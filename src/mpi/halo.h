#ifndef MPI_HALO_H
#define MPI_HALO_H
#pragma once

#include "global/gpu_compat.h"
#include "mpi_compat.h"

// Halo exchange between Cartesian neighbors. The pipeline is:
//   1. halo_build_exports: identify boundary-layer cells per direction.
//   2. halo_exchange_seeds: ship seed positions (full halo, every widening
//      iteration). Mesh build only needs seeds.
//   3. halo_build_used_subset: after the mesh converges, scan neighbor_cell
//      to identify which MPI ghosts are actually referenced by local faces,
//      then exchange a bitmap with senders so they know which subset to pack.
//   4. halo_exchange_primvars / _gradients / _v_mesh: ship only the used subset.
//      These are the per-step refresh paths and run in the hydro hot loop.

struct VMesh;
namespace hydro {
    struct primvars;
}
namespace gradients {
    struct PrimGradients;
}

namespace proteus_mpi {

#ifdef dim_3D
    constexpr int HALO_MAX_NEIGHBORS = 26;
#else
    constexpr int HALO_MAX_NEIGHBORS = 8;
#endif

    // SoA-style packed payload for primvar refresh (one MPI message per neighbor)
    struct HaloPrimCell {
        double     rho;
        POINT_TYPE v;
        double     E;
    };

    // naming: send_count/recv_count are full-halo per-neighbor arrays;
    // n_used_send/n_used_recv are used-subset totals; send_n_outer is the
    // outermost-layer split of the full halo.
    struct MpiHalo {
        int    n_neighbors;
        int    neighbor_ranks[HALO_MAX_NEIGHBORS];
        int    neighbor_dirs[HALO_MAX_NEIGHBORS][3];
        double neighbor_shift[HALO_MAX_NEIGHBORS][3];

        // managed-memory mirror of neighbor_shift (flat row-major n*3+a) so the seed-pack
        // CUDA kernel can read shifts device-side. Filled once after build_neighbor_table.
        double* neighbor_shift_flat;

        int n_mpi_capacity;
        int use_neighbor_coll;

        // full halo layout (built by halo_build_exports, used by halo_exchange_seeds)
        int n_mpi_ghosts;
        int send_count[HALO_MAX_NEIGHBORS];
        int recv_count[HALO_MAX_NEIGHBORS];
        int send_offset[HALO_MAX_NEIGHBORS + 1];
        int ghost_offset[HALO_MAX_NEIGHBORS + 1];

        int send_n_outer[HALO_MAX_NEIGHBORS];
        int recv_n_outer[HALO_MAX_NEIGHBORS];

        int*           export_indices;
        unsigned char* dir_of_slot;

        // used subset (built by halo_build_used_subset after the mesh converges,
        // consumed by halo_exchange_primvars/_gradients/_v_mesh)
        int used_send_count[HALO_MAX_NEIGHBORS];
        int used_recv_count[HALO_MAX_NEIGHBORS];
        int used_send_offset[HALO_MAX_NEIGHBORS + 1];
        int used_recv_offset[HALO_MAX_NEIGHBORS + 1];
        int n_used_send;
        int n_used_recv;
        int used_subset_ready; // 0 until halo_build_used_subset has run

        int* used_export_indices; // [used_send_slot] -> local cell k
        int* used_to_full_slot;   // [used_recv_slot] -> full ghost slot

        // scratch bitmaps reused each rebuild (one byte per slot)
        unsigned char* send_used_bitmap;
        unsigned char* recv_used_bitmap;

        // send/recv buffers
        POINT_TYPE*   sendbuf_seed;
        POINT_TYPE*   recvbuf_seed;
        HaloPrimCell* sendbuf_prim;
        HaloPrimCell* recvbuf_prim;
        POINT_TYPE*   sendbuf_v_mesh;
        POINT_TYPE*   recvbuf_v_mesh;
        POINT_TYPE*   sendbuf_grad;
        POINT_TYPE*   recvbuf_grad;

        // is_outer_layer is derived from positional packing in halo_exchange_seeds
        unsigned char* is_outer_layer;

#ifdef USE_MPI
        MPI_Comm     graph_comm;
        MPI_Datatype mpi_prim_t;
        MPI_Datatype mpi_point_t;
        MPI_Datatype mpi_grad_cell_t;
#endif
    };

    extern MpiHalo halo;

    void halo_init(int n_local, double buff);
    void halo_free();

    void halo_build_exports(const POINT_TYPE* local_seeds, int n_local, double buff, int W = 0);
    int  halo_default_width(double buff);
    void halo_remap_export_indices(const unsigned int* inv_gather, int n_local);

    // ship seed positions on the full halo. receivers populate pts[pts_mpi_base..]
    // and mesh->seeds[n_hydro..]. sets halo.n_mpi_ghosts and is_outer_layer.
    void halo_exchange_seeds(VMesh* mesh, POINT_TYPE* pts, int pts_mpi_base);

    // after mesh build, identify the subset of MPI ghosts that local faces
    // reference, exchange the bitmap with senders, and build the compact used-
    // subset arrays consumed by the per-quantity exchanges below.
    void halo_build_used_subset(VMesh* mesh);

    // per-quantity refreshes — use the compact used subset
    void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar);
    void halo_exchange_gradients(VMesh* mesh, gradients::PrimGradients* grads);
    void halo_exchange_v_mesh(VMesh* mesh);

    void halo_dt_allreduce(double* dt);

    // runtime ghost-capacity growth. Called when halo_build_exports observes an overflow
    // of total_send or total_recv past the current n_mpi_capacity. Reallocates:
    //   - this struct's send/recv/index/bitmap buffers
    //   - sim.mesh's seeds_g / v_mesh_g
    //   - sim.primvar's rho_g / v_g / E_g
    //   - sim.grads' rho_g / vx_g / vy_g / vz_g / E_g
    // new_capacity is rounded up to at least 2× the current capacity to amortize.
    // Caller is expected to retry the failing build after this returns.
    void halo_grow_capacity(int new_capacity);

} // namespace proteus_mpi

#endif // MPI_HALO_H
