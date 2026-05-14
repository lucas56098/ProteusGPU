#ifndef MPI_HALO_H
#define MPI_HALO_H
#pragma once

#include "extension.h"
#include "global/gpu_compat.h"
#include "mpi_compat.h"

// Halo exchange between Cartesian neighbors. A layer of cells at each rank-rank
// boundary is imported from the appropriate Cart neighbor; imports live in the
// extended per-cell arrays at indices [n_hydro, n_hydro + n_mpi_ghosts).

// forward decls
struct VMesh;
namespace hydro {
    struct primvars;
}
namespace gradients {
    struct PrimGradients;
}

namespace proteus_mpi {

// 3^DIMENSION - 1
#ifdef dim_3D
constexpr int HALO_MAX_NEIGHBORS = 26;
#else
constexpr int HALO_MAX_NEIGHBORS = 8;
#endif

struct MpiHalo {
    // Cart neighbor table (populated at init)
    int    n_neighbors;
    int    neighbor_ranks[HALO_MAX_NEIGHBORS];
    int    neighbor_dirs[HALO_MAX_NEIGHBORS][3];   // (dx,dy,dz) ∈ {-1,0,+1}^d \ {0}
    double neighbor_shift[HALO_MAX_NEIGHBORS][3];  // periodic-wrap shift for outgoing seeds

    // capacities (one-shot at init)
    int n_mpi_capacity;
    int per_dir_capacity;

    // per-rebuild state. send/recv buffers have per-direction stride per_dir_capacity:
    // neighbor n's slot j lives at sendbuf[n * per_dir_capacity + j].
    int n_mpi_ghosts;
    int send_count[HALO_MAX_NEIGHBORS];
    int recv_count[HALO_MAX_NEIGHBORS];
    int ghost_offset[HALO_MAX_NEIGHBORS + 1];  // imported cell j from neighbor n → [n_hydro + ghost_offset[n] + j]

    int* export_indices;

    // preallocated send/recv buffers, SoA-packed per quantity
    POINT_TYPE* sendbuf_seed;
    POINT_TYPE* recvbuf_seed;
    double*     sendbuf_rho;
    double*     recvbuf_rho;
    POINT_TYPE* sendbuf_v;
    POINT_TYPE* recvbuf_v;
    double*     sendbuf_E;
    double*     recvbuf_E;
    POINT_TYPE* sendbuf_vmesh;
    POINT_TYPE* recvbuf_vmesh;

    // gradient buffers pack (3 + DIMENSION) POINT_TYPE-sized components per cell
    POINT_TYPE* sendbuf_grad;
    POINT_TYPE* recvbuf_grad;

    // completeness sentinel: ghost is in the deepest layer (from receiver POV) of what
    // was shipped. If a local cell's K-nearest reaches such a ghost, closer cells may
    // exist beyond the halo and we must widen. send/recv layout matches the seed buffer
    // (per-direction stride per_dir_capacity); is_outer_layer is the flat by-ghost-slot
    // form, indexed as ghost_offset[n] + j.
    unsigned char* sendbuf_outer;
    unsigned char* recvbuf_outer;
    unsigned char* is_outer_layer;

#ifdef USE_MPI
    MPI_Comm    comm;
    MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
#endif
};

extern MpiHalo g_halo;

// one-shot init: builds Cart neighbor table and allocates send/recv buffers.
// per_dir_capacity is derived analytically from the brick dims and the worst-case
// halo width. single-node: no-op.
void halo_init(int n_local, double buff);
void halo_free();

// build per-rebuild export-index lists from local seeds. W = halo-layer thickness
// in buckets; pass 0 to auto-derive from buff/N_grid.
void halo_build_exports(const POINT_TYPE* local_seeds, int n_local, double buff, int W = 0);

// default halo width in buckets, matching the periodic-ghost band thickness
int halo_default_width(double buff);

// remap export_indices from old-k (pre-permute) to new-k (post-permute) so subsequent
// halo_exchange_* calls within a hydro step pack the right cells
void halo_remap_export_indices(const unsigned int* inv_gather, int n_local);

// initial exchange: seeds, primvars, v_mesh. Receivers populate the extended slots
// of mesh->seeds, primvar, mesh->v_mesh and append seed positions to pts at offset
// pts_mpi_base. Sets g_halo.n_mpi_ghosts.
void halo_exchange_initial(VMesh* mesh, hydro::primvars* primvar, POINT_TYPE* pts, int pts_mpi_base);

// refresh primvars / gradients / v_mesh in MPI ghost slots before each compute
void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar);
void halo_exchange_gradients(VMesh* mesh, gradients::PrimGradients* grads);
void halo_exchange_vmesh(VMesh* mesh);

// in-place global Allreduce(MIN) of dt
void halo_dt_allreduce(double* dt);

}  // namespace proteus_mpi

#endif  // MPI_HALO_H
