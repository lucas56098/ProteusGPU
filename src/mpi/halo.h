#ifndef MPI_HALO_H
#define MPI_HALO_H
#pragma once

// Ghost cells from the neighbour ranks: which cells to send, and the state that goes with them.
// order in one mesh build: build_exports, exchange_seeds, build_used_subset,
// then the state exchanges for as long as that mesh stands

#include "global/gpu_compat.h"
#include "mpi_compat.h"
#include <vector>

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

    // one cell's state on the wire
    struct HaloPrimCell {
        double     rho;
        POINT_TYPE v;
        double     E;
    };

    // the halo of this rank, set up once and refilled by every build
    struct MpiHalo {
        int    n_neighbors; // Cartesian neighbours that are not this rank itself
        int    neighbor_ranks[HALO_MAX_NEIGHBORS];
        int    neighbor_dirs[HALO_MAX_NEIGHBORS][3];  // -1, 0 or +1 per axis
        double neighbor_shift[HALO_MAX_NEIGHBORS][3]; // one box further on, where the seed crosses the border

        double* neighbor_shift_flat; // the same shifts, readable on the device

        int n_mpi_capacity;    // slots in every buffer
        int use_neighbor_coll; // neighbour collective, or Isend and Irecv per direction

        // exports and ghosts of the current build
        int n_mpi_ghosts;
        int send_count[HALO_MAX_NEIGHBORS];
        int recv_count[HALO_MAX_NEIGHBORS];
        int send_offset[HALO_MAX_NEIGHBORS + 1];
        int ghost_offset[HALO_MAX_NEIGHBORS + 1];

        int send_n_outer[HALO_MAX_NEIGHBORS]; // of those, from the outermost bucket layer; they come first
        int recv_n_outer[HALO_MAX_NEIGHBORS];

        int*           export_indices; // cell behind every send slot
        unsigned char* dir_of_slot;    // and the neighbour it goes to

        // the used subset: only ghosts that a local cell has as a face neighbour get state
        int used_send_count[HALO_MAX_NEIGHBORS];
        int used_recv_count[HALO_MAX_NEIGHBORS];
        int used_send_offset[HALO_MAX_NEIGHBORS + 1];
        int used_recv_offset[HALO_MAX_NEIGHBORS + 1];
        int n_used_send;
        int n_used_recv;
        int used_subset_ready;

        int* used_export_indices;
        int* used_to_full_slot; // place in the used subset -> ghost slot

        unsigned char* send_used_bitmap; // per send slot, whether the other side uses it
        unsigned char* recv_used_bitmap;

        // one pair of buffers per kind of data
        POINT_TYPE*   sendbuf_seed;
        POINT_TYPE*   recvbuf_seed;
        HaloPrimCell* sendbuf_prim;
        HaloPrimCell* recvbuf_prim;
        POINT_TYPE*   sendbuf_v_mesh;
        POINT_TYPE*   recvbuf_v_mesh;
        POINT_TYPE*   sendbuf_grad;
        POINT_TYPE*   recvbuf_grad;
#ifdef VOL_REGULARIZE
        double* sendbuf_vol;
        double* recvbuf_vol;
#endif

        unsigned char* is_outer_layer;

#ifdef USE_MPI
        MPI_Comm     graph_comm;
        MPI_Datatype mpi_prim_t;
        MPI_Datatype mpi_point_t;
        MPI_Datatype mpi_grad_cell_t;
#endif
    };

    extern MpiHalo halo;

    // neighbour table, transport mode and the buffers
    void halo_init(int n_local, double buff);
    void halo_free();

    // which cells go to which neighbour, for the seed positions of this build
    void halo_build_exports(const POINT_TYPE* local_seeds, int n_local, double buff, int W = 0);
    // bucket layers the ghost band needs
    int halo_default_width(double buff);
    // the build sorted the cells, so the export list has to follow
    void halo_remap_export_indices(const unsigned int* inv_gather, int n_local);

    // sends the export seeds and takes the neighbours' ones as ghosts
    void halo_exchange_seeds(VMesh* mesh, POINT_TYPE* pts, int pts_mpi_base);
#ifdef VOL_REGULARIZE
    void halo_exchange_volumes(VMesh* mesh);
#endif

    // a ghost seed a neighbour rank moved after the exchange
    struct MovedSeed {
        POINT_TYPE pos;
        int        ghost_slot;
    };

    struct MovedExportLists {
        std::vector<int>        js[HALO_MAX_NEIGHBORS];
        std::vector<POINT_TYPE> pos[HALO_MAX_NEIGHBORS];
    };

    int halo_collect_moved_exports(const VMesh* mesh, const std::vector<int>& moved_ks, MovedExportLists* lists);

    void halo_exchange_moved_seeds(const MovedExportLists& lists, std::vector<MovedSeed>* received);

    // finds the ghosts the local cells really touch, so the state exchanges stay small
    void halo_build_used_subset(VMesh* mesh);

    // state of those ghosts, once per use
    void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar);
    void halo_exchange_gradients(VMesh* mesh, gradients::PrimGradients* grads);
    void halo_exchange_v_mesh(VMesh* mesh);

    void halo_dt_allreduce(double* dt);
    void halo_sum_allreduce(double* v);

    // more slots, everything that is sized by them grows along
    void halo_grow_capacity(int new_capacity);

} // namespace proteus_mpi

#endif
