#ifndef MPI_HALO_H
#define MPI_HALO_H
#pragma once

// Ghost cells by request: a cell asks for the cells inside a ball, every rank whose part of the curve the
// ball touches answers. One rank asks itself for the periodic copies. Then the state that goes with them.
// Order in one mesh build: halo_begin_build, halo_request_balls (once per round), halo_build_used_subset,
// then the state exchanges for as long as that mesh stands.

#include "exchange.h"
#include "global/gpu_compat.h"
#include "mpi_compat.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct VMesh;
namespace hydro {
    struct primvars;
}
namespace gradients {
    struct PrimGradients;
}

namespace proteus_mpi {

    // one cell's state on the wire
    struct HaloPrimCell {
        double     rho;
        POINT_TYPE v;
        double     E;
    };

    // POINT_TYPEs per cell in the gradient message: rho, one per velocity axis, E, anchor
    constexpr int HALO_GRAD_COMPONENTS = 4 + DIMENSION;

    // a periodic shift s of the box, one of 3^DIMENSION, as a small code
    HD inline int shift_code(int sx, int sy, int sz) {
        return (sx + 1) + 3 * (sy + 1) + 9 * (sz + 1);
    }
    HD inline void shift_of_code(int code, double* s) {
        s[0] = (double)(code % 3 - 1);
        s[1] = (double)((code / 3) % 3 - 1);
        s[2] = (double)(code / 9 - 1);
    }
    constexpr int SHIFT_NONE = 13;

    // a ball a rank asks another one (or itself) for, in the frame of the owner
    struct GhostQuery {
        POINT_TYPE c;
        double     r;
        int        shift; // the answer comes back moved by this shift
    };

    // one cell an owner sends back
    struct GhostAnswer {
        POINT_TYPE p; // its seed, moved by the shift
        int        k; // its index on the owner
        int        shift;
    };

    // the ghosts of this build and who they came from
    struct MpiHalo {
        // ghost g: owner rank, cell on the owner, shift; it sits after the cells in the point list
        std::vector<int>                  g_owner;
        std::vector<int>                  g_k;
        std::vector<int>                  g_shift;
        std::vector<POINT_TYPE>           g_pos;
        std::vector<int>                  g_slot;       // MPI ghost slot, -1 for a copy of an own cell
        std::unordered_map<uint64_t, int> g_index;      // (owner, k, shift) -> ghost
        int                               n_mpi_ghosts; // ghosts with a slot

        // partners of this build: ranks this one asked, ranks that asked this one
        std::vector<int> asked;
        std::vector<int> askers;
        // per asker, what it got: (k, shift) packed, sorted
        std::vector<std::vector<uint64_t>> sent;

        // the used subset: only ghosts that a local cell has as a face neighbour get state
        int              used_subset_ready;
        int              n_used_send;
        int              n_used_recv;
        std::vector<int> used_send_count;     // per asker
        std::vector<int> used_recv_count;     // per asked rank
        int*             used_export_indices; // cell behind every send entry
        int*             used_to_full_slot;   // slot behind every receive entry
        int              send_capacity;

        // one pair of buffers per kind of data
        HaloPrimCell* sendbuf_prim;
        HaloPrimCell* recvbuf_prim;
        POINT_TYPE*   sendbuf_point;
        POINT_TYPE*   recvbuf_point;
        POINT_TYPE*   sendbuf_grad;
        POINT_TYPE*   recvbuf_grad;
        double*       sendbuf_double;
        double*       recvbuf_double;
    };

    extern MpiHalo halo;

    // buffers sized from an estimate; they grow when a build needs more
    void halo_init(int n_local);
    void halo_free();

    // forgets the ghosts and partners of the last build
    void halo_begin_build();

    // asks for the cells in a ball of radius radii[i] around cell cells[i], i < nb, and appends the new ghosts;
    // collective, also on one rank. All arrays in managed memory; cell_pos holds the seeds in cell order, the
    // owners answer from theirs
    void halo_request_balls(VMesh* mesh, const POINT_TYPE* cell_pos, const int* cells, const double* radii, int nb);

    // the build sorted the ghosts into the point list; this writes their seeds and neighbour indices
    void halo_write_ghosts(VMesh* mesh, POINT_TYPE* pts, uint64_t* ghost_ids, int n_hydro);

    // finds the ghosts the local cells really touch, so the state exchanges stay small
    void halo_build_used_subset(VMesh* mesh);

    // state of those ghosts, once per use
    void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar);
    void halo_exchange_gradients(VMesh* mesh, gradients::PrimGradients* grads);
    void halo_exchange_v_mesh(VMesh* mesh);
    void halo_exchange_centroids(VMesh* mesh);
#ifdef VOL_REGULARIZE
    void halo_exchange_volumes(VMesh* mesh);
#endif

    // a ghost seed its owner moved after it was sent
    struct MovedSeed {
        POINT_TYPE pos;
        int        ghost_slot;
    };

    // how many of these cells another rank holds as a ghost
    int halo_count_moved_exports(const std::vector<int>& moved_ks);

    // tells those ranks where the cells are now and takes what the others moved; collective
    void
    halo_exchange_moved_seeds(const VMesh* mesh, const std::vector<int>& moved_ks, std::vector<MovedSeed>* received);

    void halo_dt_allreduce(double* dt);
    void halo_sum_allreduce(double* v);

    // more slots, everything that is sized by them grows along
    void halo_grow_capacity(int new_capacity);

} // namespace proteus_mpi

#endif
