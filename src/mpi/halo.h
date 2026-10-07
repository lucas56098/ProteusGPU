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
#include <vector>

struct VMesh;
namespace hydro {
    struct primvars;
}
namespace gradients {
    struct PrimGradients;
}

namespace proteus_mpi {

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
        // ghost g: the answer it came with, its owner rank and its MPI ghost slot, -1 for a periodic ghost;
        // it sits after the cells in the point list
        int                   n_ghosts;
        int                   n_mpi_ghosts;
        GpuArray<GhostAnswer> g_ans;
        GpuArray<int>         g_owner;
        GpuArray<int>         g_slot;

        // partners of this build: ranks this one asked, ranks that asked this one
        std::vector<int> asked;
        std::vector<int> askers;
        // what this rank sent, (asker, k, shift) as one key, sorted; the periodic ghosts too
        GpuArray<uint64_t> sent;
        size_t             n_sent;

        // the used subset: only ghosts that a local cell has as a face neighbour get state
        int           used_subset_ready;
        Blocks        state_send;          // per asker the cells this rank sends
        Blocks        state_recv;          // per asked rank the ghosts this rank gets
        GpuArray<int> used_export_indices; // cell behind every send entry
        GpuArray<int> used_to_full_slot;   // slot behind every receive entry

        // the state on the wire, one kind at a time
        GpuArray<char> sendbuf;
        GpuArray<char> recvbuf;
    };

    extern MpiHalo halo;

    // the first ghost slots, from an estimate; they grow when a build needs more
    void halo_init(int n_local);
    void halo_free();

    // forgets the ghosts and partners of the last build
    void halo_begin_build();

    // asks for the cells in a ball of radius radii[i] around cell cells[i], i < nb, and appends the new ghosts;
    // collective, also on one rank. All arrays in gpu memory; cell_pos holds the seeds in cell order, the
    // owners answer from theirs
    void halo_request_balls(VMesh* mesh, const POINT_TYPE* cell_pos, const int* cells, const double* radii, int nb);

    // the build sorted the ghosts into the point list pts; this writes their seeds and neighbour indices
    void halo_write_ghosts(VMesh* mesh, POINT_TYPE* pts);

    // finds the ghosts the local cells really touch, so the state exchanges stay small
    void halo_build_used_subset(VMesh* mesh);

    // state of those ghosts, once per use
    void halo_exchange_primvars(hydro::primvars* primvar);
    void halo_exchange_gradients(gradients::PrimGradients* grads);
    void halo_exchange_v_mesh(VMesh* mesh);
    void halo_exchange_centroids(VMesh* mesh);
#ifdef VOL_REGULARIZE
    void halo_exchange_volumes(VMesh* mesh);
#endif

    // more slots, everything that is sized by them grows along
    void halo_grow_capacity(int new_capacity);

} // namespace proteus_mpi

#endif
