// ghost cells by request (halo.h)

#include "halo.h"

#include "decomp.h"
#include "global/allvars.h"
#include "global/structs.h"
#include "gradients/gradients.h"
#include "hydro/finite_volume_solver.h"
#include "knn/knn.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

namespace proteus_mpi {

    MpiHalo halo                = {};
    int     n_mpi_capacity      = 0;
    int     n_local_initial_max = 0;
    double  alloc_growth        = 0.0;

    // the arrays the requests and the exchanges work in, kept between builds
    struct HaloBuffers {
        // shared by steps that never run at the same time
        GpuArray<unsigned int> scan, flag, pos, run_scratch;
        PairSortArrays         sort;

        // the queries: offsets per ball, as built, grouped by target rank, received
        GpuArray<unsigned int> q_offset;
        GpuArray<GhostQuery>   q_built, q_sorted, q_in;

        // the answers: per own cell one bit per shift, per run of cells whether it has one, the hits of one
        // asker, those it did not get before, and the answers out and in
        GpuArray<unsigned int> hit_mask, run_hit, run_list, hit_offset;
        GpuArray<uint64_t>     hits, pending, sent_alt;
        GpuArray<GhostAnswer>  hit_ans, a_out, a_in;
        GpuArray<int>          stack_full;
        size_t                 n_pending = 0;

        // the used subset and the moved seeds
        GpuArray<unsigned char> used_bitmap;
        GpuArray<int>           cells, moved, m_ghost;
        GpuArray<uint64_t>      m_dest;
        GpuArray<GhostAnswer>   m_out, m_in;

        unsigned int* scan_scratch(size_t n) { return scan.fit(scan_scratch_size(n, _MPI_PACK_BLOCK_SIZE_)); }

        void free() {
            for (GpuArray<unsigned int>* a :
                 {&scan, &flag, &pos, &run_scratch, &q_offset, &hit_mask, &run_hit, &run_list, &hit_offset})
                a->free();
            for (GpuArray<uint64_t>* a : {&hits, &pending, &sent_alt, &m_dest})
                a->free();
            for (GpuArray<GhostAnswer>* a : {&hit_ans, &a_out, &a_in, &m_out, &m_in})
                a->free();
            for (GpuArray<int>* a : {&stack_full, &cells, &moved, &m_ghost})
                a->free();
            q_built.free();
            q_sorted.free();
            q_in.free();
            used_bitmap.free();
            sort.free();
            n_pending = 0;
        }
    };

    static HaloBuffers s_buffers;

    // a guess of a few cell layers around the domain; the slots grow when a build needs more
    void halo_init(int n_local) {
        halo         = MpiHalo();
        int capacity = 0;
#ifdef USE_MPI
        if (decomp.nranks > 1) {
            const double n       = (double)std::max(n_local, 1);
            const double surface = (DIMENSION == 3) ? 6.0 * std::pow(n, 2.0 / 3.0) : 4.0 * std::sqrt(n);
            capacity             = std::max(1024, (int)(4.0 * surface));
        }
#else
        (void)n_local;
#endif
        n_mpi_capacity = capacity;
    }

    void halo_free() {
        for (GpuArray<int>* a : {&halo.g_owner, &halo.g_slot, &halo.used_export_indices, &halo.used_to_full_slot})
            a->free();
        halo.g_ans.free();
        halo.sent.free();
        halo.sendbuf.free();
        halo.recvbuf.free();
        s_buffers.free();
        halo           = MpiHalo();
        n_mpi_capacity = 0;
    }

    // more slots: at least double, and take everything that is sized by them along
    void halo_grow_capacity(int new_capacity) {
        const int old_cap = n_mpi_capacity;
        const int target  = std::max(new_capacity, std::max(1024, 2 * old_cap));
        n_mpi_capacity    = target;

        if (sim.mesh) voronoi::mesh_grow_ghosts(sim.mesh, target);
        if (sim.primvar) hydro::primvar_grow_ghosts(sim.primvar, target);
        if (sim.grads) gradients::grad_grow_ghosts(sim.grads, target);

        printf("HALO: rank %d grew its ghost slots %d -> %d.\n", decomp.rank, old_cap, target);
        fflush(stdout);
    }

    // clang-format off
    // one translation unit, so the include order matters
    #include "halo_requests.cu"
    #include "halo_exchange.cu"
    // clang-format on

} // namespace proteus_mpi
