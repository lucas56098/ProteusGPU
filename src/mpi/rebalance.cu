#include "rebalance.h"

#include "../global/allvars.h"
#include "../global/log.h"
#include "../global/structs.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "decomp.h"

#include <cstdio>
#include <vector>

namespace proteus_mpi {

    RebalanceConfig rebalance_config = {0, 0, 1.10};

#ifdef USE_MPI
    // pre-rebalance imbalance, saved by rebalance_decide on rebalance steps
    // for rebalance_log_after_migration to print alongside the post probe.
    static double s_pre_imbalance = 1.0;

    // Allreduce of n_local: returns global max (int — per-rank fits int32 well past
    // our target scale), global mean (long long — the cross-rank SUM is n_global which
    // overflows int32 from ~1300^3 upward), and the max/mean imbalance ratio.
    // Shared by the probe and pre/post logs.
    static void compute_imbalance_probe(VMesh* mesh, int* n_max, long long* n_avg, double* imbalance) {
        const int       n_local_int = (int)mesh->n_hydro;
        const long long n_local_ll  = (long long)mesh->n_hydro;
        int             g_max       = n_local_int;
        long long       g_sum       = n_local_ll;
        {
            PROFILE_MPI("IMBALANCE_PROBE");
            MPI_Allreduce(&n_local_int, &g_max, 1, MPI_INT, MPI_MAX, decomp.cart_comm);
            MPI_Allreduce(&n_local_ll, &g_sum, 1, MPI_LONG_LONG, MPI_SUM, decomp.cart_comm);
        }
        *n_max     = g_max;
        *n_avg     = g_sum / (long long)decomp.nranks;
        *imbalance = (g_sum > 0) ? (double)g_max * (double)decomp.nranks / (double)g_sum : 1.0;
    }
#endif

    void rebalance_imbalance_log(int step, VMesh* mesh) {
        if (rebalance_config.imbalance_log_interval <= 0) return;
        if (step % rebalance_config.imbalance_log_interval != 0) return;
#ifdef USE_MPI
        if (decomp.nranks <= 1) return;
        int       n_max;
        long long n_avg;
        double    imbalance;
        compute_imbalance_probe(mesh, &n_max, &n_avg, &imbalance);
        if (decomp.rank == 0) {
            printf("DECOMP: imbalance=%.2f (n_max=%d, n_avg=%lld)\n", imbalance, n_max, n_avg);
            fflush(stdout);
        }
#else
        (void)step;
        (void)mesh;
#endif
    }

#ifdef USE_MPI

    // marginal cell-count histograms along each axis. hx[bx] = #cells whose bucket.x == bx
    // (summed over by, bz). pts is the position buffer to bucket against (move_mesh passes
    // post-advance positions in mesh->scratch_move so splits reflect where cells will sit
    // for the next step). hx/hy/hz are managed-memory buffers — reused across calls via
    // persistent statics inside compute_local_histograms.

    // per-cell body: atomicAdd into the three managed marginal histograms.
    HD inline void hist_body(int k, const POINT_TYPE* pts, int N_grid, double bf, int* hx, int* hy, int* hz) {
        int bx, by, bz;
        decomp_bucket_of_point(pts[k].x,
                               pts[k].y,
#ifdef dim_3D
                               pts[k].z,
#else
                               0.0,
#endif
                               N_grid,
                               bf,
                               &bx,
                               &by,
                               &bz);
        portable_atomicAdd(&hx[bx], 1);
        portable_atomicAdd(&hy[by], 1);
#ifdef dim_3D
        portable_atomicAdd(&hz[bz], 1);
#else
        (void)hz;
#endif
    }

#ifndef CPU_DEBUG
    GLOBAL static void
    kernel_histograms(int n_hydro, const POINT_TYPE* pts, int N_grid, double bf, int* hx, int* hy, int* hz) {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_hydro) return;
        hist_body(k, pts, N_grid, bf, hx, hy, hz);
    }
#endif

    static void compute_local_histograms(
        POINT_TYPE* pts, int n_hydro, int N_grid, double bf, int*& hx, int*& hy, int*& hz, int& hxyz_cap) {
        // grow managed histogram buffers if N_grid has increased (or first call)
        const int need = std::max(N_grid, 1);
        if (need > hxyz_cap) {
            if (hx) gpu_free(hx);
            if (hy) gpu_free(hy);
            if (hz) gpu_free(hz);
            hx       = (int*)gpu_malloc(sizeof(int) * (size_t)need);
            hy       = (int*)gpu_malloc(sizeof(int) * (size_t)need);
            hz       = (int*)gpu_malloc(sizeof(int) * (size_t)need);
            hxyz_cap = need;
        }
        gpu_memset(hx, 0, sizeof(int) * (size_t)N_grid);
        gpu_memset(hy, 0, sizeof(int) * (size_t)N_grid);
#ifdef dim_3D
        gpu_memset(hz, 0, sizeof(int) * (size_t)N_grid);
#else
        // 2D: every cell has bz=0; collapse to a single-bin total.
        gpu_memset(hz, 0, sizeof(int));
        hz[0] = n_hydro;
#endif

#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (n_hydro + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("HIST_K");
            kernel_histograms<<<blocks, tpb>>>(n_hydro, pts, N_grid, bf, hx, hy, hz);
        }
        GPU_SYNC();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int k = 0; k < n_hydro; k++) {
            hist_body(k, pts, N_grid, bf, hx, hy, hz);
        }
#ifndef dim_3D
        (void)0; // hz already filled above
#endif
#endif
    }

    // walk hist[], place split[c] at the smallest bucket index whose cumulative sum reaches
    // ceil(c × total / Pa). The ceiling division is what lets one extra cell land in the
    // lower-index slab when total isn't a multiple of Pa — produces strictly monotone splits.
    // Snap any empty slab to width-1 afterwards (the decomposition requires non-empty bricks).
    static void compute_splits_axis(const int* hist, int N_grid, int Pa, int* split_out) {
        split_out[0] = 0;
        if (Pa <= 1) {
            split_out[1] = N_grid;
            return;
        }

        long long total = 0;
        for (int i = 0; i < N_grid; i++)
            total += hist[i];

        long long running = 0;
        int       next_c  = 1;
        long long target  = ((long long)next_c * total + Pa - 1) / Pa;

        for (int i = 0; i < N_grid && next_c < Pa; i++) {
            running += hist[i];
            while (next_c < Pa && running >= target) {
                split_out[next_c] = i + 1;
                next_c++;
                target = ((long long)next_c * total + Pa - 1) / Pa;
            }
        }
        while (next_c < Pa) {
            split_out[next_c] = N_grid;
            next_c++;
        }
        split_out[Pa] = N_grid;

        // forward sweep: ensure each slab has width >= 1
        for (int c = 0; c < Pa; c++) {
            if (split_out[c + 1] < split_out[c] + 1) { split_out[c + 1] = split_out[c] + 1; }
        }
        // backward sweep: if the forward sweep pushed the last split past N_grid, pull
        // earlier splits left so we end at N_grid. If N_grid < Pa, splits become degenerate
        // and check_bricks_nonempty in decomp_apply_splits will exit_failure cleanly.
        split_out[Pa] = N_grid;
        for (int c = Pa; c > 0; c--) {
            if (split_out[c - 1] > split_out[c] - 1) { split_out[c - 1] = split_out[c] - 1; }
        }
    }

    bool rebalance_decide(int step, VMesh* mesh, POINT_TYPE* pts) {
        if (rebalance_config.rebalance_interval <= 0) return false;
        if (step <= 0) return false;
        if (step % rebalance_config.rebalance_interval != 0) return false;
        if (decomp.nranks <= 1) return false;

        PROFILE("BALANCE");

        // pre-rebalance probe — used either for the "Skipped" line below or saved
        // for rebalance_log_after_migration to pair with the post probe.
        int       pre_n_max;
        long long pre_n_avg;
        double    pre_imbalance;
        compute_imbalance_probe(mesh, &pre_n_max, &pre_n_avg, &pre_imbalance);

        // gate on threshold — don't disturb a healthy decomposition. The histogram-based
        // split chooser has bucket-granularity overshoot near concentrated regions (e.g.
        // contact discontinuities), so re-splitting a near-balanced state can hurt.
        if (pre_imbalance < rebalance_config.imbalance_threshold) {
            if (decomp.rank == 0) {
                printf("DECOMP: Skipped rebalancing (below threshold, imbalance=%.2f)\n", pre_imbalance);
                fflush(stdout);
            }
            return false;
        }

        const int    N_grid  = decomp.N_grid_global;
        const int    n_hydro = (int)mesh->n_hydro;
        const double bf      = mesh->buff;

        // Persistent managed histogram buffers — kept across calls so we don't
        // gpu_malloc/free per rebalance step. compute_local_histograms grows them
        // if N_grid increases (it doesn't change after begrun).
        static int* s_hx          = nullptr;
        static int* s_hy          = nullptr;
        static int* s_hz          = nullptr;
        static int  s_hxyz_cap    = 0;
        static int* s_hx_global   = nullptr;
        static int* s_hy_global   = nullptr;
        static int* s_hz_global   = nullptr;
        static int  s_hglobal_cap = 0;

        {
            PROFILE("HIST");
            compute_local_histograms(pts, n_hydro, N_grid, bf, s_hx, s_hy, s_hz, s_hxyz_cap);
        }

        const int need_global = std::max(N_grid, 1);
        if (need_global > s_hglobal_cap) {
            if (s_hx_global) gpu_free(s_hx_global);
            if (s_hy_global) gpu_free(s_hy_global);
            if (s_hz_global) gpu_free(s_hz_global);
            s_hx_global   = (int*)gpu_malloc(sizeof(int) * (size_t)need_global);
            s_hy_global   = (int*)gpu_malloc(sizeof(int) * (size_t)need_global);
            s_hz_global   = (int*)gpu_malloc(sizeof(int) * (size_t)need_global);
            s_hglobal_cap = need_global;
        }
        gpu_memset(s_hx_global, 0, sizeof(int) * (size_t)N_grid);
        gpu_memset(s_hy_global, 0, sizeof(int) * (size_t)N_grid);
#ifdef dim_3D
        gpu_memset(s_hz_global, 0, sizeof(int) * (size_t)N_grid);
#else
        gpu_memset(s_hz_global, 0, sizeof(int));
#endif

        mpi_sync_before_send(s_hx, sizeof(int) * (size_t)N_grid);
        mpi_sync_before_send(s_hy, sizeof(int) * (size_t)N_grid);
#ifdef dim_3D
        mpi_sync_before_send(s_hz, sizeof(int) * (size_t)N_grid);
#endif
        {
            PROFILE_MPI("ALLREDUCE");
            MPI_Allreduce(s_hx, s_hx_global, N_grid, MPI_INT, MPI_SUM, decomp.cart_comm);
            MPI_Allreduce(s_hy, s_hy_global, N_grid, MPI_INT, MPI_SUM, decomp.cart_comm);
#ifdef dim_3D
            MPI_Allreduce(s_hz, s_hz_global, N_grid, MPI_INT, MPI_SUM, decomp.cart_comm);
#else
            // 2D: every cell sits in bz=0. Reduction collapses to a single bin —
            // long long for symmetry with 3D (per-bin total fits int32).
            const long long hz0_ll  = (long long)s_hz[0];
            long long       n_total = 0;
            MPI_Allreduce(&hz0_ll, &n_total, 1, MPI_LONG_LONG, MPI_SUM, decomp.cart_comm);
            s_hz_global[0] = (int)n_total;
#endif
        }
        mpi_sync_after_recv(s_hx_global, sizeof(int) * (size_t)N_grid);
        mpi_sync_after_recv(s_hy_global, sizeof(int) * (size_t)N_grid);
#ifdef dim_3D
        mpi_sync_after_recv(s_hz_global, sizeof(int) * (size_t)N_grid);
#endif

        std::vector<int> sx(decomp.dims[0] + 1, 0);
        std::vector<int> sy(decomp.dims[1] + 1, 0);
        std::vector<int> sz(decomp.dims[2] + 1, 0);
        {
            PROFILE("SPLIT");
            compute_splits_axis(s_hx_global, N_grid, decomp.dims[0], sx.data());
            compute_splits_axis(s_hy_global, N_grid, decomp.dims[1], sy.data());
#ifdef dim_3D
            compute_splits_axis(s_hz_global, N_grid, decomp.dims[2], sz.data());
#else
            // 2D: z slab is the fixed {0,1} bucket; not derived from histogram.
            sz[0] = 0;
            sz[1] = 1;
#endif
        }

        // diff against current splits; identical => skip the migration entirely.
        bool same = true;
        for (int i = 0; i <= decomp.dims[0] && same; i++)
            if (sx[i] != decomp.splits[0][i]) same = false;
        for (int i = 0; i <= decomp.dims[1] && same; i++)
            if (sy[i] != decomp.splits[1][i]) same = false;
        for (int i = 0; i <= decomp.dims[2] && same; i++)
            if (sz[i] != decomp.splits[2][i]) same = false;
        if (same) {
            if (decomp.rank == 0) {
                printf("DECOMP: Skipped rebalancing (splits unchanged, imbalance=%.2f)\n", pre_imbalance);
                fflush(stdout);
            }
            return false;
        }

        // splits will change — stash pre-imbalance for the post-migration log line.
        s_pre_imbalance = pre_imbalance;
        decomp_apply_splits(sx.data(), sy.data(), sz.data());

        return true;
    }

    void rebalance_log_after_migration(VMesh* mesh) {
        if (decomp.nranks <= 1) return;
        int       n_max;
        long long n_avg;
        double    post_imbalance;
        compute_imbalance_probe(mesh, &n_max, &n_avg, &post_imbalance);
        if (decomp.rank == 0) {
            // migrated count appears on the MPI: line later in the same step — don't duplicate.
            printf("DECOMP: Rebalanced (imbalance %.2f -> %.2f)\n", s_pre_imbalance, post_imbalance);
            fflush(stdout);
        }
    }

#else // USE_MPI

    bool rebalance_decide(int, VMesh*, POINT_TYPE*) {
        return false;
    }
    void rebalance_log_after_migration(VMesh*) {}

#endif

} // namespace proteus_mpi
