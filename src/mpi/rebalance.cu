// implements the rebalancing (rebalance.h)

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

#ifdef USE_MPI
    static double s_pre_imbalance = 1.0;

    // largest cell count over all ranks, divided by the mean
    static void compute_imbalance_probe(VMesh* mesh, int* n_max, long long* n_avg, double* imbalance) {
        const int       n_local_int = (int)mesh->n_hydro;
        const long long n_local_ll  = (long long)mesh->n_hydro;
        int             g_max       = n_local_int;
        long long       g_sum       = n_local_ll;
        {
            PROFILE_MPI("IMBALANCE_PROBE");
            MPI_Allreduce(&n_local_int, &g_max, 1, MPI_INT, MPI_MAX, decomp.comm);
            MPI_Allreduce(&n_local_ll, &g_sum, 1, MPI_LONG_LONG, MPI_SUM, decomp.comm);
        }
        *n_max     = g_max;
        *n_avg     = g_sum / (long long)decomp.nranks;
        *imbalance = (g_sum > 0) ? (double)g_max * (double)decomp.nranks / (double)g_sum : 1.0;
    }
#endif

    void rebalance_imbalance_log(int step, VMesh* mesh) {
        if (sim.imbalance_log_interval <= 0) return;
        if (step % sim.imbalance_log_interval != 0) return;
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

    // new cuts from the cells at their new position, if the imbalance is worth it
    bool rebalance_decide(int step, VMesh* mesh, POINT_TYPE* pts) {
        if (sim.rebalance_interval <= 0) return false;
        if (step <= 0) return false;
        if (step % sim.rebalance_interval != 0) return false;
        if (decomp.nranks <= 1) return false;

        PROFILE("BALANCE");

        int       pre_n_max;
        long long pre_n_avg;
        double    pre_imbalance;
        compute_imbalance_probe(mesh, &pre_n_max, &pre_n_avg, &pre_imbalance);

        if (pre_imbalance < sim.imbalance_threshold) {
            if (decomp.rank == 0) {
                printf("DECOMP: Skipped rebalancing (below threshold, imbalance=%.2f)\n", pre_imbalance);
                fflush(stdout);
            }
            return false;
        }

        const int             n_hydro = (int)mesh->n_hydro;
        std::vector<uint64_t> keys((size_t)n_hydro);
        for (int k = 0; k < n_hydro; k++)
            keys[k] = knn::hilbert_key(pts[k]);
        std::vector<uint64_t> cuts;
        {
            PROFILE("CUTS");
            decomp_balanced_cuts(keys, &cuts);
        }

        // nothing to gain if the cuts come out where they already are
        bool same = true;
        for (int r = 0; r <= decomp.nranks && same; r++)
            if (cuts[r] != decomp.cuts[r]) same = false;
        if (same) {
            if (decomp.rank == 0) {
                printf("DECOMP: Skipped rebalancing (cuts unchanged, imbalance=%.2f)\n", pre_imbalance);
                fflush(stdout);
            }
            return false;
        }

        s_pre_imbalance = pre_imbalance;
        decomp_set_cuts(cuts.data());
        return true;
    }

    void rebalance_log_after_migration(VMesh* mesh) {
        if (decomp.nranks <= 1) return;
        int       n_max;
        long long n_avg;
        double    post_imbalance;
        compute_imbalance_probe(mesh, &n_max, &n_avg, &post_imbalance);
        if (decomp.rank == 0) {
            printf("DECOMP: Rebalanced (imbalance %.2f -> %.2f)\n", s_pre_imbalance, post_imbalance);
            fflush(stdout);
        }
    }

#else

    bool rebalance_decide(int, VMesh*, POINT_TYPE*) {
        return false;
    }
    void rebalance_log_after_migration(VMesh*) {}

#endif

} // namespace proteus_mpi
