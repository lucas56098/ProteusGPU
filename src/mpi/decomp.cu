// implements the domain decomposition and the rebalancing (decomp.h)

#include "decomp.h"

#include "exchange.h"
#include "global/allvars.h"
#include "io/input.h"
#include "knn/knn.h"
#include "mpi_compat.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace mpi {

    MpiDecomp decomp = {};

#ifdef USE_MPI
    static void print_cuts(const char* what);
#endif

    // the communicator of the run and an even split of the keys
    void decomp_init() {
        decomp.rank   = rank();
        decomp.nranks = nranks();
#ifdef USE_MPI
        MPI_Comm_dup(MPI_COMM_WORLD, &decomp.comm);
#endif
        decomp.cuts = gpu_alloc<uint64_t>((size_t)decomp.nranks + 1);
        for (int r = 0; r < decomp.nranks; r++) {
            int64_t lo, hi;
            decomp_even_split((int64_t)knn::DOMAIN_KEY_END, decomp.nranks, r, &lo, &hi);
            decomp.cuts[r] = (uint64_t)lo;
        }
        decomp.cuts[decomp.nranks] = knn::DOMAIN_KEY_END;
    }

    // a new table for every rank; all ranks pass the same one
    void decomp_set_cuts(const uint64_t* cuts) {
        for (int r = 0; r <= decomp.nranks; r++)
            decomp.cuts[r] = cuts[r];
        if (decomp.cuts[0] != 0 || decomp.cuts[decomp.nranks] != knn::DOMAIN_KEY_END) {
            exit_failure("DECOMP: cut table does not cover the box (%llu .. %llu)\n",
                         (unsigned long long)decomp.cuts[0],
                         (unsigned long long)decomp.cuts[decomp.nranks]);
        }
        for (int r = 0; r < decomp.nranks; r++) {
            if (decomp.cuts[r] > decomp.cuts[r + 1]) exit_failure("DECOMP: cut table not sorted at %d\n", r);
        }
    }

    void decomp_free() {
        if (decomp.cuts) gpu_free(decomp.cuts);
        decomp.cuts = nullptr;
        decomp.probe.free();
        decomp.below.free();
    }

    // a cut stops early once its count is within n_avg / CUT_TOLERANCE of the target; below that many cells
    // per rank the bisection runs to the end
    constexpr long long CUT_TOLERANCE = 10000;

    // bisection on the key value for every inner cut, all open cuts in one Allreduce per step
    void decomp_balanced_cuts(const POINT_TYPE* pts, int n, PairSort sort, std::vector<uint64_t>* cuts_out) {
        const int P = decomp.nranks;

        // the Hilbert keys of the points, sorted on the device
        {
            uint64_t*     keys = sort.keys;
            unsigned int* vals = sort.vals;
            parallel_for<_MPI_PACK_BLOCK_SIZE_>("HILBERT_KEYS", n, [=] HD(int i) {
                keys[i] = knn::hilbert_key(pts[i]);
                vals[i] = (unsigned int)i;
            });
        }
        sort.sort("KEY_SORT", (size_t)n, knn::DOMAIN_KEY_BITS);
        const uint64_t* sorted = sort.keys;

        long long n_total = (long long)n;
#ifdef USE_MPI
        {
            PROFILE_MPI("CUTS_ALLREDUCE");
            const long long n_local = n_total;
            MPI_Allreduce(&n_local, &n_total, 1, MPI_LONG_LONG, MPI_SUM, decomp.comm);
        }
#endif
        cuts_out->assign((size_t)P + 1, 0);
        (*cuts_out)[P] = knn::DOMAIN_KEY_END;
        if (P == 1) return;

        // cut r has floor-even shares below it; lo stays below the target, hi reaches it
        const long long        tol = (n_total / P) / CUT_TOLERANCE;
        std::vector<long long> target(P);
        std::vector<uint64_t>  lo(P, 0), hi(P, knn::DOMAIN_KEY_END), cut(P, 0);
        std::vector<char>      done(P, 0);
        for (int r = 1; r < P; r++) {
            int64_t a, b;
            decomp_even_split(n_total, P, r, &a, &b);
            target[r] = a;
        }
        uint64_t*  probe = decomp.probe.fit((size_t)P);
        long long* below = decomp.below.fit((size_t)P);

        // every rank sees the same counts, so all agree which cuts are still open
        std::vector<int>       open;
        std::vector<long long> local_count, global_count;
        while (true) {
            open.clear();
            for (int r = 1; r < P; r++) {
                if (!done[r] && target[r] > 0 && hi[r] - lo[r] > 1) open.push_back(r);
            }
            if (open.empty()) break;
            const int n_open = (int)open.size();

            for (int i = 0; i < n_open; i++) {
                const int r = open[i];
                probe[i]    = lo[r] + (hi[r] - lo[r]) / 2;
            }
            parallel_for<_MPI_PACK_BLOCK_SIZE_>("CUT_COUNT", n_open, [=] HD(int i) {
                below[i] = (long long)lower_bound_of(sorted, (size_t)n, probe[i]);
            });
            local_count.assign(below, below + n_open);
            global_count = local_count;
#ifdef USE_MPI
            {
                PROFILE_MPI("CUTS_ALLREDUCE");
                MPI_Allreduce(local_count.data(), global_count.data(), n_open, MPI_LONG_LONG, MPI_SUM, decomp.comm);
            }
#endif
            for (int i = 0; i < n_open; i++) {
                const int       r   = open[i];
                const uint64_t  mid = probe[i];
                const long long g   = global_count[i];
                if (tol > 0 && g - target[r] <= tol && target[r] - g <= tol) {
                    cut[r]  = mid;
                    done[r] = 1;
                } else if (g >= target[r]) {
                    hi[r] = mid;
                } else {
                    lo[r] = mid;
                }
            }
        }
        for (int r = 1; r < P; r++)
            (*cuts_out)[r] = done[r] ? cut[r] : ((target[r] > 0) ? hi[r] : 0);
    }

    void decomp_even_split(int64_t N, int P, int i, int64_t* lo, int64_t* hi) {
        const int64_t base = N / P;
        const int64_t rem  = N % P;
        *lo                = (int64_t)i * base + std::min((int64_t)i, rem);
        *hi                = *lo + base + (i < rem ? 1 : 0);
    }

    // ==========================================================
    // IC routing
    // ==========================================================

#ifdef USE_MPI

    // one IC cell on its way to the rank that owns it
    struct ICMigrant {
        double pos[DIMENSION];
        double vel[DIMENSION];
        double rho;
        double energy;
    };

    static POINT_TYPE ic_point(const ICData& ic, int k) {
        POINT_TYPE p;
        p.x = ic.pos[DIMENSION * k + 0];
        p.y = ic.pos[DIMENSION * k + 1];
#ifdef dim_3D
        p.z = ic.pos[DIMENSION * k + 2];
#endif
        return p;
    }

    // block of rank r in b, 0 if it has none
    static size_t count_of_rank(const Blocks& b, int r) {
        for (size_t i = 0; i < b.ranks.size(); i++)
            if (b.ranks[i] == r) return b.counts[i];
        return 0;
    }

    // every rank read its own rows of the IC file; cut the curve evenly, then send each cell to its owner
    void distribute_ic_parallel(ICData& ic) {
        const size_t n    = (size_t)ic.header.n_seeds;
        const int    n_in = (int)n;
        const int    me   = decomp.rank;

        if (decomp.nranks <= 1) {
            if (me == 0) printf("DECOMP: single-rank, n_local=%d (no routing)\n", n_in);
            return;
        }

        // the positions on the device, and the arrays of the sorts and the messages; all of it only here
        GpuArray<POINT_TYPE>   pts;
        PairSortArrays         sort_arrays;
        GpuArray<unsigned int> run_scratch;
        GpuArray<ICMigrant>    send, recv;
        POINT_TYPE*            p = pts.fit(n);
        for (int k = 0; k < n_in; k++)
            p[k] = ic_point(ic, k);

        // about the same number of cells on every rank
        {
            std::vector<uint64_t> cuts;
            decomp_balanced_cuts(p, n_in, sort_arrays.fit(n), &cuts);
            decomp_set_cuts(cuts.data());
        }

        // the cells grouped by owner, read order kept within an owner
        PairSort sort = sort_arrays.fit(n);
        {
            const uint64_t* cuts   = decomp.cuts;
            const int       nranks = decomp.nranks;
            uint64_t*       keys   = sort.keys;
            unsigned int*   vals   = sort.vals;
            parallel_for<_MPI_PACK_BLOCK_SIZE_>("IC_OWNER", n_in, [=] HD(int k) {
                keys[k] = (uint64_t)owner_of_point(p[k], cuts, nranks);
                vals[k] = (unsigned int)k;
            });
        }
        sort.sort("IC_SORT", n, rank_key_bits());
        const Blocks out = blocks_of_sorted_ranks(sort.keys, n, &run_scratch);

        // packed on the host, where the IC is
        ICMigrant* sendbuf = send.fit(n);
        for (int j = 0; j < n_in; j++) {
            const int  k = (int)sort.vals[j];
            ICMigrant& m = sendbuf[j];
            for (int d = 0; d < DIMENSION; d++) {
                m.pos[d] = ic.pos[DIMENSION * k + d];
                m.vel[d] = ic.vel[DIMENSION * k + d];
            }
            m.rho    = ic.rho[k];
            m.energy = ic.energy[k];
        }

        Blocks in;
        {
            PROFILE_MPI("ICDIST_PAYLOAD_WAIT");
            sparse_counts(out, &in);
            exchange_items(sendbuf, out, recv.fit(in.total), in, sizeof(ICMigrant));
        }

        // what came back is the new content of ic_data
        const int        n_out   = (int)in.total;
        const ICMigrant* recvbuf = recv.data;
        ic.pos.resize((size_t)DIMENSION * n_out);
        ic.vel.resize((size_t)DIMENSION * n_out);
        ic.rho.resize(n_out);
        ic.energy.resize(n_out);
        for (int j = 0; j < n_out; j++) {
            const ICMigrant& m = recvbuf[j];
            for (int d = 0; d < DIMENSION; d++) {
                ic.pos[DIMENSION * j + d] = m.pos[d];
                ic.vel[DIMENSION * j + d] = m.vel[d];
            }
            ic.rho[j]    = m.rho;
            ic.energy[j] = m.energy;
        }
        ic.header.n_seeds = (uint64_t)n_out;

        const int self = (int)count_of_rank(out, me);
        printf("DECOMP: rank %d routed IC: read %d, kept %d (self %d, recv %d, sent %d)\n",
               me,
               n_in,
               n_out,
               self,
               n_out - self,
               n_in - self);
        fflush(stdout);
        print_cuts("IC");

        pts.free();
        sort_arrays.free();
        run_scratch.free();
        send.free();
        recv.free();

        // no cell may be lost on the way
        const long long n_out_ll      = (long long)n_out;
        const long long n_in_ll       = (long long)n_in;
        long long       n_global_kept = 0;
        long long       n_total_in    = 0;
        {
            PROFILE_MPI("ICDIST_CONS_ALLREDUCE");
            MPI_Allreduce(&n_out_ll, &n_global_kept, 1, MPI_LONG_LONG, MPI_SUM, decomp.comm);
            MPI_Allreduce(&n_in_ll, &n_total_in, 1, MPI_LONG_LONG, MPI_SUM, decomp.comm);
        }
        if (n_global_kept != n_total_in) {
            exit_failure("DECOMP: FATAL parallel-IC cell-count mismatch — received-sum=%lld, sent-sum=%lld.\n",
                         n_global_kept,
                         n_total_in);
        }
        if (me == 0) {
            printf("DECOMP: parallel-IC cell-count check passed (sum of per-rank n_local = %lld).\n", n_global_kept);
            fflush(stdout);
        }
    }

    // where the curve is cut, as a share of it
    static void print_cuts(const char* what) {
        if (decomp.rank != 0) return;
        printf("DECOMP: %s cuts along the Hilbert curve:", what);
        for (int r = 0; r <= decomp.nranks && r <= 16; r++)
            printf(" %.4f", (double)decomp.cuts[r] / (double)knn::DOMAIN_KEY_END);
        printf("%s\n", decomp.nranks > 16 ? " ..." : "");
        fflush(stdout);
    }

#else

    void distribute_ic_parallel(ICData& ic) {
        (void)ic;
    }

#endif

    // ==========================================================
    // rebalancing
    // ==========================================================

#ifdef USE_MPI
    static double s_pre_imbalance = 1.0;
    static bool   s_new_cuts      = false; // set by a rebalance, the migration after it logs the result

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
    void rebalance(int step, VMesh* mesh) {
        if (sim.rebalance_interval <= 0) return;
        if (step <= 0) return;
        if (step % sim.rebalance_interval != 0) return;
        if (decomp.nranks <= 1) return;

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
            return;
        }

        // the kNN sort arrays are free here: the next build sorts its points again before any search
        const int          n_hydro = (int)mesh->n_hydro;
        const knn_problem* knn     = mesh->knn;
        if (n_hydro > knn->pts_capacity) {
            exit_failure(
                "[rank %d] DECOMP: %d cells, but the kNN arrays hold %d\n", decomp.rank, n_hydro, knn->pts_capacity);
        }
        PairSort sort;
        sort.keys     = knn->d_keys;
        sort.keys_alt = knn->d_keys_alt;
        sort.vals     = knn->d_permutation;
        sort.vals_alt = knn->d_perm_alt;
        sort.scratch  = knn->d_sort_scratch;
        std::vector<uint64_t> cuts;
        {
            PROFILE("CUTS");
            decomp_balanced_cuts(mesh->scratch_move, n_hydro, sort, &cuts);
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
            return;
        }

        s_pre_imbalance = pre_imbalance;
        s_new_cuts      = true;
        decomp_set_cuts(cuts.data());
    }

    void rebalance_log_after_migration(VMesh* mesh) {
        if (!s_new_cuts) return;
        s_new_cuts = false;
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

    void rebalance(int, VMesh*) {}
    void rebalance_log_after_migration(VMesh*) {}

#endif

} // namespace mpi
