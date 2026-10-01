// implements the domain decomposition (decomp.h)

#include "decomp.h"

#include "io/input.h"
#include "mpi_compat.h"
#include "profiler/profiler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace proteus_mpi {

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

    // for every inner cut the smallest key with the wanted number of keys of all ranks below it
    void decomp_balanced_cuts(std::vector<uint64_t>& local_keys, std::vector<uint64_t>* cuts_out) {
        const int P = decomp.nranks;
        std::sort(local_keys.begin(), local_keys.end());

        long long n_total = (long long)local_keys.size();
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
        std::vector<long long> target(P);
        std::vector<uint64_t>  lo(P, 0), hi(P, knn::DOMAIN_KEY_END);
        for (int r = 1; r < P; r++) {
            int64_t a, b;
            decomp_even_split(n_total, P, r, &a, &b);
            target[r] = a;
        }

        std::vector<long long> local_count(P), global_count(P);
        while (true) {
            bool open = false;
            for (int r = 1; r < P; r++) {
                if (target[r] > 0 && hi[r] - lo[r] > 1) open = true;
            }
            if (!open) break;

            // all cuts move in one step, one Allreduce each step
            for (int r = 1; r < P; r++) {
                const uint64_t mid = lo[r] + (hi[r] - lo[r]) / 2;
                local_count[r] =
                    (long long)(std::lower_bound(local_keys.begin(), local_keys.end(), mid) - local_keys.begin());
            }
            global_count = local_count;
#ifdef USE_MPI
            {
                PROFILE_MPI("CUTS_ALLREDUCE");
                MPI_Allreduce(
                    local_count.data() + 1, global_count.data() + 1, P - 1, MPI_LONG_LONG, MPI_SUM, decomp.comm);
            }
#endif
            for (int r = 1; r < P; r++) {
                if (target[r] <= 0 || hi[r] - lo[r] <= 1) continue;
                const uint64_t mid = lo[r] + (hi[r] - lo[r]) / 2;
                if (global_count[r] >= target[r])
                    hi[r] = mid;
                else
                    lo[r] = mid;
            }
        }
        for (int r = 1; r < P; r++)
            (*cuts_out)[r] = (target[r] > 0) ? hi[r] : 0;
    }

    void decomp_even_split(int64_t N, int P, int i, int64_t* lo, int64_t* hi) {
        const int64_t base = N / P;
        const int64_t rem  = N % P;
        *lo                = (int64_t)i * base + std::min((int64_t)i, rem);
        *hi                = *lo + base + (i < rem ? 1 : 0);
    }

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

    // every rank read its own rows of the IC file; cut the curve evenly, then send each cell to its owner
    void distribute_ic_parallel(ICData& ic) {
        const int n_local_in = (int)ic.header.n_seeds;
        const int my_rank    = decomp.rank;
        const int nr         = decomp.nranks;

        if (nr <= 1) {
            if (my_rank == 0) printf("DECOMP: single-rank, n_local=%d (no routing)\n", n_local_in);
            return;
        }

        // the same number of cells on every rank
        {
            std::vector<uint64_t> keys((size_t)n_local_in);
            for (int k = 0; k < n_local_in; k++)
                keys[k] = knn::hilbert_key(ic_point(ic, k));
            std::vector<uint64_t> cuts;
            decomp_balanced_cuts(keys, &cuts);
            decomp_set_cuts(cuts.data());
        }

        // owner of every cell we read, and how many go to each rank
        std::vector<int> send_counts(nr, 0);
        std::vector<int> per_cell_dest(n_local_in, -1);
        for (int k = 0; k < n_local_in; k++) {
            const int owner  = owner_of_point(ic_point(ic, k), decomp.cuts, nr);
            per_cell_dest[k] = owner;
            send_counts[owner]++;
        }

        // tell every rank how much it will get
        std::vector<int> recv_counts(nr, 0);
        {
            PROFILE_MPI("ICDIST_COUNTS_WAIT");
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, decomp.comm);
        }

        std::vector<int> send_displs(nr, 0);
        std::vector<int> recv_displs(nr, 0);
        int              total_send = 0, total_recv = 0;
        for (int r = 0; r < nr; r++) {
            send_displs[r] = total_send;
            recv_displs[r] = total_recv;
            total_send += send_counts[r];
            total_recv += recv_counts[r];
        }

        // cells sorted by their target rank
        std::vector<ICMigrant> sendbuf((size_t)total_send);
        std::vector<int>       cursor = send_displs;
        for (int k = 0; k < n_local_in; k++) {
            const int  dest = per_cell_dest[k];
            const int  slot = cursor[dest]++;
            ICMigrant& m    = sendbuf[slot];
            for (int d = 0; d < DIMENSION; d++) {
                m.pos[d] = ic.pos[DIMENSION * k + d];
                m.vel[d] = ic.vel[DIMENSION * k + d];
            }
            m.rho    = ic.rho[k];
            m.energy = ic.energy[k];
        }

        MPI_Datatype ic_migrant_t;
        MPI_Type_contiguous(sizeof(ICMigrant), MPI_BYTE, &ic_migrant_t);
        MPI_Type_commit(&ic_migrant_t);

        std::vector<ICMigrant> recvbuf((size_t)total_recv);
        {
            PROFILE_MPI("ICDIST_PAYLOAD_WAIT");
            MPI_Alltoallv(sendbuf.data(),
                          send_counts.data(),
                          send_displs.data(),
                          ic_migrant_t,
                          recvbuf.data(),
                          recv_counts.data(),
                          recv_displs.data(),
                          ic_migrant_t,
                          decomp.comm);
        }
        MPI_Type_free(&ic_migrant_t);

        // what came back is the new content of ic_data
        const int n_local_out = total_recv;
        ic.pos.resize((size_t)DIMENSION * n_local_out);
        ic.vel.resize((size_t)DIMENSION * n_local_out);
        ic.rho.resize(n_local_out);
        ic.energy.resize(n_local_out);
        for (int j = 0; j < n_local_out; j++) {
            const ICMigrant& m = recvbuf[j];
            for (int d = 0; d < DIMENSION; d++) {
                ic.pos[DIMENSION * j + d] = m.pos[d];
                ic.vel[DIMENSION * j + d] = m.vel[d];
            }
            ic.rho[j]    = m.rho;
            ic.energy[j] = m.energy;
        }
        ic.header.n_seeds = (uint64_t)n_local_out;

        printf("DECOMP: rank %d routed IC: read %d, kept %d (self %d, recv %d, sent %d)\n",
               my_rank,
               n_local_in,
               n_local_out,
               send_counts[my_rank],
               n_local_out - send_counts[my_rank],
               total_send - send_counts[my_rank]);
        fflush(stdout);
        print_cuts("IC");

        // no cell may be lost on the way
        const long long n_local_out_ll = (long long)n_local_out;
        const long long n_local_in_ll  = (long long)n_local_in;
        long long       n_global_kept  = 0;
        long long       n_total_in_ll  = 0;
        {
            PROFILE_MPI("ICDIST_CONS_ALLREDUCE");
            MPI_Allreduce(&n_local_out_ll, &n_global_kept, 1, MPI_LONG_LONG, MPI_SUM, decomp.comm);
            MPI_Allreduce(&n_local_in_ll, &n_total_in_ll, 1, MPI_LONG_LONG, MPI_SUM, decomp.comm);
        }
        if (n_global_kept != n_total_in_ll) {
            exit_failure("DECOMP: FATAL parallel-IC cell-count mismatch — received-sum=%lld, sent-sum=%lld.\n",
                         n_global_kept,
                         n_total_in_ll);
        }
        if (my_rank == 0) {
            printf("DECOMP: parallel-IC cell-count check passed (sum of per-rank n_local = %lld).\n", n_global_kept);
            fflush(stdout);
        }
    }

#else

    void distribute_ic_parallel(ICData& ic) {
        (void)ic;
    }

#endif

#ifdef USE_MPI
    // where the curve is cut, as a share of it
    static void print_cuts(const char* what) {
        if (decomp.rank != 0) return;
        printf("DECOMP: %s cuts along the Hilbert curve:", what);
        for (int r = 0; r <= decomp.nranks && r <= 16; r++)
            printf(" %.4f", (double)decomp.cuts[r] / (double)knn::DOMAIN_KEY_END);
        printf("%s\n", decomp.nranks > 16 ? " ..." : "");
        fflush(stdout);
    }
#endif

} // namespace proteus_mpi
