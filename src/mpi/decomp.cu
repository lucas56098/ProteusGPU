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

    // forward declarations
    static int  compute_global_N_grid(int64_t n_total, double buff);
    static void even_split(int N, int P, int i, int* lo, int* hi);
    static void create_cart_topology();
    static void allocate_split_tables();
    static void allocate_coord_to_rank();
    static void fill_coord_to_rank();
    static void init_splits_even(int N);
    static void apply_splits_for_this_rank();
    static void check_bricks_nonempty(int N);
    static int  keep_owned_cells(ICData& ic, int N_grid, double buff);
    static void resize_ic_to(ICData& ic, int n_kept);
    static void check_global_cell_count(int n_kept, int n_total);

    // ============================================================
    // Public entry points
    // ============================================================

    void decomp_init(int64_t n_total, double buff) {
        decomp.rank          = rank();
        decomp.nranks        = nranks();
        decomp.N_grid_global = compute_global_N_grid(n_total, buff);

        create_cart_topology();
        allocate_split_tables();
        allocate_coord_to_rank();
        init_splits_even(decomp.N_grid_global);
        apply_splits_for_this_rank();
        fill_coord_to_rank();
        check_bricks_nonempty(decomp.N_grid_global);

        if (decomp.rank == 0) {
            printf("DECOMP: dims=[%d,%d,%d] N_grid_global=%d\n",
                   decomp.dims[0],
                   decomp.dims[1],
                   decomp.dims[2],
                   decomp.N_grid_global);
        }
        printf("DECOMP: rank %d/%d coords=[%d,%d,%d] brick=[%d,%d) x [%d,%d) x [%d,%d)\n",
               decomp.rank,
               decomp.nranks,
               decomp.coords[0],
               decomp.coords[1],
               decomp.coords[2],
               decomp.b0[0],
               decomp.b1[0],
               decomp.b0[1],
               decomp.b1[1],
               decomp.b0[2],
               decomp.b1[2]);
        fflush(stdout);
    }

    void decomp_apply_splits(const int* sx, const int* sy, const int* sz) {
        const int dx = decomp.dims[0];
        const int dy = decomp.dims[1];
        const int dz = decomp.dims[2];
        for (int i = 0; i <= dx; i++)
            decomp.splits[0][i] = sx[i];
        for (int i = 0; i <= dy; i++)
            decomp.splits[1][i] = sy[i];
        for (int i = 0; i <= dz; i++)
            decomp.splits[2][i] = sz[i];
        apply_splits_for_this_rank();
        // dims don't change, but rebuilding the coord_to_rank table is a no-op
        // in cost terms (small table) and keeps the invariant that it always
        // matches the live Cart topology — cheaper than tracking when it can stale.
        fill_coord_to_rank();
        check_bricks_nonempty(decomp.N_grid_global);
    }

    int decomp_owner_of_bucket(int bx, int by, int bz) {
        // Routes through the same coord_to_rank table used by the device path; both
        // host and device callers see one source of truth for the coord -> rank map.
        return decomp_owner_of_bucket_dev(bx,
                                          by,
                                          bz,
                                          decomp.N_grid_global,
                                          decomp.dims[0],
                                          decomp.dims[1],
                                          decomp.dims[2],
                                          decomp.splits[0],
                                          decomp.splits[1],
                                          decomp.splits[2],
                                          decomp.coord_to_rank);
    }

    void distribute_ic_local(ICData& ic, double buff) {
        const int n_total = (int)ic.pos_dims[0];

        // sequential global IDs in input order (every rank reads the full IC)
        ic.global_id.resize(n_total);
        for (int i = 0; i < n_total; i++)
            ic.global_id[i] = (uint64_t)i;

        if (decomp.nranks <= 1) {
            if (decomp.rank == 0) printf("DECOMP: single-rank, n_local=%d (no filtering)\n", n_total);
            return;
        }

        const int n_kept = keep_owned_cells(ic, decomp.N_grid_global, buff);
        resize_ic_to(ic, n_kept);

        printf("DECOMP: rank %d kept %d / %d cells\n", decomp.rank, n_kept, n_total);
        fflush(stdout);

        check_global_cell_count(n_kept, n_total);
    }

    void decomp_even_split(int64_t N, int P, int i, int64_t* lo, int64_t* hi) {
        const int64_t base = N / P;
        const int64_t rem  = N % P;
        *lo                = (int64_t)i * base + std::min((int64_t)i, rem);
        *hi                = *lo + base + (i < rem ? 1 : 0);
    }

#ifdef USE_MPI

    // per-particle payload shipped through Alltoallv. one struct per cell — packs all IC fields
    // (pos, vel, rho, energy, global_id) so a single Alltoallv covers them.
    struct ICMigrant {
        double   pos[DIMENSION];
        double   vel[DIMENSION];
        double   rho;
        double   energy;
        uint64_t global_id;
    };

    void distribute_ic_parallel(ICData& ic, double buff) {
        const int n_local_in = (int)ic.pos_dims[0];
        const int my_rank    = decomp.rank;
        const int nr         = decomp.nranks;
        const int N_grid     = decomp.N_grid_global;

        // single-rank: nothing to route, all cells stay here
        if (nr <= 1) {
            if (my_rank == 0) printf("DECOMP: single-rank, n_local=%d (no routing)\n", n_local_in);
            return;
        }

        // assign destination rank per input cell
        std::vector<int> send_counts(nr, 0);
        std::vector<int> per_cell_dest(n_local_in, -1);
        for (int k = 0; k < n_local_in; k++) {
            const double px = ic.pos[DIMENSION * k + 0];
            const double py = ic.pos[DIMENSION * k + 1];
#ifdef dim_3D
            const double pz = ic.pos[DIMENSION * k + 2];
#else
            const double pz = 0.0;
#endif
            int bx, by, bz;
            decomp_bucket_of_point(px, py, pz, N_grid, buff, &bx, &by, &bz);
            const int owner = decomp_owner_of_bucket(bx, by, bz);
            if (owner < 0) {
                exit_failure("[rank %d] DECOMP: invalid owner for IC cell %d at (%g,%g,%g) → bucket (%d,%d,%d).\n",
                             my_rank,
                             k,
                             px,
                             py,
                             pz,
                             bx,
                             by,
                             bz);
            }
            per_cell_dest[k] = owner;
            send_counts[owner]++;
        }

        // exchange counts
        std::vector<int> recv_counts(nr, 0);
        {
            PROFILE_MPI("ICDIST_COUNTS_WAIT");
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, decomp.cart_comm);
        }

        // displacements + totals
        std::vector<int> send_displs(nr, 0);
        std::vector<int> recv_displs(nr, 0);
        int              total_send = 0, total_recv = 0;
        for (int r = 0; r < nr; r++) {
            send_displs[r] = total_send;
            recv_displs[r] = total_recv;
            total_send += send_counts[r];
            total_recv += recv_counts[r];
        }

        // pack outgoing
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
            m.rho       = ic.rho[k];
            m.energy    = ic.energy[k];
            m.global_id = ic.global_id[k];
        }

        // MPI datatype for ICMigrant — byte-blob, layout doesn't matter as long as senders and
        // receivers agree on sizeof(ICMigrant). Built locally to avoid a static dependency.
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
                          decomp.cart_comm);
        }
        MPI_Type_free(&ic_migrant_t);

        // unpack into ic. Self-sends pass through recvbuf as well (dest = my_rank routes through
        // MPI_Alltoallv on the local rank), so total_recv already accounts for cells we keep.
        const int n_local_out = total_recv;
        ic.pos.resize((size_t)DIMENSION * n_local_out);
        ic.vel.resize((size_t)DIMENSION * n_local_out);
        ic.rho.resize(n_local_out);
        ic.energy.resize(n_local_out);
        ic.global_id.resize(n_local_out);
        for (int j = 0; j < n_local_out; j++) {
            const ICMigrant& m = recvbuf[j];
            for (int d = 0; d < DIMENSION; d++) {
                ic.pos[DIMENSION * j + d] = m.pos[d];
                ic.vel[DIMENSION * j + d] = m.vel[d];
            }
            ic.rho[j]       = m.rho;
            ic.energy[j]    = m.energy;
            ic.global_id[j] = m.global_id;
        }
        ic.pos_dims[0] = (hsize_t)n_local_out;

        printf("DECOMP: rank %d routed IC: read %d, kept %d (self %d, recv %d, sent %d)\n",
               my_rank,
               n_local_in,
               n_local_out,
               send_counts[my_rank],
               n_local_out - send_counts[my_rank],
               total_send - send_counts[my_rank]);
        fflush(stdout);

        // conservation across all ranks — long long because the global sum is n_global
        // (few_thousand^3) which overflows int32 from ~1300^3 upward.
        const long long n_local_out_ll = (long long)n_local_out;
        const long long n_local_in_ll  = (long long)n_local_in;
        long long       n_global_kept  = 0;
        {
            PROFILE_MPI("ICDIST_CONS_ALLREDUCE");
            MPI_Allreduce(&n_local_out_ll, &n_global_kept, 1, MPI_LONG_LONG, MPI_SUM, decomp.cart_comm);
        }

        long long n_total_in_ll = 0;
        {
            PROFILE_MPI("ICDIST_CONS_ALLREDUCE");
            MPI_Allreduce(&n_local_in_ll, &n_total_in_ll, 1, MPI_LONG_LONG, MPI_SUM, decomp.cart_comm);
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

#else // !USE_MPI

    void distribute_ic_parallel(ICData& ic, double buff) {
        (void)ic;
        (void)buff;
    }

#endif // USE_MPI

    // ============================================================
    // Static helpers
    // ============================================================

    // same N_grid formula as knn::init_once, but using the global cell count so
    // every rank agrees. N_grid is the bucket grid resolution per axis (~few thousand
    // for our target scale), so the return value stays int even though the input
    // global cell count is int64.
    static int compute_global_N_grid(int64_t n_total, double buff) {
        double ghost_frac  = std::pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        double max_n_total = (double)n_total + 2.0 * ghost_frac * (double)n_total + 1.0;
        int    N           = (int)std::round(std::pow(max_n_total / 3.1, 1.0 / (double)DIMENSION));
        if (N < 1) N = 1;
        return N;
    }

    // even split of N items across P slots; slot i gets the i'th contiguous chunk
    static void even_split(int N, int P, int i, int* lo, int* hi) {
        int base = N / P;
        int rem  = N % P;
        *lo      = i * base + std::min(i, rem);
        *hi      = *lo + base + (i < rem ? 1 : 0);
    }

    static void create_cart_topology() {
#ifdef USE_MPI
        // 2D forces Pz = 1
        int dims[3] = {0, 0, 0};
#ifdef dim_2D
        dims[2]    = 1;
        int active = 2;
#else
        int active = 3;
#endif
        MPI_Dims_create(decomp.nranks, active, dims);
        if (active == 2) dims[2] = 1;

        int periods[3] = {1, 1, 1}; // periodic Cart matches physical BCs
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, /*reorder=*/0, &decomp.cart_comm);

        int coords[3] = {0, 0, 0};
        MPI_Cart_coords(decomp.cart_comm, decomp.rank, 3, coords);

        for (int a = 0; a < 3; a++) {
            decomp.dims[a]   = dims[a];
            decomp.coords[a] = coords[a];
        }
#else
        for (int a = 0; a < 3; a++) {
            decomp.dims[a]   = 1;
            decomp.coords[a] = 0;
        }
#endif
    }

    // allocate the three per-axis split tables. dims must be set by create_cart_topology first.
    // Managed memory so device kernels can read them during pack/migrate/rebalance.
    static void allocate_split_tables() {
        for (int a = 0; a < 3; a++) {
            const int n      = decomp.dims[a] + 1;
            decomp.splits[a] = gpu_alloc<int>((size_t)n);
            for (int i = 0; i < n; i++)
                decomp.splits[a][i] = 0;
        }
    }

    // allocate the coord -> rank lookup table (managed). dims must be set first.
    static void allocate_coord_to_rank() {
        const size_t n       = (size_t)decomp.dims[0] * (size_t)decomp.dims[1] * (size_t)decomp.dims[2];
        decomp.coord_to_rank = gpu_alloc<int>(n);
        for (size_t i = 0; i < n; i++)
            decomp.coord_to_rank[i] = 0;
    }

    // fill coord_to_rank[cx,cy,cz] from MPI_Cart_rank. Single-rank build: every entry is 0.
    static void fill_coord_to_rank() {
        const int dx = decomp.dims[0];
        const int dy = decomp.dims[1];
        const int dz = decomp.dims[2];
        for (int cx = 0; cx < dx; cx++) {
            for (int cy = 0; cy < dy; cy++) {
                for (int cz = 0; cz < dz; cz++) {
                    const int idx = (cx * dy + cy) * dz + cz;
#ifdef USE_MPI
                    int coords[3] = {cx, cy, cz};
                    int owner     = 0;
                    MPI_Cart_rank(decomp.cart_comm, coords, &owner);
                    decomp.coord_to_rank[idx] = owner;
#else
                    decomp.coord_to_rank[idx] = 0;
#endif
                }
            }
        }
    }

    // populate splits[a] with the even_split positions for axis a so the initial decomposition
    // matches the pre-rebalance behaviour bit-for-bit.
    static void init_splits_even(int N) {
        for (int a = 0; a < 3; a++) {
            const int P         = decomp.dims[a];
            decomp.splits[a][0] = 0;
            for (int c = 0; c < P; c++) {
                int lo, hi;
                even_split(N, P, c, &lo, &hi);
                decomp.splits[a][c + 1] = hi;
            }
        }
#ifndef dim_3D
        // 2D: lookups force bz=0, so the z slab collapses to a single bucket.
        decomp.splits[2][0] = 0;
        decomp.splits[2][1] = 1;
#endif
    }

    // derive this rank's b0/b1 from the current splits and Cart coords.
    static void apply_splits_for_this_rank() {
        for (int a = 0; a < 3; a++) {
            const int c  = decomp.coords[a];
            decomp.b0[a] = decomp.splits[a][c];
            decomp.b1[a] = decomp.splits[a][c + 1];
        }
    }

    static void check_bricks_nonempty(int N) {
        for (int a = 0; a < 3; a++) {
            if (decomp.b1[a] <= decomp.b0[a]) {
                exit_failure("[rank %d] DECOMP: axis %d brick is empty (b0=%d b1=%d, N_grid=%d, dims=%d). "
                             "Reduce nranks or use a larger IC.\n",
                             decomp.rank,
                             a,
                             decomp.b0[a],
                             decomp.b1[a],
                             N,
                             decomp.dims[a]);
            }
        }
    }

    // in-place compaction: keep only cells whose bucket lies in this rank's brick
    static int keep_owned_cells(ICData& ic, int N_grid, double buff) {
        const int n_total = (int)ic.pos_dims[0];
        int       n_kept  = 0;
        for (int i = 0; i < n_total; i++) {
            double px = ic.pos[DIMENSION * i + 0];
            double py = ic.pos[DIMENSION * i + 1];
#ifdef dim_3D
            double pz = ic.pos[DIMENSION * i + 2];
#else
            double pz = 0.0;
#endif

            int bx, by, bz;
            decomp_bucket_of_point(px, py, pz, N_grid, buff, &bx, &by, &bz);

            if (!decomp_owns_bucket(bx, by, bz)) continue;

            if (n_kept != i) {
                for (int d = 0; d < DIMENSION; d++) {
                    ic.pos[DIMENSION * n_kept + d] = ic.pos[DIMENSION * i + d];
                    ic.vel[DIMENSION * n_kept + d] = ic.vel[DIMENSION * i + d];
                }
                ic.rho[n_kept]       = ic.rho[i];
                ic.energy[n_kept]    = ic.energy[i];
                ic.global_id[n_kept] = ic.global_id[i];
            }
            n_kept++;
        }
        return n_kept;
    }

    static void resize_ic_to(ICData& ic, int n_kept) {
        ic.pos.resize((size_t)DIMENSION * n_kept);
        ic.vel.resize((size_t)DIMENSION * n_kept);
        ic.rho.resize(n_kept);
        ic.energy.resize(n_kept);
        ic.global_id.resize(n_kept);
        ic.pos_dims[0] = (hsize_t)n_kept;
    }

    // conservation: sum of per-rank n_kept must equal global n_total
    static void check_global_cell_count(int n_kept, int n_total) {
#ifdef USE_MPI
        const long long n_kept_ll     = (long long)n_kept;
        long long       n_global_kept = 0;
        {
            PROFILE_MPI("ICDIST_CONS_ALLREDUCE");
            MPI_Allreduce(&n_kept_ll, &n_global_kept, 1, MPI_LONG_LONG, MPI_SUM, decomp.cart_comm);
        }
        if (n_global_kept != (long long)n_total) {
            exit_failure(
                "DECOMP: FATAL cell-count mismatch — sum(n_kept) = %lld, expected %d.\n", n_global_kept, n_total);
        }
        if (decomp.rank == 0) {
            printf("DECOMP: cell-count check passed (sum of per-rank n_kept = %lld).\n", n_global_kept);
            fflush(stdout);
        }
#else
        (void)n_kept;
        (void)n_total;
#endif
    }

} // namespace proteus_mpi
