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

    static int  compute_global_N_grid(int64_t n_total, double buff);
    static void even_split(int N, int P, int i, int* lo, int* hi);
    static void create_cart_topology();
    static void allocate_split_tables();
    static void allocate_coord_to_rank();
    static void fill_coord_to_rank();
    static void init_splits_even(int N);
    static void apply_splits_for_this_rank();
    static void check_bricks_nonempty(int N);

    // bucket grid, Cartesian rank grid, and an even brick per rank
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

    // new split tables for every rank; all ranks pass the same ones
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
        fill_coord_to_rank();
        check_bricks_nonempty(decomp.N_grid_global);
    }

    int decomp_owner_of_bucket(int bx, int by, int bz) {
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

    void decomp_even_split(int64_t N, int P, int i, int64_t* lo, int64_t* hi) {
        const int64_t base = N / P;
        const int64_t rem  = N % P;
        *lo                = (int64_t)i * base + std::min((int64_t)i, rem);
        *hi                = *lo + base + (i < rem ? 1 : 0);
    }

#ifdef USE_MPI

    // one IC cell on its way to the rank that owns it
    struct ICMigrant {
        double   pos[DIMENSION];
        double   vel[DIMENSION];
        double   rho;
        double   energy;
        uint64_t global_id;
    };

    // every rank read its own rows of the IC file, this sends each cell to its owner
    void distribute_ic_parallel(ICData& ic, double buff) {
        const int n_local_in = (int)ic.header.n_seeds;
        const int my_rank    = decomp.rank;
        const int nr         = decomp.nranks;
        const int N_grid     = decomp.N_grid_global;

        if (nr <= 1) {
            if (my_rank == 0) printf("DECOMP: single-rank, n_local=%d (no routing)\n", n_local_in);
            return;
        }

        // owner of every cell we read, and how many go to each rank
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

        // tell every rank how much it will get
        std::vector<int> recv_counts(nr, 0);
        {
            PROFILE_MPI("ICDIST_COUNTS_WAIT");
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, decomp.cart_comm);
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
            m.rho       = ic.rho[k];
            m.energy    = ic.energy[k];
            m.global_id = ic.global_id[k];
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
                          decomp.cart_comm);
        }
        MPI_Type_free(&ic_migrant_t);

        // what came back is the new content of ic_data
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
        ic.header.n_seeds = (uint64_t)n_local_out;

        printf("DECOMP: rank %d routed IC: read %d, kept %d (self %d, recv %d, sent %d)\n",
               my_rank,
               n_local_in,
               n_local_out,
               send_counts[my_rank],
               n_local_out - send_counts[my_rank],
               total_send - send_counts[my_rank]);
        fflush(stdout);

        // no cell may be lost on the way
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

#else

    void distribute_ic_parallel(ICData& ic, double buff) {
        (void)ic;
        (void)buff;
    }

#endif

    // buckets per axis, at about 3 cells per bucket
    static int compute_global_N_grid(int64_t n_total, double buff) {
        double ghost_frac  = std::pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        double max_n_total = (double)n_total + 2.0 * ghost_frac * (double)n_total + 1.0;
        int    N           = (int)std::round(std::pow(max_n_total / 3.1, 1.0 / (double)DIMENSION));
        if (N < 1) N = 1;
        return N;
    }

    static void even_split(int N, int P, int i, int* lo, int* hi) {
        int base = N / P;
        int rem  = N % P;
        *lo      = i * base + std::min(i, rem);
        *hi      = *lo + base + (i < rem ? 1 : 0);
    }

    // rank grid from MPI, periodic on every axis
    static void create_cart_topology() {
#ifdef USE_MPI
        int dims[3] = {0, 0, 0};
#ifdef dim_2D
        dims[2]    = 1;
        int active = 2;
#else
        int active = 3;
#endif
        MPI_Dims_create(decomp.nranks, active, dims);
        if (active == 2) dims[2] = 1;

        int periods[3] = {1, 1, 1};
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &decomp.cart_comm);

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

    static void allocate_split_tables() {
        for (int a = 0; a < 3; a++) {
            const int n      = decomp.dims[a] + 1;
            decomp.splits[a] = gpu_alloc<int>((size_t)n);
            for (int i = 0; i < n; i++)
                decomp.splits[a][i] = 0;
        }
    }

    static void allocate_coord_to_rank() {
        const size_t n       = (size_t)decomp.dims[0] * (size_t)decomp.dims[1] * (size_t)decomp.dims[2];
        decomp.coord_to_rank = gpu_alloc<int>(n);
        for (size_t i = 0; i < n; i++)
            decomp.coord_to_rank[i] = 0;
    }

    // coords -> rank, so a device kernel can look the owner up
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

    // same number of buckets for every rank
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
        decomp.splits[2][0] = 0;
        decomp.splits[2][1] = 1;
#endif
    }

    // the brick this rank owns
    static void apply_splits_for_this_rank() {
        for (int a = 0; a < 3; a++) {
            const int c  = decomp.coords[a];
            decomp.b0[a] = decomp.splits[a][c];
            decomp.b1[a] = decomp.splits[a][c + 1];
        }
    }

    // with too many ranks a brick can end up without a single bucket
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

} // namespace proteus_mpi
