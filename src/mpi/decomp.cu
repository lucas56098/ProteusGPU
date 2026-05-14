#include "decomp.h"

#include "io/input.h"
#include "mpi_init.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace proteus_mpi {

MpiDecomp g_decomp = {};

// same N_grid formula as knn::init_once, but using the global cell count so
// every rank agrees
static int compute_global_N_grid(int n_total, double buff) {
    double ghost_frac  = std::pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
    int    max_n_total = (int)((double)n_total + 2.0 * ghost_frac * (double)n_total) + 1;
    int    N           = (int)std::round(std::pow((double)max_n_total / 3.1, 1.0 / (double)DIMENSION));
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

void decomp_init(int n_total, double buff) {
    g_decomp.rank          = rank();
    g_decomp.nranks        = nranks();
    g_decomp.N_grid_global = compute_global_N_grid(n_total, buff);

#ifdef USE_MPI
    // Cartesian factorization; 2D forces Pz=1
    int dims[3] = {0, 0, 0};
#ifdef dim_2D
    dims[2]    = 1;
    int active = 2;
#else
    int active = 3;
#endif
    MPI_Dims_create(g_decomp.nranks, active, dims);
    if (active == 2) dims[2] = 1;

    int periods[3] = {1, 1, 1};  // periodic Cart matches physical BCs
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, /*reorder=*/0, &g_decomp.cart_comm);

    int coords[3] = {0, 0, 0};
    MPI_Cart_coords(g_decomp.cart_comm, g_decomp.rank, 3, coords);

    g_decomp.dims[0]   = dims[0];
    g_decomp.dims[1]   = dims[1];
    g_decomp.dims[2]   = dims[2];
    g_decomp.coords[0] = coords[0];
    g_decomp.coords[1] = coords[1];
    g_decomp.coords[2] = coords[2];
#else
    g_decomp.dims[0]   = 1;
    g_decomp.dims[1]   = 1;
    g_decomp.dims[2]   = 1;
    g_decomp.coords[0] = 0;
    g_decomp.coords[1] = 0;
    g_decomp.coords[2] = 0;
#endif

    // partition the global N_grid buckets along each axis among the Cart dims
    int N = g_decomp.N_grid_global;
    even_split(N, g_decomp.dims[0], g_decomp.coords[0], &g_decomp.b0[0], &g_decomp.b1[0]);
    even_split(N, g_decomp.dims[1], g_decomp.coords[1], &g_decomp.b0[1], &g_decomp.b1[1]);
#ifdef dim_3D
    even_split(N, g_decomp.dims[2], g_decomp.coords[2], &g_decomp.b0[2], &g_decomp.b1[2]);
#else
    g_decomp.b0[2] = 0;
    g_decomp.b1[2] = 1;
#endif

    for (int a = 0; a < 3; a++) {
        if (g_decomp.b1[a] <= g_decomp.b0[a]) {
            fprintf(stderr,
                    "[rank %d] DECOMP: axis %d brick is empty (b0=%d b1=%d, N_grid=%d, dims=%d). "
                    "Reduce nranks or use a larger IC.\n",
                    g_decomp.rank, a, g_decomp.b0[a], g_decomp.b1[a], N, g_decomp.dims[a]);
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            std::exit(EXIT_FAILURE);
#endif
        }
    }

    if (g_decomp.rank == 0) {
        printf("DECOMP: dims=[%d,%d,%d] N_grid_global=%d\n", g_decomp.dims[0], g_decomp.dims[1], g_decomp.dims[2], N);
    }
    printf("DECOMP: rank %d/%d coords=[%d,%d,%d] brick=[%d,%d) x [%d,%d) x [%d,%d)\n", g_decomp.rank, g_decomp.nranks,
           g_decomp.coords[0], g_decomp.coords[1], g_decomp.coords[2], g_decomp.b0[0], g_decomp.b1[0], g_decomp.b0[1],
           g_decomp.b1[1], g_decomp.b0[2], g_decomp.b1[2]);
    fflush(stdout);
}

int decomp_owner_of_bucket(int bx, int by, int bz) {
    int N = g_decomp.N_grid_global;
    if (bx < 0 || bx >= N || by < 0 || by >= N) return -1;
#ifdef dim_3D
    if (bz < 0 || bz >= N) return -1;
#else
    bz = 0;
#endif

    auto find_coord = [&](int b, int dim_size) {
        int base = N / dim_size;
        int rem  = N % dim_size;
        int lo   = 0;
        for (int c = 0; c < dim_size; c++) {
            int hi = lo + base + (c < rem ? 1 : 0);
            if (b >= lo && b < hi) return c;
            lo = hi;
        }
        return -1;
    };

    int cx = find_coord(bx, g_decomp.dims[0]);
    int cy = find_coord(by, g_decomp.dims[1]);
    int cz = find_coord(bz, g_decomp.dims[2]);
    if (cx < 0 || cy < 0 || cz < 0) return -1;

#ifdef USE_MPI
    int coords[3] = {cx, cy, cz};
    int owner     = 0;
    MPI_Cart_rank(g_decomp.cart_comm, coords, &owner);
    return owner;
#else
    return 0;
#endif
}

void distribute_ic_local(ICData& ic, double buff) {
    const int n_total = (int)ic.pos_dims[0];

    // assign sequential global IDs in input order (every rank reads the full IC)
    ic.global_id.resize(n_total);
    for (int i = 0; i < n_total; i++) ic.global_id[i] = (uint64_t)i;

    if (g_decomp.nranks <= 1) {
        if (g_decomp.rank == 0) printf("DECOMP: single-rank, n_local=%d (no filtering)\n", n_total);
        return;
    }

    const int N_grid = g_decomp.N_grid_global;

    int n_kept = 0;
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

        // in-place compaction: copy slot i down to slot n_kept
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

    ic.pos.resize((size_t)DIMENSION * n_kept);
    ic.vel.resize((size_t)DIMENSION * n_kept);
    ic.rho.resize(n_kept);
    ic.energy.resize(n_kept);
    ic.global_id.resize(n_kept);
    ic.pos_dims[0] = (hsize_t)n_kept;

    printf("DECOMP: rank %d kept %d / %d cells\n", g_decomp.rank, n_kept, n_total);
    fflush(stdout);

#ifdef USE_MPI
    // cell-count conservation: sum of per-rank n_kept must equal global n_total
    int n_global_kept = 0;
    MPI_Allreduce(&n_kept, &n_global_kept, 1, MPI_INT, MPI_SUM, g_decomp.cart_comm);
    if (n_global_kept != n_total) {
        if (g_decomp.rank == 0) {
            fprintf(stderr,
                    "DECOMP: FATAL cell-count mismatch — sum(n_kept) = %d, expected %d.\n",
                    n_global_kept, n_total);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (g_decomp.rank == 0) {
        printf("DECOMP: cell-count check passed (sum of per-rank n_kept = %d).\n", n_global_kept);
        fflush(stdout);
    }
#endif
}

}  // namespace proteus_mpi
