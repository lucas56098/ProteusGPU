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
static int  compute_global_N_grid(int n_total, double buff);
static void even_split(int N, int P, int i, int* lo, int* hi);
static void create_cart_topology();
static void split_brick_per_axis(int N);
static void check_bricks_nonempty(int N);
static int  keep_owned_cells(ICData& ic, int N_grid, double buff);
static void resize_ic_to(ICData& ic, int n_kept);
static void check_global_cell_count(int n_kept, int n_total);

// ============================================================
// Public entry points
// ============================================================

void decomp_init(int n_total, double buff) {
    decomp.rank          = rank();
    decomp.nranks        = nranks();
    decomp.N_grid_global = compute_global_N_grid(n_total, buff);

    create_cart_topology();
    split_brick_per_axis(decomp.N_grid_global);
    check_bricks_nonempty(decomp.N_grid_global);

    if (decomp.rank == 0) {
        printf("DECOMP: dims=[%d,%d,%d] N_grid_global=%d\n",
               decomp.dims[0], decomp.dims[1], decomp.dims[2], decomp.N_grid_global);
    }
    printf("DECOMP: rank %d/%d coords=[%d,%d,%d] brick=[%d,%d) x [%d,%d) x [%d,%d)\n",
           decomp.rank, decomp.nranks,
           decomp.coords[0], decomp.coords[1], decomp.coords[2],
           decomp.b0[0], decomp.b1[0], decomp.b0[1], decomp.b1[1], decomp.b0[2], decomp.b1[2]);
    fflush(stdout);
}

int decomp_owner_of_bucket(int bx, int by, int bz) {
    const int N = decomp.N_grid_global;
    if (bx < 0 || bx >= N || by < 0 || by >= N) return -1;
#ifdef dim_3D
    if (bz < 0 || bz >= N) return -1;
#else
    bz = 0;
#endif

    auto coord_of_bucket = [&](int b, int dim_size) {
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

    const int cx = coord_of_bucket(bx, decomp.dims[0]);
    const int cy = coord_of_bucket(by, decomp.dims[1]);
    const int cz = coord_of_bucket(bz, decomp.dims[2]);
    if (cx < 0 || cy < 0 || cz < 0) return -1;

#ifdef USE_MPI
    int coords[3] = {cx, cy, cz};
    int owner     = 0;
    MPI_Cart_rank(decomp.cart_comm, coords, &owner);
    return owner;
#else
    return 0;
#endif
}

void distribute_ic_local(ICData& ic, double buff) {
    const int n_total = (int)ic.pos_dims[0];

    // sequential global IDs in input order (every rank reads the full IC)
    ic.global_id.resize(n_total);
    for (int i = 0; i < n_total; i++) ic.global_id[i] = (uint64_t)i;

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

// ============================================================
// Static helpers
// ============================================================

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

    int periods[3] = {1, 1, 1};  // periodic Cart matches physical BCs
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

// partition global N buckets along each Cart axis
static void split_brick_per_axis(int N) {
    even_split(N, decomp.dims[0], decomp.coords[0], &decomp.b0[0], &decomp.b1[0]);
    even_split(N, decomp.dims[1], decomp.coords[1], &decomp.b0[1], &decomp.b1[1]);
#ifdef dim_3D
    even_split(N, decomp.dims[2], decomp.coords[2], &decomp.b0[2], &decomp.b1[2]);
#else
    decomp.b0[2] = 0;
    decomp.b1[2] = 1;
#endif
}

static void check_bricks_nonempty(int N) {
    for (int a = 0; a < 3; a++) {
        if (decomp.b1[a] <= decomp.b0[a]) {
            exit_failure("[rank %d] DECOMP: axis %d brick is empty (b0=%d b1=%d, N_grid=%d, dims=%d). "
                "Reduce nranks or use a larger IC.\n",
                decomp.rank, a, decomp.b0[a], decomp.b1[a], N, decomp.dims[a]);
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
    int n_global_kept = 0;
    Profiler::StartTimer("MPI_REDUCE");
    MPI_Allreduce(&n_kept, &n_global_kept, 1, MPI_INT, MPI_SUM, decomp.cart_comm);
    Profiler::EndTimer("MPI_REDUCE");
    if (n_global_kept != n_total) {
        exit_failure("DECOMP: FATAL cell-count mismatch — sum(n_kept) = %d, expected %d.\n", n_global_kept, n_total);
    }
    if (decomp.rank == 0) {
        printf("DECOMP: cell-count check passed (sum of per-rank n_kept = %d).\n", n_global_kept);
        fflush(stdout);
    }
#else
    (void)n_kept;
    (void)n_total;
#endif
}

}  // namespace proteus_mpi
