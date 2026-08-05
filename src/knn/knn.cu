#include "../global/globals.h"
#include "../global/structs.h"
#include "../profiler/profiler.h"
#include "knn.h"
#include <iostream>

namespace knn {

    // forward declarations
    static void sort_points_into_grid(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

#ifndef CPU_DEBUG
    // kernels
    GLOBAL void kernel_count_cells(const POINT_TYPE*, int, int, const double*, double, int*);
    GLOBAL void kernel_compute_ptrs(int*, int*, int*, int);
    GLOBAL void kernel_scatter_points(
        const POINT_TYPE*, int, int, const double*, double, int*, const int*, POINT_TYPE*, unsigned int*);
#endif

    // ============================================================
    // init (once), prepare (per timestep), free (once)
    // ============================================================

    knn_problem* init_once(int n_hydro) {

        // worst-case total points: max_n_local + periodic ghosts + MPI ghosts
        double ghost_frac  = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        int    n_grow      = proteus_mpi::max_n_local(n_hydro);
        int    max_n_total = (int)(n_grow + 2.0 * ghost_frac * n_grow) + 1 + proteus_mpi::n_mpi_capacity;

        knn_problem* knn = gpu_alloc<knn_problem>(1);

        // pick grid resolution: ~3 points per cell on average is the sweet spot for KNN
        knn->len_pts      = max_n_total;
        knn->pts_capacity = max_n_total;
        knn->N_grid       = std::max(1, (int)round(pow(max_n_total / 3.1f, 1.0f / (float)DIMENSION)));
        knn->Npow         = (int)pow(knn->N_grid, DIMENSION);
        // bucket grid spans [-buff, 1+buff]^d; cellFromPoint uses inv_boxsize to index into it
        knn->buff                = buff;
        knn->inv_boxsize         = 1.0 / (1.0 + 2.0 * buff);
        // default to the global box; set_local_extent() re-anchors to the rank's extent each build
        knn->inv_cell_size = (double)knn->N_grid * knn->inv_boxsize;
        knn->grid_lo[0]    = -buff;
        knn->grid_lo[1]    = -buff;
#ifdef dim_2D
        knn->grid_lo[2] = 0.0;
#else
        knn->grid_lo[2] = -buff;
#endif
        knn->d_cell_offsets           = NULL;
        knn->d_cell_offset_dists      = NULL;
        knn->d_cell_offset_dists_unit = NULL;
        knn->d_permutation            = NULL;
        knn->d_counters          = NULL;
        knn->d_ptrs              = NULL;
        knn->d_globcounter       = NULL;
        knn->d_stored_points     = NULL;

        int N_max = 16;
        if (knn->N_grid < N_max) {
            std::cerr << "KNN: We don't support meshes with less than approx 12700 cells (3D)." << std::endl;
            exit(EXIT_FAILURE);
        }

        // ring-expansion offset table: grid-cell offsets ordered by Chebyshev ring distance.
        // cell_offset_dists holds the lower-bound dist^2 for the *current* cell_size; the _unit
        // copy holds it at cell_size==1 so set_local_extent() can rescale per build.
        double  cell_size         = (1.0 + 2.0 * buff) / (double)knn->N_grid; // global-box default
        int     alloc             = N_max * N_max * N_max * N_max;             // very naive upper bound
        int*    cell_offsets      = gpu_alloc<int>(alloc);
        double* cell_offset_dists = gpu_alloc<double>(alloc);
        double* cell_offset_dists_unit = gpu_alloc<double>(alloc);

        // ring 0: the home cell itself
        cell_offsets[0]           = 0;
        cell_offset_dists[0]      = 0.0;
        cell_offset_dists_unit[0] = 0.0;
        knn->N_cell_offsets       = 1;

        // rings 1..N_max-1
        for (int ring = 1; ring < N_max; ring++) {
#ifdef dim_2D
            for (int j = -N_max; j <= N_max; j++) {
                for (int i = -N_max; i <= N_max; i++) {
                    if (std::max(abs(i), abs(j)) != ring) continue;

                    int id_offset                     = i + j * knn->N_grid;
                    cell_offsets[knn->N_cell_offsets] = id_offset;

                    // lower-bound dist^2 from home cell to this ring cell, at unit cell_size and
                    // at the current cell_size. A ring-r cell is >= (r-1) cells away.
                    double du                                   = (double)(ring - 1);
                    cell_offset_dists_unit[knn->N_cell_offsets] = du * du;
                    cell_offset_dists[knn->N_cell_offsets]      = du * du * cell_size * cell_size;

                    knn->N_cell_offsets++;
                }
            }
#else
            for (int k = -N_max; k <= N_max; k++) {
                for (int j = -N_max; j <= N_max; j++) {
                    for (int i = -N_max; i <= N_max; i++) {
                        if (std::max(abs(i), std::max(abs(j), abs(k))) != ring) continue;

                        int id_offset                     = i + j * knn->N_grid + k * knn->N_grid * knn->N_grid;
                        cell_offsets[knn->N_cell_offsets] = id_offset;

                        double du                                   = (double)(ring - 1);
                        cell_offset_dists_unit[knn->N_cell_offsets] = du * du;
                        cell_offset_dists[knn->N_cell_offsets]      = du * du * cell_size * cell_size;

                        knn->N_cell_offsets++;
                    }
                }
            }
#endif
        }

        knn->d_cell_offsets           = cell_offsets;
        knn->d_cell_offset_dists      = cell_offset_dists;
        knn->d_cell_offset_dists_unit = cell_offset_dists_unit;

        // per-call grid bookkeeping (counters + prefix-sum pointers + atomic counter)
        int Npow        = knn->Npow;
        knn->d_counters = gpu_calloc<int>(Npow);
        knn->d_ptrs     = gpu_calloc<int>(Npow);

        knn->d_globcounter   = gpu_calloc<int>(1);
        knn->d_stored_points = gpu_calloc<POINT_TYPE>(max_n_total);
        knn->d_permutation   = gpu_calloc<unsigned int>(max_n_total);

        // hint GPU-preferred location for hot KNN arrays (reduces UM page faults)
        gpu_advise_gpu_preferred(knn->d_stored_points, max_n_total * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(knn->d_counters, Npow * sizeof(int));
        gpu_advise_gpu_preferred(knn->d_ptrs, Npow * sizeof(int));
        gpu_advise_gpu_preferred(knn->d_cell_offsets, knn->N_cell_offsets * sizeof(int));
        gpu_advise_gpu_preferred(knn->d_cell_offset_dists, knn->N_cell_offsets * sizeof(double));

        return knn;
    }

    // Re-anchor the KNN grid to this rank's local extent so occupancy stays ~3 pts/cell at every
    // rank count (instead of spreading N_grid cells over the whole global box). Isotropic cells
    // keep the cubic ring-distance table valid. Degenerate extent (single-rank sentinel lo==hi)
    // falls back to the global box, making non-MPI behaviour identical. Called per build, before
    // prepare(); only this rank's KNN grid is affected — the MPI/decomp grid is untouched.
    void set_local_extent(knn_problem* knn, const double* data_lo, const double* data_hi) {
        const int    N_grid       = knn->N_grid;
        const bool   extent_valid = (data_hi[0] > data_lo[0]); // same predicate as cell.cu safe-radius
        double       cell_size;

        if (extent_valid) {
            // isotropic cell sized from the largest active-axis span; origin at data_lo
            double span = data_hi[0] - data_lo[0];
            span        = std::max(span, data_hi[1] - data_lo[1]);
            knn->grid_lo[0] = data_lo[0];
            knn->grid_lo[1] = data_lo[1];
#ifdef dim_2D
            knn->grid_lo[2] = 0.0;
#else
            span            = std::max(span, data_hi[2] - data_lo[2]);
            knn->grid_lo[2] = data_lo[2];
#endif
            cell_size          = span / (double)N_grid;
            knn->inv_cell_size = 1.0 / cell_size;
        } else {
            // global-box fallback: bit-identical to init_once's mapping
            cell_size          = (1.0 + 2.0 * knn->buff) / (double)N_grid;
            knn->inv_cell_size = (double)N_grid * knn->inv_boxsize;
            knn->grid_lo[0]    = -knn->buff;
            knn->grid_lo[1]    = -knn->buff;
#ifdef dim_2D
            knn->grid_lo[2] = 0.0;
#else
            knn->grid_lo[2] = -knn->buff;
#endif
        }

        // rescale ring lower-bounds to the new cell_size (unit table holds them at cell_size==1)
        const double cs2 = cell_size * cell_size;
        for (int m = 0; m < knn->N_cell_offsets; m++) {
            knn->d_cell_offset_dists[m] = knn->d_cell_offset_dists_unit[m] * cs2;
        }
    }

    // per-timestep refresh: zero counters and rebuild the grid bucket sort
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {

        if (len_pts > knn->pts_capacity) {
            std::cerr << "KNN: Error! point count " << len_pts << " exceeds pre-allocated capacity "
                      << knn->pts_capacity << ". Increase ghost headroom." << std::endl;
            exit(EXIT_FAILURE);
        }

        knn->len_pts = len_pts;

        // reset grid counters and pointers
        gpu_memset(knn->d_counters, 0, knn->Npow * sizeof(int));
        gpu_memset(knn->d_ptrs, 0, knn->Npow * sizeof(int));
        gpu_memset(knn->d_globcounter, 0, sizeof(int));

        // bucket-sort the input points into d_stored_points, grouped by grid cell
        sort_points_into_grid(knn, pts, len_pts);
    }

    void knn_free(knn_problem** knn) {
        gpu_free((*knn)->d_cell_offsets);
        gpu_free((*knn)->d_cell_offset_dists);
        gpu_free((*knn)->d_cell_offset_dists_unit);
        gpu_free((*knn)->d_permutation);
        gpu_free((*knn)->d_counters);
        gpu_free((*knn)->d_ptrs);
        gpu_free((*knn)->d_globcounter);
        gpu_free((*knn)->d_stored_points);
        gpu_free(*knn);
        *knn = NULL;
    }

    // resize the per-call buffers (d_stored_points + d_permutation) to fit a new max_n_total.
    // N_grid is left alone — its value is computed at startup and reused; a runtime change
    // would invalidate the bucket-sort assumptions on existing data. d_stored_points contents
    // are scratch (rebuilt by prepare() each step), so no copy needed.
    void knn_grow(knn_problem* knn, int new_pts_capacity) {
        if (new_pts_capacity <= knn->pts_capacity) return;
        gpu_free(knn->d_stored_points);
        gpu_free(knn->d_permutation);
        knn->d_stored_points = gpu_calloc<POINT_TYPE>(new_pts_capacity);
        knn->d_permutation   = gpu_calloc<unsigned int>(new_pts_capacity);
        knn->pts_capacity    = new_pts_capacity;
        gpu_advise_gpu_preferred(knn->d_stored_points, new_pts_capacity * sizeof(POINT_TYPE));
    }

    // ============================================================
    // Grid sort (GPU kernels or CPU loops)
    // ============================================================

    static void sort_points_into_grid(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {

        int           N_grid        = knn->N_grid;
        int           Npow          = knn->Npow;
        const double* grid_lo       = knn->grid_lo;
        double        inv_cell_size = knn->inv_cell_size;
        int*          d_counters    = knn->d_counters;

#ifndef CPU_DEBUG
        int tpb = _KNN_BLOCK_SIZE_;

        // 1) count points per grid cell
        int blocks1 = (len_pts + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("COUNT");
            kernel_count_cells<<<blocks1, tpb>>>(pts, len_pts, N_grid, grid_lo, inv_cell_size, d_counters);
            GPU_SYNC();
        }

        // 2) compute prefix pointers via atomicAdd
        int blocks2 = (Npow + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("PTRS");
            kernel_compute_ptrs<<<blocks2, tpb>>>(d_counters, knn->d_ptrs, knn->d_globcounter, Npow);
            GPU_SYNC();
        }

        // 3) scatter points into sorted positions
        gpu_memset(d_counters, 0, Npow * sizeof(int));
        {
            PROFILE_KERNEL("SCATTER");
            kernel_scatter_points<<<blocks1, tpb>>>(pts,
                                                    len_pts,
                                                    N_grid,
                                                    grid_lo,
                                                    inv_cell_size,
                                                    d_counters,
                                                    knn->d_ptrs,
                                                    knn->d_stored_points,
                                                    knn->d_permutation);
            GPU_SYNC();
        }

#else
        // count points per grid cell
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int id = 0; id < len_pts; id++) {
            int cell = cellFromPoint(N_grid, grid_lo, inv_cell_size, pts[id]);
            portable_atomicAdd(d_counters + cell, 1);
        }

        // reserve memory ranges for each cell
        {
            int* d_ptrs        = knn->d_ptrs;
            int* d_globcounter = knn->d_globcounter;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int id = 0; id < Npow; id++) {
                int count = d_counters[id];
                if (count > 0) { d_ptrs[id] = portable_atomicAdd(d_globcounter, count); }
            }
        }

        // store points in their cell-organized locations
        {
            gpu_memset(d_counters, 0, Npow * sizeof(int));

            const int*    d_ptrs          = knn->d_ptrs;
            POINT_TYPE*   d_stored_points = knn->d_stored_points;
            unsigned int* d_permutation   = knn->d_permutation;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int id = 0; id < len_pts; id++) {
                POINT_TYPE p         = pts[id];
                int        cell      = cellFromPoint(N_grid, grid_lo, inv_cell_size, p);
                int        pos       = d_ptrs[cell] + portable_atomicAdd(d_counters + cell, 1);
                d_stored_points[pos] = p;
                d_permutation[pos]   = id;
            }
        }
#endif // CPU_DEBUG
    }

    // ============================================================
    // CUDA kernel wrappers
    // ============================================================
#ifndef CPU_DEBUG

    GLOBAL void kernel_count_cells(
        const POINT_TYPE* pts, int len_pts, int N_grid, const double* grid_lo, double inv_cell_size, int* d_counters) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= len_pts) return;
        int cell = cellFromPoint(N_grid, grid_lo, inv_cell_size, pts[id]);
        portable_atomicAdd(d_counters + cell, 1);
    }

    GLOBAL void kernel_compute_ptrs(int* d_counters, int* d_ptrs, int* d_globcounter, int Npow) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= Npow) return;
        int count = d_counters[id];
        if (count > 0) { d_ptrs[id] = portable_atomicAdd(d_globcounter, count); }
    }

    GLOBAL void kernel_scatter_points(const POINT_TYPE* pts,
                                      int               len_pts,
                                      int               N_grid,
                                      const double*     grid_lo,
                                      double            inv_cell_size,
                                      int*              d_counters,
                                      const int*        d_ptrs,
                                      POINT_TYPE*       d_stored_points,
                                      unsigned int*     d_permutation) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= len_pts) return;
        POINT_TYPE p         = pts[id];
        int        cell      = cellFromPoint(N_grid, grid_lo, inv_cell_size, p);
        int        pos       = d_ptrs[cell] + portable_atomicAdd(d_counters + cell, 1);
        d_stored_points[pos] = p;
        d_permutation[pos]   = id;
    }

#endif // !CPU_DEBUG

    // ============================================================
    // helpers (grid mapping)
    // ============================================================

    HD int cellFromPoint(int N_grid, const double* grid_lo, double inv_cell_size, POINT_TYPE point) {
        // grid origin grid_lo[a], isotropic cell width 1/inv_cell_size. Map (point - grid_lo) into
        // [0, N_grid); out-of-extent ghosts clamp into the edge cells. For the global-box fallback
        // grid_lo = -buff and inv_cell_size = N_grid/(1+2*buff), i.e. the old mapping exactly.
        int i = (int)floor((point.x - grid_lo[0]) * inv_cell_size);
        int j = (int)floor((point.y - grid_lo[1]) * inv_cell_size);

        i = imax(0, imin(i, N_grid - 1));
        j = imax(0, imin(j, N_grid - 1));

#ifdef dim_2D
        return i + j * N_grid;
#else
        int k = (int)floor((point.z - grid_lo[2]) * inv_cell_size);
        k     = imax(0, imin(k, N_grid - 1));
        return i + j * N_grid + k * N_grid * N_grid;
#endif
    }

} // namespace knn
