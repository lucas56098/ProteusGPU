#include "../global/globals.h"
#include "../profiler/profiler.h"
#include "knn.h"
#include <iostream>

namespace knn {

#ifndef CPU_DEBUG
    // kernels
    GLOBAL void kernel_count_cells(const POINT_TYPE*, int, int, int*);
    GLOBAL void kernel_compute_ptrs(int*, int*, int*, int);
    GLOBAL void kernel_scatter_points(const POINT_TYPE*, int, int, int*, const int*, POINT_TYPE*, unsigned int*);
#endif

    // ============================================================
    // init (once), prepare (per timestep), free (once)
    // ============================================================

    knn_problem* init_once(int n_hydro) {

        // worst-case total points including periodic ghosts (same formula as periodic_mesh)
        double ghost_frac  = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        int    max_n_total = (int)(n_hydro + 2.0 * ghost_frac * n_hydro) + 1;

        knn_problem* knn = gpu_alloc<knn_problem>(1);

        // pick grid resolution: ~3 points per cell on average is the sweet spot for KNN
        knn->len_pts             = max_n_total;
        knn->pts_capacity        = max_n_total;
        knn->N_grid              = std::max(1, (int)round(pow(max_n_total / 3.1f, 1.0f / (float)DIMENSION)));
        knn->Npow                = (int)pow(knn->N_grid, DIMENSION);
        knn->d_cell_offsets      = NULL;
        knn->d_cell_offset_dists = NULL;
        knn->d_permutation       = NULL;
        knn->d_counters          = NULL;
        knn->d_ptrs              = NULL;
        knn->d_globcounter       = NULL;
        knn->d_stored_points     = NULL;

        int N_max = 16;
        if (knn->N_grid < N_max) {
            std::cerr << "KNN: We don't support meshes with less than approx 12700 cells (3D)." << std::endl;
            exit(EXIT_FAILURE);
        }

        // ring-expansion offset table: grid-cell offsets ordered by Chebyshev ring distance
        int     alloc             = N_max * N_max * N_max * N_max; // very naive upper bound
        int*    cell_offsets      = gpu_alloc<int>(alloc);
        double* cell_offset_dists = gpu_alloc<double>(alloc);

        // ring 0: the home cell itself
        cell_offsets[0]      = 0;
        cell_offset_dists[0] = 0.0;
        knn->N_cell_offsets  = 1;

        // rings 1..N_max-1
        for (int ring = 1; ring < N_max; ring++) {
#ifdef dim_2D
            for (int j = -N_max; j <= N_max; j++) {
                for (int i = -N_max; i <= N_max; i++) {
                    if (std::max(abs(i), abs(j)) != ring) continue;

                    int id_offset                     = i + j * knn->N_grid;
                    cell_offsets[knn->N_cell_offsets] = id_offset;

                    // lower-bound distance from home cell to this ring cell (boxsize = 1)
                    double d = (double)(ring - 1) / (double)(knn->N_grid);
                    cell_offset_dists[knn->N_cell_offsets] = d * d;

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

                        double d = (double)(ring - 1) / (double)(knn->N_grid);
                        cell_offset_dists[knn->N_cell_offsets] = d * d;

                        knn->N_cell_offsets++;
                    }
                }
            }
#endif
        }

        knn->d_cell_offsets      = cell_offsets;
        knn->d_cell_offset_dists = cell_offset_dists;

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
        gpu_free((*knn)->d_permutation);
        gpu_free((*knn)->d_counters);
        gpu_free((*knn)->d_ptrs);
        gpu_free((*knn)->d_globcounter);
        gpu_free((*knn)->d_stored_points);
        gpu_free(*knn);
        *knn = NULL;
    }

    // ============================================================
    // Grid sort (GPU kernels or CPU loops)
    // ============================================================

    void sort_points_into_grid(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {

        int  N_grid     = knn->N_grid;
        int  Npow       = knn->Npow;
        int* d_counters = knn->d_counters;

#ifndef CPU_DEBUG
        int tpb = _KNN_BLOCK_SIZE_;

        PROFILE_GPU_START("knn_grid_sort");

        // 1) count points per grid cell
        int blocks1 = (len_pts + tpb - 1) / tpb;
        kernel_count_cells<<<blocks1, tpb>>>(pts, len_pts, N_grid, d_counters);
        GPU_LAUNCH_CHECK();

        // 2) compute prefix pointers via atomicAdd
        int blocks2 = (Npow + tpb - 1) / tpb;
        kernel_compute_ptrs<<<blocks2, tpb>>>(d_counters, knn->d_ptrs, knn->d_globcounter, Npow);
        GPU_LAUNCH_CHECK();

        // 3) scatter points into sorted positions
        gpu_memset(d_counters, 0, Npow * sizeof(int));
        kernel_scatter_points<<<blocks1, tpb>>>(
            pts, len_pts, N_grid, d_counters, knn->d_ptrs, knn->d_stored_points, knn->d_permutation);

        PROFILE_GPU_END("knn_grid_sort");

#else
        // count points per grid cell
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int id = 0; id < len_pts; id++) {
            int cell = cellFromPoint(N_grid, pts[id]);
            host_atomicAdd(d_counters + cell, 1);
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
                if (count > 0) { d_ptrs[id] = host_atomicAdd(d_globcounter, count); }
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
                int        cell      = cellFromPoint(N_grid, p);
                int        pos       = d_ptrs[cell] + host_atomicAdd(d_counters + cell, 1);
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

    GLOBAL void kernel_count_cells(const POINT_TYPE* pts, int len_pts, int N_grid, int* d_counters) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= len_pts) return;
        int cell = cellFromPoint(N_grid, pts[id]);
        atomicAdd(d_counters + cell, 1);
    }

    GLOBAL void kernel_compute_ptrs(int* d_counters, int* d_ptrs, int* d_globcounter, int Npow) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= Npow) return;
        int count = d_counters[id];
        if (count > 0) { d_ptrs[id] = atomicAdd(d_globcounter, count); }
    }

    GLOBAL void kernel_scatter_points(const POINT_TYPE* pts,
                                      int               len_pts,
                                      int               N_grid,
                                      int*              d_counters,
                                      const int*        d_ptrs,
                                      POINT_TYPE*       d_stored_points,
                                      unsigned int*     d_permutation) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= len_pts) return;
        POINT_TYPE p         = pts[id];
        int        cell      = cellFromPoint(N_grid, p);
        int        pos       = d_ptrs[cell] + atomicAdd(d_counters + cell, 1);
        d_stored_points[pos] = p;
        d_permutation[pos]   = id;
    }

#endif // !CPU_DEBUG

    // ============================================================
    // helpers (grid mapping)
    // ============================================================

    HD int cellFromPoint(int N_grid, POINT_TYPE point) {
        int i = (int)floor(point.x * (double)N_grid); // assumes boxsize = 1.0
        int j = (int)floor(point.y * (double)N_grid); // assumes boxsize = 1.0

        i = imax(0, imin(i, N_grid - 1));
        j = imax(0, imin(j, N_grid - 1));

#ifdef dim_2D
        return i + j * N_grid;
#else
        int k = (int)floor(point.z * (double)N_grid); // assumes boxsize = 1.0
        k     = imax(0, imin(k, N_grid - 1));
        return i + j * N_grid + k * N_grid * N_grid;
#endif
    }

} // namespace knn
