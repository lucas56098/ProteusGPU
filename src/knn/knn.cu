// implements the neighbour grid (knn.h)

#include "../global/globals.h"
#include "../global/structs.h"
#include "../profiler/profiler.h"
#include "knn.h"
#include <iostream>

namespace knn {

    static void sort_points_into_grid(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    // allocates the grid and the point arrays for the whole run
    knn_problem* init_once(int n_hydro, int N_grid_restored) {

        // room for the local cells after growth, their periodic ghosts and the MPI ghosts
        double ghost_frac  = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        int    n_grow      = proteus_mpi::max_n_local(n_hydro);
        int    max_n_total = (int)(n_grow + 2.0 * ghost_frac * n_grow) + 1 + proteus_mpi::n_mpi_capacity;

        knn_problem* knn = gpu_alloc<knn_problem>(1);

        knn->len_pts      = max_n_total;
        knn->pts_capacity = max_n_total;

        // about 3 points per bucket
        // a restart keeps the old size, another size would change the cell order
        knn->N_grid        = (N_grid_restored > 0) ? N_grid_restored
                                                   : std::max(1, (int)round(pow(max_n_total / 3.1f, 1.0f / (float)DIMENSION)));
        knn->Npow          = (int)pow(knn->N_grid, DIMENSION);
        knn->buff          = buff;
        knn->inv_boxsize   = 1.0 / (1.0 + 2.0 * buff);
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
        knn->d_counters               = NULL;
        knn->d_ptrs                   = NULL;
        knn->d_scan_scratch           = NULL;
        knn->d_bucket_ids             = NULL;
        knn->d_stored_points          = NULL;

        // N_max is the number of rings the search walks, and the smallest grid we take
        int N_max = 16;
        if (knn->N_grid < N_max) {
            proteus_mpi::exit_failure("KNN: We don't support meshes with less than approx 12700 cells (3D).\n");
        }

        double cell_size = (1.0 + 2.0 * buff) / (double)knn->N_grid;
        // more than enough for the offsets of N_max rings
        int     alloc                  = N_max * N_max * N_max * N_max;
        int*    cell_offsets           = gpu_alloc<int>(alloc);
        double* cell_offset_dists      = gpu_alloc<double>(alloc);
        double* cell_offset_dists_unit = gpu_alloc<double>(alloc);

        cell_offsets[0]           = 0;
        cell_offset_dists[0]      = 0.0;
        cell_offset_dists_unit[0] = 0.0;
        knn->N_cell_offsets       = 1;

        // index step and smallest distance for every ring, nearest ring first
        // a point in ring r is at least r - 1 buckets away
        for (int ring = 1; ring < N_max; ring++) {
#ifdef dim_2D
            for (int j = -N_max; j <= N_max; j++) {
                for (int i = -N_max; i <= N_max; i++) {
                    if (std::max(abs(i), abs(j)) != ring) continue;

                    int id_offset                     = i + j * knn->N_grid;
                    cell_offsets[knn->N_cell_offsets] = id_offset;

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

        int Npow        = knn->Npow;
        knn->d_counters = gpu_calloc<int>(Npow);
        knn->d_ptrs     = gpu_calloc<int>(Npow);

        knn->d_stored_points = gpu_calloc<POINT_TYPE>(max_n_total);
        knn->d_permutation   = gpu_calloc<unsigned int>(max_n_total);
        knn->d_scan_scratch  = gpu_calloc<int>((int)scan_scratch_size((size_t)Npow, _KNN_BLOCK_SIZE_));
        knn->d_bucket_ids    = gpu_calloc<int>(max_n_total);

        // the search reads these on every build
        gpu_advise_gpu_preferred(knn->d_stored_points, max_n_total * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(knn->d_counters, Npow * sizeof(int));
        gpu_advise_gpu_preferred(knn->d_ptrs, Npow * sizeof(int));
        gpu_advise_gpu_preferred(knn->d_cell_offsets, knn->N_cell_offsets * sizeof(int));
        gpu_advise_gpu_preferred(knn->d_cell_offset_dists, knn->N_cell_offsets * sizeof(double));

        return knn;
    }

    // puts the grid on the data extent, or on the box plus both bands
    void set_local_extent(knn_problem* knn, const double* data_lo, const double* data_hi) {
        const int  N_grid       = knn->N_grid;
        const bool extent_valid = (data_hi[0] > data_lo[0]);
        double     cell_size;

        if (extent_valid) {
            // one bucket size for all axes, taken from the widest one
            double span     = data_hi[0] - data_lo[0];
            span            = std::max(span, data_hi[1] - data_lo[1]);
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

        // the ring distances follow the new bucket size
        const double cs2 = cell_size * cell_size;
        for (int m = 0; m < knn->N_cell_offsets; m++) {
            knn->d_cell_offset_dists[m] = knn->d_cell_offset_dists_unit[m] * cs2;
        }
    }

    // sorts len_pts points into the grid, called once per mesh build
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {

        if (len_pts > knn->pts_capacity) {
            proteus_mpi::exit_failure(
                "KNN: Error! point count %d exceeds pre-allocated capacity %d. Increase ghost headroom.\n",
                len_pts,
                knn->pts_capacity);
        }

        knn->len_pts = len_pts;

        gpu_memset(knn->d_counters, 0, knn->Npow * sizeof(int));
        gpu_memset(knn->d_ptrs, 0, knn->Npow * sizeof(int));

        sort_points_into_grid(knn, pts, len_pts);
    }

    // frees the grid and the point arrays
    void knn_free(knn_problem** knn) {
        gpu_free((*knn)->d_cell_offsets);
        gpu_free((*knn)->d_cell_offset_dists);
        gpu_free((*knn)->d_cell_offset_dists_unit);
        gpu_free((*knn)->d_permutation);
        gpu_free((*knn)->d_counters);
        gpu_free((*knn)->d_ptrs);
        gpu_free((*knn)->d_scan_scratch);
        gpu_free((*knn)->d_bucket_ids);
        gpu_free((*knn)->d_stored_points);
        gpu_free(*knn);
        *knn = NULL;
    }

    // more room for points; the buckets stay, prepare fills the new arrays
    void knn_grow(knn_problem* knn, int new_pts_capacity) {
        if (new_pts_capacity <= knn->pts_capacity) return;
        gpu_free(knn->d_stored_points);
        gpu_free(knn->d_permutation);
        gpu_free(knn->d_bucket_ids);
        knn->d_stored_points = gpu_calloc<POINT_TYPE>(new_pts_capacity);
        knn->d_permutation   = gpu_calloc<unsigned int>(new_pts_capacity);
        knn->d_bucket_ids    = gpu_calloc<int>(new_pts_capacity);
        knn->pts_capacity    = new_pts_capacity;
        gpu_advise_gpu_preferred(knn->d_stored_points, new_pts_capacity * sizeof(POINT_TYPE));
    }

    // counting sort of the points into the buckets
    static void sort_points_into_grid(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {

        int           N_grid        = knn->N_grid;
        int           Npow          = knn->Npow;
        const double* grid_lo       = knn->grid_lo;
        double        inv_cell_size = knn->inv_cell_size;
        int*          d_counters    = knn->d_counters;
        int*          d_ptrs        = knn->d_ptrs;
        int*          bucket_ids    = knn->d_bucket_ids;
        POINT_TYPE*   stored_points = knn->d_stored_points;
        unsigned int* permutation   = knn->d_permutation;

        // points per bucket
        parallel_for<_KNN_BLOCK_SIZE_>("COUNT", len_pts, [=] HD(int id) {
            const int cell = cell_from_point(N_grid, grid_lo, inv_cell_size, pts[id]);
            portable_atomicAdd(d_counters + cell, 1);
        });

        // where each bucket starts in the sorted list
        parallel_exclusive_scan<_KNN_BLOCK_SIZE_>("PTRS", (size_t)Npow, d_counters, d_ptrs, knn->d_scan_scratch);

        gpu_memset(d_counters, 0, Npow * sizeof(int));
        // group the points by bucket, in the order the atomic gives out the slots
        parallel_for<_KNN_BLOCK_SIZE_>("SCATTER", len_pts, [=] HD(int id) {
            const int cell = cell_from_point(N_grid, grid_lo, inv_cell_size, pts[id]);
            bucket_ids[d_ptrs[cell] + portable_atomicAdd(d_counters + cell, 1)] = id;
        });

        // place by id inside the bucket, so the order does not depend on the atomic above
        parallel_for<_KNN_BLOCK_SIZE_>("RANK", len_pts, [=] HD(int slot) {
            const int id   = bucket_ids[slot];
            const int cell = cell_from_point(N_grid, grid_lo, inv_cell_size, pts[id]);
            const int base = d_ptrs[cell];
            const int end  = base + d_counters[cell];

            int rank = 0;
            for (int q = base; q < end; q++) {
                if (bucket_ids[q] < id) rank++;
            }

            stored_points[base + rank] = pts[id];
            permutation[base + rank]   = (unsigned int)id;
        });
    }

    // bucket a point falls into; a point outside gets the nearest edge bucket
    HD int cell_from_point(int N_grid, const double* grid_lo, double inv_cell_size, POINT_TYPE point) {
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
