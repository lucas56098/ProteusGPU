#include "knn.h"
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// CONSTRUCTION SITE: nothing works yet :D

namespace knn {

    // -------- initalize KNN problem --------
    knn_problem* init(POINT_TYPE* pts, int len_pts) {

        // -------- allocate the main data structure --------
        knn_problem* knn = (knn_problem*)malloc(sizeof(knn_problem));

        knn->len_pts             = len_pts;
        knn->N_grid              = std::max(1, (int)round(pow(len_pts / 3.1f, 1.0f / (float)DIMENSION)));
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

        // lets build an offset grid: allows us to quickly access pre computed ring-based neighbour pattern
        int     alloc             = N_max * N_max * N_max * N_max; // very naive upper bound
        int*    cell_offsets      = (int*)malloc(alloc * sizeof(int));
        double* cell_offset_dists = (double*)malloc(alloc * sizeof(double));

        // init first query
        cell_offsets[0]      = 0;
        cell_offset_dists[0] = 0.0;
        knn->N_cell_offsets  = 1;

        // -------- calc offsets for all rings up to N_max --------
        for (int ring = 1; ring < N_max; ring++) {
#ifdef dim_2D
            // 2D: only iterate over i and j
            for (int j = -N_max; j <= N_max; j++) {
                for (int i = -N_max; i <= N_max; i++) {
                    if (std::max(abs(i), abs(j)) != ring) continue;
                    // everything below is only executed if cell is inside current ring

                    // compute linear offset in the flattened 2D grid array
                    int id_offset                     = i + j * knn->N_grid;
                    cell_offsets[knn->N_cell_offsets] = id_offset;

                    // compute geometric distance for pruning later on
                    double d = (double)(ring - 1) / (double)(knn->N_grid); // assumes boxsize = 1.0
                    cell_offset_dists[knn->N_cell_offsets] = d * d;

                    knn->N_cell_offsets++;
                }
            }
#else
            // 3D: iterate over i, j, and k
            for (int k = -N_max; k <= N_max; k++) {
                for (int j = -N_max; j <= N_max; j++) {
                    for (int i = -N_max; i <= N_max; i++) {
                        if (std::max(abs(i), std::max(abs(j), abs(k))) != ring) continue;
                        // everything below is only executed if cell is inside current ring

                        // compute linear offset in the flattened 3D grid array
                        int id_offset                     = i + j * knn->N_grid + k * knn->N_grid * knn->N_grid;
                        cell_offsets[knn->N_cell_offsets] = id_offset;

                        // compute geometric distance for pruning later on
                        double d = (double)(ring - 1) / (double)(knn->N_grid); // assumes boxsize = 1.0
                        cell_offset_dists[knn->N_cell_offsets] = d * d;

                        knn->N_cell_offsets++;
                    }
                }
            }
#endif
        }

        // -------- allocate memory buffers and copy data --------
        knn->d_cell_offsets = (int*)realloc(cell_offsets, knn->N_cell_offsets * sizeof(int));

        knn->d_cell_offset_dists = (double*)realloc(cell_offset_dists, knn->N_cell_offsets * sizeof(double));

        POINT_TYPE* d_points = (POINT_TYPE*)malloc(len_pts * sizeof(POINT_TYPE));
        memcpy(d_points, pts, len_pts * sizeof(POINT_TYPE)); // input pts (temporary), freed after sorting into grid

        int Npow        = pow(knn->N_grid, DIMENSION);
        knn->d_counters = (int*)calloc(Npow, sizeof(int)); // pts per grid cell
        knn->d_ptrs     = (int*)calloc(Npow, sizeof(int)); // cell ptrs to start in d_stored_points

        knn->d_globcounter = (int*)calloc(1, sizeof(int)); // global counter
        knn->d_stored_points =
            (POINT_TYPE*)calloc(knn->len_pts, sizeof(POINT_TYPE)); // will be filled with sorted points
        knn->d_permutation =
            (unsigned int*)calloc(knn->len_pts, sizeof(unsigned int)); // permutation to restore original order

        // -------- reorganize input points by grid cell --------
        sort_points_into_grid(knn, d_points, len_pts);

        // no longer need original points
        free(d_points);

        return knn;
    }

    void sort_points_into_grid(knn_problem* knn, POINT_TYPE* d_points, int len_pts) {

        // -------- count points per grid cell --------
        {
            int threadsPerBlock = 256;
            int blocksPerGrid   = (len_pts + threadsPerBlock - 1) / threadsPerBlock;

#ifdef CPU_DEBUG
            cpu_count(blocksPerGrid, threadsPerBlock, d_points, len_pts, knn->N_grid, knn->d_counters);
#endif
        }

        // -------- reserve memory ranges for each cell --------
        {
            int threadsPerBlock = 4;
            int blocksPerGrid   = (pow(knn->N_grid, DIMENSION) + threadsPerBlock - 1) / threadsPerBlock;

#ifdef CPU_DEBUG
            cpu_reserve(blocksPerGrid, threadsPerBlock, knn->N_grid, knn->d_counters, knn->d_globcounter, knn->d_ptrs);
#endif
        }

        // -------- store points in their cell-organized locations -------
        {
            // reset counters: we'll reuse them for atomic allocation within each cell's range
            memset(knn->d_counters, 0x00, pow(knn->N_grid, DIMENSION) * sizeof(int));

            int threadsPerBlock = 256;
            int blocksPerGrid   = (len_pts + threadsPerBlock - 1) / threadsPerBlock;

// store oraganized points
#ifdef CPU_DEBUG
            cpu_store(blocksPerGrid,
                      threadsPerBlock,
                      d_points,
                      len_pts,
                      knn->N_grid,
                      knn->d_ptrs,
                      knn->d_counters,
                      knn->d_stored_points,
                      knn->d_permutation);
#endif
        }
    }

#ifdef CPU_DEBUG
    // counts how many poiunts are in each cell, stores in d_counters
    void
    cpu_count(int blocksPerGrid, int threadsPerBlock, POINT_TYPE* d_points, int len_pts, int N_grid, int* d_counters) {
        for (int blockId = 0; blockId < blocksPerGrid; blockId++) {
            for (int threadId = 0; threadId < threadsPerBlock; threadId++) {
                int id = threadsPerBlock * blockId + threadId;
                if (id < len_pts) {
                    int cell = cellFromPoint(N_grid, d_points[id]);
                    atomicAdd(d_counters + cell, 1);
                }
            }
        }
    }

    // uses d_counters to reserve memory ranges for each cell, stores in d_ptrs
    void cpu_reserve(
        int blocksPerGrid, int threadsPerBlock, int N_grid, const int* d_counters, int* d_globcounter, int* d_ptrs) {
        for (int blockId = 0; blockId < blocksPerGrid; blockId++) {
            for (int threadId = 0; threadId < threadsPerBlock; threadId++) {
                int id = threadsPerBlock * blockId + threadId;

                if (id < pow(N_grid, DIMENSION)) {
                    int count = d_counters[id]; // read how many points are in this cell
                    if (count > 0) {
                        d_ptrs[id] = atomicAdd(d_globcounter, count); // store starting pos in ptrs
                    }
                }
            }
        }
    }

    // stores points in their cell-organized locations
    void cpu_store(int               blocksPerGrid,
                   int               threadsPerBlock,
                   const POINT_TYPE* d_points,
                   int               len_pts,
                   int               N_grid,
                   const int*        d_ptrs,
                   int*              d_counters,
                   POINT_TYPE*       d_stored_points,
                   unsigned int*     d_permutation) {
        for (int blockId = 0; blockId < blocksPerGrid; blockId++) {
            for (int threadId = 0; threadId < threadsPerBlock; threadId++) {
                int id = threadsPerBlock * blockId + threadId;
                if (id < len_pts) {
                    // determine cell for point
                    POINT_TYPE p    = d_points[id];
                    int        cell = cellFromPoint(N_grid, p);

                    // claim a slot within the cell's range
                    int pos = d_ptrs[cell] + atomicAdd(d_counters + cell, 1);

                    d_stored_points[pos] = p;
                    d_permutation[pos]   = id;
                }
            }
        }
    }
#endif

    // get cell index from point position (will be __device__)
    int cellFromPoint(int N_grid, POINT_TYPE point) {
        int i = (int)floor(point.x * (double)N_grid); // assumes boxsize = 1.0
        int j = (int)floor(point.y * (double)N_grid); // assumes boxsize = 1.0

        i = std::max(0, std::min(i, N_grid - 1));
        j = std::max(0, std::min(j, N_grid - 1));

#ifdef dim_2D
        return i + j * N_grid;
#else
        int k = (int)floor(point.z * (double)N_grid); // assumes boxsize = 1.0
        k     = std::max(0, std::min(k, N_grid - 1));
        return i + j * N_grid + k * N_grid * N_grid;
#endif
    }

    // -------- inline per-point KNN (called from voronoi cell construction) --------
    void knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest) {
        // thread-private k-nearest arrays (stack-allocated)
        unsigned int local_knearest[_K_];
        double       local_dists[_K_];

        const POINT_TYPE* d_stored_points     = knn->d_stored_points;
        int               N_grid              = knn->N_grid;
        const int*        d_ptrs              = knn->d_ptrs;
        const int*        d_counters          = knn->d_counters;
        int               N_cell_offsets      = knn->N_cell_offsets;
        const int*        d_cell_offsets      = knn->d_cell_offsets;
        const double*     d_cell_offset_dists = knn->d_cell_offset_dists;

        POINT_TYPE p       = d_stored_points[point_in];
        int        cell_in = cellFromPoint(N_grid, p);

        for (int i = 0; i < _K_; i++) {
            local_knearest[i] = UINT_MAX;
            local_dists[i]    = DBL_MAX;
        }

        int search_cell_index = 0;

        do {
            double min_dist = d_cell_offset_dists[search_cell_index];
            if (local_dists[0] < min_dist) { break; }

            int cell = cell_in + d_cell_offsets[search_cell_index];

            if (cell >= 0 && cell < (int)pow(N_grid, DIMENSION)) {
                int cell_base = d_ptrs[cell];
                int num       = d_counters[cell];

                for (int ptr = cell_base; ptr < cell_base + num; ptr++) {
                    if (ptr == point_in) { continue; }

                    POINT_TYPE p_cmp = d_stored_points[ptr];
                    double     d     = dist2_point(p, p_cmp);

                    if (d < local_dists[0]) {
                        local_knearest[0] = ptr;
                        local_dists[0]    = d;
                        heapify(local_knearest, local_dists, 0, _K_);
                    }
                }
            }
        } while (search_cell_index++ < N_cell_offsets);

        heapsort(local_knearest, local_dists, _K_);

        for (int i = 0; i < _K_; i++) {
            out_knearest[i] = local_knearest[i];
        }
    }

    template <typename T> void inline swap_on_device(T& a, T& b) {
        T c(a);
        a = b;
        b = c;
    }

    void heapify(unsigned int* keys, double* vals, int node, int size) {
        int j = node;
        while (true) {
            int left    = 2 * j + 1;
            int right   = 2 * j + 2;
            int largest = j;
            if (left < size && vals[left] > vals[largest]) { largest = left; }
            if (right < size && vals[right] > vals[largest]) { largest = right; }
            if (largest == j) return;
            swap_on_device(vals[j], vals[largest]);
            swap_on_device(keys[j], keys[largest]);
            j = largest;
        }
    }

    void heapsort(unsigned int* keys, double* vals, int size) {
        while (size) {
            swap_on_device(vals[0], vals[size - 1]);
            swap_on_device(keys[0], keys[size - 1]);
            heapify(keys, vals, 0, --size);
        }
    }

    // -------- other --------
    void knn_free(knn_problem** knn) {
        free((*knn)->d_cell_offsets);
        free((*knn)->d_cell_offset_dists);
        free((*knn)->d_permutation);
        free((*knn)->d_counters);
        free((*knn)->d_ptrs);
        free((*knn)->d_globcounter);
        free((*knn)->d_stored_points);
        free(*knn);
        *knn = NULL;
    }

} // namespace knn
