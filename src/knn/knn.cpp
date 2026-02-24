#include "knn.h"
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cfloat>
#include <vector>

// CONSTRUCTION SITE: nothing works yet :D

namespace knn {

// -------- initalize KNN problem --------
knn_problem* init(POINT_TYPE *pts, int len_pts) {
    
    // -------- allocate the main data structure --------
    knn_problem *knn = (knn_problem*)malloc(sizeof(knn_problem));

    knn->len_pts = len_pts;
    knn->N_grid = std::max(1,(int)round(pow(len_pts / 3.1f, 1.0f / (float)DIMENSION)));
    knn->d_cell_offsets = NULL;
    knn->d_cell_offset_dists = NULL;
    knn->d_permutation = NULL;
    knn->d_counters = NULL;
    knn->d_ptrs = NULL;
    knn->d_globcounter = NULL;
    knn->d_stored_points = NULL;
    knn->d_knearests = NULL;

    int N_max = 16;
    if (knn->N_grid < N_max) {
        std::cerr << "We don't support meshes with less than approx 12700 cells." << std::endl;
        exit(EXIT_FAILURE);
    }

    // lets build an offset grid: allows us to quickly access pre computed ring-based neighbour pattern
    int alloc = N_max*N_max*N_max*N_max; // very naive upper bound
    int   *cell_offsets      =   (int*)malloc(alloc*sizeof(int));
    double *cell_offset_dists = (double*)malloc(alloc*sizeof(double));

    // init first query
    cell_offsets[0] = 0;
    cell_offset_dists[0] = 0.0;
    knn->N_cell_offsets = 1;

    // -------- calc offsets for all rings up to N_max --------
    for (int ring = 1; ring < N_max; ring++) {
#ifdef dim_2D
        // 2D: only iterate over i and j
        for (int j = -N_max; j <= N_max; j++) {
            for (int i = -N_max; i <= N_max; i++) {
                if (std::max(abs(i), abs(j)) != ring) continue;
                // everything below is only executed if cell is inside current ring

                // compute linear offset in the flattened 2D grid array
                int id_offset = i + j * knn->N_grid;
                cell_offsets[knn->N_cell_offsets] = id_offset;

                // compute geometric distance for pruning later on
                double d = _boxsize_ * (double)(ring - 1) / (double)(knn->N_grid);
                cell_offset_dists[knn->N_cell_offsets] = d*d;

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
                    int id_offset = i + j * knn->N_grid + k * knn->N_grid * knn->N_grid;
                    cell_offsets[knn->N_cell_offsets] = id_offset;

                    // compute geometric distance for pruning later on
                    double d = _boxsize_ * (double)(ring - 1) / (double)(knn->N_grid);
                    cell_offset_dists[knn->N_cell_offsets] = d*d;

                    knn->N_cell_offsets++;
                }
            }
        }
#endif
    }

    // -------- allocate memory buffers and copy data --------
    gpuMallocNCopy((void**)&knn->d_cell_offsets, cell_offsets, knn->N_cell_offsets*sizeof(int));
    free(cell_offsets);

    gpuMallocNCopy((void**)&knn->d_cell_offset_dists, cell_offset_dists, knn->N_cell_offsets*sizeof(double));
    free(cell_offset_dists);

    POINT_TYPE *d_points = NULL;
    gpuMallocNCopy((void**)&d_points, pts, len_pts*sizeof(POINT_TYPE)); // input pts to GPU (temporary), freed after sorting into grid

    int Npow = pow(knn->N_grid, DIMENSION);
    gpuMallocNMemset((void**)&knn->d_counters, 0x00, Npow*sizeof(int)); // pts per grid cell
    gpuMallocNMemset((void**)&knn->d_ptrs, 0x00, Npow*sizeof(int)); // cell ptrs to start in d_stored_points

    gpuMallocNMemset((void**)&knn->d_globcounter, 0x00, sizeof(int)); // global counter
    gpuMallocNMemset((void**)&knn->d_stored_points, 0x00, knn->len_pts*sizeof(POINT_TYPE)); // will be filled with sorted points
    gpuMallocNMemset((void**)&knn->d_permutation, 0x00, knn->len_pts*sizeof(unsigned int)); // permutation to restore original order
    gpuMallocNMemset((void**)&knn->d_knearests, 0xFF, knn->len_pts*_K_*sizeof(int)); // result indices of knn

    // -------- reorganize input points by grid cell --------
    sort_points_into_grid(knn, d_points, len_pts);

    // no longer need orignal points on GPU
    gpuFree(d_points);

    return knn;
}

void sort_points_into_grid(knn_problem* knn, POINT_TYPE* d_points, int len_pts) {

    // -------- count points per grid cell --------
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (len_pts + threadsPerBlock - 1) / threadsPerBlock;

        #ifdef CPU_DEBUG
        cpu_count(blocksPerGrid, threadsPerBlock, d_points, len_pts, knn->N_grid, knn->d_counters);
        #endif
    }

    // -------- reserve memory ranges for each cell --------
    {
        int threadsPerBlock = 4;
        int blocksPerGrid = (pow(knn->N_grid, DIMENSION) + threadsPerBlock - 1) / threadsPerBlock;

        #ifdef CPU_DEBUG
        cpu_reserve(blocksPerGrid, threadsPerBlock, knn->N_grid, knn->d_counters, knn->d_globcounter, knn->d_ptrs);
        #endif
    }

    // -------- store points in their cell-organized locations -------
    {
        // reset counters: we'll reuse them for atomic allocation within each cell's range
        gpuMemset(knn->d_counters, 0x00, pow(knn->N_grid, DIMENSION)*sizeof(int));

        int threadsPerBlock = 256;
        int blocksPerGrid = (len_pts + threadsPerBlock - 1) / threadsPerBlock;

        // store oraganized points
        #ifdef CPU_DEBUG
        cpu_store(blocksPerGrid, threadsPerBlock, d_points, len_pts, knn->N_grid, knn->d_ptrs, knn->d_counters, knn->d_stored_points, knn->d_permutation);
        #endif
    }
}

#ifdef CPU_DEBUG
// counts how many poiunts are in each cell, stores in d_counters
void cpu_count(int blocksPerGrid, int threadsPerBlock, POINT_TYPE* d_points, int len_pts, int N_grid, int* d_counters) {
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
void cpu_reserve(int blocksPerGrid, int threadsPerBlock, int N_grid, const int* d_counters, int* d_globcounter, int* d_ptrs) {
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
void cpu_store(int blocksPerGrid, int threadsPerBlock, const POINT_TYPE* d_points, int len_pts, int N_grid, const int *d_ptrs, int* d_counters, POINT_TYPE* d_stored_points, unsigned int *d_permutation) {
    for (int blockId = 0; blockId < blocksPerGrid; blockId++) {
        for (int threadId = 0; threadId < threadsPerBlock; threadId++) {
            int id = threadsPerBlock * blockId + threadId;
            if (id < len_pts) {
                // determine cell for point
                POINT_TYPE p = d_points[id];
                int cell = cellFromPoint(N_grid, p);

                // claim a slot within the cell's range
                int pos = d_ptrs[cell] + atomicAdd(d_counters + cell, 1);

                d_stored_points[pos] = p;
                d_permutation[pos] = id;
            }
        }
    }
}
#endif

// get cell index from point position (will be __device__)
int cellFromPoint(int N_grid, POINT_TYPE point) {
    int i = (int)floor(point.x * (double) N_grid / _boxsize_);
    int j = (int)floor(point.y * (double) N_grid / _boxsize_);

    i = std::max(0, std::min(i, N_grid-1));
    j = std::max(0, std::min(j, N_grid-1));

#ifdef dim_2D
    return i + j * N_grid;
#else
    int k = (int)floor(point.z * (double) N_grid / _boxsize_);
    k = std::max(0, std::min(k, N_grid-1));
    return i + j * N_grid + k * N_grid * N_grid;
#endif
}

// -------- solve KNN problem --------
void solve(knn_problem* knn) {

    int threadsPerBlock = _KNN_BLOCK_SIZE_;
    int blocksPerGrid = (knn->len_pts + threadsPerBlock - 1) / _KNN_BLOCK_SIZE_;

    #ifdef CPU_DEBUG
    cpu_knearest(blocksPerGrid, threadsPerBlock, knn->N_grid, knn->len_pts, knn->d_ptrs, knn->d_counters, knn-> d_stored_points, knn->N_cell_offsets, knn->d_cell_offsets, knn->d_cell_offset_dists, knn->d_knearests);
    #endif

}

#ifdef CPU_DEBUG
void cpu_knearest(int blocksPerGrid, int threadsPerBlock, int N_grid, int len_pts, const int* d_ptrs, const int *d_counters, const POINT_TYPE* d_stored_points, int N_cell_offsets, const int* d_cell_offsets, const double* d_cell_offset_dists, unsigned int* d_knearest) {

    // __shared__ : each thread updates its k-nearest
    unsigned int knearest[_K_ * _KNN_BLOCK_SIZE_];
    double knearest_dists[_K_ * _KNN_BLOCK_SIZE_];

    for (int blockId = 0; blockId < blocksPerGrid; blockId++) {
        for (int threadId = 0; threadId < threadsPerBlock; threadId++) {
            int point_in = threadId + blockId * threadsPerBlock;
            if (point_in >= len_pts) return;
            if (point_in % 1000000 == 0) {
                std::cout << "Processing point " << point_in << " / " << len_pts << std::endl;
            }

            // point considered by this thread
            POINT_TYPE p = d_stored_points[point_in];

            // compute cell_id of point and offset for knn storage
            int cell_in = cellFromPoint(N_grid, p);
            int offs = threadId * _K_;

            // set knearest and knearest_dist to maximum values
            for (int i = 0; i < _K_; i++) {
                knearest[offs + i] = UINT_MAX;
                knearest_dists[offs + i] = DBL_MAX;
            }

            int search_cell_index = 0;

            // expanding ring search: find knn by checking cells in order
            do {
                // get the min dist for this cell
                double min_dist = d_cell_offset_dists[search_cell_index];

                // early termination: all cells are farther: we've found the true knn
                if (knearest_dists[offs] < min_dist) {break;}

                int cell = cell_in + d_cell_offsets[search_cell_index];

                // enusre the cell is within the grid
                if (cell >= 0 && cell < pow(N_grid, DIMENSION)) {
                    int cell_base = d_ptrs[cell]; // starting idx for this cell
                    int num = d_counters[cell]; // how many pts in this cell

                    // iterate over all pts in this cell
                    for (int ptr = cell_base; ptr < cell_base + num; ptr++) {
                        
                        // skip self comparison
                        if (ptr == point_in) {continue;}

                        // load candidate ngb and calc dist
                        POINT_TYPE p_cmp = d_stored_points[ptr];
                        double d = dist2_point(p, p_cmp);


                        // if new k-nearest neighbour
                        if (d < knearest_dists[offs]) {
                            knearest[offs] = ptr;
                            knearest_dists[offs] = d;
                            heapify(knearest + offs, knearest_dists + offs, 0, _K_);
                        }
                    }
                }
            } while (search_cell_index++ < N_cell_offsets);

            // if we exhausted all rings we might not have found all knn
            // mark with DBL_MAX for diagnostics
            if (search_cell_index == N_cell_offsets) {
                std::cerr << "Not sure if we found all ngb here... Thats a problem!" << std::endl;
            }

            heapsort(knearest + offs, knearest_dists + offs, _K_);

            for (int i = 0; i < _K_; i++) {
                d_knearest[point_in * _K_ + i] = knearest[offs + i];
            }
        }
    }
}
#endif

template <typename T> void inline swap_on_device(T& a, T& b) {
    T c(a); a=b; b=c;
}

void heapify(unsigned int *keys, double *vals, int node, int size) {
    int j = node;
    while (true) { 
        int left  = 2*j+1;
        int right = 2*j+2;
        int largest = j;
        if ( left<size && vals[ left]>vals[largest]) {
            largest = left;
        }
        if (right<size && vals[right]>vals[largest]) {
            largest = right;
        }
        if (largest==j) return;
        swap_on_device(vals[j], vals[largest]);
        swap_on_device(keys[j], keys[largest]);
        j = largest;
    }
}

void heapsort(unsigned int *keys, double *vals, int size) {
    while (size) {
        swap_on_device(vals[0], vals[size-1]);
        swap_on_device(keys[0], keys[size-1]);
        heapify(keys, vals, 0, --size);
    }
}


bool verify(knn_problem* knn, double tol, int max_report) {
    if (!knn) {
        std::cerr << "KNN verify: null knn problem." << std::endl;
        return false;
    }

    int n = knn->len_pts;
    int k = _K_;
    const POINT_TYPE* pts = knn->d_stored_points;
    const unsigned int* knn_idx = knn->d_knearests;

    if (!pts || !knn_idx) {
        std::cerr << "KNN verify: missing data buffers." << std::endl;
        return false;
    }

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        std::vector<double> best_dist(k, DBL_MAX);
        std::vector<unsigned int> best_idx(k, UINT_MAX);

        const POINT_TYPE& p = pts[i];

        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double d = dist2_point(p, pts[j]);

            if (d >= best_dist[k-1]) continue;

            int pos = k - 1;
            while (pos > 0 && d < best_dist[pos-1]) {
                best_dist[pos] = best_dist[pos-1];
                best_idx[pos] = best_idx[pos-1];
                pos--;
            }
            best_dist[pos] = d;
            best_idx[pos] = (unsigned int)j;
        }

        bool ok = true;
        for (int m = 0; m < k; m++) {
            unsigned int idx = knn_idx[i * k + m];
            if (idx == UINT_MAX) {
                ok = false;
                break;
            }
            double d = dist2_point(p, pts[idx]);
            double ref = best_dist[m];
            double diff = std::abs(d - ref);
            if (diff > tol * (1.0 + ref)) {
                ok = false;
                break;
            }
        }

        if (!ok) {
            mismatches++;
            if (mismatches <= max_report) {
                std::cerr << "KNN verify mismatch at point " << i
                          << " (" << mismatches << ")" << std::endl;
            }
        }
    }

    if (mismatches > 0) {
        std::cerr << "KNN verify failed: " << mismatches << " / " << n
                  << " points mismatch." << std::endl;
        return false;
    }

    std::cout << "KNN verify passed: all points match brute-force." << std::endl;
    return true;
}

// -------- other --------
void knn_free(knn_problem** knn) {
    gpuFree((*knn)->d_cell_offsets);
    gpuFree((*knn)->d_cell_offset_dists);
    gpuFree((*knn)->d_permutation);
    gpuFree((*knn)->d_counters);
    gpuFree((*knn)->d_ptrs);
    gpuFree((*knn)->d_globcounter);
    gpuFree((*knn)->d_stored_points);
    gpuFree((*knn)->d_knearests);
    free(*knn);
    *knn = NULL;
}

// -------- get stuff from gpu to cpu --------
POINT_TYPE* get_points(knn_problem* knn) {
    POINT_TYPE* pts = (POINT_TYPE*)malloc(knn->len_pts * sizeof(POINT_TYPE));
    gpuMemcpy(pts, knn->d_stored_points, knn->len_pts * sizeof(POINT_TYPE));
    return pts;
}

unsigned int* get_knearest(knn_problem* knn) {
    unsigned int* knearest = (unsigned int*)malloc(knn->len_pts * _K_ * sizeof(int));
    gpuMemcpy(knearest, knn->d_knearests, knn->len_pts * _K_ * sizeof(int));
    return knearest;
}

unsigned int* get_permutation(knn_problem* knn) {
    unsigned int* permutation = (unsigned int*)malloc(knn->len_pts*sizeof(int));
    gpuMemcpy(permutation, knn->d_permutation, knn->len_pts*sizeof(int));
    return permutation;
}

} // namespace knn
