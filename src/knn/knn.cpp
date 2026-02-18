#include "knn.h"
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>

// CONSTRUCTION SITE: nothing works yet :D

namespace knn {

knn_problem* init(double3 *pts, int len_pts) {
    
    // -------- allocate the main data structure --------
    knn_problem *knn = (knn_problem*)malloc(sizeof(knn_problem));

    knn->len_pts = len_pts;
    knn->N_grid = std::max(1,(int)round(pow(len_pts / 3.1f, 1.0f / 3.0)));
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
    }

    // -------- allocate memory buffers and copy data --------
    gpuMallocNCopy((void**)&knn->d_cell_offsets, cell_offsets, knn->N_cell_offsets*sizeof(int));
    free(cell_offsets);

    gpuMallocNCopy((void**)&knn->d_cell_offset_dists, cell_offset_dists, knn->N_cell_offsets*sizeof(double));
    free(cell_offset_dists);

    double3 *d_points = NULL;
    gpuMallocNCopy((void**)&d_points, pts, len_pts*sizeof(double3)); // input pts to GPU (temporary), freed after sorting into grid
    gpuMallocNMemset((void**)&knn->d_counters, 0x00, knn->N_grid*knn->N_grid*knn->N_grid*sizeof(int)); // pts per grid cell
    gpuMallocNMemset((void**)&knn->d_ptrs, 0x00, knn->N_grid*knn->N_grid*knn->N_grid*sizeof(int)); // cell ptrs to start in d_stored_points
    gpuMallocNMemset((void**)&knn->d_globcounter, 0x00, sizeof(int)); // global counter
    gpuMallocNMemset((void**)&knn->d_stored_points, 0x00, knn->len_pts*sizeof(double3)); // will be filled with sorted points
    gpuMallocNMemset((void**)&knn->d_permutation, 0x00, knn->len_pts*sizeof(unsigned int)); // permutation to restore original order
    gpuMallocNMemset((void**)&knn->d_knearests, 0xFF, knn->len_pts*_K_*sizeof(int)); // result indices of knn

    // -------- reorganize input points by grid cell --------
    sort_points_into_grid(knn, d_points, len_pts);

    // no longer need orignal points on GPU
    gpuFree(d_points);

    return knn;
}

void sort_points_into_grid(knn_problem* knn, double3* d_points, int len_pts) {

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
        int blocksPerGrid = (knn->N_grid*knn->N_grid*knn->N_grid + threadsPerBlock - 1) / threadsPerBlock;

        #ifdef CPU_DEBUG
        cpu_reserve(blocksPerGrid, threadsPerBlock, knn->N_grid, knn->d_counters, knn->d_globcounter, knn->d_ptrs);
        #endif
    }

    // -------- store points in their cell-organized locations -------
    {
        // reset counters: we'll reuse them for atomic allocation within each cell's range
        gpuMemset(knn->d_counters, 0x00, knn->N_grid*knn->N_grid*knn->N_grid*sizeof(int));

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
void cpu_count(int blocksPerGrid, int threadsPerBlock, double3* d_points, int len_pts, int N_grid, int* d_counters) {
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
            if (id < N_grid*N_grid*N_grid) {
                int count = d_counters[id]; // read how many points are in this cell
                if (count > 0) {
                    d_ptrs[id] = atomicAdd(d_globcounter, count); // store starting pos in ptrs
                }
            }
        }
    }
}

// stores points in their cell-organized locations
void cpu_store(int blocksPerGrid, int threadsPerBlock, const double3* d_points, int len_pts, int N_grid, const int *d_ptrs, int* d_counters, double3* d_stored_points, unsigned int *d_permutation) {
    for (int blockId = 0; blockId < blocksPerGrid; blockId++) {
        for (int threadId = 0; threadId < threadsPerBlock; threadId++) {
            int id = threadsPerBlock * blockId + threadId;
            if (id < len_pts) {
                // determine cell for point
                double3 p = d_points[id];
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
int cellFromPoint(int N_grid, double3 point) {
    int i = (int)floor(point.x * (double) N_grid / _boxsize_);
    int j = (int)floor(point.y * (double) N_grid / _boxsize_);
    int k = (int)floor(point.z * (double) N_grid / _boxsize_);

    i = std::max(0, std::min(i, N_grid-1));
    j = std::max(0, std::min(j, N_grid-1));
    k = std::max(0, std::min(k, N_grid-1));

    return i + j * N_grid + k * N_grid * N_grid;
}


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

void printInfo() {
    std::cout << "Argh—you caught me. Watch me morph into an SPH particle, bye." << std::endl;
}

} // namespace knn
