#include "knn.h"
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <cstring>

// CONSTRUCTION SITE: nothing works yet :D

namespace knn {

knn_problem* init(double3 *pts, int len_pts) {
    
    // allocate the main data structure
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

    // calc offsets for all rings up to N_max
    for (int ring = 1; ring < N_max; ring++) {
        for (int k = -N_max; k <= N_max; k++) {
            for (int j = -N_max; j <= N_max; j++) {
                for (int i = -N_max; i <= N_max; i++) {
                    if (std::max(abs(i), std::max(abs(j), abs(k))) != ring) continue;
                    // everything below is only executed if cell is inside current ring

                    // compute linear offset in the flattened 3D grid array
                    int id_offset = i + j * knn->N_grid + k * knn->N_grid * knn->N_grid;
                    cell_offsets[knn->N_cell_offsets] = id_offset;

                    // compute geometric distance for pruning later on (assumes box size 1000.)
                    double d = 1000. * (double)(ring - 1) / (double)(knn->N_grid);
                    cell_offset_dists[knn->N_cell_offsets] = d*d;

                    knn->N_cell_offsets++;
                }
            }
        }
    }

    // allocate memory buffers and copy data
    gpuMallocNCopy((void**)&knn->d_cell_offsets, cell_offsets, knn->N_cell_offsets*sizeof(int));
    free(cell_offsets);

    gpuMallocNCopy((void**)&knn->d_cell_offset_dists, cell_offset_dists, knn->N_cell_offsets*sizeof(double));
    free(cell_offset_dists);

    // copy input pts to GPU (temporarily)
    // will be freed after kn_firstbuild
    double3 *d_points = NULL;
    gpuMallocNCopy((void**)&d_points, pts, len_pts*sizeof(double3));

    // counter how many points fall into each grid cell
    gpuMallocNMemset((void**)&knn->d_counters, 0x00, knn->N_grid*knn->N_grid*knn->N_grid*sizeof(int));

    // cell pointers: for each grid cell, where does its data start in the flat d_stored_points array?
    gpuMallocNMemset((void**)&knn->d_ptrs, 0x00, knn->N_grid*knn->N_grid*knn->N_grid*sizeof(int));

    // global counter: total number of points stored
    gpuMallocNMemset((void**)&knn->d_globcounter, 0x00, sizeof(int));

    // input points reorganized by grid cells
    gpuMallocNMemset((void**)&knn->d_stored_points, 0x00, knn->len_pts*sizeof(double3));

    // result array: for each point, store the indices of its KNN
    gpuMallocNMemset((void**)&knn->d_knearests, 0xFF, knn->len_pts*_K_*sizeof(int));

    // hier fehlt noch: reorganize input points by grid cell -> besserer fct name: sort_points_into_grid()?
    //kn_firstbuild()

    // no longer need orignal points on GPU
    gpuFree(d_points);

    return knn;
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
