#ifndef KNN_H
#define KNN_H

#include <string>
#include "global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>

/* 
 * This part of the code is heavily inspired by the work of: Nicolas Ray, Dmitry Sokolov, 
 * Sylvain Lefebvre, Bruno L'evy, "Meshless Voronoi on the GPU", ACM Trans. Graph., 
 * vol. 37, no. 6, Dec. 2018. If you build upon this code, we recommend  
 * reading and citing their paper: https://doi.org/10.1145/3272127.3275092
 */ 

typedef struct {
    int len_pts;        // number of input points
    int N_grid;        // grid resolution
    int N_cell_offsets;        // actual number of cells in the offset grid
    int *d_cell_offsets;         // cell offsets (sorted by rings), Nmax*Nmax*Nmax*Nmax
    double *d_cell_offset_dists;  // stores min dist to the cells in the rings
    unsigned int *d_permutation; // allows to restore original point order
    int *d_counters;             // counters per cell,   N_grid*N_grid*N_grid
    int *d_ptrs;                 // cell start pointers, N_grid*N_grid*N_grid
    int *d_globcounter;          // global allocation counter, 1
    POINT_TYPE *d_stored_points;     // input points sorted, numpoints 
    unsigned int *d_knearests;   // knn, allocated_points * KN
} knn_problem;

namespace knn {

// prepare the knn problem
knn_problem* init(POINT_TYPE *pts, int len_pts);
void sort_points_into_grid(knn_problem* knn, POINT_TYPE* d_points, int len_pts);
#ifdef CPU_DEBUG
void cpu_count(int blocksPerGrid, int threadsPerBlock, POINT_TYPE* d_points, int len_pts, int N_grid, int* d_counters);
void cpu_reserve(int blocksPerGrid, int threadsPerBlock, int N_grid, const int* d_counters, int* d_globcounter, int* d_ptrs);
void cpu_store(int blocksPerGrid, int threadsPerBlock, const POINT_TYPE* d_points, int len_pts, int N_grid, const int *d_ptrs, int* d_counters, POINT_TYPE* d_stored_points, unsigned int *d_permutation);
#endif
int cellFromPoint(int N_grid, POINT_TYPE point);

// solve the knn problem
void solve(knn_problem* knn);
#ifdef CPU_DEBUG
void cpu_knearest(int blocksPerGrid, int threadsPerBlock, int N_grid, int len_pts, const int* d_ptrs, const int *d_counters, const POINT_TYPE* d_stored_points, int N_cell_offsets, const int* d_cell_offsets, const double* d_cell_offset_dists, unsigned int* d_knearest);
#endif
void heapify(unsigned int *keys, double *vals, int node, int size);
template <typename T> void inline swap_on_device(T& a, T& b);
void heapsort(unsigned int *keys, double *vals, int size);

void knn_free(knn_problem** knn);
POINT_TYPE* get_points(knn_problem* knn);
unsigned int* get_knearest(knn_problem* knn);
unsigned int* get_permutation(knn_problem* knn);

// verify knn results
bool verify(knn_problem* knn, double tol = 1e-8, int max_report = 5);

void write_knn_output(knn_problem* knn, ICData& icData, InputHandler& input, OutputHandler& output);

static inline double dist2_point(const POINT_TYPE& a, const POINT_TYPE& b) {
#ifdef dim_2D
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx*dx + dy*dy;
#else
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
#endif
}

} // namespace knn

#endif // KNN_H
