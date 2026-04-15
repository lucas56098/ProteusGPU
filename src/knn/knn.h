#ifndef KNN_H
#define KNN_H

#include "../io/input.h"
#include "../io/output.h"
#include "global/allvars.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

typedef struct knn_problem {
    int           len_pts;             // number of input points (current call)
    int           pts_capacity;        // allocated capacity for d_stored_points / d_permutation
    int           N_grid;              // grid resolution
    int           Npow;                // N_grid^DIMENSION (total grid cells)
    int           N_cell_offsets;      // actual number of cells in the offset grid
    int*          d_cell_offsets;      // cell offsets (sorted by rings), Nmax*Nmax*Nmax*Nmax
    double*       d_cell_offset_dists; // stores min dist to the cells in the rings
    unsigned int* d_permutation;       // allows to restore original point order
    int*          d_counters;          // counters per cell,   N_grid*N_grid*N_grid
    int*          d_ptrs;              // cell start pointers, N_grid*N_grid*N_grid
    int*          d_globcounter;       // global allocation counter, 1
    POINT_TYPE*   d_stored_points;     // input points sorted, numpoints
} knn_problem;

namespace knn {

    // computes offset grid and allocates buffers
    knn_problem* init_once(int n_hydro);

    // resets counters and sorts points into grid
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    void sort_points_into_grid(knn_problem* knn, const POINT_TYPE* pts, int len_pts);
    int  cellFromPoint(int N_grid, POINT_TYPE point);

    // compute K nearest neighbors for a single point
    void knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest);

    void heapify(unsigned int* keys, double* vals, int node, int size);
    template <typename T> void inline swap_on_device(T& a, T& b);
    void heapsort(unsigned int* keys, double* vals, int size);

    void knn_free(knn_problem** knn);

    static inline double dist2_point(const POINT_TYPE& a, const POINT_TYPE& b) {
#ifdef dim_2D
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return dx * dx + dy * dy;
#else
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
#endif
    }

} // namespace knn

#endif // KNN_H
