#ifndef KNN_H
#define KNN_H

#include <string>
#include "global/allvars.h"
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
    int *d_counters;             // counters per cell,   dimx*dimy*dimz
    int *d_ptrs;                 // cell start pointers, dimx*dimy*dimz
    int *d_globcounter;          // global allocation counter, 1
    double3 *d_stored_points;     // input points sorted, numpoints 
    unsigned int *d_knearests;   // knn, allocated_points * KN
} knn_problem;

namespace knn {

knn_problem* init(double3 *pts, int len_pts);
//void solve();
void knn_free(knn_problem** knn);
//double3 get_points();
//unsigned int* get_knearests();
//unsigned int* get_permutation();

// mystic function
void printInfo();

} // namespace knn

#endif // KNN_H
