#ifndef KNN_H
#define KNN_H

#include "global/allvars.h"
#include <cfloat>
#include <cmath>

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
    double        buff;                // periodic ghost buffer; bucket grid spans [-buff, 1+buff]^d
    double        inv_boxsize;         // 1 / (1 + 2*buff), precomputed for cellFromPoint
} knn_problem;

namespace knn {

    // computes offset grid and allocates buffers
    knn_problem* init_once(int n_hydro);

    // resets counters and sorts points into grid
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    HD int cellFromPoint(int N_grid, double buff, double inv_boxsize, POINT_TYPE point);

    void knn_free(knn_problem** knn);

    HD static inline double dist2_point(const POINT_TYPE& a, const POINT_TYPE& b) {
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

    template <typename T> HD inline void swap_on_device(T& a, T& b) {
        T c(a);
        a = b;
        b = c;
    }

    HD inline void heapify(unsigned int* keys, double* vals, int node, int size) {
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

    HD inline void heapsort(unsigned int* keys, double* vals, int size) {
        while (size > 1) {
            swap_on_device(vals[0], vals[size - 1]);
            swap_on_device(keys[0], keys[size - 1]);
            size--;
            heapify(keys, vals, 0, size);
        }
    }

    // K nearest neighbours of point_in, sorted by distance (max-heap on stack)
    template <int K>
    HD void knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest) {
        // thread-private max-heap of K best candidates seen so far
        unsigned int local_knearest[K];
        double       local_dists[K];
        int          heap_size = 0;

        const POINT_TYPE* d_stored_points     = knn->d_stored_points;
        int               N_grid              = knn->N_grid;
        int               Npow_local          = knn->Npow;
        const int*        d_ptrs              = knn->d_ptrs;
        const int*        d_counters          = knn->d_counters;
        int               N_cell_offsets      = knn->N_cell_offsets;
        const int*        d_cell_offsets      = knn->d_cell_offsets;
        const double*     d_cell_offset_dists = knn->d_cell_offset_dists;
        int               len_pts             = knn->len_pts;

        POINT_TYPE p       = d_stored_points[point_in];
        int        cell_in = cellFromPoint(N_grid, knn->buff, knn->inv_boxsize, p);

        // initialize the heap with sentinel "infinitely far" entries
        for (int i = 0; i < K; i++) {
            local_knearest[i] = (unsigned int)point_in;
            local_dists[i]    = DBL_MAX;
        }

        // walk grid cells in expanding-distance order
        for (int search_cell_index = 0; search_cell_index < N_cell_offsets; search_cell_index++) {
            // terminate once heap is full and the ring's lower-bound exceeds our worst-in-heap
            double min_dist = d_cell_offset_dists[search_cell_index];
            if (heap_size == K && local_dists[0] < min_dist) { break; }

            int cell = cell_in + d_cell_offsets[search_cell_index];
            if (cell < 0 || cell >= Npow_local) { continue; }

            int cell_base = d_ptrs[cell];
            int num       = d_counters[cell];
            int cell_end  = cell_base + num;

            // defensive guard: skip corrupted cell ranges instead of reading OOB
            if (cell_base < 0 || num < 0 || cell_end < cell_base || cell_end > len_pts) { continue; }

            // test every point in this grid cell against the heap
            for (int ptr = cell_base; ptr < cell_end; ptr++) {
                if (ptr == point_in) { continue; }

                POINT_TYPE p_cmp = d_stored_points[ptr];
                double     d     = dist2_point(p, p_cmp);

                if (heap_size < K) {
                    // heap not yet full: append and sift up
                    int pos             = heap_size;
                    local_dists[pos]    = d;
                    local_knearest[pos] = (unsigned int)ptr;
                    heap_size++;

                    while (pos > 0) {
                        int parent = (pos - 1) / 2;
                        if (local_dists[parent] >= local_dists[pos]) { break; }
                        swap_on_device(local_dists[parent], local_dists[pos]);
                        swap_on_device(local_knearest[parent], local_knearest[pos]);
                        pos = parent;
                    }
                } else if (d < local_dists[0]) {
                    // heap full and this candidate beats the current worst — replace + sift down
                    local_dists[0]    = d;
                    local_knearest[0] = (unsigned int)ptr;
                    heapify(local_knearest, local_dists, 0, K);
                }
            }
        }

        // sort heap ascending so callers iterate near-to-far
        if (heap_size > 1) { heapsort(local_knearest, local_dists, heap_size); }

        for (int i = 0; i < K; i++) {
            out_knearest[i] = local_knearest[i];
        }
    }

} // namespace knn

#endif // KNN_H
