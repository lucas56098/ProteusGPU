#ifndef KNN_H
#define KNN_H

// Nearest neighbour search on a uniform grid of buckets.

#include "global/allvars.h"
#include <cfloat>
#include <cmath>

// the grid and the point list sorted into it, one per run
typedef struct knn_problem {
    int           len_pts;             // points in the grid right now
    int           pts_capacity;        // points the arrays can hold
    int           N_grid;              // buckets per axis, same for the whole run
    int           Npow;                // buckets in total
    int           N_cell_offsets;      // entries of the two ring arrays
    int*          d_cell_offsets;      // index step of every ring, nearest ring first
    double*       d_cell_offset_dists; // smallest distance to that ring, squared
    unsigned int* d_permutation;       // sorted point -> its index in the input list
    int*          d_counters;          // points per bucket
    int*          d_ptrs;              // first point of each bucket
    int*          d_globcounter;
    int*          d_scan_scratch;           // scratch of the scan over d_counters
    int*          d_bucket_ids;             // scratch of the sort
    POINT_TYPE*   d_stored_points;          // the points, bucket by bucket
    double        buff;                     // ghost band width, used when there is no data extent
    double        inv_boxsize;              // 1 / (box plus both bands)
    double        grid_lo[3];               // lower corner of the grid
    double        inv_cell_size;            // 1 / bucket size
    double*       d_cell_offset_dists_unit; // the same distances in bucket units, scaled when the grid changes
} knn_problem;

namespace knn {

    // allocates the grid once per run; N_grid_restored > 0 comes from a snapshot
    knn_problem* init_once(int n_hydro, int N_grid_restored);

    // sorts the points into the grid
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    // bucket a point falls into
    HD int cell_from_point(int N_grid, const double* grid_lo, double inv_cell_size, POINT_TYPE point);

    // puts the grid on this rank's data extent, or on the whole box if there is none
    void set_local_extent(knn_problem* knn, const double* data_lo, const double* data_hi);

    // frees the grid and clears the pointer
    void knn_free(knn_problem** knn);

    // room for more points, the buckets stay as they are
    void knn_grow(knn_problem* knn, int new_pts_capacity);

    // squared distance between two points
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

    // moves one node down until vals is a max heap again
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

    // sorts both arrays by increasing distance
    HD inline void heapsort(unsigned int* keys, double* vals, int size) {
        while (size > 1) {
            swap_on_device(vals[0], vals[size - 1]);
            swap_on_device(keys[0], keys[size - 1]);
            size--;
            heapify(keys, vals, 0, size);
        }
    }

    // the K nearest points of point_in, as sorted-list indices, nearest first
    template <int K> HD void knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest) {
        // the K best so far in a max heap, local_dists[0] is the worst of them
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
        int        cell_in = cell_from_point(N_grid, knn->grid_lo, knn->inv_cell_size, p);

        // if there are fewer than K neighbours, the rest stay the point itself
        for (int i = 0; i < K; i++) {
            local_knearest[i] = (unsigned int)point_in;
            local_dists[i]    = DBL_MAX;
        }

        // rings from near to far, stop when the heap is full and closer than the ring
        for (int search_cell_index = 0; search_cell_index < N_cell_offsets; search_cell_index++) {
            double min_dist = d_cell_offset_dists[search_cell_index];
            if (heap_size == K && local_dists[0] < min_dist) { break; }

            int cell = cell_in + d_cell_offsets[search_cell_index];
            if (cell < 0 || cell >= Npow_local) { continue; } // ring step left the grid

            int cell_base = d_ptrs[cell];
            int num       = d_counters[cell];
            int cell_end  = cell_base + num;

            if (cell_base < 0 || num < 0 || cell_end < cell_base || cell_end > len_pts) { continue; }

            for (int ptr = cell_base; ptr < cell_end; ptr++) {
                if (ptr == point_in) { continue; } // never itself

                POINT_TYPE p_cmp = d_stored_points[ptr];
                double     d     = dist2_point(p, p_cmp);

                // heap not full yet: add it and move it up
                if (heap_size < K) {
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
                } else if (d < local_dists[0]) { // closer than the worst: replace it and move it down
                    local_dists[0]    = d;
                    local_knearest[0] = (unsigned int)ptr;
                    heapify(local_knearest, local_dists, 0, K);
                }
            }
        }

        // the cell build clips in this order, so it needs the nearest first
        if (heap_size > 1) { heapsort(local_knearest, local_dists, heap_size); }

        for (int i = 0; i < K; i++) {
            out_knearest[i] = local_knearest[i];
        }
    }

} // namespace knn

#endif
