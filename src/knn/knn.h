#ifndef KNN_H
#define KNN_H

// Nearest neighbour search on a uniform grid of buckets.

#include "global/allvars.h"
#include <cfloat>
#include <cmath>

// the grid and the point list sorted into it, one per run
typedef struct knn_problem {
    int           len_pts;                  // points in the grid right now
    int           pts_capacity;             // points the arrays can hold
    int           N_grid;                   // buckets per axis, same for the whole run
    int           Npow;                     // buckets in total
    int           N_cell_offsets;           // entries of the two ring arrays
    int*          d_cell_offsets;           // index step of every ring, nearest ring first
    int*          d_cell_offset_axes;       // the same steps per axis, packed, to stop at the grid edge
    int           N_rings;                  // rings around the own bucket
    double        reach2;                   // the rings hold every point closer than this, squared
    double*       d_cell_offset_dists;      // smallest distance to that ring, squared
    unsigned int* d_permutation;            // sorted point -> its index in the input list
    int*          d_counters;               // points per bucket
    int*          d_ptrs;                   // first point of each bucket
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

    // one ring step per axis in one int, each axis in [-64, 63]
    HD inline int pack_ring_step(int di, int dj, int dk) {
        return (di + 64) | ((dj + 64) << 8) | ((dk + 64) << 16);
    }

    // bucket (ix, iy, iz) moved by a packed ring step is still in the grid
    HD inline bool ring_step_in_grid(int packed, int ix, int iy, int iz, int N_grid) {
        const int i = ix + (packed & 0xff) - 64;
        const int j = iy + ((packed >> 8) & 0xff) - 64;
        const int k = iz + ((packed >> 16) & 0xff) - 64;
        return i >= 0 && i < N_grid && j >= 0 && j < N_grid && k >= 0 && k < N_grid;
    }

    // bucket index -> position per axis, iz is 0 in 2D
    HD inline void bucket_coords(int cell, int N_grid, int* ix, int* iy, int* iz) {
        *ix = cell % N_grid;
        *iy = (cell / N_grid) % N_grid;
        *iz = cell / (N_grid * N_grid);
    }

    // the K nearest points of point_in, as sorted-list indices, nearest first
    template <int K> HD void knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest) {
        // the K best so far in a max heap, local_dists[0] is the worst of them
        unsigned int local_knearest[K];
        double       local_dists[K];
        int          heap_size = 0;

        const POINT_TYPE* d_stored_points     = knn->d_stored_points;
        int               N_grid              = knn->N_grid;
        const int*        d_ptrs              = knn->d_ptrs;
        const int*        d_counters          = knn->d_counters;
        int               N_cell_offsets      = knn->N_cell_offsets;
        const int*        d_cell_offsets      = knn->d_cell_offsets;
        const int*        d_cell_offset_axes  = knn->d_cell_offset_axes;
        const double*     d_cell_offset_dists = knn->d_cell_offset_dists;
        int               len_pts             = knn->len_pts;

        POINT_TYPE p       = d_stored_points[point_in];
        int        cell_in = cell_from_point(N_grid, knn->grid_lo, knn->inv_cell_size, p);
        int        ix, iy, iz;
        bucket_coords(cell_in, N_grid, &ix, &iy, &iz);

        // if there are fewer than K neighbours, the rest stay the point itself
        for (int i = 0; i < K; i++) {
            local_knearest[i] = (unsigned int)point_in;
            local_dists[i]    = DBL_MAX;
        }

        // rings from near to far, stop when the heap is full and closer than the ring
        bool stopped_early = false;
        for (int search_cell_index = 0; search_cell_index < N_cell_offsets; search_cell_index++) {
            double min_dist = d_cell_offset_dists[search_cell_index];
            if (heap_size == K && local_dists[0] < min_dist) {
                stopped_early = true;
                break;
            }

            // no wrap into the next row, the far side of the grid is not a neighbour
            if (!ring_step_in_grid(d_cell_offset_axes[search_cell_index], ix, iy, iz, N_grid)) { continue; }
            int cell = cell_in + d_cell_offsets[search_cell_index];

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

        // all rings walked: past their reach a nearer point may be missing, so those count as not found
        if (!stopped_early) {
            for (int i = 0; i < heap_size; i++) {
                if (local_dists[i] >= knn->reach2) {
                    local_knearest[i] = (unsigned int)point_in;
                    local_dists[i]    = DBL_MAX;
                }
            }
        }

        for (int i = 0; i < K; i++) {
            out_knearest[i] = local_knearest[i];
        }
    }

} // namespace knn

#endif
