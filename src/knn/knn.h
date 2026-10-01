#ifndef KNN_H
#define KNN_H

// Nearest neighbour search over one Morton sorted point list, on a bucket grid or on a tree.

#include "global/allvars.h"
#include "keys.h"
#include <cfloat>
#include <cmath>
#include <utility>
#include <vector>

// an internal node of the tree: its two children and their boxes; the box of a leaf child is its point
struct TreeNode {
    POINT_TYPE lo[2];
    POINT_TYPE hi[2];
    int        child[2];
};

// the sorted point list and the search structures, one per run
typedef struct knn_problem {
    // points in key order; both searches index them by sorted point (sid)
    int           len_pts;         // points in the list right now
    int           pts_capacity;    // points the arrays can hold
    POINT_TYPE*   d_stored_points; // the points, sorted
    unsigned int* d_permutation;   // sorted point -> its index in the input list
    uint64_t*     d_keys;          // Morton key of each sorted point
    uint64_t*     d_keys_alt;      // scratch of the sort
    unsigned int* d_perm_alt;
    unsigned int* d_sort_scratch;

    // grid only: buckets are key grid cells 2^shift wide, the reach follows their size
    int    N_grid;       // buckets per axis the grid arrays hold, same for the whole run
    int    Npow;         // buckets in total
    int    shift;        // bucket = key coordinate >> shift
    int    bucket_lo[3]; // first bucket per axis in this build
    double cell_size;    // bucket width
    double reach2;       // the grid only returns points closer than this, squared

    // grid search
    int     N_cell_offsets;           // entries of the ring arrays
    int*    d_cell_offsets;           // index step of every ring, nearest ring first
    int*    d_cell_offset_axes;       // the same steps per axis, packed, to stop at the grid edge
    int     N_rings;                  // rings around the own bucket
    double* d_cell_offset_dists;      // smallest distance to that ring, squared
    double* d_cell_offset_dists_unit; // the same in bucket units
    int*    d_counters;               // points per bucket
    int*    d_ptrs;                   // first point of each bucket

    // tree: internal nodes [0, len_pts - 1), sorted point s is the leaf len_pts - 1 + s
    TreeNode*     d_nodes;
    int*          d_parent; // of every node, -1 at the root
    unsigned int* d_visits; // scratch of the box pass
} knn_problem;

namespace knn {

#ifdef KNN_TREE
    constexpr bool USE_TREE = true;
#else
    constexpr bool USE_TREE = false;
#endif
#ifdef KNN_CROSSCHECK
    constexpr bool CROSS_CHECK = true;
#else
    constexpr bool CROSS_CHECK = false;
#endif
    constexpr bool BUILD_GRID = !USE_TREE || CROSS_CHECK;
    constexpr bool BUILD_TREE = USE_TREE || CROSS_CHECK;

    constexpr int TREE_STACK = 96; // a path has at most 63 + 31 internal nodes, one per common prefix length

    // allocates the arrays once per run; N_grid_restored > 0 comes from a snapshot
    knn_problem* init_once(int n_hydro, int N_grid_restored);

    // sorts the points and builds the search structure, once per mesh build
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    // frees everything and clears the pointer
    void knn_free(knn_problem** knn);

    // room for more points
    void knn_grow(knn_problem* knn, int new_pts_capacity);

    // the max_k nearest points of sorted point sid, as (distance^2, sid), nearest first; the grid only within its reach
    std::vector<std::pair<double, int>> nearest_on_host(const knn_problem* knn, int sid, int max_k);

    // every sorted point with distance^2 <= r2 from p
    void points_within_on_host(const knn_problem* knn, POINT_TYPE p, double r2, std::vector<int>* out);

    // sorted point sid was moved in place; the search has to find it at the new position
    void point_moved(knn_problem* knn, int sid);

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

    // strict order of the candidates: by distance, equal distances by sorted point
    HD inline bool comes_before(double da, unsigned int ia, double db, unsigned int ib) {
        return da < db || (da == db && ia < ib);
    }

    template <typename T> HD inline void swap_on_device(T& a, T& b) {
        T c(a);
        a = b;
        b = c;
    }

    // the K first candidates so far as a max heap, ids[0] is the last of them
    template <int K> struct KBest {
        unsigned int ids[K];
        double       d2[K];
        int          size;

        HD bool   full() const { return size == K; }
        HD double worst() const { return d2[0]; }

        // moves one entry down until the heap is valid again
        HD void sift_down(int node, int n) {
            int j = node;
            while (true) {
                const int left  = 2 * j + 1;
                const int right = 2 * j + 2;
                int       last  = j;
                if (left < n && comes_before(d2[last], ids[last], d2[left], ids[left])) last = left;
                if (right < n && comes_before(d2[last], ids[last], d2[right], ids[right])) last = right;
                if (last == j) return;
                swap_on_device(d2[j], d2[last]);
                swap_on_device(ids[j], ids[last]);
                j = last;
            }
        }

        // keeps it if it is among the K first
        HD void push(double d, unsigned int id) {
            if (size < K) {
                int pos  = size++;
                d2[pos]  = d;
                ids[pos] = id;
                while (pos > 0) {
                    const int parent = (pos - 1) / 2;
                    if (!comes_before(d2[parent], ids[parent], d2[pos], ids[pos])) break;
                    swap_on_device(d2[parent], d2[pos]);
                    swap_on_device(ids[parent], ids[pos]);
                    pos = parent;
                }
            } else if (comes_before(d, id, d2[0], ids[0])) {
                d2[0]  = d;
                ids[0] = id;
                sift_down(0, K);
            }
        }

        // into order, first candidate first
        HD void sort() {
            for (int n = size; n > 1; n--) {
                swap_on_device(d2[0], d2[n - 1]);
                swap_on_device(ids[0], ids[n - 1]);
                sift_down(0, n - 1);
            }
        }
    };

    // ============================================================================
    // grid search
    // ============================================================================

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

    // bucket on one axis; a point outside the grid gets the edge bucket
    HD inline int bucket_axis(double x, const knn_problem* knn, int a) {
        const int b = (int)(key_coord(x) >> knn->shift) - knn->bucket_lo[a];
        return imax(0, imin(b, knn->N_grid - 1));
    }

    // bucket a point falls into
    HD inline int cell_from_point(const knn_problem* knn, const POINT_TYPE& p) {
        const int N = knn->N_grid;
#ifdef dim_2D
        return bucket_axis(p.x, knn, 0) + bucket_axis(p.y, knn, 1) * N;
#else
        return bucket_axis(p.x, knn, 0) + bucket_axis(p.y, knn, 1) * N + bucket_axis(p.z, knn, 2) * N * N;
#endif
    }

    // the K nearest points within the reach on the grid, as sorted points, nearest first
    template <int K> HD void grid_knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest) {
        KBest<K> best;
        best.size = 0;

        const POINT_TYPE* d_stored_points     = knn->d_stored_points;
        const int         N_grid              = knn->N_grid;
        const int*        d_ptrs              = knn->d_ptrs;
        const int*        d_counters          = knn->d_counters;
        const int         N_cell_offsets      = knn->N_cell_offsets;
        const int*        d_cell_offsets      = knn->d_cell_offsets;
        const int*        d_cell_offset_axes  = knn->d_cell_offset_axes;
        const double*     d_cell_offset_dists = knn->d_cell_offset_dists;
        const int         len_pts             = knn->len_pts;

        const POINT_TYPE p       = d_stored_points[point_in];
        const int        cell_in = cell_from_point(knn, p);
        int              ix, iy, iz;
        bucket_coords(cell_in, N_grid, &ix, &iy, &iz);

        // rings from near to far, stop when the heap is full and closer than the ring
        bool stopped_early = false;
        for (int search_cell_index = 0; search_cell_index < N_cell_offsets; search_cell_index++) {
            if (best.full() && best.worst() < d_cell_offset_dists[search_cell_index]) {
                stopped_early = true;
                break;
            }

            // no wrap into the next row, the far side of the grid is not a neighbour
            if (!ring_step_in_grid(d_cell_offset_axes[search_cell_index], ix, iy, iz, N_grid)) { continue; }
            const int cell = cell_in + d_cell_offsets[search_cell_index];

            const int cell_base = d_ptrs[cell];
            const int num       = d_counters[cell];
            const int cell_end  = cell_base + num;

            if (cell_base < 0 || num < 0 || cell_end < cell_base || cell_end > len_pts) { continue; }

            for (int ptr = cell_base; ptr < cell_end; ptr++) {
                if (ptr == point_in) { continue; } // never itself
                best.push(dist2_point(p, d_stored_points[ptr]), (unsigned int)ptr);
            }
        }

        // the cell build clips in this order, so it needs the nearest first
        best.sort();

        // all rings walked: past their reach a nearer point may be missing, so those count as not found
        int found = best.size;
        if (!stopped_early) {
            while (found > 0 && best.d2[found - 1] >= knn->reach2)
                found--;
        }

        // fewer than K: the rest stay the point itself
        for (int i = 0; i < K; i++) {
            out_knearest[i] = (i < found) ? best.ids[i] : (unsigned int)point_in;
        }
    }

    // ============================================================================
    // tree search
    // ============================================================================

    // squared distance from p to the box; never larger than that of a point inside it
    HD inline double dist2_box(const POINT_TYPE& lo, const POINT_TYPE& hi, const POINT_TYPE& p) {
        const double dx = (p.x < lo.x) ? lo.x - p.x : ((p.x > hi.x) ? p.x - hi.x : 0.0);
        const double dy = (p.y < lo.y) ? lo.y - p.y : ((p.y > hi.y) ? p.y - hi.y : 0.0);
#ifdef dim_2D
        return dx * dx + dy * dy;
#else
        const double dz = (p.z < lo.z) ? lo.z - p.z : ((p.z > hi.z) ? p.z - hi.z : 0.0);
        return dx * dx + dy * dy + dz * dz;
#endif
    }

    // a box margin for rounding: a node is only skipped when it is clearly too far
    constexpr double TREE_PRUNE_SLACK = 1.0 + 1e-12;

    // the K nearest points on the tree, as sorted points, nearest first
    template <int K> HD void tree_knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest) {
        KBest<K> best;
        best.size = 0;

        const int n = knn->len_pts;
        if (n >= 2) {
            const TreeNode*  nodes = knn->d_nodes;
            const int        leaf0 = n - 1;
            const POINT_TYPE p     = knn->d_stored_points[point_in];

            // pending nodes with the distance to their box
            int    stack_node[TREE_STACK];
            double stack_d[TREE_STACK];
            int    sp   = 0;
            int    node = 0;
            while (node >= 0) {
                // leaves right away, internal children by their box
                const TreeNode& nd       = nodes[node];
                double          d_box[2] = {0.0, 0.0};
                bool            inner[2] = {false, false};
                for (int h = 0; h < 2; h++) {
                    const int c = nd.child[h];
                    if (c >= leaf0) {
                        const int s = c - leaf0;
                        if (s == point_in) continue;
                        best.push(dist2_point(p, nd.lo[h]), (unsigned int)s);
                    } else {
                        d_box[h] = dist2_box(nd.lo[h], nd.hi[h], p);
                        inner[h] = true;
                    }
                }
                bool visit[2];
                for (int h = 0; h < 2; h++) {
                    visit[h] = inner[h] && (!best.full() || d_box[h] <= best.worst() * TREE_PRUNE_SLACK);
                }

                // the nearer child next, the other one later
                if (visit[0] && visit[1]) {
                    const int near = (d_box[1] < d_box[0]) ? 1 : 0;
                    if (sp < TREE_STACK) {
                        stack_node[sp] = nd.child[1 - near];
                        stack_d[sp]    = d_box[1 - near];
                        sp++;
                    }
                    node = nd.child[near];
                } else if (visit[0] || visit[1]) {
                    node = visit[0] ? nd.child[0] : nd.child[1];
                } else {
                    // back to the nearest pending node that can still hold a better point
                    node = -1;
                    while (sp > 0) {
                        sp--;
                        const double d = stack_d[sp];
                        if (!best.full() || d <= best.worst() * TREE_PRUNE_SLACK) {
                            node = stack_node[sp];
                            break;
                        }
                    }
                }
            }
        }

        best.sort();
        for (int i = 0; i < K; i++) {
            out_knearest[i] = (i < best.size) ? best.ids[i] : (unsigned int)point_in;
        }
    }

    // the K nearest points of sorted point point_in, nearest first, the grid only within its reach;
    // missing ones are point_in itself
    template <int K> HD void knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest) {
        if (USE_TREE) {
            tree_knn_for_point<K>(point_in, knn, out_knearest);
        } else {
            grid_knn_for_point<K>(point_in, knn, out_knearest);
        }
    }

} // namespace knn

#endif
