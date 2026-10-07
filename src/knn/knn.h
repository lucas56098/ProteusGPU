#ifndef KNN_H
#define KNN_H

// Nearest neighbour search over one Morton sorted point list, on a tree.

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

// the sorted point list and the tree, one per run
typedef struct knn_problem {
    // points in key order; the search indexes them by sorted point (sid)
    int           len_pts;         // points in the list right now
    int           pts_capacity;    // points the arrays can hold
    POINT_TYPE*   d_stored_points; // the points, sorted
    unsigned int* d_permutation;   // sorted point -> its index in the input list
    uint64_t*     d_keys;          // Morton key of each sorted point
    uint64_t*     d_keys_alt;      // scratch of the sort
    unsigned int* d_perm_alt;
    unsigned int* d_sort_scratch;

    // tree: internal nodes [0, len_pts - 1), sorted point s is the leaf len_pts - 1 + s
    TreeNode*     d_nodes;
    int*          d_parent; // of every node, -1 at the root
    unsigned int* d_visits; // scratch of the box pass
} knn_problem;

namespace knn {

    constexpr int TREE_STACK = 96; // a path has at most 63 + 31 internal nodes, one per common prefix length

    // allocates the arrays once per run, for this many points
    knn_problem* init_once(int capacity);

    // sorts the points and builds the tree, once per mesh build
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    // the same when pts starts with the points of the last prepare, in the same order: only the points behind them
    // are sorted, then merged in. Same order and same tree as prepare
    void prepare_appended(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    // the caller put the points of the last prepare into its sorted order: the tree stays, the permutation
    // becomes the identity
    void take_sorted_order(knn_problem* knn);

    // frees everything and clears the pointer
    void knn_free(knn_problem** knn);

    // room for more points
    void knn_grow(knn_problem* knn, int new_pts_capacity);

    // the max_k nearest points of sorted point sid, as (distance^2, sid), nearest first
    std::vector<std::pair<double, int>> nearest_on_host(const knn_problem* knn, int sid, int max_k);

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

    // the K nearest points of sorted point point_in, nearest first; missing ones are point_in itself
    template <int K> HD void knn_for_point(int point_in, const knn_problem* knn, unsigned int* out_knearest) {
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

} // namespace knn

#endif
