// implements the neighbour search (knn.h)

#include "../global/globals.h"
#include "../global/structs.h"
#include "../profiler/profiler.h"
#include "knn.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <utility>

#include "tree.cu"

namespace knn {

    static void allocate_point_arrays(knn_problem* knn, int capacity);
    static void free_point_arrays(knn_problem* knn);
    static void sort_points(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    // allocates everything for this many points; knn_grow makes room for more
    knn_problem* init_once(int capacity) {
        knn_problem* knn = gpu_alloc<knn_problem>(1);
        std::memset(knn, 0, sizeof(knn_problem));
        allocate_point_arrays(knn, capacity);
        return knn;
    }

    // the point list, the sort scratch and the tree, for this many points
    static void allocate_point_arrays(knn_problem* knn, int capacity) {
        knn->pts_capacity    = capacity;
        knn->d_stored_points = gpu_calloc<POINT_TYPE>(capacity);
        knn->d_permutation   = gpu_calloc<unsigned int>(capacity);
        knn->d_perm_alt      = gpu_calloc<unsigned int>(capacity);
        knn->d_keys          = gpu_calloc<uint64_t>(capacity);
        knn->d_keys_alt      = gpu_calloc<uint64_t>(capacity);
        knn->d_sort_scratch  = gpu_calloc<unsigned int>(sort_scratch_size((size_t)capacity));

        // the search reads these on every build
        gpu_advise_gpu_preferred(knn->d_stored_points, capacity * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(knn->d_keys, capacity * sizeof(uint64_t));
        gpu_advise_gpu_preferred(knn->d_keys_alt, capacity * sizeof(uint64_t));

        knn->d_nodes  = gpu_calloc<TreeNode>(capacity);
        knn->d_parent = gpu_calloc<int>(2 * (size_t)capacity);
        knn->d_visits = gpu_calloc<unsigned int>(capacity);
        gpu_advise_gpu_preferred(knn->d_nodes, capacity * sizeof(TreeNode));
    }

    static void free_point_arrays(knn_problem* knn) {
        gpu_free(knn->d_stored_points);
        gpu_free(knn->d_permutation);
        gpu_free(knn->d_perm_alt);
        gpu_free(knn->d_keys);
        gpu_free(knn->d_keys_alt);
        gpu_free(knn->d_sort_scratch);
        gpu_free(knn->d_nodes);
        gpu_free(knn->d_parent);
        gpu_free(knn->d_visits);
    }

    // sorts len_pts points and builds the tree, called once per mesh build
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {

        if (len_pts > knn->pts_capacity) {
            proteus_mpi::exit_failure(
                "KNN: Error! point count %d exceeds pre-allocated capacity %d. Increase ghost headroom.\n",
                len_pts,
                knn->pts_capacity);
        }

        knn->len_pts = len_pts;
        sort_points(knn, pts, len_pts);
        build_tree(knn);
    }

    void prepare_appended(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {
        const int n_old = knn->len_pts;
        const int n_new = len_pts - n_old;

        // the new points sort behind the old ones in the key arrays; without the room, or without old points, the
        // whole list is sorted
        if (n_old == 0 || n_new < 0 || (size_t)n_old + 2 * (size_t)n_new > (size_t)knn->pts_capacity) {
            prepare(knn, pts, len_pts);
            return;
        }
        if (n_new == 0) return;
        knn->len_pts = len_pts;

        // the keys of the new points, sorted on their own
        uint64_t*     new_keys     = knn->d_keys + n_old;
        unsigned int* new_perm     = knn->d_permutation + n_old;
        uint64_t*     new_keys_alt = new_keys + n_new;
        unsigned int* new_perm_alt = new_perm + n_new;
        {
            uint64_t*     keys = new_keys;
            unsigned int* perm = new_perm;
            parallel_for<_KNN_BLOCK_SIZE_>("KEYS_NEW", n_new, [=] HD(int j) {
                keys[j] = morton_key(pts[n_old + j]);
                perm[j] = (unsigned int)(n_old + j);
            });
        }
        parallel_sort_pairs("SORT_NEW",
                            (size_t)n_new,
                            KEY_TOTAL_BITS,
                            new_keys,
                            new_perm,
                            new_keys_alt,
                            new_perm_alt,
                            knn->d_sort_scratch);

        // every point finds its place in the merged list; an old point goes before a new one with the same key, as in
        // the stable sort of the whole list
        const uint64_t*     old_keys = knn->d_keys;
        const unsigned int* old_perm = knn->d_permutation;
        const uint64_t*     nk       = new_keys;
        const unsigned int* np       = new_perm;
        uint64_t*           out_keys = knn->d_keys_alt;
        unsigned int*       out_perm = knn->d_perm_alt;
        parallel_for<_KNN_BLOCK_SIZE_>("MERGE_OLD", n_old, [=] HD(int i) {
            const size_t at = (size_t)i + lower_bound_of(nk, (size_t)n_new, old_keys[i]);
            out_keys[at]    = old_keys[i];
            out_perm[at]    = old_perm[i];
        });
        parallel_for<_KNN_BLOCK_SIZE_>("MERGE_NEW", n_new, [=] HD(int j) {
            const size_t at = (size_t)j + upper_bound_of(old_keys, (size_t)n_old, nk[j]);
            out_keys[at]    = nk[j];
            out_perm[at]    = np[j];
        });
        std::swap(knn->d_keys, knn->d_keys_alt);
        std::swap(knn->d_permutation, knn->d_perm_alt);

        const unsigned int* order  = knn->d_permutation;
        POINT_TYPE*         stored = knn->d_stored_points;
        parallel_for<_KNN_BLOCK_SIZE_>("GATHER", len_pts, [=] HD(int s) { stored[s] = pts[order[s]]; });
        build_tree(knn);
    }

    void take_sorted_order(knn_problem* knn) {
        unsigned int* perm = knn->d_permutation;
        parallel_for<_KNN_BLOCK_SIZE_>("IDENTITY", knn->len_pts, [=] HD(int s) { perm[s] = (unsigned int)s; });
    }

    // points into Morton order; equal keys stay in input order
    static void sort_points(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {
        uint64_t*     keys = knn->d_keys;
        unsigned int* perm = knn->d_permutation;
        parallel_for<_KNN_BLOCK_SIZE_>("KEYS", len_pts, [=] HD(int i) {
            keys[i] = morton_key(pts[i]);
            perm[i] = (unsigned int)i;
        });

        parallel_sort_pairs("SORT",
                            (size_t)len_pts,
                            KEY_TOTAL_BITS,
                            knn->d_keys,
                            knn->d_permutation,
                            knn->d_keys_alt,
                            knn->d_perm_alt,
                            knn->d_sort_scratch);

        const unsigned int* order  = knn->d_permutation;
        POINT_TYPE*         stored = knn->d_stored_points;
        parallel_for<_KNN_BLOCK_SIZE_>("GATHER", len_pts, [=] HD(int s) { stored[s] = pts[order[s]]; });
    }

    // frees everything and clears the pointer
    void knn_free(knn_problem** knn) {
        free_point_arrays(*knn);
        gpu_free(*knn);
        *knn = NULL;
    }

    // more room for points; prepare fills the new arrays
    void knn_grow(knn_problem* knn, int new_pts_capacity) {
        if (new_pts_capacity <= knn->pts_capacity) return;
        free_point_arrays(knn);
        allocate_point_arrays(knn, new_pts_capacity);
        knn->len_pts = 0;
    }

} // namespace knn
