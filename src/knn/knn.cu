// implements the neighbour search (knn.h)

#include "../global/globals.h"
#include "../global/structs.h"
#include "../profiler/profiler.h"
#include "knn.h"
#include <algorithm>
#include <cstring>
#include <iostream>

#include "tree.cu"

namespace knn {

    static void allocate_point_arrays(knn_problem* knn, int capacity);
    static void free_point_arrays(knn_problem* knn);
    static void sort_points(knn_problem* knn, const POINT_TYPE* pts, int len_pts);

    // allocates everything for the whole run
    knn_problem* init_once(int n_hydro) {

        // room for the local cells after growth, their periodic ghosts and the MPI ghosts
        double ghost_frac  = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        int    n_grow      = proteus_mpi::max_n_local(n_hydro);
        int    max_n_total = (int)(n_grow + 2.0 * ghost_frac * n_grow) + 1 + proteus_mpi::n_mpi_capacity;

        knn_problem* knn = gpu_alloc<knn_problem>(1);
        std::memset(knn, 0, sizeof(knn_problem));
        allocate_point_arrays(knn, max_n_total);
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
    }

} // namespace knn
