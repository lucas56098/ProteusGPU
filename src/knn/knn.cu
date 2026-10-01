// implements the neighbour search (knn.h)

#include "../global/globals.h"
#include "../global/structs.h"
#include "../profiler/profiler.h"
#include "knn.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <iostream>

#include "grid.cu"
#include "tree.cu"

namespace knn {

    namespace {
        // key grid bounds of the points of one build
        struct KeyBox {
            int lo[3];
            int hi[3];
        };
    } // namespace

    static void allocate_point_arrays(knn_problem* knn, int capacity);
    static void free_point_arrays(knn_problem* knn);
    static void sort_points(knn_problem* knn, const POINT_TYPE* pts, int len_pts);
    static void choose_buckets(knn_problem* knn);
    static void cross_check(const knn_problem* knn);

    // allocates everything for the whole run
    knn_problem* init_once(int n_hydro, int N_grid_restored) {

        // room for the local cells after growth, their periodic ghosts and the MPI ghosts
        double ghost_frac  = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        int    n_grow      = proteus_mpi::max_n_local(n_hydro);
        int    max_n_total = (int)(n_grow + 2.0 * ghost_frac * n_grow) + 1 + proteus_mpi::n_mpi_capacity;

        knn_problem* knn = gpu_alloc<knn_problem>(1);
        std::memset(knn, 0, sizeof(knn_problem));

        // about 3 points per bucket if all of them are in use; a power of two bucket size leaves between half
        // and all of them per axis in use, hence sqrt(2) more per axis
        // a restart keeps the old size, another size would change the reach
        knn->N_grid = (N_grid_restored > 0)
                          ? N_grid_restored
                          : std::max(1, (int)round(sqrt(2.0) * pow(max_n_total / 3.1, 1.0 / (double)DIMENSION)));
        knn->Npow   = (int)pow(knn->N_grid, DIMENSION);

        // N_max is the number of rings the grid search walks, and the smallest grid it takes
        const int N_max = 16;
        if (BUILD_GRID && knn->N_grid < N_max) {
            proteus_mpi::exit_failure(
                "KNN: the grid needs at least %d buckets per axis, this rank has %d. Too few cells.\n",
                N_max,
                knn->N_grid);
        }
        knn->N_rings = N_max - 1;

        if (BUILD_GRID) {
            build_ring_offsets(knn, N_max);
            knn->d_counters = gpu_calloc<int>(knn->Npow);
            knn->d_ptrs     = gpu_calloc<int>(knn->Npow);
            gpu_advise_gpu_preferred(knn->d_counters, knn->Npow * sizeof(int));
            gpu_advise_gpu_preferred(knn->d_ptrs, knn->Npow * sizeof(int));
        }

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

        if (BUILD_TREE) {
            knn->d_nodes  = gpu_calloc<TreeNode>(capacity);
            knn->d_parent = gpu_calloc<int>(2 * (size_t)capacity);
            knn->d_visits = gpu_calloc<unsigned int>(capacity);
            gpu_advise_gpu_preferred(knn->d_nodes, capacity * sizeof(TreeNode));
        }
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

    // sorts len_pts points and builds the search structure, called once per mesh build
    void prepare(knn_problem* knn, const POINT_TYPE* pts, int len_pts) {

        if (len_pts > knn->pts_capacity) {
            proteus_mpi::exit_failure(
                "KNN: Error! point count %d exceeds pre-allocated capacity %d. Increase ghost headroom.\n",
                len_pts,
                knn->pts_capacity);
        }

        knn->len_pts = len_pts;
        sort_points(knn, pts, len_pts);
        if (BUILD_GRID) {
            choose_buckets(knn);
            build_grid(knn);
        }
        if (BUILD_TREE) build_tree(knn);
        if (CROSS_CHECK) cross_check(knn);
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

    // the finest power of two bucket size whose buckets over the points fit into the grid arrays
    static void choose_buckets(knn_problem* knn) {
        const POINT_TYPE* pts = knn->d_stored_points;

        KeyBox none;
        for (int a = 0; a < 3; a++) {
            none.lo[a] = INT_MAX;
            none.hi[a] = -1;
        }
        const KeyBox box = parallel_reduce<_KNN_BLOCK_SIZE_, KeyBox>(
            "KEY_BOX",
            (size_t)knn->len_pts,
            none,
            [] HD(KeyBox a, KeyBox b) {
                KeyBox r;
                for (int c = 0; c < 3; c++) {
                    r.lo[c] = imin(a.lo[c], b.lo[c]);
                    r.hi[c] = imax(a.hi[c], b.hi[c]);
                }
                return r;
            },
            [=] HD(size_t s) {
                KeyBox r;
                r.lo[0] = r.hi[0] = (int)key_coord(pts[s].x);
                r.lo[1] = r.hi[1] = (int)key_coord(pts[s].y);
#ifdef dim_3D
                r.lo[2] = r.hi[2] = (int)key_coord(pts[s].z);
#else
                r.lo[2] = r.hi[2] = 0;
#endif
                return r;
            });

        int shift = 0;
        if (knn->len_pts > 0) {
            for (; shift < KEY_BITS; shift++) {
                bool fits = true;
                for (int a = 0; a < DIMENSION; a++) {
                    if ((box.hi[a] >> shift) - (box.lo[a] >> shift) + 1 > knn->N_grid) fits = false;
                }
                if (fits) break;
            }
        }
        knn->shift = shift;
        for (int a = 0; a < 3; a++) {
            knn->bucket_lo[a] = (a < DIMENSION && knn->len_pts > 0) ? (box.lo[a] >> shift) : 0;
        }

        // a power of two times the key cell, so the ring distances are exact
        knn->cell_size   = ldexp(1.0, shift - (KEY_BITS - 1));
        const double cs2 = knn->cell_size * knn->cell_size;
        knn->reach2      = (double)knn->N_rings * knn->N_rings * cs2;
        if (BUILD_GRID) {
            for (int m = 0; m < knn->N_cell_offsets; m++) {
                knn->d_cell_offset_dists[m] = knn->d_cell_offset_dists_unit[m] * cs2;
            }
        }
    }

    std::vector<std::pair<double, int>> nearest_on_host(const knn_problem* knn, int sid, int max_k) {
        return USE_TREE ? tree_nearest_on_host(knn, sid, max_k) : grid_nearest_on_host(knn, sid, max_k);
    }

    void points_within_on_host(const knn_problem* knn, POINT_TYPE p, double r2, std::vector<int>* out) {
        if (USE_TREE) {
            tree_points_within_on_host(knn, p, r2, out);
        } else {
            grid_points_within_on_host(knn, p, r2, out);
        }
    }

    // the grid keeps the point in its old bucket, the tree widens the boxes above it
    void point_moved(knn_problem* knn, int sid) {
        if (BUILD_TREE) tree_point_moved(knn, sid);
    }

    // frees everything and clears the pointer
    void knn_free(knn_problem** knn) {
        gpu_free((*knn)->d_cell_offsets);
        gpu_free((*knn)->d_cell_offset_axes);
        gpu_free((*knn)->d_cell_offset_dists);
        gpu_free((*knn)->d_cell_offset_dists_unit);
        gpu_free((*knn)->d_counters);
        gpu_free((*knn)->d_ptrs);
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

    // ============================================================================
    // cross check of the two searches
    // ============================================================================

    // first entry where the tree disagrees with the grid, -1 if none; past the grid's reach the grid has
    // nothing and the tree anything at least that far
    template <int K>
    HD int first_query_mismatch(int s, const knn_problem* knn, unsigned int* from_grid, unsigned int* from_tree) {
        grid_knn_for_point<K>(s, knn, from_grid);
        tree_knn_for_point<K>(s, knn, from_tree);
        const POINT_TYPE p = knn->d_stored_points[s];
        for (int i = 0; i < K; i++) {
            if (from_grid[i] != (unsigned int)s) {
                if (from_grid[i] != from_tree[i]) return i;
            } else if (from_tree[i] != (unsigned int)s &&
                       dist2_point(p, knn->d_stored_points[from_tree[i]]) < knn->reach2) {
                return i;
            }
        }
        return -1;
    }

    // points whose K nearest differ between grid and tree
    template <int K> static int count_query_mismatches(const knn_problem* knn) {
        return parallel_reduce_sum<_KNN_BLOCK_SIZE_, int>("CHECK_QUERY", (size_t)knn->len_pts, [=] HD(size_t s) {
            unsigned int from_grid[K];
            unsigned int from_tree[K];
            return (first_query_mismatch<K>((int)s, knn, from_grid, from_tree) >= 0) ? 1 : 0;
        });
    }

    // names the first point where the two differ and stops the run
    template <int K> static void report_query_mismatch(const knn_problem* knn) {
        for (int s = 0; s < knn->len_pts; s++) {
            unsigned int from_grid[K];
            unsigned int from_tree[K];
            const int    i = first_query_mismatch<K>(s, knn, from_grid, from_tree);
            if (i >= 0) {
                const POINT_TYPE p = knn->d_stored_points[s];
                proteus_mpi::exit_failure(
                    "KNN: cross check failed, K = %d, point %d, entry %d: grid %u (d^2 %.17g), tree %u (d^2 %.17g)\n",
                    K,
                    s,
                    i,
                    from_grid[i],
                    dist2_point(p, knn->d_stored_points[from_grid[i]]),
                    from_tree[i],
                    dist2_point(p, knn->d_stored_points[from_tree[i]]));
            }
        }
    }

    // both searches on the same points must give the same answer, entry by entry, as far as the grid reaches
    static void cross_check(const knn_problem* knn) {
        PROFILE("CROSS_CHECK");
        const int n = knn->len_pts;

        // every bucket is one run of the sorted list
        const int* counters = knn->d_counters;
        const int  runs     = parallel_reduce_sum<_KNN_BLOCK_SIZE_, int>("CHECK_RUNS", (size_t)n, [=] HD(size_t s) {
            return (s == 0 ||
                    cell_from_point(knn, knn->d_stored_points[s - 1]) != cell_from_point(knn, knn->d_stored_points[s]))
                            ? 1
                            : 0;
        });
        const int  used     = parallel_reduce_sum<_KNN_BLOCK_SIZE_, int>(
            "CHECK_BUCKETS", (size_t)knn->Npow, [=] HD(size_t b) { return (counters[b] > 0) ? 1 : 0; });
        if (runs != used) {
            proteus_mpi::exit_failure(
                "KNN: cross check failed, %d runs of the sorted list but %d used buckets\n", runs, used);
        }

        if (count_query_mismatches<_FAST_K_>(knn) > 0) report_query_mismatch<_FAST_K_>(knn);
        if (count_query_mismatches<_K_>(knn) > 0) report_query_mismatch<_K_>(knn);

        // the host searches of the fallback, on a sample
        const int step = std::max(1, n / 64);
        for (int s = 0; s < n; s += step) {
            auto from_tree_near = tree_nearest_on_host(knn, s, 2048);
            while (!from_tree_near.empty() && !(from_tree_near.back().first < knn->reach2))
                from_tree_near.pop_back();
            if (grid_nearest_on_host(knn, s, 2048) != from_tree_near) {
                proteus_mpi::exit_failure("KNN: cross check failed, nearest on host differ for point %d\n", s);
            }
            std::vector<int> from_grid, from_tree;
            const double     r2 = 9.0 * knn->cell_size * knn->cell_size;
            grid_points_within_on_host(knn, knn->d_stored_points[s], r2, &from_grid);
            tree_points_within_on_host(knn, knn->d_stored_points[s], r2, &from_tree);
            std::sort(from_grid.begin(), from_grid.end());
            std::sort(from_tree.begin(), from_tree.end());
            if (from_grid != from_tree) {
                proteus_mpi::exit_failure("KNN: cross check failed, points within differ for point %d\n", s);
            }
        }
        logging::root() << "KNN: grid and tree searches agree." << std::endl;
    }

} // namespace knn
