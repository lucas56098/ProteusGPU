// the bucket grid: its rings, its build and its searches on the host (included by knn.cu)

namespace knn {

    // index step and smallest distance of every ring, nearest ring first
    static void build_ring_offsets(knn_problem* knn, int N_max) {
        // more than enough for the offsets of N_max rings
        const int alloc                  = N_max * N_max * N_max * N_max;
        int*      cell_offsets           = gpu_alloc<int>(alloc);
        int*      cell_offset_axes       = gpu_alloc<int>(alloc);
        double*   cell_offset_dists      = gpu_alloc<double>(alloc);
        double*   cell_offset_dists_unit = gpu_alloc<double>(alloc);

        cell_offsets[0]           = 0;
        cell_offset_axes[0]       = pack_ring_step(0, 0, 0);
        cell_offset_dists[0]      = 0.0;
        cell_offset_dists_unit[0] = 0.0;
        int count                 = 1;

        // a point in ring r is at least r - 1 buckets away
        for (int ring = 1; ring < N_max; ring++) {
            const double du = (double)(ring - 1);
#ifdef dim_2D
            for (int j = -N_max; j <= N_max; j++) {
                for (int i = -N_max; i <= N_max; i++) {
                    if (std::max(abs(i), abs(j)) != ring) continue;
                    cell_offsets[count]           = i + j * knn->N_grid;
                    cell_offset_axes[count]       = pack_ring_step(i, j, 0);
                    cell_offset_dists_unit[count] = du * du;
                    count++;
                }
            }
#else
            for (int k = -N_max; k <= N_max; k++) {
                for (int j = -N_max; j <= N_max; j++) {
                    for (int i = -N_max; i <= N_max; i++) {
                        if (std::max(abs(i), std::max(abs(j), abs(k))) != ring) continue;
                        cell_offsets[count]           = i + j * knn->N_grid + k * knn->N_grid * knn->N_grid;
                        cell_offset_axes[count]       = pack_ring_step(i, j, k);
                        cell_offset_dists_unit[count] = du * du;
                        count++;
                    }
                }
            }
#endif
        }

        knn->d_cell_offsets           = cell_offsets;
        knn->d_cell_offset_axes       = cell_offset_axes;
        knn->d_cell_offset_dists      = cell_offset_dists;
        knn->d_cell_offset_dists_unit = cell_offset_dists_unit;
        knn->N_cell_offsets           = count;

        gpu_advise_gpu_preferred(knn->d_cell_offsets, count * sizeof(int));
        gpu_advise_gpu_preferred(knn->d_cell_offset_axes, count * sizeof(int));
        gpu_advise_gpu_preferred(knn->d_cell_offset_dists, count * sizeof(double));
    }

    // points per bucket and where each bucket starts; a bucket is a cell of the key grid,
    // so its points are one run of the sorted list
    static void build_grid(knn_problem* knn) {
        const int n = knn->len_pts;
        gpu_memset(knn->d_counters, 0, (size_t)knn->Npow * sizeof(int));
        gpu_memset(knn->d_ptrs, 0, (size_t)knn->Npow * sizeof(int));

        const knn_problem* kp       = knn;
        const POINT_TYPE*  pts      = knn->d_stored_points;
        int*               counters = knn->d_counters;
        int*               ptrs     = knn->d_ptrs;

        parallel_for<_KNN_BLOCK_SIZE_>(
            "COUNT", n, [=] HD(int s) { portable_atomicAdd(counters + cell_from_point(kp, pts[s]), 1); });

        parallel_for<_KNN_BLOCK_SIZE_>("PTRS", n, [=] HD(int s) {
            const int cell = cell_from_point(kp, pts[s]);
            if (s == 0 || cell_from_point(kp, pts[s - 1]) != cell) ptrs[cell] = s;
        });
    }

    // nearest max_k points within the reach, sorted, read straight from the grid
    static std::vector<std::pair<double, int>> grid_nearest_on_host(const knn_problem* knn, int sid, int max_k) {
        const POINT_TYPE* pts       = knn->d_stored_points;
        const POINT_TYPE  seed_pos  = pts[sid];
        const int         seed_cell = cell_from_point(knn, seed_pos);
        int               ix, iy, iz;
        bucket_coords(seed_cell, knn->N_grid, &ix, &iy, &iz);

        std::vector<std::pair<double, int>> candidates;
        candidates.reserve(max_k * 2);

        bool stopped_early = false;
        for (int ring = 0; ring < knn->N_cell_offsets; ring++) {
            const double ring_dist = knn->d_cell_offset_dists[ring];

            // once per distance: keep the max_k first, stop if the ring is strictly farther than the last
            // of them (a point at the same distance with a lower index still counts)
            if ((ring == 0 || ring_dist != knn->d_cell_offset_dists[ring - 1]) && (int)candidates.size() >= max_k) {
                std::nth_element(candidates.begin(), candidates.begin() + max_k - 1, candidates.end());
                candidates.resize(max_k);
                if (ring_dist > candidates[max_k - 1].first) {
                    stopped_early = true;
                    break;
                }
            }

            if (!ring_step_in_grid(knn->d_cell_offset_axes[ring], ix, iy, iz, knn->N_grid)) continue;
            const int cell = seed_cell + knn->d_cell_offsets[ring];

            const int cell_base  = knn->d_ptrs[cell];
            const int cell_count = knn->d_counters[cell];
            for (int i = 0; i < cell_count; i++) {
                const int other = cell_base + i;
                if (other == sid) continue;
                candidates.push_back({dist2_point(seed_pos, pts[other]), other});
            }
        }

        if ((int)candidates.size() > max_k) {
            std::nth_element(candidates.begin(), candidates.begin() + max_k - 1, candidates.end());
            candidates.resize(max_k);
        }
        std::sort(candidates.begin(), candidates.end());

        // all rings walked: the list is complete only as far as they reach
        if (!stopped_early) {
            const auto past =
                std::lower_bound(candidates.begin(), candidates.end(), std::make_pair(knn->reach2, INT_MIN));
            candidates.erase(past, candidates.end());
        }
        return candidates;
    }

    // every point with distance^2 <= r2 from p
    static void grid_points_within_on_host(const knn_problem* knn, POINT_TYPE p, double r2, std::vector<int>* out) {
        out->clear();
        const POINT_TYPE* pts = knn->d_stored_points;

        // farther than the rings reach: every point
        if (r2 >= knn->reach2) {
            for (int s = 0; s < knn->len_pts; s++) {
                if (dist2_point(p, pts[s]) <= r2) out->push_back(s);
            }
            return;
        }

        const int center = cell_from_point(knn, p);
        int       cx, cy, cz;
        bucket_coords(center, knn->N_grid, &cx, &cy, &cz);
        for (int ring = 0; ring < knn->N_cell_offsets; ring++) {
            if (knn->d_cell_offset_dists[ring] > r2) break;
            if (!ring_step_in_grid(knn->d_cell_offset_axes[ring], cx, cy, cz, knn->N_grid)) continue;
            const int cell  = center + knn->d_cell_offsets[ring];
            const int base  = knn->d_ptrs[cell];
            const int count = knn->d_counters[cell];
            for (int i = 0; i < count; i++) {
                if (dist2_point(p, pts[base + i]) <= r2) out->push_back(base + i);
            }
        }
    }

} // namespace knn
