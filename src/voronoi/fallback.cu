
// CPU rebuild of the cells no GPU tier could finish (internal.h)

namespace voronoi {

    namespace {
        enum class FallbackOutcome { ok_unchanged, ok_perturbed, failed };

        // the points that stand for a cell: itself and its ghost copies
        struct CellSids {
            std::unordered_map<int, std::vector<int>> per_cell;
            const VMesh*                              mesh = nullptr;

            void ensure_built_for(int k) {
                if (per_cell.count(k)) return;
                std::vector<int>& sids    = per_cell[k];
                const int         n_seeds = (int)mesh->n_seeds;
                for (int sid = 0; sid < n_seeds; sid++) {
                    if ((int)mesh->sid_to_neighbor[sid] == k) sids.push_back(sid);
                }
            }

            int        size_for(int k) const { return (int)per_cell.at(k).size(); }
            const int* begin_for(int k) const { return per_cell.at(k).data(); }
        };
    } // namespace

    static int             count_failed_and_prefetch_status(VMesh* mesh);
    static CellSids        build_cell_sids_for(const VMesh* mesh, const std::vector<int>& target_ks);
    static FallbackOutcome rebuild_cell_with_perturb_retry(
        VMesh* mesh, int k, double* d_stored_points, CellSids& cell_sids, double dt, Status& last_status_out);
    static std::vector<std::pair<double, int>>
    gather_nearby_seeds_sorted(double* d_stored_points, int seed_id, const knn_problem* knn, int max_candidates);
    static std::vector<std::pair<double, int>>
    sort_neighbours_by_distance(double* d_stored_points, int seed_id, int n_seeds);
    template <typename CellT>
    static bool    try_build_cell_from_neighbours_as(VMesh*                                     mesh,
                                                     int                                        k,
                                                     int                                        seed_id,
                                                     double*                                    d_stored_points,
                                                     const std::vector<std::pair<double, int>>& sorted,
                                                     bool                                       require_security,
                                                     Status&                                    last_status_out);
    static bool    rebuild_on_wide_tier(VMesh*                                     mesh,
                                        int                                        k,
                                        int                                        seed_id,
                                        double*                                    d_stored_points,
                                        const std::vector<std::pair<double, int>>& bounded,
                                        Status&                                    last_status_out);
    static double3 compute_perturbation_delta(int seed_id, int attempt, double scale);
    static void    apply_perturbation(
           double* d_stored_points, double3 delta, const int* sids, size_t n_sids, const double4_t* orig_positions);
    static void
    rewind_perturbation(double* d_stored_points, const int* sids, size_t n_sids, const double4_t* orig_positions);
#ifdef MOVING_MESH
    static void apply_vmesh_perturbation_correction(VMesh* mesh, int k, double3 delta, double dt);
#endif
    static void                           retire_face_range(VMesh* mesh, uint64_t first, uint64_t count);
    template <typename CellT> static void write_cell_to_mesh(VMesh* mesh, int k, const CellT& cell);
    static void                           reclaim_appended_slice(VMesh*              mesh,
                                                                 int                 k,
                                                                 uint64_t            fp_old,
                                                                 uint64_t            fc_old,
                                                                 unsigned long long* face_offset,
                                                                 unsigned long long  off_before);
    static std::vector<int>               collect_unique_neighbors(const VMesh* mesh, const std::vector<int>& sources);

    struct CascadeResult {
        int rebuilt = 0;
        int rounds  = 0;
    };
    static CascadeResult cascade_rebuild_affected(VMesh*                  mesh,
                                                  double*                 d_stored_points,
                                                  CellSids&               cell_sids,
                                                  const std::vector<int>& initial_affected,
                                                  double                  dt,
                                                  std::vector<int>*       newly_perturbed_out);
    static void          run_symmetry_pass(VMesh*                  mesh,
                                           double*                 d_stored_points,
                                           CellSids&               cell_sids,
                                           const std::vector<int>& initial_perturbed,
                                           double                  dt,
                                           std::vector<int>*       cascade_perturbed_out);
    static double        compute_max_security_d2(const VMesh* mesh);

    static int s_wide_tier_rebuilds = 0;

    static int     s_uncertified_rebuilds = 0;
    static double3 s_first_uncertified    = {0.0, 0.0, 0.0}; // its seed, for the message

    // a cell reaching past the points this rank has may miss a neighbour, so the run stops
    static void stop_if_uncertified(const char* who) {
        if (s_uncertified_rebuilds == 0) return;
        proteus_mpi::exit_failure("[rank %d] VORONOI: %d cell(s) built by the %s reach past the points this rank "
                                  "has (ghost band on one rank, halo under MPI), the first at (%g, %g, %g). They may "
                                  "miss a neighbour. Aborting.\n",
                                  proteus_mpi::rank(),
                                  s_uncertified_rebuilds,
                                  who,
                                  s_first_uncertified.x,
                                  s_first_uncertified.y,
                                  s_first_uncertified.z);
    }

    // rebuilds every failed cell, returns how many needed a moved seed
    int cpu_fallback_failed_cells(VMesh* mesh, int* num_failed_out, double dt, std::vector<int>* perturbed_ks_out) {
        Status* stat            = mesh->cell_status;
        double* d_stored_points = (double*)mesh->knn->d_stored_points;

        const int num_failed = count_failed_and_prefetch_status(mesh);
        if (num_failed_out) *num_failed_out = num_failed;
        if (num_failed == 0) return 0;

        const int n_hydro      = (int)mesh->n_hydro;
        s_wide_tier_rebuilds   = 0;
        s_uncertified_rebuilds = 0;

        std::vector<int> failed_ks;
        failed_ks.reserve(num_failed);
        for (int k = 0; k < n_hydro; k++) {
            if (stat[k] == success) continue;
            const Status original = stat[k];
            // only these statuses can be repaired
            if (original != security_radius_not_reached && original != needs_exact_predicates &&
                original != inconsistent_boundary && original != vertex_overflow && original != triangle_overflow &&
                original != security_radius_beyond_data) {
                proteus_mpi::exit_failure(
                    "VORONOI: cell %d failed with unrecoverable status: %d\n", (int)k, (int)original);
            }
            failed_ks.push_back(k);
            mesh->face_counts[k] = 0;
        }

        CellSids cell_sids = build_cell_sids_for(mesh, failed_ks);

        std::vector<int> perturbed_ks;
        for (int k : failed_ks) {
            Status last_status = success;
            switch (rebuild_cell_with_perturb_retry(mesh, k, d_stored_points, cell_sids, dt, last_status)) {
            case FallbackOutcome::ok_unchanged:
                break;
            case FallbackOutcome::ok_perturbed:
                perturbed_ks.push_back(k);
                if (perturbed_ks_out) perturbed_ks_out->push_back(k);
                break;
            case FallbackOutcome::failed:
                proteus_mpi::exit_failure("VORONOI: cell %d all fallback attempts FAILED, aborting.\n", (int)k);
            }
        }

        // a moved seed also changes the cells around it
        if (!perturbed_ks.empty()) {
            run_symmetry_pass(mesh, d_stored_points, cell_sids, perturbed_ks, dt, perturbed_ks_out);
        }

        if (s_wide_tier_rebuilds > 0) {
            std::cerr << "VORONOI: " << s_wide_tier_rebuilds << " cell(s) rebuilt on the wide tier (" << _BIG_MAX_P_
                      << "/" << _BIG_MAX_T_ << " slots)." << std::endl;
        }
        stop_if_uncertified("CPU fallback");
        return (int)perturbed_ks.size();
    }

    // count, and bring the status array to the host
    static int count_failed_and_prefetch_status(VMesh* mesh) {
        const int n_failed = count_failed_cells(mesh);

#ifndef CPU_DEBUG
        if (n_failed > 0) gpu_prefetch_to_cpu(mesh->cell_status, mesh->n_hydro * sizeof(Status));
#endif
        return n_failed;
    }

    // one walk over the point list for all wanted cells
    static CellSids build_cell_sids_for(const VMesh* mesh, const std::vector<int>& target_ks) {
        CellSids cs;
        cs.mesh = mesh;
        if (target_ks.empty()) return cs;

        std::unordered_set<int> target_set(target_ks.begin(), target_ks.end());
        for (int k : target_ks)
            cs.per_cell[k];

        gpu_prefetch_to_cpu(mesh->sid_to_neighbor, mesh->n_seeds * sizeof(unsigned int));

        const int n_seeds = (int)mesh->n_seeds;
        for (int sid = 0; sid < n_seeds; sid++) {
            const int k = (int)mesh->sid_to_neighbor[sid];
            if (target_set.count(k)) cs.per_cell[k].push_back(sid);
        }
        return cs;
    }

    static constexpr int FALLBACK_BOUNDED_K = 2048; // candidates of the bounded search

    static bool is_overflow(Status s) {
        return s == vertex_overflow || s == triangle_overflow;
    }

    // build with the seed moved, the step ten times larger each try; attempt 0 is the seed as it is
    static FallbackOutcome run_perturb_ladder(VMesh*                                     mesh,
                                              int                                        k,
                                              int                                        seed_id,
                                              double*                                    d_stored_points,
                                              const std::vector<std::pair<double, int>>& sorted,
                                              const int*                                 sids,
                                              size_t                                     n_sids,
                                              const double4_t*                           orig_positions,
                                              double                                     dt,
                                              bool                                       require_security,
                                              int                                        first_attempt,
                                              Status&                                    last_status_out,
                                              bool&                                      overflowed) {
        constexpr int max_perturb = 12;
        double        scale       = 1e-13;
        for (int attempt = 0; attempt < first_attempt; attempt++)
            scale *= 10.0;
        for (int attempt = first_attempt; attempt <= max_perturb; attempt++) {
            double3 delta = {0.0, 0.0, 0.0};
            if (attempt > 0) {
                delta = compute_perturbation_delta(seed_id, attempt, scale);
                apply_perturbation(d_stored_points, delta, sids, n_sids, orig_positions);
            }

            Status     attempt_status = success;
            const bool ok             = try_build_cell_from_neighbours_as<ConvexCell>(
                mesh, k, seed_id, d_stored_points, sorted, require_security, attempt_status);
            if (ok) {
                if (attempt == 0) return FallbackOutcome::ok_unchanged;
#ifdef MOVING_MESH
                apply_vmesh_perturbation_correction(mesh, k, delta, dt);
#else
                (void)dt;
#endif
                return FallbackOutcome::ok_perturbed;
            }

            last_status_out = attempt_status;
            if (is_overflow(attempt_status)) overflowed = true;
            if (attempt > 0) rewind_perturbation(d_stored_points, sids, n_sids, orig_positions);
            scale *= 10.0;
        }
        return FallbackOutcome::failed;
    }

    // one cell through the ladder: every unperturbed build first, a moved seed only after all of them
    static FallbackOutcome rebuild_cell_with_perturb_retry(
        VMesh* mesh, int k, double* d_stored_points, CellSids& cell_sids, double dt, Status& last_status_out) {
        const int seed_id = (int)mesh->real_sorted_ids[k];

        const auto bounded = gather_nearby_seeds_sorted(d_stored_points, seed_id, mesh->knn, FALLBACK_BOUNDED_K);

        // out of slots: only the wide tier can help
        if (is_overflow(mesh->cell_status[k])) {
            return rebuild_on_wide_tier(mesh, k, seed_id, d_stored_points, bounded, last_status_out)
                       ? FallbackOutcome::ok_unchanged
                       : FallbackOutcome::failed;
        }

        // the near points
        Status st = success;
        if (try_build_cell_from_neighbours_as<ConvexCell>(mesh, k, seed_id, d_stored_points, bounded, true, st)) {
            return FallbackOutcome::ok_unchanged;
        }
        bool overflowed = is_overflow(st);

        // too few of them: all points, a kick does not bring in the ones that are missing
        std::vector<std::pair<double, int>> full;
        if (st == security_radius_not_reached) {
            full = sort_neighbours_by_distance(d_stored_points, seed_id, (int)mesh->n_seeds);
            if (try_build_cell_from_neighbours_as<ConvexCell>(mesh, k, seed_id, d_stored_points, full, false, st)) {
                return FallbackOutcome::ok_unchanged;
            }
            overflowed = overflowed || is_overflow(st);
        }
        const bool full_tried = !full.empty();

        // out of slots: more slots before a kick
        bool wide_tried = false;
        if (overflowed) {
            if (rebuild_on_wide_tier(mesh, k, seed_id, d_stored_points, bounded, last_status_out)) {
                return FallbackOutcome::ok_unchanged;
            }
            wide_tried = true;
        }

        // what is left is degenerate, a small move of the seed can fix that
        cell_sids.ensure_built_for(k);
        const int*   sids   = cell_sids.begin_for(k);
        const size_t n_sids = (size_t)cell_sids.size_for(k);

        std::vector<double4_t> orig_positions(n_sids);
        for (size_t i = 0; i < n_sids; i++)
            orig_positions[i] = point_from_ptr(d_stored_points + DIMENSION * sids[i]);

        FallbackOutcome outcome = run_perturb_ladder(mesh,
                                                     k,
                                                     seed_id,
                                                     d_stored_points,
                                                     bounded,
                                                     sids,
                                                     n_sids,
                                                     orig_positions.data(),
                                                     dt,
                                                     true,
                                                     1,
                                                     last_status_out,
                                                     overflowed);
        if (outcome != FallbackOutcome::failed) return outcome;

        if (!full_tried) full = sort_neighbours_by_distance(d_stored_points, seed_id, (int)mesh->n_seeds);
        outcome = run_perturb_ladder(mesh,
                                     k,
                                     seed_id,
                                     d_stored_points,
                                     full,
                                     sids,
                                     n_sids,
                                     orig_positions.data(),
                                     dt,
                                     false,
                                     full_tried ? 1 : 0,
                                     last_status_out,
                                     overflowed);
        if (outcome != FallbackOutcome::failed) return outcome;

        // more slots only help if some attempt ran out of them
        if (!overflowed || wide_tried) return FallbackOutcome::failed;
        return rebuild_on_wide_tier(mesh, k, seed_id, d_stored_points, bounded, last_status_out)
                   ? FallbackOutcome::ok_unchanged
                   : FallbackOutcome::failed;
    }

    static std::vector<std::pair<double, int>>
    // nearest max_candidates points, sorted, read straight from the grid
    gather_nearby_seeds_sorted(double* d_stored_points, int seed_id, const knn_problem* knn, int max_candidates) {
        const double4_t seed_pos = point_from_ptr(d_stored_points + DIMENSION * seed_id);
        const int       seed_cell =
            knn::cell_from_point(knn->N_grid, knn->grid_lo, knn->inv_cell_size, knn->d_stored_points[seed_id]);
        int ix, iy, iz;
        knn::bucket_coords(seed_cell, knn->N_grid, &ix, &iy, &iz);

        std::vector<std::pair<double, int>> candidates;
        candidates.reserve(max_candidates * 2);

        double kth_dist      = DBL_MAX;
        bool   stopped_early = false;
        for (int ring = 0; ring < knn->N_cell_offsets; ring++) {
            if ((int)candidates.size() >= max_candidates && knn->d_cell_offset_dists[ring] >= kth_dist) {
                stopped_early = true;
                break;
            }

            if (!knn::ring_step_in_grid(knn->d_cell_offset_axes[ring], ix, iy, iz, knn->N_grid)) continue;
            const int cell = seed_cell + knn->d_cell_offsets[ring];

            const int cell_base  = knn->d_ptrs[cell];
            const int cell_count = knn->d_counters[cell];
            for (int i = 0; i < cell_count; i++) {
                const int sid = cell_base + i;
                if (sid == seed_id) continue;
                const double4_t other = point_from_ptr(d_stored_points + DIMENSION * sid);
                const double    dx    = other.x - seed_pos.x;
                const double    dy    = other.y - seed_pos.y;
                const double    dz    = other.z - seed_pos.z;
                candidates.push_back({dx * dx + dy * dy + dz * dz, sid});
            }

            // distance of the last candidate, the ring loop stops beyond it
            if ((int)candidates.size() >= max_candidates) {
                std::nth_element(candidates.begin(), candidates.begin() + max_candidates - 1, candidates.end());
                kth_dist = candidates[max_candidates - 1].first;
            }
        }

        if ((int)candidates.size() > max_candidates) candidates.resize(max_candidates);
        std::sort(candidates.begin(), candidates.end());

        // all rings walked: the list is complete only as far as they reach
        if (!stopped_early) {
            const auto past =
                std::lower_bound(candidates.begin(), candidates.end(), std::make_pair(knn->reach2, INT_MIN));
            candidates.erase(past, candidates.end());
        }
        return candidates;
    }

    static std::vector<std::pair<double, int>>
    // every point on the rank, sorted by distance
    sort_neighbours_by_distance(double* d_stored_points, int seed_id, int n_seeds) {
        const double4_t                     seed_pos = point_from_ptr(d_stored_points + DIMENSION * seed_id);
        std::vector<std::pair<double, int>> dists;
        dists.reserve(n_seeds - 1);
        for (int j = 0; j < n_seeds; j++) {
            if (j == seed_id) continue;
            const double4_t other = point_from_ptr(d_stored_points + DIMENSION * j);
            const double    dx    = other.x - seed_pos.x;
            const double    dy    = other.y - seed_pos.y;
            const double    dz    = other.z - seed_pos.z;
            dists.push_back({dx * dx + dy * dy + dz * dz, j});
        }
        std::sort(dists.begin(), dists.end());
        return dists;
    }

    // clips the cell with the sorted points and writes it if it came out complete
    template <typename CellT>
    static bool try_build_cell_from_neighbours_as(VMesh*                                     mesh,
                                                  int                                        k,
                                                  int                                        seed_id,
                                                  double*                                    d_stored_points,
                                                  const std::vector<std::pair<double, int>>& sorted,
                                                  bool                                       require_security,
                                                  Status&                                    last_status_out) {
        Status status = success;
        CellT  cell(seed_id, d_stored_points, &status, mesh->buff);

        bool security_reached = false;
        for (size_t di = 0; di < sorted.size(); di++) {
            const int j = sorted[di].second;
            cell.clip_by_plane(j);
            if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * j))) {
                security_reached = true;
                break;
            }
            if (status != success) break;
        }
        if (status != success) {
            last_status_out = status;
            return false;
        }
        // the last resort takes the cell without this check
        if (require_security && !security_reached) {
            last_status_out = security_radius_not_reached;
            return false;
        }

        double r2_num, r2_denom;
        cell.max_vertex_r2_ratio(&r2_num, &r2_denom);
#ifdef USE_MPI
        store_security_d2(mesh, (uint64_t)k, r2_num, r2_denom);
#endif
        if (!cell_certified_within_data(cell.voro_seed, r2_num, r2_denom, mesh->data_lo, mesh->data_hi, mesh->buff)) {
            if (s_uncertified_rebuilds == 0)
                s_first_uncertified = {cell.voro_seed.x, cell.voro_seed.y, cell.voro_seed.z};
            s_uncertified_rebuilds++;
        }

        write_cell_to_mesh(mesh, k, cell);
        mesh->cell_status[k] = success;
        return true;
    }

    // the same with the big capacities
    static bool rebuild_on_wide_tier(VMesh*                                     mesh,
                                     int                                        k,
                                     int                                        seed_id,
                                     double*                                    d_stored_points,
                                     const std::vector<std::pair<double, int>>& bounded,
                                     Status&                                    last_status_out) {
        Status st = success;
        if (try_build_cell_from_neighbours_as<BigConvexCell>(mesh, k, seed_id, d_stored_points, bounded, true, st)) {
            s_wide_tier_rebuilds++;
            return true;
        }

        const auto full = sort_neighbours_by_distance(d_stored_points, seed_id, (int)mesh->n_seeds);
        if (try_build_cell_from_neighbours_as<BigConvexCell>(mesh, k, seed_id, d_stored_points, full, false, st)) {
            s_wide_tier_rebuilds++;
            return true;
        }

        last_status_out = st;
        return false;
    }

    // offset from a hash of seed and attempt, so a rebuild gives the same one
    static double3 compute_perturbation_delta(int seed_id, int attempt, double scale) {
        unsigned int hash = (unsigned int)(seed_id * 2654435761u + attempt * 40503u);
        hash              = hash * 1103515245u + 12345u;
        const double dx   = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
        hash              = hash * 1103515245u + 12345u;
        const double dy   = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
        double       dz   = 0.0;
#ifdef dim_3D
        hash = hash * 1103515245u + 12345u;
        dz   = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
#endif
        return {dx, dy, dz};
    }

    // moves every copy of the seed, ghosts included
    static void apply_perturbation(
        double* d_stored_points, double3 delta, const int* sids, size_t n_sids, const double4_t* orig_positions) {
        for (size_t i = 0; i < n_sids; i++) {
            const int sid                        = sids[i];
            d_stored_points[DIMENSION * sid + 0] = orig_positions[i].x + delta.x;
            d_stored_points[DIMENSION * sid + 1] = orig_positions[i].y + delta.y;
#ifdef dim_3D
            d_stored_points[DIMENSION * sid + 2] = orig_positions[i].z + delta.z;
#endif
        }
    }

#ifdef MOVING_MESH
    // the seed keeps the offset, so add it to the mesh velocity, but only if it is small
    static void apply_vmesh_perturbation_correction(VMesh* mesh, int k, double3 delta, double dt) {
        if (dt <= 0.0) return;
        const double inv_dt = 1.0 / dt;
        const double d_mag  = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
        const double dv_mag = d_mag * inv_dt;

        constexpr double DV_REL     = 1e-3;
        constexpr double DV_MAX_ABS = 1e-2;
#ifdef dim_3D
        const double vm_mag = sqrt(mesh->v_mesh[k].x * mesh->v_mesh[k].x + mesh->v_mesh[k].y * mesh->v_mesh[k].y +
                                   mesh->v_mesh[k].z * mesh->v_mesh[k].z);
#else
        const double vm_mag = sqrt(mesh->v_mesh[k].x * mesh->v_mesh[k].x + mesh->v_mesh[k].y * mesh->v_mesh[k].y);
#endif
        if (dv_mag > fmin(DV_REL * vm_mag, DV_MAX_ABS)) return;

        mesh->v_mesh[k].x += delta.x * inv_dt;
        mesh->v_mesh[k].y += delta.y * inv_dt;
#ifdef dim_3D
        mesh->v_mesh[k].z += delta.z * inv_dt;
#endif
    }
#endif

    static void
    rewind_perturbation(double* d_stored_points, const int* sids, size_t n_sids, const double4_t* orig_positions) {
        for (size_t i = 0; i < n_sids; i++) {
            const int sid                        = sids[i];
            d_stored_points[DIMENSION * sid + 0] = orig_positions[i].x;
            d_stored_points[DIMENSION * sid + 1] = orig_positions[i].y;
#ifdef dim_3D
            d_stored_points[DIMENSION * sid + 2] = orig_positions[i].z;
#endif
        }
    }

    // unused slots become wall faces of zero area
    static void retire_face_range(VMesh* mesh, uint64_t first, uint64_t count) {
        for (uint64_t i = first; i < first + count; i++) {
            mesh->neighbor_cell[i] = -1;
            mesh->face_area[i]     = 0.0;
#ifdef MOVING_MESH
            for (int c = 0; c < DIMENSION - 1; c++)
                mesh->f_mid_local[i * (DIMENSION - 1) + c] = 0.0;
#endif
        }
    }

    // into the old block of faces if it fits, else a new block at the end
    template <typename CellT> static void write_cell_to_mesh(VMesh* mesh, int k, const CellT& cell) {
        const uint64_t fc_max = (uint64_t)count_cell_faces(cell);
        const uint64_t fp_old = mesh->face_ptr[k];
        const uint64_t fc_old = mesh->face_counts[k];
        if (fc_max <= fc_old) {
            const uint64_t written = extract_cell_all(cell, mesh, (uint64_t)k);
            mesh->face_counts[k]   = written;
            retire_face_range(mesh, fp_old + written, fc_old - written);
        } else {
            retire_face_range(mesh, fp_old, fc_old);
            ensure_face_capacity(mesh, mesh->num_faces + fc_max);
            mesh->face_ptr[k]      = mesh->num_faces;
            const uint64_t written = extract_cell_all(cell, mesh, (uint64_t)k);
            mesh->face_counts[k]   = written;
            mesh->num_faces += written;
        }
    }

    // moves an appended block back into the old one and gives the tail back
    static void reclaim_appended_slice(VMesh*              mesh,
                                       int                 k,
                                       uint64_t            fp_old,
                                       uint64_t            fc_old,
                                       unsigned long long* face_offset,
                                       unsigned long long  off_before) {
        const uint64_t fp_new = mesh->face_ptr[k];
        const uint64_t fc_new = mesh->face_counts[k];
        if (fc_new <= fc_old) {
            for (uint64_t i = 0; i < fc_new; i++) {
                mesh->neighbor_cell[fp_old + i] = mesh->neighbor_cell[fp_new + i];
                mesh->face_area[fp_old + i]     = mesh->face_area[fp_new + i];
#ifdef MOVING_MESH
                for (int c = 0; c < DIMENSION - 1; c++)
                    mesh->f_mid_local[(fp_old + i) * (DIMENSION - 1) + c] =
                        mesh->f_mid_local[(fp_new + i) * (DIMENSION - 1) + c];
#endif
            }
            mesh->face_ptr[k] = fp_old;
            retire_face_range(mesh, fp_old + fc_new, fc_old - fc_new);
            *face_offset = off_before;
        } else {
            retire_face_range(mesh, fp_old, fc_old);
        }
    }

    // the cells that share a face with one of these
    static std::vector<int> collect_unique_neighbors(const VMesh* mesh, const std::vector<int>& sources) {
        const int         n_hydro = (int)mesh->n_hydro;
        std::vector<bool> seen(n_hydro, false);
        std::vector<int>  result;
        for (int k : sources) {
            const uint64_t fp = mesh->face_ptr[k];
            const uint64_t fc = mesh->face_counts[k];
            for (uint64_t f = 0; f < fc; f++) {
                const int kn = mesh->neighbor_cell[fp + f];
                if (kn < 0 || kn >= n_hydro) continue;
                if (seen[kn]) continue;
                seen[kn] = true;
                result.push_back(kn);
            }
        }
        return result;
    }

    // rebuilds the affected cells, and the neighbours of those that moved again
    static CascadeResult cascade_rebuild_affected(VMesh*                  mesh,
                                                  double*                 d_stored_points,
                                                  CellSids&               cell_sids,
                                                  const std::vector<int>& initial_affected,
                                                  double                  dt,
                                                  std::vector<int>*       newly_perturbed_out) {
        constexpr int MAX_ROUNDS = 12;

        CascadeResult    result;
        std::vector<int> work_affected = initial_affected;
        // a rebuilt cell appends its faces and gives the room back when they fit in the old block
        unsigned long long face_offset = (unsigned long long)mesh->num_faces;
        int                overflow    = 0;

        while (!work_affected.empty()) {
            if (++result.rounds > MAX_ROUNDS) {
                proteus_mpi::exit_failure("VORONOI: symmetry cascade did not converge after %d rounds, aborting.\n",
                                          MAX_ROUNDS);
            }

            std::vector<int> perturbed_this_round;
            for (int kn : work_affected) {
                mesh->cell_status[kn]               = security_radius_not_reached;
                const int                seed_id    = (int)mesh->real_sorted_ids[kn];
                const uint64_t           fp_old     = mesh->face_ptr[kn];
                const uint64_t           fc_old     = mesh->face_counts[kn];
                const unsigned long long off_before = face_offset;
                compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_, uchar, VERT_TYPE>(
                    kn, seed_id, d_stored_points, mesh->knn, mesh->cell_status, mesh, &face_offset, &overflow);
                if (overflow) {
                    proteus_mpi::exit_failure("VORONOI: face overflow during symmetry rebuild — increase "
                                              "_FACE_CAPACITY_MULT_ in Config.sh.\n");
                }
                if (mesh->cell_status[kn] == success) {
                    reclaim_appended_slice(mesh, kn, fp_old, fc_old, &face_offset, off_before);
                }

                // the cell code failed here too, so take the CPU ladder
                if (mesh->cell_status[kn] != success) {
                    mesh->num_faces             = (uint64_t)face_offset;
                    Status          last_status = success;
                    FallbackOutcome outcome =
                        rebuild_cell_with_perturb_retry(mesh, kn, d_stored_points, cell_sids, dt, last_status);
                    if (outcome == FallbackOutcome::failed) {
                        proteus_mpi::exit_failure(
                            "VORONOI: symmetry rebuild for cell %d all fallback attempts FAILED.\n", (int)kn);
                    }
                    face_offset = (unsigned long long)mesh->num_faces;
                    if (outcome == FallbackOutcome::ok_perturbed) perturbed_this_round.push_back(kn);
                }
                result.rebuilt++;
            }

            if (newly_perturbed_out)
                newly_perturbed_out->insert(
                    newly_perturbed_out->end(), perturbed_this_round.begin(), perturbed_this_round.end());
            work_affected = collect_unique_neighbors(mesh, perturbed_this_round);
        }
        mesh->num_faces = (uint64_t)face_offset;
        return result;
    }

    // the cells next to a moved seed are wrong now
    static void run_symmetry_pass(VMesh*                  mesh,
                                  double*                 d_stored_points,
                                  CellSids&               cell_sids,
                                  const std::vector<int>& initial_perturbed,
                                  double                  dt,
                                  std::vector<int>*       cascade_perturbed_out) {
        const std::vector<int> affected = collect_unique_neighbors(mesh, initial_perturbed);
        const CascadeResult    result =
            cascade_rebuild_affected(mesh, d_stored_points, cell_sids, affected, dt, cascade_perturbed_out);

        std::cout << "VORONOI: " << initial_perturbed.size() << " cell(s) permanently perturbed; " << result.rebuilt
                  << " neighbour rebuild(s) over " << result.rounds << " round(s)." << std::endl;
    }

    static double compute_max_security_d2(const VMesh* mesh) {
        const double* sec = mesh->security_d2;
        return parallel_reduce<_MESH_BLOCK_SIZE_, double>(
            "MAX_SEC_D2",
            mesh->n_hydro,
            0.0,
            [] HD(double a, double b) { return a > b ? a : b; },
            [=] HD(size_t k) { return sec[k]; });
    }

    // cells that had the ghost inside their security radius
    static void collect_affected_by_moved_ghost(
        const VMesh* mesh, double4_t g_old, double4_t g_new, double search_l2, std::unordered_set<int>* affected) {
        const knn_problem* knn = mesh->knn;

        POINT_TYPE gp;
        gp.x = g_old.x;
        gp.y = g_old.y;
#ifdef dim_3D
        gp.z = g_old.z;
#endif
        const int center = knn::cell_from_point(knn->N_grid, knn->grid_lo, knn->inv_cell_size, gp);

        auto consider = [&](int sid) {
            const int k = (int)mesh->sid_to_neighbor[sid];
            if (k >= (int)mesh->n_hydro) return;
            if (affected->count(k)) return;

            const double3 s   = mesh->seeds[k];
            const double  dox = s.x - g_old.x, doy = s.y - g_old.y, doz = s.z - g_old.z;
            const double  dnx = s.x - g_new.x, dny = s.y - g_new.y, dnz = s.z - g_new.z;
            const double  d2o = dox * dox + doy * doy + doz * doz;
            const double  d2n = dnx * dnx + dny * dny + dnz * dnz;
            if (d2o <= mesh->security_d2[k] || d2n <= mesh->security_d2[k]) affected->insert(k);
        };

        // farther than the rings reach: every point
        if (search_l2 >= knn->reach2) {
            for (int sid = 0; sid < (int)mesh->n_seeds; sid++)
                consider(sid);
            return;
        }

        int cx, cy, cz;
        knn::bucket_coords(center, knn->N_grid, &cx, &cy, &cz);
        for (int ring = 0; ring < knn->N_cell_offsets; ring++) {
            if (knn->d_cell_offset_dists[ring] > search_l2) break;
            if (!knn::ring_step_in_grid(knn->d_cell_offset_axes[ring], cx, cy, cz, knn->N_grid)) continue;
            const int cell = center + knn->d_cell_offsets[ring];

            const int base  = knn->d_ptrs[cell];
            const int count = knn->d_counters[cell];
            for (int i = 0; i < count; i++)
                consider(base + i);
        }
    }

    // takes the new position of ghost seeds a neighbour rank moved and rebuilds around them
    int repair_cells_for_moved_ghosts(VMesh*                                     mesh,
                                      const std::vector<proteus_mpi::MovedSeed>& moved,
                                      double                                     dt,
                                      std::vector<int>*                          newly_perturbed_out) {
        if (moved.empty()) return 0;

        double*   d_stored_points = (double*)mesh->knn->d_stored_points;
        const int n_hydro         = (int)mesh->n_hydro;

        s_wide_tier_rebuilds   = 0;
        s_uncertified_rebuilds = 0;

        // point of the list behind each moved ghost slot
        std::unordered_map<int, int> slot_to_sid;
        {
            std::unordered_set<int> wanted;
            for (const proteus_mpi::MovedSeed& m : moved)
                wanted.insert(n_hydro + m.ghost_slot);

            gpu_prefetch_to_cpu(mesh->sid_to_neighbor, mesh->n_seeds * sizeof(unsigned int));
            const int n_seeds = (int)mesh->n_seeds;
            for (int sid = 0; sid < n_seeds; sid++) {
                const int v = (int)mesh->sid_to_neighbor[sid];
                if (v >= n_hydro && wanted.count(v)) slot_to_sid[v - n_hydro] = sid;
            }
        }

        const double max_sec = sqrt(compute_max_security_d2(mesh));

        std::unordered_set<int> affected_set;
        for (const proteus_mpi::MovedSeed& m : moved) {
            const auto it = slot_to_sid.find(m.ghost_slot);
            if (it == slot_to_sid.end()) {
                proteus_mpi::exit_failure("VORONOI: moved ghost slot %d has no KNN point in this build\n",
                                          m.ghost_slot);
            }
            const int sid = it->second;

            const double4_t g_old = point_from_ptr(d_stored_points + DIMENSION * sid);
#ifdef dim_3D
            const double4_t g_new = make_double4_t(m.pos.x, m.pos.y, m.pos.z, 1.0);
#else
            const double4_t g_new = make_double4_t(m.pos.x, m.pos.y, 0.0, 1.0);
#endif
            const double ddx = g_new.x - g_old.x;
            const double ddy = g_new.y - g_old.y;
            const double ddz = g_new.z - g_old.z;
            // a cell is affected if the old or the new position is inside its security radius
            const double L = max_sec + sqrt(ddx * ddx + ddy * ddy + ddz * ddz);

            collect_affected_by_moved_ghost(mesh, g_old, g_new, L * L, &affected_set);

            d_stored_points[DIMENSION * sid + 0] = g_new.x;
            d_stored_points[DIMENSION * sid + 1] = g_new.y;
#ifdef dim_3D
            d_stored_points[DIMENSION * sid + 2] = g_new.z;
#endif
            mesh->seeds_g[m.ghost_slot] = double3{g_new.x, g_new.y, g_new.z};
        }

        std::vector<int> affected(affected_set.begin(), affected_set.end());
        std::sort(affected.begin(), affected.end());

        CellSids cell_sids = build_cell_sids_for(mesh, std::vector<int>());

        const CascadeResult result =
            cascade_rebuild_affected(mesh, d_stored_points, cell_sids, affected, dt, newly_perturbed_out);

        std::cout << "VORONOI: MPI repair: " << moved.size() << " moved ghost seed(s) -> " << affected.size()
                  << " affected cell(s), " << result.rebuilt << " rebuild(s) over " << result.rounds << " round(s)."
                  << std::endl;

        stop_if_uncertified("MPI repair");
        if (s_wide_tier_rebuilds > 0) {
            std::cerr << "VORONOI: " << s_wide_tier_rebuilds
                      << " cell(s) rebuilt on the wide tier during the MPI repair." << std::endl;
        }
        return result.rebuilt;
    }

} // namespace voronoi
