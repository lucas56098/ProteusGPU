namespace voronoi {

    // ---- file-local types ----
    namespace {
        enum class FallbackOutcome { ok_unchanged, ok_perturbed, failed };

        // Sparse cell -> sids lookup. Only cells that need a perturb-retry build (the rare
        // failed set, plus any cascade-affected cell that compute_single_voronoi_cell can't
        // rebuild on its own) get an entry. Replaces an older dense (n_hydro-wide) flat
        // layout that walked all n_seeds three times even when only a handful of cells
        // failed. Build is one O(n_seeds) pass over the target set; lazy extension is a
        // single linear scan per newly-needed cell, executed from ensure_built_for().
        struct CellSids {
            std::unordered_map<int, std::vector<int>> per_cell;
            const VMesh*                              mesh = nullptr;

            // ensure cell k has its sid list materialised. Called lazily by
            // rebuild_cell_with_perturb_retry when k wasn't in the initial failed set.
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

    // ---- forward declarations ----
    static int      count_failed_and_prefetch_status(VMesh* mesh);
    static CellSids build_cell_sids_for(const VMesh* mesh, const std::vector<int>& target_ks);
    static FallbackOutcome
    rebuild_cell_with_perturb_retry(
        VMesh* mesh, int k, double* d_stored_points, CellSids& cell_sids, double dt, Status& last_status_out);
    static std::vector<std::pair<double, int>>
    gather_nearby_seeds_sorted(double* d_stored_points, int seed_id, const knn_problem* knn, int max_candidates);
    static std::vector<std::pair<double, int>>
                   sort_neighbours_by_distance(double* d_stored_points, int seed_id, int n_seeds);
    template <typename CellT>
    static bool try_build_cell_from_neighbours_as(VMesh*                                     mesh,
                                                  int                                        k,
                                                  int                                        seed_id,
                                                  double*                                    d_stored_points,
                                                  const std::vector<std::pair<double, int>>& sorted,
                                                  bool                                       require_security,
                                                  Status&                                    last_status_out);
    static bool rebuild_on_wide_tier(VMesh*                                     mesh,
                                     int                                        k,
                                     int                                        seed_id,
                                     double*                                    d_stored_points,
                                     const std::vector<std::pair<double, int>>& bounded,
                                     Status&                                    last_status_out);
    static double3 compute_perturbation_delta(int seed_id, int attempt, double scale);
    static void    apply_perturbation(
        double* d_stored_points, double3 delta, const int* sids, size_t n_sids, const double4* orig_positions);
    static void
    rewind_perturbation(double* d_stored_points, const int* sids, size_t n_sids, const double4* orig_positions);
#ifdef MOVING_MESH
    static void apply_vmesh_perturbation_correction(VMesh* mesh, int k, double3 delta, double dt);
#endif
    static void             retire_face_range(VMesh* mesh, hsize_t first, hsize_t count);
    template <typename CellT> static void write_cell_to_mesh(VMesh* mesh, int k, const CellT& cell);
    static void             reclaim_appended_slice(
                    VMesh* mesh, int k, hsize_t fp_old, hsize_t fc_old, unsigned long long* face_offset,
                    unsigned long long off_before);
    static std::vector<int> collect_unique_neighbors(const VMesh* mesh, const std::vector<int>& sources);
    static void             run_symmetry_pass(VMesh*                  mesh,
                                              double*                 d_stored_points,
                                              CellSids&               cell_sids,
                                              const std::vector<int>& initial_perturbed,
                                              double                  dt);

#ifndef CPU_DEBUG
    GLOBAL void kernel_count_failures(int n, const Status* stat, int* fail_count);
    static int* d_fail_count = nullptr;
#endif

    // cells escalated to the wide tier during one cpu_fallback_failed_cells call; reported
    // once at the end rather than per cell, since a stressed mesh can escalate many at once
    static int s_wide_tier_rebuilds = 0;

    // ============================================================
    // Main routines
    // ============================================================

    // retry cells that failed both GPU tiers with hash-based seed perturbation;
    // perturbed cells trigger a symmetry pass to rebuild affected neighbours.
    // `dt > 0` enables the v_mesh correction inside rebuild_cell_with_perturb_retry so the
    // perturbed cell's mesh velocity offsets by delta/dt; pass 0.0 for the initial build.
    int cpu_fallback_failed_cells(VMesh* mesh, int* num_failed_out, double dt) {
        Status* stat            = mesh->cell_status;
        double* d_stored_points = (double*)mesh->knn->d_stored_points;

        // count failed cells (and pull cell_status to host on GPU)
        const int num_failed = count_failed_and_prefetch_status(mesh);
        if (num_failed_out) *num_failed_out = num_failed;
        if (num_failed == 0) return 0;

        const int n_hydro    = (int)mesh->n_hydro;
        s_wide_tier_rebuilds = 0;

        // first pass: enumerate failed cells once so we can build a sparse cell_sids for
        // exactly that set (one O(n_seeds) walk instead of three over the entire mesh).
        std::vector<int> failed_ks;
        failed_ks.reserve(num_failed);
        for (int k = 0; k < n_hydro; k++) {
            if (stat[k] == success) continue;
            const Status original = stat[k];
            // Overflow statuses are recoverable now: they mean the cell needed more plane or
            // triangle slots than the 8-bit tier can index, which the wide tier addresses.
            // Raising _MAX_P_/_MAX_T_ at build time cannot fix them -- plane ids are uchar with
            // 255 as the no-such-plane sentinel, and Euler (V = 2F - 4) ties _MAX_T_ to the same
            // ceiling. The wide tier is the runtime answer; this gate is what kept it
            // unreachable. Anything genuinely unknown still aborts.
            if (original != security_radius_not_reached && original != needs_exact_predicates
                && original != inconsistent_boundary && original != vertex_overflow
                && original != triangle_overflow && original != security_radius_beyond_data) {
                proteus_mpi::exit_failure(
                    "VORONOI: cell %d failed with unrecoverable status: %d\n", (int)k, (int)original);
            }
            failed_ks.push_back(k);
            // a failed cell owns no live face slice this step: the build tiers write
            // face_ptr/face_counts only on success, so both still hold the PREVIOUS build's
            // layout and point into data now owned by other cells. face_counts == 0 tells
            // write_cell_to_mesh "no slot to reuse or retire — append".
            mesh->face_counts[k] = 0;
        }

        // sparse cell_sids targeted at the failed set; run_symmetry_pass extends it lazily
        // if compute_single_voronoi_cell can't rebuild some cascade neighbour and we end up
        // calling rebuild_cell_with_perturb_retry for it.
        CellSids cell_sids = build_cell_sids_for(mesh, failed_ks);

        // perturb-retry each failed cell; track which got permanently perturbed
        std::vector<int> perturbed_ks;
        for (int k : failed_ks) {
            Status last_status = success;
            switch (rebuild_cell_with_perturb_retry(mesh, k, d_stored_points, cell_sids, dt, last_status)) {
            case FallbackOutcome::ok_unchanged:
                break;
            case FallbackOutcome::ok_perturbed:
                perturbed_ks.push_back(k);
                break;
            case FallbackOutcome::failed:
                proteus_mpi::exit_failure("VORONOI: cell %d all fallback attempts FAILED, aborting.\n", (int)k);
            }
        }

        // every failed cell was recovered on this rank (otherwise exit_failure fired).
        // The caller does a single sum_global on num_failed and logs the global total
        // via logging::root() — see cpu_perturb_and_rebuild.

        // if any cell stuck at a perturbation, rebuild its neighbours
        if (!perturbed_ks.empty()) { run_symmetry_pass(mesh, d_stored_points, cell_sids, perturbed_ks, dt); }

        if (s_wide_tier_rebuilds > 0) {
            std::cerr << "VORONOI: " << s_wide_tier_rebuilds << " cell(s) rebuilt on the wide tier ("
                      << _BIG_MAX_P_ << "/" << _BIG_MAX_T_ << " slots)." << std::endl;
        }
        return (int)perturbed_ks.size();
    }

    // ============================================================
    // Helpers
    // ============================================================

    // count cells whose status is not success; on GPU also pull cell_status to host
    static int count_failed_and_prefetch_status(VMesh* mesh) {
#ifndef CPU_DEBUG
        // GPU: launch the count kernel and read the result back
        if (!d_fail_count) d_fail_count = gpu_calloc<int>(1);
        gpu_memset(d_fail_count, 0, sizeof(int));
        const int n_hydro = (int)mesh->n_hydro;
        const int tpb     = _MESH_BLOCK_SIZE_;
        const int blocks  = (n_hydro + tpb - 1) / tpb;
        kernel_count_failures<<<blocks, tpb>>>(n_hydro, mesh->cell_status, d_fail_count);
        GPU_SYNC();
        const int n_failed = *d_fail_count;

        // pull cell_status to host so the fallback loop can read it
        if (n_failed > 0) gpu_prefetch_to_cpu(mesh->cell_status, n_hydro * sizeof(Status));
        return n_failed;
#else
        // CPU: serial scan
        int n_failed = 0;
        for (hsize_t k = 0; k < mesh->n_hydro; k++)
            if (mesh->cell_status[k] != success) n_failed++;
        return n_failed;
#endif
    }

    // Build a sparse cell->sids lookup containing entries only for the given target cells.
    // A single linear sweep over n_seeds (vs the old three full passes), and the result
    // is a small unordered_map keyed by target k instead of a flat n_hydro-wide array.
    // Cascade-triggered cells that aren't in target_ks get their sid list materialised
    // on demand by CellSids::ensure_built_for().
    static CellSids build_cell_sids_for(const VMesh* mesh, const std::vector<int>& target_ks) {
        CellSids cs;
        cs.mesh = mesh;
        if (target_ks.empty()) return cs;

        // empty bucket per target so size_for(k) on a hit doesn't throw
        std::unordered_set<int> target_set(target_ks.begin(), target_ks.end());
        for (int k : target_ks)
            cs.per_cell[k];

        const int n_seeds = (int)mesh->n_seeds;
        for (int sid = 0; sid < n_seeds; sid++) {
            const int k = (int)mesh->sid_to_neighbor[sid];
            if (target_set.count(k)) cs.per_cell[k].push_back(sid);
        }
        return cs;
    }

    // K' = max candidates for the bounded-by-distance first pass. Big enough that for a
    // mostly-cartesian mesh the security radius is reached within these planes for any cell;
    // small enough that the per-cell sort runs in microseconds vs ~2 s for the n_seeds-wide
    // exhaustive sort. If the bounded set fails (rare; only on truly pathological geometry),
    // rebuild_cell_with_perturb_retry escalates to the full sort as a correctness safety net.
    static constexpr int FALLBACK_BOUNDED_K = 2048;

    // run the perturb-retry ladder against a pre-computed sorted clip list. Returns
    // FallbackOutcome::failed if every attempt (unperturbed + 12 scales) ran out without
    // producing a security-radius-certified cell. `require_security` is forwarded to
    // try_build_cell_from_neighbours: TRUE means treat list-exhaustion as failure, used when
    // `sorted` is bounded; FALSE means trust the list (used for the full-sort escalation).
    static FallbackOutcome run_perturb_ladder(VMesh*                                     mesh,
                                              int                                        k,
                                              int                                        seed_id,
                                              double*                                    d_stored_points,
                                              const std::vector<std::pair<double, int>>& sorted,
                                              const int*                                 sids,
                                              size_t                                     n_sids,
                                              const double4*                             orig_positions,
                                              double                                     dt,
                                              bool                                       require_security,
                                              Status&                                    last_status_out) {
        // ladder: attempt 0 unperturbed, then attempts 1..12 with 10x growing scale.
        constexpr int max_perturb = 12;
        double        scale       = 1e-13;
        for (int attempt = 0; attempt <= max_perturb; attempt++) {
            double3 delta = {0.0, 0.0, 0.0};
            if (attempt > 0) {
                delta = compute_perturbation_delta(seed_id, attempt, scale);
                apply_perturbation(d_stored_points, delta, sids, n_sids, orig_positions);
            }

            Status     attempt_status = success;
            const bool ok              = try_build_cell_from_neighbours_as<ConvexCell>(
                mesh, k, seed_id, d_stored_points, sorted, require_security, attempt_status);
            if (ok) {
                if (attempt == 0) return FallbackOutcome::ok_unchanged;
#ifdef MOVING_MESH
                // perturbation is a non-physical position jump; treat it as an extra step in
                // seed motion so face velocities downstream match the new geometry
                apply_vmesh_perturbation_correction(mesh, k, delta, dt);
#else
                (void)dt;
#endif
                return FallbackOutcome::ok_perturbed;
            }

            last_status_out = attempt_status;
            if (attempt > 0) rewind_perturbation(d_stored_points, sids, n_sids, orig_positions);
            scale *= 10.0;
        }
        return FallbackOutcome::failed;
    }

    // try the unperturbed cell first, then perturb the seed with growing scale on each retry.
    // First pass uses a KNN-grid-bounded clip list (K' nearest seeds) — typical fallback cells
    // converge well within K'. If every perturb attempt with the bounded list fails, escalate
    // to the n_seeds-exhaustive sort as a correctness safety net. The exhaustive list is more
    // robust than KNN's K-nearest sample for stubborn degeneracies. On ok_perturbed, the
    // surviving delta is folded into v_mesh (when dt > 0) so face velocities stay consistent
    // with the perturbed geometry.
    static FallbackOutcome
    rebuild_cell_with_perturb_retry(
        VMesh* mesh, int k, double* d_stored_points, CellSids& cell_sids, double dt, Status& last_status_out) {
        // lazy-build cell_sids[k] if a cascade in run_symmetry_pass brought us a cell that
        // wasn't in the initial failed-set; no-op when build_cell_sids_for already populated it
        cell_sids.ensure_built_for(k);

        const int seed_id = (int)mesh->real_sorted_ids[k];

        // clip list for both paths below: K' nearest candidates from the KNN grid
        const auto bounded = gather_nearby_seeds_sorted(d_stored_points, seed_id, mesh->knn, FALLBACK_BOUNDED_K);

        // The cell exhausted the 8-bit tier's plane/triangle slots. The perturb ladder cannot
        // help -- it addresses numerical near-degeneracies, and this cell simply needs more
        // slots than _MAX_P_/_MAX_T_ can express -- so go straight to the wide tier. Running
        // the ladder first would burn 13 clip passes only to overflow again on every one.
        const Status incoming = mesh->cell_status[k];
        if (incoming == vertex_overflow || incoming == triangle_overflow) {
            return rebuild_on_wide_tier(mesh, k, seed_id, d_stored_points, bounded, last_status_out)
                       ? FallbackOutcome::ok_unchanged
                       : FallbackOutcome::failed;
        }

        const int*   sids   = cell_sids.begin_for(k);
        const size_t n_sids = (size_t)cell_sids.size_for(k);

        // snapshot original positions so a failed perturb attempt can be rewound
        std::vector<double4> orig_positions(n_sids);
        for (size_t i = 0; i < n_sids; i++)
            orig_positions[i] = point_from_ptr(d_stored_points + DIMENSION * sids[i]);

        // first pass: perturb ladder against the bounded clip list
        FallbackOutcome outcome = run_perturb_ladder(mesh,
                                                     k,
                                                     seed_id,
                                                     d_stored_points,
                                                     bounded,
                                                     sids,
                                                     n_sids,
                                                     orig_positions.data(),
                                                     dt,
                                                     /*require_security=*/true,
                                                     last_status_out);
        if (outcome != FallbackOutcome::failed) return outcome;

        // escalation: exhaustive sort against every other seed. Rare path; only hit when the
        // bounded list was geometrically insufficient. Sorting all n_seeds is O(n log n) but
        // we only pay it for cells the bounded pass couldn't handle.
        const auto full = sort_neighbours_by_distance(d_stored_points, seed_id, (int)mesh->n_seeds);
        outcome         = run_perturb_ladder(mesh,
                                     k,
                                     seed_id,
                                     d_stored_points,
                                     full,
                                     sids,
                                     n_sids,
                                     orig_positions.data(),
                                     dt,
                                     /*require_security=*/false,
                                     last_status_out);
        if (outcome != FallbackOutcome::failed) return outcome;

        // Final escalation: the wide tier, but only when the ladder died on a capacity
        // overflow. A cell that arrived already overflowed took the direct path above, so
        // reaching here means the overflow first appeared during the ladder.
        if (last_status_out != vertex_overflow && last_status_out != triangle_overflow) {
            return FallbackOutcome::failed;
        }
        return rebuild_on_wide_tier(mesh, k, seed_id, d_stored_points, bounded, last_status_out)
                   ? FallbackOutcome::ok_unchanged
                   : FallbackOutcome::failed;
    }

    // gather up to `max_candidates` seeds closest to `seed_id` from the KNN spatial grid,
    // returned sorted ascending by distance². Walks the pre-computed ring offsets in
    // distance order (knn->d_cell_offsets / d_cell_offset_dists) and early-exits once the
    // next ring's lower-bound distance² exceeds the worst entry currently in the top-K' set,
    // so we touch only the buckets that can plausibly contribute. ~3 orders of magnitude
    // faster than sort_neighbours_by_distance on the 30M-seed mesh; sufficient for any
    // non-pathological cell. Caller must check `try_build_cell_from_neighbours`'s
    // require_security flag to escalate to the full sort if the bounded set is too small.
    static std::vector<std::pair<double, int>>
    gather_nearby_seeds_sorted(double* d_stored_points, int seed_id, const knn_problem* knn, int max_candidates) {
        const double4 seed_pos = point_from_ptr(d_stored_points + DIMENSION * seed_id);
        const int     seed_cell =
            knn::cellFromPoint(knn->N_grid, knn->buff, knn->inv_boxsize, knn->d_stored_points[seed_id]);

        std::vector<std::pair<double, int>> candidates;
        candidates.reserve(max_candidates * 2);

        double kth_dist = DBL_MAX;
        for (int ring = 0; ring < knn->N_cell_offsets; ring++) {
            // early-out: have enough candidates and next ring's lower bound is past our worst
            if ((int)candidates.size() >= max_candidates && knn->d_cell_offset_dists[ring] >= kth_dist) break;

            const int cell = seed_cell + knn->d_cell_offsets[ring];
            if (cell < 0 || cell >= knn->Npow) continue;

            const int cell_base  = knn->d_ptrs[cell];
            const int cell_count = knn->d_counters[cell];
            for (int i = 0; i < cell_count; i++) {
                const int sid = cell_base + i;
                if (sid == seed_id) continue;
                const double4 other = point_from_ptr(d_stored_points + DIMENSION * sid);
                const double  dx    = other.x - seed_pos.x;
                const double  dy    = other.y - seed_pos.y;
                const double  dz    = other.z - seed_pos.z;
                candidates.push_back({dx * dx + dy * dy + dz * dz, sid});
            }

            // refresh the kth-worst distance to bound the next early-out check
            if ((int)candidates.size() >= max_candidates) {
                std::nth_element(candidates.begin(), candidates.begin() + max_candidates - 1, candidates.end());
                kth_dist = candidates[max_candidates - 1].first;
            }
        }

        if ((int)candidates.size() > max_candidates) candidates.resize(max_candidates);
        std::sort(candidates.begin(), candidates.end());
        return candidates;
    }

    // sort every other seed by distance² from seed_id; the exhaustive correctness-safety-net
    // clip list invoked only when the bounded gather couldn't yield a security-certified cell.
    static std::vector<std::pair<double, int>>
    sort_neighbours_by_distance(double* d_stored_points, int seed_id, int n_seeds) {
        const double4                       seed_pos = point_from_ptr(d_stored_points + DIMENSION * seed_id);
        std::vector<std::pair<double, int>> dists;
        dists.reserve(n_seeds - 1);
        for (int j = 0; j < n_seeds; j++) {
            if (j == seed_id) continue;
            const double4 other = point_from_ptr(d_stored_points + DIMENSION * j);
            const double  dx    = other.x - seed_pos.x;
            const double  dy    = other.y - seed_pos.y;
            const double  dz    = other.z - seed_pos.z;
            dists.push_back({dx * dx + dy * dy + dz * dz, j});
        }
        std::sort(dists.begin(), dists.end());
        return dists;
    }

    // attempt one cell build by clipping against every seed in the provided distance-sorted list.
    // Returns true and writes the cell into mesh on success. When `require_security` is true,
    // a list-exhaustion without is_security_radius_reached is treated as failure — the bounded
    // gather can run out before the cell is fully enclosed, in which case we need to escalate
    // to the exhaustive sort. When false (full-sort caller) the list is already exhaustive, so
    // exhaustion is fine: the cell has been clipped against every seed and is correct.
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

        // clip plane-by-plane in distance order until security radius is hit or status fails
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
        if (require_security && !security_reached) {
            last_status_out = security_radius_not_reached;
            return false;
        }

        write_cell_to_mesh(mesh, k, cell);
        mesh->cell_status[k] = success;
        return true;
    }

    // Rebuild cell k on the wide (32-bit index) tier. Tries the bounded clip list first and
    // only pays the n_seeds-wide exhaustive sort if those candidates could not enclose the
    // cell, since that sort costs seconds per cell on a large mesh.
    //
    // No perturbation here: a capacity overflow is not a numerical near-degeneracy, and moving
    // the seed does not change how many planes the security radius requires.
    static bool rebuild_on_wide_tier(VMesh*                                     mesh,
                                     int                                        k,
                                     int                                        seed_id,
                                     double*                                    d_stored_points,
                                     const std::vector<std::pair<double, int>>& bounded,
                                     Status&                                    last_status_out) {
        Status st = success;
        if (try_build_cell_from_neighbours_as<BigConvexCell>(
                mesh, k, seed_id, d_stored_points, bounded, /*require_security=*/true, st)) {
            s_wide_tier_rebuilds++;
            return true;
        }

        const auto full = sort_neighbours_by_distance(d_stored_points, seed_id, (int)mesh->n_seeds);
        if (try_build_cell_from_neighbours_as<BigConvexCell>(
                mesh, k, seed_id, d_stored_points, full, /*require_security=*/false, st)) {
            s_wide_tier_rebuilds++;
            return true;
        }

        last_status_out = st;
        return false;
    }

    // hash-deterministic perturbation: each (seed_id, attempt, scale) maps to the same delta
    static double3 compute_perturbation_delta(int seed_id, int attempt, double scale) {
        // mix seed_id + attempt into a 16-bit field used for each axis offset
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

    // shift each sid that touches this cell by the same (dx, dy, dz)
    static void apply_perturbation(
        double* d_stored_points, double3 delta, const int* sids, size_t n_sids, const double4* orig_positions) {
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
    // The fallback shifts a degenerate seed by `delta` at time t_n+dt. The mesh velocity used
    // for the half-step's face velocities was computed before the move, so v_mesh * dt
    // accounts for only part of the seed's effective displacement. Adding delta / dt brings
    // v_mesh in line with the post-perturbation geometry, keeping face velocities consistent
    // in the second flux update. dt <= 0 (e.g. the initial mesh build) is a no-op.
    static void apply_vmesh_perturbation_correction(VMesh* mesh, int k, double3 delta, double dt) {
        if (dt <= 0.0) return;
        const double inv_dt = 1.0 / dt;
        mesh->v_mesh[k].x += delta.x * inv_dt;
        mesh->v_mesh[k].y += delta.y * inv_dt;
#ifdef dim_3D
        mesh->v_mesh[k].z += delta.z * inv_dt;
#endif
    }
#endif

    // restore the pre-perturb positions if an attempt did not succeed
    static void
    rewind_perturbation(double* d_stored_points, const int* sids, size_t n_sids, const double4* orig_positions) {
        for (size_t i = 0; i < n_sids; i++) {
            const int sid                        = sids[i];
            d_stored_points[DIMENSION * sid + 0] = orig_positions[i].x;
            d_stored_points[DIMENSION * sid + 1] = orig_positions[i].y;
#ifdef dim_3D
            d_stored_points[DIMENSION * sid + 2] = orig_positions[i].z;
#endif
        }
    }

    // ---- face-slot management ----
    //
    // The parallel build hands out face_ptr[k] via an atomic counter, so face storage order is
    // a random permutation of cells and a rebuilt cell can never shift its neighbours' slices.
    // The fallback therefore manages the face array as a slot store: a rebuilt cell reuses its
    // own slot when the new face count fits (the common case — a perturb rebuild rarely changes
    // the face count), otherwise it appends at num_faces and retires its old slice. Retired
    // entries stay inside [0, num_faces) marked inert (neighbor_cell = -1, zero area/f_mid):
    // every physics consumer walks face_ptr[k]/face_counts[k] slices and never sees them, and
    // the one flat iteration over [0, num_faces) (halo_build's mark_used_bitmap) skips negative
    // neighbour ids. This keeps the whole recovery path O(faces of rebuilt cells).
    //
    // An earlier design appended and then compacted the ENTIRE face array — an O(num_faces)
    // serial host rewrite every time a single cell was perturbed. Do not reintroduce it:
    // partial compaction is unsound (face_ptr is not k-ordered, see above) and full compaction
    // buys nothing the inert-entry invariant doesn't already provide.

    // mark [first, first + count) as retired/inert
    static void retire_face_range(VMesh* mesh, hsize_t first, hsize_t count) {
        for (hsize_t i = first; i < first + count; i++) {
            mesh->neighbor_cell[i] = -1;
            mesh->face_area[i]     = 0.0;
#ifdef MOVING_MESH
            for (int c = 0; c < DIMENSION - 1; c++)
                mesh->f_mid_local[i * (DIMENSION - 1) + c] = 0.0;
#endif
        }
    }

    // write a successfully built cell into mesh's face arrays: in place when it fits the
    // cell's existing slot, appended otherwise. face_counts[k] == 0 marks "no live slot".
    //
    // fc_max (count_cell_faces) only bounds the slot-fit decision and the capacity check;
    // face_counts[k] and num_faces advance by what extract_cell_all actually wrote, so a
    // degenerate cell that closes fewer faces than planes leaves no unwritten entry inside
    // its live slice. This path is serial, so unlike the atomic build there is no reservation
    // to give back — the append advances by the true count.
    template <typename CellT> static void write_cell_to_mesh(VMesh* mesh, int k, const CellT& cell) {
        const hsize_t fc_max = (hsize_t)count_cell_faces(cell);
        const hsize_t fp_old = mesh->face_ptr[k];
        const hsize_t fc_old = mesh->face_counts[k];
        if (fc_max <= fc_old) {
            const hsize_t written = extract_cell_all(cell, mesh, (hsize_t)k);
            mesh->face_counts[k]  = written;
            retire_face_range(mesh, fp_old + written, fc_old - written);
        } else {
            retire_face_range(mesh, fp_old, fc_old);
            ensure_face_capacity(mesh, mesh->num_faces + fc_max);
            mesh->face_ptr[k]     = mesh->num_faces;
            const hsize_t written = extract_cell_all(cell, mesh, (hsize_t)k);
            mesh->face_counts[k]  = written;
            mesh->num_faces += written;
        }
    }

    // compute_single_voronoi_cell (shared with the GPU kernels, so left untouched) always
    // appends the rebuilt slice at *face_offset and repoints face_ptr[k] there. Relocate the
    // slice back into the cell's original slot when it fits and roll the append back — the
    // cascade is serial and owns the counter, so the rollback is safe. Otherwise keep the
    // appended location and retire the original slice.
    //
    // `off_before` is the counter value from before the rebuild. The rollback restores it
    // exactly rather than subtracting face_counts[k]: the cell reserved count_cell_faces
    // slots but face_counts[k] now holds the (possibly smaller) number extract_cell_all
    // wrote, so subtracting the latter would leave the reservation's slack behind forever.
    static void reclaim_appended_slice(
        VMesh* mesh, int k, hsize_t fp_old, hsize_t fc_old, unsigned long long* face_offset,
        unsigned long long off_before) {
        const hsize_t fp_new = mesh->face_ptr[k];
        const hsize_t fc_new = mesh->face_counts[k];
        if (fc_new <= fc_old) {
            for (hsize_t i = 0; i < fc_new; i++) {
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

    // collect unique neighbour-k indices touched by any source cell
    static std::vector<int> collect_unique_neighbors(const VMesh* mesh, const std::vector<int>& sources) {
        const int         n_hydro = (int)mesh->n_hydro;
        std::vector<bool> seen(n_hydro, false);
        std::vector<int>  result;
        for (int k : sources) {
            const hsize_t fp = mesh->face_ptr[k];
            const hsize_t fc = mesh->face_counts[k];
            for (hsize_t f = 0; f < fc; f++) {
                const int kn = mesh->neighbor_cell[fp + f];
                if (kn < 0 || kn >= n_hydro) continue; // box-boundary face
                if (seen[kn]) continue;
                seen[kn] = true;
                result.push_back(kn);
            }
        }
        return result;
    }

    // after perturbation, neighbours of perturbed cells were built against the OLD seed positions
    // and need rebuilding. Cascade until no more neighbours change (or MAX_ROUNDS). Rebuilt
    // slices are folded back into each cell's own slot by reclaim_appended_slice, so the face
    // array needs no compaction.
    static void run_symmetry_pass(VMesh*                  mesh,
                                  double*                 d_stored_points,
                                  CellSids&               cell_sids,
                                  const std::vector<int>& initial_perturbed,
                                  double                  dt) {
        // each round only rebuilds the cells that actually changed in the previous round,
        // so cost is bounded by the propagation neighbourhood and ramping this up is cheap.
        constexpr int MAX_ROUNDS = 12;

        std::vector<int>   work        = initial_perturbed;
        int                rebuilt     = 0;
        int                rounds      = 0;
        unsigned long long face_offset = (unsigned long long)mesh->num_faces;
        int                overflow    = 0;

        while (!work.empty()) {
            if (++rounds > MAX_ROUNDS) {
                std::cerr << "VORONOI: symmetry cascade did not converge after " << MAX_ROUNDS << " rounds, aborting."
                          << std::endl;
                exit(EXIT_FAILURE);
            }

            // build the unique set of neighbours that need rebuilding this round
            std::vector<int> affected = collect_unique_neighbors(mesh, work);
            if (affected.empty()) break;

            // rebuild each affected cell; track which ones get perturbed for the next round
            std::vector<int> next_work;
            for (int kn : affected) {
                mesh->cell_status[kn] = security_radius_not_reached;
                const int seed_id     = (int)mesh->real_sorted_ids[kn];
                // snapshot the cell's live slot; on success the appended rebuild is folded
                // back into it (affected cells were built successfully this step, so the
                // slot is valid — unlike the initial failed set)
                const hsize_t            fp_old     = mesh->face_ptr[kn];
                const hsize_t            fc_old     = mesh->face_counts[kn];
                const unsigned long long off_before = face_offset;
                compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_, uchar, VERT_TYPE>(
                    kn, seed_id, d_stored_points, mesh->knn, mesh->cell_status, mesh, &face_offset, &overflow);
                if (overflow) {
                    std::cerr << "VORONOI: face overflow during symmetry rebuild — "
                                 "increase _FACE_CAPACITY_MULT_ in Config.sh."
                              << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (mesh->cell_status[kn] == success) {
                    reclaim_appended_slice(mesh, kn, fp_old, fc_old, &face_offset, off_before);
                }

                // KNN rebuild failed for this neighbour: fall through to the perturb path
                if (mesh->cell_status[kn] != success) {
                    mesh->num_faces         = (hsize_t)face_offset;
                    Status          last_status = success;
                    FallbackOutcome outcome =
                        rebuild_cell_with_perturb_retry(mesh, kn, d_stored_points, cell_sids, dt, last_status);
                    if (outcome == FallbackOutcome::failed) {
                        proteus_mpi::exit_failure(
                            "VORONOI: symmetry rebuild for cell %d all fallback attempts FAILED.\n", (int)kn);
                    }
                    face_offset = (unsigned long long)mesh->num_faces;
                    if (outcome == FallbackOutcome::ok_perturbed) next_work.push_back(kn);
                }
                rebuilt++;
            }
            work = std::move(next_work);
        }
        mesh->num_faces = (hsize_t)face_offset;

        std::cout << "VORONOI: " << initial_perturbed.size() << " cell(s) permanently perturbed; " << rebuilt
                  << " neighbour rebuild(s) over " << rounds << " round(s)." << std::endl;
    }

    // ============================================================
    // CUDA kernels
    // ============================================================
#ifndef CPU_DEBUG
    // count cells whose status is not success via per-thread atomic add
    GLOBAL void kernel_count_failures(int n, const Status* stat, int* fail_count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n && stat[i] != success) portable_atomicAdd(fail_count, 1);
    }
#endif

} // namespace voronoi
