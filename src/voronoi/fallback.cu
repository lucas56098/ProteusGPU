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
    rebuild_cell_with_perturb_retry(VMesh* mesh, int k, double* d_stored_points, CellSids& cell_sids, double dt);
    static std::vector<std::pair<double, int>>
    gather_nearby_seeds_sorted(double* d_stored_points, int seed_id, const knn_problem* knn, int max_candidates);
    static std::vector<std::pair<double, int>>
                   sort_neighbours_by_distance(double* d_stored_points, int seed_id, int n_seeds);
    static bool    try_build_cell_from_neighbours(VMesh*                                     mesh,
                                                  int                                        k,
                                                  int                                        seed_id,
                                                  double*                                    d_stored_points,
                                                  const std::vector<std::pair<double, int>>& sorted,
                                                  bool                                       require_security);
    static double3 compute_perturbation_delta(int seed_id, int attempt, double scale);
    static void    apply_perturbation(
        double* d_stored_points, double3 delta, const int* sids, size_t n_sids, const double4* orig_positions);
    static void
    rewind_perturbation(double* d_stored_points, const int* sids, size_t n_sids, const double4* orig_positions);
#ifdef MOVING_MESH
    static void apply_vmesh_perturbation_correction(VMesh* mesh, int k, double3 delta, double dt);
#endif
    static void             append_cell_to_mesh(VMesh* mesh, int k, const ConvexCell& cell);
    static std::vector<int> collect_unique_neighbors(const VMesh* mesh, const std::vector<int>& sources);
    static void             run_symmetry_pass(VMesh*                  mesh,
                                              double*                 d_stored_points,
                                              CellSids&               cell_sids,
                                              const std::vector<int>& initial_perturbed,
                                              double                  dt,
                                              int&                    min_touched_k);
    static void             compact_face_arrays(VMesh* mesh, int min_touched_k);

#ifndef CPU_DEBUG
    GLOBAL void kernel_count_failures(int n, const Status* stat, int* fail_count);
    static int* d_fail_count = nullptr;
#endif

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

        const int n_hydro = (int)mesh->n_hydro;

        // first pass: enumerate failed cells once so we can build a sparse cell_sids for
        // exactly that set (one O(n_seeds) walk instead of three over the entire mesh).
        std::vector<int> failed_ks;
        failed_ks.reserve(num_failed);
        for (int k = 0; k < n_hydro; k++) {
            if (stat[k] == success) continue;
            const Status original = stat[k];
            if (original != security_radius_not_reached && original != needs_exact_predicates) {
                proteus_mpi::exit_failure(
                    "VORONOI: cell %d failed with unrecoverable status: %d\n", (int)k, (int)original);
            }
            failed_ks.push_back(k);
        }

        // sparse cell_sids targeted at the failed set; run_symmetry_pass extends it lazily
        // if compute_single_voronoi_cell can't rebuild some cascade neighbour and we end up
        // calling rebuild_cell_with_perturb_retry for it.
        CellSids cell_sids = build_cell_sids_for(mesh, failed_ks);

        // perturb-retry each failed cell; track which got permanently perturbed and the
        // lowest-index k touched (needed by compact_face_arrays to bound its work).
        std::vector<int> perturbed_ks;
        int              min_touched_k = n_hydro; // sentinel: nothing touched yet
        for (int k : failed_ks) {
            switch (rebuild_cell_with_perturb_retry(mesh, k, d_stored_points, cell_sids, dt)) {
            case FallbackOutcome::ok_unchanged:
                break;
            case FallbackOutcome::ok_perturbed:
                perturbed_ks.push_back(k);
                break;
            case FallbackOutcome::failed:
                proteus_mpi::exit_failure("VORONOI: cell %d all fallback attempts FAILED, aborting.\n", (int)k);
            }
            // every failed-cell rebuild calls append_cell_to_mesh and rewrites face_ptr[k]
            if (k < min_touched_k) min_touched_k = k;
        }

        // every failed cell was recovered on this rank (otherwise exit_failure fired).
        // The caller does a single sum_global on num_failed and logs the global total
        // via logging::root() — see cpu_perturb_and_rebuild.

        // if any cell stuck at a perturbation, rebuild neighbours + compact face data
        if (!perturbed_ks.empty()) {
            run_symmetry_pass(mesh, d_stored_points, cell_sids, perturbed_ks, dt, min_touched_k);
            compact_face_arrays(mesh, min_touched_k);
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
                                              bool                                       require_security) {
        // ladder: attempt 0 unperturbed, then attempts 1..12 with 10x growing scale.
        constexpr int max_perturb = 12;
        double        scale       = 1e-13;
        for (int attempt = 0; attempt <= max_perturb; attempt++) {
            double3 delta = {0.0, 0.0, 0.0};
            if (attempt > 0) {
                delta = compute_perturbation_delta(seed_id, attempt, scale);
                apply_perturbation(d_stored_points, delta, sids, n_sids, orig_positions);
            }

            const bool ok = try_build_cell_from_neighbours(mesh, k, seed_id, d_stored_points, sorted, require_security);
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
    rebuild_cell_with_perturb_retry(VMesh* mesh, int k, double* d_stored_points, CellSids& cell_sids, double dt) {
        // lazy-build cell_sids[k] if a cascade in run_symmetry_pass brought us a cell that
        // wasn't in the initial failed-set; no-op when build_cell_sids_for already populated it
        cell_sids.ensure_built_for(k);

        const int    seed_id = (int)mesh->real_sorted_ids[k];
        const int*   sids    = cell_sids.begin_for(k);
        const size_t n_sids  = (size_t)cell_sids.size_for(k);

        // snapshot original positions so a failed perturb attempt can be rewound
        std::vector<double4> orig_positions(n_sids);
        for (size_t i = 0; i < n_sids; i++)
            orig_positions[i] = point_from_ptr(d_stored_points + DIMENSION * sids[i]);

        // first pass: bounded sort, K' candidates from the KNN grid
        const auto      bounded = gather_nearby_seeds_sorted(d_stored_points, seed_id, mesh->knn, FALLBACK_BOUNDED_K);
        FallbackOutcome outcome = run_perturb_ladder(mesh,
                                                     k,
                                                     seed_id,
                                                     d_stored_points,
                                                     bounded,
                                                     sids,
                                                     n_sids,
                                                     orig_positions.data(),
                                                     dt,
                                                     /*require_security=*/true);
        if (outcome != FallbackOutcome::failed) return outcome;

        // escalation: exhaustive sort against every other seed. Rare path; only hit when the
        // bounded list was geometrically insufficient. Sorting all n_seeds is O(n log n) but
        // we only pay it for cells the bounded pass couldn't handle.
        const auto full = sort_neighbours_by_distance(d_stored_points, seed_id, (int)mesh->n_seeds);
        return run_perturb_ladder(mesh,
                                  k,
                                  seed_id,
                                  d_stored_points,
                                  full,
                                  sids,
                                  n_sids,
                                  orig_positions.data(),
                                  dt,
                                  /*require_security=*/false);
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
    static bool try_build_cell_from_neighbours(VMesh*                                     mesh,
                                               int                                        k,
                                               int                                        seed_id,
                                               double*                                    d_stored_points,
                                               const std::vector<std::pair<double, int>>& sorted,
                                               bool                                       require_security) {
        Status     status = success;
        ConvexCell cell(seed_id, d_stored_points, &status, mesh->buff);

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
        if (status != success) return false;
        if (require_security && !security_reached) return false;

        append_cell_to_mesh(mesh, k, cell);
        mesh->cell_status[k] = success;
        return true;
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

    // write a successfully built ConvexCell into mesh's face arrays + reserve capacity
    static void append_cell_to_mesh(VMesh* mesh, int k, const ConvexCell& cell) {
        // count_cell_faces sizes the reservation; extract_cell_all reports what it actually
        // wrote, which can be fewer. num_faces still advances by the reservation, so the
        // unwritten tail stays slack that face_counts keeps anyone from reading.
        const int fc = count_cell_faces(cell);
        ensure_face_capacity(mesh, mesh->num_faces + fc);
        mesh->face_ptr[k]    = mesh->num_faces;
        mesh->face_counts[k] = extract_cell_all(cell, mesh, (hsize_t)k);
        mesh->num_faces += (hsize_t)fc;
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
    // and need rebuilding. Cascade until no more neighbours change (or MAX_ROUNDS). Every
    // affected cell's face_ptr is rewritten by append_cell_to_mesh, so we feed the lowest
    // touched k back to compact_face_arrays via min_touched_k.
    static void run_symmetry_pass(VMesh*                  mesh,
                                  double*                 d_stored_points,
                                  CellSids&               cell_sids,
                                  const std::vector<int>& initial_perturbed,
                                  double                  dt,
                                  int&                    min_touched_k) {
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
                if (kn < min_touched_k) min_touched_k = kn;
                mesh->cell_status[kn] = security_radius_not_reached;
                const int seed_id     = (int)mesh->real_sorted_ids[kn];
                compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_>(
                    kn, seed_id, d_stored_points, mesh->knn, mesh->cell_status, mesh, &face_offset, &overflow);
                if (overflow) {
                    std::cerr << "VORONOI: face overflow during symmetry rebuild — "
                                 "increase _FACE_CAPACITY_MULT_ in Config.sh."
                              << std::endl;
                    exit(EXIT_FAILURE);
                }

                // KNN rebuild failed for this neighbour: fall through to the perturb path
                if (mesh->cell_status[kn] != success) {
                    mesh->num_faces         = (hsize_t)face_offset;
                    FallbackOutcome outcome = rebuild_cell_with_perturb_retry(mesh, kn, d_stored_points, cell_sids, dt);
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

    // append-then-compact: symmetry-pass rebuilds push new face data past the original num_faces,
    // leaving gaps. Compact only from k = min_touched_k onward — every cell with smaller k
    // already lives at its correct compact offset (no gaps before it), so we copy nothing
    // and reallocate no scratch for the unchanged prefix. Saves ~(min_touched_k / n_hydro)
    // of the scratch and copy traffic; near-total saving when the perturbations cluster
    // at high k, vs the old version that always rewrote the whole face array.
    static void compact_face_arrays(VMesh* mesh, int min_touched_k) {
        const int n_hydro = (int)mesh->n_hydro;
        if (min_touched_k >= n_hydro) return; // nothing touched (defensive)
        if (min_touched_k < 0) min_touched_k = 0;

        // For k < min_touched_k: face_ptr[k] is already at the correct compact offset.
        // Compute the starting offset = sum of face_counts[0..min_touched_k-1].
        hsize_t starting_offset = 0;
        for (int k = 0; k < min_touched_k; k++)
            starting_offset += mesh->face_counts[k];

        // Total faces that need to be moved (cells k >= min_touched_k).
        hsize_t total_remaining = 0;
        for (int k = min_touched_k; k < n_hydro; k++)
            total_remaining += mesh->face_counts[k];

        // scratch sized to the moving region only (not the whole face array)
        std::vector<int>    neighbor_tmp(total_remaining);
        std::vector<double> area_tmp(total_remaining);
#ifdef MOVING_MESH
        std::vector<double> fmid_tmp(total_remaining * (DIMENSION - 1));
#endif

        // pass 1: walk cells [min_touched_k, n_hydro), copy faces into scratch in compact order,
        // rewrite face_ptr[k] to the new compact location
        hsize_t out_local = 0;
        for (int k = min_touched_k; k < n_hydro; k++) {
            const hsize_t fp = mesh->face_ptr[k];
            const hsize_t fc = mesh->face_counts[k];
            for (hsize_t i = 0; i < fc; i++) {
                neighbor_tmp[out_local + i] = mesh->neighbor_cell[fp + i];
                area_tmp[out_local + i]     = mesh->face_area[fp + i];
            }
#ifdef MOVING_MESH
            for (hsize_t i = 0; i < fc * (DIMENSION - 1); i++)
                fmid_tmp[out_local * (DIMENSION - 1) + i] = mesh->f_mid_local[fp * (DIMENSION - 1) + i];
#endif
            mesh->face_ptr[k] = starting_offset + out_local;
            out_local += fc;
        }

        // pass 2: write scratch back into mesh arrays starting at starting_offset
        for (hsize_t i = 0; i < out_local; i++) {
            mesh->neighbor_cell[starting_offset + i] = neighbor_tmp[i];
            mesh->face_area[starting_offset + i]     = area_tmp[i];
        }
#ifdef MOVING_MESH
        for (hsize_t i = 0; i < out_local * (DIMENSION - 1); i++)
            mesh->f_mid_local[starting_offset * (DIMENSION - 1) + i] = fmid_tmp[i];
#endif

        mesh->num_faces = starting_offset + out_local;
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
