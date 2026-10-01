
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

        // a built cell kept until it is written; its status lives next to it, the cell points at it
        template <typename CellT> struct KeptCell {
            Status status = success;
            CellT  cell;
            KeptCell(int seed_id, double* pts) : cell(seed_id, pts, &status) {}
        };

        // what the build without a moved seed gave for one cell; no cell means the kick ladder
        struct FirstPass {
            std::unique_ptr<KeptCell<BigConvexCell>> cell;
            std::vector<std::pair<double, int>>      start;               // kept for the kick ladder only
            bool                                     overflowed  = false; // a GPU tier ran out of slots
            Status                                   last_status = success;
        };
    } // namespace

    static int             count_failed_and_prefetch_status(VMesh* mesh);
    static CellSids        build_cell_sids_for(const VMesh* mesh, const std::vector<int>& target_ks);
    static FallbackOutcome rebuild_cell_with_perturb_retry(
        VMesh* mesh, int k, double* d_stored_points, CellSids& cell_sids, double dt, Status& last_status_out);
    static void            first_pass(const VMesh* mesh, int k, double* d_stored_points, FirstPass& r);
    static FallbackOutcome finish_cell(VMesh*     mesh,
                                       int        k,
                                       double*    d_stored_points,
                                       CellSids&  cell_sids,
                                       double     dt,
                                       FirstPass& r,
                                       Status&    last_status_out);
    template <typename CellT>
    static bool                                     clip_and_walk(CellT&                                     cell,
                                                                  const Status&                              status,
                                                                  double*                                    d_stored_points,
                                                                  const std::vector<std::pair<double, int>>& start,
                                                                  int                                        seed_id,
                                                                  const knn_problem*                         knn,
                                                                  Status&                                    last_status_out);
    template <typename CellT> static void           commit_cell(VMesh* mesh, int k, const CellT& cell);
    static bool                                     cell_is_covered(const VMesh* mesh, int k, double4_t seed);
    static void                                     count_if_uncovered(const VMesh* mesh, int k, double4_t seed);
    static std::unique_ptr<KeptCell<BigConvexCell>> build_kept(const VMesh* mesh,
                                                               int          seed_id,
                                                               double*      d_stored_points,
                                                               const std::vector<std::pair<double, int>>& start,
                                                               Status& last_status_out);
    static bool                                     try_build_cell(VMesh*                                     mesh,
                                                                   int                                        k,
                                                                   int                                        seed_id,
                                                                   double*                                    d_stored_points,
                                                                   const std::vector<std::pair<double, int>>& start,
                                                                   Status&                                    last_status_out);
    static bool    first_pass_still_valid(const FirstPass& r, const std::vector<double4_t>& moved);
    static double3 compute_perturbation_delta(int seed_id, int attempt, double scale);
    static void    apply_perturbation(knn_problem*     knn,
                                      double*          d_stored_points,
                                      double3          delta,
                                      const int*       sids,
                                      size_t           n_sids,
                                      const double4_t* orig_positions);
    static void    rewind_perturbation(
           knn_problem* knn, double* d_stored_points, const int* sids, size_t n_sids, const double4_t* orig_positions);
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

    // a cell whose sphere holds points this rank did not ask for may miss a neighbour, so the run stops
    static void stop_if_uncertified(const char* who) {
        if (s_uncertified_rebuilds == 0) return;
        proteus_mpi::exit_failure("[rank %d] VORONOI: %d cell(s) built by the %s reach past the ball they asked "
                                  "for, the first at (%g, %g, %g). They may miss a neighbour. Aborting.\n",
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

        // every attempt without a moved seed only reads the points, so the cells run side by side
        std::vector<FirstPass> passes(failed_ks.size());
        cpu_for_dynamic(failed_ks.size(),
                        [&](size_t i) { first_pass(mesh, failed_ks[i], d_stored_points, passes[i]); });

        // written in cell order; a cell that may see a point an earlier kick moved is built again here
        std::vector<int>       perturbed_ks;
        std::vector<double4_t> moved; // old and new position of every point a kick moved
        for (size_t i = 0; i < failed_ks.size(); i++) {
            const int  k = failed_ks[i];
            FirstPass& r = passes[i];
            if (!first_pass_still_valid(r, moved)) {
                r = FirstPass();
                first_pass(mesh, k, d_stored_points, r);
            }

            std::vector<double4_t> before;
            if (!r.cell) {
                cell_sids.ensure_built_for(k);
                for (int s = 0; s < cell_sids.size_for(k); s++)
                    before.push_back(point_from_ptr(d_stored_points + DIMENSION * cell_sids.begin_for(k)[s]));
            }

            Status last_status = success;
            switch (finish_cell(mesh, k, d_stored_points, cell_sids, dt, r, last_status)) {
            case FallbackOutcome::ok_unchanged:
                break;
            case FallbackOutcome::ok_perturbed:
                perturbed_ks.push_back(k);
                if (perturbed_ks_out) perturbed_ks_out->push_back(k);
                for (int s = 0; s < cell_sids.size_for(k); s++) {
                    moved.push_back(before[s]);
                    moved.push_back(point_from_ptr(d_stored_points + DIMENSION * cell_sids.begin_for(k)[s]));
                }
                break;
            case FallbackOutcome::failed:
                proteus_mpi::exit_failure("VORONOI: cell %d all fallback attempts FAILED, aborting.\n", (int)k);
            }
            r = FirstPass();
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

    static bool is_overflow(Status s) {
        return s == vertex_overflow || s == triangle_overflow;
    }

    // build with the seed moved, the step ten times larger each try
    static FallbackOutcome run_perturb_ladder(VMesh*                                     mesh,
                                              int                                        k,
                                              int                                        seed_id,
                                              double*                                    d_stored_points,
                                              const std::vector<std::pair<double, int>>& start,
                                              const int*                                 sids,
                                              size_t                                     n_sids,
                                              const double4_t*                           orig_positions,
                                              double                                     dt,
                                              Status&                                    last_status_out) {
        constexpr int max_perturb = 12;
        double        scale       = 1e-13;
        for (int attempt = 1; attempt <= max_perturb; attempt++) {
            scale *= 10.0;
            const double3 delta = compute_perturbation_delta(seed_id, attempt, scale);
            apply_perturbation(mesh->knn, d_stored_points, delta, sids, n_sids, orig_positions);

            Status attempt_status = success;
            if (try_build_cell(mesh, k, seed_id, d_stored_points, start, attempt_status)) {
#ifdef MOVING_MESH
                apply_vmesh_perturbation_correction(mesh, k, delta, dt);
#else
                (void)dt;
#endif
                return FallbackOutcome::ok_perturbed;
            }

            last_status_out = attempt_status;
            rewind_perturbation(mesh->knn, d_stored_points, sids, n_sids, orig_positions);
        }
        return FallbackOutcome::failed;
    }

    // one cell through the ladder: every unperturbed build first, a moved seed only after all of them
    static FallbackOutcome rebuild_cell_with_perturb_retry(
        VMesh* mesh, int k, double* d_stored_points, CellSids& cell_sids, double dt, Status& last_status_out) {
        FirstPass r;
        first_pass(mesh, k, d_stored_points, r);
        return finish_cell(mesh, k, d_stored_points, cell_sids, dt, r, last_status_out);
    }

    // the build that moves no seed: the slow tier's start list, then the walk, on the wide slots;
    // it only reads the points and the mesh
    static void first_pass(const VMesh* mesh, int k, double* d_stored_points, FirstPass& r) {
        const int seed_id = (int)mesh->real_sorted_ids[k];
        r.start           = knn::nearest_on_host(mesh->knn, seed_id, _K_);
        r.overflowed      = is_overflow(mesh->cell_status[k]);
        r.cell            = build_kept(mesh, seed_id, d_stored_points, r.start, r.last_status);
        if (!r.cell) return;
        r.start.clear();
        r.start.shrink_to_fit();
    }

    // a failed cell built the way the fallback will build it, to see how far it reaches; one that only a kick
    // can build gets twice its first guess
    void fallback_needs(VMesh* mesh, std::vector<int>* cells, std::vector<double>* need_d2) {
        cells->clear();
        need_d2->clear();
        const int     n_hydro  = (int)mesh->n_hydro;
        const Status* stat     = mesh->cell_status;
        auto          failed   = [=] HD(int k) { return stat[k] != success && stat[k] != security_radius_beyond_data; };
        const int     n_failed = parallel_reduce_sum<_MESH_BLOCK_SIZE_, int>(
            "FAILED", n_hydro, [=] HD(size_t k) { return failed((int)k) ? 1 : 0; });
        if (n_failed == 0) return;

        // the list of them, picked on the device
        unsigned int* flags = mesh->scan_flags;
        parallel_for<_MESH_BLOCK_SIZE_>("FAILED_FLAG", n_hydro, [=] HD(int k) { flags[k] = failed(k) ? 1u : 0u; });
        parallel_exclusive_scan<_MESH_BLOCK_SIZE_>("FAILED_SCAN", (size_t)n_hydro, flags, flags, mesh->scan_scratch);
        static int* s_failed     = nullptr;
        static int  s_failed_cap = 0;
        if (n_failed > s_failed_cap) {
            if (s_failed) gpu_free(s_failed);
            s_failed_cap = std::max(n_failed, 2 * s_failed_cap);
            s_failed     = gpu_alloc<int>(s_failed_cap);
        }
        int* list = s_failed;
        parallel_for<_MESH_BLOCK_SIZE_>("FAILED_SCATTER", n_hydro, [=] HD(int k) {
            if (failed(k)) list[flags[k]] = k;
        });
        cells->assign(s_failed, s_failed + n_failed);

        PROFILE("FALLBACK_NEEDS");
        double*             d_stored_points = (double*)mesh->knn->d_stored_points;
        std::vector<double> need(cells->size(), -1.0);
        cpu_for_dynamic(cells->size(), [&](size_t i) {
            const int k = (*cells)[i];
            FirstPass r;
            first_pass(mesh, k, d_stored_points, r);
            double d2 = 4.0 * mesh->est_r[k] * mesh->est_r[k];
            if (r.cell) {
                double r2_num, r2_denom;
                r.cell->cell.max_vertex_r2_ratio(&r2_num, &r2_denom);
                d2 = (r2_denom > 0.0) ? 4.0 * r2_num / r2_denom : 1e30;
            }
            const POINT_TYPE& c = mesh->scratch_move[k];
            if (sphere_is_covered(c,
                                  c,
                                  d2,
                                  mesh->req_r2[k],
                                  proteus_mpi::decomp.cuts,
                                  proteus_mpi::decomp.nranks,
                                  proteus_mpi::decomp.rank))
                return;
            need[i] = d2;
        });

        std::vector<int> open;
        for (size_t i = 0; i < cells->size(); i++) {
            if (need[i] < 0.0) continue;
            open.push_back((*cells)[i]);
            need_d2->push_back(need[i]);
        }
        cells->swap(open);
    }

    // writes what the first pass built, or moves the seed until the cell builds
    static FallbackOutcome finish_cell(VMesh*     mesh,
                                       int        k,
                                       double*    d_stored_points,
                                       CellSids&  cell_sids,
                                       double     dt,
                                       FirstPass& r,
                                       Status&    last_status_out) {
        if (r.cell) {
            commit_cell(mesh, k, r.cell->cell);
            // every cell here gets the wide slots, count those that needed them
            if (r.overflowed) s_wide_tier_rebuilds++;
            return FallbackOutcome::ok_unchanged;
        }

        // what is left is degenerate, a small move of the seed can fix that
        const int seed_id = (int)mesh->real_sorted_ids[k];
        last_status_out   = r.last_status;
        cell_sids.ensure_built_for(k);
        const int*   sids   = cell_sids.begin_for(k);
        const size_t n_sids = (size_t)cell_sids.size_for(k);

        std::vector<double4_t> orig_positions(n_sids);
        for (size_t i = 0; i < n_sids; i++)
            orig_positions[i] = point_from_ptr(d_stored_points + DIMENSION * sids[i]);

        return run_perturb_ladder(
            mesh, k, seed_id, d_stored_points, r.start, sids, n_sids, orig_positions.data(), dt, last_status_out);
    }

    // clips the start list until the cell is closed, the walk adds the points it was missing; false if it failed
    template <typename CellT>
    static bool clip_and_walk(CellT&                                     cell,
                              const Status&                              status,
                              double*                                    d_stored_points,
                              const std::vector<std::pair<double, int>>& start,
                              int                                        seed_id,
                              const knn_problem*                         knn,
                              Status&                                    last_status_out) {
        bool closed = false;
        for (size_t di = 0; di < start.size(); di++) {
            const int j = start[di].second;
            cell.clip_by_plane(j);
            if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * j))) {
                closed = true;
                break;
            }
            if (status != success) break;
        }
        if (status == success && !closed) close_cell_by_tree_walk(cell, seed_id, knn);
        if (status != success) {
            last_status_out = status;
            return false;
        }
        return true;
    }

    // writes a finished cell into the mesh; a cell past the points of this rank is counted
    template <typename CellT> static void commit_cell(VMesh* mesh, int k, const CellT& cell) {
        double r2_num, r2_denom;
        cell.max_vertex_r2_ratio(&r2_num, &r2_denom);
        store_security_d2(mesh, (uint64_t)k, r2_num, r2_denom);
        count_if_uncovered(mesh, k, cell.voro_seed);

        write_cell_to_mesh(mesh, k, cell);
        mesh->cell_status[k] = success;
    }

    // a cell is covered if this rank asked for a ball that holds its sphere, or has all of it anyway; the seed
    // may have been moved by a kick since the ball went out
    static bool cell_is_covered(const VMesh* mesh, int k, double4_t seed) {
        POINT_TYPE s;
        s.x = seed.x;
        s.y = seed.y;
#ifdef dim_3D
        s.z = seed.z;
#endif
        return sphere_is_covered(s,
                                 mesh->scratch_move[k],
                                 mesh->security_d2[k],
                                 mesh->req_r2[k],
                                 proteus_mpi::decomp.cuts,
                                 proteus_mpi::decomp.nranks,
                                 proteus_mpi::decomp.rank);
    }

    static void count_if_uncovered(const VMesh* mesh, int k, double4_t seed) {
        if (cell_is_covered(mesh, k, seed)) return;
        if (s_uncertified_rebuilds == 0) s_first_uncertified = {seed.x, seed.y, seed.z};
        s_uncertified_rebuilds++;
    }

    // builds a cell that outlives the call, or nothing
    static std::unique_ptr<KeptCell<BigConvexCell>> build_kept(const VMesh* mesh,
                                                               int          seed_id,
                                                               double*      d_stored_points,
                                                               const std::vector<std::pair<double, int>>& start,
                                                               Status& last_status_out) {
        std::unique_ptr<KeptCell<BigConvexCell>> kept(new KeptCell<BigConvexCell>(seed_id, d_stored_points));
        if (!clip_and_walk(kept->cell, kept->status, d_stored_points, start, seed_id, mesh->knn, last_status_out))
            return nullptr;
        return kept;
    }

    // builds the cell and writes it if it came out complete
    static bool try_build_cell(VMesh*                                     mesh,
                               int                                        k,
                               int                                        seed_id,
                               double*                                    d_stored_points,
                               const std::vector<std::pair<double, int>>& start,
                               Status&                                    last_status_out) {
        auto kept = build_kept(mesh, seed_id, d_stored_points, start, last_status_out);
        if (!kept) return false;
        commit_cell(mesh, k, kept->cell);
        return true;
    }

    // a closed cell stays right if no point a kick moved, before or after, is inside its security sphere
    static bool first_pass_still_valid(const FirstPass& r, const std::vector<double4_t>& moved) {
        if (moved.empty()) return true;
        if (!r.cell) return false;

        double r2_num, r2_denom;
        r.cell->cell.max_vertex_r2_ratio(&r2_num, &r2_denom);
        const double4_t seed = r.cell->cell.voro_seed;

        // the security test of the cell, with a margin for rounding
        for (const double4_t& m : moved) {
            const double dx = m.x - seed.x;
            const double dy = m.y - seed.y;
            const double dz = m.z - seed.z;
            if (!((dx * dx + dy * dy + dz * dz) * r2_denom > 4.0 * r2_num * (1.0 + 1e-9))) return false;
        }
        return true;
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
    static void apply_perturbation(knn_problem*     knn,
                                   double*          d_stored_points,
                                   double3          delta,
                                   const int*       sids,
                                   size_t           n_sids,
                                   const double4_t* orig_positions) {
        for (size_t i = 0; i < n_sids; i++) {
            const int sid                        = sids[i];
            d_stored_points[DIMENSION * sid + 0] = orig_positions[i].x + delta.x;
            d_stored_points[DIMENSION * sid + 1] = orig_positions[i].y + delta.y;
#ifdef dim_3D
            d_stored_points[DIMENSION * sid + 2] = orig_positions[i].z + delta.z;
#endif
            knn::point_moved(knn, sid);
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

    static void rewind_perturbation(
        knn_problem* knn, double* d_stored_points, const int* sids, size_t n_sids, const double4_t* orig_positions) {
        for (size_t i = 0; i < n_sids; i++) {
            const int sid                        = sids[i];
            d_stored_points[DIMENSION * sid + 0] = orig_positions[i].x;
            d_stored_points[DIMENSION * sid + 1] = orig_positions[i].y;
#ifdef dim_3D
            d_stored_points[DIMENSION * sid + 2] = orig_positions[i].z;
#endif
            knn::point_moved(knn, sid);
        }
    }

    // unused slots become wall faces of zero area
    static void retire_face_range(VMesh* mesh, uint64_t first, uint64_t count) {
        for (uint64_t i = first; i < first + count; i++) {
            mesh->neighbor_cell[i] = -1;
            mesh->face_area[i]     = 0.0;
            for (int c = 0; c < DIMENSION - 1; c++)
                mesh->f_mid_local[i * (DIMENSION - 1) + c] = 0.0;
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
                for (int c = 0; c < DIMENSION - 1; c++)
                    mesh->f_mid_local[(fp_old + i) * (DIMENSION - 1) + c] =
                        mesh->f_mid_local[(fp_new + i) * (DIMENSION - 1) + c];
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
                compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_, uchar, VERT_TYPE, true>(
                    kn, seed_id, d_stored_points, mesh->knn, mesh->cell_status, mesh, &face_offset, &overflow);
                if (overflow) {
                    proteus_mpi::exit_failure("VORONOI: face overflow during symmetry rebuild — increase "
                                              "_FACE_CAPACITY_MULT_ in Config.sh.\n");
                }
                if (mesh->cell_status[kn] == success) {
                    reclaim_appended_slice(mesh, kn, fp_old, fc_old, &face_offset, off_before);
                    count_if_uncovered(mesh, kn, point_from_ptr(d_stored_points + DIMENSION * seed_id));
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
        POINT_TYPE gp;
        gp.x = g_old.x;
        gp.y = g_old.y;
#ifdef dim_3D
        gp.z = g_old.z;
#endif

        std::vector<int> near;
        knn::points_within_on_host(mesh->knn, gp, search_l2, &near);
        for (const int sid : near) {
            const int k = (int)mesh->sid_to_neighbor[sid];
            if (k >= (int)mesh->n_hydro) continue;
            if (affected->count(k)) continue;

            const double3 s   = mesh->seeds[k];
            const double  dox = s.x - g_old.x, doy = s.y - g_old.y, doz = s.z - g_old.z;
            const double  dnx = s.x - g_new.x, dny = s.y - g_new.y, dnz = s.z - g_new.z;
            const double  d2o = dox * dox + doy * doy + doz * doz;
            const double  d2n = dnx * dnx + dny * dny + dnz * dnz;
            if (d2o <= mesh->security_d2[k] || d2n <= mesh->security_d2[k]) affected->insert(k);
        }
    }

    // takes the new position of ghost seeds another rank moved and rebuilds around them
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
            knn::point_moved(mesh->knn, sid);
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
