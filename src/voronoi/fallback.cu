namespace voronoi {

    // ---- file-local types ----
    namespace {
        enum class FallbackOutcome { ok_unchanged, ok_perturbed, failed };

        // sid lookup: cell k -> list of sids that touch it. The flat layout keeps the
        // inner perturb-rebuild loops cache-friendly.
        struct CellSids {
            std::vector<int> offsets; // size n_hydro + 1
            std::vector<int> flat;    // size offsets.back()

            int        size_for(int k) const { return offsets[k + 1] - offsets[k]; }
            const int* begin_for(int k) const { return flat.data() + offsets[k]; }
        };
    } // namespace

    // ---- forward declarations ----
    static int             count_failed_and_prefetch_status(VMesh* mesh);
    static CellSids        build_cell_sids(const VMesh* mesh);
    static FallbackOutcome rebuild_cell_with_perturb_retry(VMesh* mesh, int k, double* d_stored_points,
                                                           const CellSids& cell_sids);
    static std::vector<std::pair<double, int>> sort_neighbours_by_distance(double* d_stored_points, int seed_id,
                                                                           int n_seeds);
    static bool                                try_build_cell_from_neighbours(
                                       VMesh* mesh, int k, int seed_id, double* d_stored_points,
                                       const std::vector<std::pair<double, int>>& sorted);
    static void                                apply_perturbation(double* d_stored_points, int seed_id, int attempt,
                                                                  double scale, const int* sids, size_t n_sids,
                                                                  const double4* orig_positions);
    static void                                rewind_perturbation(double* d_stored_points, const int* sids, size_t n_sids,
                                                                   const double4* orig_positions);
    static void                                append_cell_to_mesh(VMesh* mesh, int k, const ConvexCell& cell);
    static std::vector<int>                    collect_unique_neighbors(const VMesh* mesh, const std::vector<int>& sources);
    static void                                run_symmetry_pass(VMesh* mesh, double* d_stored_points,
                                                                 const CellSids&         cell_sids,
                                                                 const std::vector<int>& initial_perturbed);
    static void                                compact_face_arrays(VMesh* mesh);

#ifndef CPU_DEBUG
    GLOBAL void kernel_count_failures(int n, const Status* stat, int* fail_count);
    static int* d_fail_count = nullptr;
#endif

    // ============================================================
    // Main routines
    // ============================================================

    // retry cells that failed both GPU tiers with hash-based seed perturbation;
    // perturbed cells trigger a symmetry pass to rebuild affected neighbours
    int cpu_fallback_failed_cells(VMesh* mesh) {
        Status* stat            = mesh->cell_status;
        double* d_stored_points = (double*)mesh->knn->d_stored_points;

        // count failed cells (and pull cell_status to host on GPU)
        const int num_failed = count_failed_and_prefetch_status(mesh);
        if (num_failed == 0) return 0;
        std::cout << "VORONOI: " << num_failed << " cells failed, retrying with fallback..." << std::endl;

        // perturb-retry each failed cell; track which ones got permanently perturbed
        const CellSids   cell_sids = build_cell_sids(mesh);
        std::vector<int> perturbed_ks;
        const int        n_hydro = (int)mesh->n_hydro;

        for (int k = 0; k < n_hydro; k++) {
            if (stat[k] == success) continue;

            // refuse to recover from statuses other than security_radius / needs_exact_predicates
            const Status original = stat[k];
            if (original != security_radius_not_reached && original != needs_exact_predicates) {
                proteus_mpi::exit_failure("VORONOI: cell %d failed with unrecoverable status: %d\n", (int)k,
                                          (int)original);
            }
            std::cout << "VORONOI: cell " << k << " failed with status: " << original << std::endl;

            // try the perturb-retry ladder; record cells that stuck at non-zero perturb
            switch (rebuild_cell_with_perturb_retry(mesh, k, d_stored_points, cell_sids)) {
            case FallbackOutcome::ok_unchanged: break;
            case FallbackOutcome::ok_perturbed: perturbed_ks.push_back(k); break;
            case FallbackOutcome::failed:
                proteus_mpi::exit_failure("VORONOI: cell %d all fallback attempts FAILED, aborting.\n", (int)k);
            }
        }

        // if any cell stuck at a perturbation, rebuild neighbours + compact face data
        if (!perturbed_ks.empty()) {
            run_symmetry_pass(mesh, d_stored_points, cell_sids, perturbed_ks);
            compact_face_arrays(mesh);
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

    // build a flat sid-per-cell lookup (offsets + flat) for quick membership iteration
    static CellSids build_cell_sids(const VMesh* mesh) {
        const int n_hydro = (int)mesh->n_hydro;
        const int n_seeds = (int)mesh->n_seeds;

        // step 1: count sids per cell into offsets[k+1]
        CellSids cs;
        cs.offsets.assign(n_hydro + 1, 0);
        for (int sid = 0; sid < n_seeds; sid++) {
            const unsigned int k = mesh->sid_to_neighbor[sid];
            if ((int)k < n_hydro) cs.offsets[k + 1]++;
        }
        // step 2: prefix-sum offsets[] in place
        for (int k = 0; k < n_hydro; k++)
            cs.offsets[k + 1] += cs.offsets[k];
        cs.flat.resize(cs.offsets[n_hydro]);

        // step 3: scatter sids into flat[] using a per-cell write cursor
        std::vector<int> cursor(n_hydro, 0);
        for (int sid = 0; sid < n_seeds; sid++) {
            const unsigned int k = mesh->sid_to_neighbor[sid];
            if ((int)k < n_hydro) cs.flat[cs.offsets[k] + cursor[k]++] = sid;
        }
        return cs;
    }

    // try the unperturbed cell first, then perturb the seed with growing scale on each retry.
    // The exhaustive sorted-by-distance clip list is more robust than KNN's K-nearest sample
    // for stubborn degeneracies.
    static FallbackOutcome
    rebuild_cell_with_perturb_retry(VMesh* mesh, int k, double* d_stored_points, const CellSids& cell_sids) {
        const int    seed_id = (int)mesh->real_sorted_ids[k];
        const int*   sids    = cell_sids.begin_for(k);
        const size_t n_sids  = (size_t)cell_sids.size_for(k);

        // snapshot original positions so a failed perturb attempt can be rewound
        std::vector<double4> orig_positions(n_sids);
        for (size_t i = 0; i < n_sids; i++)
            orig_positions[i] = point_from_ptr(d_stored_points + DIMENSION * sids[i]);

        // build the sorted neighbour list once; reused across every perturb attempt
        const auto sorted = sort_neighbours_by_distance(d_stored_points, seed_id, (int)mesh->n_seeds);

        // ladder: attempt 0 unperturbed, then attempts 1..9 with 10x growing scale
        constexpr int max_perturb = 9;
        double        scale       = 1e-13;
        for (int attempt = 0; attempt <= max_perturb; attempt++) {
            if (attempt > 0)
                apply_perturbation(d_stored_points, seed_id, attempt, scale, sids, n_sids, orig_positions.data());

            if (try_build_cell_from_neighbours(mesh, k, seed_id, d_stored_points, sorted)) {
                std::cout << "VORONOI: cell " << k << " fallback succeeded (attempt " << attempt << ")." << std::endl;
                return (attempt == 0) ? FallbackOutcome::ok_unchanged : FallbackOutcome::ok_perturbed;
            }

            if (attempt > 0) rewind_perturbation(d_stored_points, sids, n_sids, orig_positions.data());
            scale *= 10.0;
        }
        return FallbackOutcome::failed;
    }

    // sort every other seed by distance² from seed_id; used as the exhaustive clip list
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

    // attempt one cell build by clipping against every other seed in distance order;
    // returns true and writes the cell into mesh on success
    static bool try_build_cell_from_neighbours(VMesh*                                     mesh,
                                               int                                        k,
                                               int                                        seed_id,
                                               double*                                    d_stored_points,
                                               const std::vector<std::pair<double, int>>& sorted) {
        Status     status = success;
        ConvexCell cell(seed_id, d_stored_points, &status, mesh->buff);

        // clip plane-by-plane in distance order until security radius is hit or status fails
        for (size_t di = 0; di < sorted.size(); di++) {
            const int j = sorted[di].second;
            cell.clip_by_plane(j);
            if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * j))) break;
            if (status != success) break;
        }
        if (status != success) return false;

        append_cell_to_mesh(mesh, k, cell);
        mesh->cell_status[k] = success;
        return true;
    }

    // hash-deterministic perturbation: each (seed_id, attempt) -> same dx/dy(/dz)
    static void apply_perturbation(double*        d_stored_points,
                                   int            seed_id,
                                   int            attempt,
                                   double         scale,
                                   const int*     sids,
                                   size_t         n_sids,
                                   const double4* orig_positions) {
        // mix seed_id + attempt into a 16-bit field used for each axis offset
        unsigned int hash = (unsigned int)(seed_id * 2654435761u + attempt * 40503u);
        hash              = hash * 1103515245u + 12345u;
        const double dx   = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
        hash              = hash * 1103515245u + 12345u;
        const double dy   = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
#ifdef dim_3D
        hash            = hash * 1103515245u + 12345u;
        const double dz = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
#endif
        // shift each sid that touches this cell by the same (dx, dy, dz)
        for (size_t i = 0; i < n_sids; i++) {
            const int sid                        = sids[i];
            d_stored_points[DIMENSION * sid + 0] = orig_positions[i].x + dx;
            d_stored_points[DIMENSION * sid + 1] = orig_positions[i].y + dy;
#ifdef dim_3D
            d_stored_points[DIMENSION * sid + 2] = orig_positions[i].z + dz;
#endif
        }
    }

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
        const int fc = count_cell_faces(cell);
        ensure_face_capacity(mesh, mesh->num_faces + fc);
        mesh->face_ptr[k]    = mesh->num_faces;
        mesh->face_counts[k] = (hsize_t)fc;
        extract_cell_all(cell, mesh, (hsize_t)k);
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
    // and need rebuilding. Cascade until no more neighbours change (or MAX_ROUNDS).
    static void run_symmetry_pass(VMesh*                  mesh,
                                  double*                 d_stored_points,
                                  const CellSids&         cell_sids,
                                  const std::vector<int>& initial_perturbed) {
        constexpr int MAX_ROUNDS = 4;

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
                compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_>(kn, seed_id, d_stored_points, mesh->knn,
                                                                   mesh->cell_status, mesh, &face_offset, &overflow);
                if (overflow) {
                    std::cerr << "VORONOI: face overflow during symmetry rebuild — "
                                 "increase _FACE_CAPACITY_MULT_ in Config.sh."
                              << std::endl;
                    exit(EXIT_FAILURE);
                }

                // KNN rebuild failed for this neighbour: fall through to the perturb path
                if (mesh->cell_status[kn] != success) {
                    mesh->num_faces         = (hsize_t)face_offset;
                    FallbackOutcome outcome = rebuild_cell_with_perturb_retry(mesh, kn, d_stored_points, cell_sids);
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
    // leaving gaps. Compact tightens everything into a contiguous prefix.
    static void compact_face_arrays(VMesh* mesh) {
        const hsize_t cap     = mesh->num_faces;
        const int     n_hydro = (int)mesh->n_hydro;

        // host-side scratch sized to the upper bound (current num_faces)
        std::vector<int>    neighbor_tmp(cap);
        std::vector<double> area_tmp(cap);
#ifdef MOVING_MESH
        std::vector<double> fmid_tmp(cap * (DIMENSION - 1));
#endif

        // walk cells in k-order, copy their faces into scratch, update face_ptr to compact offset
        hsize_t out = 0;
        for (int k = 0; k < n_hydro; k++) {
            const hsize_t fp = mesh->face_ptr[k];
            const hsize_t fc = mesh->face_counts[k];
            for (hsize_t i = 0; i < fc; i++) {
                neighbor_tmp[out + i] = mesh->neighbor_cell[fp + i];
                area_tmp[out + i]     = mesh->face_area[fp + i];
            }
#ifdef MOVING_MESH
            for (hsize_t i = 0; i < fc * (DIMENSION - 1); i++)
                fmid_tmp[out * (DIMENSION - 1) + i] = mesh->f_mid_local[fp * (DIMENSION - 1) + i];
#endif
            mesh->face_ptr[k] = out;
            out += fc;
        }

        // write compacted face data back into mesh's arrays
        for (hsize_t i = 0; i < out; i++) {
            mesh->neighbor_cell[i] = neighbor_tmp[i];
            mesh->face_area[i]     = area_tmp[i];
        }
#ifdef MOVING_MESH
        for (hsize_t i = 0; i < out * (DIMENSION - 1); i++)
            mesh->f_mid_local[i] = fmid_tmp[i];
#endif
        mesh->num_faces = out;
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
