
// CPU rebuild of the cells no GPU tier could finish, with exact tests (internal.h)

namespace voronoi {

    namespace {
        // a cell built on the CPU; its status lives next to it, the cell points at it
        struct ExactCell {
            Status        status = success;
            BigConvexCell cell;
            ExactCell(int seed_id, double* pts) : cell(seed_id, pts, &status, true) {}
        };
    } // namespace

    // cells built at once before they are written, about 32 KB each
    constexpr size_t FALLBACK_CHUNK = 1024;

    static int s_cpu_built = 0; // cells written by the CPU since the last take_cpu_built()

    // cell k from its K nearest points, then the walk for what they did not close, on the wide slots
    static std::unique_ptr<ExactCell> build_exact(const VMesh* mesh, int k) {
        double*                                   pts     = (double*)mesh->knn->d_stored_points;
        const int                                 seed_id = (int)mesh->real_sorted_ids[k];
        const std::vector<std::pair<double, int>> start   = knn::nearest_on_host(mesh->knn, seed_id, _K_);

        std::unique_ptr<ExactCell> c(new ExactCell(seed_id, pts));
        bool                       closed = false;
        for (size_t i = 0; i < start.size(); i++) {
            const int j = start[i].second;
            c->cell.clip_by_plane(j);
            if (c->cell.is_security_radius_reached(point_from_ptr(pts + DIMENSION * j))) {
                closed = true;
                break;
            }
            if (c->status != success) break;
        }
        if (c->status == success && !closed) close_cell_by_tree_walk(c->cell, seed_id, mesh->knn);
        return c;
    }

    // what is left of a block of face slots becomes wall faces of zero area
    static void retire_face_range(VMesh* mesh, uint64_t first, uint64_t count) {
        for (uint64_t i = first; i < first + count; i++) {
            mesh->neighbor_cell[i] = -1;
            mesh->face_area[i]     = 0.0;
            for (int c = 0; c < DIMENSION - 1; c++)
                mesh->f_mid_local[i * (DIMENSION - 1) + c] = 0.0;
        }
    }

    // no tier is left after this one
    static void stop_for_failed_cell(int k, const ExactCell& c) {
        const double4_t s = c.cell.voro_seed;
        if (c.status == coincident_points) {
            proteus_mpi::exit_failure("VORONOI: cell %d at (%g, %g, %g) shares its position with another point. "
                                      "Two seeds may not sit at the same place.\n",
                                      k,
                                      s.x,
                                      s.y,
                                      s.z);
        }
        proteus_mpi::exit_failure("VORONOI: cell %d at (%g, %g, %g) failed on the CPU too, status %d. Raise "
                                  "_BIG_MAX_P_ / _BIG_MAX_T_ in Config.sh if it ran out of slots.\n",
                                  k,
                                  s.x,
                                  s.y,
                                  s.z,
                                  (int)c.status);
    }

    // builds the cells of a chunk side by side. A cell whose sphere this rank covers is final, no later point
    // can reach it, so it is written, in cell order behind the faces so far; need_d2 is -1 for it, and the
    // squared sphere radius for the others
    static void build_chunk(VMesh* mesh, const int* ks, size_t n, double* need_d2) {
        std::vector<std::unique_ptr<ExactCell>> built(n);
        std::vector<uint64_t>                   n_faces(n, 0);
        cpu_for_dynamic(n, [&](size_t i) {
            const int k = ks[i];
            built[i]    = build_exact(mesh, k);
            if (built[i]->status != success) return;
            double r2_num, r2_denom;
            built[i]->cell.max_vertex_r2_ratio(&r2_num, &r2_denom);
            const double d2 = security_d2_of(r2_num, r2_denom);
            if (sphere_is_covered(mesh->scratch_move[k],
                                  d2,
                                  mesh->req_r2[k],
                                  proteus_mpi::decomp.cuts,
                                  proteus_mpi::decomp.nranks,
                                  proteus_mpi::decomp.rank)) {
                need_d2[i] = -1.0;
                n_faces[i] = (uint64_t)count_cell_faces(built[i]->cell);
            } else {
                need_d2[i] = d2;
            }
        });
        for (size_t i = 0; i < n; i++) {
            if (built[i]->status != success) stop_for_failed_cell(ks[i], *built[i]);
        }

        uint64_t total = 0;
        for (size_t i = 0; i < n; i++)
            total += n_faces[i];
        ensure_face_capacity(mesh, mesh->num_faces + total);
        for (size_t i = 0; i < n; i++) {
            if (need_d2[i] >= 0.0) continue;
            mesh->face_ptr[ks[i]] = mesh->num_faces;
            mesh->num_faces += n_faces[i];
            s_cpu_built++;
        }

        cpu_for_dynamic(n, [&](size_t i) {
            if (need_d2[i] >= 0.0) return;
            const int            k    = ks[i];
            const BigConvexCell& cell = built[i]->cell;
            double               r2_num, r2_denom;
            cell.max_vertex_r2_ratio(&r2_num, &r2_denom);
            store_security_d2(mesh, (uint64_t)k, r2_num, r2_denom);
            const uint64_t written = extract_cell_all(cell, mesh, (uint64_t)k);
            mesh->face_counts[k]   = written;
            retire_face_range(mesh, mesh->face_ptr[k] + written, n_faces[i] - written);
            mesh->cell_status[k] = success;
        });
    }

    int take_cpu_built() {
        const int n = s_cpu_built;
        s_cpu_built = 0;
        return n;
    }

    // every cell no GPU tier could build is built on the CPU; the final ones are written, the others say how
    // far they reach
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
        int* list = mesh->cell_list;
        parallel_for<_MESH_BLOCK_SIZE_>("FAILED_SCATTER", n_hydro, [=] HD(int k) {
            if (failed(k)) list[flags[k]] = k;
        });
        const std::vector<int> ks(list, list + n_failed);

        PROFILE("FALLBACK");
        std::vector<double> need(ks.size());
        for (size_t first = 0; first < ks.size(); first += FALLBACK_CHUNK) {
            build_chunk(mesh, ks.data() + first, std::min(FALLBACK_CHUNK, ks.size() - first), need.data() + first);
        }
        for (size_t i = 0; i < ks.size(); i++) {
            if (need[i] < 0.0) continue;
            cells->push_back(ks[i]);
            need_d2->push_back(need[i]);
        }
    }

} // namespace voronoi
