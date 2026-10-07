// how far a cell reaches, and whether the points it needs are all here (included by voronoi.cu)

namespace voronoi {

    // the first guess of a cell's reach, in distances to its _FAST_K_-th nearest cell
    constexpr double FIRST_GUESS_FACTOR = 1.0;

    // a ball may not reach more than half the box, else it would meet its own periodic image
    constexpr double MAX_BALL_RADIUS = 0.49;

    // the sphere of 2 x the farthest corner holds every point that can cut the cell; covered when this rank
    // asked for all points in a ball around the seed that holds the sphere, or holds them itself
    HD inline bool
    sphere_is_covered(const POINT_TYPE& seed, double sec_d2, double req_r2, const uint64_t* cuts, int nranks, int me) {
        const double sec_r = sqrt(sec_d2);
        if (req_r2 > 0.0 && sec_r <= sqrt(req_r2)) return true;
        return proteus_mpi::ball_is_local(seed, sec_r, cuts, nranks, me);
    }

    // the mean cell spacing on this rank, from the share of the curve it holds
    static double mean_spacing_of_rank(int n) {
        const auto&  dc    = proteus_mpi::decomp;
        const double share = (double)(dc.cuts[dc.rank + 1] - dc.cuts[dc.rank]) / (double)knn::DOMAIN_KEY_END;
        return std::pow(share / (double)std::max(n, 1), 1.0 / (double)DIMENSION);
    }

    // a cell more than this many mean spacings from anything it does not hold needs no guess
    constexpr double LOCAL_SPACINGS = 4.0;

    // the cells with a radius above 0, in cell order, into the ball list; returns how many
    static int compact_balls(VMesh* mesh) {
        const int n = (int)mesh->n_hydro;
        if (n == 0) return 0;
        const double* r     = mesh->ball_r;
        int*          cell  = mesh->ball_cell;
        double*       rad   = mesh->ball_rad;
        unsigned int* flags = mesh->scan_flags;
        parallel_for<_MESH_BLOCK_SIZE_>("BALL_FLAG", n, [=] HD(int k) { flags[k] = (r[k] > 0.0) ? 1u : 0u; });
        const bool last = r[n - 1] > 0.0;
        parallel_exclusive_scan<_MESH_BLOCK_SIZE_>("BALL_SCAN", (size_t)n, flags, flags, mesh->scan_scratch);
        parallel_for<_MESH_BLOCK_SIZE_>("BALL_SCATTER", n, [=] HD(int k) {
            if (r[k] > 0.0) {
                cell[flags[k]] = k;
                rad[flags[k]]  = r[k];
            }
        });
        return (int)flags[n - 1] + (last ? 1 : 0);
    }

    // first guess per cell; the cells whose guess leaves what this rank holds ask for it
    void first_guess_balls(VMesh* mesh, const POINT_TYPE* cell_pos) {
        PROFILE("FIRST_GUESS");
        const int          n       = (int)mesh->n_hydro;
        const knn_problem* knn     = mesh->knn;
        double*            est_r   = mesh->est_r;
        double*            req_r2  = mesh->req_r2;
        const uint64_t*    cuts    = proteus_mpi::decomp.cuts;
        const int          nranks  = proteus_mpi::decomp.nranks;
        const int          me      = proteus_mpi::decomp.rank;
        const int*         sorted  = (const int*)mesh->real_sorted_ids;
        const double       r_local = std::fmin(LOCAL_SPACINGS * mean_spacing_of_rank(n), MAX_BALL_RADIUS);
        double*            ball_r  = mesh->ball_r;

        parallel_for<_VORO_BLOCK_SIZE_>("GUESS", n, [=] HD(int k) {
            // deep inside: the guess starts from that distance, nothing to ask
            if (proteus_mpi::ball_is_local(cell_pos[k], r_local, cuts, nranks, me)) {
                est_r[k]  = r_local;
                req_r2[k] = 0.0;
                ball_r[k] = 0.0;
                return;
            }
            unsigned int nb[_FAST_K_];
            const int    sid = sorted[k];
            knn::knn_for_point<_FAST_K_>(sid, knn, nb);
            double d2 = 0.0;
            for (int i = 0; i < _FAST_K_; i++) {
                const double e = knn::dist2_point(knn->d_stored_points[sid], knn->d_stored_points[nb[i]]);
                if (e > d2) d2 = e;
            }
            double r = FIRST_GUESS_FACTOR * sqrt(d2);
            if (!(r > 0.0) || r > MAX_BALL_RADIUS) r = MAX_BALL_RADIUS;
            est_r[k]  = r;
            req_r2[k] = proteus_mpi::ball_is_local(cell_pos[k], r, cuts, nranks, me) ? 0.0 : r * r;
            ball_r[k] = (req_r2[k] > 0.0) ? r : 0.0;
        });

        const int nb = compact_balls(mesh);
        proteus_mpi::halo_request_balls(mesh, cell_pos, mesh->ball_cell, mesh->ball_rad, nb);
    }

    // a finished cell whose sphere is not covered is left open, its sphere is what it needs next
    void certify_cells(VMesh* mesh, const POINT_TYPE* cell_pos) {
        PROFILE("CERTIFY");
        const int       n      = (int)mesh->n_hydro;
        Status*         stat   = mesh->cell_status;
        const double*   sec_d2 = mesh->security_d2;
        const double*   req_r2 = mesh->req_r2;
        const uint64_t* cuts   = proteus_mpi::decomp.cuts;
        const int       nranks = proteus_mpi::decomp.nranks;
        const int       me     = proteus_mpi::decomp.rank;

        parallel_for<_MESH_BLOCK_SIZE_>("CERTIFY", n, [=] HD(int k) {
            if (stat[k] != success) return;
            if (!sphere_is_covered(cell_pos[k], sec_d2[k], req_r2[k], cuts, nranks, me))
                stat[k] = security_radius_beyond_data;
        });
        GPU_SYNC();
    }

    // their points did not change, so a build would give the same cells; certify_cells looks at them again
    void reopen_uncertified_cells(VMesh* mesh) {
        Status* stat = mesh->cell_status;
        parallel_for<_MESH_BLOCK_SIZE_>("REOPEN_UNCERTIFIED", mesh->n_hydro, [=] HD(int k) {
            if (stat[k] == security_radius_beyond_data) stat[k] = success;
        });
    }

    // the next ball of an open cell: its sphere, but at most twice what it had, and never past half the box;
    // 0 if it cannot grow any more
    HD inline double next_ball(double req_r2, double est_r, double need_d2) {
        const double had = fmax(sqrt(req_r2), est_r);
        double       r   = fmin(sqrt(need_d2), 2.0 * had);
        if (r > MAX_BALL_RADIUS) r = MAX_BALL_RADIUS;
        return (r * r <= req_r2) ? 0.0 : r;
    }

    // the cells left open by the round ask for their next ball; collective. Returns how many cells ask, and in
    // stuck how many cannot grow any more
    int request_open_balls(VMesh* mesh, int* stuck) {
        PROFILE("OPEN_BALLS");
        const int     n      = (int)mesh->n_hydro;
        double*       need   = mesh->ball_r;
        const Status* stat   = mesh->cell_status;
        const double* sec_d2 = mesh->security_d2;
        parallel_for<_MESH_BLOCK_SIZE_>(
            "NEED", n, [=] HD(int k) { need[k] = (stat[k] == security_radius_beyond_data) ? sec_d2[k] : -1.0; });

        // the cells only the CPU can build are built now; those that reach past their ball say how far
        std::vector<int>    failed;
        std::vector<double> failed_need;
        fallback_needs(mesh, &failed, &failed_need);
        for (size_t i = 0; i < failed.size(); i++)
            need[failed[i]] = failed_need[i];

        double*       req_r2 = mesh->req_r2;
        const double* est_r  = mesh->est_r;
        parallel_for<_MESH_BLOCK_SIZE_>("NEXT", n, [=] HD(int k) {
            if (need[k] < 0.0) {
                need[k] = 0.0;
                return;
            }
            const double r = next_ball(req_r2[k], est_r[k], need[k]);
            if (r > 0.0) req_r2[k] = r * r;
            need[k] = (r > 0.0) ? r : -1.0;
        });
        *stuck = parallel_reduce_sum<_MESH_BLOCK_SIZE_, int>(
            "STUCK", n, [=] HD(size_t k) { return (need[k] < 0.0) ? 1 : 0; });

        const int nb = compact_balls(mesh);
        return nb;
    }

    // asks for the balls the last request_open_balls collected
    void send_open_balls(VMesh* mesh, const POINT_TYPE* cell_pos, int nb) {
        proteus_mpi::halo_request_balls(mesh, cell_pos, mesh->ball_cell, mesh->ball_rad, nb);
    }

} // namespace voronoi
