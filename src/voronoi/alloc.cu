
// allocates and frees the mesh (voronoi.h)

namespace voronoi {

    // a first guess of the ghosts: a few cell layers around the part of the box this rank holds; it grows
    static uint64_t ghost_guess(uint64_t n_cells) {
        const double n       = (double)std::max<uint64_t>(n_cells, 1);
        const double surface = (DIMENSION == 3) ? 6.0 * std::pow(n, 2.0 / 3.0) : 4.0 * std::sqrt(n);
        return std::max<uint64_t>(1024, (uint64_t)(8.0 * surface));
    }

    // allocates all mesh arrays, once at startup
    VMesh* allocate_mesh(uint64_t n_hydro) {
        const uint64_t ext        = (uint64_t)mpi::max_n_local((int)n_hydro);
        const uint64_t max_ghosts = ghost_guess(ext);
        const uint64_t total      = ext + max_ghosts;
        const uint64_t max_faces  = ext * _FACE_CAPACITY_MULT_;

        VMesh* mesh          = gpu_alloc<VMesh>(1);
        mesh->n_seeds        = 0;
        mesh->n_hydro        = n_hydro;
        mesh->num_faces      = 0;
        mesh->face_capacity  = max_faces;
        mesh->ghost_capacity = max_ghosts;
        mesh->total_capacity = total;

        // per cell
        mesh->seeds        = gpu_calloc<double3>(ext);
        mesh->com          = gpu_calloc<double3>(ext);
        mesh->volumes      = gpu_calloc<double>(ext);
        mesh->face_counts  = gpu_calloc<uint64_t>(ext);
        mesh->face_ptr     = gpu_calloc<uint64_t>(ext);
        mesh->min_egy_spec = sim.min_egy_spec;

        // mean cell volume, or the size from the param file
        double V_ref = 1.0 / (double)ic_data.header.n_global;
#ifdef VOL_REGULARIZE
        if (input.has_parameter("vol_ref_cell_size")) {
            const double vs = input.get_parameter_double("vol_ref_cell_size");
#ifdef dim_2D
            V_ref = vs * vs;
#else
            V_ref = vs * vs * vs;
#endif
        }
#endif
#ifdef dim_2D
        mesh->Ri_ref = sqrt(V_ref / PI);
#else
        mesh->Ri_ref = portable_cbrt(3.0 * V_ref / (4.0 * PI));
#endif

        mesh->cell_status   = gpu_alloc<Status>(ext);
        mesh->security_d2   = gpu_calloc<double>(ext);
        mesh->est_r         = gpu_calloc<double>(ext);
        mesh->req_r2        = gpu_calloc<double>(ext);
        mesh->ball_r        = gpu_calloc<double>(ext);
        mesh->ball_cell     = gpu_alloc<int>(ext);
        mesh->ball_rad      = gpu_alloc<double>(ext);
        mesh->cell_list     = gpu_alloc<int>(ext);
        mesh->face_offset   = gpu_calloc<unsigned long long>(1);
        mesh->overflow_flag = gpu_calloc<int>(1);
        mesh->n_mpi_ghosts  = 0;
#ifdef MOVING_MESH
        mesh->v_mesh = gpu_calloc<POINT_TYPE>(ext);
#endif

        // ghost arrays, none without MPI
        const int gc = mpi::n_mpi_capacity;
        if (gc > 0) {
            mesh->seeds_g   = gpu_alloc<double3>(gc);
            mesh->com_off_g = gpu_alloc<POINT_TYPE>(gc);
#ifdef VOL_REGULARIZE
            mesh->volumes_g = gpu_alloc<double>(gc);
#endif
#ifdef MOVING_MESH
            mesh->v_mesh_g = gpu_alloc<POINT_TYPE>(gc);
#endif
            gpu_advise_gpu_preferred(mesh->seeds_g, gc * sizeof(double3));
            gpu_advise_gpu_preferred(mesh->com_off_g, gc * sizeof(POINT_TYPE));
#ifdef MOVING_MESH
            gpu_advise_gpu_preferred(mesh->v_mesh_g, gc * sizeof(POINT_TYPE));
#endif
        } else {
            mesh->seeds_g   = nullptr;
            mesh->com_off_g = nullptr;
#ifdef VOL_REGULARIZE
            mesh->volumes_g = nullptr;
#endif
#ifdef MOVING_MESH
            mesh->v_mesh_g = nullptr;
#endif
        }

        // per face
        mesh->neighbor_cell = gpu_alloc<int>(max_faces);
        mesh->face_area     = gpu_alloc<double>(max_faces);
        mesh->f_mid_local   = gpu_alloc<double>(max_faces * (DIMENSION - 1));

        mesh->ghost_ids = gpu_alloc<uint64_t>(max_ghosts);

        // index maps and scratch
        mesh->real_sorted_ids = gpu_alloc<unsigned int>(ext);
        mesh->sid_to_neighbor = gpu_alloc<unsigned int>(total);
        mesh->gather_perm     = gpu_alloc<unsigned int>(ext);
        mesh->scan_flags      = gpu_alloc<unsigned int>(ext);
        mesh->scan_scratch    = gpu_alloc<unsigned int>(scan_scratch_size((size_t)ext, _MESH_BLOCK_SIZE_));

        mesh->scratch_double = gpu_alloc<double>(ext);
        mesh->scratch_point  = gpu_alloc<POINT_TYPE>(ext);

        mesh->scratch_pts  = gpu_alloc<POINT_TYPE>(total);
        mesh->scratch_move = gpu_alloc<POINT_TYPE>(ext);

        mesh->knn = knn::init_once((int)total);

        // keep the hot arrays on the device
        gpu_advise_gpu_preferred(mesh->seeds, ext * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->com, n_hydro * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->volumes, n_hydro * sizeof(double));
        gpu_advise_gpu_preferred(mesh->face_counts, n_hydro * sizeof(uint64_t));
        gpu_advise_gpu_preferred(mesh->face_ptr, n_hydro * sizeof(uint64_t));
        gpu_advise_gpu_preferred(mesh->cell_status, n_hydro * sizeof(Status));
        gpu_advise_gpu_preferred(mesh->security_d2, n_hydro * sizeof(double));
        gpu_advise_gpu_preferred(mesh->neighbor_cell, max_faces * sizeof(int));
        gpu_advise_gpu_preferred(mesh->face_area, max_faces * sizeof(double));
        gpu_advise_gpu_preferred(mesh->real_sorted_ids, n_hydro * sizeof(unsigned int));
        gpu_advise_gpu_preferred(mesh->sid_to_neighbor, total * sizeof(unsigned int));
        gpu_advise_gpu_preferred(mesh->gather_perm, n_hydro * sizeof(unsigned int));

        return mesh;
    }

    // frees what allocate_mesh took
    void free_mesh(VMesh* mesh) {
        if (!mesh) return;
        gpu_free(mesh->seeds);
        gpu_free(mesh->com);
        gpu_free(mesh->volumes);
        gpu_free(mesh->face_counts);
        gpu_free(mesh->face_ptr);
        gpu_free(mesh->cell_status);
        gpu_free(mesh->security_d2);
        gpu_free(mesh->est_r);
        gpu_free(mesh->req_r2);
        gpu_free(mesh->ball_r);
        gpu_free(mesh->ball_cell);
        gpu_free(mesh->ball_rad);
        gpu_free(mesh->cell_list);
        gpu_free(mesh->face_offset);
        gpu_free(mesh->overflow_flag);
#ifdef MOVING_MESH
        gpu_free(mesh->v_mesh);
#endif
        gpu_free(mesh->neighbor_cell);
        gpu_free(mesh->face_area);
        gpu_free(mesh->f_mid_local);
        gpu_free(mesh->ghost_ids);
        gpu_free(mesh->real_sorted_ids);
        gpu_free(mesh->sid_to_neighbor);
        gpu_free(mesh->gather_perm);
        gpu_free(mesh->scan_flags);
        gpu_free(mesh->scan_scratch);
        gpu_free(mesh->scratch_double);
        gpu_free(mesh->scratch_point);
        gpu_free(mesh->scratch_pts);
        gpu_free(mesh->scratch_move);
        if (mesh->seeds_g) gpu_free(mesh->seeds_g);
        if (mesh->com_off_g) gpu_free(mesh->com_off_g);
#ifdef VOL_REGULARIZE
        if (mesh->volumes_g) gpu_free(mesh->volumes_g);
#endif
#ifdef MOVING_MESH
        if (mesh->v_mesh_g) gpu_free(mesh->v_mesh_g);
#endif
        if (mesh->knn) { knn::knn_free(&mesh->knn); }
        gpu_free(mesh);
    }

    // new ghost arrays after n_mpi_capacity grew, old content dropped
    void mesh_grow_ghosts(VMesh* mesh, int new_cap) {
        if (mesh->seeds_g) gpu_free(mesh->seeds_g);
        mesh->seeds_g = (new_cap > 0) ? gpu_alloc<double3>(new_cap) : nullptr;
        if (mesh->com_off_g) gpu_free(mesh->com_off_g);
        mesh->com_off_g = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
#ifdef VOL_REGULARIZE
        if (mesh->volumes_g) gpu_free(mesh->volumes_g);
        mesh->volumes_g = (new_cap > 0) ? gpu_alloc<double>(new_cap) : nullptr;
#endif
#ifdef MOVING_MESH
        if (mesh->v_mesh_g) gpu_free(mesh->v_mesh_g);
        mesh->v_mesh_g = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
#endif
    }

    // room for this many ghosts behind the cells, at least double, content kept
    void mesh_ensure_ghost_capacity(VMesh* mesh, uint64_t n_ghosts) {
        if (n_ghosts <= mesh->ghost_capacity) return;
        const uint64_t ext            = mesh->total_capacity - mesh->ghost_capacity;
        const uint64_t new_max_ghosts = std::max(n_ghosts, 2 * mesh->ghost_capacity);
        const uint64_t new_total      = ext + new_max_ghosts;
        const uint64_t old_total      = mesh->total_capacity;

        POINT_TYPE* new_pts = gpu_alloc<POINT_TYPE>(new_total);
        gpu_memcpy(new_pts, mesh->scratch_pts, (size_t)old_total * sizeof(POINT_TYPE));
        gpu_free(mesh->scratch_pts);
        mesh->scratch_pts = new_pts;

        uint64_t* new_gids = gpu_alloc<uint64_t>(new_max_ghosts);
        gpu_memcpy(new_gids, mesh->ghost_ids, (size_t)mesh->ghost_capacity * sizeof(uint64_t));
        gpu_free(mesh->ghost_ids);
        mesh->ghost_ids = new_gids;

        unsigned int* new_s2n = gpu_alloc<unsigned int>(new_total);
        gpu_memcpy(new_s2n, mesh->sid_to_neighbor, (size_t)old_total * sizeof(unsigned int));
        gpu_free(mesh->sid_to_neighbor);
        mesh->sid_to_neighbor = new_s2n;

        mesh->ghost_capacity = new_max_ghosts;
        mesh->total_capacity = new_total;
        knn::knn_grow(mesh->knn, (int)new_total);
    }

} // namespace voronoi
