
// allocates and frees the mesh (voronoi.h)

namespace voronoi {

    // allocates all mesh arrays, once at startup
    VMesh* allocate_mesh(uint64_t n_hydro) {
        // cells with growth headroom + periodic ghosts + MPI ghosts
        const double   ghost_frac     = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        const uint64_t n_grow         = (uint64_t)proteus_mpi::max_n_local((int)n_hydro);
        const uint64_t max_pgh        = (uint64_t)(2.0 * ghost_frac * n_grow) + 1;
        const uint64_t max_mpi_ghosts = (uint64_t)proteus_mpi::n_mpi_capacity;
        const uint64_t max_ghosts     = max_pgh + max_mpi_ghosts;
        const uint64_t total          = n_grow + max_ghosts;
        const uint64_t max_faces      = n_grow * _FACE_CAPACITY_MULT_;
        const uint64_t ext            = (uint64_t)proteus_mpi::max_n_local((int)n_hydro);

        VMesh* mesh          = gpu_alloc<VMesh>(1);
        mesh->n_seeds        = 0;
        mesh->n_hydro        = n_hydro;
        mesh->num_faces      = 0;
        mesh->face_capacity  = max_faces;
        mesh->ghost_capacity = max_ghosts;
        mesh->total_capacity = total;
        mesh->buff           = buff;

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

        mesh->cell_status = gpu_alloc<Status>(ext);
#ifdef USE_MPI
        mesh->security_d2 = gpu_calloc<double>(ext);
#else
        mesh->security_d2 = nullptr;
#endif
        mesh->n_mpi_ghosts = 0;
        for (int a = 0; a < 3; a++) {
            mesh->data_lo[a] = 0.0;
            mesh->data_hi[a] = 0.0;
        }
#ifdef MOVING_MESH
        mesh->v_mesh      = gpu_calloc<POINT_TYPE>(ext);
        mesh->old_volumes = gpu_calloc<double>(ext);
#endif

        // ghost arrays, none without MPI
        const int gc = proteus_mpi::n_mpi_capacity;
        if (gc > 0) {
            mesh->seeds_g = gpu_alloc<double3>(gc);
#ifdef VOL_REGULARIZE
            mesh->volumes_g = gpu_alloc<double>(gc);
#endif
#ifdef MOVING_MESH
            mesh->v_mesh_g = gpu_alloc<POINT_TYPE>(gc);
#endif
            gpu_advise_gpu_preferred(mesh->seeds_g, gc * sizeof(double3));
#ifdef MOVING_MESH
            gpu_advise_gpu_preferred(mesh->v_mesh_g, gc * sizeof(POINT_TYPE));
#endif
        } else {
            mesh->seeds_g = nullptr;
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
#ifdef MOVING_MESH
        mesh->f_mid_local = gpu_alloc<double>(max_faces * (DIMENSION - 1));
#endif

        mesh->ghost_ids = gpu_alloc<uint64_t>(max_ghosts);

        // index maps and scratch
        mesh->real_sorted_ids = gpu_alloc<unsigned int>(ext);
        mesh->sid_to_neighbor = gpu_alloc<unsigned int>(total);
        mesh->gather_perm     = gpu_alloc<unsigned int>(ext);
        mesh->orig_to_k_save  = gpu_alloc<unsigned int>(ext);
        mesh->scan_flags      = gpu_alloc<unsigned int>(total);
        mesh->scan_scratch    = gpu_alloc<unsigned int>(scan_scratch_size((size_t)total, _MESH_BLOCK_SIZE_));

        mesh->scratch_uint   = gpu_alloc<unsigned int>(ext);
        mesh->scratch_double = gpu_alloc<double>(ext);
        mesh->scratch_point  = gpu_alloc<POINT_TYPE>(ext);

        mesh->scratch_pts  = gpu_alloc<POINT_TYPE>(total);
        mesh->scratch_move = gpu_alloc<POINT_TYPE>(ext);

        mesh->knn = knn::init_once((int)n_hydro, ic_data.header.knn_N_grid);

        // keep the hot arrays on the device
        gpu_advise_gpu_preferred(mesh->seeds, ext * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->com, n_hydro * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->volumes, n_hydro * sizeof(double));
        gpu_advise_gpu_preferred(mesh->face_counts, n_hydro * sizeof(uint64_t));
        gpu_advise_gpu_preferred(mesh->face_ptr, n_hydro * sizeof(uint64_t));
        gpu_advise_gpu_preferred(mesh->cell_status, n_hydro * sizeof(Status));
#ifdef USE_MPI
        gpu_advise_gpu_preferred(mesh->security_d2, n_hydro * sizeof(double));
#endif
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
#ifdef USE_MPI
        gpu_free(mesh->security_d2);
#endif
#ifdef MOVING_MESH
        gpu_free(mesh->v_mesh);
        gpu_free(mesh->old_volumes);
        gpu_free(mesh->f_mid_local);
#endif
        gpu_free(mesh->neighbor_cell);
        gpu_free(mesh->face_area);
        gpu_free(mesh->ghost_ids);
        gpu_free(mesh->real_sorted_ids);
        gpu_free(mesh->sid_to_neighbor);
        gpu_free(mesh->gather_perm);
        gpu_free(mesh->orig_to_k_save);
        gpu_free(mesh->scan_flags);
        gpu_free(mesh->scan_scratch);
        gpu_free(mesh->scratch_uint);
        gpu_free(mesh->scratch_double);
        gpu_free(mesh->scratch_point);
        gpu_free(mesh->scratch_pts);
        gpu_free(mesh->scratch_move);
        if (mesh->seeds_g) gpu_free(mesh->seeds_g);
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
#ifdef VOL_REGULARIZE
        if (mesh->volumes_g) gpu_free(mesh->volumes_g);
        mesh->volumes_g = (new_cap > 0) ? gpu_alloc<double>(new_cap) : nullptr;
#endif
#ifdef MOVING_MESH
        if (mesh->v_mesh_g) gpu_free(mesh->v_mesh_g);
        mesh->v_mesh_g = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
#endif
    }

    // longer point list and maps, content kept
    void mesh_grow_build_buffers(VMesh* mesh, int new_mpi_capacity) {
        const double   ghost_frac     = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        const uint64_t n_grow         = (uint64_t)proteus_mpi::max_n_local((int)mesh->n_hydro);
        const uint64_t max_pgh        = (uint64_t)(2.0 * ghost_frac * n_grow) + 1;
        const uint64_t new_max_ghosts = max_pgh + (uint64_t)new_mpi_capacity;
        const uint64_t new_total      = n_grow + new_max_ghosts;
        const uint64_t old_total      = mesh->total_capacity;
        if (new_total <= old_total) return;

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

        gpu_free(mesh->scan_flags);
        gpu_free(mesh->scan_scratch);
        mesh->scan_flags   = gpu_alloc<unsigned int>(new_total);
        mesh->scan_scratch = gpu_alloc<unsigned int>(scan_scratch_size((size_t)new_total, _MESH_BLOCK_SIZE_));

        mesh->ghost_capacity = new_max_ghosts;
        mesh->total_capacity = new_total;
    }

} // namespace voronoi
