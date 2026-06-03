namespace voronoi {

    VMesh* allocate_mesh(hsize_t n_hydro) {
        // n_grow = max post-migration n_local. Periodic and MPI ghosts coexist in pts[] and ghost_ids,
        // and capacities scale with n_grow so all mesh-build buffers survive migration imbalance.
        // ext sizes the per-cell SoA arrays for n_grow + MPI ghost slots.
        const double  ghost_frac     = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        const hsize_t n_grow         = (hsize_t)proteus_mpi::max_n_local((int)n_hydro);
        const hsize_t max_pgh        = (hsize_t)(2.0 * ghost_frac * n_grow) + 1;
        const hsize_t max_mpi_ghosts = (hsize_t)proteus_mpi::n_mpi_capacity;
        const hsize_t max_ghosts     = max_pgh + max_mpi_ghosts;
        const hsize_t total          = n_grow + max_ghosts;
        const hsize_t max_faces      = n_grow * _FACE_CAPACITY_MULT_;
        const hsize_t ext            = (hsize_t)proteus_mpi::alloc_per_cell_size((int)n_hydro);

        VMesh* mesh          = gpu_alloc<VMesh>(1);
        mesh->n_seeds        = 0;
        mesh->n_hydro        = n_hydro;
        mesh->num_faces      = 0;
        mesh->face_capacity  = max_faces;
        mesh->ghost_capacity = max_ghosts;
        mesh->total_capacity = total;
        mesh->buff           = buff;

        // per-cell — every n_hydro-indexed array is sized ext so it survives migration
        // growth and the MPI ghost band [n_hydro, n_hydro + n_mpi_ghosts)
        mesh->seeds       = gpu_calloc<double3>(ext);
        mesh->com         = gpu_calloc<double3>(ext);
        mesh->volumes     = gpu_calloc<double>(ext);
        mesh->face_counts = gpu_calloc<hsize_t>(ext);
        mesh->face_ptr    = gpu_calloc<hsize_t>(ext);
        mesh->cell_status    = gpu_alloc<Status>(ext);
        mesh->cell_hit_outer = gpu_calloc<unsigned char>(ext);
        mesh->pts_mpi_base   = 0;
#ifdef MOVING_MESH
        mesh->v_mesh      = gpu_calloc<POINT_TYPE>(ext);
        mesh->old_volumes = gpu_calloc<double>(ext);
#endif

        // per-face
        mesh->neighbor_cell = gpu_alloc<int>(max_faces);
        mesh->face_area     = gpu_alloc<double>(max_faces);
#ifdef MOVING_MESH
        mesh->f_mid_local = gpu_alloc<double>(max_faces * (DIMENSION - 1));
#endif

        // ghost mapping
        mesh->ghost_ids = gpu_alloc<hsize_t>(max_ghosts);

        // index maps; ext-sized so permute_inplace's live↔scratch swap stays uniform-size
        mesh->real_sorted_ids  = gpu_alloc<unsigned int>(ext);
        mesh->sid_to_neighbor  = gpu_alloc<unsigned int>(total);
        mesh->cell_to_original = gpu_alloc<unsigned int>(ext);
        mesh->gather_perm      = gpu_alloc<unsigned int>(ext);
        mesh->orig_to_k_save   = gpu_alloc<unsigned int>(ext);
        for (hsize_t i = 0; i < n_hydro; i++)
            mesh->cell_to_original[i] = (unsigned int)i;

        // typed scratch pools — ext-sized (see above)
        mesh->scratch_uint   = gpu_alloc<unsigned int>(ext);
        mesh->scratch_double = gpu_alloc<double>(ext);
        mesh->scratch_point  = gpu_alloc<POINT_TYPE>(ext);

        // mesh-build scratch
        mesh->scratch_pts  = gpu_alloc<POINT_TYPE>(total);
        mesh->scratch_move = gpu_alloc<POINT_TYPE>(ext);

        // device counters
        mesh->d_real_counter = gpu_calloc<int>(1);

        // KNN cache
        mesh->knn = knn::init_once((int)n_hydro);

        // hint GPU-preferred placement for hot arrays
        gpu_advise_gpu_preferred(mesh->seeds, ext * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->com, n_hydro * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->volumes, n_hydro * sizeof(double));
        gpu_advise_gpu_preferred(mesh->face_counts, n_hydro * sizeof(hsize_t));
        gpu_advise_gpu_preferred(mesh->face_ptr, n_hydro * sizeof(hsize_t));
        gpu_advise_gpu_preferred(mesh->cell_status, n_hydro * sizeof(Status));
        gpu_advise_gpu_preferred(mesh->neighbor_cell, max_faces * sizeof(int));
        gpu_advise_gpu_preferred(mesh->face_area, max_faces * sizeof(double));
        gpu_advise_gpu_preferred(mesh->real_sorted_ids, n_hydro * sizeof(unsigned int));
        gpu_advise_gpu_preferred(mesh->sid_to_neighbor, total * sizeof(unsigned int));
        gpu_advise_gpu_preferred(mesh->cell_to_original, n_hydro * sizeof(unsigned int));
        gpu_advise_gpu_preferred(mesh->gather_perm, n_hydro * sizeof(unsigned int));

        return mesh;
    }

    void free_mesh(VMesh* mesh) {
        if (!mesh) return;
        gpu_free(mesh->seeds);
        gpu_free(mesh->com);
        gpu_free(mesh->volumes);
        gpu_free(mesh->face_counts);
        gpu_free(mesh->face_ptr);
        gpu_free(mesh->cell_status);
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
        gpu_free(mesh->cell_to_original);
        gpu_free(mesh->gather_perm);
        gpu_free(mesh->orig_to_k_save);
        gpu_free(mesh->scratch_uint);
        gpu_free(mesh->scratch_double);
        gpu_free(mesh->scratch_point);
        gpu_free(mesh->scratch_pts);
        gpu_free(mesh->scratch_move);
        gpu_free(mesh->d_real_counter);
        if (mesh->knn) { knn::knn_free(&mesh->knn); }
        gpu_free(mesh);
    }

} // namespace voronoi
