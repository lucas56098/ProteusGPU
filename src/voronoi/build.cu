
// one build of the mesh (internal.h)

namespace voronoi {

    static void check_seed_capacity(const VMesh* mesh, int n_total);
    static void save_orig_to_k_for_lookup(VMesh* mesh);
    static void clear_cell_arrays(VMesh* mesh);
    static void build_index_maps(VMesh* mesh, int iter);
    static void compute_gather_perm(VMesh* mesh);
    static void permute_persistent_state(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* primvar_aux);
    template <typename T> static void permute_inplace(T*& live, T*& scratch, uint64_t n, const unsigned int* perm);
    static void                       compute_cells(VMesh* mesh);

    static void allocate_cell_scratch(uint64_t n_hydro);
    static void run_fast_cell_kernel(VMesh* mesh);
    static int  collect_failed_cells(VMesh* mesh);
    static int  count_failed_cells(const VMesh* mesh);
    static void print_cell_build_summary(uint64_t n_hydro, int n_failed);
    static void run_slow_cell_kernel(VMesh* mesh, int n_failed);
    static void read_face_count_from_gpu(VMesh* mesh);

    static int* d_failed_indices          = nullptr; // cells the fast tier could not finish
    static int  d_failed_indices_capacity = 0;
#ifndef CPU_DEBUG
    static uint64_t* d_face_offset   = nullptr; // face counter of the cell kernels
    static int*      d_overflow_flag = nullptr;
#else
    static unsigned long long s_cpu_face_offset   = 0;
    static int                s_cpu_overflow_flag = 0;
#endif

    // sorts the points, maps the indices, builds all cells
    void compute_mesh(VMesh*           mesh,
                      POINT_TYPE*      pts_data,
                      int              n_total,
                      hydro::primvars* primvar,
                      hydro::primvars* primvar_aux,
                      int              iter) {
        {
            PROFILE("KNN_PREP");
            // sort the points into the neighbour grid
            knn::set_local_extent(mesh->knn, mesh->data_lo, mesh->data_hi);
            knn::prepare(mesh->knn, (const POINT_TYPE*)pts_data, n_total);
        }

        check_seed_capacity(mesh, n_total);
        mesh->n_seeds   = (uint64_t)n_total;
        mesh->num_faces = 0;

        {
            PROFILE("PERMUTE");
            // the first round also fixes the cell order of this step
            build_index_maps(mesh, iter);
            if (iter == 0) {
                save_orig_to_k_for_lookup(mesh);
                compute_gather_perm(mesh);
                permute_persistent_state(mesh, primvar, primvar_aux);
            }
        }

        {
            PROFILE("CELLS");
            clear_cell_arrays(mesh);
            compute_cells(mesh);
        }
    }

    // stops the run if the point list is longer than the arrays
    static void check_seed_capacity(const VMesh* mesh, int n_total) {
        if ((uint64_t)n_total > mesh->total_capacity) {
            proteus_mpi::exit_failure("VORONOI: Error! point count %d exceeds pre-allocated capacity %llu. "
                                      "Increase ghost headroom.\n",
                                      n_total,
                                      (unsigned long long)mesh->total_capacity);
        }
    }

    // keeps the cell order of the first round
    static void save_orig_to_k_for_lookup(VMesh* mesh) {
        const uint64_t n_hydro = mesh->n_hydro;
        gpu_memcpy(mesh->orig_to_k_save, mesh->scratch_uint, n_hydro * sizeof(unsigned int));
    }

    // faces and status of the new build
    static void clear_cell_arrays(VMesh* mesh) {
        const uint64_t n_hydro = mesh->n_hydro;
        gpu_memset(mesh->face_counts, 0, n_hydro * sizeof(uint64_t));
        gpu_memset(mesh->face_ptr, 0, n_hydro * sizeof(uint64_t));
        Status* stat = mesh->cell_status;
        parallel_for<_MESH_BLOCK_SIZE_>("INIT", n_hydro, [=] HD(size_t i) { stat[i] = security_radius_not_reached; });
    }

    // maps input point <-> sorted point <-> cell k
    static void build_index_maps(VMesh* mesh, int iter) {
        const int      n_total = (int)mesh->n_seeds;
        const uint64_t n_hydro = mesh->n_hydro;

        const unsigned int* dperm           = mesh->knn->d_permutation;
        unsigned int*       real_sorted_ids = mesh->real_sorted_ids;
        unsigned int*       sid_to_neighbor = mesh->sid_to_neighbor;
        unsigned int*       orig_to_k       = mesh->scratch_uint;

        if (iter == 0) {
            unsigned int* flags   = mesh->scan_flags;
            unsigned int* scratch = mesh->scan_scratch;

            // flag the real points; after the scan the flag of a real point is its cell index
            parallel_for<_MESH_BLOCK_SIZE_>(
                "INDEX_FLAG", n_total, [=] HD(int sid) { flags[sid] = ((uint64_t)dperm[sid] < n_hydro) ? 1u : 0u; });

            parallel_exclusive_scan<_MESH_BLOCK_SIZE_>("INDEX_SCAN", (size_t)n_total, flags, flags, scratch);

            parallel_for<_MESH_BLOCK_SIZE_>("INDEX_P1", n_total, [=] HD(int sid) {
                const unsigned int orig = dperm[sid];
                if ((uint64_t)orig < n_hydro) {
                    const unsigned int k = flags[sid];
                    real_sorted_ids[k]   = (unsigned int)sid;
                    sid_to_neighbor[sid] = k;
                    orig_to_k[orig]      = k;
                }
            });

            // all real points must have got a cell
            const uint64_t n_reals =
                (n_total > 0) ? (uint64_t)flags[n_total - 1] + (((uint64_t)dperm[n_total - 1] < n_hydro) ? 1 : 0) : 0;

            if (n_reals != n_hydro) {
                proteus_mpi::exit_failure(
                    "VORONOI: build_index_maps: counted %llu reals but n_hydro = %llu. Aborting.\n",
                    (unsigned long long)n_reals,
                    (unsigned long long)n_hydro);
            }
        } else {
            // later rounds keep the order of the first one
            const unsigned int* orig_to_k_save = mesh->orig_to_k_save;
            parallel_for<_MESH_BLOCK_SIZE_>("INDEX_P1", n_total, [=] HD(int sid) {
                const unsigned int orig = dperm[sid];
                if ((uint64_t)orig < n_hydro) {
                    const unsigned int k = orig_to_k_save[orig];
                    real_sorted_ids[k]   = (unsigned int)sid;
                    sid_to_neighbor[sid] = k;
                    orig_to_k[orig]      = k;
                }
            });
        }

        // ghosts: periodic -> its cell, MPI -> its own slot
        const uint64_t* ghost_ids = mesh->ghost_ids;
        parallel_for<_MESH_BLOCK_SIZE_>("INDEX_P2", n_total, [=] HD(int sid) {
            const unsigned int orig = dperm[sid];
            if ((uint64_t)orig >= n_hydro) {
                const uint64_t     g = (uint64_t)orig - n_hydro;
                const unsigned int v = (unsigned int)ghost_ids[g];
                sid_to_neighbor[sid] = (v >= (unsigned int)n_hydro) ? v : orig_to_k[v];
            }
        });
    }

    // cell k came from input point gather_perm[k]
    static void compute_gather_perm(VMesh* mesh) {
        const uint64_t      n        = mesh->n_hydro;
        const unsigned int* perm     = mesh->knn->d_permutation;
        const unsigned int* sorted   = mesh->real_sorted_ids;
        unsigned int*       gathered = mesh->gather_perm;

        parallel_for<_MESH_BLOCK_SIZE_>("GATHER_PERM", n, [=] HD(size_t k) { gathered[k] = perm[sorted[k]]; });
    }

    // writes live[perm[k]] into the scratch and swaps the pointers
    template <typename T> static void permute_inplace(T*& live, T*& scratch, uint64_t n, const unsigned int* perm) {
        T* src = live;
        T* dst = scratch;
        parallel_for<_MESH_BLOCK_SIZE_>("PERMUTE", n, [=] HD(size_t k) { dst[k] = src[perm[k]]; });
        std::swap(live, scratch);
    }

    // sorts everything that outlives the step into the new cell order
    static void permute_persistent_state(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* primvar_aux) {
        const uint64_t      n    = mesh->n_hydro;
        const unsigned int* perm = mesh->gather_perm;

        if (primvar) {
            permute_inplace(primvar->rho, mesh->scratch_double, n, perm);
            permute_inplace(primvar->v, mesh->scratch_point, n, perm);
            permute_inplace(primvar->E, mesh->scratch_double, n, perm);
        }
        if (primvar_aux) {
            permute_inplace(primvar_aux->rho, mesh->scratch_double, n, perm);
            permute_inplace(primvar_aux->v, mesh->scratch_point, n, perm);
            permute_inplace(primvar_aux->E, mesh->scratch_double, n, perm);
        }
#ifdef MOVING_MESH
        permute_inplace(mesh->v_mesh, mesh->scratch_point, n, perm);
        permute_inplace(mesh->old_volumes, mesh->scratch_double, n, perm);
#endif

#ifndef CPU_DEBUG
        GPU_SYNC();
#endif
    }

    // fast tier for all cells, slow tier for what failed
    static void compute_cells(VMesh* mesh) {
        allocate_cell_scratch(mesh->n_hydro);

        run_fast_cell_kernel(mesh);

        const int n_failed = collect_failed_cells(mesh);
        print_cell_build_summary(mesh->n_hydro, n_failed);

        if (n_failed > 0) run_slow_cell_kernel(mesh, n_failed);

        read_face_count_from_gpu(mesh);
    }

    // scratch of the cell kernels, kept between builds
    static void allocate_cell_scratch(uint64_t n_hydro) {
        if (d_failed_indices_capacity < (int)n_hydro) {
            if (d_failed_indices) gpu_free(d_failed_indices);
            d_failed_indices          = gpu_alloc<int>((int)n_hydro);
            d_failed_indices_capacity = (int)n_hydro;
        }
#ifndef CPU_DEBUG
        if (!d_face_offset) {
            d_face_offset   = gpu_calloc<uint64_t>(1);
            d_overflow_flag = gpu_calloc<int>(1);
        }
        gpu_memset(d_face_offset, 0, sizeof(uint64_t));
        gpu_memset(d_overflow_flag, 0, sizeof(int));
#else
        s_cpu_face_offset   = 0;
        s_cpu_overflow_flag = 0;
#endif
    }

    static unsigned long long* cell_face_offset() {
#ifndef CPU_DEBUG
        return (unsigned long long*)d_face_offset;
#else
        return &s_cpu_face_offset;
#endif
    }

    static int* cell_overflow_flag() {
#ifndef CPU_DEBUG
        return d_overflow_flag;
#else
        return &s_cpu_overflow_flag;
#endif
    }

    // all cells, small capacities
    static void run_fast_cell_kernel(VMesh* mesh) {
        double*             pts   = (double*)mesh->knn->d_stored_points;
        const knn_problem*  knn   = mesh->knn;
        Status*             stat  = mesh->cell_status;
        unsigned long long* foff  = cell_face_offset();
        int*                oflag = cell_overflow_flag();

        parallel_for<_VORO_BLOCK_SIZE_, 16, Sched::Dynamic>("FAST", mesh->n_hydro, [=] HD(int k) {
            const int seed_id = (int)mesh->real_sorted_ids[k];
            compute_single_voronoi_cell<_FAST_K_, _FAST_MAX_P_, _FAST_MAX_T_, uchar, VERT_TYPE>(
                k, seed_id, pts, knn, stat, mesh, foff, oflag);
        });
    }

    // index list of the cells that did not end with success
    static int collect_failed_cells(VMesh* mesh) {
        const int n_hydro = (int)mesh->n_hydro;
        if (n_hydro == 0) return 0;

        PROFILE("COLLECT");
        const Status* stat    = mesh->cell_status;
        unsigned int* flags   = mesh->scan_flags;
        unsigned int* scratch = mesh->scan_scratch;
        int*          out     = d_failed_indices;

        parallel_for<_MESH_BLOCK_SIZE_>(
            "COLLECT_FLAG", n_hydro, [=] HD(int k) { flags[k] = (stat[k] != success) ? 1u : 0u; });

        parallel_exclusive_scan<_MESH_BLOCK_SIZE_>("COLLECT_SCAN", (size_t)n_hydro, flags, flags, scratch);

        parallel_for<_MESH_BLOCK_SIZE_>("COLLECT_SCATTER", n_hydro, [=] HD(int k) {
            if (stat[k] != success) out[flags[k]] = k;
        });

        return (int)flags[n_hydro - 1] + ((stat[n_hydro - 1] != success) ? 1 : 0);
    }

    // cells no tier has built yet
    static int count_failed_cells(const VMesh* mesh) {
        const Status* stat = mesh->cell_status;
        return parallel_reduce_sum<_MESH_BLOCK_SIZE_, int>(
            "COUNT_FAILED", mesh->n_hydro, [=] HD(size_t k) { return (stat[k] != success) ? 1 : 0; });
    }

    static void print_cell_build_summary(uint64_t n_hydro, int n_failed) {
        const int n_global        = logging::sum_global((int)n_hydro);
        const int n_failed_global = logging::sum_global(n_failed);
        logging::root() << "VORONOI: Generated " << n_global << " cells. ("
                        << (100.0 * n_failed_global / (double)n_global) << "% slow tier)" << std::endl;
    }

    // again with the full capacities
    static void run_slow_cell_kernel(VMesh* mesh, int n_failed) {
        const int*          failed_ks = d_failed_indices;
        double*             pts       = (double*)mesh->knn->d_stored_points;
        const knn_problem*  knn       = mesh->knn;
        Status*             stat      = mesh->cell_status;
        unsigned long long* foff      = cell_face_offset();
        int*                oflag     = cell_overflow_flag();

        parallel_for<_VORO_BLOCK_SIZE_, 8, Sched::Dynamic>("SLOW", n_failed, [=] HD(int i) {
            const int k       = failed_ks[i];
            const int seed_id = (int)mesh->real_sorted_ids[k];
            compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_, uchar, VERT_TYPE>(
                k, seed_id, pts, knn, stat, mesh, foff, oflag);
        });
    }

    // total faces, aborts if they did not fit
    static void read_face_count_from_gpu(VMesh* mesh) {
#ifndef CPU_DEBUG
        GPU_SYNC();
        mesh->num_faces         = *d_face_offset;
        const int overflow_flag = *d_overflow_flag;
#else
        mesh->num_faces         = (uint64_t)s_cpu_face_offset;
        const int overflow_flag = s_cpu_overflow_flag;
#endif
        if (overflow_flag) {
            proteus_mpi::exit_failure("VORONOI: Error! face offset exceeds pre-allocated face capacity %llu. "
                                      "Increase _FACE_CAPACITY_MULT_ in Config.sh.\n",
                                      (unsigned long long)mesh->face_capacity);
        }
    }

} // namespace voronoi
