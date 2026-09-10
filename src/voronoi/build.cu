namespace voronoi {

    // ---- forward declarations ----
    static void check_seed_capacity(const VMesh* mesh, int n_total);
    static void save_orig_to_k_for_lookup(VMesh* mesh);
    static void clear_cell_arrays(VMesh* mesh);
    static void build_index_maps(VMesh* mesh, int iter);
    static void compute_gather_perm(VMesh* mesh);
    static void permute_persistent_state(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* primvar_aux);
    template <typename T> static void permute_inplace(T*& live, T*& scratch, hsize_t n, const unsigned int* perm);
    static void                       compute_cells(VMesh* mesh);

    static void allocate_cell_scratch(hsize_t n_hydro);
    static void run_fast_cell_kernel(VMesh* mesh);
    static int  collect_failed_cells(VMesh* mesh);
    static void print_cell_build_summary(hsize_t n_hydro, int n_failed);
    static void run_slow_cell_kernel(VMesh* mesh, int n_failed);
    static void read_face_count_from_gpu(VMesh* mesh);

#ifndef CPU_DEBUG
#endif

    // ---- per-step scratch for cell construction ----
    // the failed-cell list is built on both backends, so it lives outside the split
    static int* d_failed_indices          = nullptr;
    static int  d_failed_indices_capacity = 0;
#ifndef CPU_DEBUG
    static hsize_t* d_face_offset   = nullptr;
    static int*     d_overflow_flag = nullptr;
#else
    static unsigned long long s_cpu_face_offset   = 0; // running face offset across fast + slow tiers
    static int                s_cpu_overflow_flag = 0; // set if face writes exceed pre-allocated capacity
#endif

    // ============================================================
    // Main routines
    // ============================================================

    // build the Voronoi cells for the current seed + ghost buffer
    //   iter == 0: full pipeline — atomic-counter pass1 + save orig_to_k + permute primvar
    //   iter  > 0: lookup-mode pass1 using saved orig_to_k; primvar already aligned
    void compute_mesh(VMesh*           mesh,
                      POINT_TYPE*      pts_data,
                      int              n_total,
                      hydro::primvars* primvar,
                      hydro::primvars* primvar_aux,
                      int              iter) {
        // KNN spatial sort over the augmented seed buffer
        {
            PROFILE("KNN_PREP");
            // anchor the KNN grid to this rank's current extent (W/brick already set by
            // set_data_extent_for_build) so cell occupancy stays ~3 pts/cell at any rank count
            knn::set_local_extent(mesh->knn, mesh->data_lo, mesh->data_hi);
            knn::prepare(mesh->knn, (const POINT_TYPE*)pts_data, n_total);
        }

        // commit augmented seed count + reset face counter
        check_seed_capacity(mesh, n_total);
        mesh->n_seeds   = (hsize_t)n_total;
        mesh->num_faces = 0;

        // build orig <-> k <-> sid index maps; iter 0 also permutes primvar into new-k order
        {
            PROFILE("PERMUTE");
            build_index_maps(mesh, iter);
            if (iter == 0) {
                save_orig_to_k_for_lookup(mesh);
                compute_gather_perm(mesh);
                permute_persistent_state(mesh, primvar, primvar_aux);
            }
        }

        // reset per-cell arrays, run fast tier, then slow tier on failed cells
        {
            PROFILE("CELLS");
            clear_cell_arrays(mesh);
            compute_cells(mesh);
        }
    }

    // ============================================================
    // Helpers
    // ============================================================

    // abort if the augmented seed count exceeds the pre-allocated capacity
    static void check_seed_capacity(const VMesh* mesh, int n_total) {
        if ((hsize_t)n_total > mesh->total_capacity) {
            proteus_mpi::exit_failure("VORONOI: Error! point count %d exceeds pre-allocated capacity %llu. "
                                      "Increase ghost headroom.\n",
                                      n_total,
                                      (unsigned long long)mesh->total_capacity);
        }
    }

    // snapshot pass-1's orig_to_k mapping so iter > 0 can reproduce the same k assignment
    static void save_orig_to_k_for_lookup(VMesh* mesh) {
        const hsize_t n_hydro = mesh->n_hydro;
        gpu_memcpy(mesh->orig_to_k_save, mesh->scratch_uint, n_hydro * sizeof(unsigned int));
    }

    // clear per-cell arrays and reset cell_status to security_radius_not_reached
    static void clear_cell_arrays(VMesh* mesh) {
        const hsize_t n_hydro = mesh->n_hydro;
        gpu_memset(mesh->face_counts, 0, n_hydro * sizeof(hsize_t));
        gpu_memset(mesh->face_ptr, 0, n_hydro * sizeof(hsize_t));
        Status* stat = mesh->cell_status;
        parallel_for<_MESH_BLOCK_SIZE_>("INIT", n_hydro, [=] HD(size_t i) { stat[i] = security_radius_not_reached; });
    }

    // build real_sorted_ids[k] -> sid and sid_to_neighbor[sid] -> k (both passes).
    // The pass-1 orig->k map is stashed in scratch_uint so pass 2 can resolve periodic ghosts.
    static void build_index_maps(VMesh* mesh, int iter) {
        const int     n_total = (int)mesh->n_seeds;
        const hsize_t n_hydro = mesh->n_hydro;

        const unsigned int* dperm           = mesh->knn->d_permutation;
        unsigned int*       real_sorted_ids = mesh->real_sorted_ids;
        unsigned int*       sid_to_neighbor = mesh->sid_to_neighbor;
        unsigned int*       orig_to_k       = mesh->scratch_uint;

        // pass 1: assign each real seed an output index k
        if (iter == 0) {
            unsigned int* flags   = mesh->scan_flags;
            unsigned int* scratch = mesh->scan_scratch;

            parallel_for<_MESH_BLOCK_SIZE_>(
                "INDEX_FLAG", n_total, [=] HD(int sid) { flags[sid] = ((hsize_t)dperm[sid] < n_hydro) ? 1u : 0u; });

            parallel_exclusive_scan<_MESH_BLOCK_SIZE_>("INDEX_SCAN", (size_t)n_total, flags, flags, scratch);

            parallel_for<_MESH_BLOCK_SIZE_>("INDEX_P1", n_total, [=] HD(int sid) {
                const unsigned int orig = dperm[sid];
                if ((hsize_t)orig < n_hydro) {
                    const unsigned int k = flags[sid];
                    real_sorted_ids[k]   = (unsigned int)sid;
                    sid_to_neighbor[sid] = k;
                    orig_to_k[orig]      = k;
                }
            });

            // exclusive scan, so the total is the last slot plus whether the last sid was real
            const hsize_t n_reals =
                (n_total > 0) ? (hsize_t)flags[n_total - 1] + (((hsize_t)dperm[n_total - 1] < n_hydro) ? 1 : 0) : 0;

            if (n_reals != n_hydro) {
                proteus_mpi::exit_failure(
                    "VORONOI: build_index_maps: counted %llu reals but n_hydro = %llu. Aborting.\n",
                    (unsigned long long)n_reals,
                    (unsigned long long)n_hydro);
            }
        } else {
            // iter > 0: reuse iter-0's orig_to_k_save for stable k assignment
            const unsigned int* orig_to_k_save = mesh->orig_to_k_save;
            parallel_for<_MESH_BLOCK_SIZE_>("INDEX_P1", n_total, [=] HD(int sid) {
                const unsigned int orig = dperm[sid];
                if ((hsize_t)orig < n_hydro) {
                    const unsigned int k = orig_to_k_save[orig];
                    real_sorted_ids[k]   = (unsigned int)sid;
                    sid_to_neighbor[sid] = k;
                    orig_to_k[orig]      = k; // populate scratch_uint so pass 2 can resolve periodic ghosts
                }
            });
        }

        // pass 2: resolve ghost sids — MPI ghosts hold ext-array indices, periodic ghosts hold source orig
        const hsize_t* ghost_ids = mesh->ghost_ids;
        parallel_for<_MESH_BLOCK_SIZE_>("INDEX_P2", n_total, [=] HD(int sid) {
            const unsigned int orig = dperm[sid];
            if ((hsize_t)orig >= n_hydro) {
                const hsize_t      g = (hsize_t)orig - n_hydro;
                const unsigned int v = (unsigned int)ghost_ids[g];
                sid_to_neighbor[sid] = (v >= (unsigned int)n_hydro) ? v : orig_to_k[v];
            }
        });
    }

    // gather_perm[new_k] = d_permutation[real_sorted_ids[new_k]] = old_k
    // (step N's k IS step N+1's input orig, hence "new_k -> old_k")
    static void compute_gather_perm(VMesh* mesh) {
        const hsize_t       n        = mesh->n_hydro;
        const unsigned int* perm     = mesh->knn->d_permutation;
        const unsigned int* sorted   = mesh->real_sorted_ids;
        unsigned int*       gathered = mesh->gather_perm;

        parallel_for<_MESH_BLOCK_SIZE_>("GATHER_PERM", n, [=] HD(size_t k) { gathered[k] = perm[sorted[k]]; });
    }

    // out-of-place gather then pointer swap. The permutation only touches [0, n);
    // the MPI-ghost-slot region [n, ext) is copied verbatim so it survives the swap.
    template <typename T> static void permute_inplace(T*& live, T*& scratch, hsize_t n, const unsigned int* perm) {
        const hsize_t ext = (hsize_t)proteus_mpi::extended_size((int)n);
        T*            src = live;
        T*            dst = scratch;
        parallel_for<_MESH_BLOCK_SIZE_>("PERMUTE", n, [=] HD(size_t k) { dst[k] = src[perm[k]]; });

        // the MPI-ghost-slot region [n, ext) carries over untouched
        if (ext > n) { gpu_memcpy(scratch + n, live + n, (ext - n) * sizeof(T)); }
        std::swap(live, scratch);
    }

    // permute every per-cell array that must carry across the rebuild into new-k order
    static void permute_persistent_state(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* primvar_aux) {
        const hsize_t       n    = mesh->n_hydro;
        const unsigned int* perm = mesh->gather_perm;

        permute_inplace(mesh->cell_to_original, mesh->scratch_uint, n, perm);

        // primary primvars
        if (primvar) {
            permute_inplace(primvar->rho, mesh->scratch_double, n, perm);
            permute_inplace(primvar->v, mesh->scratch_point, n, perm);
            permute_inplace(primvar->E, mesh->scratch_double, n, perm);
        }
        // auxiliary primvars (the second slot used by the RK stages)
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

    // ============================================================
    // Cell construction (fast tier → collect failures → slow tier)
    //
    // CPU-side fallback for cells that fail both GPU tiers is invoked separately
    // by compute_periodic_mesh after the halo-widening loop, so cells can be
    // re-attempted with wider halos before resorting to seed perturbation.
    // ============================================================

    // run fast voronoi kernel on every cell, slow voronoi kernel on cells that fail
    static void compute_cells(VMesh* mesh) {
        // allocate / reuse the per-step GPU scratch buffers
        allocate_cell_scratch(mesh->n_hydro);

        // first attempt: fast voronoi kernel for all cells
        run_fast_cell_kernel(mesh);

        // collect cells that did not converge under the fast kernel
        const int n_failed = collect_failed_cells(mesh);
        print_cell_build_summary(mesh->n_hydro, n_failed);

        // second attempt: slow voronoi kernel on just the failed cells
        if (n_failed > 0) run_slow_cell_kernel(mesh, n_failed);

        // copy total face count back from device and check for overflow
        read_face_count_from_gpu(mesh);
    }

    // allocate / resize the per-step scratch buffers used by the cell-construction kernels
    static void allocate_cell_scratch(hsize_t n_hydro) {
        // grow the failed-indices buffer if n_hydro outgrew it (one-shot per growth)
        if (d_failed_indices_capacity < (int)n_hydro) {
            if (d_failed_indices) gpu_free(d_failed_indices);
            d_failed_indices          = gpu_alloc<int>((int)n_hydro);
            d_failed_indices_capacity = (int)n_hydro;
        }
#ifndef CPU_DEBUG
        // first call: allocate the singleton scratch slots
        if (!d_face_offset) {
            d_face_offset   = gpu_calloc<hsize_t>(1);
            d_overflow_flag = gpu_calloc<int>(1);
        }
        // zero the per-step counters
        gpu_memset(d_face_offset, 0, sizeof(hsize_t));
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

    // dispatch the fast voronoi kernel over n_hydro cells
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

    // List the cells that did not finish under the fast kernel, in index order, on both
    // backends. The slot used to come from an atomic cursor, so the slow tier's work list was
    // ordered by which thread got there first; and the CPU built no list at all, leaving its
    // slow tier to rescan all n_hydro looking for the few failures.
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

        // exclusive scan, so the total is the last offset plus whether the last cell failed
        return (int)flags[n_hydro - 1] + ((stat[n_hydro - 1] != success) ? 1 : 0);
    }

    // print "Generated N cells. (X% slow tier)" for the current build
    static void print_cell_build_summary(hsize_t n_hydro, int n_failed) {
        const int n_global        = logging::sum_global((int)n_hydro);
        const int n_failed_global = logging::sum_global(n_failed);
        logging::root() << "VORONOI: Generated " << n_global << " cells. ("
                        << (100.0 * n_failed_global / (double)n_global) << "% slow tier)" << std::endl;
    }

    // dispatch the slow voronoi kernel over the cells that failed the fast tier
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

    // sync the device, copy num_faces back to host, abort on face-buffer overflow
    static void read_face_count_from_gpu(VMesh* mesh) {
#ifndef CPU_DEBUG
        GPU_SYNC();
        mesh->num_faces         = *d_face_offset;
        const int overflow_flag = *d_overflow_flag;
#else
        mesh->num_faces         = (hsize_t)s_cpu_face_offset;
        const int overflow_flag = s_cpu_overflow_flag;
#endif
        if (overflow_flag) {
            proteus_mpi::exit_failure("VORONOI: Error! face offset exceeds pre-allocated face capacity %llu. "
                                      "Increase _FACE_CAPACITY_MULT_ in Config.sh.\n",
                                      (unsigned long long)mesh->face_capacity);
        }
    }

} // namespace voronoi
