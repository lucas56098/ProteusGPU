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
    template <typename T> GLOBAL void kernel_gather(hsize_t n, const T* in, const unsigned int* perm, T* out);
    GLOBAL void                       kernel_build_index_pass1_compact_reals(int                 n_total,
                                                                             hsize_t             n_hydro,
                                                                             const unsigned int* d_permutation,
                                                                             unsigned int*       real_sorted_ids,
                                                                             unsigned int*       sid_to_neighbor,
                                                                             unsigned int*       orig_to_k,
                                                                             int*                counter);
    GLOBAL void                       kernel_build_index_pass1_lookup(int                 n_total,
                                                                      hsize_t             n_hydro,
                                                                      const unsigned int* d_permutation,
                                                                      const unsigned int* orig_to_k_save,
                                                                      unsigned int*       real_sorted_ids,
                                                                      unsigned int*       sid_to_neighbor,
                                                                      unsigned int*       orig_to_k);
    GLOBAL void                       kernel_build_index_pass2_remap_ghosts(int                 n_total,
                                                                            hsize_t             n_hydro,
                                                                            const unsigned int* d_permutation,
                                                                            const hsize_t*      ghost_ids,
                                                                            const unsigned int* orig_to_k,
                                                                            unsigned int*       sid_to_neighbor);
    GLOBAL void                       kernel_init_cell_status(int n, Status* stat);
    GLOBAL void kernel_collect_failed_cells(int n, const Status* stat, int* failed_indices, int* failed_count);
    GLOBAL void kernel_compute_voronoi_cells_fast(int                n_hydro,
                                                  double*            d_stored_points,
                                                  const knn_problem* knn,
                                                  Status*            stat,
                                                  VMesh*             mesh,
                                                  hsize_t*           face_offset,
                                                  int*               overflow_flag);
    GLOBAL void kernel_compute_voronoi_cells_slow(int                n_failed,
                                                  const int*         failed_ks,
                                                  double*            d_stored_points,
                                                  const knn_problem* knn,
                                                  Status*            stat,
                                                  VMesh*             mesh,
                                                  hsize_t*           face_offset,
                                                  int*               overflow_flag);
#endif

    // ---- per-step GPU scratch (GPU) / accumulators (CPU) for cell construction ----
#ifndef CPU_DEBUG
    static hsize_t* d_face_offset             = nullptr;
    static int*     d_overflow_flag           = nullptr;
    static int*     d_failed_indices          = nullptr;
    static int*     d_failed_count            = nullptr;
    static int      d_failed_indices_capacity = 0;
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
            std::cerr << "VORONOI: Error! point count " << n_total << " exceeds pre-allocated capacity "
                      << mesh->total_capacity << ". Increase ghost headroom." << std::endl;
            exit(EXIT_FAILURE);
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
        gpu_memset(mesh->outer_halo_hit, 0, sizeof(int));
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (int)((n_hydro + tpb - 1) / tpb);
        {
            PROFILE_KERNEL("INIT");
            kernel_init_cell_status<<<blocks, tpb>>>((int)n_hydro, mesh->cell_status);
            GPU_SYNC();
        }
#else
        for (hsize_t i = 0; i < n_hydro; i++)
            mesh->cell_status[i] = security_radius_not_reached;
#endif
    }

    // build real_sorted_ids[k] -> sid and sid_to_neighbor[sid] -> k (both passes).
    // The pass-1 orig->k map is stashed in scratch_uint so pass 2 can resolve periodic ghosts.
    static void build_index_maps(VMesh* mesh, int iter) {
        const int     n_total = (int)mesh->n_seeds;
        const hsize_t n_hydro = mesh->n_hydro;

#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (n_total + tpb - 1) / tpb;

        // pass 1: assign each real seed an output index k
        if (iter == 0) {
            // iter 0: count + emit via atomic counter (compact reals into [0, n_hydro))
            gpu_memset(mesh->d_real_counter, 0, sizeof(int));
            kernel_build_index_pass1_compact_reals<<<blocks, tpb>>>(n_total,
                                                                    n_hydro,
                                                                    mesh->knn->d_permutation,
                                                                    mesh->real_sorted_ids,
                                                                    mesh->sid_to_neighbor,
                                                                    mesh->scratch_uint,
                                                                    mesh->d_real_counter);
            GPU_SYNC();
        } else {
            // iter > 0: reuse iter-0's orig_to_k_save for stable k assignment
            kernel_build_index_pass1_lookup<<<blocks, tpb>>>(n_total,
                                                             n_hydro,
                                                             mesh->knn->d_permutation,
                                                             mesh->orig_to_k_save,
                                                             mesh->real_sorted_ids,
                                                             mesh->sid_to_neighbor,
                                                             mesh->scratch_uint);
            GPU_SYNC();
        }

        // pass 2: resolve ghost sids — MPI ghosts hold ext-array indices, periodic ghosts hold source orig
        kernel_build_index_pass2_remap_ghosts<<<blocks, tpb>>>(
            n_total, n_hydro, mesh->knn->d_permutation, mesh->ghost_ids, mesh->scratch_uint, mesh->sid_to_neighbor);
        GPU_SYNC();
        GPU_SYNC();

        // sanity check on iter 0: pass-1 must have visited exactly n_hydro reals
        if (iter == 0 && (hsize_t)*mesh->d_real_counter != n_hydro) {
            std::cerr << "VORONOI: build_index_maps: counted " << *mesh->d_real_counter
                      << " reals but n_hydro = " << n_hydro << ". Aborting." << std::endl;
            exit(EXIT_FAILURE);
        }
#else
        const unsigned int* dperm = mesh->knn->d_permutation;

        // pass 1: assign each real seed an output index k
        if (iter == 0) {
            // iter 0: serial fold over sids, compact reals into [0, n_hydro)
            unsigned int k = 0;
            for (int sid = 0; sid < n_total; sid++) {
                const unsigned int orig = dperm[sid];
                if ((hsize_t)orig < n_hydro) {
                    mesh->real_sorted_ids[k]   = (unsigned int)sid;
                    mesh->sid_to_neighbor[sid] = k;
                    mesh->scratch_uint[orig]   = k;
                    k++;
                }
            }
            if ((hsize_t)k != n_hydro) {
                std::cerr << "VORONOI: build_index_maps: counted " << k << " reals but n_hydro = " << n_hydro
                          << ". Aborting." << std::endl;
                exit(EXIT_FAILURE);
            }
        } else {
            // iter > 0: reuse iter-0's orig_to_k_save for stable k assignment
            for (int sid = 0; sid < n_total; sid++) {
                const unsigned int orig = dperm[sid];
                if ((hsize_t)orig < n_hydro) {
                    const unsigned int k       = mesh->orig_to_k_save[orig];
                    mesh->real_sorted_ids[k]   = (unsigned int)sid;
                    mesh->sid_to_neighbor[sid] = k;
                    mesh->scratch_uint[orig]   = k;
                }
            }
        }

        // pass 2: resolve ghost sids — MPI ghosts hold ext-array indices, periodic ghosts hold source orig
        for (int sid = 0; sid < n_total; sid++) {
            const unsigned int orig = dperm[sid];
            if ((hsize_t)orig >= n_hydro) {
                const hsize_t      g       = (hsize_t)orig - n_hydro;
                const unsigned int v       = (unsigned int)mesh->ghost_ids[g];
                mesh->sid_to_neighbor[sid] = (v >= (unsigned int)n_hydro) ? v : mesh->scratch_uint[v];
            }
        }
#endif
    }

    // gather_perm[new_k] = d_permutation[real_sorted_ids[new_k]] = old_k
    // (step N's k IS step N+1's input orig, hence "new_k -> old_k")
    static void compute_gather_perm(VMesh* mesh) {
        const hsize_t n = mesh->n_hydro;
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (int)((n + tpb - 1) / tpb);
        kernel_gather<unsigned int>
            <<<blocks, tpb>>>(n, mesh->knn->d_permutation, mesh->real_sorted_ids, mesh->gather_perm);
        GPU_SYNC();
#else
        for (hsize_t k = 0; k < n; k++) {
            mesh->gather_perm[k] = mesh->knn->d_permutation[mesh->real_sorted_ids[k]];
        }
#endif
    }

    // out-of-place gather then pointer swap. The permutation only touches [0, n);
    // the MPI-ghost-slot region [n, ext) is copied verbatim so it survives the swap.
    template <typename T> static void permute_inplace(T*& live, T*& scratch, hsize_t n, const unsigned int* perm) {
        const hsize_t ext = (hsize_t)proteus_mpi::extended_size((int)n);
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (int)((n + tpb - 1) / tpb);
        kernel_gather<T><<<blocks, tpb>>>(n, live, perm, scratch);
        GPU_SYNC();
        if (ext > n) { gpu_memcpy(scratch + n, live + n, (ext - n) * sizeof(T)); }
#else
        for (hsize_t k = 0; k < n; k++)
            scratch[k] = live[perm[k]];
        for (hsize_t k = n; k < ext; k++)
            scratch[k] = live[k];
#endif
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
#ifndef CPU_DEBUG
        // first call: allocate the singleton scratch slots
        if (!d_face_offset) {
            d_face_offset   = gpu_calloc<hsize_t>(1);
            d_overflow_flag = gpu_calloc<int>(1);
            d_failed_count  = gpu_calloc<int>(1);
        }
        // grow the failed-indices buffer if n_hydro outgrew it (one-shot per growth)
        if (d_failed_indices_capacity < (int)n_hydro) {
            if (d_failed_indices) gpu_free(d_failed_indices);
            d_failed_indices          = gpu_alloc<int>((int)n_hydro);
            d_failed_indices_capacity = (int)n_hydro;
        }
        // zero the per-step counters
        gpu_memset(d_face_offset, 0, sizeof(hsize_t));
        gpu_memset(d_overflow_flag, 0, sizeof(int));
        gpu_memset(d_failed_count, 0, sizeof(int));
#else
        (void)n_hydro;
        s_cpu_face_offset   = 0;
        s_cpu_overflow_flag = 0;
#endif
    }

    // dispatch the fast voronoi kernel over n_hydro cells
    static void run_fast_cell_kernel(VMesh* mesh) {
        const int n_hydro = (int)mesh->n_hydro;
#ifndef CPU_DEBUG
        const int tpb    = _VORO_BLOCK_SIZE_;
        const int blocks = (n_hydro + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("FAST");
            kernel_compute_voronoi_cells_fast<<<blocks, tpb>>>(n_hydro,
                                                               (double*)mesh->knn->d_stored_points,
                                                               mesh->knn,
                                                               mesh->cell_status,
                                                               mesh,
                                                               d_face_offset,
                                                               d_overflow_flag);
            GPU_SYNC();
        }
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int k = 0; k < n_hydro; k++) {
            if (s_cpu_overflow_flag) continue;
            const int seed_id = (int)mesh->real_sorted_ids[k];
            compute_single_voronoi_cell<_FAST_K_, _FAST_MAX_P_, _FAST_MAX_T_, uchar, VERT_TYPE>(k,
                                                                              seed_id,
                                                                              (double*)mesh->knn->d_stored_points,
                                                                              mesh->knn,
                                                                              mesh->cell_status,
                                                                              mesh,
                                                                              &s_cpu_face_offset,
                                                                              &s_cpu_overflow_flag);
        }
#endif
    }

    // count cells that did not finish under the fast kernel and (GPU) emit their k indices
    static int collect_failed_cells(VMesh* mesh) {
        const int n_hydro = (int)mesh->n_hydro;
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (n_hydro + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("COLLECT");
            kernel_collect_failed_cells<<<blocks, tpb>>>(n_hydro, mesh->cell_status, d_failed_indices, d_failed_count);
        }
        GPU_SYNC();
        return *d_failed_count;
#else
        int n_failed = 0;
        for (int k = 0; k < n_hydro; k++)
            if (mesh->cell_status[k] != success) n_failed++;
        return n_failed;
#endif
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
        const int n_hydro = (int)mesh->n_hydro;
#ifndef CPU_DEBUG
        const int tpb    = _VORO_BLOCK_SIZE_;
        const int blocks = (n_failed + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("SLOW");
            kernel_compute_voronoi_cells_slow<<<blocks, tpb>>>(n_failed,
                                                               d_failed_indices,
                                                               (double*)mesh->knn->d_stored_points,
                                                               mesh->knn,
                                                               mesh->cell_status,
                                                               mesh,
                                                               d_face_offset,
                                                               d_overflow_flag);
            GPU_SYNC();
        }
#else
        (void)n_failed;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int k = 0; k < n_hydro; k++) {
            if (s_cpu_overflow_flag) continue;
            if (mesh->cell_status[k] == success) continue;
            const int seed_id = (int)mesh->real_sorted_ids[k];
            compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_, uchar, VERT_TYPE>(k,
                                                               seed_id,
                                                               (double*)mesh->knn->d_stored_points,
                                                               mesh->knn,
                                                               mesh->cell_status,
                                                               mesh,
                                                               &s_cpu_face_offset,
                                                               &s_cpu_overflow_flag);
        }
#endif
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
            std::cerr << "VORONOI: Error! face offset exceeds pre-allocated face capacity " << mesh->face_capacity
                      << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // ============================================================
    // CUDA kernels
    // ============================================================
#ifndef CPU_DEBUG

    // gather: out[k] = in[perm[k]] for k in [0, n)
    template <typename T> GLOBAL void kernel_gather(hsize_t n, const T* in, const unsigned int* perm, T* out) {
        hsize_t k = (hsize_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (k < n) out[k] = in[perm[k]];
    }

    // pass-1 compact-reals: each real sid grabs a unique k via atomic counter
    GLOBAL void kernel_build_index_pass1_compact_reals(int                 n_total,
                                                       hsize_t             n_hydro,
                                                       const unsigned int* d_permutation,
                                                       unsigned int*       real_sorted_ids,
                                                       unsigned int*       sid_to_neighbor,
                                                       unsigned int*       orig_to_k,
                                                       int*                counter) {
        const int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= n_total) return;
        const unsigned int orig = d_permutation[sid];
        if ((hsize_t)orig < n_hydro) {
            const int k          = portable_atomicAdd(counter, 1);
            real_sorted_ids[k]   = (unsigned int)sid;
            sid_to_neighbor[sid] = (unsigned int)k;
            orig_to_k[orig]      = (unsigned int)k;
        }
    }

    // pass-1 lookup: each real sid reuses iter-0's saved orig_to_k for stable k assignment
    GLOBAL void kernel_build_index_pass1_lookup(int                 n_total,
                                                hsize_t             n_hydro,
                                                const unsigned int* d_permutation,
                                                const unsigned int* orig_to_k_save,
                                                unsigned int*       real_sorted_ids,
                                                unsigned int*       sid_to_neighbor,
                                                unsigned int*       orig_to_k) {
        const int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= n_total) return;
        const unsigned int orig = d_permutation[sid];
        if ((hsize_t)orig < n_hydro) {
            const unsigned int k = orig_to_k_save[orig];
            real_sorted_ids[k]   = (unsigned int)sid;
            sid_to_neighbor[sid] = k;
            orig_to_k[orig]      = k; // populate scratch_uint so pass-2 can resolve periodic ghosts
        }
    }

    // pass-2: resolve ghost sids — MPI ghost = ext-array index (>= n_hydro), periodic = source orig
    GLOBAL void kernel_build_index_pass2_remap_ghosts(int                 n_total,
                                                      hsize_t             n_hydro,
                                                      const unsigned int* d_permutation,
                                                      const hsize_t*      ghost_ids,
                                                      const unsigned int* orig_to_k,
                                                      unsigned int*       sid_to_neighbor) {
        const int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= n_total) return;
        const unsigned int orig = d_permutation[sid];
        if ((hsize_t)orig >= n_hydro) {
            const hsize_t      g = (hsize_t)orig - n_hydro;
            const unsigned int v = (unsigned int)ghost_ids[g];
            sid_to_neighbor[sid] = (v >= (unsigned int)n_hydro) ? v : orig_to_k[v];
        }
    }

    // initialise cell status to security_radius_not_reached before each cell build
    GLOBAL void kernel_init_cell_status(int n, Status* stat) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) stat[i] = security_radius_not_reached;
    }

    // collect indices of cells whose status is not success into failed_indices[0 .. *failed_count)
    GLOBAL void kernel_collect_failed_cells(int n, const Status* stat, int* failed_indices, int* failed_count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n && stat[i] != success) {
            const int slot       = portable_atomicAdd(failed_count, 1);
            failed_indices[slot] = i;
        }
    }

    // fast-tier per-cell kernel: small K, small face capacity
    GLOBAL __launch_bounds__(_VORO_BLOCK_SIZE_, 16) void kernel_compute_voronoi_cells_fast(int     n_hydro,
                                                                                           double* d_stored_points,
                                                                                           const knn_problem* knn,
                                                                                           Status*            stat,
                                                                                           VMesh*             mesh,
                                                                                           hsize_t* face_offset,
                                                                                           int*     overflow_flag) {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_hydro) return;
        const int seed_id = (int)mesh->real_sorted_ids[k];
        compute_single_voronoi_cell<_FAST_K_, _FAST_MAX_P_, _FAST_MAX_T_, uchar, VERT_TYPE>(
            k, seed_id, d_stored_points, knn, stat, mesh, (unsigned long long*)face_offset, overflow_flag);
    }

    // slow-tier per-cell kernel: bigger K + larger face capacity, run only on cells that failed fast tier
    GLOBAL __launch_bounds__(_VORO_BLOCK_SIZE_, 8) void kernel_compute_voronoi_cells_slow(int        n_failed,
                                                                                          const int* failed_ks,
                                                                                          double*    d_stored_points,
                                                                                          const knn_problem* knn,
                                                                                          Status*            stat,
                                                                                          VMesh*             mesh,
                                                                                          hsize_t* face_offset,
                                                                                          int*     overflow_flag) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_failed) return;
        const int k       = failed_ks[i];
        const int seed_id = (int)mesh->real_sorted_ids[k];
        compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_, uchar, VERT_TYPE>(
            k, seed_id, d_stored_points, knn, stat, mesh, (unsigned long long*)face_offset, overflow_flag);
    }

#endif // !CPU_DEBUG

} // namespace voronoi
