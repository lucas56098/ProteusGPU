
// one build of the mesh (internal.h)

namespace voronoi {

    static void                       check_seed_capacity(const VMesh* mesh, int n_total);
    static void                       clear_cell_arrays(VMesh* mesh);
    static void                       build_index_maps(VMesh* mesh);
    static void                       permute_persistent_state(VMesh*                    mesh,
                                                               hydro::primvars*          primvar,
                                                               hydro::ConsVars*          cons,
                                                               gradients::PrimGradients* grads);
    template <typename T> static void permute_inplace(T*& live, T*& scratch, uint64_t n, const unsigned int* perm);
    static void                       compute_cells(VMesh* mesh, bool only_open);

    static void allocate_cell_scratch(uint64_t n_hydro, uint64_t first_face);
    static int  reopen_cells(VMesh* mesh);
    static void run_fast_cell_kernel(VMesh* mesh, int n_listed);
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

    // the cells alone in Morton order give the cell order of this step; the state and the positions follow,
    // so input point k is cell k from now on, and the tree is built again over the cells in that order
    void fix_cell_order(VMesh*                    mesh,
                        POINT_TYPE*               cell_pos,
                        hydro::primvars*          primvar,
                        hydro::ConsVars*          cons,
                        gradients::PrimGradients* grads) {
        const uint64_t n = mesh->n_hydro;
        knn::prepare(mesh->knn, (const POINT_TYPE*)cell_pos, (int)n);
        mesh->n_seeds = n;

        const unsigned int* dperm    = mesh->knn->d_permutation;
        unsigned int*       gathered = mesh->gather_perm;
        parallel_for<_MESH_BLOCK_SIZE_>("GATHER_PERM", n, [=] HD(size_t k) { gathered[k] = dperm[k]; });
        permute_persistent_state(mesh, primvar, cons, grads);

        POINT_TYPE* tmp = mesh->scratch_point;
        parallel_for<_MESH_BLOCK_SIZE_>("PERMUTE_POS", n, [=] HD(size_t k) { tmp[k] = cell_pos[gathered[k]]; });
        gpu_memcpy(cell_pos, tmp, n * sizeof(POINT_TYPE));

        knn::prepare(mesh->knn, (const POINT_TYPE*)cell_pos, (int)n);
        build_index_maps(mesh);
    }

    // sorts the points, maps the indices, builds all cells or only those not finished yet
    void compute_mesh(VMesh* mesh, POINT_TYPE* pts, int n_total, bool only_open) {
        {
            PROFILE("KNN_PREP");
            knn::prepare(mesh->knn, (const POINT_TYPE*)pts, n_total);
        }

        check_seed_capacity(mesh, n_total);
        mesh->n_seeds = (uint64_t)n_total;

        {
            PROFILE("INDEX_MAPS");
            build_index_maps(mesh);
        }

        {
            PROFILE("CELLS");
            if (!only_open) {
                mesh->num_faces = 0;
                clear_cell_arrays(mesh);
            }
            compute_cells(mesh, only_open);
        }
    }

    // stops the run if the point list is longer than the arrays
    static void check_seed_capacity(const VMesh* mesh, int n_total) {
        if ((uint64_t)n_total > mesh->total_capacity) {
            proteus_mpi::exit_failure("VORONOI: Error! point count %d exceeds pre-allocated capacity %llu.\n",
                                      n_total,
                                      (unsigned long long)mesh->total_capacity);
        }
    }

    // faces and status of the new build
    static void clear_cell_arrays(VMesh* mesh) {
        const uint64_t n_hydro = mesh->n_hydro;
        gpu_memset(mesh->face_counts, 0, n_hydro * sizeof(uint64_t));
        gpu_memset(mesh->face_ptr, 0, n_hydro * sizeof(uint64_t));
        Status* stat = mesh->cell_status;
        parallel_for<_MESH_BLOCK_SIZE_>("INIT", n_hydro, [=] HD(size_t i) { stat[i] = security_radius_not_reached; });
    }

    // sorted point <-> neighbour index; the cells come first in the input, as cell k, the ghosts after them
    static void build_index_maps(VMesh* mesh) {
        const int       n_total         = (int)mesh->n_seeds;
        const uint64_t  n_hydro         = mesh->n_hydro;
        const unsigned* dperm           = mesh->knn->d_permutation;
        unsigned int*   real_sorted_ids = mesh->real_sorted_ids;
        unsigned int*   sid_to_neighbor = mesh->sid_to_neighbor;
        const uint64_t* ghost_ids       = mesh->ghost_ids;

        // a copy of an own cell stands for that cell, any other ghost for its slot
        parallel_for<_MESH_BLOCK_SIZE_>("INDEX", n_total, [=] HD(int sid) {
            const unsigned int orig = dperm[sid];
            if ((uint64_t)orig < n_hydro) {
                real_sorted_ids[orig] = (unsigned int)sid;
                sid_to_neighbor[sid]  = orig;
            } else {
                sid_to_neighbor[sid] = (unsigned int)ghost_ids[(uint64_t)orig - n_hydro];
            }
        });
    }

    // writes live[perm[k]] into the scratch and swaps the pointers
    template <typename T> static void permute_inplace(T*& live, T*& scratch, uint64_t n, const unsigned int* perm) {
        T* src = live;
        T* dst = scratch;
        parallel_for<_MESH_BLOCK_SIZE_>("PERMUTE", n, [=] HD(size_t k) { dst[k] = src[perm[k]]; });
        std::swap(live, scratch);
    }

    // sorts everything that outlives the step into the new cell order
    static void permute_persistent_state(VMesh*                    mesh,
                                         hydro::primvars*          primvar,
                                         hydro::ConsVars*          cons,
                                         gradients::PrimGradients* grads) {
        const uint64_t      n    = mesh->n_hydro;
        const unsigned int* perm = mesh->gather_perm;

        if (primvar) {
            permute_inplace(primvar->rho, mesh->scratch_double, n, perm);
            permute_inplace(primvar->v, mesh->scratch_point, n, perm);
            permute_inplace(primvar->E, mesh->scratch_double, n, perm);
        }
        if (cons) {
            permute_inplace(cons->mass, mesh->scratch_double, n, perm);
            permute_inplace(cons->momentum, mesh->scratch_point, n, perm);
            permute_inplace(cons->energy, mesh->scratch_double, n, perm);
        }
        if (grads) {
            permute_inplace(grads->rho, mesh->scratch_point, n, perm);
            permute_inplace(grads->vx, mesh->scratch_point, n, perm);
            permute_inplace(grads->vy, mesh->scratch_point, n, perm);
#ifdef dim_3D
            permute_inplace(grads->vz, mesh->scratch_point, n, perm);
#endif
            permute_inplace(grads->E, mesh->scratch_point, n, perm);
            permute_inplace(grads->anchor, mesh->scratch_point, n, perm);
        }
#ifdef MOVING_MESH
        permute_inplace(mesh->v_mesh, mesh->scratch_point, n, perm);
#endif

#ifndef CPU_DEBUG
        GPU_SYNC();
#endif
    }

    // fast tier for all cells or the reopened ones, slow tier for what failed; a cell that is finished keeps its
    // faces, the faces of the others come after them
    static void compute_cells(VMesh* mesh, bool only_open) {
        allocate_cell_scratch(mesh->n_hydro, only_open ? mesh->num_faces : 0);

        const int n_listed = only_open ? reopen_cells(mesh) : -1;
        run_fast_cell_kernel(mesh, n_listed);

        const int n_failed = collect_failed_cells(mesh);
        if (!only_open) print_cell_build_summary(mesh->n_hydro, n_failed);

        if (n_failed > 0) run_slow_cell_kernel(mesh, n_failed);

        read_face_count_from_gpu(mesh);
    }

    // the cells not finished yet into the list, their old faces out of the way
    static int reopen_cells(VMesh* mesh) {
        const int  n_open = collect_failed_cells(mesh);
        const int* list   = d_failed_indices;
        Status*    stat   = mesh->cell_status;
        parallel_for<_MESH_BLOCK_SIZE_>("REOPEN", n_open, [=] HD(int i) {
            const int      k     = list[i];
            const uint64_t first = mesh->face_ptr[k];
            for (uint64_t f = first; f < first + mesh->face_counts[k]; f++) {
                mesh->neighbor_cell[f] = -1;
                mesh->face_area[f]     = 0.0;
                for (int c = 0; c < DIMENSION - 1; c++)
                    mesh->f_mid_local[f * (DIMENSION - 1) + c] = 0.0;
            }
            mesh->face_counts[k] = 0;
            stat[k]              = security_radius_not_reached;
        });
        return n_open;
    }

    // scratch of the cell kernels, kept between builds; new faces start at first_face
    static void allocate_cell_scratch(uint64_t n_hydro, uint64_t first_face) {
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
        *d_face_offset = first_face;
        gpu_memset(d_overflow_flag, 0, sizeof(int));
#else
        s_cpu_face_offset   = first_face;
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

    // all cells, or the first n_listed of the list, small capacities
    static void run_fast_cell_kernel(VMesh* mesh, int n_listed) {
        double*             pts   = (double*)mesh->knn->d_stored_points;
        const knn_problem*  knn   = mesh->knn;
        Status*             stat  = mesh->cell_status;
        unsigned long long* foff  = cell_face_offset();
        int*                oflag = cell_overflow_flag();
        const int*          list  = d_failed_indices;
        const int           n     = (n_listed < 0) ? (int)mesh->n_hydro : n_listed;

        parallel_for<_VORO_BLOCK_SIZE_, 16, Sched::Dynamic>("FAST", n, [=] HD(int i) {
            const int k       = (n_listed < 0) ? i : list[i];
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
            compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_, uchar, VERT_TYPE, true>(
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
