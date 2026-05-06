#include "../global/allvars.h"
#include "../knn/knn.h"
#include "../profiler/profiler.h"
#include "cell.h"
#include "voronoi.h"

#include "cell.cu"
#include "geometry.cu"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

namespace voronoi {

    // forward declarations
    void compute_cells(int, knn_problem*, Status*, VMesh*, const unsigned int*);
    void cpu_fallback_failed_cells(int, double*, Status*, VMesh*, const unsigned int*);

    template <int K, int MAX_P, int MAX_T>
    HD void compute_single_voronoi_cell(
        int, int, double*, const knn_problem*, Status*, VMesh*, const unsigned int*, unsigned long long*, int*);

#ifndef CPU_DEBUG
    // kernels
    GLOBAL void kernel_init_cell_status(int, Status*);
    GLOBAL void kernel_count_failures(int, const Status*, int*);
    GLOBAL void kernel_collect_failed_cells(int, const Status*, int*, int*);
    GLOBAL void kernel_compute_voronoi_cells_fast(
        int, double*, const knn_problem*, Status*, VMesh*, const unsigned int*, hsize_t*, int*);
    GLOBAL void kernel_compute_voronoi_cells_slow(
        int, int, const int*, double*, const knn_problem*, Status*, VMesh*, const unsigned int*, hsize_t*, int*);
#endif

    // ============================================================
    // Allocation and initialization
    // ============================================================

    VMesh* allocate_mesh(hsize_t n_hydro) {

        // worst-case ghost estimate with 2x safety margin
        double  ghost_frac  = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        hsize_t max_ghosts  = (hsize_t)(2.0 * ghost_frac * n_hydro) + 1;
        hsize_t max_n_total = n_hydro + max_ghosts;
        hsize_t max_faces   = max_n_total * _FACE_CAPACITY_MULT_;

        VMesh* mesh          = gpu_alloc<VMesh>(1);
        mesh->n_seeds        = 0;
        mesh->n_hydro        = 0;
        mesh->num_faces      = 0;
        mesh->cell_capacity  = max_n_total;
        mesh->face_capacity  = max_faces;
        mesh->ghost_capacity = max_ghosts;

        // cache buff so device code (BasicConvexCell::plane_for) doesn't have to read the host global
        mesh->buff = buff;

        // per-cell arrays
        mesh->seeds       = gpu_calloc<double3>(max_n_total);
        mesh->com         = gpu_calloc<double3>(max_n_total);
        mesh->volumes     = gpu_calloc<double>(max_n_total);
        mesh->face_counts = gpu_calloc<hsize_t>(max_n_total);
        mesh->face_ptr    = gpu_calloc<hsize_t>(max_n_total);

        // per-face arrays
        mesh->neighbor_cell = gpu_alloc<int>(max_faces);
        mesh->face_area     = gpu_alloc<double>(max_faces);
#ifdef MOVING_MESH
        mesh->f_mid_local = gpu_alloc<double>(max_faces * (DIMENSION - 1));
#endif

        // ghost mapping
        mesh->ghost_ids = gpu_alloc<hsize_t>(max_ghosts);

        // per-cell status flags (reused each timestep)
        mesh->cell_status = gpu_alloc<Status>(max_n_total);

        // hint GPU-preferred location for hot arrays (reduces UM page faults)
        gpu_advise_gpu_preferred(mesh->seeds, max_n_total * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->com, max_n_total * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->volumes, max_n_total * sizeof(double));
        gpu_advise_gpu_preferred(mesh->face_counts, max_n_total * sizeof(hsize_t));
        gpu_advise_gpu_preferred(mesh->face_ptr, max_n_total * sizeof(hsize_t));
        gpu_advise_gpu_preferred(mesh->neighbor_cell, max_faces * sizeof(int));
        gpu_advise_gpu_preferred(mesh->face_area, max_faces * sizeof(double));
        gpu_advise_gpu_preferred(mesh->cell_status, max_n_total * sizeof(Status));

        // moving mesh per-cell arrays (zero-initialized for first dt_CFL call)
#ifdef MOVING_MESH
        mesh->v_mesh      = gpu_calloc<POINT_TYPE>(n_hydro);
        mesh->old_volumes = gpu_calloc<double>(n_hydro);
#endif

        // scratch buffers for ghost-augmented point arrays
        mesh->scratch_pts     = gpu_alloc<POINT_TYPE>(max_n_total);
        mesh->scratch_pts_cap = max_n_total;

        // scratch buffer for moved seed positions (only n_hydro needed)
        mesh->scratch_move     = gpu_alloc<POINT_TYPE>(n_hydro);
        mesh->scratch_move_cap = n_hydro;

        // KNN cache
        mesh->knn = knn::init_once((int)n_hydro);

        return mesh;
    }

    void free_mesh(VMesh* mesh) {
        if (mesh) {
            gpu_free(mesh->seeds);
            gpu_free(mesh->com);
            gpu_free(mesh->volumes);
            gpu_free(mesh->face_counts);
            gpu_free(mesh->face_ptr);
            gpu_free(mesh->neighbor_cell);
            gpu_free(mesh->face_area);
#ifdef MOVING_MESH
            gpu_free(mesh->f_mid_local);
            gpu_free(mesh->v_mesh);
            gpu_free(mesh->old_volumes);
#endif
            gpu_free(mesh->ghost_ids);
            gpu_free(mesh->cell_status);

            gpu_free(mesh->scratch_pts);
            mesh->scratch_pts     = nullptr;
            mesh->scratch_pts_cap = 0;
            gpu_free(mesh->scratch_move);
            mesh->scratch_move     = nullptr;
            mesh->scratch_move_cap = 0;

            if (mesh->knn) { knn::knn_free(&mesh->knn); }

            gpu_free(mesh);
        }
    }

    void compute_mesh(VMesh* mesh, POINT_TYPE* pts_data, int num_points) {

        // grid-bucket-sort all points so neighbour queries become local
        PROFILE_START("KNN (par)");
        knn::prepare(mesh->knn, (const POINT_TYPE*)pts_data, num_points);
        PROFILE_END("KNN (par)");

        PROFILE_START("VORONOI (par)");

        if ((hsize_t)num_points > mesh->cell_capacity) {
            std::cerr << "VORONOI: Error! cell count " << num_points << " exceeds pre-allocated capacity "
                      << mesh->cell_capacity << ". Increase ghost headroom." << std::endl;
            exit(EXIT_FAILURE);
        }

        // reset per-cell counters before the kernel rewrites them
        mesh->n_seeds   = (hsize_t)num_points;
        mesh->num_faces = 0;
        mesh->n_hydro   = 0;
        gpu_memset(mesh->face_counts, 0, num_points * sizeof(hsize_t));
        gpu_memset(mesh->face_ptr, 0, num_points * sizeof(hsize_t));

        // mark all cells as not-yet-converged; ConvexCell constructor flips them to success
#ifndef CPU_DEBUG
        {
            int tpb    = _MESH_BLOCK_SIZE_;
            int blocks = (num_points + tpb - 1) / tpb;
            kernel_init_cell_status<<<blocks, tpb>>>(num_points, mesh->cell_status);
            GPU_LAUNCH_CHECK();
        }
#else
        for (int i = 0; i < num_points; i++)
            mesh->cell_status[i] = security_radius_not_reached;
#endif

        // build cells (fast tier + slow tier + CPU fallback)
        compute_cells(num_points, mesh->knn, mesh->cell_status, mesh, mesh->knn->d_permutation);

        PROFILE_END("VORONOI (par)");
    }

    // ============================================================
    // Main routines
    // ============================================================

    // managed-memory scratch counters and the compact failed-cell list (allocated lazily)
#ifndef CPU_DEBUG
    static hsize_t* d_face_offset             = nullptr;
    static int*     d_overflow_flag           = nullptr;
    static int*     d_fail_count              = nullptr;
    static int*     d_failed_indices          = nullptr; // seed_ids the fast tier did not finish
    static int*     d_failed_count            = nullptr; // length of d_failed_indices
    static int      d_failed_indices_capacity = 0;
#endif

    void
    compute_cells(int N_seedpts, knn_problem* knn, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original) {

#ifndef CPU_DEBUG
        const int threadsPerBlock = _VORO_BLOCK_SIZE_;
        const int blocksPerGrid   = (N_seedpts + threadsPerBlock - 1) / threadsPerBlock;

        // lazy allocate / grow scratch buffers
        if (!d_face_offset) {
            d_face_offset   = gpu_calloc<hsize_t>(1);
            d_overflow_flag = gpu_calloc<int>(1);
            d_failed_count  = gpu_calloc<int>(1);
        }
        if (d_failed_indices_capacity < N_seedpts) {
            if (d_failed_indices) gpu_free(d_failed_indices);
            d_failed_indices          = gpu_alloc<int>(N_seedpts);
            d_failed_indices_capacity = N_seedpts;
        }
        gpu_memset(d_face_offset, 0, sizeof(hsize_t));
        gpu_memset(d_overflow_flag, 0, sizeof(int));
        gpu_memset(d_failed_count, 0, sizeof(int));

        // fast tier: small per-thread arrays, higher occupancy
        PROFILE_GPU_START("kernel_compute_voronoi_cells_fast");
        kernel_compute_voronoi_cells_fast<<<blocksPerGrid, threadsPerBlock>>>(N_seedpts,
                                                                              (double*)knn->d_stored_points,
                                                                              knn,
                                                                              stat,
                                                                              mesh,
                                                                              sorted_to_original,
                                                                              d_face_offset,
                                                                              d_overflow_flag);
        PROFILE_GPU_END("kernel_compute_voronoi_cells_fast");

        // compact failed seed_ids so the slow kernel runs one thread per failed cell
        {
            int tpb        = _MESH_BLOCK_SIZE_;
            int collect_bl = (N_seedpts + tpb - 1) / tpb;
            PROFILE_GPU_START("kernel_collect_failed_cells");
            kernel_collect_failed_cells<<<collect_bl, tpb>>>(N_seedpts, stat, d_failed_indices, d_failed_count);
            PROFILE_GPU_END("kernel_collect_failed_cells");
        }

        GPU_SYNC();
        int n_failed = *d_failed_count;

        std::cout << "VORONOI: Generated " << N_seedpts << " cells. ("
                  << (100.0 * n_failed / (double)N_seedpts) << "% slow tier)" << std::endl;

        // slow tier: full-size arrays, runs only on the compacted failed cells
        if (n_failed > 0) {
            int slow_blocks = (n_failed + threadsPerBlock - 1) / threadsPerBlock;
            PROFILE_GPU_START("kernel_compute_voronoi_cells_slow");
            kernel_compute_voronoi_cells_slow<<<slow_blocks, threadsPerBlock>>>(n_failed,
                                                                                N_seedpts,
                                                                                d_failed_indices,
                                                                                (double*)knn->d_stored_points,
                                                                                knn,
                                                                                stat,
                                                                                mesh,
                                                                                sorted_to_original,
                                                                                d_face_offset,
                                                                                d_overflow_flag);
            PROFILE_GPU_END("kernel_compute_voronoi_cells_slow");
        }

        GPU_SYNC();
        hsize_t total_faces = *d_face_offset;
        int     overflow    = *d_overflow_flag;
        mesh->num_faces     = total_faces;

        if (overflow) {
            std::cerr << "VORONOI: Error! face offset exceeds pre-allocated face capacity " << mesh->face_capacity
                      << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
            exit(EXIT_FAILURE);
        }
#else
        // CPU path: serial (or OpenMP) loop over all cells, using slow-tier limits
        unsigned long long face_offset        = 0;
        int                face_overflow_flag = 0;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int seed_id = 0; seed_id < N_seedpts; seed_id++) {
            if (face_overflow_flag) continue;
            compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_>(seed_id,
                                                               N_seedpts,
                                                               (double*)knn->d_stored_points,
                                                               knn,
                                                               stat,
                                                               mesh,
                                                               sorted_to_original,
                                                               &face_offset,
                                                               &face_overflow_flag);
        }
        mesh->num_faces = (hsize_t)face_offset;

        if (face_overflow_flag) {
            std::cerr << "VORONOI: Error! face offset exceeds pre-allocated face capacity " << mesh->face_capacity
                      << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
            exit(EXIT_FAILURE);
        }
#endif

        cpu_fallback_failed_cells(N_seedpts, (double*)knn->d_stored_points, stat, mesh, sorted_to_original);
    }

    // ============================================================
    // CUDA kernel wrappers
    // ============================================================
#ifndef CPU_DEBUG

    GLOBAL void kernel_init_cell_status(int n, Status* stat) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        stat[i] = security_radius_not_reached;
    }

    GLOBAL void kernel_count_failures(int n, const Status* stat, int* fail_count) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        if (stat[i] != success) { atomicAdd(fail_count, 1); }
    }

    GLOBAL void kernel_collect_failed_cells(int n, const Status* stat, int* failed_indices, int* failed_count) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        if (stat[i] != success) {
            int slot              = atomicAdd(failed_count, 1);
            failed_indices[slot] = i;
        }
    }

    GLOBAL __launch_bounds__(_VORO_BLOCK_SIZE_, 16)
        void kernel_compute_voronoi_cells_fast(int                 N_seedpts,
                                               double*             d_stored_points,
                                               const knn_problem*  knn,
                                               Status*             gpu_stat,
                                               VMesh*              mesh,
                                               const unsigned int* sorted_to_original,
                                               hsize_t*            face_offset,
                                               int*                overflow_flag) {
        int seed_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (seed_id >= N_seedpts) return;
        compute_single_voronoi_cell<_FAST_K_, _FAST_MAX_P_, _FAST_MAX_T_>(seed_id,
                                                                          N_seedpts,
                                                                          d_stored_points,
                                                                          knn,
                                                                          gpu_stat,
                                                                          mesh,
                                                                          sorted_to_original,
                                                                          (unsigned long long*)face_offset,
                                                                          overflow_flag);
    }

    GLOBAL __launch_bounds__(_VORO_BLOCK_SIZE_, 8)
        void kernel_compute_voronoi_cells_slow(int                 n_failed,
                                               int                 N_seedpts,
                                               const int*          failed_indices,
                                               double*             d_stored_points,
                                               const knn_problem*  knn,
                                               Status*             gpu_stat,
                                               VMesh*              mesh,
                                               const unsigned int* sorted_to_original,
                                               hsize_t*            face_offset,
                                               int*                overflow_flag) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_failed) return;
        int seed_id = failed_indices[i];
        compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_>(seed_id,
                                                           N_seedpts,
                                                           d_stored_points,
                                                           knn,
                                                           gpu_stat,
                                                           mesh,
                                                           sorted_to_original,
                                                           (unsigned long long*)face_offset,
                                                           overflow_flag);
    }

#endif // !CPU_DEBUG

    // ============================================================
    // Per-cell work functions (called by kernels and CPU loops)
    // ============================================================

    template <int K, int MAX_P, int MAX_T>
    HD void compute_single_voronoi_cell(int                 seed_id,
                                        int                 N_seedpts,
                                        double*             d_stored_points,
                                        const knn_problem*  knn,
                                        Status*             stat,
                                        VMesh*              mesh,
                                        const unsigned int* sorted_to_original,
                                        unsigned long long* face_offset,
                                        int*                overflow_flag) {

        // K nearest neighbours sorted by distance
        unsigned int local_knn[K];
        knn::knn_for_point<K>(seed_id, knn, local_knn);

        // clip the bounding cell by each neighbour's bisector, in distance order
        BasicConvexCell<MAX_P, MAX_T> cell(seed_id, d_stored_points, &(stat[seed_id]), mesh->buff);

        for (int v = 0; v < K; v++) {
            unsigned int z = local_knn[v];
            cell.clip_by_plane(z);
            if (stat[seed_id] != success) { break; }

            // early out once the cell is enclosed by the security sphere
            if (v >= 2 * DIMENSION &&
                cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * z))) {
                break;
            }
        }

        // K wasn't enough — fall through to slow tier / CPU fallback
        if (!cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * local_knn[K - 1]))) {
            stat[seed_id] = security_radius_not_reached;
        }

        if (stat[seed_id] == success) {
            // map sorted-grid index back to the original input index
            hsize_t original_id = (hsize_t)sorted_to_original[seed_id];

            // claim contiguous face slots via the global atomic counter
            int fc                         = count_cell_faces(cell);
            mesh->face_counts[original_id] = (hsize_t)fc;
            hsize_t my_offset              = (hsize_t)portable_atomicAdd(face_offset, (unsigned long long)fc);

            if (my_offset + (hsize_t)fc > mesh->face_capacity) {
                portable_atomicExch(overflow_flag, 1);
                return;
            }
            mesh->face_ptr[original_id] = my_offset;

            // write volume, centroid, and per-face data to the global mesh
            extract_cell_all(cell, mesh, original_id);

            // remap neighbour ids in this face range to original input indices
            hsize_t face_end = my_offset + (hsize_t)fc;
            for (hsize_t fi = my_offset; fi < face_end; fi++) {
                int& nid = mesh->neighbor_cell[fi];
                if (nid >= 0 && nid < N_seedpts) { nid = (int)sorted_to_original[nid]; }
            }
        }
    }

    // ============================================================
    // CPU fallback for cells that failed both GPU tiers
    // ============================================================

    void cpu_fallback_failed_cells(
        int N_seedpts, double* d_stored_points, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original) {
        int num_failed = 0;

        // count failed cells, prefetch status to host
#ifndef CPU_DEBUG
        if (!d_fail_count) d_fail_count = gpu_calloc<int>(1);
        gpu_memset(d_fail_count, 0, sizeof(int));
        {
            int tpb    = _MESH_BLOCK_SIZE_;
            int blocks = (N_seedpts + tpb - 1) / tpb;
            kernel_count_failures<<<blocks, tpb>>>(N_seedpts, stat, d_fail_count);
        }
        GPU_SYNC();
        num_failed = *d_fail_count;
        if (num_failed == 0) return;
        gpu_prefetch_to_cpu(stat, N_seedpts * sizeof(Status));
#else
        for (int i = 0; i < N_seedpts; i++) {
            if (stat[i] != success) { num_failed++; }
        }
        if (num_failed == 0) return;
#endif

        std::cout << "VORONOI: " << num_failed << " cells failed, retrying with fallback..." << std::endl;

        // retry each failed cell on the CPU, perturbing the seed if degeneracies persist
        for (int i = 0; i < N_seedpts; i++) {
            if (stat[i] == success) continue;

            // overflow-style failures aren't recoverable here — abort with diagnostic
            Status original_status = stat[i];
            if (original_status != security_radius_not_reached && original_status != needs_exact_predicates) {
                std::cerr << "VORONOI: cell " << i << " failed with unrecoverable status: " << original_status
                          << std::endl;
                exit(EXIT_FAILURE);
            }
            std::cout << "VORONOI: cell " << i << " failed with status: " << original_status << std::endl;

            // sort all other points by distance to seed (exhaustive clip in distance order)
            double4 seed_pos = point_from_ptr(d_stored_points + DIMENSION * i);

            std::vector<std::pair<double, int>> dists;
            dists.reserve(N_seedpts - 1);
            for (int j = 0; j < N_seedpts; j++) {
                if (j == i) continue;
                double4 other = point_from_ptr(d_stored_points + DIMENSION * j);
                double  dx    = other.x - seed_pos.x;
                double  dy    = other.y - seed_pos.y;
                double  dz    = other.z - seed_pos.z;
                double  dist2 = dx * dx + dy * dy + dz * dz;
                dists.push_back({dist2, j});
            }
            std::sort(dists.begin(), dists.end());

            bool         cell_ok       = false;
            int          max_perturb   = 9;
            double       perturb_scale = 1e-13;

            // try unperturbed first, then perturb the seed with growing magnitude
            for (int attempt = 0; attempt <= max_perturb; attempt++) {
                if (attempt > 0) {
                    // hash-based pseudo-random offset to break exact-arithmetic degeneracies
                    unsigned int hash = (unsigned int)(i * 2654435761u + attempt * 40503u);
                    for (int d = 0; d < DIMENSION; d++) {
                        hash                               = hash * 1103515245u + 12345u;
                        double r                           = ((double)(hash & 0xFFFF) / 32768.0 - 1.0);
                        d_stored_points[DIMENSION * i + d] = (d == 0   ? seed_pos.x
                                                              : d == 1 ? seed_pos.y
                                                                       : seed_pos.z) +
                                                             r * perturb_scale;
                    }
                    perturb_scale *= 10.0;
                }

                // run cell construction over all neighbours in distance order
                Status     fallback_status = success;
                ConvexCell cell(i, d_stored_points, &fallback_status, mesh->buff);
                for (size_t di = 0; di < dists.size(); di++) {
                    int j = dists[di].second;
                    cell.clip_by_plane(j);
                    if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * j))) break;
                    if (fallback_status != success) break;
                }

                // restore the original seed so later steps don't see the perturbation
                if (attempt > 0) {
                    d_stored_points[DIMENSION * i + 0] = seed_pos.x;
                    d_stored_points[DIMENSION * i + 1] = seed_pos.y;
#ifdef dim_3D
                    d_stored_points[DIMENSION * i + 2] = seed_pos.z;
#endif
                }

                if (fallback_status == success) {
                    // serial append into the mesh (host-only path, no atomics needed)
                    hsize_t original_id = (hsize_t)sorted_to_original[i];
                    int     fc          = count_cell_faces(cell);
                    ensure_face_capacity(mesh, mesh->num_faces + fc);
                    hsize_t face_start             = mesh->num_faces;
                    mesh->face_ptr[original_id]    = face_start;
                    mesh->face_counts[original_id] = (hsize_t)fc;
                    extract_cell_all(cell, mesh, original_id);
                    mesh->num_faces += (hsize_t)fc;

                    // remap neighbour ids (same as the GPU path)
                    for (hsize_t fi = face_start; fi < face_start + (hsize_t)fc; fi++) {
                        int& nid = mesh->neighbor_cell[fi];
                        if (nid >= 0 && (hsize_t)nid < (hsize_t)N_seedpts) { nid = (int)sorted_to_original[nid]; }
                    }
                    std::cout << "VORONOI: cell " << i << " fallback (perturbed, attempt " << attempt << ") succeeded."
                              << std::endl;
                    cell_ok = true;
                    break;
                }
            }

            if (!cell_ok) {
                std::cerr << "VORONOI: cell " << i << " all fallback attempts FAILED, aborting." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

} // namespace voronoi
