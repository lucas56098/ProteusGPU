#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "../profiler/profiler.h"
#include "cell.h"
#include "voronoi.h"

#include "cell.cu"
#include "geometry.cu"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

namespace voronoi {

    // forward declarations
    void compute_cells(int, knn_problem*, Status*, VMesh*, const unsigned int*);
    void cpu_fallback_failed_cells(int, double*, Status*, VMesh*, const unsigned int*);
    void cpu_compute_cell(int, int, int, double*, const knn_problem*, Status*, VMesh*, const unsigned int*);

    HD void compute_single_voronoi_cell(
        int, int, double*, const knn_problem*, Status*, VMesh*, const unsigned int*, unsigned long long*, int*);

#ifndef CPU_DEBUG
    // kernels
    GLOBAL void kernel_init_cell_status(int, Status*);
    GLOBAL void kernel_count_failures(int, const Status*, int*);
    GLOBAL void kernel_compute_voronoi_cells(
        int, double*, const knn_problem*, Status*, VMesh*, const unsigned int*, hsize_t*, int*);
#endif

    // ============================================================
    // allocate, free, compute
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

        // per-cell arrays
        mesh->seeds       = gpu_calloc<double3>(max_n_total);
        mesh->com         = gpu_calloc<double3>(max_n_total);
        mesh->volumes     = gpu_calloc<double>(max_n_total);
        mesh->face_counts = gpu_calloc<hsize_t>(max_n_total);
        mesh->face_ptr    = gpu_calloc<hsize_t>(max_n_total);

        // per-face arrays
        mesh->neighbor_cell = gpu_alloc<int>(max_faces);
        mesh->face_area     = gpu_alloc<compact_t>(max_faces);
#ifdef MOVING_MESH
        mesh->f_mid_local = gpu_alloc<compact_t>(max_faces * (DIMENSION - 1));
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
        gpu_advise_gpu_preferred(mesh->face_area, max_faces * sizeof(compact_t));
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
#ifdef DEBUG_MODE
        std::cout << "VORONOI: Computing Voronoi mesh..." << std::endl;
#endif

        // -------- KNN --------
        PROFILE_START("KNN (par)");
        knn::prepare(mesh->knn, (const POINT_TYPE*)pts_data, num_points);
#ifdef DEBUG_MODE
        std::cout << "KNN: problem initialized." << std::endl;
#endif
        PROFILE_END("KNN (par)");

        // -------- VORONOI --------
        PROFILE_START("VORONOI (par)");

        if ((hsize_t)num_points > mesh->cell_capacity) {
            std::cerr << "VORONOI: Error! cell count " << num_points << " exceeds pre-allocated capacity "
                      << mesh->cell_capacity << ". Increase ghost headroom." << std::endl;
            exit(EXIT_FAILURE);
        }

        mesh->n_seeds   = (hsize_t)num_points;
        mesh->num_faces = 0;
        mesh->n_hydro   = 0;
        gpu_memset(mesh->face_counts, 0, num_points * sizeof(hsize_t));
        gpu_memset(mesh->face_ptr, 0, num_points * sizeof(hsize_t));

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
        compute_cells(num_points, mesh->knn, mesh->cell_status, mesh, mesh->knn->d_permutation);

        PROFILE_END("VORONOI (par)");
    }

    // ============================================================
    // Cell construction (GPU kernel call or CPU loops)
    // ============================================================

    // host-side pointers to managed atomic counters (allocated lazily)
#ifndef CPU_DEBUG
    static hsize_t* d_face_offset   = nullptr;
    static int*     d_overflow_flag = nullptr;
    static int*     d_fail_count    = nullptr;
#endif

    void
    compute_cells(int N_seedpts, knn_problem* knn, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original) {

        int threadsPerBlock = _VORO_BLOCK_SIZE_;
        int blocksPerGrid   = (N_seedpts + threadsPerBlock - 1) / threadsPerBlock;

#ifdef DEBUG_MODE
        std::cout << "VORONOI: computing cells" << std::endl;
#endif

#ifndef CPU_DEBUG
        if (!d_face_offset) {
            d_face_offset   = gpu_calloc<hsize_t>(1);
            d_overflow_flag = gpu_calloc<int>(1);
        }
        gpu_memset(d_face_offset, 0, sizeof(hsize_t));
        gpu_memset(d_overflow_flag, 0, sizeof(int));

        PROFILE_GPU_START("kernel_compute_voronoi_cells");
        kernel_compute_voronoi_cells<<<blocksPerGrid, threadsPerBlock>>>(N_seedpts,
                                                                         (double*)knn->d_stored_points,
                                                                         knn,
                                                                         stat,
                                                                         mesh,
                                                                         sorted_to_original,
                                                                         d_face_offset,
                                                                         d_overflow_flag);
        PROFILE_GPU_END("kernel_compute_voronoi_cells");

        hsize_t total_faces;
        int     overflow;
        gpu_memcpy(&total_faces, d_face_offset, sizeof(hsize_t));
        gpu_memcpy(&overflow, d_overflow_flag, sizeof(int));
        mesh->num_faces = total_faces;

        if (overflow) {
            std::cerr << "VORONOI: Error! face offset exceeds pre-allocated face capacity " << mesh->face_capacity
                      << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
            exit(EXIT_FAILURE);
        }
#else
        cpu_compute_cell(blocksPerGrid,
                         threadsPerBlock,
                         N_seedpts,
                         (double*)knn->d_stored_points,
                         knn,
                         stat,
                         mesh,
                         sorted_to_original);
#endif

#ifdef DEBUG_MODE
        std::cout << "\nVORONOI: cells computed" << std::endl;
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

    GLOBAL __launch_bounds__(64, 8) void kernel_compute_voronoi_cells(int                 N_seedpts,
                                                                      double*             d_stored_points,
                                                                      const knn_problem*  knn,
                                                                      Status*             gpu_stat,
                                                                      VMesh*              mesh,
                                                                      const unsigned int* sorted_to_original,
                                                                      hsize_t*            face_offset,
                                                                      int*                overflow_flag) {
        int seed_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (seed_id >= N_seedpts) return;
        compute_single_voronoi_cell(seed_id,
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
    // Per-cell Voronoi construction (called by kernel and CPU loop)
    // ============================================================

    HD void compute_single_voronoi_cell(int                 seed_id,
                                        int                 N_seedpts,
                                        double*             d_stored_points,
                                        const knn_problem*  knn,
                                        Status*             stat,
                                        VMesh*              mesh,
                                        const unsigned int* sorted_to_original,
                                        unsigned long long* face_offset,
                                        int*                overflow_flag) {
        unsigned int local_knn[_K_];
        knn::knn_for_point(seed_id, knn, local_knn);

        ConvexCell cell(seed_id, d_stored_points, &(stat[seed_id]));

        for (int v = 0; v < _K_; v++) {
            unsigned int z = local_knn[v];
            cell.clip_by_plane(z);
            if (stat[seed_id] != success) { break; }
            if (v >= 2 * DIMENSION &&
                cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * z))) {
                break;
            }
        }
        if (!cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * local_knn[_K_ - 1]))) {
            stat[seed_id] = security_radius_not_reached;
        }

        if (stat[seed_id] == success) {
            hsize_t original_id = (hsize_t)sorted_to_original[seed_id];

            int fc                         = count_cell_faces(cell);
            mesh->face_counts[original_id] = (hsize_t)fc;

            hsize_t my_offset = (hsize_t)portable_atomicAdd(face_offset, (unsigned long long)fc);

            if (my_offset + (hsize_t)fc > mesh->face_capacity) {
                portable_atomicExch(overflow_flag, 1);
                return;
            }

            mesh->face_ptr[original_id] = my_offset;

            extract_cell_all(cell, mesh, original_id);

            hsize_t face_end = my_offset + (hsize_t)fc;
            for (hsize_t fi = my_offset; fi < face_end; fi++) {
                int& nid = mesh->neighbor_cell[fi];
                if (nid >= 0 && nid < N_seedpts) { nid = (int)sorted_to_original[nid]; }
            }
        }
    }

    // ============================================================
    // CPU helpers (fallback for failed cells, host-side cell loop)
    // ============================================================

    void cpu_fallback_failed_cells(
        int N_seedpts, double* d_stored_points, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original) {
        int num_failed = 0;

#ifndef CPU_DEBUG
        if (!d_fail_count) d_fail_count = gpu_calloc<int>(1);
        gpu_memset(d_fail_count, 0, sizeof(int));
        {
            int tpb    = _MESH_BLOCK_SIZE_;
            int blocks = (N_seedpts + tpb - 1) / tpb;
            kernel_count_failures<<<blocks, tpb>>>(N_seedpts, stat, d_fail_count);
        }
        gpu_memcpy(&num_failed, d_fail_count, sizeof(int));
        if (num_failed == 0) return;
        gpu_prefetch_to_cpu(stat, N_seedpts * sizeof(Status));
#else
        for (int i = 0; i < N_seedpts; i++) {
            if (stat[i] != success) { num_failed++; }
        }
        if (num_failed == 0) return;
#endif

        std::cout << "VORONOI: " << num_failed << " cells failed, retrying with fallback..." << std::endl;

        for (int i = 0; i < N_seedpts; i++) {
            if (stat[i] == success) continue;

            Status original_status = stat[i];

            if (original_status != security_radius_not_reached && original_status != needs_exact_predicates) {
                std::cerr << "VORONOI: cell " << i << " failed with unrecoverable status: " << original_status
                          << std::endl;
                exit(EXIT_FAILURE);
            }

            std::cout << "VORONOI: cell " << i << " failed with status: " << original_status << std::endl;

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

            bool cell_ok = false;

            int    max_perturb   = 9;
            double perturb_scale = 1e-13;

            for (int attempt = 0; attempt <= max_perturb; attempt++) {
                if (attempt > 0) {
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

                Status     fallback_status = success;
                ConvexCell cell(i, d_stored_points, &fallback_status);

                for (size_t di = 0; di < dists.size(); di++) {
                    int j = dists[di].second;
                    cell.clip_by_plane(j);
                    if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * j))) break;
                    if (fallback_status != success) break;
                }

                if (attempt > 0) {
                    d_stored_points[DIMENSION * i + 0] = seed_pos.x;
                    d_stored_points[DIMENSION * i + 1] = seed_pos.y;
#ifdef dim_3D
                    d_stored_points[DIMENSION * i + 2] = seed_pos.z;
#endif
                }

                if (fallback_status == success) {
                    hsize_t original_id = (hsize_t)sorted_to_original[i];
                    int     fc          = count_cell_faces(cell);
                    ensure_face_capacity(mesh, mesh->num_faces + fc);
                    hsize_t face_start             = mesh->num_faces;
                    mesh->face_ptr[original_id]    = face_start;
                    mesh->face_counts[original_id] = (hsize_t)fc;
                    extract_cell_all(cell, mesh, original_id);
                    mesh->num_faces += (hsize_t)fc;
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

    void cpu_compute_cell(int                 blocksPerGrid,
                          int                 threadsPerBlock,
                          int                 N_seedpts,
                          double*             d_stored_points,
                          const knn_problem*  knn,
                          Status*             gpu_stat,
                          VMesh*              mesh,
                          const unsigned int* sorted_to_original) {

        (void)blocksPerGrid;
        (void)threadsPerBlock;

        unsigned long long face_offset        = 0;
        int                face_overflow_flag = 0;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, _VORO_BLOCK_SIZE_)
#endif
        for (int seed_id = 0; seed_id < N_seedpts; seed_id++) {
            if (face_overflow_flag) continue;
#ifdef DEBUG_MODE
            if (seed_id % 10000 == 0 || seed_id == N_seedpts - 1) {
#ifdef USE_OPENMP
#pragma omp critical(voro_progress_print)
#endif
                std::cout << "\rVORONOI: processing cell " << seed_id + 1 << " / " << N_seedpts << std::flush;
            }
#endif

            compute_single_voronoi_cell(seed_id,
                                        N_seedpts,
                                        d_stored_points,
                                        knn,
                                        gpu_stat,
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
    }

} // namespace voronoi
