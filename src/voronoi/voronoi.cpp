#include "voronoi.h"
#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "../profiler/profiler.h"
#include "cell.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

namespace voronoi {

    VMesh* allocate_mesh(hsize_t n_hydro) {

        // worst-case ghost estimate with 2x safety margin
        double  ghost_frac  = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        hsize_t max_ghosts  = (hsize_t)(2.0 * ghost_frac * n_hydro) + 1;
        hsize_t max_n_total = n_hydro + max_ghosts;
        hsize_t max_faces   = max_n_total * _FACE_CAPACITY_MULT_;

        // allocate VMesh with fixed worst-case capacities
        VMesh* mesh          = (VMesh*)malloc(sizeof(VMesh));
        mesh->n_seeds        = 0;
        mesh->n_hydro        = 0;
        mesh->num_faces      = 0;
        mesh->cell_capacity  = max_n_total;
        mesh->face_capacity  = max_faces;
        mesh->ghost_capacity = max_ghosts;

        // per-cell arrays
        mesh->seeds       = (double3*)calloc(max_n_total, sizeof(double3));
        mesh->com         = (double3*)calloc(max_n_total, sizeof(double3));
        mesh->volumes     = (double*)calloc(max_n_total, sizeof(double));
        mesh->face_counts = (hsize_t*)calloc(max_n_total, sizeof(hsize_t));
        mesh->face_ptr    = (hsize_t*)calloc(max_n_total, sizeof(hsize_t));

        // per-face arrays
        mesh->neighbor_cell = (int*)malloc(max_faces * sizeof(int));
        mesh->face_area     = (compact_t*)malloc(max_faces * sizeof(compact_t));
#ifdef MOVING_MESH
        mesh->f_mid_local = (compact_t*)malloc(max_faces * (DIMENSION - 1) * sizeof(compact_t));
#endif

        // ghost mapping
        mesh->ghost_ids = (hsize_t*)malloc(max_ghosts * sizeof(hsize_t));

        // per-cell status flags (reused each timestep)
        mesh->cell_status = (Status*)malloc(max_n_total * sizeof(Status));

        // moving mesh per-cell arrays (zero-initialized for first dt_CFL call)
#ifdef MOVING_MESH
        mesh->v_mesh      = (POINT_TYPE*)calloc(n_hydro, sizeof(POINT_TYPE));
        mesh->old_volumes = (double*)calloc(n_hydro, sizeof(double));
#endif

        // scratch buffers for ghost-augmented point arrays
        mesh->scratch_pts     = (POINT_TYPE*)malloc(max_n_total * sizeof(POINT_TYPE));
        mesh->scratch_pts_cap = max_n_total;

        // scratch buffer for moved seed positions (only n_hydro needed)
        mesh->scratch_move     = (POINT_TYPE*)malloc(n_hydro * sizeof(POINT_TYPE));
        mesh->scratch_move_cap = n_hydro;

        // KNN cache
        mesh->knn = knn::init_once((int)n_hydro);

        return mesh;
    }

    // Free VMesh and all persistent voronoi buffers.
    void free_mesh(VMesh* mesh) {
        if (mesh) {
            free(mesh->seeds);
            free(mesh->com);
            free(mesh->volumes);
            free(mesh->face_counts);
            free(mesh->face_ptr);
            free(mesh->neighbor_cell);
            free(mesh->face_area);
#ifdef MOVING_MESH
            free(mesh->f_mid_local);
#endif
#ifdef MOVING_MESH
            free(mesh->v_mesh);
            free(mesh->old_volumes);
#endif
            free(mesh->ghost_ids);
            free(mesh->cell_status);
            free(mesh);
        }

        free(mesh->scratch_pts);
        mesh->scratch_pts     = nullptr;
        mesh->scratch_pts_cap = 0;
        free(mesh->scratch_move);
        mesh->scratch_move     = nullptr;
        mesh->scratch_move_cap = 0;

        if (mesh->knn) { knn::knn_free(&mesh->knn); }
    }

    // computes the mesh
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

        // verify capacity
        if ((hsize_t)num_points > mesh->cell_capacity) {
            std::cerr << "VORONOI: Error! cell count " << num_points << " exceeds pre-allocated capacity "
                      << mesh->cell_capacity << ". Increase ghost headroom." << std::endl;
            exit(EXIT_FAILURE);
        }

        // reset mesh for new computation
        mesh->n_seeds   = (hsize_t)num_points;
        mesh->num_faces = 0;
        mesh->n_hydro   = 0;
        memset(mesh->face_counts, 0, num_points * sizeof(hsize_t));
        memset(mesh->face_ptr, 0, num_points * sizeof(hsize_t));

        // compute voronoi cells
        for (int i = 0; i < num_points; i++)
            mesh->cell_status[i] = security_radius_not_reached;
        compute_cells(num_points, mesh->knn, mesh->cell_status, mesh, mesh->knn->d_permutation);

        PROFILE_END("VORONOI (par)");
    }

    // compute voronoi cells from knn results and store in VMesh
    void
    compute_cells(int N_seedpts, knn_problem* knn, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original) {

        // compute cell kernel
        int threadsPerBlock = _VORO_BLOCK_SIZE_;
        int blocksPerGrid   = N_seedpts / threadsPerBlock + 1;

#ifdef DEBUG_MODE
        std::cout << "VORONOI: computing cells" << std::endl;
#endif
        cpu_compute_cell(blocksPerGrid,
                         threadsPerBlock,
                         N_seedpts,
                         (double*)knn->d_stored_points,
                         knn,
                         stat,
                         mesh,
                         sorted_to_original);
#ifdef DEBUG_MODE
        std::cout << "\nVORONOI: cells computed" << std::endl;
#endif

        // face arrays are kept at face_capacity (no shrinkage) to avoid
        // heap fragmentation from repeated realloc cycles in moving-mesh mode.

        // check if any cells failed and retry with cpu fallback
        cpu_fallback_failed_cells(N_seedpts, (double*)knn->d_stored_points, stat, mesh, sorted_to_original);
    }

    // cpu fallback for cells that failed during knn-based construction
    void cpu_fallback_failed_cells(
        int N_seedpts, double* d_stored_points, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original) {
        int num_failed = 0;
        for (int i = 0; i < N_seedpts; i++) {
            if (stat[i] != success) { num_failed++; }
        }
        if (num_failed == 0) return;

        std::cout << "VORONOI: " << num_failed << " cells failed, retrying with fallback..." << std::endl;

        for (int i = 0; i < N_seedpts; i++) {
            if (stat[i] == success) continue;

            Status original_status = stat[i];

            // for unexpected errors (triangle/vertex overflow, inconsistent boundary) abort
            if (original_status != security_radius_not_reached && original_status != needs_exact_predicates) {
                std::cerr << "VORONOI: cell " << i << " failed with unrecoverable status: " << original_status
                          << std::endl;
                exit(EXIT_FAILURE);
            }

            std::cout << "VORONOI: cell " << i << " failed with status: " << original_status << std::endl;

            // sort all other seed indices by distance to this seed
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

            // needs exact predicates or more ngb: retry with increasing perturbation to break degeneracy
            int    max_perturb   = 5;
            double perturb_scale = 1e-13;

            for (int attempt = 0; attempt <= max_perturb; attempt++) {
                if (attempt > 0) {
                    // deterministic pseudo-random perturbation based on seed id and attempt
                    unsigned int hash = (unsigned int)(i * 2654435761u + attempt * 40503u);
                    for (int d = 0; d < DIMENSION; d++) {
                        hash                               = hash * 1103515245u + 12345u;
                        double r                           = ((double)(hash & 0xFFFF) / 32768.0 - 1.0); // [-1, 1]
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

                // restore original position after perturbation
                if (attempt > 0) {
                    d_stored_points[DIMENSION * i + 0] = seed_pos.x;
                    d_stored_points[DIMENSION * i + 1] = seed_pos.y;
#ifdef dim_3D
                    d_stored_points[DIMENSION * i + 2] = seed_pos.z;
#endif
                }

                if (fallback_status == success) {
                    hsize_t original_id = (hsize_t)sorted_to_original[i];
                    double4 vertices[_MAX_T_];
                    for (int vi = 0; vi < cell.nb_t; vi++)
                        vertices[vi] = cell.compute_vertex_point(cell.triangle[vi], true);
                    int fc = extract_cell_data(cell, vertices, mesh, original_id);
                    ensure_face_capacity(mesh, mesh->num_faces + fc);
                    hsize_t face_start             = mesh->num_faces;
                    mesh->face_ptr[original_id]    = face_start;
                    mesh->face_counts[original_id] = (hsize_t)fc;
                    extract_cell_faces(cell, vertices, mesh, original_id);
                    mesh->num_faces += (hsize_t)fc;
                    // convert neighbor IDs from sorted to original indexing
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
        // face arrays are kept at face_capacity (no shrinkage)
    }

#ifdef CPU_DEBUG
    // cpu debug version of cell computation kernel
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

        // face arrays are pre-allocated with generous capacity (n_pts * 16) by caller
        hsize_t face_offset        = 0;
        int     face_overflow_flag = 0;

        // ──── single pass (parallel): KNN + cell construction + atomic face-slot reservation ────
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

            // inline KNN (computed once per cell)
            unsigned int local_knn[_K_];
            knn::knn_for_point(seed_id, knn, local_knn);

            // construct Voronoi cell (once per cell)
            ConvexCell cell(seed_id, d_stored_points, &(gpu_stat[seed_id]));

            for (int v = 0; v < _K_; v++) {
                unsigned int z = local_knn[v];
                cell.clip_by_plane(z);
                if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * z))) { break; }
                if (gpu_stat[seed_id] != success) { break; }
            }
            if (!cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * local_knn[_K_ - 1]))) {
                gpu_stat[seed_id] = security_radius_not_reached;
            }

            if (gpu_stat[seed_id] == success) {
                hsize_t original_id = (hsize_t)sorted_to_original[seed_id];

                // compute vertices once, used by both extract passes
                double4 vertices[_MAX_T_];
                for (int vi = 0; vi < cell.nb_t; vi++)
                    vertices[vi] = cell.compute_vertex_point(cell.triangle[vi], true);

                // count faces and extract per-cell data (seeds, com, volumes)
                int fc                         = extract_cell_data(cell, vertices, mesh, original_id);
                mesh->face_counts[original_id] = (hsize_t)fc;

                // atomically reserve a contiguous block of face slots (lock-free)
                hsize_t my_offset;
#ifdef USE_OPENMP
#pragma omp atomic capture
#endif
                {
                    my_offset = face_offset;
                    face_offset += (hsize_t)fc;
                }

                // check that reserved slot fits in pre-allocated capacity
                if (my_offset + (hsize_t)fc > mesh->face_capacity) {
                    face_overflow_flag = 1;
                    continue;
                }

                mesh->face_ptr[original_id] = my_offset;

                // write face data at the reserved offset (non-overlapping, no contention)
                extract_cell_faces(cell, vertices, mesh, original_id);

                // convert neighbor IDs from sorted to original indexing
                hsize_t face_end = my_offset + (hsize_t)fc;
                for (hsize_t fi = my_offset; fi < face_end; fi++) {
                    int& nid = mesh->neighbor_cell[fi];
                    if (nid >= 0 && nid < N_seedpts) { nid = (int)sorted_to_original[nid]; }
                }
            }
        }

        mesh->num_faces = face_offset;

        if (face_overflow_flag) {
            std::cerr << "VORONOI: Error! face offset exceeds pre-allocated face capacity " << mesh->face_capacity
                      << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
#endif

} // namespace voronoi