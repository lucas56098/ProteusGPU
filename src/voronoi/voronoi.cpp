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

    // ----------------------------------------------
    // -------- main voronoi mesh generation --------
    // ----------------------------------------------
    VMesh* compute_mesh(POINT_TYPE* pts_data, int num_points, VMesh* reuse) {
#ifdef DEBUG_MODE
        std::cout << "VORONOI: Computing Voronoi mesh..." << std::endl;
#endif

        // -------- KNN PROBLEM --------
        PROFILE_START("KNN (par)");
        // define knn problem
        knn_problem* knn = NULL;

        // prepare knn problem
        int n_pts = num_points;
        knn       = knn::init((POINT_TYPE*)pts_data, n_pts);
#ifdef DEBUG_MODE
        std::cout << "KNN: problem initialized." << std::endl;
#endif

        PROFILE_END("KNN (par)");
        // -------- VORONOI MESH GENERATION --------
        PROFILE_START("VORONOI (par)");

        // allocate or reuse Vmesh struct
        std::vector<Status> stat(n_pts, security_radius_not_reached);
        hsize_t             initial_face_capacity = (hsize_t)n_pts * 16;
        VMesh*              mesh;

        if (reuse) {
            mesh = reuse;
            // grow per-cell arrays only if needed
            if ((hsize_t)n_pts > mesh->cell_capacity) {
                mesh->seeds         = (double3*)realloc(mesh->seeds, n_pts * sizeof(double3));
                mesh->com           = (double3*)realloc(mesh->com, n_pts * sizeof(double3));
                mesh->volumes       = (double*)realloc(mesh->volumes, n_pts * sizeof(double));
                mesh->face_counts   = (hsize_t*)realloc(mesh->face_counts, n_pts * sizeof(hsize_t));
                mesh->face_ptr      = (hsize_t*)realloc(mesh->face_ptr, n_pts * sizeof(hsize_t));
                mesh->cell_capacity = (hsize_t)n_pts;
            }
            // grow face arrays only if needed
            if (initial_face_capacity > mesh->face_capacity) {
                mesh->neighbor_cell = (int*)realloc(mesh->neighbor_cell, initial_face_capacity * sizeof(int));
                mesh->face_area     = (compact_t*)realloc(mesh->face_area, initial_face_capacity * sizeof(compact_t));
#ifdef MOVING_MESH
                mesh->f_mid_local =
                    (compact_t*)realloc(mesh->f_mid_local, initial_face_capacity * (DIMENSION - 1) * sizeof(compact_t));
#endif
#ifdef DEBUG_MODE
                mesh->edge_coords =
                    (double*)realloc(mesh->edge_coords, initial_face_capacity * DIMENSION * 4 * sizeof(double));
                mesh->edge_coords_offsets =
                    (hsize_t*)realloc(mesh->edge_coords_offsets, initial_face_capacity * sizeof(hsize_t));
#endif
                mesh->face_capacity = initial_face_capacity;
            }
            // reset for new computation
            mesh->n_seeds   = (hsize_t)n_pts;
            mesh->num_faces = 0;
            mesh->n_hydro   = 0;
#ifdef DEBUG_MODE
            mesh->num_edge_coord_verts = 0;
#endif
            memset(mesh->face_counts, 0, n_pts * sizeof(hsize_t));
            memset(mesh->face_ptr, 0, n_pts * sizeof(hsize_t));
        } else {
            mesh = allocate_vmesh((hsize_t)n_pts, initial_face_capacity);
        }

        // compute voronoi cells from knn results, writing directly to original positions
        compute_cells(n_pts, knn, stat, mesh, knn->d_permutation);

        PROFILE_END("VORONOI (par)");

        // free KNN resources
        knn::knn_free(&knn);
        return mesh;
    }

    // compute voronoi cells from knn results and store in VMesh
    void compute_cells(int                  N_seedpts,
                       knn_problem*         knn,
                       std::vector<Status>& stat,
                       VMesh*               mesh,
                       const unsigned int*  sorted_to_original) {

#ifdef DEBUG_MODE
        extern hsize_t edge_coords_capacity_global;
        edge_coords_capacity_global = mesh->n_seeds * 16 * 4; // initial estimate
#endif

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
                         stat.data(),
                         mesh,
                         sorted_to_original);
#ifdef DEBUG_MODE
        std::cout << "\nVORONOI: cells computed" << std::endl;
#endif

        // face arrays are kept at face_capacity (no shrinkage) to avoid
        // heap fragmentation from repeated realloc cycles in moving-mesh mode.

        // check if any cells failed and retry with cpu fallback
        cpu_fallback_failed_cells(N_seedpts, (double*)knn->d_stored_points, stat.data(), mesh, sorted_to_original);
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
                    hsize_t face_start  = mesh->num_faces;
                    extract_cell_to_vmesh(cell, mesh, original_id);
                    // convert neighbor IDs from sorted to original indexing
                    for (hsize_t fi = face_start; fi < mesh->num_faces; fi++) {
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
        hsize_t face_offset = 0;

        // ──── single pass (parallel): KNN + cell construction + atomic face-slot reservation ────
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, _VORO_BLOCK_SIZE_)
#endif
        for (int seed_id = 0; seed_id < N_seedpts; seed_id++) {
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

                // count faces and extract per-cell data (seeds, com, volumes)
                int fc                         = extract_cell_percell_count(cell, mesh, original_id);
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
                mesh->face_ptr[original_id] = my_offset;

                // write face data at the reserved offset (non-overlapping, no contention)
                extract_cell_percell_faces(cell, mesh, original_id);

                // convert neighbor IDs from sorted to original indexing
                hsize_t face_end = my_offset + (hsize_t)fc;
                for (hsize_t fi = my_offset; fi < face_end; fi++) {
                    int& nid = mesh->neighbor_cell[fi];
                    if (nid >= 0 && nid < N_seedpts) { nid = (int)sorted_to_original[nid]; }
                }
            }
        }

        mesh->num_faces = face_offset;
    }
#endif

    // ----------------------------------------------
    // ------ mesh allocation and deallocation ------
    // ----------------------------------------------
    VMesh* allocate_vmesh(hsize_t n_seeds, hsize_t initial_face_capacity) {
        VMesh* mesh         = (VMesh*)malloc(sizeof(VMesh));
        mesh->n_seeds       = n_seeds;
        mesh->num_faces     = 0;
        mesh->cell_capacity = n_seeds;
        mesh->face_capacity = initial_face_capacity;
        mesh->n_hydro       = 0;
        mesh->ghost_ids     = NULL;

        // per-cell arrays (known size)
        mesh->seeds       = (double3*)calloc(n_seeds, sizeof(double3));
        mesh->com         = (double3*)calloc(n_seeds, sizeof(double3));
        mesh->volumes     = (double*)calloc(n_seeds, sizeof(double));
        mesh->face_counts = (hsize_t*)calloc(n_seeds, sizeof(hsize_t));
        mesh->face_ptr    = (hsize_t*)calloc(n_seeds, sizeof(hsize_t));

        // face arrays (initial capacity, grown dynamically during extraction)
        mesh->neighbor_cell = (int*)malloc(initial_face_capacity * sizeof(int));
        mesh->face_area     = (compact_t*)malloc(initial_face_capacity * sizeof(compact_t));

#ifdef MOVING_MESH
        mesh->f_mid_local = (compact_t*)malloc(initial_face_capacity * (DIMENSION - 1) * sizeof(compact_t));
#endif

#ifdef DEBUG_MODE
        mesh->edge_coords =
            (double*)malloc(initial_face_capacity * DIMENSION * 4 * sizeof(double)); // estimate ~4 verts per face
        mesh->edge_coords_offsets  = (hsize_t*)malloc(initial_face_capacity * sizeof(hsize_t));
        mesh->num_edge_coord_verts = 0;
#endif

        // ghosts are manually allocated in compute periodic mesh

        return mesh;
    }

    void free_vmesh(VMesh* mesh) {
        if (!mesh) return;
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
#ifdef DEBUG_MODE
        free(mesh->edge_coords);
        free(mesh->edge_coords_offsets);
#endif
        free(mesh->ghost_ids);
        free(mesh);
    }

} // namespace voronoi