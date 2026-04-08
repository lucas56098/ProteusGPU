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
    VMesh* compute_mesh(POINT_TYPE* pts_data, int num_points) {
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

        // solve knn problem
        knn::solve(knn);
#ifdef DEBUG_MODE
        std::cout << "\n";
#endif
#ifdef DEBUG_MODE
        std::cout << "KNN: problem solved." << std::endl;
#endif

// optional verify and output file
#ifdef VERIFY
        if (!knn::verify(knn)) { exit(EXIT_FAILURE); }
#endif
#if defined(USE_HDF5) && defined(WRITE_KNN_OUTPUT)
        knn::write_knn_output(knn);
#endif

        PROFILE_END("KNN (par)");
        // -------- VORONOI MESH GENERATION --------
        PROFILE_START("VORONOI (par)");

        // allocate Vmesh struct
        std::vector<Status> stat(n_pts, security_radius_not_reached);
        hsize_t             initial_face_capacity = (hsize_t)n_pts * 16;
        VMesh*              mesh                  = allocate_vmesh((hsize_t)n_pts, initial_face_capacity);

        // compute voronoi cells from knn results
        compute_cells(n_pts, knn, stat, mesh);

        // reorder VMesh from sorted KNN order back to original input order
        unpermute_vmesh(mesh, knn->d_permutation);

        PROFILE_END("VORONOI (par)");

        // free KNN resources
        knn::knn_free(&knn);
        return mesh;
    }

    // compute voronoi cells from knn results and store in VMesh
    void compute_cells(int N_seedpts, knn_problem* knn, std::vector<Status>& stat, VMesh* mesh) {

        // initial capacities for face arrays
        hsize_t face_capacity = mesh->n_seeds * 16;
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
                         knn->d_knearests,
                         stat.data(),
                         mesh,
                         face_capacity);
#ifdef DEBUG_MODE
        std::cout << "\nVORONOI: cells computed" << std::endl;
#endif

        // shrink face arrays to actual size
        if (mesh->num_faces > 0) {
            mesh->neighbor_cell = (int*)realloc(mesh->neighbor_cell, mesh->num_faces * sizeof(int));
            mesh->face_area     = (double*)realloc(mesh->face_area, mesh->num_faces * sizeof(double));
#ifdef DEBUG_MODE
            mesh->edge_coords_offsets = (hsize_t*)realloc(mesh->edge_coords_offsets, mesh->num_faces * sizeof(hsize_t));
            mesh->edge_coords =
                (double*)realloc(mesh->edge_coords, mesh->num_edge_coord_verts * DIMENSION * sizeof(double));
#endif
        }

        // check if any cells failed and retry with cpu fallback
        cpu_fallback_failed_cells(N_seedpts, (double*)knn->d_stored_points, stat.data(), mesh);
    }

    // cpu fallback for cells that failed during knn-based construction
    void cpu_fallback_failed_cells(int N_seedpts, double* d_stored_points, Status* stat, VMesh* mesh) {
        int num_failed = 0;
        for (int i = 0; i < N_seedpts; i++) {
            if (stat[i] != success) { num_failed++; }
        }
        if (num_failed == 0) return;

        std::cout << "VORONOI: " << num_failed << " cells failed, retrying with fallback..." << std::endl;

        hsize_t fallback_face_capacity = mesh->num_faces;

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

            if (original_status == security_radius_not_reached) {
                // security radius not reached: retry once with all seeds
                Status     fallback_status = success;
                ConvexCell cell(i, d_stored_points, &fallback_status);

                for (size_t di = 0; di < dists.size(); di++) {
                    int j = dists[di].second;
                    cell.clip_by_plane(j);
                    if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * j))) break;
                    if (fallback_status != success) break;
                }

                if (fallback_status == success) {
                    extract_cell_to_vmesh(cell, mesh, (hsize_t)i, fallback_face_capacity);
                    std::cout << "VORONOI: cell " << i << " fallback (all seeds) succeeded." << std::endl;
                    cell_ok = true;
                } else {
                    std::cerr << "VORONOI: cell " << i
                              << " fallback (all seeds) failed with status: " << fallback_status << std::endl;
                }

            } else if (original_status == needs_exact_predicates) {
                // needs exact predicates: retry with increasing perturbation to break degeneracy
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
                        extract_cell_to_vmesh(cell, mesh, (hsize_t)i, fallback_face_capacity);
                        std::cout << "VORONOI: cell " << i << " fallback (perturbed, attempt " << attempt
                                  << ") succeeded." << std::endl;
                        cell_ok = true;
                        break;
                    }
                }
            }

            if (!cell_ok) {
                std::cerr << "VORONOI: cell " << i << " all fallback attempts FAILED, aborting." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // re-shrink face arrays after fallback
        if (mesh->num_faces > 0) {
            mesh->neighbor_cell = (int*)realloc(mesh->neighbor_cell, mesh->num_faces * sizeof(int));
            mesh->face_area     = (double*)realloc(mesh->face_area, mesh->num_faces * sizeof(double));
#ifdef MOVING_MESH
            mesh->f_mid = (POINT_TYPE*)realloc(mesh->f_mid, mesh->num_faces * sizeof(POINT_TYPE));
#endif
#ifdef DEBUG_MODE
            mesh->edge_coords_offsets = (hsize_t*)realloc(mesh->edge_coords_offsets, mesh->num_faces * sizeof(hsize_t));
            mesh->edge_coords =
                (double*)realloc(mesh->edge_coords, mesh->num_edge_coord_verts * DIMENSION * sizeof(double));
#endif
        }
    }

#ifdef CPU_DEBUG
    // cpu debug version of cell computation kernel
    void cpu_compute_cell(int           blocksPerGrid,
                          int           threadsPerBlock,
                          int           N_seedpts,
                          double*       d_stored_points,
                          unsigned int* d_knearests,
                          Status*       gpu_stat,
                          VMesh*        mesh,
                          hsize_t&      face_capacity) {

        (void)blocksPerGrid;
        (void)threadsPerBlock;

        // per-cell extraction results for ALL cells
        std::vector<CellExtractionResult> results(N_seedpts);
        for (int i = 0; i < N_seedpts; i++) {
            results[i].valid = false;
        }

        // parallel region over all seed points
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

            // create and initalize convex cell
            ConvexCell cell(seed_id, d_stored_points, &(gpu_stat[seed_id]));

            // clip cell by _K_ nearest neighbor planes
            for (int v = 0; v < _K_; v++) {

                unsigned int z = d_knearests[_K_ * seed_id + v];
                cell.clip_by_plane(z);

                // security radius early exit
                if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * z))) { break; }

                // gpu stat failure return...
                if (gpu_stat[seed_id] != success) { break; }
            }
            // check if we are sure that the cell is correct
            if (!cell.is_security_radius_reached(
                    point_from_ptr(d_stored_points + DIMENSION * d_knearests[_K_ * (seed_id + 1) - 1]))) {
                gpu_stat[seed_id] = security_radius_not_reached;
            }

            // extract per-cell data and face info into thread-local buffer
            if (gpu_stat[seed_id] == success) { extract_cell_percell(cell, mesh, (hsize_t)seed_id, results[seed_id]); }
        }

        // serial merge — set face_ptr/face_counts and copy face data into VMesh
        for (int seed_id = 0; seed_id < N_seedpts; seed_id++) {
            if (!results[seed_id].valid) continue;

            mesh->face_ptr[seed_id]    = mesh->num_faces;
            mesh->face_counts[seed_id] = (hsize_t)results[seed_id].face_count;
            ensure_face_capacity(mesh, face_capacity, mesh->num_faces + results[seed_id].face_count);

            for (int f = 0; f < results[seed_id].face_count; f++) {
                hsize_t             fi  = mesh->num_faces;
                const CellFaceInfo& cfi = results[seed_id].faces[f];

                mesh->neighbor_cell[fi] = cfi.neighbor_id;
                mesh->face_area[fi]     = cfi.face_area;
#ifdef MOVING_MESH
                mesh->f_mid[fi] = cfi.f_mid;
#endif
#ifdef DEBUG_MODE
                const auto& fv                = cfi.face_verts;
                mesh->edge_coords_offsets[fi] = (hsize_t)fv.size();
                ensure_edge_coords_capacity(mesh, mesh->num_edge_coord_verts + fv.size());
                hsize_t ec = mesh->num_edge_coord_verts;
                for (size_t vi = 0; vi < fv.size(); vi++) {
                    mesh->edge_coords[(ec + vi) * DIMENSION + 0] = fv[vi].x;
                    mesh->edge_coords[(ec + vi) * DIMENSION + 1] = fv[vi].y;
#ifdef dim_3D
                    mesh->edge_coords[(ec + vi) * DIMENSION + 2] = fv[vi].z;
#endif
                }
                mesh->num_edge_coord_verts += fv.size();
#endif
                mesh->num_faces++;
            }
        }
    }
#endif

    // ----------------------------------------------
    // ------ mesh allocation and deallocation ------
    // ----------------------------------------------
    VMesh* allocate_vmesh(hsize_t n_seeds, hsize_t initial_face_capacity) {
        VMesh* mesh     = (VMesh*)malloc(sizeof(VMesh));
        mesh->n_seeds   = n_seeds;
        mesh->num_faces = 0;
        mesh->n_hydro   = 0;
        mesh->ghost_ids = NULL;

        // per-cell arrays (known size)
        mesh->seeds       = (double3*)calloc(n_seeds, sizeof(double3));
        mesh->com         = (double3*)calloc(n_seeds, sizeof(double3));
        mesh->volumes     = (double*)calloc(n_seeds, sizeof(double));
        mesh->face_counts = (hsize_t*)calloc(n_seeds, sizeof(hsize_t));
        mesh->face_ptr    = (hsize_t*)calloc(n_seeds, sizeof(hsize_t));

        // face arrays (initial capacity, grown dynamically during extraction)
        mesh->neighbor_cell = (int*)malloc(initial_face_capacity * sizeof(int));
        mesh->face_area     = (double*)malloc(initial_face_capacity * sizeof(double));

#ifdef MOVING_MESH
        mesh->f_mid = (POINT_TYPE*)malloc(initial_face_capacity * sizeof(POINT_TYPE));
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
        free(mesh->f_mid);
#endif
#ifdef DEBUG_MODE
        free(mesh->edge_coords);
        free(mesh->edge_coords_offsets);
#endif
        free(mesh->ghost_ids);
        free(mesh);
    }

    // ----------------------------------------------
    // ------ restore original input pts order ------
    // ----------------------------------------------
    void unpermute_vmesh(VMesh* mesh, const unsigned int* sorted_to_original) {
        hsize_t n = mesh->n_seeds;

        if (n == 0 || sorted_to_original == NULL) return;

        // build inverse map: original id -> sorted id.
        std::vector<int> original_to_sorted(n, -1);
        for (hsize_t sorted = 0; sorted < n; sorted++) {
            unsigned int original = sorted_to_original[sorted];
            if (original < n) { original_to_sorted[original] = (int)sorted; }
        }

        // new cell-wise arrays in original input order.
        double3* new_seeds       = (double3*)malloc(n * sizeof(double3));
        double3* new_com         = (double3*)malloc(n * sizeof(double3));
        double*  new_volumes     = (double*)malloc(n * sizeof(double));
        hsize_t* new_face_counts = (hsize_t*)malloc(n * sizeof(hsize_t));
        hsize_t* new_face_ptr    = (hsize_t*)malloc(n * sizeof(hsize_t));

        // face arrays are rebuilt as contiguous blocks per (now unpermuted) cell.
        int*    new_neighbor_cell = (int*)malloc(mesh->num_faces * sizeof(int));
        double* new_face_area     = (double*)malloc(mesh->num_faces * sizeof(double));
#ifdef MOVING_MESH
        POINT_TYPE* new_f_mid = (POINT_TYPE*)malloc(mesh->num_faces * sizeof(POINT_TYPE));
#endif

#ifdef DEBUG_MODE
        hsize_t* new_edge_coords_offsets = (hsize_t*)malloc(mesh->num_faces * sizeof(hsize_t));
        double*  new_edge_coords         = (double*)malloc(mesh->num_edge_coord_verts * DIMENSION * sizeof(double));
        // Prefix starts for the old flattened edge-coordinates buffer.
        std::vector<hsize_t> old_edge_start(mesh->num_faces + 1, 0);
        for (hsize_t fi = 0; fi < mesh->num_faces; fi++) {
            old_edge_start[fi + 1] = old_edge_start[fi] + mesh->edge_coords_offsets[fi];
        }
        hsize_t new_edge_coord_cursor = 0;
#endif

        hsize_t face_cursor = 0;

        for (hsize_t original = 0; original < n; original++) {
            int sorted = original_to_sorted[original];
            if (sorted < 0) continue;

            hsize_t sorted_idx = (hsize_t)sorted;
            hsize_t count      = mesh->face_counts[sorted_idx];
            hsize_t start      = mesh->face_ptr[sorted_idx];

            // per-cell scalars move to original slot; faces are appended at face_cursor.
            new_seeds[original]       = mesh->seeds[sorted_idx];
            new_com[original]         = mesh->com[sorted_idx];
            new_volumes[original]     = mesh->volumes[sorted_idx];
            new_face_counts[original] = count;
            new_face_ptr[original]    = face_cursor;

            for (hsize_t f = 0; f < count; f++) {
                hsize_t old_fi = start + f;
                hsize_t new_fi = face_cursor + f;

                // neighbor ids are still in sorted indexing; convert back to original ids.
                int sorted_neighbor = mesh->neighbor_cell[old_fi];
                if (sorted_neighbor >= 0 && (hsize_t)sorted_neighbor < n) {
                    new_neighbor_cell[new_fi] = (int)sorted_to_original[sorted_neighbor];
                } else {
                    new_neighbor_cell[new_fi] = sorted_neighbor;
                }

                new_face_area[new_fi] = mesh->face_area[old_fi];

#ifdef MOVING_MESH
                new_f_mid[new_fi] = mesh->f_mid[old_fi];
#endif

#ifdef DEBUG_MODE
                hsize_t verts_in_face           = mesh->edge_coords_offsets[old_fi];
                hsize_t old_edge_coord_cursor   = old_edge_start[old_fi];
                new_edge_coords_offsets[new_fi] = verts_in_face;

                // copy the variable-length face vertex block in flat storage.
                for (hsize_t vi = 0; vi < verts_in_face; vi++) {
                    new_edge_coords[(new_edge_coord_cursor + vi) * DIMENSION + 0] =
                        mesh->edge_coords[(old_edge_coord_cursor + vi) * DIMENSION + 0];
                    new_edge_coords[(new_edge_coord_cursor + vi) * DIMENSION + 1] =
                        mesh->edge_coords[(old_edge_coord_cursor + vi) * DIMENSION + 1];
#ifdef dim_3D
                    new_edge_coords[(new_edge_coord_cursor + vi) * DIMENSION + 2] =
                        mesh->edge_coords[(old_edge_coord_cursor + vi) * DIMENSION + 2];
#endif
                }
                new_edge_coord_cursor += verts_in_face;
#endif
            }

            face_cursor += count;
        }

        // swap in rebuilt arrays.
        free(mesh->seeds);
        free(mesh->com);
        free(mesh->volumes);
        free(mesh->face_counts);
        free(mesh->face_ptr);
        free(mesh->neighbor_cell);
        free(mesh->face_area);
#ifdef MOVING_MESH
        free(mesh->f_mid);
#endif

#ifdef DEBUG_MODE
        free(mesh->edge_coords_offsets);
        free(mesh->edge_coords);
#endif

        mesh->seeds         = new_seeds;
        mesh->com           = new_com;
        mesh->volumes       = new_volumes;
        mesh->face_counts   = new_face_counts;
        mesh->face_ptr      = new_face_ptr;
        mesh->neighbor_cell = new_neighbor_cell;
        mesh->face_area     = new_face_area;
#ifdef MOVING_MESH
        mesh->f_mid = new_f_mid;
#endif

#ifdef DEBUG_MODE
        mesh->edge_coords_offsets  = new_edge_coords_offsets;
        mesh->edge_coords          = new_edge_coords;
        mesh->num_edge_coord_verts = new_edge_coord_cursor;
#endif
    }

} // namespace voronoi