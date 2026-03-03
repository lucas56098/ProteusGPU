#include "voronoi.h"
#include "../global/allvars.h"
#include "../knn/knn.h"
#include "../io/input.h"
#include "../io/output.h"
#include "cell.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

namespace voronoi {

// ----------------------------------------------
// -------- main voronoi mesh generation --------
// ----------------------------------------------
    VMesh* compute_mesh(POINT_TYPE* pts_data, ICData& icData, InputHandler& input, OutputHandler& output) {
        (void)input;   // unused unless VERIFY or WRITE_KNN_OUTPUT is enabled
        (void)output;  // unused unless VERIFY or WRITE_KNN_OUTPUT is enabled
        std::cout << "VORONOI: Computing Voronoi mesh..." << std::endl;

        // -------- KNN PROBLEM --------
        // define knn problem
        knn_problem *knn = NULL;

        // prepare knn problem
        int n_pts = icData.seedpos_dims[0];
        knn = knn::init((POINT_TYPE*) pts_data, n_pts);
        std::cout << "KNN: problem initialized." << std::endl;

        // solve knn problem
        knn::solve(knn);
        std::cout << "\nKNN: problem solved." << std::endl;

        // optional verify and output file
        #ifdef VERIFY
        if (!knn::verify(knn)) {exit(EXIT_FAILURE);}
        #endif
        #if defined(USE_HDF5) && defined(WRITE_KNN_OUTPUT)
        knn::write_knn_output(knn, icData, input, output);
        #endif

        // -------- VORONOI MESH GENERATION --------
        
        // allocate Vmesh struct
        std::vector<Status> stat(n_pts, security_radius_not_reached);
        hsize_t initial_face_capacity = (hsize_t)n_pts * 16;
        VMesh* mesh = allocate_vmesh((hsize_t)n_pts, initial_face_capacity);

        // compute voronoi cells from knn results
        compute_cells(n_pts, knn, stat, mesh);

        // reorder VMesh from sorted KNN order back to original input order
        unsigned int* knn_permutation = knn::get_permutation(knn);
        unpermute_vmesh(mesh, knn_permutation);
        free(knn_permutation);

        // free KNN resources
        knn::knn_free(&knn);

        return mesh;
    }

    // compute voronoi cells from knn results and store in VMesh
    void compute_cells(int N_seedpts, knn_problem* knn, std::vector<Status>& stat, VMesh* mesh) {
        // buffer to store (success, failure reason) for each cell
        GPUBuffer<Status> gpu_stat(stat);

        // initial capacities for face arrays
        hsize_t face_capacity = mesh->n_seeds * 16;
        #ifdef DEBUG_MODE
        edge_coords_capacity_global = mesh->n_seeds * 16 * 4; // initial estimate
        #endif

        // compute cell kernel
        int threadsPerBlock = _VORO_BLOCK_SIZE_;
        int blocksPerGrid = N_seedpts/threadsPerBlock + 1;
        
        std::cout << "VORONOI: computing cells" << std::endl;
        cpu_compute_cell(blocksPerGrid, threadsPerBlock, N_seedpts, (double*)knn->d_stored_points, knn->d_knearests, gpu_stat.gpu_data, mesh, face_capacity);
        std::cout << "\nVORONOI: cells computed" << std::endl;

        // shrink face arrays to actual size
        if (mesh->num_faces > 0) {
            mesh->neighbor_cell = (int*)realloc(mesh->neighbor_cell, mesh->num_faces * sizeof(int));
            mesh->face_normal = (double3*)realloc(mesh->face_normal, mesh->num_faces * sizeof(double3));
            mesh->face_area = (double*)realloc(mesh->face_area, mesh->num_faces * sizeof(double));
            #ifdef DEBUG_MODE
            mesh->edge_coords_offsets = (hsize_t*)realloc(mesh->edge_coords_offsets, mesh->num_faces * sizeof(hsize_t));
            mesh->edge_coords = (double*)realloc(mesh->edge_coords, mesh->num_edge_coord_verts * DIMENSION * sizeof(double));
            #endif
        }

        // check if any cells failed...
        gpu_stat.gpu2cpu();
        for (int i = 0; i < N_seedpts; i++) {
            if (gpu_stat.cpu_data[i] != success) {
                std::cout << "VORONOI: cell " << i << " failed with status: " << gpu_stat.cpu_data[i] << std::endl;
                // do cpu fallback here i guess
            }
        }
    }

#ifdef CPU_DEBUG
    // cpu debug version of cell computation kernel
    void cpu_compute_cell(int blocksPerGrid, int threadsPerBlock, int N_seedpts, double* d_stored_points, unsigned int* d_knearests, Status* gpu_stat, VMesh* mesh, hsize_t& face_capacity) {

        // emulate kernel
        for (blockId.x = 0; blockId.x < blocksPerGrid; blockId.x++) {
            for (threadId.x = 0; threadId.x < threadsPerBlock; threadId.x++) {

                // global seed_id
                int seed_id = threadsPerBlock * blockId.x + threadId.x;
                if (seed_id >= N_seedpts) {break;}
                if (seed_id % 1000000 == 0 || seed_id == N_seedpts - 1) {
                    std::cout << "\rVORONOI: processing cell " << seed_id+1 << " / " << N_seedpts << std::flush;
                }

                //create and initalize convex cell
                ConvexCell cell(seed_id, d_stored_points, &(gpu_stat[seed_id]));

                // clip cell by _K_ nearest neighbor planes
                for (int v = 0; v < _K_; v++) {

                    unsigned int z = d_knearests[_K_ * seed_id + v];
                    cell.clip_by_plane(z);

                    // security radius early exit
                    if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION*z))) {break;}

                    // gpu stat failure return...
                    if (gpu_stat[seed_id] != success) {break;}

                }
                // check if we are sure that the cell is correct
                if (!cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * d_knearests[_K_ * (seed_id+1) -1]))) {
                    gpu_stat[seed_id] = security_radius_not_reached;
                }

                // store cell in Vmesh if successful
                if (gpu_stat[seed_id] == success) {
                    extract_cell_to_vmesh(cell, mesh, (hsize_t)seed_id, face_capacity);
                }
            }
        }
    }
#endif

// ----------------------------------------------
// ------ mesh allocation and deallocation ------
// ----------------------------------------------
    VMesh* allocate_vmesh(hsize_t n_seeds, hsize_t initial_face_capacity) {
        VMesh* mesh = (VMesh*)malloc(sizeof(VMesh));
        mesh->n_seeds = n_seeds;
        mesh->num_faces = 0;

        // per-cell arrays (known size)
        mesh->cell_ids = (hsize_t*)calloc(n_seeds, sizeof(hsize_t));
        mesh->seeds = (double3*)calloc(n_seeds, sizeof(double3));
        mesh->volumes = (double*)calloc(n_seeds, sizeof(double));
        mesh->face_counts = (hsize_t*)calloc(n_seeds, sizeof(hsize_t));
        mesh->face_ptr = (hsize_t*)calloc(n_seeds, sizeof(hsize_t));

        // face arrays (initial capacity, grown dynamically during extraction)
        mesh->neighbor_cell = (int*)malloc(initial_face_capacity * sizeof(int));
        mesh->face_normal = (double3*)malloc(initial_face_capacity * sizeof(double3));
        mesh->face_area = (double*)malloc(initial_face_capacity * sizeof(double));

        #ifdef DEBUG_MODE
        mesh->edge_coords = (double*)malloc(initial_face_capacity * DIMENSION * 4 * sizeof(double)); // estimate ~4 verts per face
        mesh->edge_coords_offsets = (hsize_t*)malloc(initial_face_capacity * sizeof(hsize_t));
        mesh->num_edge_coord_verts = 0;
        #endif

        return mesh;
    }

    void free_vmesh(VMesh* mesh) {
        if (!mesh) return;
        free(mesh->cell_ids);
        free(mesh->seeds);
        free(mesh->volumes);
        free(mesh->face_counts);
        free(mesh->face_ptr);
        free(mesh->neighbor_cell);
        free(mesh->face_normal);
        free(mesh->face_area);
        #ifdef DEBUG_MODE
        free(mesh->edge_coords);
        free(mesh->edge_coords_offsets);
        #endif
        free(mesh);
    }

// ----------------------------------------------
// ------ restore original input pts order ------
// ----------------------------------------------
    void unpermute_vmesh(VMesh* mesh, const unsigned int* sorted_to_original) {
        hsize_t n = mesh->n_seeds;

        if (n == 0 || sorted_to_original == NULL) return;

        // Build inverse map: original id -> sorted id.
        std::vector<int> original_to_sorted(n, -1);
        for (hsize_t sorted = 0; sorted < n; sorted++) {
            unsigned int original = sorted_to_original[sorted];
            if (original < n) {
                original_to_sorted[original] = (int)sorted;
            }
        }

        // New cell-wise arrays in original input order.
        hsize_t* new_cell_ids = (hsize_t*)malloc(n * sizeof(hsize_t));
        double3* new_seeds = (double3*)malloc(n * sizeof(double3));
        double* new_volumes = (double*)malloc(n * sizeof(double));
        hsize_t* new_face_counts = (hsize_t*)malloc(n * sizeof(hsize_t));
        hsize_t* new_face_ptr = (hsize_t*)malloc(n * sizeof(hsize_t));

        // Face arrays are rebuilt as contiguous blocks per (now unpermuted) cell.
        int* new_neighbor_cell = (int*)malloc(mesh->num_faces * sizeof(int));
        double3* new_face_normal = (double3*)malloc(mesh->num_faces * sizeof(double3));
        double* new_face_area = (double*)malloc(mesh->num_faces * sizeof(double));

        #ifdef DEBUG_MODE
        hsize_t* new_edge_coords_offsets = (hsize_t*)malloc(mesh->num_faces * sizeof(hsize_t));
        double* new_edge_coords = (double*)malloc(mesh->num_edge_coord_verts * DIMENSION * sizeof(double));
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
            hsize_t count = mesh->face_counts[sorted_idx];
            hsize_t start = mesh->face_ptr[sorted_idx];

            // Per-cell scalars move to original slot; faces are appended at face_cursor.
            new_cell_ids[original] = original;
            new_seeds[original] = mesh->seeds[sorted_idx];
            new_volumes[original] = mesh->volumes[sorted_idx];
            new_face_counts[original] = count;
            new_face_ptr[original] = face_cursor;

            for (hsize_t f = 0; f < count; f++) {
                hsize_t old_fi = start + f;
                hsize_t new_fi = face_cursor + f;

                // Neighbor ids are still in sorted indexing; convert back to original ids.
                int sorted_neighbor = mesh->neighbor_cell[old_fi];
                if (sorted_neighbor >= 0 && (hsize_t)sorted_neighbor < n) {
                    new_neighbor_cell[new_fi] = (int)sorted_to_original[sorted_neighbor];
                } else {
                    new_neighbor_cell[new_fi] = sorted_neighbor;
                }

                new_face_normal[new_fi] = mesh->face_normal[old_fi];
                new_face_area[new_fi] = mesh->face_area[old_fi];

                #ifdef DEBUG_MODE
                hsize_t verts_in_face = mesh->edge_coords_offsets[old_fi];
                hsize_t old_edge_coord_cursor = old_edge_start[old_fi];
                new_edge_coords_offsets[new_fi] = verts_in_face;

                // Copy the variable-length face vertex block in flat storage.
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

        // Swap in rebuilt arrays.
        free(mesh->cell_ids);
        free(mesh->seeds);
        free(mesh->volumes);
        free(mesh->face_counts);
        free(mesh->face_ptr);
        free(mesh->neighbor_cell);
        free(mesh->face_normal);
        free(mesh->face_area);

        #ifdef DEBUG_MODE
        free(mesh->edge_coords_offsets);
        free(mesh->edge_coords);
        #endif

        mesh->cell_ids = new_cell_ids;
        mesh->seeds = new_seeds;
        mesh->volumes = new_volumes;
        mesh->face_counts = new_face_counts;
        mesh->face_ptr = new_face_ptr;
        mesh->neighbor_cell = new_neighbor_cell;
        mesh->face_normal = new_face_normal;
        mesh->face_area = new_face_area;

        #ifdef DEBUG_MODE
        mesh->edge_coords_offsets = new_edge_coords_offsets;
        mesh->edge_coords = new_edge_coords;
        mesh->num_edge_coord_verts = new_edge_coord_cursor;
        #endif
    }

// ----------------------------------------------
// -------- convert VMesh to MeshCellData -------
// ----------------------------------------------
#ifdef USE_HDF5
    void vmesh_to_meshdata(VMesh* mesh, MeshCellData& meshData) {
        int n_pts = (int)mesh->n_seeds;

        // header
        meshData.header.dimension = DIMENSION;
        meshData.header.extent = 1.0;
        meshData.header.n = n_pts;
        meshData.header.k = _K_;
        meshData.header.nmax = _MAX_P_;
        meshData.header.seed = 0;
        #ifdef DEBUG_MODE
        meshData.header.store_edge_coords = true;
        #else
        meshData.header.store_edge_coords = false;
        #endif

        meshData.seeds_dims = {(hsize_t)n_pts, DIMENSION};

        // cell ids
        meshData.cell_ids.resize(n_pts);
        for (int i = 0; i < n_pts; i++) {
            meshData.cell_ids[i] = (int)mesh->cell_ids[i];
        }

        // seeds (flatten double3 to flat double array)
        meshData.seeds.resize(n_pts * DIMENSION);
        for (int i = 0; i < n_pts; i++) {
            meshData.seeds[i * DIMENSION + 0] = mesh->seeds[i].x;
            meshData.seeds[i * DIMENSION + 1] = mesh->seeds[i].y;
            #ifdef dim_3D
            meshData.seeds[i * DIMENSION + 2] = mesh->seeds[i].z;
            #endif
        }

        // volumes
        meshData.volumes.resize(n_pts);
        for (int i = 0; i < n_pts; i++) {
            meshData.volumes[i] = mesh->volumes[i];
        }

        // face counts
        meshData.face_counts.resize(n_pts);
        for (int i = 0; i < n_pts; i++) {
            meshData.face_counts[i] = (int)mesh->face_counts[i];
        }

        // face data
        hsize_t nf = mesh->num_faces;
        meshData.faces.neighbor_cell.resize(nf);
        meshData.faces.normal.resize(nf * DIMENSION);
        meshData.faces.area.resize(nf);
        meshData.faces.normal_dims = {nf, DIMENSION};

        for (hsize_t f = 0; f < nf; f++) {
            meshData.faces.neighbor_cell[f] = mesh->neighbor_cell[f];
            meshData.faces.normal[f * DIMENSION + 0] = mesh->face_normal[f].x;
            meshData.faces.normal[f * DIMENSION + 1] = mesh->face_normal[f].y;
            #ifdef dim_3D
            meshData.faces.normal[f * DIMENSION + 2] = mesh->face_normal[f].z;
            #endif
            meshData.faces.area[f] = mesh->face_area[f];
        }

        #ifdef DEBUG_MODE
        // edge coords
        hsize_t total_verts = mesh->num_edge_coord_verts;
        meshData.faces.edge_coords.resize(total_verts * DIMENSION);
        for (hsize_t v = 0; v < total_verts * DIMENSION; v++) {
            meshData.faces.edge_coords[v] = mesh->edge_coords[v];
        }
        meshData.faces.edge_coords_dims = {total_verts, DIMENSION};

        meshData.faces.edge_coords_offsets.resize(nf);
        for (hsize_t f = 0; f < nf; f++) {
            meshData.faces.edge_coords_offsets[f] = (int)mesh->edge_coords_offsets[f];
        }
        #endif

        std::cout << "VORONOI: converted VMesh to MeshCellData (" << n_pts << " cells, " << nf << " faces)" << std::endl;
    }


#endif // USE_HDF5

} // namespace voronoi