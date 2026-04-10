#ifndef VORONOI_H
#define VORONOI_H

#include "../global/allvars.h"
#include "../io/input.h"
#include "../knn/knn.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

/*
 * This part of the code is heavily inspired by the work of: Nicolas Ray, Dmitry Sokolov,
 * Sylvain Lefebvre, Bruno L'evy, "Meshless Voronoi on the GPU", ACM Trans. Graph.,
 * vol. 37, no. 6, Dec. 2018. If you build upon this code, we recommend
 * reading and citing their paper: https://doi.org/10.1145/3272127.3275092
 */

// voronoi mesh struct used for hydro solver
struct VMesh {

    // stored once
    hsize_t n_seeds;       // number of cells
    hsize_t n_hydro;       // number of hydro cells (n_ghost = n_seeds - n_hydro)
    hsize_t num_faces;     // total number of faces in the mesh
    hsize_t cell_capacity; // allocated capacity of per-cell arrays (>= n_seeds)
    hsize_t face_capacity; // allocated capacity of face arrays (>= num_faces)

    // stored for all cells
    double3* seeds;       // seedpoints
    double3* com;         // cell centroid
    double*  volumes;     // area in 2D, volume in 3D
    hsize_t* face_counts; // number of faces per cell
    hsize_t* face_ptr;    // pointer to start of each cell's faces in the face arrays

    // stored for all faces
    int*       neighbor_cell; // global id of neighboring cell for each face
    compact_t* face_area;     // edge length in 2D, face area in 3D
#ifdef MOVING_MESH
    compact_t* f_mid_local; // (DIMENSION-1) tangent-space offsets per face from seed midpoint
#endif
#ifdef DEBUG_MODE
    double*  edge_coords;          // flat array of all face vertex coordinates (DIMENSION doubles per vertex)
    hsize_t* edge_coords_offsets;  // number of vertices per face
    hsize_t  num_edge_coord_verts; // total number of edge coord vertices
#endif

    // stored for all ghost cells
    hsize_t* ghost_ids; // ids of the corresponding original cell (i.e. the ghost cell with id (n_hydro-1) + 4
                        // has ghost_ids[4])
};

namespace voronoi {

    // allocation and deallocation of VMesh
    VMesh* allocate_vmesh(hsize_t n_seeds, hsize_t initial_face_capacity);
    void   free_vmesh(VMesh* mesh);

    // main mesh computation (pass reuse to recycle an existing mesh's buffers)
    VMesh* compute_mesh(POINT_TYPE* pts_data, int num_points, VMesh* reuse = nullptr);
    void   compute_cells(int                  N_seedpts,
                         knn_problem*         knn,
                         std::vector<Status>& stat,
                         VMesh*               mesh,
                         const unsigned int*  sorted_to_original);

    // cpu fallback for cells that failed during knn-based construction
    void cpu_fallback_failed_cells(
        int N_seedpts, double* d_stored_points, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original);

// kernels
#ifdef CPU_DEBUG
    void cpu_compute_cell(int                 blocksPerGrid,
                          int                 threadsPerBlock,
                          int                 N_seedpts,
                          double*             d_stored_points,
                          const knn_problem*  knn,
                          Status*             gpu_stat,
                          VMesh*              mesh,
                          const unsigned int* sorted_to_original);
#endif

} // namespace voronoi

// helper
inline hsize_t hydro_index(hsize_t neighbor_raw, const VMesh* mesh) {
    if (neighbor_raw < mesh->n_hydro) { return neighbor_raw; }
    return mesh->ghost_ids[neighbor_raw - mesh->n_hydro];
}

// shared helper: extract primitive state for cell i (handles ghost mapping)
inline prim get_state(hsize_t i, const VMesh* mesh, const primvars* primvar) {
    prim    state_i;
    hsize_t index = hydro_index(i, mesh);

    state_i.rho = primvar->rho[index];
    state_i.v.x = primvar->v[index].x;
    state_i.v.y = primvar->v[index].y;
#ifdef dim_3D
    state_i.v.z = primvar->v[index].z;
#endif
    state_i.E = primvar->E[index];

    return state_i;
}

#endif // VORONOI_H
