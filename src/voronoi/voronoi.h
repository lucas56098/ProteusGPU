#ifndef VORONOI_H
#define VORONOI_H

#include "../global/allvars.h"
#include "../io/input.h"
#include "../knn/knn.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

// Voronoi mesh data (allocated once at startup with worst-case capacity)
struct VMesh {

    // current counts
    hsize_t n_seeds;   // number of cells
    hsize_t n_hydro;   // number of hydro cells (n_ghost = n_seeds - n_hydro)
    hsize_t num_faces; // number of faces

    // capacities (fixed)
    hsize_t cell_capacity;  // max per-cell array size
    hsize_t face_capacity;  // max face array size
    hsize_t ghost_capacity; // max ghost_ids array size

    // per-cell
    double3* seeds;       // seed positions
    double3* com;         // cell centroids
    double*  volumes;     // cell volumes (area in 2D)
    hsize_t* face_counts; // number of faces per cell
    hsize_t* face_ptr;    // offset into face arrays for each cell

    // per-hydro-cell
#ifdef MOVING_MESH
    POINT_TYPE* v_mesh;      // mesh point velocities
    double*     old_volumes; // volumes before mesh movement
#endif

    // per-face
    int*       neighbor_cell; // neighbor cell index for each face
    compact_t* face_area;     // face area (edge length in 2D)
#ifdef MOVING_MESH
    compact_t* f_mid_local; // (DIMENSION-1) tangent-space offsets per face
#endif

    // ghost mapping
    hsize_t* ghost_ids; // ghost index -> original hydro index

    // per-cell status (reused each timestep)
    voronoi::Status* cell_status; // cell construction status flags

    // scratch buffers (allocated once, reused each timestep)
    POINT_TYPE*  scratch_pts;      // ghost-augmented point buffer
    hsize_t      scratch_pts_cap;  // capacity of scratch_pts
    POINT_TYPE*  scratch_move;     // moved seed positions buffer
    hsize_t      scratch_move_cap; // capacity of scratch_move
    knn_problem* knn;              // KNN data structure
};

namespace voronoi {

    // ---- lifecycle (called once) ----
    VMesh* allocate_mesh(hsize_t n_hydro);
    void   free_mesh(VMesh* mesh);

    // ---- mesh computation (called every timestep) ----
    void compute_periodic_mesh(VMesh* mesh, POINT_TYPE* pts_data, hsize_t num_points);
    void move_mesh(VMesh* mesh, double dt);
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads);

    // ---- internal (used by compute_periodic_mesh) ----
    void compute_mesh(VMesh* mesh, POINT_TYPE* pts_data, int num_points);
    void
    compute_cells(int N_seedpts, knn_problem* knn, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original);
    void cpu_fallback_failed_cells(
        int N_seedpts, double* d_stored_points, Status* stat, VMesh* mesh, const unsigned int* sorted_to_original);

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

// ---- shared inline helpers ----

inline hsize_t hydro_index(hsize_t neighbor_raw, const VMesh* mesh) {
    if (neighbor_raw < mesh->n_hydro) { return neighbor_raw; }
    return mesh->ghost_ids[neighbor_raw - mesh->n_hydro];
}

inline hydro::prim get_state(hsize_t i, const VMesh* mesh, const hydro::primvars* primvar) {
    hydro::prim state_i;
    hsize_t     index = hydro_index(i, mesh);

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
