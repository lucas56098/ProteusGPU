#ifndef VORONOI_H
#define VORONOI_H

#include "../global/allvars.h"
#include "../io/input.h"
#include "../knn/knn.h"

namespace hydro {
    struct primvars;
}

// Voronoi mesh data — allocated once at startup with worst-case capacity, reused every step.
//
// Index spaces a cell can be addressed in:
//   orig: [0, n_seeds): position in the unsorted KNN input (real if orig < n_hydro, ghost if >=)
//   sid:  [0, n_seeds): KNN's spatial-sort index;  d_permutation[sid] -> orig
//   k:    [0, n_hydro): real-sorted-id; the canonical address for all per-cell data below
struct VMesh {

    // current counts
    hsize_t n_seeds;   // n_total = reals + ghosts fed to KNN this build
    hsize_t n_hydro;   // number of real cells; size of every per-cell array below
    hsize_t num_faces; // number of faces this build

    // fixed capacities (set in allocate_mesh)
    hsize_t face_capacity;
    hsize_t ghost_capacity;
    hsize_t total_capacity; // max n_seeds = n_hydro + max_ghosts

    // per-cell arrays (size n_hydro, indexed by k)
    double3*         seeds;
    double3*         com;
    double*          volumes;
    hsize_t*         face_counts;
    hsize_t*         face_ptr;
    voronoi::Status* cell_status;
#ifdef MOVING_MESH
    POINT_TYPE* v_mesh;
    double*     old_volumes;
#endif

    // per-face arrays (size face_capacity)
    int*    neighbor_cell; // [face_idx] -> neighbor k in [0, n_hydro), or -1 for box-boundary
    double* face_area;
#ifdef MOVING_MESH
    double* f_mid_local;
#endif

    // ghost slot -> source-real previous-step k  (size ghost_capacity)
    hsize_t* ghost_ids;

    // index maps rebuilt every step
    unsigned int* real_sorted_ids;  // [k] -> sid;       size n_hydro
    unsigned int* sid_to_neighbor;  // [sid] -> k;       size total_capacity
    unsigned int* cell_to_original; // [k] -> file id;   size n_hydro
    unsigned int* gather_perm;      // [new_k] -> old_k; size n_hydro

    // stable [orig] -> k mapping captured after iter 0 of the halo-widening loop,
    // reused by later iterations so primvar order stays aligned across them
    unsigned int* orig_to_k_save;   // [orig] -> k;      size n_hydro

    // typed scratch pools — one per type, reused across every permute_inplace<T> call
    unsigned int* scratch_uint;     // size n_hydro
    double*       scratch_double;   // size n_hydro
    POINT_TYPE*   scratch_point;    // size n_hydro

    // mesh-build scratch
    POINT_TYPE* scratch_pts;   // ghost-augmented point buffer; size total_capacity
    POINT_TYPE* scratch_move;  // post-move seed buffer;        size n_hydro

    int* d_real_counter; // build_index_maps atomic counter, size 1

    // periodic-buffer width: bounding box covers [-buff, 1+buff]^d; reals stay in [0, 1]^d
    double buff;

    // set during the cell build if any of cell k's K-nearest is an outermost-layer MPI ghost
    unsigned char* cell_hit_outer;
    int            pts_mpi_base;

    // KNN cache
    knn_problem* knn;
};

namespace voronoi {

    VMesh* allocate_mesh(hsize_t n_hydro);
    void   free_mesh(VMesh* mesh);

    // periodic ghost generation + mesh rebuild over the extended [-buff, 1+buff]^d domain.
    void compute_periodic_mesh(VMesh*           mesh,
                               POINT_TYPE*      pts_data,
                               hsize_t          num_points,
                               hydro::primvars* primvar,
                               hydro::primvars* primvar_aux);

    // mesh-point velocity (gas velocity + Lloyd regularization)
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads);

    // advance seeds by v_mesh*dt with periodic wrap, then rebuild the mesh
    void move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::primvars* primvar_aux);

} // namespace voronoi

// neighbor_cell stores real-sorted indices in [0, n_hydro); read primvars at k directly.
HD inline hydro::prim get_state(hsize_t k, const hydro::primvars* primvar) {
    hydro::prim s;
    s.rho = primvar->rho[k];
    s.v.x = primvar->v[k].x;
    s.v.y = primvar->v[k].y;
#ifdef dim_3D
    s.v.z = primvar->v[k].z;
#endif
    s.E = primvar->E[k];
    return s;
}

#endif // VORONOI_H
