#ifndef VORONOI_H
#define VORONOI_H

// The mesh of one rank: cells, faces and the index maps of a build.

#include "../global/allvars.h"
#include "../knn/knn.h"
#include <cstdint>

namespace hydro {
    struct primvars;
    struct ConsVars;
} // namespace hydro

// allocated once in allocate_mesh, lives until endrun
struct VMesh {

    uint64_t n_seeds; // points of the current build: cells + ghosts
    uint64_t n_hydro; // cells on this rank
    uint64_t num_faces;

    uint64_t face_capacity;
    uint64_t ghost_capacity;
    uint64_t total_capacity;

    // per cell
    double3*         seeds;
    double3*         com;
    double*          volumes;
    uint64_t*        face_counts;
    uint64_t*        face_ptr; // first face of the cell
    voronoi::Status* cell_status;
#ifdef MOVING_MESH
    POINT_TYPE* v_mesh;
#endif

    double* security_d2; // (2 x farthest vertex)^2, a seed outside of that cannot cut the cell
    double* est_r;       // first guess of how far the cell reaches
    double* req_r2;      // squared radius of the ball the cell asked for in this build, 0 if none

    // the balls of one request round: per cell the radius it asks for, then the cells that ask and their radii
    double* ball_r;
    int*    ball_cell;
    double* ball_rad;

    // the cells one step of the cell build works on, and the face counter and overflow flag of its kernels
    int*                cell_list;
    unsigned long long* face_offset;
    int*                overflow_flag;

#ifdef VOL_REGULARIZE
    double* volumes_g;
#endif

    // per MPI ghost slot
    double3*    seeds_g;
    POINT_TYPE* com_off_g; // centroid - seed
#ifdef MOVING_MESH
    POINT_TYPE* v_mesh_g;
#endif

    // per face, cell k owns [face_ptr[k], face_ptr[k] + face_counts[k])
    int*    neighbor_cell; // -1 on the box wall
    double* face_area;
    double* f_mid_local; // face centre - middle of the two seeds, tangential

    uint64_t* ghost_ids; // copy of an own cell -> that cell, ghost of another rank -> n_hydro + slot

    // index maps of the current build
    unsigned int* real_sorted_ids; // cell k -> sorted point
    unsigned int* sid_to_neighbor; // sorted point -> neighbour index
    unsigned int* gather_perm;     // cell k -> input point

    unsigned int* scan_flags;
    unsigned int* scan_scratch;

    // scratch
    double*     scratch_double;
    POINT_TYPE* scratch_point;

    POINT_TYPE* scratch_pts;  // point list of the build
    POINT_TYPE* scratch_move; // seeds after the move, in k-order during the build

    double min_egy_spec;

    double Ri_ref; // radius of a cell of the reference volume

    int n_mpi_ghosts;

    knn_problem* knn;
};

namespace voronoi {

    // allocate and grow
    VMesh* allocate_mesh(uint64_t n_hydro);
    void   free_mesh(VMesh* mesh);

    void mesh_grow_ghosts(VMesh* mesh, int new_cap);

    // room for this many ghosts behind the cells of a build
    void mesh_ensure_ghost_capacity(VMesh* mesh, uint64_t n_ghosts);

    // build the mesh for the current seed positions
    void compute_periodic_mesh(VMesh*                    mesh,
                               POINT_TYPE*               pts_data,
                               uint64_t                  num_points,
                               hydro::primvars*          primvar,
                               hydro::ConsVars*          cons,
                               gradients::PrimGradients* grads,
                               double                    dt);

    // move it
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads);

    void
    move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::ConsVars* cons, gradients::PrimGradients* grads);

} // namespace voronoi

HD inline hydro::prim get_state(uint64_t k, const hydro::primvars* primvar) {
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

// k < n_hydro: cell, above: MPI ghost
HD inline double3 get_seed_at(int k, int n_hydro, const VMesh* mesh) {
    return (k < n_hydro) ? mesh->seeds[k] : mesh->seeds_g[k - n_hydro];
}

// centroid - seed of a cell or an MPI ghost
HD inline POINT_TYPE get_com_off_at(int k, int n_hydro, const VMesh* mesh) {
    return (k < n_hydro) ? point_diff_periodic(mesh->com[k], mesh->seeds[k]) : mesh->com_off_g[k - n_hydro];
}

// face centroid seen from a seed: half_dx is +-0.5 (seed j - seed i), g the frame of the face
HD inline POINT_TYPE face_centroid_from_seed(const POINT_TYPE& half_dx, const geom& g, const double* f_mid_local) {
#ifdef dim_2D
    const double cx = f_mid_local[0] * g.m.x;
    const double cy = f_mid_local[0] * g.m.y;
    POINT_TYPE   r  = {half_dx.x + cx, half_dx.y + cy};
#else
    const double cx = f_mid_local[0] * g.m.x + f_mid_local[1] * g.p.x;
    const double cy = f_mid_local[0] * g.m.y + f_mid_local[1] * g.p.y;
    const double cz = f_mid_local[0] * g.m.z + f_mid_local[1] * g.p.z;
    POINT_TYPE   r  = {half_dx.x + cx, half_dx.y + cy, half_dx.z + cz};
#endif
    return r;
}

#ifdef MOVING_MESH
HD inline POINT_TYPE get_vmesh_at(int k, int n_hydro, const VMesh* mesh) {
    return (k < n_hydro) ? mesh->v_mesh[k] : mesh->v_mesh_g[k - n_hydro];
}

#ifdef VOL_REGULARIZE
HD inline double get_volume_at(int k, int n_hydro, const VMesh* mesh) {
    return (k < n_hydro) ? mesh->volumes[k] : mesh->volumes_g[k - n_hydro];
}
#endif
#endif

HD inline hydro::prim get_state_at(int k, int n_hydro, const hydro::primvars* primvar) {
    hydro::prim s;
    if (k < n_hydro) {
        s.rho = primvar->rho[k];
        s.v.x = primvar->v[k].x;
        s.v.y = primvar->v[k].y;
#ifdef dim_3D
        s.v.z = primvar->v[k].z;
#endif
        s.E = primvar->E[k];
    } else {
        const int g = k - n_hydro;
        s.rho       = primvar->rho_g[g];
        s.v.x       = primvar->v_g[g].x;
        s.v.y       = primvar->v_g[g].y;
#ifdef dim_3D
        s.v.z = primvar->v_g[g].z;
#endif
        s.E = primvar->E_g[g];
    }
    return s;
}

#endif
