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
    // face array entries in use this build. The CPU fallback reuses a rebuilt cell's slot
    // where it fits and retires the old slice in place, so this range can contain inert
    // holes (neighbor_cell = -1, zero area). Slice-based consumers never see them; a flat
    // scan over [0, num_faces) must skip negative neighbour ids.
    hsize_t num_faces;

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

    // per-cell squared security diameter (2R)^2, R = distance from seed to the farthest cell
    // vertex, stored by store_security_d2 (cell.cu) from the pair every successful build
    // already computes for data-extent certification. A seed can influence cell k only if it
    // lies within distance 2R of seed k, so
    //     |s - seeds[k]|^2 > security_d2[k]  =>  cell k provably unaffected by seed s.
    // Valid for cells with cell_status == success; failed cells carry the previous build's
    // value until the CPU fallback rebuilds them. nullptr on non-MPI builds: nothing reads it
    // there, and the array would cost HBM the single-rank runs do not have to spare.
    double* security_d2;

#ifdef VOL_REGULARIZE
    double* volumes_g; // ghost cell volumes, populated by halo_exchange_volumes
#endif

    // MPI ghost SoA storage (size proteus_mpi::n_mpi_capacity). seeds_g[slot] is the
    // ghost cell's seed position; v_mesh_g[slot] its mesh velocity. Populated by
    // halo_exchange_seeds and halo_exchange_v_mesh. nullptr on single-rank.
    double3* seeds_g;
#ifdef MOVING_MESH
    POINT_TYPE* v_mesh_g;
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
    unsigned int* orig_to_k_save; // [orig] -> k;      size n_hydro

    // typed scratch pools — one per type, reused across every permute_inplace<T> call
    unsigned int* scratch_uint;   // size n_hydro
    double*       scratch_double; // size n_hydro
    POINT_TYPE*   scratch_point;  // size n_hydro

    // mesh-build scratch
    POINT_TYPE* scratch_pts;  // ghost-augmented point buffer; size total_capacity
    POINT_TYPE* scratch_move; // post-move seed buffer;        size n_hydro

    int* d_real_counter; // build_index_maps atomic counter, size 1

    // minimum specific internal energy for the hydro positivity/temperature floor (0 disables).
    // Mirror of sim.min_egy_spec, copied here so the flux kernel can read it device-side.
    double min_egy_spec;

    // effective radius of the mean-volume cell (V_ref = 1/N_global, unit box); reference
    // for the size-equalizing mesh drift. Set once in allocate_mesh.
    double Ri_ref;

    // periodic-buffer width: bounding box covers [-buff, 1+buff]^d; reals stay in [0, 1]^d
    double buff;

    int n_mpi_ghosts;

    // Soundness guard: physical extent of valid neighbour data this rank can see — own brick
    // plus W buckets of MPI halo, clamped to the extended [-buff, 1+buff]^d domain. Every cell
    // that reaches its security radius is checked against this in both build tiers
    // (cell_certified_within_data): if 2R exceeds the seed's distance to the nearest face, a
    // seed the rank does not hold could still clip the cell, and it is flagged
    // security_radius_beyond_data so the widen-W loop grows the halo rather than silently
    // tessellating against an incomplete neighbour set.
    // data_hi[a] == data_lo[a] disables the check (single rank, where the periodic ghost band
    // already covers every direction; also the 2D z-axis).
    double data_lo[3];
    double data_hi[3];

    // KNN cache
    knn_problem* knn;
};

namespace voronoi {

    VMesh* allocate_mesh(hsize_t n_hydro);
    void   free_mesh(VMesh* mesh);

    // resize MPI ghost arrays (seeds_g, v_mesh_g) to new_cap. Contents discarded;
    // halo_exchange_seeds / _v_mesh repopulate before any reader.
    void mesh_grow_ghosts(VMesh* mesh, int new_cap);

    // resize mesh-build buffers (scratch_pts, ghost_ids, sid_to_neighbor) to fit a new
    // halo capacity. Existing periodic-ghost data in scratch_pts is preserved through the
    // realloc so an in-progress compute_periodic_mesh build survives the grow.
    void mesh_grow_build_buffers(VMesh* mesh, int new_mpi_capacity);

    // periodic ghost generation + mesh rebuild over the extended [-buff, 1+buff]^d domain.
    // `dt` is the timestep of the move that produced these seed positions; passed through to
    // the CPU fallback so a perturbed cell can offset v_mesh by delta/dt and keep face
    // velocities consistent with the perturbed geometry. Pass 0.0 for the initial build,
    // where no v_mesh correction applies.
    void compute_periodic_mesh(VMesh*           mesh,
                               POINT_TYPE*      pts_data,
                               hsize_t          num_points,
                               hydro::primvars* primvar,
                               hydro::primvars* primvar_aux,
                               double           dt);

    // mesh-point velocity (gas velocity + Lloyd regularization)
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads);

    // advance seeds by v_mesh*dt with periodic wrap, then rebuild the mesh
    void move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::primvars* primvar_aux);

} // namespace voronoi

// own-cell read: k strictly < n_hydro guaranteed by callers.
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

// ghost-aware seed read.
HD inline double3 get_seed_at(int k, int n_hydro, const VMesh* mesh) {
    return (k < n_hydro) ? mesh->seeds[k] : mesh->seeds_g[k - n_hydro];
}

#ifdef MOVING_MESH
HD inline POINT_TYPE get_vmesh_at(int k, int n_hydro, const VMesh* mesh) {
    return (k < n_hydro) ? mesh->v_mesh[k] : mesh->v_mesh_g[k - n_hydro];
}

#ifdef VOL_REGULARIZE
// ghost-aware cell-volume read; neighbours in a face loop may be MPI ghosts (>= n_hydro).
HD inline double get_volume_at(int k, int n_hydro, const VMesh* mesh) {
    return (k < n_hydro) ? mesh->volumes[k] : mesh->volumes_g[k - n_hydro];
}
#endif
#endif

// ghost-aware read: k may be a neighbor at >= n_hydro; falls back to primvar->*_g[].
// Used in flux + gradient face loops where neighbours may be MPI ghosts.
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

#endif // VORONOI_H
