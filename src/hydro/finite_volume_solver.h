#ifndef FINITE_VOLUME_SOLVER_H
#define FINITE_VOLUME_SOLVER_H

#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../mpi/halo.h"
#include "../voronoi/voronoi.h"
#include <cmath>

namespace hydro {

    // initialization and memory management — operate on global `sim`
    void init_hydro(); // allocates primvar (from IC) + per-step scratch (prim_new, grads)
    void free_hydro(); // frees everything init_hydro allocated

    // main routines
    void   hydro_step(double dt, VMesh* mesh, primvars* primvar);
    void   apply_flux_update(double                          dt_update,
                             double                          dt_extrap,
                             const VMesh*                    mesh,
                             const primvars*                 prim_old,
                             const gradients::PrimGradients* grads,
                             primvars*                       prim_new);
    double calc_timestep(double CFL, const VMesh* mesh, const primvars* primvar);

    // HD helpers
    HD void apply_spatial_extrapolation(const prim                    state,
                                        const gradients::PrimGradient gradient,
                                        POINT_TYPE                    dx,
                                        prim*                         st_extrap);
    HD void apply_time_extrapolation(prim state_i, gradients::PrimGradient grad_i, double dt_extrap, prim* st_extrap);
    HD void keep_state_physical(prim* state);
    HD void rotate_to_face(prim* state, geom* g);
    HD void rotate_from_face(prim* state, geom* g);
#ifdef MOVING_MESH
    HD void get_vel_face(hsize_t       i,
                         hsize_t       index_j,
                         POINT_TYPE    v_mesh_i,
                         POINT_TYPE    v_mesh_j,
                         const double* f_mid_local,
                         const VMesh*  mesh,
                         geom          g,
                         POINT_TYPE*   vel_face,
                         POINT_TYPE*   vel_face_turned);
    HD void convert_state_to_local_frame(prim* st, POINT_TYPE vel_face);
    HD void convert_flux_to_lab_frame(flux_t* flux, POINT_TYPE vel_face_turned);
#endif

    // allocate per-cell SoA + (optionally) MPI ghost SoA. with_ghosts=true sizes the
    // _g arrays from proteus_mpi::n_mpi_capacity; flip to false for buffers like prim_new
    // that are never read at neighbor indices and so never need ghost slots.
    inline void allocate_prim_buffer(hsize_t n_hydro, primvars* primvar, bool with_ghosts) {
        const hsize_t ext = (hsize_t)proteus_mpi::alloc_per_cell_size((int)n_hydro);
        primvar->rho      = gpu_alloc<double>(ext);
        primvar->v        = gpu_alloc<POINT_TYPE>(ext);
        primvar->E        = gpu_alloc<double>(ext);

        gpu_advise_gpu_preferred(primvar->rho, ext * sizeof(double));
        gpu_advise_gpu_preferred(primvar->v, ext * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(primvar->E, ext * sizeof(double));

        const int gc = with_ghosts ? proteus_mpi::n_mpi_capacity : 0;
        if (gc > 0) {
            primvar->rho_g = gpu_alloc<double>(gc);
            primvar->v_g   = gpu_alloc<POINT_TYPE>(gc);
            primvar->E_g   = gpu_alloc<double>(gc);
            gpu_advise_gpu_preferred(primvar->rho_g, gc * sizeof(double));
            gpu_advise_gpu_preferred(primvar->v_g, gc * sizeof(POINT_TYPE));
            gpu_advise_gpu_preferred(primvar->E_g, gc * sizeof(double));
        } else {
            primvar->rho_g = nullptr;
            primvar->v_g   = nullptr;
            primvar->E_g   = nullptr;
        }
    }

    inline void free_prim_buffer(primvars* primvar) {
        gpu_free(primvar->rho);
        gpu_free(primvar->v);
        gpu_free(primvar->E);
        if (primvar->rho_g) gpu_free(primvar->rho_g);
        if (primvar->v_g) gpu_free(primvar->v_g);
        if (primvar->E_g) gpu_free(primvar->E_g);
        primvar->rho_g = nullptr;
        primvar->v_g   = nullptr;
        primvar->E_g   = nullptr;
    }

    // resize the ghost arrays to new_cap (>= current). Ghost contents are not preserved —
    // halo_exchange_primvars repopulates them before any reader. Called by halo_grow_capacity
    // when the post-rebalance halo exceeds the current ghost-buffer size.
    inline void primvar_grow_ghosts(primvars* primvar, int new_cap) {
        if (primvar->rho_g) gpu_free(primvar->rho_g);
        if (primvar->v_g) gpu_free(primvar->v_g);
        if (primvar->E_g) gpu_free(primvar->E_g);
        primvar->rho_g = (new_cap > 0) ? gpu_alloc<double>(new_cap) : nullptr;
        primvar->v_g   = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
        primvar->E_g   = (new_cap > 0) ? gpu_alloc<double>(new_cap) : nullptr;
        if (new_cap > 0) {
            gpu_advise_gpu_preferred(primvar->rho_g, new_cap * sizeof(double));
            gpu_advise_gpu_preferred(primvar->v_g, new_cap * sizeof(POINT_TYPE));
            gpu_advise_gpu_preferred(primvar->E_g, new_cap * sizeof(double));
        }
    }

} // namespace hydro

#endif // FINITE_VOLUME_SOLVER