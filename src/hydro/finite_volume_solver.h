#ifndef FINITE_VOLUME_SOLVER_H
#define FINITE_VOLUME_SOLVER_H

// The finite volume step: fluxes over the faces, and the CFL timestep.

#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../mpi/halo.h"
#include "../voronoi/voronoi.h"
#include <cmath>

namespace hydro {

    void init_hydro();
    void free_hydro();

    // one full step
    void hydro_step(double dt, VMesh* mesh, primvars* primvar);
    void apply_flux_update(double                          dt_update,
                           double                          dt_extrap,
                           const VMesh*                    mesh,
                           const primvars*                 prim_old,
                           const gradients::PrimGradients* grads,
                           primvars*                       prim_new);
    // smallest CFL step over all cells and ranks
    double calc_timestep(double CFL, const VMesh* mesh, const primvars* primvar);

    // per face, all called from flux_update_for_cell
    HD void apply_spatial_extrapolation(const prim                    state,
                                        const gradients::PrimGradient gradient,
                                        POINT_TYPE                    dx,
                                        prim*                         st_extrap);
    HD void apply_time_extrapolation(prim state_i, gradients::PrimGradient grad_i, double dt_extrap, prim* st_extrap);
    HD void keep_state_physical(prim* state, double min_egy_spec);
    HD void rotate_to_face(prim* state, geom* g);
    HD void rotate_from_face(prim* state, geom* g);
#ifdef MOVING_MESH
    HD void get_vel_face(uint64_t      i,
                         uint64_t      index_j,
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

    // one set of primvars: cells with growth headroom, ghosts only for the current state
    inline void allocate_prim_buffer(uint64_t n_hydro, primvars* primvar, bool with_ghosts) {
        const uint64_t ext = (uint64_t)proteus_mpi::alloc_per_cell_size((int)n_hydro);
        primvar->rho       = gpu_alloc<double>(ext);
        primvar->v         = gpu_alloc<POINT_TYPE>(ext);
        primvar->E         = gpu_alloc<double>(ext);

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

#endif