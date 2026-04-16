#ifndef FINITE_VOLUME_SOLVER_H
#define FINITE_VOLUME_SOLVER_H

#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../voronoi/periodic_mesh.h"
#include "../voronoi/voronoi.h"
#include <cmath>

namespace hydro {

    // initialization and memory management
    primvars* init(int n_hydro);
    void      free_prim(primvars** primvar);
    void      allocate_hydro_buffers(hsize_t n_hydro);
    void      free_hydro_buffers();

    // main routines
    void   hydro_step(double dt, VMesh* mesh, primvars* primvar);
    void   apply_flux_update(double                          dt_update,
                             double                          dt_extrap,
                             const VMesh*                    mesh,
                             const primvars*                 prim_old,
                             const gradients::PrimGradients* grads,
                             primvars*                       prim_new);
    double dt_CFL(double CFL, const VMesh* mesh, const primvars* primvar);

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
    HD void get_vel_face(hsize_t          i,
                         hsize_t          index_j,
                         POINT_TYPE       v_mesh_i,
                         POINT_TYPE       v_mesh_j,
                         const compact_t* f_mid_local,
                         const VMesh*     mesh,
                         geom             g,
                         POINT_TYPE*      vel_face,
                         POINT_TYPE*      vel_face_turned);
    HD void convert_state_to_local_frame(prim* st, POINT_TYPE vel_face);
    HD void convert_flux_to_lab_frame(flux_t* flux, POINT_TYPE vel_face_turned);
#endif

    inline void allocate_prim_buffer(hsize_t n_hydro, primvars* primvar) {
        primvar->rho = gpu_alloc<double>(n_hydro);
        primvar->v   = gpu_alloc<POINT_TYPE>(n_hydro);
        primvar->E   = gpu_alloc<double>(n_hydro);

        gpu_advise_gpu_preferred(primvar->rho, n_hydro * sizeof(double));
        gpu_advise_gpu_preferred(primvar->v, n_hydro * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(primvar->E, n_hydro * sizeof(double));
    }

    inline void free_prim_buffer(primvars* primvar) {
        gpu_free(primvar->rho);
        gpu_free(primvar->v);
        gpu_free(primvar->E);
    }

} // namespace hydro

#endif // FINITE_VOLUME_SOLVER