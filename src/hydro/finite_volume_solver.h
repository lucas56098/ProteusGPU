#ifndef FINITE_VOLUME_SOLVER_H
#define FINITE_VOLUME_SOLVER_H

#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../voronoi/periodic_mesh.h"
#include "../voronoi/voronoi.h"
#include <cmath>

namespace hydro {

    // init hydrostruct from IC data
    primvars* init(int n_hydro);
    void      free_prim(primvars** primvar);

    // allocate/free persistent hydro buffers (prim_new, gradients)
    void allocate_hydro_buffers(hsize_t n_hydro);
    void free_hydro_buffers();

    // RK2 hydro stepping
    void hydro_step(double dt, VMesh* mesh, primvars* primvar);
    void apply_flux_update(double                          dt_update,
                           double                          dt_extrap,
                           const VMesh*                    mesh,
                           const primvars*                 prim_old,
                           const gradients::PrimGradients* grads,
                           const POINT_TYPE*               v_mesh,
                           primvars*                       prim_new);

    // spatial and time extrapolation of states
    void apply_spatial_extrapolation(const prim                    state,
                                     const gradients::PrimGradient gradient,
                                     POINT_TYPE                    dx,
                                     prim*                         st_extrap);
    void apply_time_extrapolation(prim state_i, gradients::PrimGradient grad_i, double dt_extrap, prim* st_extrap);

    // cfl criterion
    double dt_CFL(double CFL, const VMesh* mesh, const primvars* primvar);

    // helper
#ifdef MOVING_MESH
    void get_vel_face(hsize_t          i,
                      hsize_t          index_j,
                      POINT_TYPE       v_mesh_i,
                      POINT_TYPE       v_mesh_j,
                      const compact_t* f_mid_local,
                      const VMesh*     mesh,
                      geom             g,
                      POINT_TYPE*      vel_face,
                      POINT_TYPE*      vel_face_turned);
    void convert_state_to_local_frame(prim* st, POINT_TYPE vel_face);
    void convert_flux_to_lab_frame(flux_t* flux, POINT_TYPE vel_face_turned);
#endif
    void rotate_to_face(prim* state, geom* g);
    void rotate_from_face(prim* state, geom* g);
    void keep_state_physical(prim* state);

    inline void allocate_prim_buffer(hsize_t n_hydro, primvars* primvar) {
        primvar->rho = (double*)malloc(n_hydro * sizeof(double));
        primvar->v   = (POINT_TYPE*)malloc(n_hydro * sizeof(POINT_TYPE));
        primvar->E   = (double*)malloc(n_hydro * sizeof(double));
    }

    inline void free_prim_buffer(primvars* primvar) {
        free(primvar->rho);
        free(primvar->v);
        free(primvar->E);
    }

} // namespace hydro

#endif // FINITE_VOLUME_SOLVER