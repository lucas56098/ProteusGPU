#ifndef FINITE_VOLUME_SOLVER_H
#define FINITE_VOLUME_SOLVER_H

#include "../begrun/begrun.h"
#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "../voronoi/voronoi.h"
#include "riemann.h"
#include <climits>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <vector>

namespace hydro {

    // init hydrostruct from IC data
    primvars* init(int n_hydro);
    void      free_prim(primvars** primvar);

    // RK2 hydro stepping
    void hydro_step(double dt, const VMesh* mesh, primvars* primvar);
    void apply_flux_update(double               dt_update,
                           double               dt_extrap,
                           const VMesh*         mesh,
                           const primvars*      prim_old,
                           const PrimGradients* grads,
                           primvars*            prim_new);

    // spatial and time extrapolation of states
    void apply_spatial_extrapolation(const prim state, const PrimGradients gradient, POINT_TYPE dx, prim* st_extrap);
    void apply_time_extrapolation(prim state_i, PrimGradients grad_i, double dt_extrap, prim* st_extrap);

    // cfl criterion
    double dt_CFL(double CFL, const VMesh* mesh, const primvars* primvar);

    // helper
    void        keep_state_physical(prim* state);
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