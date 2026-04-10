#ifndef HYDRO_GRADIENTS_H
#define HYDRO_GRADIENTS_H

#include "../global/allvars.h"
#include "../voronoi/voronoi.h"
#include <cmath>
#include <cstdlib>

namespace gradients {

    // compute and free gradients
    void compute_prim_gradients(const VMesh* mesh, const primvars* primvar, PrimGradients* grads);

    // calc dW/dt ("time gradients") based on states and gradients
    void time_gradient(prim state_i, PrimGradients grad_i, prim* dWdt);

    // limiter used for spatial gradients
    inline void limit_single_gradient(
        const double value, const double min_value, const double max_value, const POINT_TYPE& d, GRAD_TYPE* grad);

    // helper
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

} // namespace gradients

#endif // HYDRO_GRADIENTS_H
