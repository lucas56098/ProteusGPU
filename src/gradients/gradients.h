#ifndef HYDRO_GRADIENTS_H
#define HYDRO_GRADIENTS_H

#include "../global/allvars.h"
#include "../voronoi/voronoi.h"
#include <cmath>
#include <cstdlib>

namespace gradients {

    // compute and free gradients
    void compute_prim_gradients(const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads);

    // calc dW/dt ("time gradients") based on states and gradients
    HD inline void time_gradient(hydro::prim state_i, PrimGradient grad_i, hydro::prim* dWdt) {

        // precomputed helpers
        double v2   = point_dot(state_i.v, state_i.v);
        double divv = grad_i.vx.x + grad_i.vy.y;
        double kinx = state_i.v.x * grad_i.vx.x + state_i.v.y * grad_i.vy.x;
        double kiny = state_i.v.x * grad_i.vx.y + state_i.v.y * grad_i.vy.y;
#ifdef dim_3D
        divv += grad_i.vz.z;
        kinx += state_i.v.z * grad_i.vz.x;
        kiny += state_i.v.z * grad_i.vz.y;
        const double kinz = state_i.v.x * grad_i.vx.z + state_i.v.y * grad_i.vy.z + state_i.v.z * grad_i.vz.z;
#endif

        // pressure and its spatial derivatives
        const double P     = (gamma_eos - 1.0) * (state_i.E - 0.5 * state_i.rho * v2);
        const double dP_dx = (gamma_eos - 1.0) * (grad_i.E.x - 0.5 * (v2 * grad_i.rho.x + 2.0 * state_i.rho * kinx));
        const double dP_dy = (gamma_eos - 1.0) * (grad_i.E.y - 0.5 * (v2 * grad_i.rho.y + 2.0 * state_i.rho * kiny));
#ifdef dim_3D
        const double dP_dz = (gamma_eos - 1.0) * (grad_i.E.z - 0.5 * (v2 * grad_i.rho.z + 2.0 * state_i.rho * kinz));
#endif

        // compute drho/dt
        dWdt->rho = -(state_i.v.x * grad_i.rho.x + state_i.v.y * grad_i.rho.y + state_i.rho * divv);
#ifdef dim_3D
        dWdt->rho -= state_i.v.z * grad_i.rho.z;
#endif

        // compute dv/dt
        double inv_rho = 1.0 / state_i.rho;
        dWdt->v.x      = -(state_i.v.x * grad_i.vx.x + state_i.v.y * grad_i.vx.y) - dP_dx * inv_rho;
        dWdt->v.y      = -(state_i.v.x * grad_i.vy.x + state_i.v.y * grad_i.vy.y) - dP_dy * inv_rho;
#ifdef dim_3D
        dWdt->v.x -= state_i.v.z * grad_i.vx.z;
        dWdt->v.y -= state_i.v.z * grad_i.vy.z;
        dWdt->v.z =
            -(state_i.v.x * grad_i.vz.x + state_i.v.y * grad_i.vz.y + state_i.v.z * grad_i.vz.z) - dP_dz * inv_rho;
#endif

        // compute dE/dt
        dWdt->E = -(state_i.v.x * (grad_i.E.x + dP_dx) + state_i.v.y * (grad_i.E.y + dP_dy) + (state_i.E + P) * divv);
#ifdef dim_3D
        dWdt->E -= state_i.v.z * (grad_i.E.z + dP_dz);
#endif
    }

    // limiter used for spatial gradients (returns limiting factor for one variable at one face)
    HD inline double limit_single_gradient(
        const double value, const double min_value, const double max_value, const POINT_TYPE& d, const POINT_TYPE& grad);

} // namespace gradients

#endif // HYDRO_GRADIENTS_H
