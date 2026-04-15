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
    void time_gradient(hydro::prim state_i, PrimGradient grad_i, hydro::prim* dWdt);

    // limiter used for spatial gradients (returns limiting factor for one variable at one face)
    inline double limit_single_gradient(
        const double value, const double min_value, const double max_value, const POINT_TYPE& d, const GRAD_TYPE& grad);

} // namespace gradients

#endif // HYDRO_GRADIENTS_H
