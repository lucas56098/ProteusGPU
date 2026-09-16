#ifndef RIEMANN_H
#define RIEMANN_H

// HLLC Riemann solver and the pressure of the ideal gas.

#include "../global/allvars.h"

namespace hydro {

    // both states are in the face frame, x along the normal
    HD flux_t riemann_hllc(prim state_i, prim state_j);

    HD double get_P_ideal_gas(const prim* state);
} // namespace hydro

#endif