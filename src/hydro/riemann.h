#ifndef RIEMANN_H
#define RIEMANN_H

#include "../global/allvars.h"

namespace hydro {

    flux_t riemann_hll(prim state_i, prim state_j);
    flux_t riemann_hllc(prim state_i, prim state_j);

    flux_t get_flux(const prim* state);
    double get_P_ideal_gas(const prim* state);
} // namespace hydro

#endif // RIEMANN_H