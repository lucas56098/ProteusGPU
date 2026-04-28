#ifndef RIEMANN_H
#define RIEMANN_H

#include "../global/allvars.h"

namespace hydro {

    HD flux_t riemann_hllc(prim state_i, prim state_j);

    HD double get_P_ideal_gas(const prim* state);
} // namespace hydro

#endif // RIEMANN_H