#ifndef RIEMANN_H
#define RIEMANN_H

#include "../begrun/begrun.h"
#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "../voronoi/voronoi.h"
#include <climits>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <vector>

namespace hydro {

    prim riemann_hll(prim state_i, prim state_j);
    prim riemann_hllc(prim state_i, prim state_j);

    prim   get_flux(prim* state);
    double get_P_ideal_gas(prim* state);
} // namespace hydro

#endif // RIEMANN_H