#ifndef RIEMANN_H
#define RIEMANN_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <climits>
#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "../begrun/begrun.h"
#include "../voronoi/voronoi.h"

struct geom {
    double3 n;  // normal
    double3 m;  // 1. tangential
    double3 p;  // 2. tangential
};

namespace hydro {

    prim riemann_hll(hsize_t i, hsize_t j, prim state_i, prim state_j, const VMesh* mesh);

    void rotate_to_face(prim* state, geom* g);
    void rotate_from_face(prim* state, geom* g);

    geom compute_geom(double3 normal);
    prim get_flux(prim* state);
    double get_P_ideal_gas(prim* state);
}

#endif // RIEMANN_H