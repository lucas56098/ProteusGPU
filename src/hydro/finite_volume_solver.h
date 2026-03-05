#ifndef FINITE_VOLUME_SOLVER_H
#define FINITE_VOLUME_SOLVER_H

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
#include "riemann.h"

namespace hydro {

    // init hydrostruct from IC data
    primvars* init(int n_hydro);
    void free_prim(primvars** primvar);

    void hydro_step(double dt, const VMesh* mesh, primvars* primvar);

    prim get_state_j(hsize_t i, int j, const VMesh* mesh, primvars* primvar);

    double dt_CFL(double CFL, const VMesh* mesh, const primvars* primvar);

}

#endif // FINITE_VOLUME_SOLVER