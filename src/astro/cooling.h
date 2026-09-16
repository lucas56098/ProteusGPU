#ifndef ASTRO_COOLING_H
#define ASTRO_COOLING_H
#pragma once

// Radiative cooling from a table of temperature and cooling rate.

#include "../global/gpu_compat.h"

#ifdef COOLING

namespace astro {

    // the table and the factors that go with it, filled in cooling_init
    struct CoolingTable {
        int     N;     // rows in the table
        double* T;     // temperature of each row
        double* L;     // cooling rate there
        double* alpha; // power law slope between two rows
        double* Y;     // cooling time function, see cooling.cu
        double  T_ref;
        double  L_ref;
        double  T_floor; // no cell cools below this
        double  C_T;     // specific energy -> temperature
        double  C_dY;    // cooling rate -> steps in Y
    };

    void cooling_init();
    void cooling_apply(double dt_half);

} // namespace astro

#endif
#endif
