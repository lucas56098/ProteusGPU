#ifndef ASTRO_COOLING_H
#define ASTRO_COOLING_H
#pragma once

#include "../global/gpu_compat.h"

#ifdef COOLING

namespace astro {

    // Piecewise-power-law cooling table with the precomputed Townsend (2009) TEF. Node arrays are
    // physical cgs and live in managed memory; C_T and C_dY fold the unit system into the per-cell
    // update so the kernel stays in code units.
    struct CoolingTable {
        int     N;       // node count
        double* T;       // node temperatures [K]         (size N, ascending)
        double* L;       // Lambda_N at nodes [erg cm^3/s] (size N)
        double* alpha;   // power-law slope per segment    (size N-1)
        double* Y;       // TEF at nodes                   (size N)
        double  T_ref;   // reference (top) node
        double  L_ref;
        double  T_floor; // cooling stops at/below this [K]
        double  C_T;     // T[K]     = C_T  * (e_int_code / rho_code)
        double  C_dY;    // TEF step = C_dY * rho_code * dt_code
    };

    void cooling_init();
    void cooling_apply(double dt_half);

} // namespace astro

#endif // COOLING
#endif // ASTRO_COOLING_H
