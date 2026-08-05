#ifndef ASTRO_STARS_H
#define ASTRO_STARS_H
#pragma once

#include "../global/gpu_compat.h"

namespace astro {

#ifdef SF_FEEDBACK

    // precomputed stellar-feedback constants, all in code units (filled once by stars_init)
    struct StarParams {
        double cx, cy, cz; // box center

        // SNIa: dE = snia_Ce * rho_BCG * dt,  drho = snia_Cm * rho_BCG * dt
        double snia_Ce;
        double snia_Cm;
        double bcg_norm;  // M_BCG*R_BCG/(2*pi); rho_BCG(r) = bcg_norm / (r*(r+bcg_R)^3)
        double bcg_R;     // Hernquist scale radius
        double bcg_rsoft; // floor on r in rho_BCG to tame the central cusp

        // particle-free SF heating (thermostat on cold dense gas)
        double sf_eff;             // efficiency (converted gas rest-mass -> heat)
        double sf_c2;              // c^2 in code units
        double sf_G;               // G in code units (for the free-fall time)
        double sf_rho_thresh;      // density above which SF can act (from n_SF)
        double sf_T_max;           // temperature below which SF can act [K]
        double sf_C_T;             // T[K] = sf_C_T * e_int/rho
        double sf_r_in2, sf_r_out2; // R_acc^2, R_SF^2 (code units, squared)
    };

    void stars_init();
    void stars_apply(double dt_half);

#endif // SF_FEEDBACK

} // namespace astro

#endif // ASTRO_STARS_H
