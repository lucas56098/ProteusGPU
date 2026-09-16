#ifndef ASTRO_STARS_H
#define ASTRO_STARS_H
#pragma once

// Mass and energy the old stars of the central galaxy give back, and star formation feedback.

#include "../global/gpu_compat.h"

namespace astro {

#ifdef SF_FEEDBACK

    // set once in stars_init, in code units
    struct StarParams {
        double cx, cy, cz;

        // supernovae Ia: energy and mass per stellar mass and time
        double snia_Ce;
        double snia_Cm;
        // stellar density of the galaxy, a Hernquist profile
        double bcg_norm;
        double bcg_R;
        double bcg_rsoft;

        // star formation: share of the rest mass that comes back as energy
        double sf_eff;
        double sf_c2;
        double sf_G;
        double sf_rho_thresh; // dense enough to form stars
        double sf_T_max;      // and cold enough
        double sf_C_T;
        double sf_r_in2, sf_r_out2; // only in the shell between these radii
    };

    void stars_init();
    void stars_apply(double dt_half);

#endif

} // namespace astro

#endif
