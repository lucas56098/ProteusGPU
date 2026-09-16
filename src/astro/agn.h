#ifndef ASTRO_AGN_H
#define ASTRO_AGN_H
#pragma once

// Black hole feedback: cold gas that falls in comes back as heat, and as a jet.

#include "../global/gpu_compat.h"

#if defined(AGN_THERMAL) || defined(AGN_KINETIC)
#define AGN_ENABLED
#endif

namespace astro {

#ifdef AGN_ENABLED

    // set once in agn_init, in code units
    struct AgnParams {
        double cx, cy, cz;

        // what feeds the black hole: cold gas inside this radius
        double r_acc2;
        double T_cold_acc;
        double C_T;   // specific energy -> temperature
        double t_acc; // the cold gas drains over this time
        double eta;   // share of the infalling rest mass that comes out again
        double c2;

        // thermal part: heat a sphere of this radius
        double r_T2;
        double inv_V_T;
        double f_T;
        double T_max;
        double cs2_max; // sound speed at T_max, the timestep needs it

#ifdef AGN_KINETIC
        // kinetic part: two jets along y, above and below the centre
        double f_K;
        double r_jet2;
        double L_jet;
        double h_jet;
        double inv_Vjet;
        double v_jet;
        double v_cap;
#endif
    };

    // agn_prepare runs once per step, the others read what it found
    void             agn_init();
    void             agn_prepare();
    bool             agn_is_firing();
    const AgnParams& agn_params();
    void             agn_apply(double dt_half);

#endif

} // namespace astro

#endif
