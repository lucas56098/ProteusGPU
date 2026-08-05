#ifndef ASTRO_AGN_H
#define ASTRO_AGN_H
#pragma once

#include "../global/gpu_compat.h"

// either channel pulls in the shared cold-accretion (CCA) trigger
#if defined(AGN_THERMAL) || defined(AGN_KINETIC)
#define AGN_ENABLED
#endif

namespace astro {

#ifdef AGN_ENABLED

    // precomputed AGN constants, all in code units (filled once by agn_init)
    struct AgnParams {
        double cx, cy, cz; // box center

        // cold-accretion trigger (CCA)
        double r_acc2;     // R_acc^2
        double T_cold_acc; // cold-gas threshold [K]
        double C_T;        // T[K] = C_T * e_int/rho
        double t_acc;      // accretion timescale (code time)
        double eta;        // accretion -> energy efficiency
        double c2;         // c^2 (code)

        // thermal channel
        double r_T2;    // R_T^2
        double inv_V_T; // 1 / deposition-region volume
        double f_T;     // thermal energy fraction
        double T_max;   // thermal temperature cap [K]
        double cs2_max; // sound-speed^2 ceiling from T_max, used by CFL when AGN fires

#ifdef AGN_KINETIC
        // kinetic channel: two launch zones on a fixed axis (y), offset +/-L_jet from center
        double f_K;      // kinetic energy fraction
        double r_jet2;   // jet cross-radius^2 (perpendicular to the axis)
        double L_jet;    // inner edge offset of each zone along the axis
        double h_jet;    // zone thickness along the axis
        double inv_Vjet; // 1 / volume of ONE launch zone
        double v_jet;    // jet velocity = sqrt(2*eta)*c (code)
        double v_cap;    // velocity cap (code), safety limiter
#endif
    };

    void   agn_init();
    void   agn_prepare();       // reduce cold mass once per step, cache for CFL + apply
    double agn_m_cold_cached(); // read cached m_cold (0 before first prepare)
    bool   agn_is_firing();     // == (m_cold_cached > 0)
    const AgnParams& agn_params();
    void   agn_apply(double dt_half);

#endif // AGN_ENABLED

} // namespace astro

#endif // ASTRO_AGN_H
