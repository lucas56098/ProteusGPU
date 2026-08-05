#ifndef ASTRO_LIMITERS_H
#define ASTRO_LIMITERS_H
#pragma once

#include "../global/gpu_compat.h"

namespace astro {

#ifdef LIMITERS

    // hard clamps on T and |v| inside r < R_lim, matching the paper's central-region safety limits
    // (Fournier et al. Sect. 2.1). Purely per-cell -- no MPI reduction needed.
    struct LimiterParams {
        double cx, cy, cz; // box center
        double r_lim2;     // R_lim^2 (code units)
        double T_max;      // [K]
        double C_T;        // T[K] = C_T * e_int/rho
        double e_max_c;    // e_int/rho ceiling from T_max: e_int <= rho * e_max_c
        double v_cap;      // |v| ceiling (code units)
        double v_cap2;     // v_cap^2
    };

    void limiters_init();
    void limiters_apply();
    const LimiterParams& limiters_params();

#endif // LIMITERS

} // namespace astro

#endif // ASTRO_LIMITERS_H
