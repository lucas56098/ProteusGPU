#ifndef ASTRO_LIMITERS_H
#define ASTRO_LIMITERS_H
#pragma once

// Safety net around the centre: caps the temperature and the speed of a cell.

#include "../global/gpu_compat.h"

namespace astro {

#ifdef LIMITERS

    // set once in limiters_init, in code units
    struct LimiterParams {
        double cx, cy, cz;
        double r_lim2; // only inside this radius
        double T_max;
        double C_T;
        double e_max_c; // T_max as specific energy
        double v_cap;
        double v_cap2;
    };

    void limiters_init();
    void limiters_apply();

#endif

} // namespace astro

#endif
