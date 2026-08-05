#ifndef ASTRO_GRAVITY_H
#define ASTRO_GRAVITY_H
#pragma once

#include "../global/gpu_compat.h"

#if defined(NFW) || defined(HERNQUIST) || defined(SMBH)
#define GRAVITY_ENABLED
#endif

namespace astro {

#ifdef GRAVITY_ENABLED

    // precomputed potential constants, all in code units (filled once by gravity_init)
    struct GravityParams {
        double cx, cy, cz; // potential center = box center
#ifdef NFW
        double nfw_A;  // G*M_NFW / (ln(1+c) - c/(1+c))
        double nfw_Rs; // scale radius
#endif
#ifdef HERNQUIST
        double hq_GM; // G*M_BCG
        double hq_R;  // Hernquist scale radius
#endif
#ifdef SMBH
        double bh_GM;   // G*M_BH
        double bh_eps2; // Plummer softening, squared
#endif
    };

    void gravity_init();
    void gravity_apply(double dt_half); // kick every local cell by g*dt_half

#endif // GRAVITY_ENABLED

} // namespace astro

#endif // ASTRO_GRAVITY_H
