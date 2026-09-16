#ifndef ASTRO_GRAVITY_H
#define ASTRO_GRAVITY_H
#pragma once

// A fixed outside potential: dark matter halo, central galaxy and black hole.

#include "../global/gpu_compat.h"

#if defined(NFW) || defined(HERNQUIST) || defined(SMBH)
#define GRAVITY_ENABLED
#endif

namespace astro {

#ifdef GRAVITY_ENABLED

    // set once in gravity_init, in code units
    struct GravityParams {
        double cx, cy, cz; // centre of the potential, the middle of the box
#ifdef NFW
        double nfw_A;
        double nfw_Rs;
#endif
#ifdef HERNQUIST
        double hq_GM;
        double hq_R;
#endif
#ifdef SMBH
        double bh_GM;
        double bh_eps2;
#endif
    };

    void gravity_init();
    void gravity_apply(double dt_half);

#endif

} // namespace astro

#endif
