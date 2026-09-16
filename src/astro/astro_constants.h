#ifndef ASTRO_CONSTANTS_H
#define ASTRO_CONSTANTS_H
#pragma once

// Physical constants in cgs, and the gas composition.

namespace astro {

    constexpr double SOLAR_MASS_G = 1.989e33;
    constexpr double KPC_IN_CM    = 3.085677581e21;
    constexpr double MPC_IN_CM    = 3.085677581e24;
    constexpr double KM_S_IN_CGS  = 1.0e5;
    constexpr double YEAR_IN_S    = 3.15576e7;

    constexpr double BOLTZMANN      = 1.380649e-16;
    constexpr double PROTONMASS     = 1.67262192e-24;
    constexpr double SPEED_OF_LIGHT = 2.99792458e10;

    // fully ionised gas of primordial composition
    constexpr double HYDROGEN_MASSFRAC = 0.76;
    constexpr double MEAN_MOL_WEIGHT   = 0.6;                             // mass per particle, in proton masses
    constexpr double MEAN_MOL_WEIGHT_E = 2.0 / (1.0 + HYDROGEN_MASSFRAC); // mass per electron

} // namespace astro

#endif
