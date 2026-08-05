#ifndef ASTRO_CONSTANTS_H
#define ASTRO_CONSTANTS_H
#pragma once

// cgs physical constants + gas composition for the astro modules. Grows as modules need.

namespace astro {

    // unit conversions
    constexpr double SOLAR_MASS_G = 1.989e33;       // g
    constexpr double KPC_IN_CM    = 3.085677581e21; // cm
    constexpr double MPC_IN_CM    = 3.085677581e24; // cm
    constexpr double KM_S_IN_CGS  = 1.0e5;          // cm/s per km/s
    constexpr double YEAR_IN_S    = 3.15576e7;      // s per Julian year

    // physical constants
    constexpr double BOLTZMANN     = 1.380649e-16;  // erg/K
    constexpr double PROTONMASS    = 1.67262192e-24; // g
    constexpr double SPEED_OF_LIGHT = 2.99792458e10; // cm/s

    // gas composition (fully ionized; hot-ICM values).
    constexpr double HYDROGEN_MASSFRAC = 0.76;                          // X_H
    constexpr double MEAN_MOL_WEIGHT   = 0.6;                           // mu
    constexpr double MEAN_MOL_WEIGHT_E = 2.0 / (1.0 + HYDROGEN_MASSFRAC); // mu_e (electrons)

} // namespace astro

#endif // ASTRO_CONSTANTS_H
