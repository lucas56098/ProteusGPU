#pragma once

// The physical constants in cgs, the gas, and the code units with what follows from them.

#include "config/config.h"

namespace astro {

    constexpr double SOLAR_MASS_IN_G            = 1.989e33;
    constexpr double KPC_IN_CM                  = 3.085677581e21;
    constexpr double MPC_IN_CM                  = 3.085677581e24;
    constexpr double KM_IN_CM                   = 1.0e5;
    constexpr double YEAR_IN_S                  = 3.15576e7;
    constexpr double BOLTZMANN_CONSTANT         = 1.380649e-16;
    constexpr double PROTON_MASS                = 1.67262192e-24;
    constexpr double SPEED_OF_LIGHT_IN_CM_PER_S = 2.99792458e10;
    constexpr double GRAVITATIONAL_CONSTANT     = 6.67430e-8;

    // the gas: fully ionised, of primordial composition
    constexpr double HYDROGEN_MASS_FRACTION = 0.76;
    constexpr double MEAN_MOLECULAR_WEIGHT  = 0.6; // in proton masses
    constexpr double MASS_PER_ELECTRON      = 2.0 / (1.0 + HYDROGEN_MASS_FRACTION); // in proton masses

    // cgs value of one code length, mass and velocity
    struct Units {
        double length_in_cm         = 1.0;
        double mass_in_g            = 1.0;
        double velocity_in_cm_per_s = 1.0;
    };

    struct CodeUnits {
        Units  base;
        double time_in_s;
        double density_in_g_per_cm3;
        double gravitational_constant; // in code units
        double speed_of_light; // in code units
        double kelvin_per_specific_energy;
    };

    inline CodeUnits code_units_of(const Units& units) {
        CodeUnits code;
        code.base                   = units;
        code.time_in_s              = units.length_in_cm / units.velocity_in_cm_per_s;
        code.density_in_g_per_cm3   = units.mass_in_g / (units.length_in_cm * units.length_in_cm * units.length_in_cm);
        code.gravitational_constant = GRAVITATIONAL_CONSTANT * units.mass_in_g * code.time_in_s * code.time_in_s /
                                      (units.length_in_cm * units.length_in_cm * units.length_in_cm);
        code.speed_of_light             = SPEED_OF_LIGHT_IN_CM_PER_S / units.velocity_in_cm_per_s;
        code.kelvin_per_specific_energy = (config::adiabatic_index - 1.0) * MEAN_MOLECULAR_WEIGHT * PROTON_MASS *
                                          units.velocity_in_cm_per_s * units.velocity_in_cm_per_s / BOLTZMANN_CONSTANT;
        return code;
    }

    inline double from_kpc(double kpc, const CodeUnits& code) {
        return kpc * KPC_IN_CM / code.base.length_in_cm;
    }
    inline double from_solar_masses(double solar_masses, const CodeUnits& code) {
        return solar_masses * SOLAR_MASS_IN_G / code.base.mass_in_g;
    }
    inline double from_years(double years, const CodeUnits& code) {
        return years * YEAR_IN_S / code.time_in_s;
    }

} // namespace astro
