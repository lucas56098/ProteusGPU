#pragma once

// Caps on the speed and the temperature of the gas near the centre.

#include "astro/gas_of_cell.h"
#include "astro/parameters.h"

#include <cmath>

namespace astro {

    // in code units
    struct Limiters {
        double radius_squared          = 0.0;
        double speed_cap_squared       = 0.0;
        double largest_specific_energy = 0.0; // the internal energy per mass of the temperature cap
    };

    inline Limiters limiters_of(const Parameters& parameters, const CodeUnits& units) {
        Limiters     limiters;
        const double radius              = from_kpc(parameters.limited_radius_in_kpc, units);
        const double speed_cap           = parameters.speed_cap_in_speed_of_light * units.speed_of_light;
        limiters.radius_squared          = radius * radius;
        limiters.speed_cap_squared       = speed_cap * speed_cap;
        limiters.largest_specific_energy = parameters.temperature_cap_in_k / units.kelvin_per_specific_energy;
        return limiters;
    }

    // inside the radius the speed first, then the energy of the temperature cap
    HD inline void limit(const Limiters& l, const Vec& position, CellGas& gas) {
        if (norm2(from_centre(position)) >= l.radius_squared) return;
        double v2 = norm2(gas.velocity);
        if (v2 > l.speed_cap_squared) {
            gas.velocity = sqrt(l.speed_cap_squared / v2) * gas.velocity;
            v2           = l.speed_cap_squared;
        }
        const double largest = 0.5 * gas.density * v2 + gas.density * l.largest_specific_energy;
        if (gas.energy > largest) gas.energy = largest;
    }

} // namespace astro
