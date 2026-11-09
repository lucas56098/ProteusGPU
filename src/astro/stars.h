#pragma once

// The stars of the central Hernquist galaxy: mass loss, supernovae Ia, and star formation feedback without particles.

#include "astro/gas_of_cell.h"
#include "astro/parameters.h"

#include <cmath>

namespace astro {

    // in code units
    struct Stars {
        double supernova_energy_per_mass_and_time = 0.0;
        double mass_loss_per_mass_and_time        = 0.0;
        double galaxy_mass_radius_over_2_pi       = 0.0;
        double galaxy_radius                      = 1.0;
        double softening_radius                   = 1.0; // the density is that of this radius further in
        double formation_efficiency               = 0.0;
        double speed_of_light_squared             = 0.0;
        double gravitational_constant             = 0.0;
        double formation_density                  = 0.0; // dense enough to form stars
        double formation_temperature              = 0.0; // and cold enough, in K
        double kelvin_per_specific_energy         = 0.0;
        double inner_radius_squared               = 0.0; // only in the shell between these radii
        double outer_radius_squared               = 0.0;
    };

    inline Stars stars_of(const Parameters& parameters, const CodeUnits& units) {
        Stars        stars;
        const double M = from_solar_masses(parameters.galaxy_mass_in_solar_masses, units);
        const double R = from_kpc(parameters.galaxy_radius_in_kpc, units);
        const double rate_per_s_and_g =
            parameters.supernova_rate_per_year_and_solar_mass / (YEAR_IN_S * SOLAR_MASS_IN_G);
        stars.supernova_energy_per_mass_and_time = rate_per_s_and_g * parameters.supernova_energy_in_erg *
                                                   units.time_in_s /
                                                   (units.base.velocity_in_cm_per_s * units.base.velocity_in_cm_per_s);
        stars.mass_loss_per_mass_and_time  = parameters.mass_loss_rate_per_s * units.time_in_s;
        stars.galaxy_mass_radius_over_2_pi = M * R / (2.0 * M_PI);
        stars.galaxy_radius                = R;
        stars.softening_radius             = 0.1 * R;
        stars.formation_efficiency         = parameters.star_formation_efficiency;
        stars.speed_of_light_squared       = units.speed_of_light * units.speed_of_light;
        stars.gravitational_constant       = units.gravitational_constant;
        stars.formation_density            = parameters.star_formation_density_in_per_cm3 * PROTON_MASS /
                                  (HYDROGEN_MASS_FRACTION * units.density_in_g_per_cm3);
        stars.formation_temperature      = parameters.star_formation_temperature_in_k;
        stars.kelvin_per_specific_energy = units.kelvin_per_specific_energy;
        const double inner               = from_kpc(parameters.accretion_radius_in_kpc, units);
        const double outer               = from_kpc(parameters.star_formation_radius_in_kpc, units);
        stars.inner_radius_squared       = inner * inner;
        stars.outer_radius_squared       = outer * outer;
        return stars;
    }

    // mass and energy of the stars here; then dense cold gas in the shell turns a share of its rest energy into heat
    HD inline void let_stars_act(const Stars& s, const Vec& position, double time, CellGas& gas) {
        const double r2 = norm2(from_centre(position));
        const double r  = sqrt(r2);

        const double rs              = fmax(r, s.softening_radius);
        const double rr              = rs + s.galaxy_radius;
        const double stellar_density = s.galaxy_mass_radius_over_2_pi / (rs * rr * rr * rr);
        const double dm              = s.mass_loss_per_mass_and_time * stellar_density * time;
        if (dm > 0.0) add_mass_at_rest_with_the_cell(dm, gas);
        gas.energy += s.supernova_energy_per_mass_and_time * stellar_density * time;

        if (r2 > s.inner_radius_squared && r2 < s.outer_radius_squared && gas.density > s.formation_density) {
            const double internal = internal_energy(gas);
            if (internal > 0.0 && s.kelvin_per_specific_energy * internal / gas.density < s.formation_temperature) {
                const double free_fall_time = sqrt(3.0 * M_PI / (32.0 * s.gravitational_constant * gas.density));
                gas.energy += s.formation_efficiency * gas.density * s.speed_of_light_squared * time / free_fall_time;
            }
        }
    }

} // namespace astro
