#pragma once

// The static potential: an NFW halo, a Hernquist galaxy and a softened black hole, at the middle of the box.

#include "astro/gas_of_cell.h"
#include "astro/parameters.h"

#include <cmath>

namespace astro {

    // in code units; a part that is switched off pulls with zero
    struct Potential {
        double halo_strength     = 0.0; // G M / (ln(1 + c) - c / (1 + c))
        double halo_scale_radius = 1.0;
        double galaxy_gm         = 0.0;
        double galaxy_radius     = 0.0;
        double black_hole_gm     = 0.0;
        double softening_squared = 1.0;
    };

    // the NFW halo from its mass and concentration, its density scale of an overdensity of 200
    inline Potential potential_of(const Parameters& parameters, const CodeUnits& units) {
        Potential    potential;
        const double G = units.gravitational_constant;
        if constexpr (config::nfw) {
            const double M     = from_solar_masses(parameters.halo_mass_in_solar_masses, units);
            const double c     = parameters.halo_concentration;
            const double H0    = parameters.hubble_constant_in_km_per_s_mpc * KM_IN_CM / MPC_IN_CM * units.time_in_s;
            const double mc    = std::log(1.0 + c) - c / (1.0 + c);
            const double rho_s = 200.0 * c * c * c * H0 * H0 / (8.0 * M_PI * G * mc);
            potential.halo_scale_radius = std::cbrt(M / (4.0 * M_PI * rho_s * mc));
            potential.halo_strength     = G * M / mc;
        }
        if constexpr (config::hernquist) {
            potential.galaxy_gm     = G * from_solar_masses(parameters.galaxy_mass_in_solar_masses, units);
            potential.galaxy_radius = from_kpc(parameters.galaxy_radius_in_kpc, units);
        }
        if constexpr (config::smbh) {
            const double softening      = from_kpc(parameters.black_hole_softening_in_kpc, units);
            potential.black_hole_gm     = G * from_solar_masses(parameters.black_hole_mass_in_solar_masses, units);
            potential.softening_squared = softening * softening;
        }
        return potential;
    }

    HD inline double pull_at_radius(const Potential& p, double r) {
        double g = 0.0;
        if constexpr (config::nfw) {
            const double x = r / p.halo_scale_radius;
            g += p.halo_strength / (r * r) * (natural_log(1.0 + x) - x / (1.0 + x));
        }
        if constexpr (config::hernquist) {
            const double rr = r + p.galaxy_radius;
            g += p.galaxy_gm / (rr * rr);
        }
        if constexpr (config::smbh) {
            const double s = r * r + p.softening_squared;
            g += p.black_hole_gm * r / (s * sqrt(s));
        }
        return g;
    }

    // the velocity gets the acceleration over the time; the internal energy stays; no pull right at the centre
    HD inline void pull(const Potential& p, const Vec& position, double time, CellGas& gas) {
        const Vec    to_centre = -1.0 * from_centre(position);
        const double r         = sqrt(norm2(to_centre));
        if (r < 1e-9) return;
        const double kinetic_before = 0.5 * gas.density * norm2(gas.velocity);
        gas.velocity += (pull_at_radius(p, r) / r * time) * to_centre;
        gas.energy += 0.5 * gas.density * norm2(gas.velocity) - kinetic_before;
    }

} // namespace astro
