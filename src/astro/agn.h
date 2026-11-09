#pragma once

// The black hole at the middle of the box: accretion of the cold gas near it, a heated sphere and two jets along y.

#include "astro/gas_of_cell.h"
#include "astro/parameters.h"

#include <cmath>

namespace astro {

    // no gas the black hole heats gets hotter than this, in K
    constexpr double HOTTEST_HEATED_GAS = 5.0e9;

    // in code units
    struct Agn {
        double accretion_radius_squared   = 0.0;
        double cold_temperature           = 0.0; // in K
        double kelvin_per_specific_energy = 1.0;
        double accretion_time             = 1.0;
        double efficiency                 = 0.0;
        double speed_of_light_squared     = 0.0;

        // the heated sphere, a disc in 2D
        double heated_radius_squared  = 0.0;
        double one_over_heated_volume = 0.0;
        double thermal_share          = 0.0;
        double sound_speed_of_hottest = 0.0;

        // the jets: cylinders along y (strips in 2D), from jet_distance to jet_distance + jet_height
        double kinetic_share       = 0.0;
        double jet_radius_squared  = 0.0;
        double jet_distance        = 0.0;
        double jet_height          = 0.0;
        double one_over_jet_volume = 0.0;
        double jet_speed           = 0.0;
        double jet_speed_cap       = 0.0;
    };

    inline Agn agn_of(const Parameters& parameters, const CodeUnits& units) {
        Agn          agn;
        const double c                 = units.speed_of_light;
        const double accretion_radius  = from_kpc(parameters.accretion_radius_in_kpc, units);
        agn.accretion_radius_squared   = accretion_radius * accretion_radius;
        agn.cold_temperature           = parameters.cold_temperature_in_k;
        agn.kelvin_per_specific_energy = units.kelvin_per_specific_energy;
        agn.accretion_time             = from_years(1.0e6 * parameters.accretion_time_in_myr, units);
        agn.efficiency                 = parameters.feedback_efficiency;
        agn.speed_of_light_squared     = c * c;

        const double R             = from_kpc(parameters.heated_radius_in_kpc, units);
        agn.heated_radius_squared  = R * R;
        agn.one_over_heated_volume = (config::dim == 2) ? 1.0 / (M_PI * R * R) : 1.0 / (4.0 / 3.0 * M_PI * R * R * R);
        agn.thermal_share          = parameters.thermal_share;
        agn.sound_speed_of_hottest = std::sqrt(config::adiabatic_index * (config::adiabatic_index - 1.0) *
                                               HOTTEST_HEATED_GAS / units.kelvin_per_specific_energy);

        if constexpr (config::agn_kinetic) {
            const double r         = from_kpc(parameters.jet_radius_in_kpc, units);
            agn.kinetic_share      = parameters.kinetic_share;
            agn.jet_radius_squared = r * r;
            agn.jet_distance       = from_kpc(parameters.jet_distance_in_kpc, units);
            agn.jet_height         = from_kpc(parameters.jet_height_in_kpc, units);
            agn.one_over_jet_volume =
                (config::dim == 2) ? 1.0 / (2.0 * r * agn.jet_height) : 1.0 / (M_PI * r * r * agn.jet_height);
            agn.jet_speed     = std::sqrt(2.0 * agn.efficiency) * c;
            agn.jet_speed_cap = parameters.jet_speed_cap_in_speed_of_light * c;
        }
        return agn;
    }

    HD inline bool feeds_the_black_hole(const Agn& a, const Vec& offset, const CellGas& gas) {
        if (norm2(offset) >= a.accretion_radius_squared) return false;
        const double internal = internal_energy(gas);
        return internal > 0.0 && a.kelvin_per_specific_energy * internal / gas.density < a.cold_temperature;
    }

    HD inline bool in_the_heated_sphere(const Agn& a, const Vec& offset) {
        return norm2(offset) < a.heated_radius_squared;
    }

    HD inline bool in_a_jet(const Agn& a, const Vec& offset) {
        double across = offset[0] * offset[0];
        if constexpr (config::dim == 3) across += offset[2] * offset[2];
        const double along = fabs(offset[1]);
        return across < a.jet_radius_squared && along > a.jet_distance && along < a.jet_distance + a.jet_height;
    }

    // what the black hole does in a half of a cell's step
    struct AgnDose {
        double drained_share; // of the cold gas near the centre
        double heat; // per volume of the heated sphere
        double mass; // per volume of the heated sphere
        double jet_mass; // per volume of a jet
    };

    HD inline AgnDose dose_of(const Agn& a, double accretion_rate, double power, double time) {
        AgnDose dose{fmin(time / a.accretion_time, 1.0), 0.0, 0.0, 0.0};
        if constexpr (config::agn_thermal) {
            dose.heat = a.thermal_share * power * a.one_over_heated_volume * time;
            dose.mass = a.thermal_share * accretion_rate * a.one_over_heated_volume * time;
        }
        if constexpr (config::agn_kinetic) {
            dose.jet_mass =
                0.5 * (1.0 - a.efficiency) * a.kinetic_share * accretion_rate * a.one_over_jet_volume * time;
        }
        return dose;
    }

    // the cold gas loses the share that fell in; the heated sphere gets heat and mass; a jet adds mass at its speed
    HD inline void feed_back(const Agn& a, const Vec& position, const AgnDose& dose, CellGas& gas) {
        const Vec offset = from_centre(position);
        if (feeds_the_black_hole(a, offset, gas)) {
            gas.density *= (1.0 - dose.drained_share);
            gas.energy *= (1.0 - dose.drained_share);
        }

        if constexpr (config::agn_thermal) {
            if (in_the_heated_sphere(a, offset)) {
                if (dose.mass > 0.0) add_mass_at_rest_with_the_cell(dose.mass, gas);
                gas.energy += dose.heat;
                const double hottest = 0.5 * gas.density * norm2(gas.velocity) +
                                       gas.density * HOTTEST_HEATED_GAS / a.kelvin_per_specific_energy;
                if (gas.energy > hottest) gas.energy = hottest;
            }
        }

        if constexpr (config::agn_kinetic) {
            if (dose.jet_mass > 0.0 && in_a_jet(a, offset)) {
                const double sign        = (offset[1] > 0.0) ? 1.0 : -1.0;
                const double new_density = gas.density + dose.jet_mass;
                Vec          v           = (gas.density / new_density) * gas.velocity;
                v[1] += dose.jet_mass * sign * a.jet_speed / new_density;
                const double speed = sqrt(norm2(v));
                if (speed > a.jet_speed_cap) v = (a.jet_speed_cap / speed) * v;
                gas.velocity = v;
                gas.density  = new_density;
                gas.energy += 0.5 * dose.jet_mass * a.jet_speed * a.jet_speed;
            }
        }
    }

} // namespace astro
