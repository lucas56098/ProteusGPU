#pragma once

// load, store and the ideal gas of gas.h (included by gas.h)

#include <cmath>

namespace gas {

    // the floors, in code units
    constexpr double DENSITY_FLOOR      = 1e-12;
    constexpr double PRESSURE_FLOOR     = 1e-12;
    constexpr double EMPTY_CELL_DENSITY = 1e-13;

    HD inline Primitive load(const PrimitivePointers& w, int cell) {
        return Primitive{w.density[cell], w.velocity[cell], w.pressure[cell]};
    }

    HD inline Conserved load(const ConservedPointers& u, int cell) {
        return Conserved{u.mass[cell], u.momentum[cell], u.energy[cell]};
    }

    HD inline void store(const PrimitivePointers& w, int cell, const Primitive& value) {
        w.density[cell]  = value.density;
        w.velocity[cell] = value.velocity;
        w.pressure[cell] = value.pressure;
    }

    HD inline void store(const ConservedPointers& u, int cell, const Conserved& value) {
        u.mass[cell]     = value.mass;
        u.momentum[cell] = value.momentum;
        u.energy[cell]   = value.energy;
    }

    HD inline double energy_density(const Primitive& w) {
        return w.pressure / (config::adiabatic_index - 1.0) + 0.5 * w.density * norm2(w.velocity);
    }

    HD inline double sound_speed(const Primitive& w) {
        return sqrt(config::adiabatic_index * w.pressure / w.density);
    }

    HD inline double pressure_of(double density, const Vec& velocity, double energy_per_volume) {
        return (config::adiabatic_index - 1.0) * (energy_per_volume - 0.5 * density * norm2(velocity));
    }

    HD inline Conserved conserved_of(const Primitive& w, double volume) {
        const double mass = w.density * volume;
        return Conserved{mass, mass * w.velocity, energy_density(w) * volume};
    }

    HD inline Primitive primitive_of(const Conserved& u, double volume) {
        const double density  = u.mass / volume;
        const Vec    velocity = u.momentum / u.mass;
        const double pressure = (config::adiabatic_index - 1.0) * (u.energy / volume - 0.5 * density * norm2(velocity));
        return Primitive{density, velocity, pressure};
    }

    HD inline Primitive in_frame(const Primitive& w, const Frame& frame) {
        Primitive result   = w;
        result.velocity[0] = dot(w.velocity, frame.normal);
        for (int t = 0; t < config::dim - 1; t++)
            result.velocity[t + 1] = dot(w.velocity, frame.tangent[t]);
        return result;
    }

    HD inline Flux out_of_frame(const Flux& f, const Frame& frame) {
        Flux result     = f;
        result.momentum = f.momentum[0] * frame.normal;
        for (int t = 0; t < config::dim - 1; t++)
            result.momentum += f.momentum[t + 1] * frame.tangent[t];
        return result;
    }

    HD inline Primitive in_moving_frame(const Primitive& w, const Vec& frame_velocity) {
        Primitive result = w;
        result.velocity  = w.velocity - frame_velocity;
        return result;
    }

    // the energy takes the momentum flux of the moving frame, before it gains its share of the mass flux
    HD inline Flux out_of_moving_frame(const Flux& f, const Vec& frame_velocity) {
        Flux result     = f;
        result.momentum = f.momentum + f.mass * frame_velocity;
        result.energy   = f.energy + dot(f.momentum, frame_velocity) + 0.5 * f.mass * norm2(frame_velocity);
        return result;
    }

    HD inline Primitive with_floors(Primitive w, double lowest_specific_energy) {
        if (w.density < DENSITY_FLOOR) w.density = DENSITY_FLOOR;
        const double lowest_pressure = (lowest_specific_energy > 0.0)
                                           ? (config::adiabatic_index - 1.0) * w.density * lowest_specific_energy
                                           : PRESSURE_FLOOR;
        if (w.pressure < lowest_pressure) w.pressure = lowest_pressure;
        return w;
    }

    HD inline Primitive with_temperature_floor(Primitive w, const Vec& velocity_before, double lowest_specific_energy) {
        if (w.density < EMPTY_CELL_DENSITY) {
            return Primitive{EMPTY_CELL_DENSITY,
                             velocity_before,
                             (config::adiabatic_index - 1.0) * EMPTY_CELL_DENSITY * lowest_specific_energy};
        }
        const double lowest_pressure = (config::adiabatic_index - 1.0) * w.density * lowest_specific_energy;
        if (w.pressure < lowest_pressure) w.pressure = lowest_pressure;
        return w;
    }

} // namespace gas
