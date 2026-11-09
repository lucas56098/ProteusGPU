#pragma once

// The gas in one cell: primitive and conserved variables, and the ideal gas.

#include "math/math.h"

namespace gas {

    struct Primitive {
        double density;
        Vec    velocity;
        double pressure;
    };

    // what a cell holds: mass, momentum, total energy
    struct Conserved {
        double mass;
        Vec    momentum;
        double energy;
    };

    // what flows through a face, per area and time
    struct Flux {
        double mass;
        Vec    momentum;
        double energy;
    };

    // one array per variable, for kernels
    struct PrimitivePointers {
        double* density;
        Vec*    velocity;
        double* pressure;
    };
    struct ConservedPointers {
        double* mass;
        Vec*    momentum;
        double* energy;
    };

    HD inline Primitive load(const PrimitivePointers& w, int cell);
    HD inline Conserved load(const ConservedPointers& u, int cell);
    HD inline void store(const PrimitivePointers& w, int cell, const Primitive& value);
    HD inline void store(const ConservedPointers& u, int cell, const Conserved& value);

    // total energy per volume: P / (gamma - 1) + rho v^2 / 2
    HD inline double energy_density(const Primitive& w);

    HD inline double sound_speed(const Primitive& w);

    HD inline double pressure_of(double density, const Vec& velocity, double energy_per_volume);

    HD inline Conserved conserved_of(const Primitive& w, double volume);
    HD inline Primitive primitive_of(const Conserved& u, double volume);

    // the velocity along the normal and tangents of a frame, and the momentum of a flux back along the axes of the box
    HD inline Primitive in_frame(const Primitive& w, const Frame& frame);
    HD inline Flux out_of_frame(const Flux& f, const Frame& frame);

    // gas seen from a frame moving at frame_velocity, and a flux found in that frame as seen from the frame at rest
    HD inline Primitive in_moving_frame(const Primitive& w, const Vec& frame_velocity);
    HD inline Flux out_of_moving_frame(const Flux& f, const Vec& frame_velocity);

    // density and pressure at least their floors; the pressure floor is that of lowest_specific_energy if above zero
    HD inline Primitive with_floors(Primitive w, double lowest_specific_energy);

    // at least lowest_specific_energy; a nearly empty cell gets EMPTY_CELL_DENSITY, velocity_before and that energy
    HD inline Primitive with_temperature_floor(Primitive w, const Vec& velocity_before, double lowest_specific_energy);

} // namespace gas

// how the functions above work
#include "state.h"
