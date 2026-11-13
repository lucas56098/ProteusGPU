#pragma once

// The CFL step of a cell, the fluxes over its faces from both sides, and the update of its conserved variables.

#include "gradients/gradients.h"
#include "riemann/riemann.h"
#include "timesteps/timesteps.h"

namespace hydro {

    // CFL step of every cell of the list into cfl_step[c]; mesh_velocity is nullptr on a mesh at rest
    void cfl_steps(const mesh::MeshPointers&     mesh,
                   const gas::PrimitivePointers& state,
                   const Vec*                    mesh_velocity,
                   double                        cfl_fraction,
                   const mesh::CellList&         cells,
                   double*                       cfl_step);

    // the cells whose thermal energy came out negative, and the energy that setting it to zero added
    struct ThermalFloorCount {
        long long cells  = 0;
        double    energy = 0.0;
    };

    // conserved variables of the cells of the list from their state and volume, and back with the temperature floor
    void conserved_from_primitive(const mesh::MeshPointers&     mesh,
                                  const gas::PrimitivePointers& state,
                                  const mesh::CellList&         cells,
                                  const gas::ConservedPointers& conserved);
    // as in Arepo (update_primitive_variables, MinEgySpec 0), a negative thermal energy is set to zero, and the total
    // energy to the kinetic one, before the temperature floor
    ThermalFloorCount primitive_from_conserved(const mesh::MeshPointers&     mesh,
                                               const gas::ConservedPointers& conserved,
                                               const mesh::CellList&         cells,
                                               double                        lowest_specific_energy,
                                               const gas::PrimitivePointers& state);

    // the mass of every cell of the list at the start of its step, which the flux limiter compares with
    void masses_at_start_of_step(const mesh::MeshPointers&     mesh,
                                 const gas::PrimitivePointers& state,
                                 const mesh::CellList&         cells,
                                 double*                       step_mass);

    // over how many faces the cells of the list took a flux, and how many of these fluxes the limiter cut
    struct FluxCount {
        long long faces   = 0;
        long long limited = 0;
    };

    // the conserved variables of the cells of the list into the backup, by place in the list, and back
    void save_conserved(const mesh::CellList& cells, const gas::ConservedPointers& conserved, gas::Conserved* backup);
    void restore_conserved(const mesh::CellList&         cells,
                           const gas::Conserved*         backup,
                           const gas::ConservedPointers& conserved);

    // how many cells of the list would lose more than the limiter allows
    long long count_over_exposed(const mesh::CellList& cells, const double* step_mass, const double* exposure);

    // takes the fluxes over the faces of the cells of the list out of their conserved variables, each for its share of
    // the time; mesh_velocity (nullptr on a mesh at rest) covers the cells and the ghost cells. The exposure of a cell
    // is the mass it would lose over its whole step through all its faces. The limiter: every flux out of a cell whose
    // exposure is above MOST_OF_UPWIND_MASS of its mass at the start of its step is cut by the same factor, so all its
    // faces together take at most that share (Arepo cuts each face alone). step_mass and exposure cover the cells and
    // the ghost cells; exposure is nullptr without the limiter, which then writes the exposures of the list into
    // exposure_out if that is not nullptr
    FluxCount apply_fluxes(const mesh::MeshPointers&          mesh,
                           const gas::PrimitivePointers&      state,
                           const gradients::GradientPointers& gradients,
                           const Vec*                         mesh_velocity,
                           const timesteps::StepPointers&     steps,
                           timesteps::Half                    half,
                           double                             lowest_specific_energy,
                           const double*                      step_mass,
                           const double*                      exposure,
                           double*                            exposure_out,
                           const mesh::CellList&              cells,
                           const gas::ConservedPointers&      conserved);

    // stops the run if a cell of the list has density <= 0, energy <= 0 or a NaN; collective
    void stop_if_unphysical(const gas::PrimitivePointers& state,
                            const gas::ConservedPointers& conserved,
                            const mesh::CellList&         cells);

    // CFL step of one cell; sound_at_least and speed_at_least are for a source about to bring those speeds
    HD inline double cfl_step_of(const gas::Primitive& w,
                                 double                volume,
                                 const Vec&            mesh_velocity,
                                 double                cfl_fraction,
                                 double                sound_at_least,
                                 double                speed_at_least) {
        const Vec against_mesh = config::moving_mesh ? w.velocity - mesh_velocity : w.velocity;
        double    sound        = (w.density > 0.0 && w.pressure > 0.0) ? gas::sound_speed(w) : 0.0;
        double    speed        = sqrt(norm2(against_mesh));
        if (sound_at_least > sound) sound = sound_at_least;
        if (speed_at_least > speed) speed = speed_at_least;
        return cfl_fraction * (radius_of_ball(volume) / (sound + speed));
    }

} // namespace hydro
