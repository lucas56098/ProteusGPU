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

    // conserved variables of the cells of the list from their state and volume, and back with the temperature floor
    void conserved_from_primitive(const mesh::MeshPointers&     mesh,
                                  const gas::PrimitivePointers& state,
                                  const mesh::CellList&         cells,
                                  const gas::ConservedPointers& conserved);
    void primitive_from_conserved(const mesh::MeshPointers&     mesh,
                                  const gas::ConservedPointers& conserved,
                                  const mesh::CellList&         cells,
                                  double                        lowest_specific_energy,
                                  const gas::PrimitivePointers& state);

    // takes the fluxes over the faces of the cells of the list out of their conserved variables, each for its share of
    // the time; mesh_velocity (nullptr on a mesh at rest) covers the cells and the ghost cells
    void apply_fluxes(const mesh::MeshPointers&          mesh,
                      const gas::PrimitivePointers&      state,
                      const gradients::GradientPointers& gradients,
                      const Vec*                         mesh_velocity,
                      const timesteps::StepPointers&     steps,
                      timesteps::Half                    half,
                      double                             lowest_specific_energy,
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
