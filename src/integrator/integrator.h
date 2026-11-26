#pragma once

// The algorithm of a run: the loop over the syncs (START, TIME, MESH, FINISH, OUTPUT).

namespace simulation {
    struct Simulation;
} // namespace simulation

namespace integrator {

    // from the state begrun left in sim up to time_end; collective
    void run(simulation::Simulation& sim);

} // namespace integrator
