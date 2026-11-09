#pragma once

// The algorithm of a run: begrun, the loop over the syncs (START, TIME, MESH, FINISH, OUTPUT), endrun.

namespace integrator {

    // from the initial conditions or the last snapshot up to time_end; collective
    void run(int argc, char* argv[]);

} // namespace integrator
