#pragma once

// Starts a run and finishes it.

#include "simulation/simulation.h"

namespace begrun {

    // hello, parameters, device, the cells, first cuts, first mesh and snapshot 0; collective
    void begrun(int argc, char* argv[], simulation::Simulation& sim);

    // timer summary, memory and the last line of the log; collective
    void endrun(const simulation::Simulation& sim);

} // namespace begrun
