#pragma once

// Everything a run works on: begrun fills it, the loop of the integrator goes on with it.

#include "astro/astro.h"
#include "cells/cells.h"
#include "domain/domain.h"
#include "ghosts/ghosts.h"
#include "mesh/mesh.h"
#include "params/params.h"
#include "signals/signals.h"
#include "timesteps/timesteps.h"

#include <chrono>
#include <string>

namespace simulation {

    // the next snapshot: its number and when it is due
    struct Snapshots {
        int    number;
        double next_time;
    };

    struct Simulation {
        params::RunParameters parameters;
        cells::Cells          cells;
        astro::Sources        sources;
        signals::Signals      signals;
        domain::Domain        domain;
        ghosts::Ghosts        ghosts;
        mesh::Mesh            mesh;
        timesteps::TopStep    top;
        timesteps::Sync       sync{};
        Snapshots             snapshots{};
        int                   step       = 0; // top steps done so far
        double                start_time = 0.0; // of this run, also after a restart
        std::string           profile; // file name of the profile

        std::chrono::steady_clock::time_point start_of_run;
    };

} // namespace simulation
