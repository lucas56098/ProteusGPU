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
        long long             fluxes         = 0; // of this rank in the top step so far, for the log
        long long             limited_fluxes = 0; // of these, cut by the flux limiter
        int                   flux_passes    = 0; // halves of fluxes in the top step so far
        int                   redone_passes  = 0; // of these, done again with the limiter
        long long             floored_cells  = 0; // with a negative thermal energy set to zero, in the top step so far
        double                floor_energy   = 0.0; // the energy that added
        int                   step           = 0; // top steps done so far
        double                start_time     = 0.0; // of this run, also after a restart
        std::string           profile; // file name of the profile

        std::chrono::steady_clock::time_point start_of_run;
    };

} // namespace simulation
