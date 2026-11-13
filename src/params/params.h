#pragma once

// Reads the command line and param.txt, with every parameter of a run defined once.

#include "astro/astro.h"

#include <string>

namespace params {

    // the parameters of a run; on the CPU only
    struct RunParameters {
        std::string ic_file;
        std::string output_directory;
        double      output_dt           = 0.0; // time between two snapshots
        bool        from_last_snapshot  = false; // go on from the last snapshot in output_directory
        bool        write_snapshots     = true; // optional write_snapshots = 0: only the profile, as for timing runs
        double      time_end            = 0.0;
        double      cfl_fraction        = 0.0; // the share of its CFL step a cell takes
        double      memory_limit_in_gb  = 0.0; // per rank
        double      reference_cell_size = 0.0; // of the volume regularization; 0 for the mean cell

        // with ASTRO_PHYSICS: the code units and the sources of the switches turned on
        astro::Parameters astro;
    };

    // reads the command line, <parameter file> [0|1, 1 to restart], and the file; stops the run if anything is wrong
    RunParameters read_parameters(int argc, char* argv[]);

} // namespace params
