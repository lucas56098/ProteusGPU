#pragma once

// The initial conditions in and the snapshots out, as HDF5 files that all ranks read and write together.

#include "astro/astro.h"
#include "cells/cells.h"
#include "mesh/mesh.h"
#include "profile/profile.h"
#include "timesteps/timesteps.h"

#include <string>

namespace io {

    // reads the initial conditions into cells, an even share of the rows per rank; id = row in the file; collective
    void read_initial_conditions(const char* path, cells::Cells& cells);

    // makes the output directory; stops the run if it already holds a first snapshot; collective
    void prepare_output_directory(const std::string& directory);

    // number of the last snapshot in the directory, -1 if there is none; collective
    int last_snapshot_number(const std::string& directory);

    // the time and the number of top steps of a snapshot
    struct SnapshotHeader {
        double time;
        int    step;
    };

    // reads snapshot_<number>.hdf5 into cells, with the ids, pressures and mesh velocities of the file; collective
    SnapshotHeader read_snapshot(const std::string& directory, int number, cells::Cells& cells);

    // writes snapshot_<number>.hdf5, one file for all ranks, each rank its cells as a block of rows; collective
    void write_snapshot(const std::string&             directory,
                        int                            number,
                        double                         time,
                        int                            step,
                        const astro::Units&            units,
                        const cells::Cells&            cells,
                        const mesh::MeshPointers&      mesh,
                        const ghosts::GhostPointers&   ghosts,
                        const timesteps::StepPointers& steps);

    // writes the profile of the run so far, anew each time: seconds of every timer per rank and top step; collective
    void write_profile(const std::string& directory, const std::string& name, const profile::Profile& profile);

} // namespace io
