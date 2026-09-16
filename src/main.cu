// main simulation routine

#include "astro/sources.h"
#include "begrun/begrun.h"
#include "global/allvars.h"
#include "hydro/finite_volume_solver.h"
#include "io/output.h"
#include "mpi/halo.h"
#include "mpi/mpi_compat.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

/*=========================================================================
        _____           _                    _____ _____  _    _
       |  __ \         | |                  / ____|  __ \| |  | |
       | |__) | __ ___ | |_ ___ _   _ ___  | |  __| |__) | |  | |
       |  ___/ '__/ _ \| __/ _ \ | | / __| | | |_ |  ___/| |  | |
       | |   | | | (_) | ||  __/ |_| \__ \ | |__| | |    | |__| |
       |_|   |_|  \___/ \__\___|\__,_|___/  \_____|_|     \____/

       GPU-accelerated moving mesh hydrodynamics for astrophysics
===========================================================================
Version: 0.8
Authors: Lucas Schleuss, Dylan Nelson
Institution: Institute of Theoretical Astrophysics, Heidelberg University
===========================================================================*/

int main(int argc, char* argv[]) {

    // start MPI and pick this rank's GPU
    proteus_mpi::init(&argc, &argv);

    // params, IC or snapshot, first mesh
    begrun::begrun(argc, argv);

    // snapshot at t = 0
    if (sim.snap_num == 0) { output.write_snapshot(); }

    {
        PROFILE("HYDRO");
        // time loop
        while (sim.t_sim < sim.t_end) {

            // per-step setup of the source terms
            astro::sources_prepare();

            // CFL timestep over all cells and ranks
            double dt = hydro::calc_timestep(sim.CFL, sim.mesh, sim.primvar);

            print_log();

            // half step sources, hydro step, half step sources
            astro::apply_sources_first_half(0.5 * dt);
            hydro::hydro_step(dt, sim.mesh, sim.primvar);
            astro::apply_sources_second_half(0.5 * dt);
            sim.t_sim += dt;

            // snapshot at every output time and at the end of the run
            if (sim.t_sim >= sim.t_nextoutput || sim.t_sim >= sim.t_end) { output.write_snapshot(); }

            // one row per step in profile.hdf5
            Profiler::log_timestep(sim.step);
            sim.step++;
        }
    }

    // free everything and print the summary
    begrun::endrun();
    proteus_mpi::finalize();
    return 0;
}
