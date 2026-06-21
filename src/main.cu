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

    // setup simulation
    proteus_mpi::init(&argc, &argv);
    begrun::begrun(argc, argv);

    // snapshot at t=0
    if (sim.snap_num == 0) { output.write_snapshot(); }

    // hydro loop
    {
        PROFILE("HYDRO");
        while (sim.t_sim < sim.t_end) {

            // calculate dt
            double dt = hydro::calc_timestep(sim.CFL, sim.mesh, sim.primvar);

            // print diagnostics
            print_log();

            // hydro step
            hydro::hydro_step(dt, sim.mesh, sim.primvar);
            sim.t_sim += dt;

            // write snapshot
            if (sim.t_sim >= sim.t_nextoutput || sim.t_sim >= sim.t_end) { output.write_snapshot(); }

            // log profiling times
            Profiler::LogTimestep(sim.step);
            sim.step++;
        }
    } // HYDRO scope ends here

    // clean up
    begrun::endrun();
    proteus_mpi::finalize();
    return 0;
}
