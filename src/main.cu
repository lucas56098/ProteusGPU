/* main simulation routine */
#include "astro/sources.h"
#include "astro/agn.h"
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
            #ifdef AGN_ENABLED
            // one global reduction per step; the cached cold mass feeds both the CFL ceilings
            // below and both Strang halves
            astro::agn_prepare();
            #endif
            double dt = hydro::calc_timestep(sim.CFL, sim.mesh, sim.primvar);

            // print diagnostics
            print_log();

            // hydro step
            // Strang split: half a step of source terms on either side of the hydro update.
            // No-ops unless an astro module is compiled in.
            astro::apply_sources_first_half(0.5 * dt);
            hydro::hydro_step(dt, sim.mesh, sim.primvar);
            astro::apply_sources_second_half(0.5 * dt);
            sim.t_sim += dt;

            // write snapshot
            if (sim.t_sim >= sim.t_nextoutput || sim.t_sim >= sim.t_end) { output.write_snapshot(); }

            // log profiling times
            Profiler::LogTimestep(sim.step);
            sim.step++;
        }
    }

    // clean up
    begrun::endrun();
    proteus_mpi::finalize();
    return 0;
}
