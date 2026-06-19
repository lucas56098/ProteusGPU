#include "begrun/begrun.h"
#include "global/allvars.h"
#include "hydro/finite_volume_solver.h"
#include "io/output.h"
#include "mpi/halo.h"
#include "mpi/mpi_compat.h"
#include "mpi/rebalance.h"
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

    // init MPI
    proteus_mpi::init(&argc, &argv);

    // setup simulation
    begrun::begrun(argc, argv);

    // snapshot at t=0
    if (sim.snap_num == 0) {
        output.snapshot(sim.snap_num, sim.mesh, sim.primvar, sim.t_sim, sim.step);
        sim.snap_num += 1;
    }

    // hydro loop — block scopes the HYDRO profiler timer to the loop only
    {
        PROFILE("HYDRO");
        while (sim.t_sim < sim.t_end) {

            // calculate dt
            double dt = hydro::dt_CFL(sim.CFL, sim.mesh, sim.primvar);
            proteus_mpi::halo_dt_allreduce(&dt);

            // limit dt to t_nextoutput or t_end
            if (sim.t_sim + dt > sim.t_nextoutput) { dt = sim.t_nextoutput - sim.t_sim; }
            if (sim.t_sim + dt > sim.t_end) { dt = sim.t_end - sim.t_sim; }

            // print step, dt, ETA
            print_log(sim.step, sim.wall_start, sim.t_sim, dt, sim.t_start, sim.t_end);

            // diagnostic load-imbalance probe. The rebalance trigger itself fires
            // inside voronoi::move_mesh so it shares the mesh build with the regular step.
            proteus_mpi::rebalance_imbalance_log(sim.step, sim.mesh);

            // hydro step
            hydro::hydro_step(dt, sim.mesh, sim.primvar);
            sim.t_sim += dt;

            // write snapshot
            if (sim.t_sim >= sim.t_nextoutput || sim.t_sim >= sim.t_end) {
                output.snapshot(sim.snap_num, sim.mesh, sim.primvar, sim.t_sim, sim.step);
                sim.t_nextoutput += sim.output_dt;
                sim.snap_num += 1;
            }

            // write per-step entry into profile.hdf5
            Profiler::LogTimestep(sim.step);
            sim.step++;
        }
    } // HYDRO scope ends here

    // clean up
    begrun::endrun();
    proteus_mpi::finalize();
    return 0;
}
