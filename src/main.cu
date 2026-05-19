#include "begrun/begrun.h"
#include "global/allvars.h"
#include "hydro/finite_volume_solver.h"
#include "io/input.h"
#include "io/output.h"
#include "mpi/halo.h"
#include "mpi/mpi_init.h"
#include "profiler/profiler.h"
#include "voronoi/periodic_mesh.h"
#include "voronoi/voronoi.h"
#include <chrono>
#include <iostream>

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

    // initialize MPI (no-op single-node) and pin GPU per rank
    proteus_mpi::init(&argc, &argv);

    Profiler::StartTimer("TOTAL_RUNTIME");
    const auto wall_start = std::chrono::steady_clock::now();

    // parse param file, read IC (or latest snapshot for a restart)
    begrun::StartState state    = begrun::begrun(argc, argv);
    double             t_sim    = state.t_sim;
    int                snap_num = state.snap_num;

    // initialize primitive variables (rho, v, E) from IC
    hsize_t          n_hydro = icData.pos_dims[0];
    hydro::primvars* primvar = hydro::init(n_hydro);
    hydro::allocate_hydro_buffers(n_hydro);
    hydro::primvars* prim_new = hydro::prim_new_buffer();

    // build the initial Voronoi mesh from the seed positions
    VMesh* mesh = voronoi::allocate_mesh(n_hydro);
    voronoi::compute_periodic_mesh(mesh, (POINT_TYPE*)icData.pos.data(), n_hydro, primvar, prim_new);

    // IC arrays are no longer needed — free their host memory
    begrun::free_initial_conditions();

    // report initial memory usage
    print_max_memory_usage();

    if (t_sim > 0.0) {
        logging::root() << "HYDRO: restarted from t = " << t_sim << " (snap_num = " << snap_num << ")" << std::endl;
    } else {
        logging::root() << "HYDRO: started" << std::endl;
    }

    // diagnostic output text files
    std::string profile_log_path = input.getParameter("output_directory") + "/" + "profile.txt";

    logging::FileLogger profile_log(profile_log_path);

    // simulation control parameters
    const double t_start = t_sim;
    double       t_end   = input.getParameterDouble("time_end");
    double       CFL     = input.getParameterDouble("CFL_frac");
    int          step    = 0;

    double output_dt    = input.getParameterDouble("output_dt");
    double t_nextoutput = t_sim + output_dt;

    if (snap_num == 0) {
        // write snapshot at t=0 (if not restarted)
        output.snapshot(snap_num, mesh, primvar, icData.pos_dims[0], t_sim);
        snap_num += 1;
    }

    // hydro loop
    Profiler::StartTimer("HYDRO_MAIN");

    while (t_sim < t_end) {

        // CFL timestep: local min, then Allreduce(MIN) so every rank uses the same dt
        double dt = hydro::dt_CFL(CFL, mesh, primvar);
        proteus_mpi::halo_dt_allreduce(&dt);

        // limit dt to the next snapshot or t_end
        bool snap_to_output = false;
        bool snap_to_end    = false;
        if (t_sim + dt > t_nextoutput) {
            dt             = t_nextoutput - t_sim;
            snap_to_output = true;
        }
        if (t_sim + dt > t_end) {
            dt             = t_end - t_sim;
            snap_to_output = false;
            snap_to_end    = true;
        }

        // log info
        print_log(step, wall_start, t_sim, dt, t_start, t_end);

        Profiler::PrintTimestep(step, profile_log.root());

        // hydro step
        hydro::hydro_step(dt, mesh, primvar);

        // snap t_sim exactly to the boundary we clamped to
        if (snap_to_end)
            t_sim = t_end;
        else if (snap_to_output)
            t_sim = t_nextoutput;
        else
            t_sim += dt;
        step++;

        // write snapshot
        if (t_sim >= t_nextoutput || t_sim >= t_end) {
            output.snapshot(snap_num, mesh, primvar, icData.pos_dims[0], t_sim);
            t_nextoutput += output_dt;
            snap_num += 1;
        }
    }
    Profiler::EndTimer("HYDRO_MAIN");
    logging::root() << "HYDRO: Finished after " << step << " steps at t = " << t_sim << std::endl;

    // free all simulation buffers
    voronoi::free_mesh(mesh);
    hydro::free_prim(&primvar);
    hydro::free_hydro_buffers();
    proteus_mpi::halo_free();

    // wall-clock runtime + peak memory + profiler dump
    const double total_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    logging::root() << "MAIN: Done. (Total runtime = " << total_wall_s << " seconds (wallclock)" << std::endl;

    print_max_memory_usage();

    Profiler::EndTimer("TOTAL_RUNTIME");
    Profiler::PrintResults();

    proteus_mpi::finalize();

    return 0;
}
