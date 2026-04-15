#include "begrun/begrun.h"
#include "global/allvars.h"
#include "hydro/finite_volume_solver.h"
#include "hydro/riemann.h"
#include "io/input.h"
#include "io/output.h"
#include "knn/knn.h"
#include "profiler/profiler.h"
#include "voronoi/periodic_mesh.h"
#include "voronoi/voronoi.h"
#include <chrono>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>
#include <sys/stat.h>

/*========================================================================
          _____           _                    _____ _____  _    _
         |  __ \         | |                  / ____|  __ \| |  | |
         | |__) | __ ___ | |_ ___ _   _ ___  | |  __| |__) | |  | |
         |  ___/ '__/ _ \| __/ _ \ | | / __| | | |_ |  ___/| |  | |
         | |   | | | (_) | ||  __/ |_| \__ \ | |__| | |    | |__| |
         |_|   |_|  \___/ \__\___|\__,_|___/  \_____|_|     \____/

==========================================================================
A GPU accelerated Moving-Mesh Hydrodynamics Code for Exascale Astrophysics
==========================================================================
Version: 0.5
Authors: Lucas Schleuss, Dylan Nelson
Institution: Institute of Theoretical Astrophysics, Heidelberg University
========================================================================*/

// main routine
int main(int argc, char* argv[]) {
    PROFILE_START("TOTAL_RUNTIME");

    const auto wall_start = std::chrono::steady_clock::now();

    // load input parameters and IC
    begrun::StartState state    = begrun::begrun(argc, argv);
    double             t_sim    = state.t_sim;
    int                snap_num = state.snap_num;

    // init hydro primvars from IC
    hydro::primvars* primvar = hydro::init(icData.seedpos_dims[0]);

    // compute voronoi mesh (allocate once, reuse every timestep)
    hsize_t n_hydro = icData.seedpos_dims[0];
    VMesh*  mesh    = voronoi::allocate_mesh(n_hydro);
    voronoi::compute_periodic_mesh(mesh, (POINT_TYPE*)icData.seedpos.data(), n_hydro);

    // allocate persistent hydro buffers (prim_new, gradients)
    hydro::allocate_hydro_buffers(n_hydro);

    // free IC data no longer needed
    begrun::free_initial_conditions();

    if (t_sim > 0.0) {
        std::cout << "HYDRO: restarted from t = " << t_sim << " (snap_num = " << snap_num << ")" << std::endl;
    } else {
        std::cout << "HYDRO: started" << std::endl;
    }

    // init simulation parameters
    const double t_start = t_sim;
    double       t_end   = input.getParameterDouble("time_end");
    double       CFL     = input.getParameterDouble("CFL_frac");
    int          step    = 0;

    double output_dt    = input.getParameterDouble("output_dt");
    double t_nextoutput = t_sim + output_dt;
    int    next_log     = 1;

#ifdef USE_HDF5
    if (snap_num == 0) {
        // write snapshot at t=0 (if not restarted)
        output.snapshot(snap_num, mesh, primvar, icData.seedpos_dims[0], t_sim);
        snap_num += 1;
    }
#endif

    // main hydro loop
    PROFILE_START("HYDRO_MAIN");
    while (t_sim < t_end) {

        // CFL timestep
        double dt = hydro::dt_CFL(CFL, mesh, primvar);

        // limit dt to next snapshot/t_end
        if (t_sim + dt > t_nextoutput) { dt = t_nextoutput - t_sim; }
        if (t_sim + dt > t_end) { dt = t_end - t_sim; }

        // hydro step
        hydro::hydro_step(dt, mesh, primvar);
        t_sim += dt;
        step++;

        // print log info
        if (step >= next_log || t_sim >= t_end) { print_log(step, wall_start, t_sim, dt, t_start, t_end, &next_log); }

#ifdef USE_HDF5
        // write snapshot
        if (t_sim >= t_nextoutput || t_sim >= t_end) {
            output.snapshot(snap_num, mesh, primvar, icData.seedpos_dims[0], t_sim);
            t_nextoutput += output_dt;
            snap_num += 1;
        }
#endif
    }
    PROFILE_END("HYDRO_MAIN");
    std::cout << "HYDRO: Finished after " << step << " steps at t = " << t_sim << std::endl;

    // delete mesh & hydro
    voronoi::free_mesh(mesh);
    hydro::free_prim(&primvar);
    hydro::free_hydro_buffers();

    // final printouts
    const double total_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    std::cout << "MAIN: Runtime = " << total_wall_s << " seconds" << std::endl;
    print_max_memory_usage();
    std::cout << "MAIN: Done." << std::endl;

    PROFILE_END("TOTAL_RUNTIME");
    PROFILE_PRINT_RESULTS();

    return 0;
}
