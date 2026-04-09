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

#include <stdio.h>
#include <vector>

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

int main(int argc, char* argv[]) {
    PROFILE_START("TOTAL_RUNTIME");

    const auto wall_start = std::chrono::steady_clock::now();

    // say hi and fill/prepare structs
    begrun::begrun(argc, argv);

    // init hydro values
    primvars* primvar = hydro::init(icData.seedpos_dims[0]);

    // compute voronoi mesh
    VMesh* mesh = voronoi::compute_periodic_mesh((POINT_TYPE*)icData.seedpos.data(), icData.seedpos_dims[0]);

    // free IC data no longer needed
    begrun::free_initial_conditions();

    // start timestep loop
    std::cout << "HYDRO: started" << std::endl;

    double t_sim = 0.0;
    double t_end = std::stof(input.getParameter("time_end"));
    double CFL   = std::stof(input.getParameter("CFL_frac"));
    int    step  = 0;

    double output_dt    = std::stof(input.getParameter("output_dt"));
    double t_nextoutput = t_sim + output_dt;
    int    snap_num = 0, next_log = 1;

// write first snapshot at t=0
#ifdef USE_HDF5
    output.snapshot(snap_num, mesh, primvar, icData.seedpos_dims[0], t_sim);
    snap_num += 1;
#endif

    PROFILE_START("HYDRO_MAIN");
    while (t_sim < t_end) {
        double dt = hydro::dt_CFL(CFL, mesh, primvar);

        // go at most to next output time
        if (t_sim + dt > t_nextoutput) { dt = t_nextoutput - t_sim; }

        // make sure we exactly hit t_end
        if (t_sim + dt > t_end) { dt = t_end - t_sim; }

        mesh = hydro::hydro_step(dt, mesh, primvar);
        t_sim += dt;
        step++;

        if (step >= next_log || t_sim >= t_end) {
            const double elapsed_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
            std::cout << "HYDRO: Step " << step << "  t = " << t_sim << "  dt = " << dt
                      << "  ETA = " << format_hms((t_sim > 0.0) ? elapsed_s * (t_end - t_sim) / t_sim : 0.0)
                      << std::endl;
            const int guess = (elapsed_s > 1e-12) ? static_cast<int>(std::round(3.0 * step / elapsed_s)) : 1;
            const int bucket =
                static_cast<int>(std::pow(10.0, std::floor(std::log10(static_cast<double>(guess > 1 ? guess : 1)))));
            next_log = ((step / bucket) + 1) * bucket;
        }

// write output
#ifdef USE_HDF5
        if (t_sim >= t_nextoutput || t_sim == t_end) {

            output.snapshot(snap_num, mesh, primvar, icData.seedpos_dims[0], t_sim);

            t_nextoutput += output_dt;
            snap_num += 1;
        }
#endif
    }
    PROFILE_END("HYDRO_MAIN");

    std::cout << "HYDRO: Finished after " << step << " steps at t = " << t_sim << std::endl;

    // delete mesh & hydro
    voronoi::free_vmesh(mesh);
    hydro::free_prim(&primvar);

    const double total_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    std::cout << "MAIN: Runtime = " << format_hms(total_wall_s) << std::endl;
    std::cout << "MAIN: Done." << std::endl;

    PROFILE_END("TOTAL_RUNTIME");
    PROFILE_PRINT_RESULTS();

    return 0;
}
