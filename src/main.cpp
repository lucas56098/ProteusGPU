#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <climits>
#include "global/allvars.h"
#include "io/input.h"
#include "io/output.h"
#include "knn/knn.h"
#include "begrun/begrun.h"
#include "voronoi/voronoi.h"
#include "voronoi/periodic_mesh.h"
#include "hydro/finite_volume_solver.h"
#include "hydro/riemann.h"
#include "profiler/profiler.h"

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
Version: 0.2
Authors: Lucas Schleuss, Dylan Nelson
Institution: Institute of Theoretical Astrophysics, Heidelberg University
========================================================================*/

int main(int argc, char* argv[]) {
    PROFILE_START("TOTAL_RUNTIME");

    // say hi and fill/prepare structs
    begrun::begrun(argc, argv);

    // init hydro values
    primvars* primvar = hydro::init(icData.seedpos_dims[0]);

    // compute voronoi mesh
    VMesh* mesh = voronoi::compute_periodic_mesh((POINT_TYPE*) icData.seedpos.data(), icData.seedpos_dims[0]);

    // start timestep loop
    std::cout << "Hydro started" << std::endl;

    double t_sim = 0.0;
    double t_end = std::stof(input.getParameter("time_end"));
    double CFL = 0.4;
    int step = 0;

    double output_dt = std::stof(input.getParameter("output_dt"));
    double t_nextoutput = t_sim + output_dt;
    int snap_num = 0;

    PROFILE_START("HYDRO_MAIN");
    while (t_sim < t_end) {
        double dt = hydro::dt_CFL(CFL, mesh, primvar);

	// go at most to next output time
	if (t_sim + dt > t_nextoutput) { dt = t_nextoutput - t_sim; }

        // make sure we exactly hit t_end
        if (t_sim + dt > t_end) { dt = t_end - t_sim; }

        hydro::hydro_step(dt, mesh, primvar);
        t_sim += dt;
        step++;

	// write output
        #ifdef USE_HDF5
        if (t_sim >= t_nextoutput) {
            PROFILE_START("SNAPSHOTS");
            MeshCellData meshData;
            voronoi::vmesh_to_meshdata(mesh, meshData);

            std::string output_file = "snapshot_" + std::to_string(snap_num) + ".hdf5";
            if (!output.writeSnapshot(output_file, meshData, primvar, icData.seedpos_dims[0], t_sim)) { exit(EXIT_FAILURE); }
            t_nextoutput += output_dt;
            snap_num += 1;
            PROFILE_END("SNAPSHOTS");
        }
        #endif

        if (step % 10 == 0) {
            std::cout << "Step " << step << "  t = " << t_sim << "  dt = " << dt << std::endl;
        }
    }
    PROFILE_END("HYDRO_MAIN");

    std::cout << "Finished after " << step << " steps at t = " << t_sim << std::endl;
    std::cout << "Hydro finished" << std::endl;

    // convert VMesh to MeshCellData and write to HDF5 file
    PROFILE_START("SNAPSHOTS");
    #ifdef USE_HDF5
    MeshCellData meshData;
    voronoi::vmesh_to_meshdata(mesh, meshData);

    std::string output_file = "snapshot_" + std::to_string(snap_num) + ".hdf5";
    if (!output.writeSnapshot(output_file, meshData, primvar, icData.seedpos_dims[0], t_sim)) { exit(EXIT_FAILURE); }
    #endif
    PROFILE_END("SNAPSHOTS");

    // delete mesh & hydro
    voronoi::free_vmesh(mesh);
    hydro::free_prim(&primvar);

    std::cout << "MAIN: Done." << std::endl;

    PROFILE_END("TOTAL_RUNTIME");
    PROFILE_PRINT_RESULTS();

    return 0;
}
