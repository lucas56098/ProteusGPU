#include <iostream>
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
Version: 0.0.1 
Authors: Lucas Schleuss, Dylan Nelson
Institution: Institute of Theoretical Astrophysics, Heidelberg University
========================================================================*/

int main(int argc, char* argv[]) {

    // say hi and fill/prepare structs
    begrun::begrun(argc, argv);

    // init hydro values
    primvars* primvar = hydro::init(icData.seedpos_dims[0]);

    // compute voronoi mesh
    VMesh* mesh = voronoi::compute_periodic_mesh((POINT_TYPE*) icData.seedpos.data(), icData.seedpos_dims[0]);

    std::cout << "Hydro started" << std::endl;

    double t_sim = 0.0;
    double t_end = 0.1;
    double CFL = 0.4;
    int step = 0;

    while (t_sim < t_end) {
        double dt = hydro::dt_CFL(CFL, mesh, primvar);

        // make sure we exactly hit t_end
        if (t_sim + dt > t_end) { dt = t_end - t_sim; }

        hydro::hydro_step(dt, mesh, primvar);
        t_sim += dt;
        step++;

        if (step % 3 == 0) {
            std::cout << "Step " << step << "  t = " << t_sim << "  dt = " << dt << std::endl;
        }
    }

    std::cout << "Finished after " << step << " steps at t = " << t_sim << std::endl;

    std::cout << "Hydro finished" << std::endl;

    // convert VMesh to MeshCellData and write to HDF5 file
    #ifdef USE_HDF5
    MeshCellData meshData;
    voronoi::vmesh_to_meshdata(mesh, meshData);

    std::string mesh_output_file = input.getParameter("output_mesh_file");
    if (!output.writeMeshFile(mesh_output_file, meshData, primvar, icData.seedpos_dims[0])) { exit(EXIT_FAILURE); }
    #endif

    // delete mesh & hydro
    voronoi::free_vmesh(mesh);
    hydro::free_prim(&primvar);

    std::cout << "MAIN: Done." << std::endl;
    return 0;
}
