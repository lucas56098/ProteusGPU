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

    // compute voronoi mesh
    VMesh* mesh = voronoi::compute_periodic_mesh((POINT_TYPE*) icData.seedpos.data(), icData.seedpos_dims[0]);

    // convert VMesh to MeshCellData and write to HDF5 file
    #ifdef USE_HDF5
    MeshCellData meshData;
    voronoi::vmesh_to_meshdata(mesh, meshData);

    std::string mesh_output_file = input.getParameter("output_mesh_file");
    if (!output.writeMeshFile(mesh_output_file, meshData)) { exit(EXIT_FAILURE); }
    #endif

    // delete mesh
    voronoi::free_vmesh(mesh);

    std::cout << "MAIN: Done." << std::endl;
    return 0;
}
