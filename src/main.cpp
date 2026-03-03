#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include "global/allvars.h"
#include "io/input.h"
#include "io/output.h"
#include "knn/knn.h"
#include "begrun/begrun.h"
#include "voronoi/voronoi.h"

int main(int argc, char* argv[]) {
    // welcome :D

    // structs for input, output and IC handling
    InputHandler input;
    ICData icData;
    OutputHandler output;

    // say hi and fill/prepare structs
    begrun::begrun(argc, argv, input, icData, output);

    // -------- actual code starts here --------
    VMesh* mesh = voronoi::compute_mesh((POINT_TYPE*) icData.seedpos.data(), icData, input, output);

    // convert VMesh to MeshCellData and write to HDF5 file
    #ifdef USE_HDF5
    MeshCellData meshData;
    voronoi::vmesh_to_meshdata(mesh, meshData);

    std::string mesh_output_file = input.getParameter("output_mesh_file");
    if (!output.writeMeshFile(mesh_output_file, meshData)) { exit(EXIT_FAILURE); }
    #endif

    voronoi::free_vmesh(mesh);

    std::cout << "MAIN: Done." << std::endl;

    return 0;
}
