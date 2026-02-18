#include <iostream>
#include <stdio.h>
#include <vector>
#include "global/allvars.h"
#include "io/input.h"
#include "io/output.h"
#include "knn/knn.h"
#include "begrun/begrun.h"

int main(int argc, char* argv[]) {
    
    // welcome
    begrun::print_banner();

    // load param.txt
    InputHandler input = begrun::loadInputFiles(argc, argv);
    
    // initalize output handler
    std::string outputDir = input.getParameter("output_directory");
    OutputHandler output(outputDir);
    if (!output.initialize()) return EXIT_FAILURE;

    // read IC file
    ICData icData;
    if (!input.readICFile(input.getParameter("ic_file"), icData)) {return EXIT_FAILURE;}

    std::vector<double> pts = icData.seedpos;


    // --- actual code starts here ---
    _K_ = input.getParameterInt("knn_k");

    // define knn problem
    knn_problem *knn = NULL;

    // prepare knn problem
    knn = knn::init((double3*) pts.data(), icData.seedpos_dims[0]);



    knn::printInfo();

    knn::knn_free(&knn);

    // write mesh file
    //MeshCellData meshData;
    // here meshData would have to be filled with actual data
    //if (!output.writeMeshFile(input.getParameter("output_mesh_file"), meshData)) {return EXIT_FAILURE;}

    return 0;
}
