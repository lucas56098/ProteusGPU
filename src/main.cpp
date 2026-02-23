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


    // -------- actual code starts here --------
    voronoi::compute_mesh((POINT_TYPE*) pts.data(), icData, input, output);

    std::cout << "Done." << std::endl;

    return 0;
}
