#include "begrun.h"
#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../profiler/profiler.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

namespace begrun {

    // initalize Proteus
    void begrun(int argc, char* argv[]) {

        PROFILE_START("BEGRUN");

        // welcome message
        print_banner();

// print useful information
#ifdef dim_2D
        std::cout << "BEGRUN: Running in 2D mode" << std::endl;
#elif dim_3D
        std::cout << "BEGRUN: Running in 3D mode" << std::endl;
#endif

#ifdef CPU_DEBUG
        std::cout << "BEGRUN: CPU mode enabled" << std::endl;
#endif

#ifdef DRY_RUN
        std::cout << "Dry run for CI test successful, exiting." << std::endl;
        exit(EXIT_SUCCESS);
#endif

        // load param.txt
        input = loadInputFiles(argc, argv);

        // read IC file
        if (!input.readICFile(input.getParameter("ic_file"), icData)) { exit(EXIT_FAILURE); }

        // adapt buff for periodic bc to IC resolution (naive, will have to be improved later)
        buff = (1. / pow(icData.seedpos_dims[0], 1. / ((double)DIMENSION))) *
               4; // for approx unform grid this would be 4 layers of ghost points...
#ifdef MOVING_MESH
        // read mesh regularization parameters
        CellShapingSpeed  = input.getParameterDouble("CellShapingSpeed");
        CellShapingFactor = input.getParameterDouble("CellShapingFactor");
        std::cout << "BEGRUN: CellShapingSpeed  = " << CellShapingSpeed << std::endl;
        std::cout << "BEGRUN: CellShapingFactor = " << CellShapingFactor << std::endl;
#endif

        // init output folder
        output = OutputHandler(input.getParameter("output_directory"));
        if (!output.initialize()) { exit(EXIT_FAILURE); }

        PROFILE_END("BEGRUN");
    }

    void free_initial_conditions() {
        std::vector<double>().swap(icData.seedpos);
        std::vector<double>().swap(icData.rho);
        std::vector<double>().swap(icData.vel);
        std::vector<double>().swap(icData.Energy);
    }

    // prints welcome message
    void print_banner() {
        std::cout << "==========================================================================" << std::endl;
        std::cout
            << R"(                                                                                                                                                       
          _____           _                    _____ _____  _    _ 
         |  __ \         | |                  / ____|  __ \| |  | |
         | |__) | __ ___ | |_ ___ _   _ ___  | |  __| |__) | |  | |
         |  ___/ '__/ _ \| __/ _ \ | | / __| | | |_ |  ___/| |  | |
         | |   | | | (_) | ||  __/ |_| \__ \ | |__| | |    | |__| |
         |_|   |_|  \___/ \__\___|\__,_|___/  \_____|_|     \____/ 

    )" << std::endl;
        std::cout << "==========================================================================" << std::endl;
        std::cout << "A GPU accelerated Moving-Mesh Hydrodynamics Code for Exascale Astrophysics" << std::endl;
        std::cout << "==========================================================================" << std::endl;
        std::cout << "Version: 0.5" << std::endl;
        std::cout << "Build date: " << __DATE__ << " " << __TIME__ << std::endl;
        std::cout << "Authors: Lucas Schleuss, Dylan Nelson" << std::endl;
        std::cout << "Institution: Institute of Theoretical Astrophysics, Heidelberg University" << std::endl;
        std::cout << "==========================================================================" << std::endl;
    }

    // loads input parameters from param.txt into InputHandler
    InputHandler loadInputFiles(int argc, char* argv[]) {

        // default is param.txt, otherwise ./ProteusGPU <param_file>
        std::string paramFile = "./ics/param.txt";
        if (argc > 1) { paramFile = argv[1]; }

        // load parameters into InputHandler
        InputHandler input(paramFile);
        if (!input.loadParameters()) {
            std::cerr << "BEGRUN: Failed to load parameters. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }

        return input;
    }

} // namespace begrun
