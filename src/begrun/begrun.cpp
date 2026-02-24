#include <iostream>
#include <stdio.h>
#include <thread>
#include <chrono>
#include "begrun.h"
#include "../io/input.h"
#include "../global/allvars.h"

namespace begrun {

void print_banner() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << R"(                                                                                                                                                       
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
    std::cout << "Version: 0.0.1" << std::endl;
    std::cout << "Build date: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Authors: Lucas Schleuss, Dylan Nelson" << std::endl;
    std::cout << "Institution: Institute of Theoretical Astrophysics, Heidelberg University" << std::endl;
    std::cout << "==========================================================================" << std::endl;


#if (!defined(dim_3D) && !defined(dim_2D)) || (defined(dim_3D) && defined(dim_2D))
    #error "Choose a dimension in Config.sh: [dim_3D] OR [dim_2D]"
#elif dim_2D
    std::cout << "Running in 2D mode" << std::endl;
#elif dim_3D
    std::cout << "Running in 3D mode" << std::endl;
#endif

#ifdef CPU_DEBUG
    std::cout << "CPU debug mode enabled" << std::endl;
#endif

    // early exit for CI test
#ifdef DRY_RUN
    exit(EXIT_SUCCESS);
#endif
}

InputHandler loadInputFiles(int argc, char* argv[]) {

    std::string paramFile = "param.txt";
    if (argc > 1) {
        paramFile = argv[1];
    }

    InputHandler input(paramFile);
    if (!input.loadParameters()) {
        std::cerr << "Failed to load parameters. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    _boxsize_ = input.getParameterDouble("box_size");

    return input;
}

} // namespace begrun
