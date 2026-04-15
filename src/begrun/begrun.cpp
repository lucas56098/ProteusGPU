#include "begrun.h"
#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../profiler/profiler.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace begrun {

    // initalize Proteus
    StartState begrun(int argc, char* argv[]) {
        PROFILE_START("BEGRUN");

        print_banner();

        // print basic info
#ifdef dim_2D
#ifdef CPU_DEBUG
        std::cout << "BEGRUN: Running 2D mode on CPU" << std::endl;
#else
        std::cout << "BEGRUN: Running 2D mode on GPU" << std::endl;
#endif
#elif dim_3D
#ifdef CPU_DEBUG
        std::cout << "BEGRUN: Running 3D mode on GPU" << std::endl;
#else
        std::cout << "BEGRUN: Running 3D mode on GPU" << std::endl;
#endif
#endif

#ifdef DRY_RUN
        std::cout << "Dry run for CI test successful, exiting." << std::endl;
        exit(EXIT_SUCCESS);
#endif

        // load param.txt
        input = load_params(argc, argv);

        // 0 = fresh start (default), 1 = restart from latest snapshot
        int restart_flag = (argc > 2) ? std::atoi(argv[2]) : 0;

        StartState state = {0.0, 0};

        if (restart_flag == 1) {
            // find latest snapshot
            std::string out_dir  = input.getParameter("output_directory");
            int         latest_n = InputHandler::findLatestSnapshot(out_dir);
            if (latest_n < 0) {
                std::cerr << "RESTART: Error! No snapshots found in " << out_dir << std::endl;
                exit(EXIT_FAILURE);
            }

            std::string snap_path = out_dir + "snapshot_" + std::to_string(latest_n) + ".hdf5";
            std::cout << "RESTART: Loading snapshot " << snap_path << std::endl;

            // load IC from snapshot
            if (!input.readSnapshotFile(snap_path, icData, state.t_sim)) { exit(EXIT_FAILURE); }
            state.snap_num = latest_n + 1;
        } else {
            // load IC from IC file
            if (!input.readICFile(input.getParameter("ic_file"), icData)) { exit(EXIT_FAILURE); }
        }

        // adapt buffer for periodic bc to resolution
        buff = (1. / pow(icData.seedpos_dims[0], 1. / ((double)DIMENSION))) * 4;

#ifdef MOVING_MESH
        // mesh regularization parameters (compile-time constants from Config.sh)
        std::cout << "BEGRUN: CellShapingSpeed  = " << CellShapingSpeed << std::endl;
        std::cout << "BEGRUN: CellShapingFactor = " << CellShapingFactor << std::endl;
#endif

        // init output folder
        output = OutputHandler(input.getParameter("output_directory"));
        if (!output.initialize()) { exit(EXIT_FAILURE); }

        PROFILE_END("BEGRUN");
        return state;
    }

    // free IC data not needed anymore
    void free_initial_conditions() {
        std::vector<double>().swap(icData.seedpos);
        std::vector<double>().swap(icData.rho);
        std::vector<double>().swap(icData.vel);
        std::vector<double>().swap(icData.Energy);
    }

    // prints Proteus banner
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

    // loads parameters from param.txt
    InputHandler load_params(int argc, char* argv[]) {

        std::string paramFile = "./ics/param.txt";
        if (argc > 1) { paramFile = argv[1]; }

        InputHandler input(paramFile);
        if (!input.loadParameters()) {
            std::cerr << "BEGRUN: Failed to load parameters. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
        return input;
    }

} // namespace begrun
