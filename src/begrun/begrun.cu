#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../mpi/decomp.h"
#include "../mpi/halo.h"
#include "../mpi/migrate.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "begrun.h"
#include <cmath>
#include <cstdlib>
#include <iostream>


namespace begrun {

    // setup: banner, CUDA init, params, IC, output dir
    StartState begrun(int argc, char* argv[]) {
        PROFILE_START("BEGRUN");

        // welcome messages
        print_banner();
#ifdef dim_2D
#ifdef CPU_DEBUG
        logging::root() << "BEGRUN: Running 2D mode on CPU" << std::endl;
#else
        logging::root() << "BEGRUN: Running 2D mode on GPU" << std::endl;
#endif
#elif dim_3D
#ifdef CPU_DEBUG
        logging::root() << "BEGRUN: Running 3D mode on CPU" << std::endl;
#else
        logging::root() << "BEGRUN: Running 3D mode on GPU" << std::endl;
#endif
#endif

        // parallelization summary: MPI, OpenMP, and GPUs
#ifdef USE_MPI
        logging::root() << "BEGRUN: MPI ranks      = " << proteus_mpi::nranks() << " (" << proteus_mpi::node_local_size()
                        << " per node)" << std::endl;
#endif
#ifdef USE_OPENMP
        logging::root() << "BEGRUN: OpenMP threads = " << logging::omp_threads() << " (per rank)" << std::endl;
#endif
#ifndef CPU_DEBUG
        const int n_gpus  = proteus_mpi::gpus_per_node();
        const int n_local = proteus_mpi::node_local_size();
        logging::root() << "BEGRUN: GPUs per node  = " << n_gpus << " (" << n_local << " ranks/node, "
                        << (double)n_local / (n_gpus > 0 ? n_gpus : 1) << " ranks/GPU)" << std::endl;
#endif

#ifdef DRY_RUN
        logging::root() << "Dry run for CI test successful, exiting." << std::endl;
        exit(EXIT_SUCCESS);
#endif

#ifndef CPU_DEBUG
        // init GPU
        cudaDeviceSetLimit(cudaLimitStackSize, 8192); // (3D slow voronoi kernel needs ~5.7KB stack/thread -> use 8KB)
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        logging::root() << "CUDA: rank 0 on device " << dev << " (" << prop.name << "), SM " << prop.major << "."
                        << prop.minor << std::endl;
#endif

        // parse the parameter file
        input = load_params(argc, argv);

        // argv[2]: 0 (or absent) = fresh start; 1 = restart from latest snapshot
        int restart_flag = (argc > 2) ? std::atoi(argv[2]) : 0;

        StartState state = {0.0, 0};

        // identify existing snapshots in the output directory
        std::string out_dir  = input.getParameter("output_directory");
        int         latest_n = InputHandler::findLatestSnapshot(out_dir);

        // todo: could simply always (assume) restart is desired if latest_n > 0, and drop need for restart_flag
        if (restart_flag == 1) {
            if (latest_n < 0) {
                std::cerr << "RESTART: Error! No snapshots found in " << out_dir << std::endl;
                exit(EXIT_FAILURE);
            }

            std::string snap_path = out_dir + "snapshot_" + std::to_string(latest_n) + ".hdf5";
            logging::root() << "RESTART: Loading snapshot " << snap_path << std::endl;

            if (!input.readSnapshotFile(snap_path, icData, state.t_sim)) { exit(EXIT_FAILURE); }
            state.snap_num = latest_n + 1;
        } else {
            // fresh IC from the file in param.txt
            if (!input.readICFile(input.getParameter("ic_file"), icData)) { exit(EXIT_FAILURE); }

            // refuse to silently overwrite an existing snapshot series
            if (latest_n > 0) {
                std::cerr << "RESTART: Stopping! Found existing snapshots in " << out_dir << " but no restart-flag." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // periodic ghost band thickness scales with mean inter-particle spacing
        buff = (1. / pow(icData.pos_dims[0], 1. / ((double)DIMENSION))) * 4;

        // domain decomposition, halo buffers, migration headroom
        proteus_mpi::decomp_init((int)icData.pos_dims[0], buff);
        proteus_mpi::distribute_ic_local(icData, buff);
        proteus_mpi::halo_init((int)icData.pos_dims[0], buff);
        proteus_mpi::migrate_init((int)icData.pos_dims[0]);

        // create output directory if missing
        output = OutputHandler(input.getParameter("output_directory"));
        if (!output.initialize()) { exit(EXIT_FAILURE); }

        PROFILE_END("BEGRUN");
        return state;
    }

    // free IC data not needed anymore
    void free_initial_conditions() {
        std::vector<double>().swap(icData.pos);
        std::vector<double>().swap(icData.rho);
        std::vector<double>().swap(icData.vel);
        std::vector<double>().swap(icData.energy);
        std::vector<uint64_t>().swap(icData.global_id);
    }

    // prints Proteus banner
    void print_banner() {
        std::ostream& out = logging::root();
        out << "==========================================================================" << std::endl;
        out << R"(
          _____           _                    _____ _____  _    _
         |  __ \         | |                  / ____|  __ \| |  | |
         | |__) | __ ___ | |_ ___ _   _ ___  | |  __| |__) | |  | |
         |  ___/ '__/ _ \| __/ _ \ | | / __| | | |_ |  ___/| |  | |
         | |   | | | (_) | ||  __/ |_| \__ \ | |__| | |    | |__| |
         |_|   |_|  \___/ \__\___|\__,_|___/  \_____|_|     \____/

    )" << std::endl;
        out << "       GPU-accelerated moving mesh hydrodynamics for astrophysics" << std::endl;
        out << "==========================================================================" << std::endl;
        #ifndef GIT_COMMIT
        out << "Version: 0.8" << std::endl;
        #else
        #ifdef GIT_DIFFSTAT
        out << "Version: 0.8 (commit " << GIT_COMMIT << ", " << GIT_DIFFSTAT << ")" << std::endl;
        #else
        out << "Version: 0.8 (commit " << GIT_COMMIT << ")" << std::endl;
        #endif
        #endif
        out << "Build date: " << __DATE__ << " " << __TIME__ << std::endl;
        out << "Authors: Lucas Schleuss, Dylan Nelson" << std::endl;
        out << "Institution: Institute of Theoretical Astrophysics, Heidelberg University" << std::endl;
        out << "==========================================================================" << std::endl;
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
