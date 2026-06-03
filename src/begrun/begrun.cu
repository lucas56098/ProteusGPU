#include "../global/allvars.h"
#include "../hydro/finite_volume_solver.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../mpi/decomp.h"
#include "../mpi/halo.h"
#include "../mpi/migrate.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "begrun.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace begrun {

    // forward declarations
    static void         print_banner();
    static void         log_run_mode();
    static void         init_gpu();
    static InputHandler load_params(int argc, char* argv[]);
    static void         load_initial_conditions(int argc, char* argv[]);
    static void         init_decomposition();
    static void         init_hydro_and_mesh();
    static void         init_run_config();
    static void         free_initial_conditions();

    // restart state populated by load_initial_conditions and consumed by init_decomposition
    static bool s_is_restart       = false;
    static int  s_restart_n_global = 0;

    // ============================================================
    // Main routines
    // ============================================================

    // setup simulation
    void begrun(int argc, char* argv[]) {
        Profiler::StartTimer("TOTAL_RUNTIME");
        sim.wall_start = std::chrono::steady_clock::now();
        Profiler::StartTimer("BEGRUN");

        // initial printouts
        print_banner();
        log_run_mode();

        // GPU init
        init_gpu();

        // load params and IC
        input = load_params(argc, argv);
        load_initial_conditions(argc, argv);

        // decomposition
        init_decomposition();

        // init hydro from IC + built initial voronoi mesh
        init_hydro_and_mesh();

        // sim parameters from params
        init_run_config();

        print_max_memory_usage();
        Profiler::EndTimer("BEGRUN");
    }

    // free everything begrun built + print final summary
    void endrun() {
        logging::root() << "HYDRO: Finished after " << sim.step << " steps at t = " << sim.t_sim << std::endl;

        // free mesh/hydro/halo
        voronoi::free_mesh(sim.mesh);
        hydro::free_hydro();
        proteus_mpi::halo_free();
        sim.mesh = nullptr;

        // wall-clock + peak memory + profiler dump
        const double total_wall_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - sim.wall_start).count();
        logging::root() << "MAIN: Done. (Total runtime = " << total_wall_s << " seconds)" << std::endl;
        print_max_memory_usage();

        Profiler::EndTimer("TOTAL_RUNTIME");
        Profiler::PrintResults();
    }

    // ============================================================
    // Helpers
    // ============================================================

    // print Proteus banner with version + build info
    static void print_banner() {
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

    // log dimension/CPU-or-GPU + parallelization summary
    static void log_run_mode() {
        std::ostream& out = logging::root();
#ifdef dim_2D
#ifdef CPU_DEBUG
        out << "BEGRUN: Running 2D mode on CPU" << std::endl;
#else
        out << "BEGRUN: Running 2D mode on GPU" << std::endl;
#endif
#elif dim_3D
#ifdef CPU_DEBUG
        out << "BEGRUN: Running 3D mode on CPU" << std::endl;
#else
        out << "BEGRUN: Running 3D mode on GPU" << std::endl;
#endif
#endif

#ifdef USE_MPI
        out << "BEGRUN: MPI ranks      = " << proteus_mpi::nranks() << " (" << proteus_mpi::node_local_size()
            << " per node)" << std::endl;
#endif
#ifdef USE_OPENMP
        out << "BEGRUN: OpenMP threads = " << logging::omp_threads() << " (per rank)" << std::endl;
#endif
#ifndef CPU_DEBUG
        const int n_gpus  = proteus_mpi::gpus_per_node();
        const int n_local = proteus_mpi::node_local_size();
        out << "BEGRUN: GPUs per node  = " << n_gpus << " (" << n_local << " ranks/node, "
            << (double)n_local / (n_gpus > 0 ? n_gpus : 1) << " ranks/GPU)" << std::endl;
#endif

#ifdef DRY_RUN // early exit only used for github CI
        out << "Dry run for CI test successful, exiting." << std::endl;
        exit(EXIT_SUCCESS);
#endif
    }

    // set CUDA device limits + identify device
    static void init_gpu() {
#ifndef CPU_DEBUG
        cudaDeviceSetLimit(cudaLimitStackSize, 8192); // (3D slow voronoi kernel needs ~5.7KB stack/thread -> use 8KB)
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        logging::root() << "CUDA: rank 0 on device " << dev << " (" << prop.name << "), SM " << prop.major << "."
                        << prop.minor << std::endl;
#endif
    }

    // load param.txt
    static InputHandler load_params(int argc, char* argv[]) {
        std::string paramFile = "./ics/param.txt";
        if (argc > 1) { paramFile = argv[1]; } // use default or otherwise argv[1]

        InputHandler input(paramFile);
        if (!input.loadParameters()) {
            std::cerr << "BEGRUN: Failed to load parameters. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
        return input;
    }

    // load initial conditions from IC file or (if restart) latest snapshot
    static void load_initial_conditions(int argc, char* argv[]) {

        // is restarting from snapshots specified?
        const int restart_flag = (argc > 2) ? std::atoi(argv[2]) : 0;

        sim.t_sim    = 0.0;
        sim.snap_num = 0;
        sim.step     = 0;

        const int         my_nranks = proteus_mpi::nranks();
        const int         my_rank   = proteus_mpi::rank();
        const std::string out_dir   = input.getParameter("output_directory");
        const int         latest_n  = InputHandler::findLatestSnapshot(out_dir, my_nranks, my_rank);

        if (restart_flag == 1) {
            // restarting from snapshots

            if (latest_n < 0) {
                std::cerr << "RESTART: Error! No snapshots found in " << out_dir
                          << " (matching this run's rank count = " << my_nranks << ")" << std::endl;
                exit(EXIT_FAILURE);
            }

            // each rank reads its own per-rank file under multi-rank, plain filename under single-rank
            const std::string suffix    = (my_nranks > 1) ? ("." + std::to_string(my_rank) + ".hdf5") : ".hdf5";
            const std::string snap_path = out_dir + "snapshot_" + std::to_string(latest_n) + suffix;
            logging::root() << "RESTART: Loading snapshot snapshot_" << latest_n
                            << ((my_nranks > 1) ? ".<rank>.hdf5" : ".hdf5") << " from " << out_dir << std::endl;

            // read snapshot
            SnapshotHeader snap;
            if (!input.readSnapshotFile(snap_path, icData, snap)) { exit(EXIT_FAILURE); }

            // validate topology matches this run
            if (snap.nranks != my_nranks) {
                std::cerr << "RESTART: Error! Snapshot was written with " << snap.nranks << " ranks, but this run has "
                          << my_nranks << ". Restart requires the same nranks." << std::endl;
                exit(EXIT_FAILURE);
            }
            if (snap.rank != my_rank) {
                std::cerr << "RESTART: Error! Snapshot file claims rank " << snap.rank << " but this rank is "
                          << my_rank << " (filename / rank mismatch)." << std::endl;
                exit(EXIT_FAILURE);
            }

            // hand off to init_decomposition / log line
            sim.t_sim          = snap.t_sim;
            sim.step           = snap.step;
            sim.snap_num       = latest_n + 1;
            s_is_restart       = true;
            s_restart_n_global = snap.n_global;
        } else {
            // no restart: read IC file
            if (!input.readICFile(input.getParameter("ic_file"), icData)) { exit(EXIT_FAILURE); }

            // refuse to silently overwrite an existing snapshot series
            if (latest_n > 0) {
                std::cerr << "RESTART: Stopping! Found existing snapshots in " << out_dir << " but no restart-flag."
                          << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // setup decomposition
    static void init_decomposition() {

        // global cell count: fresh start has the full IC on every rank; restart has per-rank slices,
        // so we take n_global from the snapshot header instead of icData.pos_dims[0].
        const size_t n_global = s_is_restart ? (size_t)s_restart_n_global : icData.pos_dims[0];

        // periodic ghost band thickness scales with mean inter-particle spacing
        buff = (1. / pow(n_global, 1. / ((double)DIMENSION))) * 4;

        // domain decomposition (same nranks -> same Cart layout as the snapshot was written with)
        proteus_mpi::decomp_init((int)n_global, buff);

        if (!s_is_restart) {
            // fresh start: filter the global IC down to this rank's brick
            proteus_mpi::distribute_ic_local(icData, buff);
        }
        // restart: each rank's icData is already its own partition, skip filtering

        // local cell count owned by this rank
        sim.n_hydro = icData.pos_dims[0];

        // sanity check: sum of per-rank local counts must equal n_global (catches a missing
        // or truncated rank file). Same pattern as distribute_ic_local's conservation check.
        if (s_is_restart) {
            const int n_global_kept = logging::sum_global((int)sim.n_hydro);
            if (n_global_kept != s_restart_n_global) {
                std::cerr << "RESTART: FATAL cell-count mismatch — sum(per-rank n_local) = " << n_global_kept
                          << ", expected " << s_restart_n_global << " (from snapshot header)." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // halo + migration buffers sized from local count
        proteus_mpi::halo_init((int)sim.n_hydro, buff);
        proteus_mpi::migrate_init((int)sim.n_hydro);
    }

    // init hydro from IC + built initial voronoi mesh
    static void init_hydro_and_mesh() {

        // create output directory if missing
        output = OutputHandler(input.getParameter("output_directory"));
        if (!output.initialize()) { exit(EXIT_FAILURE); }

        // primitive variables (from IC)
        hydro::init_hydro();

        // initial Voronoi mesh from the seed positions
        sim.mesh = voronoi::allocate_mesh(sim.n_hydro);
        voronoi::compute_periodic_mesh(
            sim.mesh, (POINT_TYPE*)icData.pos.data(), sim.n_hydro, sim.primvar, sim.prim_new);

        // IC no longer needed
        free_initial_conditions();

        if (sim.t_sim > 0.0) {
            logging::root() << "HYDRO: restarted from t = " << sim.t_sim << " (snap_num = " << sim.snap_num
                            << ", step = " << sim.step << ", nranks = " << proteus_mpi::nranks()
                            << ", n_global = " << s_restart_n_global << ")" << std::endl;
        } else {
            logging::root() << "HYDRO: started" << std::endl;
        }
    }

    // sim parameters from params
    static void init_run_config() {

        sim.t_start      = sim.t_sim;
        sim.t_end        = input.getParameterDouble("time_end");
        sim.CFL          = input.getParameterDouble("CFL_frac");
        sim.output_dt    = input.getParameterDouble("output_dt");
        sim.t_nextoutput = sim.t_sim + sim.output_dt;
        // sim.step is set in load_initial_conditions (0 for fresh runs, snapshot value on restart)

        // per-timestep profile log; on restart, trim past the snapshot's step and seed counters
        const std::string profile_path = input.getParameter("output_directory") + "/profile.txt";
        if (s_is_restart) { Profiler::ResumeFromLog(profile_path, sim.step); }
        sim.profile_log = logging::FileLogger(profile_path);
    }

    // drop the IC arrays once primvar + mesh are built from them
    void free_initial_conditions() {
        std::vector<double>().swap(icData.pos);
        std::vector<double>().swap(icData.rho);
        std::vector<double>().swap(icData.vel);
        std::vector<double>().swap(icData.energy);
        std::vector<uint64_t>().swap(icData.global_id);
        std::vector<hsize_t>().swap(icData.pos_dims);
    }

} // namespace begrun
