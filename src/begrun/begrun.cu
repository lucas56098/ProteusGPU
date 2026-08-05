/* begrun: calls everything to start or end a run */
#include "../global/allvars.h"
#include "../hydro/finite_volume_solver.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../mpi/decomp.h"
#include "../mpi/halo.h"
#include "../mpi/migrate.h"
#include "../mpi/mpi_compat.h"
#include "../mpi/rebalance.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "begrun.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace begrun {

    // forward declarations
    static void initial_printouts();
    static void restart_from_snapshot(const int latest_snap_n, std::string out_dir);
    static void prepare_sim_struct();
    static void load_IC_fields();
    static void init_exch_buffers();
    static void init_hydro_and_mesh();
    static void free_initial_conditions();

    // ============================================================
    // setup / end a simulation run
    // ============================================================

    // setup simulation
    void begrun(int argc, char* argv[]) {

        Profiler::StartTotalTimer();
        PROFILE("BEGRUN");

        initial_printouts();

        // load parameters
        input.loadParameters(argc > 1 ? argv[1] : "./ics/param.txt");

        // init OutputHandler
        std::string out_dir = input.getParameter("output_directory");
        output              = OutputHandler(out_dir);
        if (!output.initialize()) { exit(EXIT_FAILURE); }

        // find latest snap
        const int latest_snap_n = InputHandler::findLatestSnapshot(out_dir, proteus_mpi::nranks(), proteus_mpi::rank());

        // do we restart?
        icData.header.restart_flag = (argc > 2) && (std::atoi(argv[2]) == 1);
        if (icData.header.restart_flag) {

            // restart sim
            restart_from_snapshot(latest_snap_n, out_dir);

        } else {

            // new sim from IC file
            icData.header.ic_filename = input.getParameter("ic_file");

            // read IC header (fields are read after domain decomp)
            hsize_t n_total = 0;
            if (!input.readICHeader(icData.header.ic_filename, icData.header, n_total)) { exit(EXIT_FAILURE); }
            icData.header.n_global = n_total;

            // refuse to silently overwrite existing snapshots
            if (latest_snap_n > 0) {
                std::cerr << "RESTART: Stopping! Found existing snapshots but no restart-flag." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // write parameters into sim struct
        prepare_sim_struct();

        // create profile log
        const std::string profile_path = input.getParameter("output_directory") + "/profile.hdf5";
        Profiler::OpenProfileLog(profile_path, icData.header.restart_flag ? sim.step : -1);

        // decompose domain
        buff = (1. / pow((double)icData.header.n_global, 1. / ((double)DIMENSION))) * 4;
        proteus_mpi::decomp_init(icData.header.n_global, buff);

        // load IC fields into icData
        if (!icData.header.restart_flag) { load_IC_fields(); };

        // init halo and migrate for exchange
        init_exch_buffers();

        // init hydro from icData + build initial voronoi mesh
        init_hydro_and_mesh();
    }

    // free everything + print summary
    void endrun() {
        logging::root() << "\nHYDRO: Finished after " << sim.step << " steps at t = " << sim.t_sim << std::endl;

        // free mesh/hydro/halo
        voronoi::free_mesh(sim.mesh);
        hydro::free_hydro();
        proteus_mpi::halo_free();
        sim.mesh = nullptr;

        Profiler::StopTotalTimer(); // accumulates the final TOTAL time
        Profiler::PrintResults();
        Profiler::CloseProfileLog();

        // peak memory + final runtime
        print_max_memory_usage();
        logging::root() << "MAIN: Done. (Total runtime = " << Profiler::TotalSeconds() << " s)" << std::endl;
    }

    // ============================================================
    // helpers
    // ============================================================

    // print banner + build info
    static void initial_printouts() {
        std::ostream& out = logging::root();

        // welcome banner
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

        // version
        out << "Version: 0.8";
#if defined(GIT_DIFFSTAT) && defined(GIT_COMMIT)
        out << " (commit " << GIT_COMMIT << ", " << GIT_DIFFSTAT << ")";
#elif defined(GIT_COMMIT)
        out << " (commit " << GIT_COMMIT << ")";
#endif
        out << "\nBuild date: " << __DATE__ << " " << __TIME__ << std::endl;
        out << "Authors: Lucas Schleuss, Dylan Nelson" << std::endl;
        out << "Institution: Institute of Theoretical Astrophysics, Heidelberg University" << std::endl;
        out << "==========================================================================" << std::endl;
        out << "BEGRUN: Running " << DIMENSION << "D mode on " << RUN_MODE << std::endl;

        // MPI info
#ifdef USE_MPI
        out << "BEGRUN: MPI ranks = " << proteus_mpi::nranks() << " (" << proteus_mpi::node_local_size() << " per node)"
            << std::endl;

        proteus_mpi::report_gpu_aware_mpi();
#endif

        // OpenMP info
#ifdef USE_OPENMP
        out << "BEGRUN: OpenMP threads = " << logging::omp_threads() << " (per rank)" << std::endl;
#endif

        // GPU info
#ifndef CPU_DEBUG
        const int n_gpus  = proteus_mpi::gpus_per_node();
        const int n_local = proteus_mpi::node_local_size();
        out << "BEGRUN: GPUs per node  = " << n_gpus << " (" << n_local << " ranks/node, "
            << (double)n_local / (n_gpus > 0 ? n_gpus : 1) << " ranks/GPU)" << std::endl;
#endif
#ifndef CPU_DEBUG
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "CUDA: rank " << proteus_mpi::rank() << " on device " << dev << " (" << prop.name << "), SM "
                  << prop.major << "." << prop.minor << std::endl;
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

        // early exit (used only in github CI)
#ifdef DRY_RUN
        out << "Dry run for CI test successful, exiting." << std::endl;
        exit(EXIT_SUCCESS);
#endif
    }

    // restarting from snapshots
    static void restart_from_snapshot(const int latest_snap_n, std::string out_dir) {

        // is there a snapshot to restart from?
        if (latest_snap_n < 0) {
            std::cerr << "RESTART: Error! No snapshots found in " << out_dir
                      << " (matching this run's rank count = " << proteus_mpi::nranks() << ")" << std::endl;
            exit(EXIT_FAILURE);
        }

        // snapshot filepath belonging to this rank
        const std::string suffix =
            (proteus_mpi::nranks() > 1) ? ("." + std::to_string(proteus_mpi::rank()) + ".hdf5") : ".hdf5";
        const std::string snap_path = out_dir + "snapshot_" + std::to_string(latest_snap_n) + suffix;
        logging::root() << "RESTART: Loading snapshot snapshot_" << latest_snap_n
                        << ((proteus_mpi::nranks() > 1) ? ".<rank>.hdf5" : ".hdf5") << " from " << out_dir << std::endl;

        // read snapshot
        SnapshotHeader snap;
        if (!input.readSnapshotFile(snap_path, icData, snap)) { exit(EXIT_FAILURE); }

        // does nranks equal the snapshot one?
        if (snap.nranks != proteus_mpi::nranks()) {
            std::cerr << "RESTART: Error! Snapshot was written with " << snap.nranks << " ranks, but this run has "
                      << proteus_mpi::nranks() << ". Restart requires the same nranks." << std::endl;
            exit(EXIT_FAILURE);
        }

        // does my rank equal the snapshot one?
        if (snap.rank != proteus_mpi::rank()) {
            std::cerr << "RESTART: Error! Snapshot file claims rank " << snap.rank << " but this rank is "
                      << proteus_mpi::rank() << " (filename / rank mismatch)." << std::endl;
            exit(EXIT_FAILURE);
        }

        // read sim info from snap
        sim.t_sim              = snap.t_sim;
        sim.step               = snap.step;
        sim.snap_num           = latest_snap_n + 1;
        icData.header.n_global = snap.n_global;
        // load_IC_fields() is the only other writer of n_hydro and is skipped on restart
        sim.n_hydro = icData.header.n_seeds;

        // restore profiler timings from snapshot
        if (!snap.profiler_cum.empty()) Profiler::SeedFromCumulative(snap.profiler_cum);
    }

    // sim parameters from params
    static void prepare_sim_struct() {

        // write parameters into sim struct
        sim.t_start      = sim.t_sim;
        sim.t_end        = input.getParameterDouble("time_end");
        sim.CFL          = input.getParameterDouble("CFL_frac");
        sim.output_dt    = input.getParameterDouble("output_dt");
        sim.t_nextoutput = sim.t_sim + sim.output_dt;
#ifdef USE_MPI
        sim.rebalance_interval     = (int)input.getParameterDouble("rebalance_interval");
        sim.imbalance_log_interval = (int)input.getParameterDouble("imbalance_log_interval");
        sim.imbalance_threshold    = input.getParameterDouble("imbalance_threshold");
#endif
    }

    // load IC seeds and primitive variables into icData
    static void load_IC_fields() {
#ifdef USE_MPI

        // evenly split n_global by rank number
        int64_t my_lo = 0, my_hi = 0;
        proteus_mpi::decomp_even_split(
            icData.header.n_global, proteus_mpi::nranks(), proteus_mpi::rank(), &my_lo, &my_hi);
        const hsize_t row_lo  = (hsize_t)my_lo;
        const hsize_t n_local = (hsize_t)(my_hi - my_lo);

        // each rank reads part of the IC
        if (!input.readICChunkParallel(icData.header.ic_filename, icData, row_lo, n_local)) { exit(EXIT_FAILURE); }

        // exch the seedpoints to the ranks where they belong
        proteus_mpi::distribute_ic_parallel(icData, buff);
#else
        // read the whole file
        if (!input.readICFile(icData.header.ic_filename, icData)) { exit(EXIT_FAILURE); }
#endif
        // local cell count
        sim.n_hydro = icData.header.n_seeds;

        // sanity check: sum of per-rank local counts must equal n_global
        if (icData.header.restart_flag) {
            const long long n_global_kept = logging::sum_global((long long)sim.n_hydro);
            if (n_global_kept != (long long)icData.header.n_global) {
                std::cerr << "RESTART: FATAL cell-count mismatch — sum(per-rank n_local) = " << n_global_kept
                          << ", expected " << icData.header.n_global << " (from snapshot header)." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // init halo and migrate for exchange
    static void init_exch_buffers() {

        // store initial max n_local across ranks
        proteus_mpi::n_local_initial_max = logging::max_global((int)sim.n_hydro);

        // init halo/migrate
        proteus_mpi::halo_init((int)sim.n_hydro, buff);
        proteus_mpi::migrate_init((int)sim.n_hydro);
    }

    // init hydro from IC + built initial voronoi mesh
    static void init_hydro_and_mesh() {

        // primitive variables (from icData)
        hydro::init_hydro();

        // allocate mesh
        sim.mesh = voronoi::allocate_mesh(sim.n_hydro);

        // initial build
        voronoi::compute_periodic_mesh(
            sim.mesh, (POINT_TYPE*)icData.pos.data(), sim.n_hydro, sim.primvar, sim.prim_new, 0.0);

        // IC no longer needed
        free_initial_conditions();

        if (sim.t_sim > 0.0) {
            logging::root() << "HYDRO: restarted from t = " << sim.t_sim << " (snap_num = " << sim.snap_num
                            << ", step = " << sim.step << ", nranks = " << proteus_mpi::nranks()
                            << ", n_global = " << icData.header.n_global << ")" << std::endl;
        } else {
            logging::root() << "HYDRO: started from IC" << std::endl;
        }
        print_max_memory_usage();
    }

    // drop the IC arrays once primvar + mesh are built from them
    void free_initial_conditions() {
        std::vector<double>().swap(icData.pos);
        std::vector<double>().swap(icData.rho);
        std::vector<double>().swap(icData.vel);
        std::vector<double>().swap(icData.energy);
        std::vector<uint64_t>().swap(icData.global_id);
    }

} // namespace begrun
