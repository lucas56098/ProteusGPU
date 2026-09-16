/* begrun: calls everything to start or end a run */
#include "../astro/sources.h"
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
    static void init_units();
    static void load_IC_fields();
    static void init_exch_buffers();
    static void init_hydro_and_mesh();
    static void free_initial_conditions();
#ifdef USE_MPI
    static void restore_decomp_splits();
#endif

    // ============================================================
    // setup / end a simulation run
    // ============================================================

    // setup simulation
    void begrun(int argc, char* argv[]) {

        Profiler::start_total_timer();
        PROFILE("BEGRUN");

        initial_printouts();

        // load parameters
        if (!input.load_parameters(argc > 1 ? argv[1] : "./ics/param.txt")) {
            proteus_mpi::exit_failure("BEGRUN: could not load the parameter file.\n");
        }

        // init OutputHandler
        std::string out_dir = input.get_parameter("output_directory");
        output              = OutputHandler(out_dir);
        if (!output.initialize()) { proteus_mpi::exit_failure("BEGRUN: output directory setup failed.\n"); }

        // find latest snap
        const int latest_snap_n =
            InputHandler::find_latest_snapshot(out_dir, proteus_mpi::nranks(), proteus_mpi::rank());

        // do we restart?
        ic_data.header.restart_flag = (argc > 2) && (std::atoi(argv[2]) == 1);
        if (ic_data.header.restart_flag) {

            // restart sim
            (void)input.has_parameter("ic_file");
            restart_from_snapshot(latest_snap_n, out_dir);

        } else {

            // new sim from IC file
            ic_data.header.ic_filename = input.get_parameter("ic_file");

            // read IC header (fields are read after domain decomp)
            uint64_t n_total = 0;
            if (!input.read_ic_header(ic_data.header.ic_filename, ic_data.header, n_total)) {
                proteus_mpi::exit_failure("BEGRUN: could not read IC header from %s\n",
                                          ic_data.header.ic_filename.c_str());
            }
            ic_data.header.n_global = n_total;

            // refuse to silently overwrite existing snapshots
            if (latest_snap_n > 0) {
                proteus_mpi::exit_failure("RESTART: Stopping! Found existing snapshots but no restart-flag.\n");
            }
        }

        // write parameters into sim struct
        prepare_sim_struct();

        // set up the code-unit system + astro source terms (only when astro physics is compiled in)
        init_units();
        astro::sources_init();

        // create profile log
        const std::string profile_path = input.get_parameter("output_directory") + "/profile.hdf5";
        Profiler::open_profile_log(profile_path, ic_data.header.restart_flag ? sim.step : -1);

        // decompose domain
        buff = (1. / pow((double)ic_data.header.n_global, 1. / ((double)DIMENSION))) * 4;
        proteus_mpi::decomp_init(ic_data.header.n_global, buff);

#ifdef USE_MPI
        if (ic_data.header.restart_flag) { restore_decomp_splits(); }
#endif

        // load IC fields into ic_data
        if (!ic_data.header.restart_flag) { load_IC_fields(); };

        // init halo and migrate for exchange
        init_exch_buffers();

        // init hydro from ic_data + build initial voronoi mesh
        init_hydro_and_mesh();

        // every parameter has been read by now
        input.warn_unread_parameters();
    }

    // free everything + print summary
    void endrun() {
        logging::root() << "\nHYDRO: Finished after " << sim.step << " steps at t = " << sim.t_sim << std::endl;

        // free mesh/hydro/halo
        voronoi::free_mesh(sim.mesh);
        hydro::free_hydro();
        proteus_mpi::halo_free();
        sim.mesh = nullptr;

        Profiler::stop_total_timer(); // accumulates the final TOTAL time
        Profiler::print_results();
        Profiler::close_profile_log();

        // peak memory + final runtime
        print_max_memory_usage();
        logging::root() << "MAIN: Done. (Total runtime = " << Profiler::total_seconds() << " s)" << std::endl;
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

    // read the code-unit base factors from param.txt. Without ASTRO_PHYSICS nothing converts
    // against them, so the parameters are not required and units keeps its cgs-identity default.
    static void init_units() {
#ifdef ASTRO_PHYSICS
        units.set_base(input.get_parameter_double("UnitLength_in_cm"),
                       input.get_parameter_double("UnitMass_in_g"),
                       input.get_parameter_double("UnitVelocity_in_cm_per_s"));

        logging::root() << "UNITS: 1 code unit = " << units.UnitLength_in_cm << " cm, " << units.UnitMass_in_g << " g, "
                        << units.UnitVelocity_in_cm_per_s << " cm/s" << std::endl;
#endif
    }

#ifdef USE_MPI
    static void restore_decomp_splits() {
        const auto& dc = proteus_mpi::decomp;
        for (int a = 0; a < 3; a++) {
            const size_t want = (size_t)dc.dims[a] + 1;
            if (ic_data.header.decomp_splits[a].size() != want) {
                proteus_mpi::exit_failure("RESTART: Error! snapshot split table for axis %d has %zu entries, "
                                          "this run's decomposition needs %zu.\n",
                                          a,
                                          ic_data.header.decomp_splits[a].size(),
                                          want);
            }
        }
        proteus_mpi::decomp_apply_splits(ic_data.header.decomp_splits[0].data(),
                                         ic_data.header.decomp_splits[1].data(),
                                         ic_data.header.decomp_splits[2].data());
    }
#endif

    // restarting from snapshots
    static void restart_from_snapshot(const int latest_snap_n, std::string out_dir) {

        // is there a snapshot to restart from?
        if (latest_snap_n < 0) {
            proteus_mpi::exit_failure("RESTART: Error! No snapshots found in %s (matching this run's rank "
                                      "count = %d)\n",
                                      out_dir.c_str(),
                                      proteus_mpi::nranks());
        }

        // snapshot filepath belonging to this rank
        const std::string suffix =
            (proteus_mpi::nranks() > 1) ? ("." + std::to_string(proteus_mpi::rank()) + ".hdf5") : ".hdf5";
        const std::string snap_path = out_dir + "snapshot_" + std::to_string(latest_snap_n) + suffix;
        logging::root() << "RESTART: Loading snapshot snapshot_" << latest_snap_n
                        << ((proteus_mpi::nranks() > 1) ? ".<rank>.hdf5" : ".hdf5") << " from " << out_dir << std::endl;

        // read snapshot
        SnapshotHeader snap;
        if (!input.read_snapshot_file(snap_path, ic_data, snap)) {
            proteus_mpi::exit_failure("RESTART: could not read snapshot %s\n", snap_path.c_str());
        }

        // does nranks equal the snapshot one?
        if (snap.nranks != proteus_mpi::nranks()) {
            proteus_mpi::exit_failure("RESTART: Error! Snapshot was written with %d ranks, but this run has %d. "
                                      "Restart requires the same nranks.\n",
                                      snap.nranks,
                                      proteus_mpi::nranks());
        }

        // does my rank equal the snapshot one?
        if (snap.rank != proteus_mpi::rank()) {
            proteus_mpi::exit_failure("RESTART: Error! Snapshot file claims rank %d but this rank is %d "
                                      "(filename / rank mismatch).\n",
                                      snap.rank,
                                      proteus_mpi::rank());
        }

        // read sim info from snap
        sim.t_sim               = snap.t_sim;
        sim.step                = snap.step + (latest_snap_n > 0 ? 1 : 0);
        sim.snap_num            = latest_snap_n + 1;
        ic_data.header.n_global = snap.n_global;
        // load_IC_fields() is the only other writer of n_hydro and is skipped on restart
        sim.n_hydro = ic_data.header.n_seeds;

        // restore profiler timings from snapshot
        if (!snap.profiler_cum.empty()) Profiler::seed_from_cumulative(snap.profiler_cum);
    }

    // sim parameters from params
    static void prepare_sim_struct() {

        // write parameters into sim struct
        sim.t_start      = sim.t_sim;
        sim.t_end        = input.get_parameter_double("time_end");
        sim.CFL          = input.get_parameter_double("CFL_frac");
        sim.output_dt    = input.get_parameter_double("output_dt");
        sim.t_nextoutput = sim.t_sim + sim.output_dt;
#ifdef USE_MPI
        sim.rebalance_interval     = input.get_parameter_int("rebalance_interval");
        sim.imbalance_log_interval = input.get_parameter_int("imbalance_log_interval");
        sim.imbalance_threshold    = input.get_parameter_double("imbalance_threshold");
#endif
    }

    // load IC seeds and primitive variables into ic_data
    static void load_IC_fields() {
#ifdef USE_MPI

        // evenly split n_global by rank number
        int64_t my_lo = 0, my_hi = 0;
        proteus_mpi::decomp_even_split(
            ic_data.header.n_global, proteus_mpi::nranks(), proteus_mpi::rank(), &my_lo, &my_hi);
        const uint64_t row_lo  = (uint64_t)my_lo;
        const uint64_t n_local = (uint64_t)(my_hi - my_lo);

        // each rank reads part of the IC
        if (!input.read_ic_chunk_parallel(ic_data.header.ic_filename, ic_data, row_lo, n_local)) {
            proteus_mpi::exit_failure("BEGRUN: parallel IC read failed for %s\n", ic_data.header.ic_filename.c_str());
        }

        // exch the seedpoints to the ranks where they belong
        proteus_mpi::distribute_ic_parallel(ic_data, buff);
#else
        // read the whole file
        if (!input.read_ic_file(ic_data.header.ic_filename, ic_data)) {
            proteus_mpi::exit_failure("BEGRUN: IC read failed for %s\n", ic_data.header.ic_filename.c_str());
        }
#endif
        // local cell count
        sim.n_hydro = ic_data.header.n_seeds;
    }

    // halo and migration buffers
    static void init_exch_buffers() {

        // store initial max n_local across ranks
        proteus_mpi::n_local_initial_max = logging::max_global((int)sim.n_hydro);

        // init halo/migrate
        proteus_mpi::halo_init((int)sim.n_hydro, buff);
        proteus_mpi::migrate_init((int)sim.n_hydro);
    }

    // init hydro from IC + built initial voronoi mesh
    static void init_hydro_and_mesh() {

        // primitive variables (from ic_data)
        hydro::init_hydro();

        // allocate mesh
        sim.mesh = voronoi::allocate_mesh(sim.n_hydro);

#ifdef MOVING_MESH
        if (!ic_data.v_mesh.empty()) {
            for (uint64_t i = 0; i < sim.n_hydro; i++) {
                sim.mesh->v_mesh[i].x = ic_data.v_mesh[DIMENSION * i];
                sim.mesh->v_mesh[i].y = ic_data.v_mesh[DIMENSION * i + 1];
#ifdef dim_3D
                sim.mesh->v_mesh[i].z = ic_data.v_mesh[DIMENSION * i + 2];
#endif
            }
        }
#endif

        // initial build
        voronoi::compute_periodic_mesh(
            sim.mesh, (POINT_TYPE*)ic_data.pos.data(), sim.n_hydro, sim.primvar, sim.prim_new, 0.0);

        // IC no longer needed
        free_initial_conditions();

        if (sim.t_sim > 0.0) {
            logging::root() << "HYDRO: restarted from t = " << sim.t_sim << " (snap_num = " << sim.snap_num
                            << ", step = " << sim.step << ", nranks = " << proteus_mpi::nranks()
                            << ", n_global = " << ic_data.header.n_global << ")" << std::endl;
        } else {
            logging::root() << "HYDRO: started from IC" << std::endl;
        }
        print_max_memory_usage();
    }

    // drop the IC arrays once primvar + mesh are built from them
    void free_initial_conditions() {
        std::vector<double>().swap(ic_data.pos);
        std::vector<double>().swap(ic_data.rho);
        std::vector<double>().swap(ic_data.vel);
        std::vector<double>().swap(ic_data.energy);
        std::vector<uint64_t>().swap(ic_data.global_id);
#ifdef MOVING_MESH
        std::vector<double>().swap(ic_data.v_mesh);
#endif
    }

} // namespace begrun
