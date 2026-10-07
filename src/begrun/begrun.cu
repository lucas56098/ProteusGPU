// implements begrun (begrun.h)

#include "../astro/sources.h"
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

    // steps of begrun, in call order
    static void initial_printouts();
    static void restart_from_snapshot(const int latest_snap_n, std::string out_dir);
    static void prepare_sim_struct();
    static void init_units();
    static void load_IC_fields();
    static void init_exch_buffers();
    static void init_hydro_and_mesh();
    static void free_initial_conditions();
#ifdef USE_MPI
    static void restore_decomp_cuts();
#endif

    // sets up everything the main loop needs
    void begrun(int argc, char* argv[]) {

        Profiler::start_total_timer();
        PROFILE("BEGRUN");

        initial_printouts();

        // argv[1] is the parameter file
        if (!input.load_parameters(argc > 1 ? argv[1] : "./ics/param.txt")) {
            mpi::exit_failure("BEGRUN: could not load the parameter file.\n");
        }

        // output directory, created if it is not there yet
        std::string out_dir = input.get_parameter("output_directory");
        output              = OutputHandler(out_dir);
        if (!output.initialize()) { mpi::exit_failure("BEGRUN: output directory setup failed.\n"); }

        // highest snapshot number already in that directory, -1 if there is none
        const int latest_snap_n = InputHandler::find_latest_snapshot(out_dir, mpi::nranks(), mpi::rank());

        // argv[2] == 1 continues from the latest snapshot
        ic_data.header.restart_flag = (argc > 2) && (std::atoi(argv[2]) == 1);
        if (ic_data.header.restart_flag) {

            // count ic_file as read, a restart does not use it
            (void)input.has_parameter("ic_file");
            restart_from_snapshot(latest_snap_n, out_dir);

        } else {

            // fresh run: header now, the cells later
            ic_data.header.ic_filename = input.get_parameter("ic_file");

            uint64_t n_total = 0;
            if (!input.read_ic_header(ic_data.header.ic_filename, ic_data.header, n_total)) {
                mpi::exit_failure("BEGRUN: could not read IC header from %s\n", ic_data.header.ic_filename.c_str());
            }
            ic_data.header.n_global = n_total;

            // never write into a snapshot series that already exists
            if (latest_snap_n > 0) {
                mpi::exit_failure("RESTART: Stopping! Found existing snapshots but no restart-flag.\n");
            }
        }

        prepare_sim_struct();

        init_units();
        astro::sources_init();

        // on a restart the profile log continues the old file
        const std::string profile_path = input.get_parameter("output_directory") + "/profile.hdf5";
        Profiler::open_profile_log(profile_path, ic_data.header.restart_flag ? sim.step : -1);

        mpi::decomp_init();

#ifdef USE_MPI
        // cuts of the snapshot, before any cell is placed
        if (ic_data.header.restart_flag) { restore_decomp_cuts(); }
#endif

        // a restart already got its cells out of the snapshot
        if (!ic_data.header.restart_flag) { load_IC_fields(); };

        init_exch_buffers();

        init_hydro_and_mesh();

        // every parameter this build reads has been read by now
        input.warn_unread_parameters();

        // snapshot at t = 0
        if (sim.snap_num == 0) { output.write_snapshot(); }
    }

    // tears the run down and prints the final numbers
    void endrun() {
        logging::root() << "\nHYDRO: Finished after " << sim.step << " steps at t = " << sim.t_sim << std::endl;

        voronoi::free_mesh(sim.mesh);
        hydro::free_hydro();
        mpi::halo_free();
        mpi::migrate_free();
        mpi::decomp_free();
        sim.mesh = nullptr;

        Profiler::stop_total_timer();
        Profiler::print_results();
        Profiler::close_profile_log();

        print_max_memory_usage();
        logging::root() << "MAIN: Done. (Total runtime = " << Profiler::total_seconds() << " s)" << std::endl;
    }

    // banner, version and what this run is: dimension, backend, ranks, threads, GPUs
    static void initial_printouts() {
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

#ifdef USE_MPI
        out << "BEGRUN: MPI ranks = " << mpi::nranks() << " (" << mpi::node_local_size() << " per node)" << std::endl;

        mpi::report_gpu_aware_mpi();
#endif

#ifdef USE_OPENMP
        out << "BEGRUN: OpenMP threads = " << logging::omp_threads() << " (per rank)" << std::endl;
#endif

#ifndef CPU_DEBUG
        const int n_gpus  = mpi::gpus_per_node();
        const int n_local = mpi::node_local_size();
        out << "BEGRUN: GPUs per node  = " << n_gpus << " (" << n_local << " ranks/node, "
            << (double)n_local / (n_gpus > 0 ? n_gpus : 1) << " ranks/GPU)" << std::endl;
#endif
#ifndef CPU_DEBUG
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "CUDA: rank " << mpi::rank() << " on device " << dev << " (" << prop.name << "), SM " << prop.major
                  << "." << prop.minor << std::endl;
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

// CI build check: stop before the run touches anything
#ifdef DRY_RUN
        out << "Dry run for CI test successful, exiting." << std::endl;
        exit(EXIT_SUCCESS);
#endif
    }

    // code units from the param file (ASTRO_PHYSICS only)
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
    // puts back the cuts the snapshot ran with
    static void restore_decomp_cuts() {
        const std::vector<int64_t>& c    = ic_data.header.decomp_cuts;
        const size_t                want = (size_t)mpi::decomp.nranks + 1;
        if (c.size() != want) {
            mpi::exit_failure(
                "RESTART: Error! snapshot cut table has %zu entries, this run needs %zu.\n", c.size(), want);
        }
        const std::vector<uint64_t> cuts(c.begin(), c.end());
        mpi::decomp_set_cuts(cuts.data());
    }
#endif

    // reads this rank's latest snapshot and continues time, step and snapshot counters
    static void restart_from_snapshot(const int latest_snap_n, std::string out_dir) {

        if (latest_snap_n < 0) {
            mpi::exit_failure("RESTART: Error! No snapshots found in %s (matching this run's rank "
                              "count = %d)\n",
                              out_dir.c_str(),
                              mpi::nranks());
        }

        const std::string suffix    = (mpi::nranks() > 1) ? ("." + std::to_string(mpi::rank()) + ".hdf5") : ".hdf5";
        const std::string snap_path = out_dir + "snapshot_" + std::to_string(latest_snap_n) + suffix;
        logging::root() << "RESTART: Loading snapshot snapshot_" << latest_snap_n
                        << ((mpi::nranks() > 1) ? ".<rank>.hdf5" : ".hdf5") << " from " << out_dir << std::endl;

        SnapshotHeader snap;
        if (!input.read_snapshot_file(snap_path, ic_data, snap)) {
            mpi::exit_failure("RESTART: could not read snapshot %s\n", snap_path.c_str());
        }

        if (snap.nranks != mpi::nranks()) {
            mpi::exit_failure("RESTART: Error! Snapshot was written with %d ranks, but this run has %d. "
                              "Restart requires the same nranks.\n",
                              snap.nranks,
                              mpi::nranks());
        }

        if (snap.rank != mpi::rank()) {
            mpi::exit_failure("RESTART: Error! Snapshot file claims rank %d but this rank is %d "
                              "(filename / rank mismatch).\n",
                              snap.rank,
                              mpi::rank());
        }

        sim.t_sim = snap.t_sim;
        // a snapshot stores the step it was written in, and snapshot_0 was written before the first step
        sim.step                = snap.step + (latest_snap_n > 0 ? 1 : 0);
        sim.snap_num            = latest_snap_n + 1;
        ic_data.header.n_global = snap.n_global;
        sim.n_hydro             = ic_data.header.n_seeds;

        // timers continue instead of starting at zero
        if (!snap.profiler_cum.empty()) Profiler::seed_from_cumulative(snap.profiler_cum);
    }

    // run limits and intervals from the param file
    static void prepare_sim_struct() {

        // time this run starts at, used for the ETA
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

    // reads the cells of the IC file
    static void load_IC_fields() {
#ifdef USE_MPI

        int64_t my_lo = 0, my_hi = 0;
        // every rank reads an equal share of the rows, not the cells it will own
        mpi::decomp_even_split(ic_data.header.n_global, mpi::nranks(), mpi::rank(), &my_lo, &my_hi);
        const uint64_t row_lo  = (uint64_t)my_lo;
        const uint64_t n_local = (uint64_t)(my_hi - my_lo);

        if (!input.read_ic_chunk_parallel(ic_data.header.ic_filename, ic_data, row_lo, n_local)) {
            mpi::exit_failure("BEGRUN: parallel IC read failed for %s\n", ic_data.header.ic_filename.c_str());
        }

        // cut the curve evenly and send every cell to its owner
        mpi::distribute_ic_parallel(ic_data);
#else
        if (!input.read_ic_file(ic_data.header.ic_filename, ic_data)) {
            mpi::exit_failure("BEGRUN: IC read failed for %s\n", ic_data.header.ic_filename.c_str());
        }
#endif
        sim.n_hydro = ic_data.header.n_seeds;
    }

    // headroom of the per-cell arrays and the first ghost slots
    static void init_exch_buffers() {

        // all ranks size their per-cell arrays from the largest rank, with the headroom the user asks for
        mpi::n_local_initial_max = logging::max_global((int)sim.n_hydro);
        mpi::alloc_growth        = input.get_parameter_double("alloc_growth");
        if (!(mpi::alloc_growth >= 1.0)) {
            mpi::exit_failure("BEGRUN: alloc_growth = %g in the param file, it must be at least 1.\n",
                              mpi::alloc_growth);
        }

        mpi::halo_init((int)sim.n_hydro);
    }

    // hydro arrays, mesh allocation and the first mesh build
    static void init_hydro_and_mesh() {

        hydro::init_hydro();

        sim.mesh = voronoi::allocate_mesh(sim.n_hydro);

#ifdef MOVING_MESH
        // v_mesh of a snapshot; the build permutes it into k-order together with the primvars
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

        // first build
        voronoi::compute_periodic_mesh(
            sim.mesh, (POINT_TYPE*)ic_data.pos.data(), sim.n_hydro, sim.primvar, sim.cons, nullptr);

        // the cells live in the mesh now
        free_initial_conditions();

        if (sim.t_sim > 0.0) {
            logging::root() << "HYDRO: restarted from t = " << sim.t_sim << " (snap_num = " << sim.snap_num
                            << ", step = " << sim.step << ", nranks = " << mpi::nranks()
                            << ", n_global = " << ic_data.header.n_global << ")" << std::endl;
        } else {
            logging::root() << "HYDRO: started from IC" << std::endl;
        }
        print_max_memory_usage();
    }

    // releases the IC vectors; swapping with an empty vector also frees the capacity
    void free_initial_conditions() {
        std::vector<double>().swap(ic_data.pos);
        std::vector<double>().swap(ic_data.rho);
        std::vector<double>().swap(ic_data.vel);
        std::vector<double>().swap(ic_data.energy);
#ifdef MOVING_MESH
        std::vector<double>().swap(ic_data.v_mesh);
#endif
    }

} // namespace begrun
