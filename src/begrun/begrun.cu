// implements starting and finishing a run (begrun.h)

#include "begrun/begrun.h"

#include "config/config.h"
#include "device/device.h"
#include "io/io.h"
#include "logging/logging.h"
#include "mpi/mpi.h"
#include "timers/timers.h"

#include <chrono>
#include <string>

namespace begrun {

    namespace {

        constexpr double START_TIME = 0.0;

        // where a run starts: its time, the top steps done so far, and the next snapshot
        struct Start {
            double                time;
            int                   step;
            simulation::Snapshots snapshots;
        };

        void print_hello();
        Start read_cells(const params::RunParameters& parameters, cells::Cells& cells);
        double reference_radius_of(const params::RunParameters& parameters, const cells::Cells& cells);

    } // namespace

    void begrun(int argc, char* argv[], simulation::Simulation& sim) {
        sim.start_of_run = std::chrono::steady_clock::now();
        print_hello();
        sim.parameters = params::read_parameters(argc, argv);
        device::init(mpi::node_rank(), mpi::node_size(), sim.parameters.memory_limit_in_gb);

        const params::RunParameters& parameters = sim.parameters;
        const Start                  start      = read_cells(parameters, sim.cells);
        sim.sources.set_up(parameters.astro);
        logging::print("HYDRO: Initialized hydro for %lld particles",
                       mpi::sum_over_ranks((long long)sim.cells.n_cells));

        {
            const timers::Scope timer("CUTS", timers::Type::communication);
            sim.domain.cut_by_number_of_cells(sim.cells.position.data(), sim.cells.n_cells);
        }
        {
            const timers::Scope timer("MIGRATE", timers::Type::communication);
            cells::move_to_owners(sim.cells, sim.domain);
        }
        if (mpi::nranks() > 1) {
            logging::print_labelled("DECOMP",
                                    "Distributed cells onto %d ranks. Imbalance %.2f",
                                    mpi::nranks(),
                                    domain::imbalance(sim.cells.n_cells));
        }
        {
            const timers::Scope timer("ORDER");
            cells::sort_along_curve(sim.cells);
        }

        {
            const timers::Scope timer("BUILD");
            sim.mesh.build(sim.cells.position.data(), sim.cells.n_cells, sim.domain, sim.ghosts);
            cells::make_room_for(sim.cells, sim.cells.n_cells, sim.mesh.pointers().n_ghost_cells);
        }
        sim.mesh.print_builds(sim.ghosts);
        if constexpr (config::regularization_speed_by_volume > 0.0) {
            sim.mesh.set_reference_radius(reference_radius_of(parameters, sim.cells));
        }
        if (parameters.from_last_snapshot) {
            logging::print("HYDRO: restarted from t = %g (snap_num = %d, step = %d, nranks = %d, n_global = %lld)",
                           start.time,
                           start.snapshots.number,
                           start.step,
                           mpi::nranks(),
                           mpi::sum_over_ranks((long long)sim.cells.n_cells));
        } else {
            logging::print("HYDRO: started from IC");
        }
        device::print_max_memory_usage();

        sim.sync = timesteps::first_sync(start.time);
        if (!parameters.from_last_snapshot) {
            const timers::Scope timer("IO_SNAPSHOT");
            io::prepare_output_directory(parameters.output_directory);
            io::write_snapshot(parameters.output_directory,
                               0,
                               start.time,
                               0,
                               parameters.astro.units,
                               sim.cells,
                               sim.mesh.pointers(),
                               sim.ghosts.pointers(),
                               sim.top.pointers(sim.sync, sim.cells.steps));
        }

        sim.snapshots  = start.snapshots;
        sim.step       = start.step;
        sim.start_time = start.time;
        sim.profile =
            parameters.from_last_snapshot ? "profile_" + std::to_string(start.step) + ".hdf5" : "profile.hdf5";
    }

    void endrun(const simulation::Simulation& sim) {
        timers::print_summary();
        device::print_max_memory_usage();
        logging::print("MAIN: Done. (Total runtime = %g s)",
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - sim.start_of_run).count());
    }

    namespace {

        // what this code is and what this run runs on
        void print_hello() {
            const char* const LINE = "==========================================================================";
            logging::print("%s", LINE);
            logging::print("%s", R"(
          _____           _                    _____ _____  _    _
         |  __ \         | |                  / ____|  __ \| |  | |
         | |__) | __ ___ | |_ ___ _   _ ___  | |  __| |__) | |  | |
         |  ___/ '__/ _ \| __/ _ \ | | / __| | | |_ |  ___/| |  | |
         | |   | | | (_) | ||  __/ |_| \__ \ | |__| | |    | |__| |
         |_|   |_|  \___/ \__\___|\__,_|___/  \_____|_|     \____/
)");
            logging::print("       GPU-accelerated moving mesh hydrodynamics for astrophysics");
            logging::print("%s", LINE);
#ifdef GIT_DIFFSTAT
            logging::print("Version: 0.8 (commit %s, %s)", GIT_COMMIT, GIT_DIFFSTAT);
#else
            logging::print("Version: 0.8 (commit %s)", GIT_COMMIT);
#endif
            logging::print("Build date: %s %s", __DATE__, __TIME__);
            logging::print("Authors: Lucas Schleuss (MPA Garching), Dylan Nelson (ITA Heidelberg)");
            logging::print("%s", LINE);
            logging::print("\nBEGRUN: Running %dD mode on %s", config::dim, config::use_gpu ? "GPU" : "CPU");
            if constexpr (config::use_mpi) {
                logging::print("BEGRUN: MPI ranks = %d (%d per node)", mpi::nranks(), mpi::node_size());
                for (const std::string& line : mpi::about_library())
                    logging::print("BEGRUN: %s", line.c_str());
            }
            if constexpr (config::use_openmp) {
                logging::print("BEGRUN: OpenMP threads = %d (per rank)", device::n_threads());
            }
        }

        // reads the initial conditions, or on a restart the last snapshot; the next one is due output_dt after it
        Start read_cells(const params::RunParameters& parameters, cells::Cells& cells) {
            const timers::Scope timer("READ");
            if (!parameters.from_last_snapshot) {
                io::read_initial_conditions(parameters.ic_file.c_str(), cells);
                return Start{START_TIME, 0, simulation::Snapshots{1, START_TIME + parameters.output_dt}};
            }
            const int last = io::last_snapshot_number(parameters.output_directory);
            if (last < 0) {
                logging::stop_run("RESTART: no snapshot in %s to go on from", parameters.output_directory.c_str());
            }
            const io::SnapshotHeader header = io::read_snapshot(parameters.output_directory, last, cells);
            return Start{header.time, header.step, simulation::Snapshots{last + 1, header.time + parameters.output_dt}};
        }

        // radius of a cell of vol_ref_cell_size, without it of the mean cell of the box; collective
        double reference_radius_of(const params::RunParameters& parameters, const cells::Cells& cells) {
            const double size = parameters.reference_cell_size;
            if (size > 0.0) return radius_of_ball((config::dim == 2) ? size * size : size * size * size);
            return radius_of_ball(1.0 / (double)mpi::sum_over_ranks((long long)cells.n_cells));
        }

    } // namespace

} // namespace begrun
