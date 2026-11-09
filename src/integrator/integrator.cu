// implements the algorithm of a run (integrator.h)

#include "integrator/integrator.h"

#include "astro/astro.h"
#include "begrun/begrun.h"
#include "cells/cells.h"
#include "device/device.h"
#include "domain/domain.h"
#include "ghosts/ghosts.h"
#include "hydro/hydro.h"
#include "io/io.h"
#include "logging/logging.h"
#include "mesh/mesh.h"
#include "mesh_motion/mesh_motion.h"
#include "mpi/mpi.h"
#include "signals/signals.h"
#include "simulation/simulation.h"
#include "timers/timers.h"
#include "timesteps/timesteps.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

namespace integrator {

    namespace {

        // the cells a sync works on
        struct CellsOfSync {
            mesh::CellList active;
            mesh::CellList with_faces;
        };

        using Clock = std::chrono::steady_clock;

        constexpr const char* LINE_BETWEEN_TOP_STEPS =
            "==========================================================================";

        // what the log keeps from one sync to the next
        struct LogOfRun {
            Clock::time_point   start_of_run;
            double              start_time;
            double              time_end;
            timesteps::BinRange bins_that_begin; // of the active cells at the last START
            timesteps::BinRange bins_that_end; // of the active cells at the last TIME
        };

        void begin_steps(simulation::Simulation& sim, const CellsOfSync& of_sync);
        void find_cfl_steps(const params::RunParameters& parameters,
                            const mesh::CellList&        list,
                            const mesh::Mesh&            mesh,
                            const astro::Sources&        sources,
                            double                       longest_step_that_matters,
                            signals::Signals&            signals,
                            cells::Cells&                cells);
        mesh::CellList go_to_next_sync(simulation::Simulation& sim);
        void end_steps_longer_than_their_cfl_step(const params::RunParameters& parameters,
                                                  const mesh::Mesh&            mesh,
                                                  ghosts::Ghosts&              ghosts,
                                                  const astro::Sources&        sources,
                                                  signals::Signals&            signals,
                                                  const timesteps::Sync&       sync,
                                                  timesteps::TopStep&          top,
                                                  cells::Cells&                cells);
        void move_mesh(simulation::Simulation& sim, mesh::CellList& active, device::Array<Vec>& seeds);
        double cut_again_if_unbalanced(domain::Domain& domain, cells::Cells& cells);
        double imbalance_of_ranks(const cells::Cells& cells);
        mesh::CellList build_around_active_cells(const timesteps::Sync& sync,
                                                 timesteps::TopStep&    top,
                                                 const domain::Domain&  domain,
                                                 ghosts::Ghosts&        ghosts,
                                                 mesh::Mesh&            mesh,
                                                 const mesh::CellList&  ending,
                                                 device::Array<Vec>&    seeds,
                                                 cells::Cells&          cells);
        void build_cells_of_list(const mesh::CellList&     list,
                                 const timesteps::Sync&    sync,
                                 const timesteps::TopStep& top,
                                 double                    drift_margin,
                                 const domain::Domain&     domain,
                                 ghosts::Ghosts&           ghosts,
                                 mesh::Mesh&               mesh,
                                 device::Array<Vec>&       seeds,
                                 cells::Cells&             cells);
        void find_seeds_at_sync(const timesteps::Sync&    sync,
                                const timesteps::TopStep& top,
                                const mesh::Mesh&         mesh,
                                device::Array<Vec>&       seeds,
                                cells::Cells&             cells);
        void end_steps(simulation::Simulation& sim, CellsOfSync& of_sync);
        void write_snapshot_if_due(simulation::Simulation& sim);
        void print_begin_of_steps(const timesteps::Sync&    sync,
                                  const timesteps::TopStep& top,
                                  int                       step,
                                  const mesh::Mesh&         mesh,
                                  const CellsOfSync&        of_sync,
                                  const cells::Cells&       cells,
                                  LogOfRun&                 of_run);
        void print_top_step(const timesteps::Sync&    sync,
                            const timesteps::TopStep& top,
                            int                       step,
                            const cells::Cells&       cells,
                            const LogOfRun&           of_run);
        void print_sync(const timesteps::Sync&      sync,
                        const mesh::CellList&       active,
                        const timesteps::CellSteps& steps,
                        LogOfRun&                   of_run);
        void print_fluxes(const mesh::Mesh&              mesh,
                          const timesteps::StepPointers& at_sync,
                          timesteps::Half                half,
                          const mesh::CellList&          with_faces,
                          const timesteps::BinRange&     bins);
        std::string text_of_bins(const timesteps::BinRange& bins);
        std::string hours_minutes_seconds(double seconds);

    } // namespace

    void run(int argc, char* argv[]) {
        simulation::Simulation sim;
        begrun::begrun(argc, argv, sim);

        // one pass per sync; every cell is active at the first one
        CellsOfSync        of_sync{mesh::every_cell(sim.mesh.pointers()), mesh::every_cell(sim.mesh.pointers())};
        device::Array<Vec> seeds("integrator: seeds at the sync", 0);
        LogOfRun           of_run{sim.start_of_run, sim.start_time, sim.parameters.time_end, {0, 0}, {0, 0}};
        logging::align_labelled_text();
        {
            const timers::Scope timer("LOOP");
            while (sim.sync.time < sim.parameters.time_end) {

                // START
                begin_steps(sim, of_sync);
                {
                    const timers::Scope timer("LOG");
                    print_begin_of_steps(sim.sync, sim.top, sim.step, sim.mesh, of_sync, sim.cells, of_run);
                }

                // TIME
                of_sync.active = go_to_next_sync(sim);
                {
                    const timers::Scope timer("LOG");
                    print_sync(sim.sync, of_sync.active, sim.cells.steps, of_run);
                }

                // MESH
                move_mesh(sim, of_sync.active, seeds);
                if constexpr (config::moving_mesh) {
                    const timers::Scope timer("LOG");
                    sim.mesh.print_builds(sim.ghosts);
                }

                // FINISH
                end_steps(sim, of_sync);
                {
                    const timers::Scope timer("LOG");
                    print_fluxes(sim.mesh,
                                 sim.top.pointers(sim.sync, sim.cells.steps),
                                 timesteps::Half::second,
                                 of_sync.with_faces,
                                 of_run.bins_that_end);
                }

                if (sim.sync.ends_top_step) {

                    sim.step++;
                    if constexpr (config::profiling) timers::record_top_step(sim.step, sim.sync.time);
                    {
                        const timers::Scope timer("LOG");
                        sim.top.print_syncs_and_steps(sim.cells.n_cells);
                    }

                    // OUTPUT
                    write_snapshot_if_due(sim);
                }
            }
        }
        if constexpr (config::individual_timesteps) logging::print("\n%s", LINE_BETWEEN_TOP_STEPS);
        if constexpr (config::profiling)
            io::write_profile(sim.parameters.output_directory, sim.profile, timers::profile_of_run());
        logging::print("\nHYDRO: Finished after %d steps at t = %g", sim.step, sim.sync.time);

        begrun::endrun(sim);
    }

    namespace {

        // ==========================================================
        // START, TIME, MESH, FINISH and OUTPUT
        // ==========================================================

        // START: new bins for the active cells, then their first half of sources, gradients and fluxes
        void begin_steps(simulation::Simulation& sim, const CellsOfSync& of_sync) {
            const mesh::MeshPointers          on_mesh   = sim.mesh.pointers();
            const gas::PrimitivePointers      state     = cells::primitive_of(sim.cells);
            const gas::ConservedPointers      conserved = cells::conserved_of(sim.cells);
            const gradients::GradientPointers gradients = cells::gradients_of(sim.cells);
            const mesh::CellList&             active    = of_sync.active;

            const timers::Scope timer("START");

            const double stop_time = std::min(sim.snapshots.next_time, sim.parameters.time_end);
            {
                const timers::Scope timer("AGN_PREPARE");
                sim.sources.prepare(sim.sync, sim.cells.id.data(), sim.cells.position.data(), state, on_mesh);
            }
            find_cfl_steps(sim.parameters,
                           active,
                           sim.mesh,
                           sim.sources,
                           sim.top.longest_step_that_matters(sim.sync, stop_time),
                           sim.signals,
                           sim.cells);
            {
                const timers::Scope timer("BINS");
                sim.top.begin_steps(
                    active, sim.cells.cfl_step.data(), stop_time, on_mesh, sim.ghosts, sim.sync, sim.cells.steps);
            }
            const timesteps::StepPointers at_sync = sim.top.pointers(sim.sync, sim.cells.steps);

            {
                const timers::Scope timer("SOURCES");
                sim.sources.apply_first_half(
                    sim.top.times_of_sources(
                        timesteps::Half::first, active, sim.cells.steps, sim.cells.source_clock.data()),
                    active,
                    sim.cells.position.data(),
                    state);
            }

            // the neighbours on other ranks get the primitive variables before the gradients
            {
                const timers::Scope timer("GHOST_CELLS", timers::Type::communication);
                cells::copy_primitive_variables_to_ghost_cells(sim.cells, sim.ghosts, active);
            }
            {
                const timers::Scope timer("GRAD");
                gradients::compute_gradients(
                    on_mesh, state, sim.cells.mesh_velocity.data(), at_sync, active, gradients);
            }
            if constexpr (config::moving_mesh) {
                const timers::Scope timer("MESH_VELOCITY");
                mesh_motion::mesh_velocities(on_mesh, state, gradients.density, active, sim.cells.mesh_velocity.data());
                if constexpr (config::individual_timesteps) {
                    sim.top.remember_fastest_mesh_speed(active, sim.cells.mesh_velocity.data());
                }
            }

            // the neighbours on other ranks get the gradients and mesh velocities before the fluxes
            {
                const timers::Scope timer("GHOST_CELLS", timers::Type::communication);
                cells::copy_gradients_to_ghost_cells(sim.cells, sim.ghosts, active);
            }
            {
                const timers::Scope timer("PRIM_TO_CONS");
                hydro::conserved_from_primitive(on_mesh, state, active, conserved);
            }
            {
                const timers::Scope timer("FLUX");
                hydro::apply_fluxes(on_mesh,
                                    state,
                                    gradients,
                                    sim.cells.mesh_velocity.data(),
                                    at_sync,
                                    timesteps::Half::first,
                                    sim.sources.lowest_specific_energy(),
                                    of_sync.with_faces,
                                    conserved);
            }
        }

        // CFL step of every cell of the list, shortened near the AGN and by signals from other cells; collective
        void find_cfl_steps(const params::RunParameters& parameters,
                            const mesh::CellList&        list,
                            const mesh::Mesh&            mesh,
                            const astro::Sources&        sources,
                            double                       longest_step_that_matters,
                            signals::Signals&            signals,
                            cells::Cells&                cells) {
            const mesh::MeshPointers     on_mesh = mesh.pointers();
            const gas::PrimitivePointers state   = cells::primitive_of(cells);
            {
                const timers::Scope timer("CFL");
                hydro::cfl_steps(
                    on_mesh, state, cells.mesh_velocity.data(), parameters.cfl_fraction, list, cells.cfl_step.data());
                sources.shorten_cfl_steps_near_the_agn(list,
                                                       cells.position.data(),
                                                       state,
                                                       cells.mesh_velocity.data(),
                                                       on_mesh,
                                                       parameters.cfl_fraction,
                                                       cells.cfl_step.data());
            }
            if constexpr (config::individual_timesteps) {
                const timers::Scope timer("SIGNALS");
                signals.shorten_cfl_steps(
                    list, mesh, state, parameters.cfl_fraction, longest_step_that_matters, cells.cfl_step.data());
            }
        }

        // TIME: the next sync and its active cells; when the AGN switched on, too long steps end there too; collective
        mesh::CellList go_to_next_sync(simulation::Simulation& sim) {
            const timers::Scope timer("TIME");
            sim.sync = sim.top.next_sync(sim.cells.steps, sim.cells.n_cells);
            if (sim.sources.agn_switched_on() && !sim.sync.ends_top_step) {
                end_steps_longer_than_their_cfl_step(
                    sim.parameters, sim.mesh, sim.ghosts, sim.sources, sim.signals, sim.sync, sim.top, sim.cells);
            }
            return sim.top.active_cells(sim.sync, sim.mesh.pointers(), sim.cells.steps);
        }

        void end_steps_longer_than_their_cfl_step(const params::RunParameters& parameters,
                                                  const mesh::Mesh&            mesh,
                                                  ghosts::Ghosts&              ghosts,
                                                  const astro::Sources&        sources,
                                                  signals::Signals&            signals,
                                                  const timesteps::Sync&       sync,
                                                  timesteps::TopStep&          top,
                                                  cells::Cells&                cells) {
            const mesh::MeshPointers on_mesh = mesh.pointers();
            find_cfl_steps(parameters,
                           mesh::every_cell(on_mesh),
                           mesh,
                           sources,
                           top.longest_step_that_matters(sync, parameters.time_end),
                           signals,
                           cells);
            top.end_steps_longer_than(cells.cfl_step.data(), sync, on_mesh, ghosts, cells.steps);
        }

        // MESH: moves the seeds; builds around the active cells, or the whole mesh at the end of a top step; collective
        void move_mesh(simulation::Simulation& sim, mesh::CellList& active, device::Array<Vec>& seeds) {

            if constexpr (!config::moving_mesh) return;
            const timers::Scope timer("MESH");

            if (config::individual_timesteps && !sim.sync.ends_top_step) {
                active = build_around_active_cells(
                    sim.sync, sim.top, sim.domain, sim.ghosts, sim.mesh, active, seeds, sim.cells);
                return;
            }

            cells::move_seeds(sim.cells, active, sim.top.pointers(sim.sync, sim.cells.steps));
            if (!sim.sync.ends_top_step) return;

            double imbalance_before = 0.0;
            {
                const timers::Scope timer("CUTS", timers::Type::communication);
                imbalance_before = cut_again_if_unbalanced(sim.domain, sim.cells);
            }
            {
                const timers::Scope timer("MIGRATE", timers::Type::communication);
                cells::move_to_owners(sim.cells, sim.domain);
            }
            if (imbalance_before > 0.0) {
                logging::print_labelled("DECOMP",
                                        "Rebalanced at t = %g. Imbalance %.2f -> %.2f",
                                        sim.sync.time,
                                        imbalance_before,
                                        imbalance_of_ranks(sim.cells));
            }
            {
                const timers::Scope timer("ORDER");
                cells::sort_along_curve(sim.cells);
            }
            {
                const timers::Scope timer("BUILD");
                sim.mesh.build(sim.cells.position.data(), sim.cells.n_cells, sim.domain, sim.ghosts);
            }
            const mesh::MeshPointers on_mesh = sim.mesh.pointers();
            cells::make_room_for(sim.cells, sim.cells.n_cells, on_mesh.n_ghost_cells);
            active = mesh::every_cell(on_mesh);
            {
                const timers::Scope timer("GHOST_CELLS", timers::Type::communication);
                cells::copy_everything_to_ghost_cells(sim.cells, sim.ghosts);
            }
        }

        // new cuts if the ranks are unbalanced; returns the imbalance before them, 0 without new cuts; collective
        double cut_again_if_unbalanced(domain::Domain& domain, cells::Cells& cells) {
            if constexpr (config::individual_timesteps) {
                device::Array<long> work("integrator: work of the cells in the top step", cells.n_cells);
                timesteps::work_of_cells(cells.steps, cells.n_cells, work.data());
                if (!domain::is_unbalanced_by_work(work.data(), cells.n_cells)) return 0.0;
                const double before = domain::imbalance_by_work(work.data(), cells.n_cells);
                domain.cut_by_work(cells.position.data(), work.data(), cells.n_cells);
                return before;
            } else {
                if (!domain::is_unbalanced(cells.n_cells)) return 0.0;
                const double before = domain::imbalance(cells.n_cells);
                domain.cut_by_number_of_cells(cells.position.data(), cells.n_cells);
                return before;
            }
        }

        // what cut_again_if_unbalanced measures, for the cells where they are now; collective
        double imbalance_of_ranks(const cells::Cells& cells) {
            if constexpr (config::individual_timesteps) {
                device::Array<long> work("integrator: work of the cells in the top step", cells.n_cells);
                timesteps::work_of_cells(cells.steps, cells.n_cells, work.data());
                return domain::imbalance_by_work(work.data(), cells.n_cells);
            } else {
                return domain::imbalance(cells.n_cells);
            }
        }

        // builds the ending cells and their neighbours, so every face that opens or closes is in the mesh; collective
        mesh::CellList build_around_active_cells(const timesteps::Sync& sync,
                                                 timesteps::TopStep&    top,
                                                 const domain::Domain&  domain,
                                                 ghosts::Ghosts&        ghosts,
                                                 mesh::Mesh&            mesh,
                                                 const mesh::CellList&  ending,
                                                 device::Array<Vec>&    seeds,
                                                 cells::Cells&          cells) {
            const timers::Scope timer("BUILD_AROUND_ACTIVE");
            const double        drift_margin = top.drift_margin(sync);
            mesh.forget_faces();
            find_seeds_at_sync(sync, top, mesh, seeds, cells);

            build_cells_of_list(ending, sync, top, drift_margin, domain, ghosts, mesh, seeds, cells);
            mesh::CellList neighbours;
            {
                const timers::Scope timer("NEIGHBOURS", timers::Type::communication);
                neighbours = mesh.neighbours_not_built(ending, ghosts);
            }
            build_cells_of_list(neighbours, sync, top, drift_margin, domain, ghosts, mesh, seeds, cells);

            cells::move_seeds(cells, ending, top.pointers(sync, cells.steps));
            return ending;
        }

        // builds the cells of the list; when ghost cells came with it, refills all ghost cells and seeds; collective
        void build_cells_of_list(const mesh::CellList&     list,
                                 const timesteps::Sync&    sync,
                                 const timesteps::TopStep& top,
                                 double                    drift_margin,
                                 const domain::Domain&     domain,
                                 ghosts::Ghosts&           ghosts,
                                 mesh::Mesh&               mesh,
                                 device::Array<Vec>&       seeds,
                                 cells::Cells&             cells) {
            const int ghost_cells_before = mesh.pointers().n_ghost_cells;
            {
                const timers::Scope timer("BUILD_LIST");
                mesh.build_list(seeds.data(), cells.n_cells, list, domain, drift_margin, ghosts);
            }
            const int n_ghost_cells = mesh.pointers().n_ghost_cells;
            if (!mpi::on_any_rank(n_ghost_cells > ghost_cells_before)) return;

            const timers::Scope timer("GHOST_CELLS", timers::Type::communication);
            cells::make_room_for(cells, cells.n_cells, n_ghost_cells);
            cells::copy_everything_to_ghost_cells(cells, ghosts);
            find_seeds_at_sync(sync, top, mesh, seeds, cells);
        }

        void find_seeds_at_sync(const timesteps::Sync&    sync,
                                const timesteps::TopStep& top,
                                const mesh::Mesh&         mesh,
                                device::Array<Vec>&       seeds,
                                cells::Cells&             cells) {
            const int n_ghost_cells = mesh.pointers().n_ghost_cells;
            seeds.grow_to_at_least(cells.n_cells + n_ghost_cells);
            cells::seeds_at_sync(cells, n_ghost_cells, top.pointers(sync, cells.steps), seeds.data());
        }

        // FINISH: fluxes over the closing faces, then the active cells get their new state and second half of sources
        void end_steps(simulation::Simulation& sim, CellsOfSync& of_sync) {
            const timers::Scope      timer("FINISH");
            const mesh::MeshPointers on_mesh = sim.mesh.pointers();
            {
                const timers::Scope timer("FACES_AT_SYNC");
                of_sync.with_faces =
                    sim.top.cells_with_faces_at_sync(sim.sync, of_sync.active, on_mesh, sim.cells.steps);
            }

            const gas::PrimitivePointers      state     = cells::primitive_of(sim.cells);
            const gas::ConservedPointers      conserved = cells::conserved_of(sim.cells);
            const gradients::GradientPointers gradients = cells::gradients_of(sim.cells);
            const timesteps::StepPointers     at_sync   = sim.top.pointers(sim.sync, sim.cells.steps);

            // a cell in the middle of its step takes the fluxes in its conserved variables only
            {
                const timers::Scope timer("FLUX");
                hydro::apply_fluxes(on_mesh,
                                    state,
                                    gradients,
                                    sim.cells.mesh_velocity.data(),
                                    at_sync,
                                    timesteps::Half::second,
                                    sim.sources.lowest_specific_energy(),
                                    of_sync.with_faces,
                                    conserved);
            }
            {
                const timers::Scope timer("CONS_TO_PRIM");
                hydro::primitive_from_conserved(
                    on_mesh, conserved, of_sync.active, sim.sources.lowest_specific_energy(), state);
            }

            {
                const timers::Scope timer("SOURCES");
                sim.sources.apply_second_half(
                    sim.top.times_of_sources(
                        timesteps::Half::second, of_sync.active, sim.cells.steps, sim.cells.source_clock.data()),
                    of_sync.active,
                    sim.cells.position.data(),
                    state);
            }
            {
                const timers::Scope timer("UNPHYS_CHECK", timers::Type::communication);
                hydro::stop_if_unphysical(state, conserved, of_sync.active);
            }
        }

        // OUTPUT: a snapshot at the end of a top step that reaches its due time or the end of the run
        void write_snapshot_if_due(simulation::Simulation& sim) {
            const params::RunParameters& parameters = sim.parameters;
            const bool reached = sim.sync.time >= sim.snapshots.next_time || sim.sync.time >= parameters.time_end;
            if (!sim.sync.ends_top_step || !reached) return;

            const timers::Scope timer("IO_SNAPSHOT");
            if constexpr (config::profiling) {
                io::write_profile(parameters.output_directory, sim.profile, timers::profile_of_run());
            }
            io::write_snapshot(parameters.output_directory,
                               sim.snapshots.number,
                               sim.sync.time,
                               sim.step,
                               parameters.astro.units,
                               sim.cells,
                               sim.mesh.pointers(),
                               sim.ghosts.pointers(),
                               sim.top.pointers(sim.sync, sim.cells.steps));
            sim.snapshots.number++;
            sim.snapshots.next_time += parameters.output_dt;
        }

        // ==========================================================
        // the log
        // ==========================================================

        // the lines of a new top step, else an empty line, then the fluxes; collective
        void print_begin_of_steps(const timesteps::Sync&    sync,
                                  const timesteps::TopStep& top,
                                  int                       step,
                                  const mesh::Mesh&         mesh,
                                  const CellsOfSync&        of_sync,
                                  const cells::Cells&       cells,
                                  LogOfRun&                 of_run) {
            if (sync.ends_top_step) {
                print_top_step(sync, top, step, cells, of_run);
            } else {
                logging::print("%s", "");
            }
            of_run.bins_that_begin = timesteps::bins_of(of_sync.active, cells.steps);
            print_fluxes(mesh,
                         top.pointers(sync, cells.steps),
                         timesteps::Half::first,
                         of_sync.with_faces,
                         of_run.bins_that_begin);
        }

        // with individual timesteps also a line above it and the cells of every bin; collective
        void print_top_step(const timesteps::Sync&    sync,
                            const timesteps::TopStep& top,
                            int                       step,
                            const cells::Cells&       cells,
                            const LogOfRun&           of_run) {
            const double      elapsed = std::chrono::duration<double>(Clock::now() - of_run.start_of_run).count();
            const double      left    = (sync.time > of_run.start_time)
                                            ? elapsed * (of_run.time_end - sync.time) / (sync.time - of_run.start_time)
                                            : 0.0;
            const std::string eta     = hours_minutes_seconds(left);
            if constexpr (config::individual_timesteps) {
                logging::print("\n%s\n", LINE_BETWEEN_TOP_STEPS);
                logging::print("SIM: Top-Step %d  t = %g  dt = %g  ETA = %s",
                               step,
                               sync.time,
                               top.length_of_this_top_step(),
                               eta.c_str());
                top.print_cells_per_bin(cells.steps, cells.n_cells);
            } else {
                logging::print("\nSIM: Step %d  t = %g  dt = %g  ETA = %s",
                               step,
                               sync.time,
                               top.length_of_this_top_step(),
                               eta.c_str());
            }
        }

        // with individual timesteps: where the sync lies in the top step, and its active cells; collective
        void print_sync(const timesteps::Sync&      sync,
                        const mesh::CellList&       active,
                        const timesteps::CellSteps& steps,
                        LogOfRun&                   of_run) {
            if constexpr (!config::individual_timesteps) return;

            of_run.bins_that_end = timesteps::bins_of(active, steps);
            const int finest     = of_run.bins_that_begin.finest;
            logging::print_labelled("SIM",
                                    "Sync %d/%d  t = %g (active: bin %s, %lld cells)",
                                    sync.tick >> (timesteps::FINEST_BIN - finest),
                                    1 << finest,
                                    sync.time,
                                    text_of_bins(of_run.bins_that_end).c_str(),
                                    mpi::sum_over_ranks((long long)active.n));
        }

        // counts the faces of the list that get a flux in this half, each from both sides; collective
        void print_fluxes(const mesh::Mesh&              mesh,
                          const timesteps::StepPointers& at_sync,
                          timesteps::Half                half,
                          const mesh::CellList&          with_faces,
                          const timesteps::BinRange&     bins) {
            const mesh::MeshPointers on_mesh  = mesh.pointers();
            const long long          here     = device::sum(with_faces.n, [=] HD(long i) {
                const int cell  = with_faces[(int)i];
                const int first = on_mesh.first_face[cell];
                long long n     = 0;
                for (int face = first; face < first + on_mesh.n_faces[cell]; face++) {
                    if (timesteps::face_share(at_sync, cell, on_mesh.neighbour[face], half).time != 0.0) n++;
                }
                return n;
            });
            const long long          n_fluxes = mpi::sum_over_ranks(here);
            const char*              label    = (half == timesteps::Half::first) ? "START" : "FINISH";
            if constexpr (config::individual_timesteps) {
                logging::print_labelled(label, "%lld fluxes (bin %s)", n_fluxes, text_of_bins(bins).c_str());
            } else {
                logging::print_labelled(label, "%lld fluxes", n_fluxes);
            }
        }

        // "2" for one bin, "0-2" for several
        std::string text_of_bins(const timesteps::BinRange& bins) {
            if (bins.coarsest == bins.finest) return std::to_string(bins.finest);
            return std::to_string(bins.coarsest) + "-" + std::to_string(bins.finest);
        }

        std::string hours_minutes_seconds(double seconds) {
            const long long total = (long long)(std::max(seconds, 0.0) + 0.5);
            char            text[64];
            std::snprintf(text, sizeof text, "%02lld:%02lld:%02lld", total / 3600, (total % 3600) / 60, total % 60);
            return std::string(text);
        }

    } // namespace

} // namespace integrator
