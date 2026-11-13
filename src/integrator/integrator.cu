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
#include "profile/profile.h"
#include "signals/signals.h"
#include "simulation/simulation.h"
#include "timers/timers.h"
#include "timesteps/timesteps.h"

#include <algorithm>
#include <chrono>

namespace integrator {

    namespace {

        // the cells a sync works on
        struct CellsOfSync {
            mesh::CellList active;
            mesh::CellList with_faces;
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
        void apply_limited_fluxes(simulation::Simulation& sim, timesteps::Half half, const mesh::CellList& with_faces);
        void print_limited_fluxes(simulation::Simulation& sim);
        void write_snapshot_if_due(simulation::Simulation& sim);
        double seconds_left(const simulation::Simulation& sim);

    } // namespace

    void run(int argc, char* argv[]) {
        simulation::Simulation sim;
        begrun::begrun(argc, argv, sim);

        // one pass per sync; every cell is active at the first one
        CellsOfSync        of_sync{mesh::every_cell(sim.mesh.pointers()), mesh::every_cell(sim.mesh.pointers())};
        device::Array<Vec> seeds("integrator: seeds at the sync", 0);
        logging::align_labelled_text();
        {
            const timers::Scope timer("LOOP");
            while (sim.sync.time < sim.parameters.time_end) {

                // START
                begin_steps(sim, of_sync);
                sim.top.print_begin_of_steps(
                    sim.sync, sim.step, seconds_left(sim), sim.mesh.pointers(), of_sync.active, sim.cells.steps);

                // TIME
                of_sync.active = go_to_next_sync(sim);
                sim.top.print_sync(sim.sync, of_sync.active, sim.cells.steps);

                // MESH
                move_mesh(sim, of_sync.active, seeds);
                if constexpr (config::moving_mesh) sim.mesh.print_builds(sim.ghosts);

                // FINISH
                end_steps(sim, of_sync);
                sim.top.print_end_of_steps();

                if (sim.sync.ends_top_step) {

                    sim.step++;
                    if constexpr (config::profiling) timers::record_top_step(sim.step, sim.sync.time);
                    sim.top.print_syncs_and_steps(sim.cells.n_cells);
                    print_limited_fluxes(sim);

                    // OUTPUT
                    write_snapshot_if_due(sim);
                }
            }
        }
        if constexpr (config::individual_timesteps) logging::print("\n%s", timesteps::LINE_BETWEEN_TOP_STEPS);
        if constexpr (config::profiling)
            io::write_profile(sim.parameters.output_directory, sim.profile, profile::profile_of_run());
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
                sim.sources.prepare(sim.sync, sim.cells.position.data(), state, on_mesh);
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
                const timers::Scope timer("GHOST_CELLS");
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

            // the neighbours on other ranks get the gradients, mesh velocities and masses before the fluxes
            hydro::masses_at_start_of_step(on_mesh, state, active, sim.cells.step_mass.data());
            {
                const timers::Scope timer("GHOST_CELLS");
                cells::copy_gradients_to_ghost_cells(sim.cells, sim.ghosts, active);
            }
            {
                const timers::Scope timer("PRIM_TO_CONS");
                hydro::conserved_from_primitive(on_mesh, state, active, conserved);
            }
            {
                const timers::Scope timer("FLUX");
                apply_limited_fluxes(sim, timesteps::Half::first, of_sync.with_faces);
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
                const timers::Scope timer("CUTS");
                imbalance_before = cut_again_if_unbalanced(sim.domain, sim.cells);
            }
            {
                const timers::Scope timer("MIGRATE");
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
                const timers::Scope timer("GHOST_CELLS");
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
                const timers::Scope timer("NEIGHBOURS");
                neighbours = mesh.neighbours_not_built(ending, ghosts);
            }
            build_cells_of_list(neighbours, sync, top, drift_margin, domain, ghosts, mesh, seeds, cells);

            cells::move_seeds(cells, ending, top.pointers(sync, cells.steps));
            return ending;
        }

        // builds the cells of the list; ghost cells that came with it get their values and seeds; collective
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

            const timers::Scope timer("GHOST_CELLS");
            cells::make_room_for(cells, cells.n_cells, n_ghost_cells);
            cells::copy_everything_to_new_ghost_cells(cells, ghosts);
            if (n_ghost_cells > ghost_cells_before) find_seeds_at_sync(sync, top, mesh, seeds, cells);
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

            const gas::PrimitivePointers state     = cells::primitive_of(sim.cells);
            const gas::ConservedPointers conserved = cells::conserved_of(sim.cells);

            // a cell in the middle of its step takes the fluxes in its conserved variables only
            {
                const timers::Scope timer("FLUX");
                apply_limited_fluxes(sim, timesteps::Half::second, of_sync.with_faces);
            }
            {
                const timers::Scope            timer("CONS_TO_PRIM");
                const hydro::ThermalFloorCount floored = hydro::primitive_from_conserved(
                    on_mesh, conserved, of_sync.active, sim.sources.lowest_specific_energy(), state);
                sim.floored_cells += floored.cells;
                sim.floor_energy += floored.energy;
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
                const timers::Scope timer("UNPHYS_CHECK");
                hydro::stop_if_unphysical(state, conserved, of_sync.active);
            }
        }

        // optimistic: the fluxes without the limiter, which give the exposures; only if a cell on some rank would lose
        // too much does every cell of the list go back to its conserved variables before, and the fluxes are taken
        // again with the limiter, once the ghost cells have the exposures; collective
        void apply_limited_fluxes(simulation::Simulation& sim, timesteps::Half half, const mesh::CellList& with_faces) {
            const mesh::MeshPointers          on_mesh   = sim.mesh.pointers();
            const gas::PrimitivePointers      state     = cells::primitive_of(sim.cells);
            const gas::ConservedPointers      conserved = cells::conserved_of(sim.cells);
            const gradients::GradientPointers gradients = cells::gradients_of(sim.cells);
            const timesteps::StepPointers     at_sync   = sim.top.pointers(sim.sync, sim.cells.steps);
            const double                      lowest    = sim.sources.lowest_specific_energy();
            const double*                     step_mass = sim.cells.step_mass.data();
            double*                           exposure  = sim.cells.outflow_exposure.data();
            const auto                        fluxes    = [&](const double* limit_by, double* exposure_out) {
                return hydro::apply_fluxes(on_mesh,
                                           state,
                                           gradients,
                                           sim.cells.mesh_velocity.data(),
                                           at_sync,
                                           half,
                                           lowest,
                                           step_mass,
                                           limit_by,
                                           exposure_out,
                                           with_faces,
                                           conserved);
            };

            sim.cells.conserved_backup.grow_to_at_least(with_faces.n);
            hydro::save_conserved(with_faces, conserved, sim.cells.conserved_backup.data());
            hydro::FluxCount count = fluxes(nullptr, exposure);
            sim.flux_passes++;
            if (mpi::sum_over_ranks(hydro::count_over_exposed(with_faces, step_mass, exposure)) > 0) {
                hydro::restore_conserved(with_faces, sim.cells.conserved_backup.data(), conserved);
                cells::copy_outflow_exposure_to_ghost_cells(sim.cells, sim.ghosts);
                count = fluxes(exposure, nullptr);
                sim.redone_passes++;
            }
            sim.fluxes += count.faces;
            sim.limited_fluxes += count.limited;
        }

        // the fluxes and thermal floors of the top step over all ranks, as Arepo's FLUX line; collective
        void print_limited_fluxes(simulation::Simulation& sim) {
            long long number[3] = {sim.fluxes, sim.limited_fluxes, sim.floored_cells};
            mpi::sum_over_ranks(number, 3);
            const double added = mpi::sum_over_ranks(sim.floor_energy);
            logging::print("HYDRO: %lld fluxes over faces, %lld of them limited (fraction %g; %d of %d halves done "
                           "again); %lld cells with negative thermal energy set to zero, adding %g",
                           number[0],
                           number[1],
                           (double)number[1] / ((double)number[0] + 1e-30),
                           sim.redone_passes,
                           sim.flux_passes,
                           number[2],
                           added);
            sim.flux_passes    = 0;
            sim.redone_passes  = 0;
            sim.fluxes         = 0;
            sim.limited_fluxes = 0;
            sim.floored_cells  = 0;
            sim.floor_energy   = 0.0;
        }

        // OUTPUT: a snapshot at the end of a top step that reaches its due time or the end of the run
        void write_snapshot_if_due(simulation::Simulation& sim) {
            const params::RunParameters& parameters = sim.parameters;
            const bool reached = sim.sync.time >= sim.snapshots.next_time || sim.sync.time >= parameters.time_end;
            if (!sim.sync.ends_top_step || !reached) return;

            const timers::Scope timer("IO_SNAPSHOT");
            if constexpr (config::profiling) {
                io::write_profile(parameters.output_directory, sim.profile, profile::profile_of_run());
            }
            if (parameters.write_snapshots)
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

        // from how fast the run went so far; 0 as long as no time has passed
        double seconds_left(const simulation::Simulation& sim) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - sim.start_of_run).count();
            return (sim.sync.time > sim.start_time)
                       ? elapsed * (sim.parameters.time_end - sim.sync.time) / (sim.sync.time - sim.start_time)
                       : 0.0;
        }

    } // namespace

} // namespace integrator
