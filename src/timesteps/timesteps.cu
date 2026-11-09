// implements the top step, the steps of the cells and the syncs (timesteps.h)

#include "timesteps/timesteps.h"

#include "logging/logging.h"
#include "mpi/mpi.h"
#include "timers/timers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace timesteps {

    namespace {

        // a top step takes the one of this many lengths with the least work per time
        constexpr int TOP_STEP_CANDIDATES = 17;

        // more than rounding one move to the box precision can change a seed's distance to anything
        constexpr double ROUNDING_OF_A_MOVE = 0x1p-50;

        // the lengths a top step may take, and the work of each in half cell steps, which is a whole number
        struct TopStepCandidates {
            double length[TOP_STEP_CANDIDATES];
        };
        struct WorkOfCandidates {
            long long half_cell_steps[TOP_STEP_CANDIDATES];
        };

        double length_of_top_step(const mesh::CellList& cells, const double* cfl_step);
        TopStepCandidates candidates_up_to(double longest, double smallest_cfl_step);
        WorkOfCandidates work_of_candidates(const mesh::CellList&    cells,
                                            const double*            cfl_step,
                                            const TopStepCandidates& candidates);
        HD int bin_for_step(double top_length, double step);
        void needs_from_cfl_steps(const mesh::CellList& active,
                                  const double*         cfl_step,
                                  double                top_length,
                                  CellSteps&            steps);
        void limit_needs_by_face_neighbours(const mesh::CellList&     active,
                                            const mesh::MeshPointers& mesh,
                                            ghosts::Ghosts&           ghosts,
                                            device::Array<int>&       next_need,
                                            CellSteps&                steps);
        void steps_from_needs(const mesh::CellList& active, int tick, CellSteps& steps);
        void print_top_step(const TopStep&   top,
                            const Sync&      sync,
                            int              step,
                            double           seconds_left,
                            const CellSteps& steps,
                            int              n_cells);
        BinRange bins_of(const mesh::CellList& list, const CellSteps& steps);
        void print_fluxes(const mesh::MeshPointers& mesh,
                          const StepPointers&       at_sync,
                          Half                      half,
                          const mesh::CellList&     with_faces,
                          const BinRange&           bins);
        std::string text_of_bins(const BinRange& bins);

        HD int ticks_of_bin(int bin) {
            return TICKS_PER_TOP_STEP >> bin;
        }

    } // namespace

    Sync first_sync(double start_time) {
        return Sync{0, start_time, true};
    }

    // a top step that would run past stop_time ends exactly there
    void TopStep::begin_steps(const mesh::CellList&     active,
                              const double*             cfl_step,
                              double                    stop_time,
                              const mesh::MeshPointers& mesh,
                              ghosts::Ghosts&           ghosts,
                              Sync&                     sync,
                              CellSteps&                steps) {
        if (sync.ends_top_step) {
            const double length = length_of_top_step(active, cfl_step);
            const bool   cut    = sync.time + length > stop_time;
            start_time          = sync.time;
            this->length        = cut ? stop_time - sync.time : length;
            end_time            = cut ? stop_time : sync.time + length;
            tick_length         = std::ldexp(this->length, -FINEST_BIN);
            n_syncs             = 0;
            n_cell_steps        = 0;
            fastest_mesh_speed  = 0.0;
            sync.tick           = 0;
        }
        n_cell_steps += active.n;

        if constexpr (config::individual_timesteps) {
            needs_from_cfl_steps(active, cfl_step, this->length, steps);
            limit_needs_by_face_neighbours(active, mesh, ghosts, next_need, steps);
            steps_from_needs(active, sync.tick, steps);
            ghosts.copy_changed_to_ghost_cells(
                mesh.n_cells, active.cell, active.n, [&](auto function) { for_each_field(steps, function); });
        }
    }

    // at a top step's end every candidate of at least half this is cut at stop_time; inside one such a step gets bin 0
    double TopStep::longest_step_that_matters(const Sync& sync, double stop_time) const {
        return 4.0 * (sync.ends_top_step ? stop_time - sync.time : length);
    }

    Sync TopStep::next_sync(const CellSteps& steps, int n_cells) const {
        int tick = TICKS_PER_TOP_STEP;
        if constexpr (config::individual_timesteps) {
            const int* end = steps.end.data();
            tick           = mpi::min_over_ranks(device::min(n_cells, [=] HD(long cell) { return end[cell]; }));
        }
        const bool   ends_top_step = (tick == TICKS_PER_TOP_STEP);
        const double time          = ends_top_step ? end_time : start_time + (double)tick * tick_length;
        return Sync{tick, time, ends_top_step};
    }

    mesh::CellList TopStep::active_cells(const Sync& sync, const mesh::MeshPointers& mesh, const CellSteps& steps) {
        n_syncs++;
        if (sync.ends_top_step) return mesh::every_cell(mesh);

        active.grow_to_at_least(mesh.n_cells);
        int*       list     = active.data();
        const int* end      = steps.end.data();
        const int  tick     = sync.tick;
        const int  n_active = device::indices_where(mesh.n_cells, [=] HD(int cell) { return end[cell] == tick; }, list);
        return mesh::CellList{n_active, list};
    }

    // a step counts as too long only by more than round-off; the planned end stays, as the faces opened with it
    void TopStep::end_steps_longer_than(const double*             cfl_step,
                                        const Sync&               sync,
                                        const mesh::MeshPointers& mesh,
                                        ghosts::Ghosts&           ghosts,
                                        CellSteps&                steps) {
        if constexpr (!config::individual_timesteps) return;
        active.grow_to_at_least(mesh.n_cells);
        int*         ended       = active.data();
        int*         end         = steps.end.data();
        const int*   start       = steps.start.data();
        const int    tick        = sync.tick;
        const double tick_length = this->tick_length;
        const auto   too_long    = [=] HD(int cell) {
            return end[cell] > tick && (double)(end[cell] - start[cell]) * tick_length > cfl_step[cell] * (1.0 + 1e-12);
        };
        const int n_ended = device::indices_where(mesh.n_cells, too_long, ended);
        device::parallel_for(n_ended, [=] HD(long i) { end[ended[i]] = tick; });

        const long long all = mpi::sum_over_ranks((long long)n_ended);
        if (all == 0) return;
        ghosts.copy_changed_to_ghost_cells(mesh.n_cells, ended, n_ended, [&](auto function) { function(steps.end); });
        logging::print("TIMESTEPS: a source switched on at t = %.10g, %lld cells end their step now", sync.time, all);
    }

    mesh::CellList TopStep::cells_with_faces_at_sync(const Sync&               sync,
                                                     const mesh::CellList&     active,
                                                     const mesh::MeshPointers& mesh,
                                                     const CellSteps&          steps) {
        if (active.n == mesh.n_cells) return mesh::every_cell(mesh);

        has_faces.grow_to_at_least(mesh.n_cells);
        with_faces.grow_to_at_least(mesh.n_cells);
        has_faces.fill(false);

        bool*      marked = has_faces.data();
        const int* end    = steps.end.data();
        const int  tick   = sync.tick;
        device::parallel_for(active.n, [=] HD(long i) {
            const int cell = active[(int)i];
            marked[cell]   = true;
            for (int face = mesh.first_face[cell]; face < mesh.first_face[cell] + mesh.n_faces[cell]; face++) {
                if (mesh.neighbour[face] < mesh.n_cells) marked[mesh.neighbour[face]] = true;
            }
        });

        // the cells next to an active ghost cell
        if constexpr (config::moving_mesh) {
            // only the cells built at this sync have faces, so this looks at few of them
            device::parallel_for(mesh.n_cells, [=] HD(long cell) {
                for (int face = mesh.first_face[cell]; face < mesh.first_face[cell] + mesh.n_faces[cell]; face++) {
                    if (mesh.neighbour[face] >= mesh.n_cells && end[mesh.neighbour[face]] == tick) marked[cell] = true;
                }
            });
        } else {
            device::parallel_for(mesh.n_ghost_cells, [=] HD(long ghost_cell) {
                if (end[mesh.n_cells + ghost_cell] != tick) return;
                for (int i = mesh.first_cell_next_to_ghost_cell[ghost_cell];
                     i < mesh.first_cell_next_to_ghost_cell[ghost_cell + 1];
                     i++)
                    marked[mesh.cell_next_to_ghost_cell[i]] = true;
            });
        }

        const int n_marked =
            device::indices_where(mesh.n_cells, [=] HD(int cell) { return marked[cell]; }, with_faces.data());
        return mesh::CellList{n_marked, with_faces.data()};
    }

    void TopStep::remember_fastest_mesh_speed(const mesh::CellList& active, const Vec* mesh_velocity) {
        const double fastest = mpi::max_over_ranks(
            device::max(active.n, [=] HD(long i) { return sqrt(norm2(mesh_velocity[active[(int)i]])); }));
        fastest_mesh_speed = std::max(fastest_mesh_speed, fastest);
    }

    // a seed moves at most once per sync
    double TopStep::drift_margin(const Sync& sync) const {
        const double since_start_of_top_step = (double)sync.tick * tick_length;
        return fastest_mesh_speed * since_start_of_top_step * (1.0 + 1e-12) + (double)n_syncs * ROUNDING_OF_A_MOVE;
    }

    SourceTimes TopStep::times_of_sources(Half                  half,
                                          const mesh::CellList& active,
                                          const CellSteps&      steps,
                                          double*               source_clock) {
        if constexpr (!config::astro_physics || !config::individual_timesteps) {
            return SourceTimes{nullptr, 0.5 * length};
        } else {
            const int* start       = steps.start.data();
            const int* planned_end = steps.planned_end.data();
            const int* end         = steps.end.data();
            source_time.grow_to_at_least(steps.end.size());
            double*      clock       = source_clock;
            double*      time        = source_time.data();
            const double tick_length = this->tick_length;
            const bool   first       = (half == Half::first);
            device::parallel_for(active.n, [=] HD(long i) {
                const int cell = active[(int)i];
                if (first && start[cell] == 0) clock[cell] = 0.0;
                const double target =
                    first ? start[cell] + 0.5 * (double)(planned_end[cell] - start[cell]) : (double)end[cell];
                const bool moves = target > clock[cell];
                time[cell]       = moves ? (target - clock[cell]) * tick_length : 0.0;
                clock[cell]      = moves ? target : clock[cell];
            });
            return SourceTimes{time, 0.0};
        }
    }

    StepPointers TopStep::pointers(const Sync& sync, const CellSteps& steps) const {
        return StepPointers{sync.tick, tick_length, steps.start.data(), steps.planned_end.data(), steps.end.data()};
    }

    void TopStep::print_begin_of_steps(const Sync&               sync,
                                       int                       step,
                                       double                    seconds_left,
                                       const mesh::MeshPointers& mesh,
                                       const mesh::CellList&     active,
                                       const mesh::CellList&     with_faces,
                                       const CellSteps&          steps) {
        const timers::Scope timer("LOG");
        if (sync.ends_top_step) {
            print_top_step(*this, sync, step, seconds_left, steps, mesh.n_cells);
        } else {
            logging::print("%s", "");
        }
        bins_that_begin = bins_of(active, steps);
        print_fluxes(mesh, pointers(sync, steps), Half::first, with_faces, bins_that_begin);
    }

    // with individual timesteps only
    void TopStep::print_sync(const Sync& sync, const mesh::CellList& active, const CellSteps& steps) {
        if constexpr (!config::individual_timesteps) return;

        const timers::Scope timer("LOG");
        bins_that_end    = bins_of(active, steps);
        const int finest = bins_that_begin.finest;
        logging::print_labelled("SIM",
                                "Sync %d/%d  t = %g (active: bin %s, %lld cells)",
                                sync.tick >> (FINEST_BIN - finest),
                                1 << finest,
                                sync.time,
                                text_of_bins(bins_that_end).c_str(),
                                mpi::sum_over_ranks((long long)active.n));
    }

    void TopStep::print_end_of_steps(const Sync&               sync,
                                     const mesh::MeshPointers& mesh,
                                     const mesh::CellList&     with_faces,
                                     const CellSteps&          steps) const {
        const timers::Scope timer("LOG");
        print_fluxes(mesh, pointers(sync, steps), Half::second, with_faces, bins_that_end);
    }

    // counted bin after bin, up to the finest one used, so that the counts need no atomics
    void TopStep::print_cells_per_bin(const CellSteps& steps, int n_cells) const {
        if constexpr (!config::individual_timesteps) return;

        const int* bin    = steps.bin.data();
        const int  finest = mpi::max_over_ranks(device::max(n_cells, [=] HD(long cell) { return bin[cell]; }));
        long long  counts[FINEST_BIN + 1] = {};
        for (int b = 0; b <= finest; b++)
            counts[b] = device::sum(n_cells, [=] HD(long cell) { return (long long)(bin[cell] == b); });
        mpi::sum_over_ranks(counts, finest + 1);

        char per_bin[32 * (FINEST_BIN + 1)] = "";
        int  written                        = 0;
        for (int b = 0; b <= finest; b++)
            written += snprintf(
                per_bin + written, sizeof(per_bin) - written, " %d: %lld%s", b, counts[b], (b < finest) ? " " : "");
        logging::print("TIMESTEPS: cells per bin%s\n", per_bin);
    }

    void TopStep::print_syncs_and_steps(int n_cells) const {
        if constexpr (!config::individual_timesteps) return;

        const timers::Scope timer("LOG");
        const long long     steps_of_cells = mpi::sum_over_ranks(n_cell_steps);
        const long long     all_cells      = mpi::sum_over_ranks((long long)n_cells);
        logging::print(
            "\nTIMESTEPS: %d syncs, %.4g steps per cell", n_syncs, (double)steps_of_cells / (double)all_cells);
    }

    void work_of_cells(const CellSteps& steps, int n_cells, long* work) {
        const int* bin = steps.bin.data();
        device::parallel_for(n_cells, [=] HD(long cell) { work[cell] = 2 * (1l << bin[cell]) + 1; });
    }

    namespace {

        // smallest CFL step with one bin, else the candidate with the least work per time; collective
        double length_of_top_step(const mesh::CellList& cells, const double* cfl_step) {
            const auto   cfl_step_of = [=] HD(long i) { return cfl_step[cells[(int)i]]; };
            const double smallest    = mpi::min_over_ranks(device::min(cells.n, cfl_step_of));
            if constexpr (!config::individual_timesteps) {
                return smallest;
            } else {
                const double            largest = mpi::max_over_ranks(device::max(cells.n, cfl_step_of));
                const double            longest = std::min(largest, std::ldexp(smallest, FINEST_BIN - BINS_KEPT_FREE));
                const TopStepCandidates candidates = candidates_up_to(longest, smallest);
                WorkOfCandidates        work       = work_of_candidates(cells, cfl_step, candidates);
                mpi::sum_over_ranks(work.half_cell_steps, TOP_STEP_CANDIDATES);

                int best = 0;
                for (int c = 1; c < TOP_STEP_CANDIDATES; c++) {
                    const double per_time      = (double)work.half_cell_steps[c] / candidates.length[c];
                    const double best_per_time = (double)work.half_cell_steps[best] / candidates.length[best];
                    if (per_time < best_per_time) best = c;
                }
                return candidates.length[best];
            }
        }

        // from the longest down to half of it, evenly in the logarithm; the last fits the smallest CFL step exactly
        TopStepCandidates candidates_up_to(double longest, double smallest_cfl_step) {
            TopStepCandidates candidates;
            for (int c = 0; c < TOP_STEP_CANDIDATES - 1; c++)
                candidates.length[c] = longest * std::exp2(-(double)c / (TOP_STEP_CANDIDATES - 1));
            const int doublings = std::max(0, (int)std::floor(std::log2(longest / smallest_cfl_step)));
            candidates.length[TOP_STEP_CANDIDATES - 1] = std::ldexp(smallest_cfl_step, doublings);
            return candidates;
        }

        // every cell costs the steps of its bin, and half a cell step for the build of the whole mesh
        WorkOfCandidates work_of_candidates(const mesh::CellList&    cells,
                                            const double*            cfl_step,
                                            const TopStepCandidates& candidates) {
            return device::reduce(
                cells.n,
                WorkOfCandidates{},
                [=] HD(long i) {
                    WorkOfCandidates work;
                    for (int c = 0; c < TOP_STEP_CANDIDATES; c++) {
                        const int bin           = bin_for_step(candidates.length[c], cfl_step[cells[(int)i]]);
                        work.half_cell_steps[c] = 2 * (1ll << bin) + 1;
                    }
                    return work;
                },
                [] HD(const WorkOfCandidates& a, const WorkOfCandidates& b) {
                    WorkOfCandidates sum;
                    for (int c = 0; c < TOP_STEP_CANDIDATES; c++)
                        sum.half_cell_steps[c] = a.half_cell_steps[c] + b.half_cell_steps[c];
                    return sum;
                });
        }

        // the coarsest bin whose step fits into step; FINEST_BIN + 1 for none, also for a step that is not a number
        HD int bin_for_step(double top_length, double step) {
            int bin = 0;
            while (bin <= FINEST_BIN && !(ldexp(top_length, -bin) <= step))
                bin++;
            return bin;
        }

        HD int largest_need_of_face_neighbours(const mesh::MeshPointers& mesh, const int* needed_bin, int cell) {
            const double negligible = mesh::negligible_area_of(mesh, cell);
            int          largest    = 0;
            for (int face = mesh.first_face[cell]; face < mesh.first_face[cell] + mesh.n_faces[cell]; face++) {
                if (mesh.area[face] <= negligible) continue;
                const int need = needed_bin[mesh.neighbour[face]];
                largest        = (need > largest) ? need : largest;
            }
            return largest;
        }

        // the bin of the need, or finer until the tick is a multiple of its step
        HD int bin_starting_at(int tick, int needed_bin) {
            int bin = needed_bin;
            while (tick % ticks_of_bin(bin) != 0)
                bin++;
            return bin;
        }

        // the run stops if a CFL step is below the finest bin or not a number; collective
        void needs_from_cfl_steps(const mesh::CellList& active,
                                  const double*         cfl_step,
                                  double                top_length,
                                  CellSteps&            steps) {
            int* needed_bin = steps.needed_bin.data();
            device::parallel_for(active.n, [=] HD(long i) {
                const int cell   = active[(int)i];
                needed_bin[cell] = bin_for_step(top_length, cfl_step[cell]);
            });

            const long long too_fine = mpi::sum_over_ranks(
                device::sum(active.n, [=] HD(long i) { return (long long)(needed_bin[active[(int)i]] > FINEST_BIN); }));
            if (too_fine > 0) {
                logging::stop_run(
                    "TIMESTEPS: %lld cells have a CFL step below the top step of %g / 2^%d, or one that is "
                    "not a number; the run stops",
                    too_fine,
                    top_length,
                    FINEST_BIN);
            }
        }

        // every need at least a face neighbour's need minus one, sweep after sweep until none changes; collective
        void limit_needs_by_face_neighbours(const mesh::CellList&     active,
                                            const mesh::MeshPointers& mesh,
                                            ghosts::Ghosts&           ghosts,
                                            device::Array<int>&       next_need,
                                            CellSteps&                steps) {
            next_need.grow_to_at_least(active.n);
            int* next       = next_need.data();
            int* needed_bin = steps.needed_bin.data();

            long long n_changed = 1;
            while (n_changed > 0) {
                ghosts.copy_changed_to_ghost_cells(
                    mesh.n_cells, active.cell, active.n, [&](auto function) { function(steps.needed_bin); });
                device::parallel_for(active.n, [=] HD(long i) {
                    const int cell    = active[(int)i];
                    const int limited = largest_need_of_face_neighbours(mesh, needed_bin, cell) - 1;
                    next[i]           = (limited > needed_bin[cell]) ? limited : needed_bin[cell];
                });
                n_changed = mpi::sum_over_ranks(device::sum(
                    active.n, [=] HD(long i) { return (long long)(next[i] != needed_bin[active[(int)i]]); }));
                device::parallel_for(active.n, [=] HD(long i) { needed_bin[active[(int)i]] = next[i]; });
            }
        }

        void steps_from_needs(const mesh::CellList& active, int tick, CellSteps& steps) {
            int*       bin         = steps.bin.data();
            int*       start       = steps.start.data();
            int*       planned_end = steps.planned_end.data();
            int*       end         = steps.end.data();
            const int* needed_bin  = steps.needed_bin.data();
            device::parallel_for(active.n, [=] HD(long i) {
                const int cell    = active[(int)i];
                bin[cell]         = bin_starting_at(tick, needed_bin[cell]);
                start[cell]       = tick;
                planned_end[cell] = tick + ticks_of_bin(bin[cell]);
                end[cell]         = planned_end[cell];
            });
        }

        // with individual timesteps also a line above it and the cells of every bin; collective
        void print_top_step(const TopStep&   top,
                            const Sync&      sync,
                            int              step,
                            double           seconds_left,
                            const CellSteps& steps,
                            int              n_cells) {
            const std::string eta = logging::hours_minutes_seconds(seconds_left);
            if constexpr (config::individual_timesteps) {
                logging::print("\n%s\n", LINE_BETWEEN_TOP_STEPS);
                logging::print("SIM: Top-Step %d  t = %g  dt = %g  ETA = %s",
                               step,
                               sync.time,
                               top.length_of_this_top_step(),
                               eta.c_str());
                top.print_cells_per_bin(steps, n_cells);
            } else {
                logging::print("\nSIM: Step %d  t = %g  dt = %g  ETA = %s",
                               step,
                               sync.time,
                               top.length_of_this_top_step(),
                               eta.c_str());
            }
        }

        // the coarsest and finest bin of the cells of the list, over all ranks; collective
        BinRange bins_of(const mesh::CellList& list, const CellSteps& steps) {
            if constexpr (!config::individual_timesteps) return BinRange{0, 0};

            const int* bin     = steps.bin.data();
            const int coarsest = mpi::min_over_ranks(device::min(list.n, [=] HD(long i) { return bin[list[(int)i]]; }));
            const int finest   = mpi::max_over_ranks(device::max(list.n, [=] HD(long i) { return bin[list[(int)i]]; }));
            return BinRange{coarsest, finest};
        }

        // counts the faces of the list that get a flux in this half, each from both sides; collective
        void print_fluxes(const mesh::MeshPointers& mesh,
                          const StepPointers&       at_sync,
                          Half                      half,
                          const mesh::CellList&     with_faces,
                          const BinRange&           bins) {
            const long long here     = device::sum(with_faces.n, [=] HD(long i) {
                const int cell  = with_faces[(int)i];
                const int first = mesh.first_face[cell];
                long long n     = 0;
                for (int face = first; face < first + mesh.n_faces[cell]; face++) {
                    if (face_share(at_sync, cell, mesh.neighbour[face], half).time != 0.0) n++;
                }
                return n;
            });
            const long long n_fluxes = mpi::sum_over_ranks(here);
            const char*     label    = (half == Half::first) ? "START" : "FINISH";
            if constexpr (config::individual_timesteps) {
                logging::print_labelled(label, "%lld fluxes (bin %s)", n_fluxes, text_of_bins(bins).c_str());
            } else {
                logging::print_labelled(label, "%lld fluxes", n_fluxes);
            }
        }

        // "2" for one bin, "0-2" for several
        std::string text_of_bins(const BinRange& bins) {
            if (bins.coarsest == bins.finest) return std::to_string(bins.finest);
            return std::to_string(bins.coarsest) + "-" + std::to_string(bins.finest);
        }

    } // namespace

} // namespace timesteps
