#pragma once

// Who steps when: the top step and its ticks, the step of every cell, the syncs, and the shares of the faces.

#include "config/config.h"
#include "device/device.h"
#include "ghosts/ghosts.h"
#include "mesh/mesh.h"

namespace timesteps {

    struct StepPointers;

    // a top step has 2^FINEST_BIN ticks; a step of bin b lasts 2^(FINEST_BIN - b) of them
    constexpr int FINEST_BIN         = 29;
    constexpr int TICKS_PER_TOP_STEP = 1 << FINEST_BIN;

    // a top step begins with this many finer bins free, so that a cell whose CFL step shrinks can still go finer
    constexpr int BINS_KEPT_FREE = 3;

    // the line in the log between two top steps
    constexpr const char* LINE_BETWEEN_TOP_STEPS =
        "==========================================================================";

    // a tick where at least one step ends; at the end of a top step begin_steps moves it to tick 0 of the next
    struct Sync {
        int    tick; // in the current top step
        double time;
        bool   ends_top_step; // every step ends here
    };

    // a face gets the first half of its flux when it opens, the second when it closes
    enum class Half { first, second };

    // the time the sources of a cell get in one half of its step: of_cell[c] with individual timesteps, else every
    struct SourceTimes {
        const double* of_cell;
        double        every;
        HD double operator[](int cell) const { return of_cell ? of_cell[cell] : every; }
    };

    struct FaceShare {
        double time; // the time its flux counts for, never below 0; 0: no flux at this sync
        double since_start_of_cell; // how far each side is predicted: the time since its step began
        double since_start_of_neighbour;
    };

    // the step of every cell and ghost cell; with one bin the arrays stay empty
    struct CellSteps {
        device::Array<int> bin{"cells: bins of the steps", 0}; // 0: the whole top step; b: 2^-b of it
        device::Array<int> start{"cells: starts of the steps", 0}; // the tick it began at
        device::Array<int> planned_end{"cells: planned ends of the steps", 0}; // its faces opened with this plan
        device::Array<int> end{"cells: ends of the steps", 0}; // the planned end, or the sync a source ended it at
        device::Array<int> needed_bin{"cells: needed bins", 0}; // what its CFL step and its face neighbours ask for
    };

    // calls function(array) for every array of the steps; with one bin for none
    template <typename Function> void for_each_field(CellSteps& steps, Function function) {
        if constexpr (config::individual_timesteps) {
            function(steps.bin);
            function(steps.start);
            function(steps.planned_end);
            function(steps.end);
            function(steps.needed_bin);
        }
    }

    // the coarsest and finest bin of some cells
    struct BinRange {
        int coarsest;
        int finest;
    };

    // a sync that ends a top step, where every cell begins its first step
    Sync first_sync(double start_time);

    // the top step the run is in, and the work of the syncs
    class TopStep {
      public:
        // START: a new top step if one ended, then the bins and steps of the active cells; collective
        void begin_steps(const mesh::CellList&     active,
                         const double*             cfl_step,
                         double                    stop_time,
                         const mesh::MeshPointers& mesh,
                         ghosts::Ghosts&           ghosts,
                         Sync&                     sync,
                         CellSteps&                steps);

        // a CFL step at least this long gives the same top step and bins as any longer one
        double longest_step_that_matters(const Sync& sync, double stop_time) const;

        double length_of_this_top_step() const { return length; }

        // TIME: the next sync, the earliest end of a step on any rank; collective
        Sync next_sync(const CellSteps& steps, int n_cells) const;

        // every cell at the end of a top step, else those whose step ends at the sync; holds until the next call
        mesh::CellList active_cells(const Sync& sync, const mesh::MeshPointers& mesh, const CellSteps& steps);

        // a source switched on: every step longer than its new CFL step ends at this sync; collective
        void end_steps_longer_than(const double*             cfl_step,
                                   const Sync&               sync,
                                   const mesh::MeshPointers& mesh,
                                   ghosts::Ghosts&           ghosts,
                                   CellSteps&                steps);

        // the active cells, their face neighbours, the cells next to an active ghost cell; holds until the next call
        mesh::CellList cells_with_faces_at_sync(const Sync&               sync,
                                                const mesh::CellList&     active,
                                                const mesh::MeshPointers& mesh,
                                                const CellSteps&          steps);

        // keeps the fastest mesh speed of the top step, over all ranks; collective
        void remember_fastest_mesh_speed(const mesh::CellList& active, const Vec* mesh_velocity);

        // how far a seed may have moved since the start of the top step
        double drift_margin(const Sync& sync) const;

        // time the sources of the active cells get in this half; source_clock[c] is the tick they ran up to
        SourceTimes times_of_sources(Half                  half,
                                     const mesh::CellList& active,
                                     const CellSteps&      steps,
                                     double*               source_clock);

        // what kernels need at a sync
        StepPointers pointers(const Sync& sync, const CellSteps& steps) const;

        // START in the log: a new top step or an empty line, then the first halves of the fluxes; collective
        void print_begin_of_steps(const Sync&               sync,
                                  int                       step,
                                  double                    seconds_left,
                                  const mesh::MeshPointers& mesh,
                                  const mesh::CellList&     active,
                                  const CellSteps&          steps);

        // TIME in the log: where the sync lies in the top step, and its active cells; collective
        void print_sync(const Sync& sync, const mesh::CellList& active, const CellSteps& steps);

        // FINISH in the log: the second halves of the fluxes
        void print_end_of_steps() const;

        // one line in the log with the cells of every bin, over all ranks; collective
        void print_cells_per_bin(const CellSteps& steps, int n_cells) const;

        // one line in the log with the syncs and the mean steps per cell, over all ranks; collective
        void print_syncs_and_steps(int n_cells) const;

      private:
        double start_time  = 0.0;
        double end_time    = 0.0; // exactly stop_time when the top step was cut there
        double length      = 0.0;
        double tick_length = 0.0; // the length / 2^FINEST_BIN

        double fastest_mesh_speed = 0.0; // over all ranks

        // what this rank did in the top step so far, for the log
        int       n_syncs      = 0;
        long long n_cell_steps = 0;

        // of the active cells at the last START and TIME, for the log
        BinRange bins_that_begin{0, 0};
        BinRange bins_that_end{0, 0};

        device::Array<int>  active{"timesteps: active cells", 0};
        device::Array<int>  with_faces{"timesteps: cells whose faces open or close", 0};
        device::Array<bool> has_faces{"timesteps: marks of the cells whose faces open or close", 0};
        device::Array<int>  next_need{"timesteps: needs of the next sweep", 0};

        device::Array<double> source_time{"timesteps: times of the sources in a half", 0};
    };

    // work of every cell in the top step, in half cell steps: the steps of its last bin, and half a step for the build
    void work_of_cells(const CellSteps& steps, int n_cells, long* work);

    // share of the face between cell and neighbour in this half of its flux
    HD inline FaceShare face_share(const StepPointers& steps, int cell, int neighbour, Half half);

    // time since the step of a cell or ghost cell began, and the tick it began at
    HD inline double since_start(const StepPointers& steps, int cell);
    HD inline double planned_length_of_step(const StepPointers& steps, int cell);
    HD inline int start_of_step(const StepPointers& steps, int cell);

    // plain values for kernels; the arrays are null with one bin
    struct StepPointers {
        int        tick;
        double     tick_length;
        const int* start;
        const int* planned_end;
        const int* end;
    };

} // namespace timesteps

#include "face_share.h"
