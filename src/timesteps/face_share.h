#pragma once

// the share of a face at a sync, and the time since the start of a step (included by timesteps.h)

namespace timesteps {

    // a face is planned open p ticks, from the later start to the earlier planned end; it gets p / 2 when it opens
    // and when it closes the rest of the d ticks it was open, d - p / 2, never below 0
    HD inline FaceShare face_share(const StepPointers& steps, int cell, int neighbour, Half half) {
        if constexpr (!config::individual_timesteps) {
            const int    ticks_open          = TICKS_PER_TOP_STEP;
            const int    ticks_of_half       = (half == Half::first) ? ticks_open / 2 : ticks_open - ticks_open / 2;
            const double since_start_of_both = since_start(steps, cell);
            return FaceShare{(double)ticks_of_half * steps.tick_length, since_start_of_both, since_start_of_both};
        } else {
            const int    start_of_cell      = steps.start[cell];
            const int    start_of_neighbour = steps.start[neighbour];
            const int    opened             = (start_of_cell > start_of_neighbour) ? start_of_cell : start_of_neighbour;
            const int    planned_close      = (steps.planned_end[cell] < steps.planned_end[neighbour])
                                                  ? steps.planned_end[cell]
                                                  : steps.planned_end[neighbour];
            const double half_of_plan       = 0.5 * (double)(planned_close - opened);

            double ticks = 0.0;
            if (half == Half::first && (start_of_cell == steps.tick || start_of_neighbour == steps.tick)) {
                ticks = half_of_plan;
            }
            if (half == Half::second && (steps.end[cell] == steps.tick || steps.end[neighbour] == steps.tick)) {
                ticks = fmax((double)(steps.tick - opened) - half_of_plan, 0.0);
            }
            return FaceShare{ticks * steps.tick_length, since_start(steps, cell), since_start(steps, neighbour)};
        }
    }

    HD inline double since_start(const StepPointers& steps, int cell) {
        if constexpr (!config::individual_timesteps) {
            return (double)steps.tick * steps.tick_length;
        } else {
            return (double)(steps.tick - steps.start[cell]) * steps.tick_length;
        }
    }

    HD inline double planned_length_of_step(const StepPointers& steps, int cell) {
        if constexpr (!config::individual_timesteps) {
            return (double)TICKS_PER_TOP_STEP * steps.tick_length;
        } else {
            return (double)(steps.planned_end[cell] - steps.start[cell]) * steps.tick_length;
        }
    }

    HD inline int start_of_step(const StepPointers& steps, int cell) {
        if constexpr (!config::individual_timesteps) {
            return 0;
        } else {
            return steps.start[cell];
        }
    }

} // namespace timesteps
