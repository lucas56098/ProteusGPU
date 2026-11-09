#pragma once

// Where a seed is at a given tick, from its position and velocity at the start of its step.

#include "math/math.h"

namespace seed {

    struct Seed {
        Vec position_at_start;
        Vec velocity;
        int start_tick;
    };

    HD inline Vec position_at_tick(const Seed& seed, int tick, double tick_length) {
        const double since_start = (double)(tick - seed.start_tick) * tick_length;
        return rounded_to_box_precision(seed.position_at_start + since_start * seed.velocity);
    }

} // namespace seed
