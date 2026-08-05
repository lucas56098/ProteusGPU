#ifndef ASTRO_SOURCES_H
#define ASTRO_SOURCES_H
#pragma once

// Strang-split source terms (gravity, cooling, feedback) wrapped around the hydro step. Each
// source is gated by its Config.sh toggle; with none set these are no-ops.

namespace astro {

    void sources_init();

    // first half before hydro_step, second half after, each over dt_half. Second half runs the
    // sources in reversed order to keep the split symmetric (2nd order) with multiple sources.
    void apply_sources_first_half(double dt_half);
    void apply_sources_second_half(double dt_half);

} // namespace astro

#endif // ASTRO_SOURCES_H
