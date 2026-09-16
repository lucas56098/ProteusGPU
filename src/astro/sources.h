#ifndef ASTRO_SOURCES_H
#define ASTRO_SOURCES_H
#pragma once

// The source terms, wrapped around the hydro step; each one needs its own Config.sh flag.

namespace astro {

    void sources_init();
    void sources_prepare();

    // half a step before and half a step after the hydro, the second half in reverse order
    void apply_sources_first_half(double dt_half);
    void apply_sources_second_half(double dt_half);

} // namespace astro

#endif
