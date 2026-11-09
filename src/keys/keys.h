#pragma once

// Morton keys on a grid over [-0.5, 1.5), Hilbert keys on its [0, 1) part with one bit less per axis.

#include "math/math.h"

#include <cstdint>

namespace keys {

    // for points in [-0.5, 1.5) (box + ghosts), used in the tree and for the order of the cells
    HD inline uint64_t morton_key(const Vec& p);

    // for points in [0, 1) (box only), used in domain decomp
    HD inline uint64_t hilbert_key(const Vec& p);

    constexpr int HILBERT_LEVELS = (config::dim == 2) ? 30 : 20;

    // first and last Hilbert key of the cube of side 2^-level at corner * 2^-level
    HD inline void hilbert_keys_of_cube(int level, const unsigned int* corner, uint64_t* first, uint64_t* last);

} // namespace keys

#include "hilbert.h"
#include "morton.h"
