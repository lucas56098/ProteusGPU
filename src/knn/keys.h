#ifndef KNN_KEYS_H
#define KNN_KEYS_H
#pragma once

// The key grid: one integer grid per axis over [-0.5, 1.5) for the whole run, and the Morton key on it.
// Every cube of the unit box with side 1/2 or less is a cell of this grid, so a Peano-Hilbert key on
// the same integers splits the box into cells that are each one run of the Morton order.

#include "global/allvars.h"
#include <cmath>
#include <cstdint>

namespace knn {

#ifdef dim_2D
    constexpr int KEY_BITS = 31; // per axis
#else
    constexpr int KEY_BITS = 21;
#endif
    constexpr int          KEY_TOTAL_BITS = DIMENSION * KEY_BITS;
    constexpr double       KEY_SCALE      = (double)(1ull << (KEY_BITS - 1)); // key cells per box length
    constexpr double       KEY_OFFSET     = (double)(1ull << (KEY_BITS - 2)); // x = -0.5 is key coordinate 0
    constexpr unsigned int KEY_MAX        = (unsigned int)((1ull << KEY_BITS) - 1);

    // key grid coordinate on one axis; the scale is a power of two, so only floor rounds
    HD inline unsigned int key_coord(double x) {
        const double c = floor(x * KEY_SCALE) + KEY_OFFSET;
        if (!(c >= 0.0)) return 0;
        if (c >= (double)KEY_MAX) return KEY_MAX;
        return (unsigned int)c;
    }

    // puts DIMENSION - 1 zero bits between the bits of v
    HD inline uint64_t spread_bits(unsigned int v) {
        uint64_t x = v;
#ifdef dim_2D
        x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
        x = (x | (x << 2)) & 0x3333333333333333ull;
        x = (x | (x << 1)) & 0x5555555555555555ull;
#else
        x &= 0x1FFFFFull;
        x = (x | (x << 32)) & 0x001F00000000FFFFull;
        x = (x | (x << 16)) & 0x001F0000FF0000FFull;
        x = (x | (x << 8)) & 0x100F00F00F00F00Full;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3ull;
        x = (x | (x << 2)) & 0x1249249249249249ull;
#endif
        return x;
    }

    // Morton key of a point, the bits of the axes interleaved
    HD inline uint64_t morton_key(const POINT_TYPE& p) {
#ifdef dim_2D
        return spread_bits(key_coord(p.x)) | (spread_bits(key_coord(p.y)) << 1);
#else
        return spread_bits(key_coord(p.x)) | (spread_bits(key_coord(p.y)) << 1) | (spread_bits(key_coord(p.z)) << 2);
#endif
    }

} // namespace knn

#endif
