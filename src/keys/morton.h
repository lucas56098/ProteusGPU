#pragma once

// how the Morton keys and the grid of both keys are computed (included by keys.h)

#include <cmath>

namespace keys {

    HD inline uint64_t spread_bits(unsigned int v);
    HD inline unsigned int grid_cell(double x);

    // bits per axis, so that all axes fit into 64 bits
    constexpr int GRID_BITS = (config::dim == 2) ? 31 : 21;

    // the grid covers [-0.5, 1.5): one box length is half the grid, the cells below 0 come first
    constexpr double       CELLS_PER_BOX_LENGTH = (double)(1ull << (GRID_BITS - 1));
    constexpr unsigned int CELLS_BELOW_BOX      = 1u << (GRID_BITS - 2);
    constexpr unsigned int LAST_CELL            = (unsigned int)((1ull << GRID_BITS) - 1);

    // interleaves the bits of all axes, axis 0 lowest
    HD inline uint64_t morton_key(const Vec& p) {
        uint64_t key = 0;
        for (int d = 0; d < config::dim; d++)
            key |= spread_bits(grid_cell(p[d])) << d;
        return key;
    }

    // spreads the bits of v: bit b goes to bit dim * b
    HD inline uint64_t spread_bits(unsigned int v) {
        uint64_t x = v;
        if constexpr (config::dim == 2) {
            x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
            x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
            x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
            x = (x | (x << 2)) & 0x3333333333333333ull;
            x = (x | (x << 1)) & 0x5555555555555555ull;
        } else {
            x &= 0x1FFFFFull;
            x = (x | (x << 32)) & 0x001F00000000FFFFull;
            x = (x | (x << 16)) & 0x001F0000FF0000FFull;
            x = (x | (x << 8)) & 0x100F00F00F00F00Full;
            x = (x | (x << 4)) & 0x10C30C30C30C30C3ull;
            x = (x | (x << 2)) & 0x1249249249249249ull;
        }
        return x;
    }

    // cell of x on the grid; the edge cell outside [-0.5, 1.5)
    HD inline unsigned int grid_cell(double x) {
        const double cell = floor(x * CELLS_PER_BOX_LENGTH) + (double)CELLS_BELOW_BOX;
        if (!(cell >= 0.0)) return 0;
        if (cell >= (double)LAST_CELL) return LAST_CELL;
        return (unsigned int)cell;
    }

} // namespace keys
