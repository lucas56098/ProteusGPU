#pragma once

// how the Hilbert keys are computed (included by keys.h)

#include "morton.h"

namespace keys {

    HD inline uint64_t hilbert_index(unsigned int cell[config::dim], int bits);
    HD inline unsigned int box_cell(double x);

    // the box [0, 1) is the middle half of the grid, so one bit less per axis
    constexpr int          BOX_BITS      = GRID_BITS - 1;
    constexpr unsigned int LAST_BOX_CELL = (1u << BOX_BITS) - 1;
    static_assert(BOX_BITS == HILBERT_LEVELS, "a cube of the last level is one cell of the grid");

    HD inline uint64_t hilbert_key(const Vec& p) {
        unsigned int cell[config::dim];
        for (int d = 0; d < config::dim; d++)
            cell[d] = box_cell(p[d]);
        return hilbert_index(cell, BOX_BITS);
    }

    HD inline void hilbert_keys_of_cube(int level, const unsigned int* corner, uint64_t* first, uint64_t* last) {
        const int    unused_bits = config::dim * (HILBERT_LEVELS - level);
        unsigned int cell[config::dim];
        for (int d = 0; d < config::dim; d++)
            cell[d] = corner[d];
        *first = (level == 0) ? 0 : hilbert_index(cell, level) << unused_bits;
        *last  = *first + ((1ull << unused_bits) - 1);
    }

    // Hilbert index of a cell with the given bits per axis, after Skilling (2004); changes cell
    HD inline uint64_t hilbert_index(unsigned int cell[config::dim], int bits) {
        const unsigned int highest_bit = 1u << (bits - 1);

        // Skilling's AxesToTranspose: from the highest bit down, invert or exchange the lower bits
        for (unsigned int bit = highest_bit; bit > 1; bit >>= 1) {
            const unsigned int lower_bits = bit - 1;
            for (int d = 0; d < config::dim; d++) {
                if (cell[d] & bit) {
                    cell[0] ^= lower_bits;
                } else {
                    const unsigned int exchange = (cell[0] ^ cell[d]) & lower_bits;
                    cell[0] ^= exchange;
                    cell[d] ^= exchange;
                }
            }
        }

        // Gray code
        for (int d = 1; d < config::dim; d++)
            cell[d] ^= cell[d - 1];
        unsigned int flip = 0;
        for (unsigned int bit = highest_bit; bit > 1; bit >>= 1)
            if (cell[config::dim - 1] & bit) flip ^= bit - 1;
        for (int d = 0; d < config::dim; d++)
            cell[d] ^= flip;

        uint64_t index = 0;
        for (int b = bits - 1; b >= 0; b--)
            for (int d = 0; d < config::dim; d++)
                index = (index << 1) | ((cell[d] >> b) & 1u);
        return index;
    }

    // cell of x on the grid of the box; the edge cell outside [0, 1)
    HD inline unsigned int box_cell(double x) {
        const unsigned int cell = grid_cell(x);
        if (cell < CELLS_BELOW_BOX) return 0;
        return (cell - CELLS_BELOW_BOX > LAST_BOX_CELL) ? LAST_BOX_CELL : cell - CELLS_BELOW_BOX;
    }

} // namespace keys
