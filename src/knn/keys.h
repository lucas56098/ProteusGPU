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

    // the unit box is the middle half of the key grid; its own grid has one bit less per axis
    constexpr int      DOMAIN_BITS     = KEY_BITS - 1;
    constexpr int      DOMAIN_KEY_BITS = DIMENSION * DOMAIN_BITS;
    constexpr uint64_t DOMAIN_KEY_END  = 1ull << DOMAIN_KEY_BITS; // one past the last Hilbert key

    // coordinate of x on the grid of the unit box, clamped to it
    HD inline unsigned int domain_coord(double x) {
        const unsigned int c  = key_coord(x);
        const unsigned int lo = (unsigned int)KEY_OFFSET;
        const unsigned int n  = 1u << DOMAIN_BITS;
        if (c < lo) return 0;
        return (c - lo >= n) ? n - 1 : c - lo;
    }

    // Hilbert index of grid cell X with bits per axis (Skilling 2004); X is overwritten.
    // The first DIMENSION * L bits of it are the index of the cube of side 2^-L that holds the cell.
    HD inline uint64_t hilbert_index(unsigned int* X, int bits) {
        for (unsigned int Q = (bits > 0) ? (1u << (bits - 1)) : 0u; Q > 1; Q >>= 1) {
            const unsigned int P = Q - 1;
            for (int i = 0; i < DIMENSION; i++) {
                if (X[i] & Q) {
                    X[0] ^= P;
                } else {
                    const unsigned int t = (X[0] ^ X[i]) & P;
                    X[0] ^= t;
                    X[i] ^= t;
                }
            }
        }
        for (int i = 1; i < DIMENSION; i++)
            X[i] ^= X[i - 1];
        unsigned int t = 0;
        for (unsigned int Q = (bits > 0) ? (1u << (bits - 1)) : 0u; Q > 1; Q >>= 1) {
            if (X[DIMENSION - 1] & Q) t ^= Q - 1;
        }
        for (int i = 0; i < DIMENSION; i++)
            X[i] ^= t;

        uint64_t h = 0;
        for (int b = bits - 1; b >= 0; b--) {
            for (int i = 0; i < DIMENSION; i++)
                h = (h << 1) | ((X[i] >> b) & 1u);
        }
        return h;
    }

    // Hilbert key of a point of the unit box, the domain decomposition orders cells by it
    HD inline uint64_t hilbert_key(const POINT_TYPE& p) {
        unsigned int X[DIMENSION];
        X[0] = domain_coord(p.x);
        X[1] = domain_coord(p.y);
#ifdef dim_3D
        X[2] = domain_coord(p.z);
#endif
        return hilbert_index(X, DOMAIN_BITS);
    }

} // namespace knn

#endif
