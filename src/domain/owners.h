#pragma once

// how the owner of a point and the ranks a ball touches are found (included by domain.h)

#include "keys/keys.h"

namespace domain {

    // a cube of the Hilbert curve: side 2^-level, lower corner at corner[d] * 2^-level; it holds one stretch of keys
    struct Cube {
        int          level;
        unsigned int corner[config::dim];
    };

    // a split leaves all but one child waiting, and the last level does not split; that bounds the cubes of a walk
    constexpr int CHILDREN_OF_CUBE  = 1 << config::dim;
    constexpr int MAX_WAITING_CUBES = (CHILDREN_OF_CUBE - 1) * keys::HILBERT_LEVELS + 1;

    HD inline int owner_of_key(const DomainPointers& domain, uint64_t key);
    HD inline double squared_distance_to_cube(const Cube& cube, const Vec& position);
    HD inline Cube child_of(const Cube& cube, int child);

    HD inline int owner_of(const DomainPointers& domain, const Vec& position) {
        return owner_of_key(domain, keys::hilbert_key(position));
    }

    // from the whole box down: a cube the ball reaches gives its rank if all its keys have one owner, else it splits
    template <typename Function>
    HD void for_each_rank_touched_by_ball(const DomainPointers& domain,
                                          const Vec&            center,
                                          double                radius,
                                          Function              function) {
        Cube waiting[MAX_WAITING_CUBES];
        int  n_waiting       = 0;
        waiting[n_waiting++] = Cube{0, {}};
        while (n_waiting > 0) {
            const Cube cube = waiting[--n_waiting];
            if (squared_distance_to_cube(cube, center) > radius * radius) continue;

            uint64_t first, last;
            keys::hilbert_keys_of_cube(cube.level, cube.corner, &first, &last);
            const int owner = owner_of_key(domain, first);
            if (owner == owner_of_key(domain, last)) {
                function(owner);
                continue;
            }
            for (int child = 0; child < CHILDREN_OF_CUBE; child++)
                waiting[n_waiting++] = child_of(cube, child);
        }
    }

    // a ball that reaches past an edge of the box holds periodic copies, which no rank holds as its own cells
    HD inline bool ball_is_inside_this_rank(const DomainPointers& domain, const Vec& center, double radius) {
        for (int d = 0; d < config::dim; d++) {
            if (center[d] - radius < 0.0 || center[d] + radius >= 1.0) return false;
        }
        bool only_this_rank = true;
        for_each_rank_touched_by_ball(domain, center, radius, [&](int rank) {
            if (rank != domain.this_rank) only_this_rank = false;
        });
        return only_this_rank;
    }

    // the last rank whose cut lies at or below key; a rank without keys owns nothing
    HD inline int owner_of_key(const DomainPointers& domain, uint64_t key) {
        int low  = 0;
        int high = domain.n_ranks;
        while (high - low > 1) {
            const int middle = (low + high) / 2;
            if (domain.cuts[middle] <= key) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return low;
    }

    // zero inside the cube; its walls are exact, as its side is a power of two
    HD inline double squared_distance_to_cube(const Cube& cube, const Vec& position) {
        const double side = 1.0 / (double)(1ull << cube.level);
        Vec          gap{};
        for (int d = 0; d < config::dim; d++) {
            const double low  = cube.corner[d] * side;
            const double high = low + side;
            if (position[d] < low) gap[d] = low - position[d];
            if (position[d] > high) gap[d] = position[d] - high;
        }
        return norm2(gap);
    }

    // bit d of child is the step of the child along axis d
    HD inline Cube child_of(const Cube& cube, int child) {
        Cube result{};
        result.level = cube.level + 1;
        for (int d = 0; d < config::dim; d++)
            result.corner[d] = 2 * cube.corner[d] + ((child >> d) & 1);
        return result;
    }

} // namespace domain
