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

    // room_in_own_cube is taken a little smaller, so that its rounding never lets a ball skip the walk that would touch
    // another rank
    constexpr double ROOM_SAFETY = 1e-10;
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

    // a ball that reaches past an edge of the box holds periodic copies, which no rank holds as its own cells; most
    // balls lie well inside a cube of this rank, which needs no walk
    HD inline bool ball_is_inside_this_rank(const DomainPointers& domain, const Vec& center, double radius) {
        for (int d = 0; d < config::dim; d++) {
            if (center[d] - radius < 0.0 || center[d] + radius >= 1.0) return false;
        }
        if (radius < (1.0 - ROOM_SAFETY) * room_in_own_cube(domain, center)) return true;
        bool only_this_rank = true;
        for_each_rank_touched_by_ball(domain, center, radius, [&](int rank) {
            if (rank != domain.this_rank) only_this_rank = false;
        });
        return only_this_rank;
    }

    // the cube of a level around a point holds the keys that share the top bits of the key of the point; the levels go
    // from the whole box down, so the first cube with only keys of this rank is the largest
    HD inline double room_in_own_cube(const DomainPointers& domain, const Vec& position) {
        const uint64_t first_here = domain.cuts[domain.this_rank];
        const uint64_t end_here   = domain.cuts[domain.this_rank + 1];
        const uint64_t key        = keys::hilbert_key(position);
        if (key < first_here || key >= end_here) return 0.0;
        for (int level = 0; level <= keys::HILBERT_LEVELS; level++) {
            const int      unused = config::dim * (keys::HILBERT_LEVELS - level);
            const uint64_t first  = (key >> unused) << unused;
            const uint64_t last   = first + ((1ull << unused) - 1);
            if (first < first_here || last >= end_here) continue;

            const double side = 1.0 / (double)(1ull << level);
            double       room = INFINITY;
            for (int d = 0; d < config::dim; d++) {
                const double low = (double)(keys::box_cell(position[d]) >> (keys::HILBERT_LEVELS - level)) * side;
                room             = fmin(room, fmin(position[d] - low, low + side - position[d]));
            }
            return fmax(room, 0.0);
        }
        return 0.0;
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
