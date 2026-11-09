#pragma once

// the nearest neighbours, then the walk over the tree that finds the points they missed (included by voronoi_cell.h)

#include "cell.h"

namespace voronoi_cell {

    // the walk counts a box as inside the last neighbour only if it is this share nearer, against rounding
    constexpr double ROUNDING_OF_A_BOX = 1e-10;

    template <int K, int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void cut_by_points_the_walk_finds(Cell<MAX_PLANES, MAX_CORNERS, EXACT>& cell,
                                         const ngbtree::TreePointers&          tree,
                                         int                                   seed,
                                         const int*                            neighbours);
    HD inline bool is_among(int point, const int* list, int n);

    // the neighbours cut the cell nearest first, until the next one lies outside the security radius
    template <int K, bool WALK, int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void build_cell(const ngbtree::TreePointers&          tree,
                       int                                   seed,
                       Cell<MAX_PLANES, MAX_CORNERS, EXACT>& cell,
                       Handover&                             handover) {
        static_assert(WALK || K <= 64, "a handover has one bit per neighbour");
        int neighbours[K];
        ngbtree::nearest_neighbours<K>(tree, seed, neighbours);

        // the faces of the handover stand for all its neighbours, which are the nearest of these
        int first = 0;
        if constexpr (WALK) {
            first = (handover.n_neighbours < K) ? handover.n_neighbours : K;
            for (int k = 0; k < first; k++) {
                if (!((handover.faces >> k) & 1)) continue;
                cell.cut_by(neighbours[k]);
                if (cell.status() != Status::ok) return;
            }
        }

        // -1 after the last neighbour: the tree has fewer than K other points, and all of them have cut
        for (int k = first; k < K; k++) {
            if (neighbours[k] < 0 || cell.is_finished_before(neighbours[k])) return;
            cell.cut_by(neighbours[k]);
            if (cell.status() != Status::ok) return;
        }

        // every other point lies at least as far as the last neighbour
        if (cell.is_finished_before(neighbours[K - 1])) return;
        if constexpr (WALK) {
            cut_by_points_the_walk_finds<K>(cell, tree, seed, neighbours);
        } else {
            cell.mark_as_needing_more_points();
            handover = Handover{cell.faces_among(neighbours, K), K};
        }
    }

    // cuts by the points the K neighbours missed: those in boxes that reach into a corner sphere, with the margin
    template <int K, int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void cut_by_points_the_walk_finds(Cell<MAX_PLANES, MAX_CORNERS, EXACT>& cell,
                                         const ngbtree::TreePointers&          tree,
                                         int                                   seed,
                                         const int*                            neighbours) {
        // squared distances as nearest_neighbours computes them
        const Vec    at     = tree.positions[seed];
        const double last   = norm2(at - tree.positions[neighbours[K - 1]]);
        const double inside = last * (1.0 - ROUNDING_OF_A_BOX);

        Vec    centre[MAX_CORNERS];
        double squared_radius[MAX_CORNERS];
        int    n_spheres = cell.corner_spheres(centre, squared_radius, inside);

        const auto box_reaches_into_a_sphere = [&](const Vec& box_min, const Vec& box_max) {
            if (cell.status() != Status::ok) return false;
            if (ngbtree::farthest_squared_distance_to_box(at, box_min, box_max) < inside) return false;
            for (int s = 0; s < n_spheres; s++) {
                const double squared_distance = ngbtree::squared_distance_to_box(centre[s], box_min, box_max);
                // a sphere the doubles could not place lets every box in
                if (!(squared_distance > SECURITY_MARGIN * squared_radius[s])) return true;
            }
            return false;
        };
        const auto cut_by_new_point = [&](int point) {
            if (point == seed) return;
            const double distance = norm2(at - tree.positions[point]);
            if (distance < last || (distance == last && is_among(point, neighbours, K))) return;
            cell.cut_by(point);
            n_spheres = cell.corner_spheres(centre, squared_radius, inside);
        };
        ngbtree::walk_nearest_first(tree, at, box_reaches_into_a_sphere, cut_by_new_point);
    }

    // a point inside the sphere of a corner is closer to it than the seed, so its plane cuts the corner away
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD int Cell<MAX_PLANES, MAX_CORNERS, EXACT>::corner_spheres(Vec*    centre,
                                                                double* squared_radius,
                                                                double  inside) const {
        int n = 0;
        for (int c = 0; c < n_corners; c++) {
            const UndividedPosition corner = position_of(corners[c]);
            centre[n]                      = corner.numerator / corner.denominator;
            squared_radius[n]              = norm2(centre[n] - seed_position);
            // with the margin a sphere reaches at most twice its radius from the seed
            if (!(4.0 * SECURITY_MARGIN * squared_radius[n] < inside)) n++;
        }
        return n;
    }

    HD inline bool is_among(int point, const int* list, int n) {
        for (int i = 0; i < n; i++) {
            if (list[i] == point) return true;
        }
        return false;
    }

} // namespace voronoi_cell
