#pragma once

// how a Cell is built: the start box, the cuts, the security radius (included by voronoi_cell.h)

#include "exact_tests.h"

#include <cmath>

namespace voronoi_cell {

    // the walls of the start box, a hair past the margin of half a box: at -WALL_LOW and WALL_HIGH on every axis
    constexpr double WALL_LOW  = 0.5 + 1e-14;
    constexpr double WALL_HIGH = 1.5 + 1e-14;

    // rounding bound of a determinant, times the product of its row scales; about 20 roundings, with a wide margin
    constexpr double UNCERTAIN_DETERMINANT = 1e-13;

    // a point exactly at the security radius can still cut in a tie, and corners are rounded
    constexpr double SECURITY_MARGIN = 1.0 + 1e-6;

    // determinant squared / product of squared normals; below this an exact cell computes the corner from the points
    constexpr double WELL_POSED = 1e-8;

    HD inline double det2x2(double a11, double a12, double a21, double a22);
    HD inline double det3x3(double a11,
                            double a12,
                            double a13,
                            double a21,
                            double a22,
                            double a23,
                            double a31,
                            double a32,
                            double a33);
    HD inline void minors_of_det3x3(double  a11,
                                    double  a12,
                                    double  a21,
                                    double  a22,
                                    double  a31,
                                    double  a32,
                                    double* minor);
    HD inline void minors_of_det4x4(double  a11,
                                    double  a12,
                                    double  a13,
                                    double  a21,
                                    double  a22,
                                    double  a23,
                                    double  a31,
                                    double  a32,
                                    double  a33,
                                    double  a41,
                                    double  a42,
                                    double  a43,
                                    double* minor);

    // plane 2 d is the low wall of axis d, 2 d + 1 the high one; every corner lists its walls in the same turning order
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD Cell<MAX_PLANES, MAX_CORNERS, EXACT>::Cell(int seed, const Vec* positions)
        : positions(positions), seed_position(positions[seed]) {
        static_assert(MAX_PLANES > 2 * config::dim, "room for the walls and at least one cut");

        for (int p = 0; p < 2 * config::dim; p++)
            point_of_plane[p] = -1;
        n_planes      = 2 * config::dim;
        n_cut_away    = 0;
        first_in_ring = END_OF_RING;

        if constexpr (config::dim == 2) {
            constexpr int box_corners[4][2] = {{2, 0}, {1, 2}, {3, 1}, {0, 3}};
            for (int c = 0; c < 4; c++) {
                for (int d = 0; d < 2; d++)
                    corners[c].plane[d] = (PlaneIndex)box_corners[c][d];
                fill_corner(corners[c]);
            }
            n_corners = 4;
        } else {
            constexpr int box_corners[8][3] = {
                {2, 5, 0}, {5, 3, 0}, {1, 5, 2}, {5, 1, 3}, {4, 2, 0}, {4, 0, 3}, {2, 4, 1}, {4, 3, 1}};
            for (int c = 0; c < 8; c++) {
                for (int d = 0; d < 3; d++)
                    corners[c].plane[d] = (PlaneIndex)box_corners[c][d];
                fill_corner(corners[c]);
            }
            n_corners = 8;
        }
    }

    // the corners on the far side go, and new ones appear where the new plane meets the ring of planes around them
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::cut_by(int point) {
        const bool farthest_was_known = farthest_corner_known;
        farthest_corner_known         = false;

        const int new_plane = add_plane(point);
        if (current_status != Status::ok) return;

        park_cut_away_corners(new_plane);
        if (current_status != Status::ok) return;
        if (n_cut_away == 0) {
            n_planes--;
            farthest_corner_known = farthest_was_known;
            return;
        }

        link_ring_around_cut_away_corners();
        if (current_status != Status::ok || first_in_ring == END_OF_RING) return;
        add_corners_along_ring(new_plane);
    }

    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD bool Cell<MAX_PLANES, MAX_CORNERS, EXACT>::is_finished_before(int point) const {
        if (!farthest_corner_known) find_farthest_corner();
        const double squared_distance = norm2(positions[point] - seed_position);
        return squared_distance * farthest_denominator > 4.0 * SECURITY_MARGIN * farthest_numerator;
    }

    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD int Cell<MAX_PLANES, MAX_CORNERS, EXACT>::add_plane(int point) {
        if (n_planes >= MAX_PLANES) {
            current_status = Status::too_many_planes;
            return -1;
        }
        point_of_plane[n_planes] = point;
        return n_planes++;
    }

    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::park_cut_away_corners(int new_plane) {
        double      scale;
        const Plane equation = plane(new_plane, &scale);
        n_cut_away           = 0;
        int i                = 0;
        while (i < n_corners) {
            if (is_cut_away(corners[i], new_plane, equation, scale)) {
                n_corners--;
                const Corner parked = corners[i];
                corners[i]          = corners[n_corners];
                corners[n_corners]  = parked;
                n_cut_away++;
            } else {
                i++;
            }
        }
    }

    // the ring of planes around the hole the cut away corners leave; in 3D it grows by one corner at a time
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::link_ring_around_cut_away_corners() {
        for (int p = 0; p < n_planes; p++)
            next_in_ring[p] = END_OF_RING;
        first_in_ring = END_OF_RING;

        if constexpr (config::dim == 2) {
            int touched[MAX_PLANES];
            for (int p = 0; p < n_planes; p++)
                touched[p] = 0;
            for (int r = 0; r < n_cut_away; r++) {
                touched[corners[n_corners + r].plane[0]]++;
                touched[corners[n_corners + r].plane[1]]++;
            }
            // the ends of the hole: the two lines that touch one cut away corner
            PlaneIndex ends[2];
            int        n_ends = 0;
            for (int p = 0; p < n_planes; p++) {
                if (touched[p] != 1) continue;
                if (n_ends < 2) ends[n_ends] = (PlaneIndex)p;
                n_ends++;
            }
            if (n_ends != 2) {
                current_status = Status::broken_ring;
                return;
            }
            first_in_ring         = ends[0];
            next_in_ring[ends[0]] = ends[1];
            next_in_ring[ends[1]] = ends[0];
        } else {
            constexpr int MAX_TRIES = (MAX_CORNERS > 255) ? 100 * MAX_CORNERS : 10000;
            int           n_tries   = 0;
            int           t         = n_corners;
            while (n_cut_away > 0) {
                if (n_tries++ > MAX_TRIES) {
                    current_status = Status::broken_ring;
                    return;
                }

                const Corner corner = corners[t];
                bool         in_ring[3];
                bool         next_is_reverse[3];
                for (int e = 0; e < 3; e++) {
                    in_ring[e]         = (next_in_ring[corner.plane[e]] != END_OF_RING);
                    next_is_reverse[e] = (next_in_ring[corner.plane[(e + 1) % 3]] == corner.plane[e]);
                }

                // a corner that would split the ring waits; the first corner starts the ring
                bool keeps_ring_simple = true;
                for (int e = 0; e < 3; e++) {
                    if (!next_is_reverse[e] && !next_is_reverse[(e + 1) % 3] && in_ring[(e + 1) % 3]) {
                        keeps_ring_simple = false;
                    }
                }
                if (!next_is_reverse[0] && !next_is_reverse[1] && !next_is_reverse[2]) {
                    if (first_in_ring == END_OF_RING) {
                        for (int e = 0; e < 3; e++)
                            next_in_ring[corner.plane[e]] = corner.plane[(e + 1) % 3];
                        first_in_ring = corner.plane[0];
                    } else {
                        keeps_ring_simple = false;
                    }
                }
                if (!keeps_ring_simple) {
                    t++;
                    if (t == n_corners + n_cut_away) t = n_corners;
                    continue;
                }

                // its edges into the ring; the planes now inside the hole drop out
                for (int e = 0; e < 3; e++) {
                    if (!next_is_reverse[e]) next_in_ring[corner.plane[e]] = corner.plane[(e + 1) % 3];
                }
                for (int e = 0; e < 3; e++) {
                    if (!(next_is_reverse[e] && next_is_reverse[(e + 1) % 3])) continue;
                    const PlaneIndex inside = corner.plane[(e + 1) % 3];
                    if (first_in_ring == inside) first_in_ring = next_in_ring[inside];
                    next_in_ring[inside] = END_OF_RING;
                }

                corners[t]                          = corners[n_corners + n_cut_away - 1];
                corners[n_corners + n_cut_away - 1] = corner;
                t                                   = n_corners;
                n_cut_away--;
            }
        }
    }

    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::add_corners_along_ring(int new_plane) {
        PlaneIndex here = first_in_ring;
        do {
            const PlaneIndex next = next_in_ring[here];
            if (next == END_OF_RING) {
                current_status = Status::broken_ring;
                return;
            }
            add_corner((PlaneIndex)new_plane, here, next);
            if (current_status != Status::ok) return;
            here = next;
        } while (here != first_in_ring);
    }

    // in 2D the two lines go in the order that turns the same way as the other corners
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::add_corner(PlaneIndex a, PlaneIndex b, PlaneIndex c) {
        if (n_corners + 1 >= MAX_CORNERS) {
            current_status = Status::too_many_corners;
            return;
        }
        Corner& corner = corners[n_corners];
        if constexpr (config::dim == 2) {
            (void)c;
            const Plane pa   = plane(a);
            const Plane pb   = plane(b);
            double      turn = det2x2(pa.normal[0], pa.normal[1], pb.normal[0], pb.normal[1]);
            if constexpr (EXACT && !config::compiling_for_gpu) turn = exactly_turns_left(a, b) ? 1.0 : -1.0;
            corner.plane[0] = (turn > 0.0) ? b : a;
            corner.plane[1] = (turn > 0.0) ? a : b;
        } else {
            corner.plane[0] = a;
            corner.plane[1] = b;
            corner.plane[2] = c;
        }
        fill_corner(corner);
        n_corners++;
    }

    // the minors and the bound that is_cut_away needs, and the distance that find_farthest_corner compares
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::fill_corner(Corner& corner) const {
        double      scale_a, scale_b;
        const Plane a = plane(corner.plane[0], &scale_a);
        const Plane b = plane(corner.plane[1], &scale_b);
        if constexpr (config::dim == 2) {
            minors_of_det3x3(a.normal[0], b.normal[0], a.normal[1], b.normal[1], a.offset, b.offset, corner.minor);
            corner.bound_scale = UNCERTAIN_DETERMINANT * scale_a * scale_b;
        } else {
            double      scale_c;
            const Plane c = plane(corner.plane[2], &scale_c);
            minors_of_det4x4(a.normal[0],
                             b.normal[0],
                             c.normal[0],
                             a.normal[1],
                             b.normal[1],
                             c.normal[1],
                             a.normal[2],
                             b.normal[2],
                             c.normal[2],
                             a.offset,
                             b.offset,
                             c.offset,
                             corner.minor);
            corner.bound_scale = UNCERTAIN_DETERMINANT * scale_a * scale_b * scale_c;
        }
        const UndividedPosition position = position_of(corner);
        corner.distance_numerator        = norm2(position.numerator - seed_position * position.denominator);
        corner.distance_denominator      = position.denominator * position.denominator;
    }

    // on the far side when the determinant with the new plane is positive; an exact cell asks the exact tests instead
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD bool Cell<MAX_PLANES, MAX_CORNERS, EXACT>::is_cut_away(const Corner& corner,
                                                              int           new_plane,
                                                              const Plane&  equation,
                                                              double        scale) {
        if constexpr (EXACT && !config::compiling_for_gpu) return exactly_cut_away(corner, new_plane);

        const double* m = corner.minor;
        double        determinant;
        if constexpr (config::dim == 2) {
            determinant = m[2] * equation.normal[0] - m[1] * equation.normal[1] + m[0] * equation.offset;
        } else {
            determinant = m[3] * equation.normal[0] - m[2] * equation.normal[1] + m[1] * equation.normal[2] -
                          m[0] * equation.offset;
        }
        const double bound = corner.bound_scale * scale;
        if (!(fabs(determinant) > bound)) current_status = Status::needs_exact_tests;
        return determinant > 0.0;
    }

    // a wall, or the plane halfway between the seed and its point, normal towards the seed; scale bounds its numbers
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD typename Cell<MAX_PLANES, MAX_CORNERS, EXACT>::Plane Cell<MAX_PLANES, MAX_CORNERS, EXACT>::plane(
        int     p,
        double* scale) const {
        Plane result{};
        if (p < 2 * config::dim) {
            const bool low       = (p % 2 == 0);
            result.normal[p / 2] = low ? 1.0 : -1.0;
            result.offset        = low ? WALL_LOW : WALL_HIGH;
            if (scale) *scale = 1.0 + result.offset;
            return result;
        }

        const Vec point        = positions[point_of_plane[p]];
        const Vec normal       = seed_position - point;
        const Vec twice_middle = seed_position + point;
        result.normal          = normal;
        result.offset          = -dot(twice_middle, normal) * 0.5;
        if (scale) {
            double size_of_normal = 0.0, size_of_offset = 0.0;
            for (int d = 0; d < config::dim; d++) {
                size_of_normal += fabs(normal[d]);
                size_of_offset += fabs(twice_middle[d] * normal[d]);
            }
            *scale = size_of_normal + 0.5 * size_of_offset;
        }
        return result;
    }

    // the plane equations by Cramer's rule; with nearly parallel planes an exact cell uses the points instead
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD typename Cell<MAX_PLANES, MAX_CORNERS, EXACT>::UndividedPosition Cell<MAX_PLANES, MAX_CORNERS, EXACT>::
        position_of(const Corner& corner) const {
        UndividedPosition result{};
        const Plane       a = plane(corner.plane[0]);
        const Plane       b = plane(corner.plane[1]);
        double            normals_squared;
        if constexpr (config::dim == 2) {
            result.numerator[0] = -det2x2(a.offset, a.normal[1], b.offset, b.normal[1]);
            result.numerator[1] = -det2x2(a.normal[0], a.offset, b.normal[0], b.offset);
            result.denominator  = det2x2(a.normal[0], a.normal[1], b.normal[0], b.normal[1]);
            normals_squared     = norm2(a.normal) * norm2(b.normal);
        } else {
            const Plane c       = plane(corner.plane[2]);
            result.numerator[0] = -det3x3(a.offset,
                                          a.normal[1],
                                          a.normal[2],
                                          b.offset,
                                          b.normal[1],
                                          b.normal[2],
                                          c.offset,
                                          c.normal[1],
                                          c.normal[2]);
            result.numerator[1] = -det3x3(a.normal[0],
                                          a.offset,
                                          a.normal[2],
                                          b.normal[0],
                                          b.offset,
                                          b.normal[2],
                                          c.normal[0],
                                          c.offset,
                                          c.normal[2]);
            result.numerator[2] = -det3x3(a.normal[0],
                                          a.normal[1],
                                          a.offset,
                                          b.normal[0],
                                          b.normal[1],
                                          b.offset,
                                          c.normal[0],
                                          c.normal[1],
                                          c.offset);
            result.denominator  = det3x3(a.normal[0],
                                        a.normal[1],
                                        a.normal[2],
                                        b.normal[0],
                                        b.normal[1],
                                        b.normal[2],
                                        c.normal[0],
                                        c.normal[1],
                                        c.normal[2]);
            normals_squared     = norm2(a.normal) * norm2(b.normal) * norm2(c.normal);
        }
        if constexpr (EXACT && !config::compiling_for_gpu) {
            if (!(result.denominator * result.denominator > WELL_POSED * normals_squared))
                return exact_position_of(corner);
        }
        (void)normals_squared;
        return result;
    }

    // compared as fractions, without dividing
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::find_farthest_corner() const {
        double max_numerator   = 0.0;
        double max_denominator = 1.0;
        for (int i = 0; i < n_corners; i++) {
            const double numerator   = corners[i].distance_numerator;
            const double denominator = corners[i].distance_denominator;
            if (numerator * max_denominator > max_numerator * denominator) {
                max_numerator   = numerator;
                max_denominator = denominator;
            }
        }
        farthest_numerator    = max_numerator;
        farthest_denominator  = max_denominator;
        farthest_corner_known = true;
    }

    // ==========================================================
    // the exact tests, CPU only
    // ==========================================================

    // a corner is the centre of the sphere through the seed and its points; a new point inside it cuts the corner away
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    bool Cell<MAX_PLANES, MAX_CORNERS, EXACT>::exactly_cut_away(const Corner& corner, int new_plane) {
        double points[config::dim + 2][config::dim];
        point_for_exact_tests(-1, points[0]);
        for (int d = 0; d < config::dim; d++)
            point_for_exact_tests(corner.plane[d], points[d + 1]);
        point_for_exact_tests(new_plane, points[config::dim + 1]);

        const double* through[config::dim + 1];
        for (int i = 0; i <= config::dim; i++)
            through[i] = points[i];
        const int side = exact_tests::in_circumsphere(through, points[config::dim + 1]);
        if (side < 0) current_status = Status::two_points_at_one_place;
        return side == 1;
    }

    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    typename Cell<MAX_PLANES, MAX_CORNERS, EXACT>::UndividedPosition Cell<MAX_PLANES, MAX_CORNERS, EXACT>::
        exact_position_of(const Corner& corner) const {
        double points[config::dim + 1][config::dim];
        point_for_exact_tests(-1, points[0]);
        for (int d = 0; d < config::dim; d++)
            point_for_exact_tests(corner.plane[d], points[d + 1]);

        const double* through[config::dim + 1];
        for (int i = 0; i <= config::dim; i++)
            through[i] = points[i];
        double centre[config::dim];
        exact_tests::circumcentre(through, centre);

        UndividedPosition result{};
        for (int d = 0; d < config::dim; d++)
            result.numerator[d] = centre[d];
        result.denominator = 1.0;
        return result;
    }

    // in 2D: whether the points of lines a and b and the seed turn counterclockwise
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    bool Cell<MAX_PLANES, MAX_CORNERS, EXACT>::exactly_turns_left(int a, int b) const {
        double points[3][config::dim];
        point_for_exact_tests(a, points[0]);
        point_for_exact_tests(b, points[1]);
        point_for_exact_tests(-1, points[2]);
        const double* through[3] = {points[0], points[1], points[2]};
        return exact_tests::orientation(through) > 0;
    }

    // the point whose plane with the seed is p: its own point, or the seed mirrored in a wall; -1 is the seed
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::point_for_exact_tests(int p, double* out) const {
        const Vec point = (p >= 2 * config::dim) ? positions[point_of_plane[p]] : seed_position;
        for (int d = 0; d < config::dim; d++)
            out[d] = point[d];
        if (p < 0 || p >= 2 * config::dim) return;
        const int axis = p / 2;
        out[axis]      = (p % 2 == 0) ? -2.0 * WALL_LOW - out[axis] : 2.0 * WALL_HIGH - out[axis];
    }

    // ==========================================================
    // determinants, each in one fixed order of operations
    // ==========================================================

    HD inline double det2x2(double a11, double a12, double a21, double a22) {
        return a11 * a22 - a12 * a21;
    }

    HD inline double det3x3(double a11,
                            double a12,
                            double a13,
                            double a21,
                            double a22,
                            double a23,
                            double a31,
                            double a32,
                            double a33) {
        return a11 * det2x2(a22, a23, a32, a33) - a21 * det2x2(a12, a13, a32, a33) + a31 * det2x2(a12, a13, a22, a23);
    }

    // the determinant is minor[2] a13 - minor[1] a23 + minor[0] a33
    HD inline void minors_of_det3x3(double  a11,
                                    double  a12,
                                    double  a21,
                                    double  a22,
                                    double  a31,
                                    double  a32,
                                    double* minor) {
        minor[0] = det2x2(a11, a12, a21, a22);
        minor[1] = det2x2(a11, a12, a31, a32);
        minor[2] = det2x2(a21, a22, a31, a32);
    }

    // the determinant is minor[3] a14 - minor[2] a24 + minor[1] a34 - minor[0] a44
    HD inline void minors_of_det4x4(double  a11,
                                    double  a12,
                                    double  a13,
                                    double  a21,
                                    double  a22,
                                    double  a23,
                                    double  a31,
                                    double  a32,
                                    double  a33,
                                    double  a41,
                                    double  a42,
                                    double  a43,
                                    double* minor) {
        const double m12 = a21 * a12 - a11 * a22;
        const double m13 = a31 * a12 - a11 * a32;
        const double m14 = a41 * a12 - a11 * a42;
        const double m23 = a31 * a22 - a21 * a32;
        const double m24 = a41 * a22 - a21 * a42;
        const double m34 = a41 * a32 - a31 * a42;
        minor[0]         = m23 * a13 - m13 * a23 + m12 * a33;
        minor[1]         = m24 * a13 - m14 * a23 + m12 * a43;
        minor[2]         = m34 * a13 - m14 * a33 + m13 * a43;
        minor[3]         = m34 * a23 - m24 * a33 + m23 * a43;
    }

} // namespace voronoi_cell
