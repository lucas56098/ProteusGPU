#pragma once

// volume, centroid, faces and security radius of a finished cell (included by voronoi_cell.h)

#include "cell.h"

#include <cmath>

namespace voronoi_cell {

    // sums the simplices between the seed and the faces; faces and corners go in fixed orders, so the bits do not
    // depend on the order of the cuts; corners are relative to the seed, so small cells keep their digits
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    template <typename FaceFunction>
    HD CellGeometry Cell<MAX_PLANES, MAX_CORNERS, EXACT>::compute_geometry(FaceFunction store_face) const {
        Vec corner_position[MAX_CORNERS];
        for (int c = 0; c < n_corners; c++)
            corner_position[c] = position_relative_to_seed(corners[c]);

        PlaneIndex face[MAX_PLANES];
        const int  n_faces = faces_in_fixed_order(face);

        double volume = 0.0;
        Vec    volume_times_centroid{};
        for (int f = 0; f < n_faces; f++) {
            int       ring[MAX_PLANES];
            const int n_ring = corners_around_face(face[f], ring);
            const Vec first  = corner_position[ring[0]];

            double area          = 0.0;
            Vec    face_centroid = first;
            if constexpr (config::dim == 2) {
                const Vec    last          = corner_position[ring[1]];
                const double triangle_area = 0.5 * det2x2(first[0], first[1], last[0], last[1]);
                area                       = sqrt(norm2(last - first));
                face_centroid              = 0.5 * (first + last);
                volume += triangle_area;
                volume_times_centroid += (triangle_area / 3.0) * (first + last);
            } else {
                Vec area_times_centroid{};
                for (int i = 1; i + 1 < n_ring; i++) {
                    const Vec    second             = corner_position[ring[i]];
                    const Vec    third              = corner_position[ring[i + 1]];
                    const double triangle_area      = 0.5 * sqrt(norm2(cross(second - first, third - first)));
                    const double tetrahedron_volume = dot(first, cross(second, third)) / 6.0;
                    area += triangle_area;
                    area_times_centroid += (triangle_area / 3.0) * (first + second + third);
                    volume += tetrahedron_volume;
                    volume_times_centroid += (tetrahedron_volume / 4.0) * (first + second + third);
                }
                if (area > 0.0) face_centroid = area_times_centroid / area;
            }
            store_face(point_of_plane[face[f]], area, seed_position + face_centroid);
        }
        return CellGeometry{volume, volume_times_centroid / volume};
    }

    // with the same margin as is_finished_before
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD double Cell<MAX_PLANES, MAX_CORNERS, EXACT>::security_radius() const {
        if (!farthest_corner_known) find_farthest_corner();
        return 2.0 * sqrt(SECURITY_MARGIN * farthest_numerator / farthest_denominator);
    }

    // the walls first, by number; then the planes by the position of their point, x, then y, then z
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD bool Cell<MAX_PLANES, MAX_CORNERS, EXACT>::plane_comes_before(int a, int b) const {
        const bool a_is_wall = (a < 2 * config::dim);
        const bool b_is_wall = (b < 2 * config::dim);
        if (a_is_wall || b_is_wall) return a_is_wall && (!b_is_wall || a < b);

        const Vec& position_a = positions[point_of_plane[a]];
        const Vec& position_b = positions[point_of_plane[b]];
        for (int d = 0; d < config::dim; d++) {
            if (position_a[d] != position_b[d]) return position_a[d] < position_b[d];
        }
        return false;
    }

    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD int Cell<MAX_PLANES, MAX_CORNERS, EXACT>::number_of_faces() const {
        bool has_corners[MAX_PLANES];
        mark_planes_with_corners(has_corners);
        int n_faces = 0;
        for (int p = 0; p < n_planes; p++) {
            if (has_corners[p]) n_faces++;
        }
        return n_faces;
    }

    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD int Cell<MAX_PLANES, MAX_CORNERS, EXACT>::faces_in_fixed_order(PlaneIndex* face) const {
        bool has_corners[MAX_PLANES];
        mark_planes_with_corners(has_corners);

        int n_faces = 0;
        for (int p = 0; p < n_planes; p++) {
            if (!has_corners[p]) continue;
            int i = n_faces++;
            while (i > 0 && plane_comes_before(p, face[i - 1])) {
                face[i] = face[i - 1];
                i--;
            }
            face[i] = (PlaneIndex)p;
        }
        return n_faces;
    }

    // a plane is dropped only when it cut nothing, so the planes after the walls are those of some points, in order
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD uint64_t Cell<MAX_PLANES, MAX_CORNERS, EXACT>::faces_among(const int* point, int n) const {
        static_assert(MAX_PLANES <= 64, "the planes with corners are bits of one word");
        uint64_t with_corners = 0;
        for (int c = 0; c < n_corners; c++)
            for (int d = 0; d < config::dim; d++)
                with_corners |= (uint64_t)1 << corners[c].plane[d];

        uint64_t faces = 0;
        int      p     = 2 * config::dim;
        for (int k = 0; k < n && p < n_planes; k++) {
            if (point_of_plane[p] != point[k]) continue;
            if ((with_corners >> p) & 1) faces |= (uint64_t)1 << k;
            p++;
        }
        return faces;
    }

    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void Cell<MAX_PLANES, MAX_CORNERS, EXACT>::mark_planes_with_corners(bool* has_corners) const {
        for (int p = 0; p < n_planes; p++)
            has_corners[p] = false;
        for (int c = 0; c < n_corners; c++)
            for (int d = 0; d < config::dim; d++)
                has_corners[corners[c].plane[d]] = true;
    }

    // the corners of a face in order, counterclockwise in 2D, with the normal out of the cell in 3D; returns how many
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD int Cell<MAX_PLANES, MAX_CORNERS, EXACT>::corners_around_face(int face, int* ring) const {
        if constexpr (config::dim == 2) {
            ring[0] = 0;
            ring[1] = 0;
            for (int c = 0; c < n_corners; c++) {
                if (corners[c].plane[0] == face) ring[0] = c;
                if (corners[c].plane[1] == face) ring[1] = c;
            }
            return 2;
        } else {
            // every corner on the face, with the two planes that follow the face in its turning order
            int        corner_on_face[MAX_PLANES];
            PlaneIndex second_plane[MAX_PLANES];
            PlaneIndex third_plane[MAX_PLANES];
            int        n = 0;
            for (int corner = 0; corner < n_corners; corner++) {
                for (int e = 0; e < 3; e++) {
                    if (corners[corner].plane[e] != face) continue;
                    corner_on_face[n] = corner;
                    second_plane[n]   = corners[corner].plane[(e + 1) % 3];
                    third_plane[n]    = corners[corner].plane[(e + 2) % 3];
                    n++;
                }
            }

            // from the corner whose second plane comes first; the next corner has this one's third plane as its second
            int here = 0;
            for (int i = 1; i < n; i++) {
                if (plane_comes_before(second_plane[i], second_plane[here])) here = i;
            }
            for (int k = 0; k < n; k++) {
                ring[k]  = corner_on_face[here];
                int next = 0;
                while (next < n && second_plane[next] != third_plane[here])
                    next++;
                if (next == n) return k + 1;
                here = next;
            }
            return n;
        }
    }

    // sorts the planes into the fixed order before solving; their order in a corner only tells which way it turns
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD Vec Cell<MAX_PLANES, MAX_CORNERS, EXACT>::position_relative_to_seed(Corner corner) const {
        for (int i = 1; i < config::dim; i++) {
            for (int j = i; j > 0 && plane_comes_before(corner.plane[j], corner.plane[j - 1]); j--) {
                const PlaneIndex swapped = corner.plane[j];
                corner.plane[j]          = corner.plane[j - 1];
                corner.plane[j - 1]      = swapped;
            }
        }
        const UndividedPosition position = position_of(corner);
        return position.numerator / position.denominator - seed_position;
    }

} // namespace voronoi_cell
