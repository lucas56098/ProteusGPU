#pragma once

// One Voronoi cell, cut out of a box by its neighbours (Ray et al. 2018), with exact tests where doubles cannot decide.

#include "math/math.h"
#include "ngbtree/ngbtree.h"

#include <cstdint>
#include <type_traits>

namespace voronoi_cell {

    // how a build stands; anything but ok leaves the cell unfinished
    enum class Status {
        ok,
        needs_more_points,
        too_many_planes,
        too_many_corners,
        broken_ring, // the ring of planes around the cut away corners does not close
        needs_exact_tests, // a cut too close to call with doubles
        two_points_at_one_place, // found by the exact tests
    };

    // the volume of a cell, its area in 2D, and its centroid relative to its seed
    struct CellGeometry {
        double volume;
        Vec    centroid_from_seed;
    };

    // left for a build with more neighbours: bit k of faces if neighbour k is a face; n_neighbours is 0 for none
    struct Handover {
        uint64_t faces;
        int      n_neighbours;
    };

    // a cell while it is built; an EXACT cell (CPU only) decides every cut with exact tests
    template <int MAX_PLANES, int MAX_CORNERS, bool EXACT = false> class Cell {
      public:
        // starts as the box plus a margin of half a box on every side
        HD Cell(int seed, const Vec* positions);

        // cuts the cell by the plane halfway between the seed and point; the side of the seed stays
        HD void cut_by(int point);

        // whether point lies outside the security radius: then it, and every point farther away, cannot cut the cell
        HD bool is_finished_before(int point) const;

        HD Status status() const { return current_status; }

        HD void mark_as_needing_more_points() { current_status = Status::needs_more_points; }

        // the corner spheres through the seed that reach at least squared distance inside from it; returns how many
        HD int corner_spheres(Vec* centre, double* squared_radius, double inside) const;

        // volume and centroid of a finished cell, and store_face(neighbour, area, centroid) for every face in a fixed
        // order; neighbour is -1 for a wall of the start box; in 2D the area of a face is its length
        template <typename FaceFunction> HD CellGeometry compute_geometry(FaceFunction store_face) const;

        HD int number_of_faces() const;

        // bit k for every point[k] whose plane is a face, k < n <= 64; the planes came from these points, in order
        HD uint64_t faces_among(const int* point, int n) const;

        // twice the distance from the seed to the farthest corner, with the margin; points beyond it cannot cut
        HD double security_radius() const;

      private:
        // one byte while there is room for at most 254 planes; END_OF_RING is none
        using PlaneIndex                        = typename std::conditional<(MAX_PLANES < 255), uint8_t, int>::type;
        static constexpr PlaneIndex END_OF_RING = (PlaneIndex)(-1);

        // the points x with normal . x + offset >= 0 are inside
        struct Plane {
            Vec    normal;
            double offset;
        };

        // a corner: its config::dim planes, and the minors, bound and squared distance from the seed its tests need
        struct Corner {
            PlaneIndex plane[config::dim];
            double     minor[config::dim + 1];
            double     bound_scale;
            double     distance_numerator;
            double     distance_denominator;
        };

        struct UndividedPosition {
            Vec    numerator;
            double denominator;
        };

        const Vec* positions;
        Vec        seed_position;
        Status     current_status = Status::ok;

        // the first 2 * config::dim planes are the walls of the box, -1 in point_of_plane
        int point_of_plane[MAX_PLANES];
        int n_planes;

        // the corners the last cut removed are parked behind the n_corners corners
        Corner corners[MAX_CORNERS];
        int    n_corners;
        int    n_cut_away;

        PlaneIndex first_in_ring;
        PlaneIndex next_in_ring[MAX_PLANES];

        // squared distance to the farthest corner as numerator / denominator, while no cut changed the cell
        mutable bool   farthest_corner_known = false;
        mutable double farthest_numerator    = 0.0;
        mutable double farthest_denominator  = 1.0;

        HD int add_plane(int point);
        HD void park_cut_away_corners(int plane);
        HD void link_ring_around_cut_away_corners();
        HD void add_corners_along_ring(int plane);
        HD void add_corner(PlaneIndex a, PlaneIndex b, PlaneIndex c);
        HD void fill_corner(Corner& corner) const;
        HD bool is_cut_away(const Corner& corner, int new_plane, const Plane& equation, double scale);
        HD Plane plane(int p, double* scale = nullptr) const;
        HD UndividedPosition position_of(const Corner& corner) const;
        HD void find_farthest_corner() const;

        HD bool plane_comes_before(int a, int b) const;
        HD int faces_in_fixed_order(PlaneIndex* face) const;
        HD void mark_planes_with_corners(bool* has_corners) const;
        HD int corners_around_face(int face, int* ring) const;
        HD Vec position_relative_to_seed(Corner corner) const;

        // the exact tests, CPU only
        bool exactly_cut_away(const Corner& corner, int new_plane);
        UndividedPosition exact_position_of(const Corner& corner) const;
        bool exactly_turns_left(int a, int b) const;
        void point_for_exact_tests(int p, double* out) const;
    };

    // cuts cell, made as Cell(seed, tree.positions), by the K nearest neighbours of seed and, if WALK, the points a
    // walk finds; without WALK a cell they do not finish needs more points, and the handover gets its faces
    template <int K, bool WALK, int MAX_PLANES, int MAX_CORNERS, bool EXACT>
    HD void build_cell(const ngbtree::TreePointers&          tree,
                       int                                   seed,
                       Cell<MAX_PLANES, MAX_CORNERS, EXACT>& cell,
                       Handover&                             handover);

} // namespace voronoi_cell

#include "build_cell.h"
#include "cell.h"
#include "geometry.h"
