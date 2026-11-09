#pragma once

// Exact tests on points, for the cells built on the CPU.

namespace voronoi_cell {
    namespace exact_tests {

        // sign of p: +1 if counterclockwise in 2D, or if p[0], p[1], p[2] turn clockwise seen from p[3] in 3D
        int orientation(const double* const* p);

        // 1 if q is inside the circle (sphere) through p, 0 if outside, -1 if p is flat or q sits on a point of p;
        // a tie goes by the (x, y, z) order of the points, so every rank decides it the same way
        int in_circumsphere(const double* const* p, const double* q);

        // centre of the circle (sphere) through p, exact up to the last rounding; for corners of nearly parallel planes
        void circumcentre(const double* const* p, double* out);

    } // namespace exact_tests
} // namespace voronoi_cell
