#ifndef PREDICATES_H
#define PREDICATES_H

// Exact tests on points for the cell build on the CPU. A point is DIMENSION doubles; p holds the seed first,
// then the DIMENSION points whose bisectors with it meet in a cell vertex. Host only.

namespace voronoi {
    namespace exact {

        // sign of the orientation of p: in 2D counterclockwise is +1, in 3D p[3] below the plane through
        // p[0], p[1], p[2] (counterclockwise seen from above) is +1
        int orientation(const double* const* p);

        // 1 if q is inside the circle (sphere) through p, 0 if outside. On it, the tie rule decides: the
        // lifted points are raised by an infinitesimal that grows with their (x, y, z) order, so every rank
        // decides a tie the same way. -1 if p is flat or q sits on a point of p
        int in_circumsphere(const double* const* p, const double* q);

        // the centre of the circle (sphere) through p, exact up to the last rounding; for the vertices whose
        // planes are nearly parallel, where the plane equations lose all digits
        void circumcentre(const double* const* p, double* out);

    } // namespace exact
} // namespace voronoi

#endif
