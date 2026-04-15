#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#pragma once

#include "gpu_compat.h"
#include "structs.h"
#include <cmath>

// vector math helpers for voronoi mesh generation and hydro

inline double4 minus4(double4 A, double4 B) {
    return make_double4(A.x - B.x, A.y - B.y, A.z - B.z, A.w - B.w);
}
inline double4 plus4(double4 A, double4 B) {
    return make_double4(A.x + B.x, A.y + B.y, A.z + B.z, A.w + B.w);
}
inline double dot4(double4 A, double4 B) {
    return A.x * B.x + A.y * B.y + A.z * B.z + A.w * B.w;
}
inline double dot3(double4 A, double4 B) {
    return A.x * B.x + A.y * B.y + A.z * B.z;
}
inline double4 mul3(double s, double4 A) {
    return make_double4(s * A.x, s * A.y, s * A.z, 1.);
}
inline double4 cross3(double4 A, double4 B) {
    return make_double4(A.y * B.z - A.z * B.y, A.z * B.x - A.x * B.z, A.x * B.y - A.y * B.x, 0);
}

inline double4 point_from_ptr(double* f) {
#ifdef dim_2D
    return make_double4(f[0], f[1], 0, 1);
#else
    return make_double4(f[0], f[1], f[2], 1);
#endif
}

// determinants
inline double det2x2(double a11, double a12, double a21, double a22) {
    return a11 * a22 - a12 * a21;
}

inline double
det3x3(double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33) {
    return a11 * det2x2(a22, a23, a32, a33) - a21 * det2x2(a12, a13, a32, a33) + a31 * det2x2(a12, a13, a22, a23);
}

inline double det4x4(double a11,
                     double a12,
                     double a13,
                     double a14,
                     double a21,
                     double a22,
                     double a23,
                     double a24,
                     double a31,
                     double a32,
                     double a33,
                     double a34,
                     double a41,
                     double a42,
                     double a43,
                     double a44) {

    double m12 = a21 * a12 - a11 * a22;
    double m13 = a31 * a12 - a11 * a32;
    double m14 = a41 * a12 - a11 * a42;
    double m23 = a31 * a22 - a21 * a32;
    double m24 = a41 * a22 - a21 * a42;
    double m34 = a41 * a32 - a31 * a42;

    double m123 = m23 * a13 - m13 * a23 + m12 * a33;
    double m124 = m24 * a13 - m14 * a23 + m12 * a43;
    double m134 = m34 * a13 - m14 * a33 + m13 * a43;
    double m234 = m34 * a23 - m24 * a33 + m23 * a43;

    return (m234 * a14 - m134 * a24 + m124 * a34 - m123 * a44);
}

// weighted least squares solvers
#ifdef dim_2D
inline bool solve_weighted_lsq_2d(double m00, double m01, double m11, double b0, double b1, GRAD_TYPE* grad) {
    double det = m00 * m11 - m01 * m01;
    if (fabs(det) < 1e-14) { return false; }

    double inv00 = m11 / det;
    double inv01 = -m01 / det;
    double inv11 = m00 / det;

    grad->x = (compact_t)(inv00 * b0 + inv01 * b1);
    grad->y = (compact_t)(inv01 * b0 + inv11 * b1);
    return true;
}
#else
inline bool solve_weighted_lsq_3d(double     m00,
                                  double     m01,
                                  double     m02,
                                  double     m11,
                                  double     m12,
                                  double     m22,
                                  double     b0,
                                  double     b1,
                                  double     b2,
                                  GRAD_TYPE* grad) {
    double a11 = m00;
    double a12 = m01;
    double a13 = m02;
    double a21 = m01;
    double a22 = m11;
    double a23 = m12;
    double a31 = m02;
    double a32 = m12;
    double a33 = m22;

    double det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
    if (fabs(det) < 1e-18) { return false; }

    double inv00 = (a22 * a33 - a23 * a32) / det;
    double inv01 = (a13 * a32 - a12 * a33) / det;
    double inv02 = (a12 * a23 - a13 * a22) / det;
    double inv10 = (a23 * a31 - a21 * a33) / det;
    double inv11 = (a11 * a33 - a13 * a31) / det;
    double inv12 = (a13 * a21 - a11 * a23) / det;
    double inv20 = (a21 * a32 - a22 * a31) / det;
    double inv21 = (a12 * a31 - a11 * a32) / det;
    double inv22 = (a11 * a22 - a12 * a21) / det;

    grad->x = (compact_t)(inv00 * b0 + inv01 * b1 + inv02 * b2);
    grad->y = (compact_t)(inv10 * b0 + inv11 * b1 + inv12 * b2);
    grad->z = (compact_t)(inv20 * b0 + inv21 * b1 + inv22 * b2);
    return true;
}
#endif

// min/max helpers
inline void get_minmax3(double& m, double& M, double x1, double x2, double x3) {
    m = fmin(fmin(x1, x2), x3);
    M = fmax(fmax(x1, x2), x3);
}

// periodic boundary condition helper (assumes boxsize 1)
inline double wrap_periodic_delta(double d) {
    if (d > 0.5) d -= 1.0;
    if (d < -0.5) d += 1.0;
    return d;
}

// POINT_TYPE math
inline double point_dot(const POINT_TYPE& a, const POINT_TYPE& b) {
#ifdef dim_2D
    return a.x * b.x + a.y * b.y;
#else
    return a.x * b.x + a.y * b.y + a.z * b.z;
#endif
}

inline POINT_TYPE point_mul(double s, const POINT_TYPE& p) {
#ifdef dim_2D
    POINT_TYPE out = {s * p.x, s * p.y};
#else
    POINT_TYPE out = {s * p.x, s * p.y, s * p.z};
#endif
    return out;
}

inline POINT_TYPE point_diff(const double3& a, const double3& b) {
#ifdef dim_2D
    POINT_TYPE out = {wrap_periodic_delta(a.x - b.x), wrap_periodic_delta(a.y - b.y)};
#else
    POINT_TYPE out = {wrap_periodic_delta(a.x - b.x), wrap_periodic_delta(a.y - b.y), wrap_periodic_delta(a.z - b.z)};
#endif
    return out;
}

// overloads for GRAD_TYPE (active only when SAVE_MEMORY makes GRAD_TYPE != POINT_TYPE)
#ifdef SAVE_MEMORY
inline double point_dot(const GRAD_TYPE& a, const POINT_TYPE& b) {
#ifdef dim_2D
    return (double)a.x * b.x + (double)a.y * b.y;
#else
    return (double)a.x * b.x + (double)a.y * b.y + (double)a.z * b.z;
#endif
}

inline double point_dot(const GRAD_TYPE& a, const GRAD_TYPE& b) {
#ifdef dim_2D
    return (double)a.x * (double)b.x + (double)a.y * (double)b.y;
#else
    return (double)a.x * (double)b.x + (double)a.y * (double)b.y + (double)a.z * (double)b.z;
#endif
}

inline GRAD_TYPE point_mul(double s, const GRAD_TYPE& p) {
#ifdef dim_2D
    return {(compact_t)(s * p.x), (compact_t)(s * p.y)};
#else
    return {(compact_t)(s * p.x), (compact_t)(s * p.y), (compact_t)(s * p.z)};
#endif
}
#endif

// computes orthonormal basis {n, m, p} from a raw (unnormalized) direction vector
inline geom compute_geom(double3 delta) {
    geom g;

    double nn = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    g.n       = {delta.x / nn, delta.y / nn, delta.z / nn};

    if (g.n.x != 0.0 || g.n.y != 0.0) {
        g.m = {-g.n.y, g.n.x, 0.0};
    } else {
        g.m = {1.0, 0.0, 0.0};
    }

    double mm = sqrt(g.m.x * g.m.x + g.m.y * g.m.y + g.m.z * g.m.z);
    g.m       = {g.m.x / mm, g.m.y / mm, g.m.z / mm};

    g.p = {g.n.y * g.m.z - g.n.z * g.m.y, g.n.z * g.m.x - g.n.x * g.m.z, g.n.x * g.m.y - g.n.y * g.m.x};

    return g;
}

#endif // MATH_UTILS_H
