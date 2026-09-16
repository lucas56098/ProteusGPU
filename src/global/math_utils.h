#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#pragma once

// Math helpers for host and device: cube root, 4-vectors, determinants, least squares, periodic wrap.

#include "gpu_compat.h"
#include "structs.h"
#include <cfloat>
#include <cmath>

// cube root with the same result on CPU and GPU
// libm cbrt is not correctly rounded, which would break the bitwise CPU == GPU guarantee
HD inline double portable_cbrt(double a) {
    if (a == 0.0) return a;
    if (!(a == a)) return a;
    const bool neg = (a < 0.0);
    double     x   = neg ? -a : a;
    if (x > DBL_MAX) return a;

    int    e;
    double m = frexp(x, &e);
    int    r = e % 3;
    if (r < 0) r += 3;
    e -= r;
    m = ldexp(m, r);

    double y = 0.4748 + 0.6529 * m - 0.1090 * m * m;
    for (int i = 0; i < 4; i++) {
        y = (2.0 * y + m / (y * y)) / 3.0;
    }
    y += (m - y * y * y) / (3.0 * y * y);

    const double res = ldexp(y, e / 3);
    return neg ? -res : res;
}

// double4_t helpers; dot3 and cross3 use x, y, z only
HD inline double4_t minus4(double4_t A, double4_t B) {
    return make_double4_t(A.x - B.x, A.y - B.y, A.z - B.z, A.w - B.w);
}
HD inline double4_t plus4(double4_t A, double4_t B) {
    return make_double4_t(A.x + B.x, A.y + B.y, A.z + B.z, A.w + B.w);
}
HD inline double dot3(double4_t A, double4_t B) {
    return A.x * B.x + A.y * B.y + A.z * B.z;
}
HD inline double4_t cross3(double4_t A, double4_t B) {
    return make_double4_t(A.y * B.z - A.z * B.y, A.z * B.x - A.x * B.z, A.x * B.y - A.y * B.x, 0);
}

// one point out of a flat DIM-stride array, w = 1
HD inline double4_t point_from_ptr(double* f) {
#ifdef dim_2D
    return make_double4_t(f[0], f[1], 0, 1);
#else
    return make_double4_t(f[0], f[1], f[2], 1);
#endif
}

// determinants used by the cell clipping and the least squares solve
HD inline double det2x2(double a11, double a12, double a21, double a22) {
    return a11 * a22 - a12 * a21;
}

HD inline double
det3x3(double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33) {
    return a11 * det2x2(a22, a23, a32, a33) - a21 * det2x2(a12, a13, a32, a33) + a31 * det2x2(a12, a13, a22, a23);
}

HD inline double det4x4(double a11,
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

#ifdef dim_2D
// solves the symmetric least squares system for one gradient
// a singular system gives a zero gradient and false
HD inline bool solve_weighted_lsq_2d(double m00, double m01, double m11, double b0, double b1, POINT_TYPE* grad) {
    double det = det2x2(m00, m01, m01, m11);
    if (fabs(det) < 1e-14) {
        grad->x = 0.0;
        grad->y = 0.0;
        return false;
    }

    double inv00 = m11 / det;
    double inv01 = -m01 / det;
    double inv11 = m00 / det;

    grad->x = inv00 * b0 + inv01 * b1;
    grad->y = inv01 * b0 + inv11 * b1;
    return true;
}
#else
// same in 3D
HD inline bool solve_weighted_lsq_3d(double      m00,
                                     double      m01,
                                     double      m02,
                                     double      m11,
                                     double      m12,
                                     double      m22,
                                     double      b0,
                                     double      b1,
                                     double      b2,
                                     POINT_TYPE* grad) {

    const double det = det3x3(m00, m01, m02, m01, m11, m12, m02, m12, m22);
    if (fabs(det) < 1e-18) {
        grad->x = 0.0;
        grad->y = 0.0;
        grad->z = 0.0;
        return false;
    }

    const double inv_det = 1.0 / det;
    const double inv00   = det2x2(m11, m12, m12, m22) * inv_det;
    const double inv11   = det2x2(m00, m02, m02, m22) * inv_det;
    const double inv22   = det2x2(m00, m01, m01, m11) * inv_det;
    const double inv01   = -det2x2(m01, m02, m12, m22) * inv_det;
    const double inv02   = det2x2(m01, m02, m11, m12) * inv_det;
    const double inv12   = -det2x2(m00, m02, m01, m12) * inv_det;

    grad->x = inv00 * b0 + inv01 * b1 + inv02 * b2;
    grad->y = inv01 * b0 + inv11 * b1 + inv12 * b2;
    grad->z = inv02 * b0 + inv12 * b1 + inv22 * b2;
    return true;
}
#endif

// min and max of three values
HD inline void get_minmax3(double& m, double& M, double x1, double x2, double x3) {
    m = fmin(fmin(x1, x2), x3);
    M = fmax(fmax(x1, x2), x3);
}

// shortest signed distance in the periodic box of size 1
HD inline double wrap_periodic_delta(double d) {
    if (d > 0.5) d -= 1.0;
    if (d < -0.5) d += 1.0;
    return d;
}

// POINT_TYPE helpers, 2D or 3D depending on the build
HD inline double point_dot(const POINT_TYPE& a, const POINT_TYPE& b) {
#ifdef dim_2D
    return a.x * b.x + a.y * b.y;
#else
    return a.x * b.x + a.y * b.y + a.z * b.z;
#endif
}

HD inline POINT_TYPE point_mul(double s, const POINT_TYPE& p) {
#ifdef dim_2D
    POINT_TYPE out = {s * p.x, s * p.y};
#else
    POINT_TYPE out = {s * p.x, s * p.y, s * p.z};
#endif
    return out;
}

// a - b with the periodic wrap on every axis
HD inline POINT_TYPE point_diff_periodic(const double3& a, const double3& b) {
#ifdef dim_2D
    POINT_TYPE out = {wrap_periodic_delta(a.x - b.x), wrap_periodic_delta(a.y - b.y)};
#else
    POINT_TYPE out = {wrap_periodic_delta(a.x - b.x), wrap_periodic_delta(a.y - b.y), wrap_periodic_delta(a.z - b.z)};
#endif
    return out;
}

// orthonormal frame around delta: n along it, m and p tangential
HD inline geom compute_geom(double3 delta) {
    geom g;

    double nn     = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    double inv_nn = 1.0 / nn;
    g.n           = {delta.x * inv_nn, delta.y * inv_nn, delta.z * inv_nn};

    const double xy_sq = g.n.x * g.n.x + g.n.y * g.n.y;
    if (xy_sq > 1e-12) {
        double inv_mm = 1.0 / sqrt(xy_sq);
        g.m           = {-g.n.y * inv_mm, g.n.x * inv_mm, 0.0};
    } else {
        g.m = {1.0, 0.0, 0.0};
    }

    g.p = {g.n.y * g.m.z - g.n.z * g.m.y, g.n.z * g.m.x - g.n.x * g.m.z, g.n.x * g.m.y - g.n.y * g.m.x};

    return g;
}

#endif
