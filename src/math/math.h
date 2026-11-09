#pragma once

// Math helpers: vector, orthonormal frame, math functions, radius of ball, periodic box.

#include "config/config.h"

struct Vec {
    double c[config::dim];

    HD double& operator[](int d) { return c[d]; }
    HD double operator[](int d) const { return c[d]; }
};

HD inline Vec operator+(Vec a, const Vec& b);
HD inline Vec operator-(Vec a, const Vec& b);
HD inline Vec operator-(Vec a);
HD inline Vec operator*(double s, Vec a);
HD inline Vec operator*(Vec a, double s);
HD inline Vec operator/(Vec a, double s);
HD inline Vec& operator+=(Vec& a, const Vec& b);
HD inline Vec& operator-=(Vec& a, const Vec& b);

HD inline double dot(const Vec& a, const Vec& b);
HD inline Vec cross(const Vec& a, const Vec& b);
HD inline double norm2(const Vec& a);

struct Frame {
    Vec normal;
    Vec tangent[config::dim - 1];
};
HD inline Frame frame_around(const Vec& direction);

HD inline double radius_of_ball(double volume);

// math functions, bitwise equal on CPU and GPU
HD inline double natural_log(double x);
HD inline double exponential(double x);
HD inline double power(double base, double exponent);

// a - b, to the nearest periodic image
HD inline Vec periodic_difference(const Vec& a, const Vec& b);

// rounds to box precision and wraps into the box
HD inline Vec wrapped_into_box(const Vec& p);

HD inline Vec rounded_to_box_precision(Vec p);

// the definitions of the functions above
#include "frame.h"
#include "functions.h"
#include "periodic.h"
#include "vector.h"
