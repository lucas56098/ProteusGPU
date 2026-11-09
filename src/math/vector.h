#pragma once

// the vector operations (included by math.h)

HD inline Vec operator+(Vec a, const Vec& b) {
    for (int d = 0; d < config::dim; d++)
        a[d] += b[d];
    return a;
}

HD inline Vec operator-(Vec a, const Vec& b) {
    for (int d = 0; d < config::dim; d++)
        a[d] -= b[d];
    return a;
}

HD inline Vec operator-(Vec a) {
    for (int d = 0; d < config::dim; d++)
        a[d] = -a[d];
    return a;
}

HD inline Vec operator*(double s, Vec a) {
    for (int d = 0; d < config::dim; d++)
        a[d] *= s;
    return a;
}

HD inline Vec operator*(Vec a, double s) {
    return s * a;
}

HD inline Vec operator/(Vec a, double s) {
    for (int d = 0; d < config::dim; d++)
        a[d] /= s;
    return a;
}

HD inline Vec& operator+=(Vec& a, const Vec& b) {
    a = a + b;
    return a;
}

HD inline Vec& operator-=(Vec& a, const Vec& b) {
    a = a - b;
    return a;
}

HD inline double dot(const Vec& a, const Vec& b) {
    double sum = 0.0;
    for (int d = 0; d < config::dim; d++)
        sum += a[d] * b[d];
    return sum;
}

HD inline double norm2(const Vec& a) {
    return dot(a, a);
}

HD inline Vec cross(const Vec& a, const Vec& b) {
    Vec result{};
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}
