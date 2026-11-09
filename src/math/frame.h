#pragma once

// the orthonormal frame around a direction (included by math.h)

#include <cmath>

constexpr double NEARLY_ALONG_Z = 1e-12;

HD inline Frame frame_around(const Vec& direction) {
    Frame frame{};
    frame.normal = (1.0 / sqrt(norm2(direction))) * direction;

    const Vec&   n        = frame.normal;
    const double in_plane = n[0] * n[0] + n[1] * n[1];
    if (in_plane > NEARLY_ALONG_Z) {
        const double one_over_length = 1.0 / sqrt(in_plane);
        frame.tangent[0][0]          = -n[1] * one_over_length;
        frame.tangent[0][1]          = n[0] * one_over_length;
    } else {
        frame.tangent[0][0] = 1.0;
    }
    if constexpr (config::dim == 3) frame.tangent[config::dim - 2] = cross(n, frame.tangent[0]);
    return frame;
}
