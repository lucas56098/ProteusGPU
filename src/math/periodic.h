#pragma once

// the periodic unit box (included by math.h)

HD inline Vec periodic_difference(const Vec& a, const Vec& b) {
    Vec difference = a - b;
    for (int d = 0; d < config::dim; d++) {
        if (difference[d] > 0.5) difference[d] -= 1.0;
        if (difference[d] < -0.5) difference[d] += 1.0;
    }
    return difference;
}

HD inline Vec wrapped_into_box(const Vec& p) {
    Vec wrapped = rounded_to_box_precision(p);
    for (int d = 0; d < config::dim; d++) {
        if (wrapped[d] < 0.0) wrapped[d] += 1.0;
        if (wrapped[d] >= 1.0) wrapped[d] -= 1.0;
    }
    return wrapped;
}

// rounds every coordinate in [-1, 2) to a multiple of 2^-51; adding +- 1 is then exact
HD inline Vec rounded_to_box_precision(Vec p) {
    for (int d = 0; d < config::dim; d++)
        p[d] = (p[d] >= 0.0) ? (p[d] + 2.0) - 2.0 : (p[d] - 2.0) + 2.0;
    return p;
}
