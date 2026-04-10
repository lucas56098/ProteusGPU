#include "geometry.h"
#include "math_utils.h"
#include <cmath>

// computes orthonormal basis {n, m, p} from a raw (unnormalized) direction vector
geom compute_geom(double3 delta) {
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
