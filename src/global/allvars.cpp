#include "allvars.h"
#include "../io/input.h"
#include "../io/output.h"

#if defined(CPU_DEBUG) && !defined(USE_OPENMP)
int threadId;
#endif
// structs for input, output and IC handling
InputHandler  input;
ICData        icData;
OutputHandler output;
double        buff    = 0.5; // will be reduced once IC loaded
double        _gamma_ = 5. / 3.;

double CellShapingSpeed  = 0.5;
double CellShapingFactor = 1.0;

// computes the normal vector for a face between two seedpoints
double3 compute_face_normal(double3 seed_i, double3 seed_j) {
    double dx  = wrap_periodic_delta(seed_j.x - seed_i.x);
    double dy  = wrap_periodic_delta(seed_j.y - seed_i.y);
    double dz  = wrap_periodic_delta(seed_j.z - seed_i.z);
    double len = sqrt(dx * dx + dy * dy + dz * dz);
    return {dx / len, dy / len, dz / len};
}

// returns geometry (normalized basis)
geom compute_geom(double3 normal) {
    geom g;

    double nn = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    g.n       = {normal.x / nn, normal.y / nn, normal.z / nn};

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