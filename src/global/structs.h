#ifndef STRUCTS_H
#define STRUCTS_H
#pragma once

#include "gpu_compat.h"

// status codes for voronoi mesh generation
enum Status {
    triangle_overflow           = 0,
    vertex_overflow             = 1,
    inconsistent_boundary       = 2,
    security_radius_not_reached = 3,
    success                     = 4,
    needs_exact_predicates      = 5
};

// hydro primitive variable arrays (SoA layout)
struct primvars {
    double*     rho;
    POINT_TYPE* v;
    double*     E; // energy per unit volume (so in the state vector we have E = rho e)
};

// single-cell primitive state
struct prim {
    double rho = 0;
#ifdef dim_2D
    POINT_TYPE v = {0., 0.};
#else
    POINT_TYPE v = {0., 0., 0.};
#endif
    double E = 0;
};

// gradient data per cell
struct PrimGradients {
    GRAD_TYPE rho;
    GRAD_TYPE vx;
    GRAD_TYPE vy;
#ifdef dim_3D
    GRAD_TYPE vz;
#endif
    GRAD_TYPE E;
};

// geometry (normalized basis for face orientation)
struct geom {
    double3 n; // normal
    double3 m; // 1. tangential
    double3 p; // 2. tangential
};

#endif // STRUCTS_H
