#ifndef GEOMETRY_H
#define GEOMETRY_H
#pragma once

#include "gpu_compat.h"
#include "structs.h"

// compute face normal and geometry basis
double3 compute_face_normal(double3 seed_i, double3 seed_j);
geom    compute_geom(double3 normal);

#endif // GEOMETRY_H
