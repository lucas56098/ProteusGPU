#ifndef GEOMETRY_H
#define GEOMETRY_H
#pragma once

#include "gpu_compat.h"
#include "structs.h"

// compute orthonormal basis from raw (unnormalized) direction vector
geom compute_geom(double3 delta);

#endif // GEOMETRY_H
