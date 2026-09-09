#ifndef ALLVARS_H
#define ALLVARS_H
#pragma once
#include "globals.h"    // stores global variables
#include "gpu_compat.h" // helpers for GPU/CPU compatibility
#include "log.h"        // root-only / MPI-aggregated logging wrappers
#include "math_utils.h" // math helpers
#include "parallel.h"   // one dispatch for per-cell work, CUDA kernel or OpenMP loop
#include "structs.h"    // globally used structs
#include "units.h"      // code-unit system (astro modules convert against it)

#endif // ALLVARS_H