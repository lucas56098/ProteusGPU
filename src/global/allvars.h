#ifndef ALLVARS_H
#define ALLVARS_H
#include <cstddef>
#include <cstdlib>
#include <cstring>

// here should be some global structs and inline functions i guess

typedef struct{
    double x, y, z;
} double3;

#ifdef dim_2D
// code runs in 2D mode
#define DIMENSION 2
#else
// code runs in 3D mode
#define DIMENSION 3
#endif

#pragma once
extern int _K_;

// abstraction layer to later switch between CPU_DEBUG, CUDA and HIP defines
// for now just CPU stuff
inline void gpuMalloc(void **ptr, size_t size) {
    *ptr = malloc(size);
}

inline void gpuMallocNCopy(void **dst, const void *src, size_t size) {
    *dst = malloc(size);
    memcpy(*dst, src, size);
}

inline void gpuMallocNMemset(void **ptr, int value, size_t size) {
    *ptr = malloc(size);
    memset(*ptr, value, size);
}

inline void gpuFree(void *ptr) {
    free(ptr);
}

#endif // ALLVARS_H