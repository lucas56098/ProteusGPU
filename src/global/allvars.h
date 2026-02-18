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
extern double _boxsize_;

// abstraction layer to later switch between CPU_DEBUG, CUDA and HIP defines
// for now just CPU stuff
inline void gpuMalloc(void **ptr, size_t size) {
#ifdef CPU_DEBUG
    *ptr = malloc(size);
#endif
}

inline void gpuMallocNCopy(void **dst, const void *src, size_t size) {
#ifdef CPU_DEBUG    
    *dst = malloc(size);
    memcpy(*dst, src, size);
#endif
}

inline void gpuMemcpy(void *dst, const void *src, size_t size) {
#ifdef CPU_DEBUG    
    memcpy(dst, src, size);
#endif
}

inline void gpuMallocNMemset(void **ptr, int value, size_t size) {
#ifdef CPU_DEBUG
    *ptr = malloc(size);
    memset(*ptr, value, size);
#endif
}

inline void gpuMemset(void *ptr, int value, size_t size) {
#ifdef CPU_DEBUG
    memset(ptr, value, size);
#endif
}

inline void gpuFree(void *ptr) {
#ifdef CPU_DEBUG
    free(ptr);
#endif
}

#ifdef CPU_DEBUG
inline int atomicAdd(int* addr, int val)
{
    int old = *addr;
    *addr += val;
    return old;
}
#endif

#endif // ALLVARS_H