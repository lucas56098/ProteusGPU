#ifndef ALLVARS_H
#define ALLVARS_H
#include <cstddef>
#include <cstdlib>
#include <cstring>

// here should be some global structs and inline functions i guess
#ifdef CPU_DEBUG
typedef struct{
    double x, y;
} double2;

typedef struct{
    double x, y, z;
} double3;

struct double4 {
    double x, y, z, w;
};

inline double4 make_double4(double x, double y, double z, double w) {
    return {x, y, z, w};
}

typedef unsigned char uchar;
struct uchar3 {
    uchar x, y, z;
};

inline uchar3 make_uchar3(uchar x, uchar y, uchar z) {
    return {x, y, z};
}
#endif

#ifdef dim_2D
// code runs in 2D mode
#define DIMENSION 2
typedef double2 POINT_TYPE;
#else
// code runs in 3D mode
#define DIMENSION 3
typedef double3 POINT_TYPE;
#endif

#pragma once
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
    // for cuda memcpy needs to be split into cudaMemcpyHostToDevice and cudaMemcpyDeviceToHost, but for now we just do memcpy
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