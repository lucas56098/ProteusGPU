#ifndef ALLVARS_H
#define ALLVARS_H
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// global defines, structs and inline functions.

// eventually i should split these into data_types, gpu_utils, math_utils :D

enum Status {
        triangle_overflow = 0,
        vertex_overflow = 1,
        inconsistent_boundary = 2,
        security_radius_not_reached = 3,
        success = 4,
        needs_exact_predicates = 5
    };

#ifdef CPU_DEBUG
// explicitly define types that exist in CUDA and HIP but not for CPU only

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

struct uchar2 {
    uchar x, y;
};

inline uchar2 make_uchar2(uchar x, uchar y) {
    return {x, y};
}

struct int3 {
    int x, y, z;
};
#endif

// point and vertex types
#ifdef dim_2D
#define DIMENSION 2
typedef double2 POINT_TYPE;
typedef uchar2 VERT_TYPE;
#else
#define DIMENSION 3
typedef double3 POINT_TYPE;
typedef uchar3 VERT_TYPE;
#endif

#ifdef CPU_DEBUG
extern int3 blockId;
extern int3 threadId;
#endif

// structs for input, output and IC handling
class InputHandler;
struct ICData;
class OutputHandler;

extern InputHandler input;
extern ICData icData;
extern OutputHandler output;
extern double buff; // buffer for the periodic bc (box will then be 1 + 2*buff long)

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
    // for cuda memcpy needs to be split into cudaMemcpyHostToDevice and cudaMemcpyDeviceToHost
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

// further helpers for voronoi mesh generation
inline double4 minus4(double4 A, double4 B) {
    return make_double4(A.x-B.x, A.y-B.y, A.z-B.z, A.w-B.w);
}
inline double4 plus4(double4 A, double4 B) {
    return make_double4(A.x+B.x, A.y+B.y, A.z+B.z, A.w+B.w);
}
inline double dot4(double4 A, double4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z + A.w*B.w;
}
inline double dot3(double4 A, double4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z;
}
inline double4 mul3(double s, double4 A) {
    return make_double4(s*A.x, s*A.y, s*A.z, 1.);
}
inline double4 cross3(double4 A, double4 B) {
    return make_double4(A.y*B.z - A.z*B.y, A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x, 0);
}

inline double4 point_from_ptr(double* f) {
#ifdef dim_2D
    return make_double4(f[0], f[1], 0, 1);
#else
    return make_double4(f[0], f[1], f[2], 1);
#endif
}

    inline double det2x2(double a11, double a12, double a21, double a22) {
        return a11*a22 - a12*a21;
    }

    inline double det3x3(double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33) {
        return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
    }

    inline double det4x4(
        double a11, double a12, double a13, double a14,
        double a21, double a22, double a23, double a24,               
        double a31, double a32, double a33, double a34,  
        double a41, double a42, double a43, double a44) {

        double m12 = a21*a12 - a11*a22;
        double m13 = a31*a12 - a11*a32;
        double m14 = a41*a12 - a11*a42;
        double m23 = a31*a22 - a21*a32;
        double m24 = a41*a22 - a21*a42;
        double m34 = a41*a32 - a31*a42;
    
        double m123 = m23*a13 - m13*a23 + m12*a33;
        double m124 = m24*a13 - m14*a23 + m12*a43;
        double m134 = m34*a13 - m14*a33 + m13*a43;
        double m234 = m34*a23 - m24*a33 + m23*a43;
    
        return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
    }

    inline void get_minmax3(double& m, double& M, double x1, double x2, double x3) {
        m = std::min(std::min(x1, x2), x3);
        M = std::max(std::max(x1, x2), x3);
    }


#endif // ALLVARS_H