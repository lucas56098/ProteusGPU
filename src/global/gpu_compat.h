#ifndef GPU_COMPAT_H
#define GPU_COMPAT_H
#pragma once

// GPU type emulations for CPU_DEBUG mode, and dimension-dependent typedefs.

#ifdef CPU_DEBUG
// explicitly define types that exist in CUDA and HIP but not for CPU only

typedef struct {
    double x, y;
} double2;

typedef struct {
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

inline int atomicAdd(int* addr, int val) {
#ifdef USE_OPENMP
    int old;
#pragma omp atomic capture
    {
        old = *addr;
        *addr += val;
    }
    return old;
#else
    int old = *addr;
    *addr += val;
    return old;
#endif
}
#endif

// point and vertex types
#ifdef dim_2D
#define DIMENSION 2
typedef double2 POINT_TYPE;
typedef uchar2  VERT_TYPE;
#else
#define DIMENSION 3
typedef double3 POINT_TYPE;
typedef uchar3  VERT_TYPE;
#endif

// compact types for memory-sensitive arrays (gradients, face_area, f_mid)
#ifdef CPU_DEBUG
typedef struct {
    float x, y;
} float2;
typedef struct {
    float x, y, z;
} float3;
#endif

#ifdef SAVE_MEMORY
typedef float compact_t;
#ifdef dim_2D
typedef float2 GRAD_TYPE;
#else
typedef float3 GRAD_TYPE;
#endif
#else
typedef double     compact_t;
typedef POINT_TYPE GRAD_TYPE;
#endif

#endif // GPU_COMPAT_H
