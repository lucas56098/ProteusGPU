#ifndef GPU_COMPAT_H
#define GPU_COMPAT_H
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

// CUDA / CPU_DEBUG mode switching

#ifdef CPU_DEBUG

// wrappers and empty defines
#define HD
#define DEVICE
#define GLOBAL
#define GPU_SYNC()

#define CUDA_CHECK(call) ((void)0)

// memory wrappers
inline void* gpu_malloc(size_t bytes) {
    return malloc(bytes);
}
inline void gpu_free(void* ptr) {
    free(ptr);
}
inline void gpu_memset(void* ptr, int val, size_t bytes) {
    memset(ptr, val, bytes);
}
inline void gpu_memcpy(void* dst, const void* src, size_t bytes) {
    memcpy(dst, src, bytes);
}
inline void gpu_advise_gpu_preferred(void*, size_t) {}
inline void gpu_prefetch(void*, size_t) {}
inline void gpu_prefetch_to_cpu(void*, size_t) {}

// emulate CUDA types
// float
typedef struct {
    float x, y;
} float2;
typedef struct {
    float x, y, z;
} float3;

// double
typedef struct {
    double x, y;
} double2;

typedef struct {
    double x, y, z;
} double3;

typedef struct {
    double x, y, z, w;
} double4;

inline double4 make_double4(double x, double y, double z, double w) {
    return {x, y, z, w};
}

// char
typedef unsigned char uchar;

typedef struct {
    uchar x, y;
} uchar2;

inline uchar2 make_uchar2(uchar x, uchar y) {
    return {x, y};
}

typedef struct {
    uchar x, y, z;
} uchar3;

inline uchar3 make_uchar3(uchar x, uchar y, uchar z) {
    return {x, y, z};
}

// emulate atomic add
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

inline int host_atomicAdd(int* addr, int val) {
    return atomicAdd(addr, val);
}

#else // CUDA mode

// kernel/function macros
#define HD __host__ __device__
#define DEVICE __device__
#define GLOBAL __global__

// syncs and error checking
#define GPU_SYNC()
    do {
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } while (0)

#define GPU_LAUNCH_CHECK() CUDA_CHECK(cudaPeekAtLastError())

#define CUDA_CHECK(call)
    do {
        cudaError_t err = (call);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    } while (0)

inline size_t& g_gpu_bytes_current() {
    static size_t v = 0;
    return v;
}
inline size_t& g_gpu_bytes_peak() {
    static size_t v = 0;
    return v;
}
inline std::unordered_map<void*, size_t>& g_gpu_allocs() {
    static std::unordered_map<void*, size_t> m;
    return m;
}

// memory wrappers
inline void* gpu_malloc(size_t bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    g_gpu_allocs()[p] = bytes;
    g_gpu_bytes_current() += bytes;
    if (g_gpu_bytes_current() > g_gpu_bytes_peak()) {
        g_gpu_bytes_peak() = g_gpu_bytes_current();
    }
    return p;
}
inline void gpu_free(void* ptr) {
    if (ptr) {
        auto& m  = g_gpu_allocs();
        auto  it = m.find(ptr);
        if (it != m.end()) {
            g_gpu_bytes_current() -= it->second;
            m.erase(it);
        }
    }
    CUDA_CHECK(cudaFree(ptr));
}
inline void gpu_memset(void* ptr, int val, size_t bytes) {
    CUDA_CHECK(cudaMemset(ptr, val, bytes));
}
inline void gpu_memcpy(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
}

// advice to store on gpu
inline void gpu_advise_gpu_preferred(void* ptr, size_t bytes) {
    int dev;
    cudaGetDevice(&dev);
#if CUDART_VERSION >= 12000
    cudaMemLocation loc = {};
    loc.type = cudaMemLocationTypeDevice;
    loc.id = dev;
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, loc));
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, loc));
#else
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, dev));
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, dev));
#endif
}

// prefetch managed memory to the GPU
inline void gpu_prefetch(void* ptr, size_t bytes) {
    int dev;
    cudaGetDevice(&dev);
#if CUDART_VERSION >= 12000
    cudaMemLocation loc = {};
    loc.type = cudaMemLocationTypeDevice;
    loc.id = dev;
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0));
#else
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, dev, 0));
#endif
}
// prefetch managed memory to the CPU
inline void gpu_prefetch_to_cpu(void* ptr, size_t bytes) {
#if CUDART_VERSION >= 12000
    cudaMemLocation loc = {};
    loc.type = cudaMemLocationTypeHost;
    loc.id = 0;
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0));
#else
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, 0));
#endif
}

typedef unsigned char uchar;

inline int host_atomicAdd(int* addr, int val) {
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

#endif // CPU_DEBUG

// allocation helpers
template <typename T> inline T* gpu_alloc(size_t count) {
    return static_cast<T*>(gpu_malloc(count * sizeof(T)));
}

template <typename T> inline T* gpu_calloc(size_t count) {
    T* p = gpu_alloc<T>(count);
    gpu_memset(p, 0, count * sizeof(T));
    return p;
}

// integer min/max
HD inline int imin(int a, int b) {
    return a < b ? a : b;
}
HD inline int imax(int a, int b) {
    return a > b ? a : b;
}

// typedefs
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

// atomic add/exch that work on host and device
HD inline unsigned long long portable_atomicAdd(unsigned long long* addr, unsigned long long val) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(addr, val);
#elif defined(USE_OPENMP)
    unsigned long long old;
#pragma omp atomic capture
    {
        old = *addr;
        *addr += val;
    }
    return old;
#else
    unsigned long long old = *addr;
    *addr += val;
    return old;
#endif
}

HD inline int portable_atomicExch(int* addr, int val) {
#if defined(__CUDA_ARCH__)
    return atomicExch(addr, val);
#elif defined(USE_OPENMP)
    int old;
#pragma omp atomic capture
    {
        old   = *addr;
        *addr = val;
    }
    return old;
#else
    int old = *addr;
    *addr   = val;
    return old;
#endif
}

#endif // GPU_COMPAT_H
