#ifndef GPU_COMPAT_H
#define GPU_COMPAT_H
#pragma once

// Backend layer: CPU_DEBUG and CUDA behind the same names (HD, GLOBAL, gpu_alloc, ...)

#include "../mpi/mpi_compat.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

typedef unsigned char uchar;

// CPU build: the device macros vanish, gpu_* is plain malloc / memcpy
#ifdef CPU_DEBUG
#define RUN_MODE "CPU"

#define HD
#define GLOBAL
#define GPU_SYNC()
#define LAUNCH_BOUNDS(threads, min_blocks)

#define CUDA_CHECK(call) ((void)0)

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
inline void gpu_prefetch_to_cpu(void*, size_t) {}
inline void gpu_prefetch_to_gpu(void*, size_t) {}

// the CUDA vector types, defined here for the CPU build
typedef struct {
    double x, y;
} double2;

typedef struct {
    double x, y, z;
} double3;

typedef struct {
    double x, y, z, w;
} double4_t;

typedef struct {
    uchar x, y;
} uchar2;

typedef struct {
    uchar x, y, z;
} uchar3;

typedef struct {
    int x, y;
} int2;

typedef struct {
    int x, y, z;
} int3;

#else
// CUDA build: managed memory, so one pointer is valid on host and device
#define RUN_MODE "GPU"

#define HD __host__ __device__
#define GLOBAL __global__

// min_blocks is dropped without CUDA_FAST_MATH: IEEE-faithful code needs more registers than that
// occupancy target allows, and nvlink then refuses to link
#ifdef CUDA_FAST_MATH
#define LAUNCH_BOUNDS(threads, min_blocks) __launch_bounds__(threads, min_blocks)
#else
#define LAUNCH_BOUNDS(threads, min_blocks) __launch_bounds__(threads)
#endif

#define GPU_SYNC()                                                                                                     \
    do {                                                                                                               \
        CUDA_CHECK(cudaPeekAtLastError());                                                                             \
        CUDA_CHECK(cudaDeviceSynchronize());                                                                           \
    } while (0)

// stops the run at the first CUDA error, naming file and line
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            mpi::exit_failure("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));               \
        }                                                                                                              \
    } while (0)

// bookkeeping for the GPU part of the memory report
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

inline void* gpu_malloc(size_t bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    g_gpu_allocs()[p] = bytes;
    g_gpu_bytes_current() += bytes;
    if (g_gpu_bytes_current() > g_gpu_bytes_peak()) { g_gpu_bytes_peak() = g_gpu_bytes_current(); }
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

// unified memory hints: keep these pages on the device
inline void gpu_advise_gpu_preferred(void* ptr, size_t bytes) {
    int dev;
    cudaGetDevice(&dev);
    cudaMemLocation loc = {};
    loc.type            = cudaMemLocationTypeDevice;
    loc.id              = dev;
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, loc));
    CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, loc));
}

inline void gpu_prefetch_to_cpu(void* ptr, size_t bytes) {
    cudaMemLocation loc = {};
    loc.type            = cudaMemLocationTypeHost;
    loc.id              = 0;
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0));
}

inline void gpu_prefetch_to_gpu(void* ptr, size_t bytes) {
    int dev;
    cudaGetDevice(&dev);
    cudaMemLocation loc = {};
    loc.type            = cudaMemLocationTypeDevice;
    loc.id              = dev;
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0));
}

#endif

// typed allocation
template <typename T> inline T* gpu_alloc(size_t count) {
    return static_cast<T*>(gpu_malloc(count * sizeof(T)));
}

template <typename T> inline T* gpu_calloc(size_t count) {
    T* p = gpu_alloc<T>(count);
    gpu_memset(p, 0, count * sizeof(T));
    return p;
}

// an array in gpu memory that grows when a call needs more and is kept for the next call; freed by hand,
// a destructor would run after the CUDA runtime is gone
template <typename T> struct GpuArray {
    T*     data     = nullptr;
    size_t capacity = 0;

    // room for n, at least double what it had; the content is not kept
    T* fit(size_t n) {
        if (n <= capacity) return data;
        if (data) gpu_free(data);
        capacity = (n > 2 * capacity) ? n : 2 * capacity;
        data     = gpu_alloc<T>(capacity);
        return data;
    }

    // the same, the content kept
    T* grow(size_t n) {
        if (n <= capacity) return data;
        const size_t new_capacity = (n > 2 * capacity) ? n : 2 * capacity;
        T*           p            = gpu_alloc<T>(new_capacity);
        if (data) {
            gpu_memcpy(p, data, capacity * sizeof(T));
            gpu_free(data);
        }
        data     = p;
        capacity = new_capacity;
        return data;
    }

    void free() {
        if (data) gpu_free(data);
        data     = nullptr;
        capacity = 0;
    }
};

// int min / max usable on the device
HD inline int imin(int a, int b) {
    return a < b ? a : b;
}
HD inline int imax(int a, int b) {
    return a > b ? a : b;
}

// first index of the sorted a[0, n) whose value is not below v, and the first one above v
template <typename T> HD inline size_t lower_bound_of(const T* a, size_t n, T v) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (a[mid] < v)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}
template <typename T> HD inline size_t upper_bound_of(const T* a, size_t n, T v) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (v < a[mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

#ifndef CPU_DEBUG
// CUDA 13 deprecates double4 in favour of double4_16a / double4_32a; we want the 16 byte aligned one
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13
typedef double4_16a double4_t;
#else
typedef double4 double4_t;
#endif
#endif

HD inline double4_t make_double4_t(double x, double y, double z, double w) {
    double4_t v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}

// dimension and the point / vertex types
#ifdef dim_2D
#define DIMENSION 2
typedef double2 POINT_TYPE;
typedef uchar2  VERT_TYPE;
typedef int2    BIG_VERT_TYPE;
#else
#define DIMENSION 3
typedef double3 POINT_TYPE;
typedef uchar3  VERT_TYPE;
typedef int3    BIG_VERT_TYPE;
#endif

// atomicAdd on the device, an OpenMP atomic on the host
template <typename T> HD inline T portable_atomicAdd(T* addr, T val) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(addr, val);
#else
    T old;
#ifdef USE_OPENMP
#pragma omp atomic capture
#endif
    {
        old = *addr;
        *addr += val;
    }
    return old;
#endif
}

// same for a bitwise or
HD inline unsigned int portable_atomicOr(unsigned int* addr, unsigned int val) {
#if defined(__CUDA_ARCH__)
    return atomicOr(addr, val);
#else
    return __atomic_fetch_or(addr, val, __ATOMIC_RELAXED);
#endif
}

// set bits of a word
HD inline int portable_popcount(unsigned int x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}

// same for an atomic exchange
template <typename T> HD inline T portable_atomicExch(T* addr, T val) {
#if defined(__CUDA_ARCH__)
    return atomicExch(addr, val);
#else
    T old;
#ifdef USE_OPENMP
#pragma omp atomic capture
#endif
    {
        old   = *addr;
        *addr = val;
    }
    return old;
#endif
}

#endif
