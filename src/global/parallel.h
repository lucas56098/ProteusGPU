#ifndef PARALLEL_H
#define PARALLEL_H
#pragma once

#include "../profiler/profiler.h"
#include "gpu_compat.h"

// CPU scheduling policy
enum class Sched { Static, Dynamic };

#ifndef CPU_DEBUG

// the one copy of the index-guard boilerplate that every kernel body used to repeat
template <int BLOCK, int MIN_BLOCKS, typename F>
GLOBAL void LAUNCH_BOUNDS(BLOCK, MIN_BLOCKS) kernel_parallel_apply(size_t n, F f) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f(i);
}

#endif // !CPU_DEBUG

template <typename F> inline void cpu_for_static(size_t n, F f) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        f(i);
    }
}

template <typename F> inline void cpu_for_dynamic(size_t n, F f) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n; i++) {
        f(i);
    }
}

template <int BLOCK, int MIN_BLOCKS = 1, Sched SCHED = Sched::Static, typename F>
inline void parallel_for(const char* name, size_t n, F f) {
    (void)name; // unused when profiling is compiled out
    if (n == 0) return;

#ifndef CPU_DEBUG
    PROFILE_KERNEL(name);
    kernel_parallel_apply<BLOCK, MIN_BLOCKS><<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(n, f);
    GPU_SYNC();
#else
    // a CPU scope rather than a kernel one: same label, no GPU events to query
    PROFILE(name);

    if (SCHED == Sched::Dynamic) {
        cpu_for_dynamic(n, f);
    } else {
        cpu_for_static(n, f);
    }
#endif
}

inline size_t scan_scratch_size(size_t n, int block) {
    size_t total = 0;
    while (n > 1) {
        n = (n + (size_t)block - 1) / (size_t)block;
        total += n;
    }
    return total > 0 ? total : 1;
}

#ifndef CPU_DEBUG
template <int BLOCK, typename T> GLOBAL void kernel_scan_block(size_t n, const T* in, T* out, T* block_sums) {
    __shared__ T buf[2][BLOCK];
    const size_t i   = (size_t)blockIdx.x * BLOCK + threadIdx.x;
    const int    tid = threadIdx.x;

    const T v    = (i < n) ? in[i] : (T)0;
    int     pout = 0, pin = 1;
    buf[pout][tid] = v;
    __syncthreads();

    for (int offset = 1; offset < BLOCK; offset *= 2) {
        pin            = pout;
        pout           = 1 - pout;
        buf[pout][tid] = (tid >= offset) ? buf[pin][tid] + buf[pin][tid - offset] : buf[pin][tid];
        __syncthreads();
    }

    const T incl = buf[pout][tid];
    if (i < n) out[i] = incl - v; // inclusive -> exclusive
    if (tid == BLOCK - 1) block_sums[blockIdx.x] = incl;
}

template <int BLOCK, typename T> GLOBAL void kernel_scan_add(size_t n, T* out, const T* block_offsets) {
    const size_t i = (size_t)blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) out[i] += block_offsets[blockIdx.x];
}

template <int BLOCK, typename T> inline void scan_device(size_t n, const T* in, T* out, T* scratch) {
    const size_t nb = (n + BLOCK - 1) / BLOCK;
    kernel_scan_block<BLOCK, T><<<(unsigned int)nb, BLOCK>>>(n, in, out, scratch);
    GPU_SYNC();
    if (nb > 1) {
        scan_device<BLOCK, T>(nb, scratch, scratch, scratch + nb);
        kernel_scan_add<BLOCK, T><<<(unsigned int)nb, BLOCK>>>(n, out, scratch);
        GPU_SYNC();
    }
}

#endif // !CPU_DEBUG

template <int BLOCK, typename T>
inline void parallel_exclusive_scan(const char* name, size_t n, const T* in, T* out, T* scratch) {
    (void)name;
    (void)scratch;
    if (n == 0) return;

#ifndef CPU_DEBUG
    PROFILE_KERNEL(name);
    scan_device<BLOCK, T>(n, in, out, scratch);
#else
    PROFILE(name);
    const size_t CHUNKS = 1024;
    const size_t chunk  = (n + CHUNKS - 1) / CHUNKS;
    T            totals[CHUNKS];

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < CHUNKS; c++) {
        const size_t lo = c * chunk;
        const size_t hi = (lo + chunk < n) ? lo + chunk : n;
        T            s  = 0;
        for (size_t i = lo; i < hi; i++) {
            s += in[i];
        }
        totals[c] = s;
    }

    T running = 0;
    for (size_t c = 0; c < CHUNKS; c++) {
        const T t = totals[c];
        totals[c] = running;
        running += t;
    }

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < CHUNKS; c++) {
        const size_t lo = c * chunk;
        const size_t hi = (lo + chunk < n) ? lo + chunk : n;
        T            r  = totals[c];
        for (size_t i = lo; i < hi; i++) {
            const T v = in[i]; // read before write, so in == out is safe
            out[i]    = r;
            r += v;
        }
    }
#endif
}

template <typename T> inline T* reduce_scratch(size_t need) {
    static T*     buf = nullptr;
    static size_t cap = 0;
    if (need > cap) {
        if (buf) gpu_free(buf);
        buf = gpu_alloc<T>(need);
        cap = need;
    }
    return buf;
}

#ifndef CPU_DEBUG
template <int BLOCK, typename T, typename Op, typename F>
GLOBAL void kernel_reduce_level(size_t n, T identity, Op op, F f, T* out) {
    __shared__ T buf[BLOCK];
    const int    tid = threadIdx.x;
    const size_t i   = (size_t)blockIdx.x * BLOCK + (size_t)tid;

    buf[tid] = (i < n) ? f(i) : identity;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = op(buf[tid], buf[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = buf[0];
}
#endif // !CPU_DEBUG

template <int BLOCK, typename T, typename Op, typename F>
inline void reduce_level(size_t n, T identity, Op op, F f, T* out) {
    const size_t nb = (n + BLOCK - 1) / BLOCK;
#ifndef CPU_DEBUG
    kernel_reduce_level<BLOCK, T><<<(unsigned int)nb, BLOCK>>>(n, identity, op, f, out);
    GPU_SYNC();
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t b = 0; b < nb; b++) {
        T buf[BLOCK];
        for (int t = 0; t < BLOCK; t++) {
            const size_t i = b * (size_t)BLOCK + (size_t)t;
            buf[t]         = (i < n) ? f(i) : identity;
        }
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            for (int t = 0; t < s; t++) {
                buf[t] = op(buf[t], buf[t + s]);
            }
        }
        out[b] = buf[0];
    }
#endif
}

template <int BLOCK, typename T, typename Op, typename F>
inline T parallel_reduce(const char* name, size_t n, T identity, Op op, F f) {
    static_assert(BLOCK >= 2 && (BLOCK & (BLOCK - 1)) == 0, "parallel_reduce needs a power-of-two BLOCK >= 2");
    (void)name;
    if (n == 0) return identity;

#ifndef CPU_DEBUG
    PROFILE_KERNEL(name);
#else
    PROFILE(name);
#endif

    T* out = reduce_scratch<T>(scan_scratch_size(n, BLOCK));

    reduce_level<BLOCK, T>(n, identity, op, f, out);
    size_t cur = (n + BLOCK - 1) / BLOCK;

    while (cur > 1) {
        const T* in   = out;
        T*       next = out + cur;
        reduce_level<BLOCK, T>(cur, identity, op, [in] HD(size_t i) { return in[i]; }, next);
        out = next;
        cur = (cur + BLOCK - 1) / BLOCK;
    }
    return out[0];
}

template <int BLOCK, typename T, typename F> inline T parallel_reduce_sum(const char* name, size_t n, F f) {
    return parallel_reduce<BLOCK, T>(name, n, (T)0, [] HD(T a, T b) { return a + b; }, f);
}

#endif // PARALLEL_H
