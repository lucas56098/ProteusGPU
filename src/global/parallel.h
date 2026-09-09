#ifndef PARALLEL_H
#define PARALLEL_H
#pragma once

#include "../profiler/profiler.h"
#include "gpu_compat.h"

// Per-cell work is dispatched twice throughout the code — once as a CUDA kernel, once as
// an OpenMP loop — and the two differ only in scaffolding. parallel_for holds that
// scaffolding in one place: the caller passes the loop body as an HD lambda and it runs
// on whichever backend was compiled in.
//
//     parallel_for<_GRAD_BLOCK_SIZE_, 2>("GRAD_KERNEL", mesh->n_hydro,
//         [=] HD(size_t i) { compute_gradient_for_cell(i, mesh, primvar, grads); });
//
// The lambda's type is a template parameter, so the body inlines into the kernel exactly
// as a hand-written one does. A function pointer would cost an indirect call per cell.
//
// BLOCK is threads per block, MIN_BLOCKS the occupancy target — the two arguments that
// used to sit in LAUNCH_BOUNDS on each kernel. They are template parameters because
// __launch_bounds__ needs them at compile time.

#ifndef CPU_DEBUG

// the one copy of the index-guard boilerplate that every kernel body used to repeat
template <int BLOCK, int MIN_BLOCKS, typename F>
GLOBAL void LAUNCH_BOUNDS(BLOCK, MIN_BLOCKS) kernel_parallel_apply(size_t n, F f) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f(i);
}

#endif // !CPU_DEBUG

template <int BLOCK, int MIN_BLOCKS = 1, typename F> inline void parallel_for(const char* name, size_t n, F f) {
    (void)name; // unused when profiling is compiled out
    if (n == 0) return;

#ifndef CPU_DEBUG
    PROFILE_KERNEL(name);
    kernel_parallel_apply<BLOCK, MIN_BLOCKS><<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(n, f);
    GPU_SYNC();
#else
    // a CPU scope rather than a kernel one: same label, no GPU events to query
    PROFILE(name);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        f(i);
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

#endif // PARALLEL_H
