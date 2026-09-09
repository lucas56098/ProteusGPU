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

#endif // PARALLEL_H
