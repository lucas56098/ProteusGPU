#ifndef PARALLEL_H
#define PARALLEL_H
#pragma once

// Launches per-cell work: parallel_for, parallel_reduce, parallel_exclusive_scan and parallel_sort_pairs,
// one kernel on CUDA and an OpenMP loop on CPU. A device lambda cannot capture a namespace-scope global, so copy such a
// global into a local before using it in a body.

#include "../profiler/profiler.h"
#include "gpu_compat.h"
#include <cstdint>
#include <utility>

// CPU loop schedule; dynamic pays off where the cost per cell varies a lot
enum class Sched { Static, Dynamic };

#ifndef CPU_DEBUG

template <int BLOCK, int MIN_BLOCKS, typename F>
// the generic kernel behind parallel_for; F is a template parameter, so the body inlines like a hand-written one
GLOBAL void LAUNCH_BOUNDS(BLOCK, MIN_BLOCKS) kernel_parallel_apply(size_t n, F f) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f(i);
}

#endif

// plain loops, threaded only with USE_OPENMP
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
// runs f(i) for every i in [0, n)
inline void parallel_for(const char* name, size_t n, F f) {
    (void)name;
    if (n == 0) return;

#ifndef CPU_DEBUG
    PROFILE_KERNEL(name);
    kernel_parallel_apply<BLOCK, MIN_BLOCKS><<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(n, f);
    GPU_SYNC();
#else
    PROFILE(name);

    if (SCHED == Sched::Dynamic) {
        cpu_for_dynamic(n, f);
    } else {
        cpu_for_static(n, f);
    }
#endif
}

// scratch elements a scan over n values needs
inline size_t scan_scratch_size(size_t n, int block) {
    size_t total = 0;
    while (n > 1) {
        n = (n + (size_t)block - 1) / (size_t)block;
        total += n;
    }
    return total > 0 ? total : 1;
}

#ifndef CPU_DEBUG
// three phases: scan inside each block, recurse on the block totals, add the offsets back
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
    if (i < n) out[i] = incl - v;
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

#endif

template <int BLOCK, typename T>
// exclusive prefix sum of in into out; in == out is allowed
// take offsets from here whenever they must be reproducible, an atomic cursor hands them out in race order
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
            const T v = in[i];
            out[i]    = r;
            r += v;
        }
    }
#endif
}

// scratch of the reductions, grown on demand and kept
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
// one level of the tree: BLOCK values per block, folded pairwise
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
#endif

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
// folds f(i) over [0, n) with op; identity pads the last block
// the tree shape follows the index alone, so the result is the same on CPU and GPU and for any thread count
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

// sum wrapper
template <int BLOCK, typename T, typename F> inline T parallel_reduce_sum(const char* name, size_t n, F f) {
    return parallel_reduce<BLOCK, T>(name, n, (T)0, [] HD(T a, T b) { return a + b; }, f);
}

// ============================================================================
// stable radix sort of (key, value) pairs
// ============================================================================

constexpr int    SORT_RADIX_BITS    = 8;
constexpr int    SORT_RADIX         = 1 << SORT_RADIX_BITS; // also the CUDA block size
constexpr int    SORT_ITEMS         = 8;                    // keys per thread and tile on CUDA
constexpr size_t SORT_TILE          = (size_t)SORT_RADIX * SORT_ITEMS;
constexpr size_t SORT_CPU_CHUNKS    = 1024; // CPU: chunks of SORT_CPU_MIN_CHUNK keys, at most this many
constexpr size_t SORT_CPU_MIN_CHUNK = 4096;
constexpr int    SORT_CUDA_WARPS    = SORT_RADIX / 32;
constexpr size_t SORT_CPU_SCRATCH   = (size_t)SORT_RADIX * SORT_CPU_CHUNKS;

// scratch elements a sort of n pairs needs
inline size_t sort_scratch_size(size_t n) {
    const size_t tiles = (n + SORT_TILE - 1) / SORT_TILE;
    const size_t hist  = (size_t)SORT_RADIX * (tiles > 0 ? tiles : 1);
    const size_t gpu   = hist + scan_scratch_size(hist, SORT_RADIX);
    return gpu > SORT_CPU_SCRATCH ? gpu : SORT_CPU_SCRATCH;
}

#ifndef CPU_DEBUG
// keys per digit in each tile, digit-major so the scan gives every (digit, tile) its first slot
static GLOBAL void LAUNCH_BOUNDS(SORT_RADIX, 1)
    kernel_sort_count(size_t n, int shift, const uint64_t* keys, unsigned int* hist, size_t tiles) {
    __shared__ unsigned int s_hist[SORT_RADIX];
    const int               t = threadIdx.x;
    s_hist[t]                 = 0;
    __syncthreads();

    const size_t base = (size_t)blockIdx.x * SORT_TILE;
    for (int j = 0; j < SORT_ITEMS; j++) {
        const size_t i = base + (size_t)j * SORT_RADIX + (size_t)t;
        if (i < n) atomicAdd(&s_hist[(keys[i] >> shift) & (SORT_RADIX - 1)], 1u);
    }
    __syncthreads();
    hist[(size_t)t * tiles + blockIdx.x] = s_hist[t];
}

// moves each pair to its slot; inside a tile the rank of a key among the same digit follows the input order
static GLOBAL void LAUNCH_BOUNDS(SORT_RADIX, 1) kernel_sort_scatter(size_t              n,
                                                                    int                 shift,
                                                                    const uint64_t*     keys,
                                                                    const unsigned int* vals,
                                                                    uint64_t*           keys_out,
                                                                    unsigned int*       vals_out,
                                                                    const unsigned int* first_slot,
                                                                    size_t              tiles) {
    __shared__ unsigned int s_next[SORT_RADIX];
    __shared__ unsigned int s_warp[SORT_CUDA_WARPS][SORT_RADIX];
    const int               t    = threadIdx.x;
    const int               lane = t & 31;
    const int               w    = t >> 5;
    s_next[t]                    = first_slot[(size_t)t * tiles + blockIdx.x];

    // one key per thread and round, rounds in input order
    const size_t base = (size_t)blockIdx.x * SORT_TILE;
    for (int j = 0; j < SORT_ITEMS; j++) {
        const size_t i     = base + (size_t)j * SORT_RADIX + (size_t)t;
        const bool   valid = i < n;
        uint64_t     key   = 0;
        unsigned int val   = 0;
        int          d     = SORT_RADIX; // matches no real digit
        if (valid) {
            key = keys[i];
            val = vals[i];
            d   = (int)((key >> shift) & (SORT_RADIX - 1));
        }
        for (int ww = 0; ww < SORT_CUDA_WARPS; ww++)
            s_warp[ww][t] = 0;
        __syncthreads();

        // rank inside the warp, and how many of each digit every warp has
        const unsigned int peers = __match_any_sync(0xffffffffu, d);
        const int          below = __popc(peers & ((1u << lane) - 1u));
        if (valid && below == 0) s_warp[w][d] = (unsigned int)__popc(peers);
        __syncthreads();

        if (valid) {
            unsigned int pos = s_next[d] + (unsigned int)below;
            for (int ww = 0; ww < w; ww++)
                pos += s_warp[ww][d];
            keys_out[pos] = key;
            vals_out[pos] = val;
        }
        __syncthreads();

        unsigned int total = 0;
        for (int ww = 0; ww < SORT_CUDA_WARPS; ww++)
            total += s_warp[ww][t];
        s_next[t] += total;
    }
}
#endif

// sorts the pairs by the low key_bits of the key; equal keys keep their input order, so the result
// is unique and the same on CPU and GPU. The pointers may come back swapped with the _alt ones.
inline void parallel_sort_pairs(const char*    name,
                                size_t         n,
                                int            key_bits,
                                uint64_t*&     keys,
                                unsigned int*& vals,
                                uint64_t*&     keys_alt,
                                unsigned int*& vals_alt,
                                unsigned int*  scratch) {
    (void)name;
    if (n <= 1) return;
    const int passes = (key_bits + SORT_RADIX_BITS - 1) / SORT_RADIX_BITS;

#ifndef CPU_DEBUG
    PROFILE_KERNEL(name);
    const size_t  tiles = (n + SORT_TILE - 1) / SORT_TILE;
    const size_t  hist  = (size_t)SORT_RADIX * tiles;
    unsigned int* scan  = scratch + hist;
    for (int p = 0; p < passes; p++) {
        const int shift = p * SORT_RADIX_BITS;
        kernel_sort_count<<<(unsigned int)tiles, SORT_RADIX>>>(n, shift, keys, scratch, tiles);
        GPU_SYNC();
        scan_device<SORT_RADIX, unsigned int>(hist, scratch, scratch, scan);
        kernel_sort_scatter<<<(unsigned int)tiles, SORT_RADIX>>>(
            n, shift, keys, vals, keys_alt, vals_alt, scratch, tiles);
        GPU_SYNC();
        std::swap(keys, keys_alt);
        std::swap(vals, vals_alt);
    }
#else
    PROFILE(name);
    // a chunk costs a table of SORT_RADIX counts per pass, so few keys get few chunks; stable, so the
    // result is the same for every chunk count
    const size_t want   = (n + SORT_CPU_MIN_CHUNK - 1) / SORT_CPU_MIN_CHUNK;
    const size_t chunks = (want < SORT_CPU_CHUNKS) ? want : SORT_CPU_CHUNKS;
    const size_t chunk  = (n + chunks - 1) / chunks;
    for (int p = 0; p < passes; p++) {
        const int           shift = p * SORT_RADIX_BITS;
        const uint64_t*     k_in  = keys;
        const unsigned int* v_in  = vals;
        uint64_t*           k_out = keys_alt;
        unsigned int*       v_out = vals_alt;

        // keys per digit in each chunk, digit-major
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t c = 0; c < chunks; c++) {
            unsigned int count[SORT_RADIX] = {0};
            const size_t lo                = c * chunk;
            const size_t hi                = (lo + chunk < n) ? lo + chunk : n;
            for (size_t i = lo; i < hi; i++)
                count[(k_in[i] >> shift) & (SORT_RADIX - 1)]++;
            for (int d = 0; d < SORT_RADIX; d++)
                scratch[(size_t)d * chunks + c] = count[d];
        }

        unsigned int running = 0;
        for (size_t e = 0; e < (size_t)SORT_RADIX * chunks; e++) {
            const unsigned int v = scratch[e];
            scratch[e]           = running;
            running += v;
        }

        // each chunk fills its slots in input order
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t c = 0; c < chunks; c++) {
            unsigned int next[SORT_RADIX];
            for (int d = 0; d < SORT_RADIX; d++)
                next[d] = scratch[(size_t)d * chunks + c];
            const size_t lo = c * chunk;
            const size_t hi = (lo + chunk < n) ? lo + chunk : n;
            for (size_t i = lo; i < hi; i++) {
                const unsigned int pos = next[(k_in[i] >> shift) & (SORT_RADIX - 1)]++;
                k_out[pos]             = k_in[i];
                v_out[pos]             = v_in[i];
            }
        }
        std::swap(keys, keys_alt);
        std::swap(vals, vals_alt);
    }
#endif
}

// the arrays of one parallel_sort_pairs: fill keys and vals, sort, then they hold the sorted pairs
struct PairSort {
    uint64_t*     keys     = nullptr;
    unsigned int* vals     = nullptr;
    uint64_t*     keys_alt = nullptr;
    unsigned int* vals_alt = nullptr;
    unsigned int* scratch  = nullptr;

    void sort(const char* name, size_t n, int key_bits) {
        parallel_sort_pairs(name, n, key_bits, keys, vals, keys_alt, vals_alt, scratch);
    }
};

// arrays a PairSort can be made of, kept between calls
struct PairSortArrays {
    GpuArray<uint64_t>     keys, keys_alt;
    GpuArray<unsigned int> vals, vals_alt, scratch;

    // a sort of n pairs on these arrays
    PairSort fit(size_t n) {
        PairSort s;
        s.keys     = keys.fit(n);
        s.keys_alt = keys_alt.fit(n);
        s.vals     = vals.fit(n);
        s.vals_alt = vals_alt.fit(n);
        s.scratch  = scratch.fit(sort_scratch_size(n));
        return s;
    }

    void free() {
        keys.free();
        keys_alt.free();
        vals.free();
        vals_alt.free();
        scratch.free();
    }
};

#endif
