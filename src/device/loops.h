#pragma once

// the loops and what they use (included by device.h)

#include <limits>
#include <type_traits>

namespace device {

    // threads per GPU block, and values per block of the reduction tree
    constexpr int BLOCK_SIZE = 256;

#ifdef CUDA
    template <typename Function> __global__ void call_function_per_gpu_thread(long n, Function function);
    template <int THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MULTIPROCESSOR, typename Function>
    __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
        call_function_per_gpu_thread_with_occupancy(long n, Function function);
    template <typename Result, typename ValueFunction, typename CombineFunction>
    __global__ void reduce_one_block_per_gpu_block(long            n,
                                                   Result          start,
                                                   ValueFunction   value_of_index,
                                                   CombineFunction combine,
                                                   Result*         results);
#endif
    template <typename Number> struct Plus;
    template <typename Number> struct SmallerOfTwo;
    template <typename Number> struct LargerOfTwo;
    template <typename Result, typename ValueFunction, typename CombineFunction>
    void reduce_one_level(long n, Result start, ValueFunction value_of_index, CombineFunction combine, Result* results);

    // defined in device.cu, for the files of device only

    // waits for the GPU; stops the run if its work failed
    void wait_for_gpu(const char* what);

    // temporary memory for one loop; counts towards the memory limit
    void* allocate_temporary(long bytes);
    void free_temporary(void* memory, long bytes);

#ifdef CUDA
    void stop_on_cuda_error(cudaError_t result, const char* what);
#endif

    template <typename Function> void parallel_for(long n, Function function) {
        if (n <= 0) return;
#ifdef CUDA
        const timers::Scope timer("GPU", timers::Type::gpu);
        call_function_per_gpu_thread<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n, function);
        wait_for_gpu("parallel_for");
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < n; i++)
            function(i);
#endif
    }

#ifdef CUDA
    template <typename Function> __global__ void call_function_per_gpu_thread(long n, Function function) {
        const long i = (long)blockIdx.x * BLOCK_SIZE + threadIdx.x;
        if (i < n) function(i);
    }
#endif

    // on the CPU as parallel_for_on_cpu
    template <int THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MULTIPROCESSOR, typename Function>
    void parallel_for(long n, Function function) {
        if (n <= 0) return;
#ifdef CUDA
        const timers::Scope timer("GPU", timers::Type::gpu);
        call_function_per_gpu_thread_with_occupancy<THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR>
            <<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(n, function);
        wait_for_gpu("parallel_for with a given occupancy");
#else
        parallel_for_on_cpu(n, function);
#endif
    }

#ifdef CUDA
    template <int THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MULTIPROCESSOR, typename Function>
    __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
        call_function_per_gpu_thread_with_occupancy(long n, Function function) {
        const long i = (long)blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
        if (i < n) function(i);
    }
#endif

    // threads take one i at a time, as the work per i can differ a lot
    template <typename Function> void parallel_for_on_cpu(long n, Function function) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (long i = 0; i < n; i++)
            function(i);
    }

    template <typename ValueFunction> auto sum(long n, ValueFunction value_of_index) {
        using Number = decltype(value_of_index(0L));
        return reduce(n, Number(0), value_of_index, Plus<Number>{});
    }

    template <typename ValueFunction> auto min(long n, ValueFunction value_of_index) {
        using Number         = decltype(value_of_index(0L));
        using Limits         = std::numeric_limits<Number>;
        const Number largest = Limits::has_infinity ? Limits::infinity() : Limits::max();
        return reduce(n, largest, value_of_index, SmallerOfTwo<Number>{});
    }

    template <typename ValueFunction> auto max(long n, ValueFunction value_of_index) {
        using Number          = decltype(value_of_index(0L));
        using Limits          = std::numeric_limits<Number>;
        const Number smallest = Limits::has_infinity ? -Limits::infinity() : Limits::lowest();
        return reduce(n, smallest, value_of_index, LargerOfTwo<Number>{});
    }

    template <typename Number> struct Plus {
        HD Number operator()(Number a, Number b) const { return a + b; }
    };
    template <typename Number> struct SmallerOfTwo {
        HD Number operator()(Number a, Number b) const { return (b < a) ? b : a; }
    };
    template <typename Number> struct LargerOfTwo {
        HD Number operator()(Number a, Number b) const { return (b > a) ? b : a; }
    };

    // level by level: every BLOCK_SIZE values become one, until one is left
    template <typename Result, typename ValueFunction, typename CombineFunction>
    Result reduce(long n, Result start, ValueFunction value_of_index, CombineFunction combine) {
        static_assert(std::is_trivially_copyable<Result>::value, "a reduction combines plain values");
        if (n <= 0) return start;

        long    n_results = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        Result* results   = static_cast<Result*>(allocate_temporary(n_results * (long)sizeof(Result)));
        reduce_one_level(n, start, value_of_index, combine, results);

        while (n_results > 1) {
            Result*    inputs   = results;
            const long n_inputs = n_results;
            n_results           = (n_inputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
            results             = static_cast<Result*>(allocate_temporary(n_results * (long)sizeof(Result)));
            reduce_one_level(n_inputs, start, [=] HD(long i) { return inputs[i]; }, combine, results);
            free_temporary(inputs, n_inputs * (long)sizeof(Result));
        }

        Result result;
        copy_bytes(&result, results, (long)sizeof(Result));
        free_temporary(results, (long)sizeof(Result));
        return result;
    }

    // one level: every block halves its values until one is left, in the same order on CPU and GPU
    template <typename Result, typename ValueFunction, typename CombineFunction>
    void reduce_one_level(long            n,
                          Result          start,
                          ValueFunction   value_of_index,
                          CombineFunction combine,
                          Result*         results) {
        const long n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
#ifdef CUDA
        const timers::Scope timer("GPU", timers::Type::gpu);
        reduce_one_block_per_gpu_block<<<n_blocks, BLOCK_SIZE>>>(n, start, value_of_index, combine, results);
        wait_for_gpu("reduce");
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long b = 0; b < n_blocks; b++) {
            Result block[BLOCK_SIZE];
            for (int t = 0; t < BLOCK_SIZE; t++) {
                const long i = b * BLOCK_SIZE + t;
                block[t]     = (i < n) ? value_of_index(i) : start;
            }
            for (int half = BLOCK_SIZE / 2; half > 0; half /= 2)
                for (int t = 0; t < half; t++)
                    block[t] = combine(block[t], block[t + half]);
            results[b] = block[0];
        }
#endif
    }

#ifdef CUDA
    template <typename Result, typename ValueFunction, typename CombineFunction>
    __global__ void reduce_one_block_per_gpu_block(long            n,
                                                   Result          start,
                                                   ValueFunction   value_of_index,
                                                   CombineFunction combine,
                                                   Result*         results) {
        __shared__ alignas(Result) unsigned char memory[BLOCK_SIZE * sizeof(Result)];
        Result*                                  block = reinterpret_cast<Result*>(memory);

        const int  t = threadIdx.x;
        const long i = (long)blockIdx.x * BLOCK_SIZE + t;
        block[t]     = (i < n) ? value_of_index(i) : start;
        __syncthreads();
        for (int half = BLOCK_SIZE / 2; half > 0; half /= 2) {
            if (t < half) block[t] = combine(block[t], block[t + half]);
            __syncthreads();
        }
        if (t == 0) results[blockIdx.x] = block[0];
    }
#endif

    // marks every i with 1 or 0, turns the marks into offsets, and writes each marked i to its offset
    template <typename Integer, typename ConditionFunction>
    Integer indices_where(Integer n, ConditionFunction condition, Integer* indices) {
        if (n <= 0) return 0;
        const long bytes   = (long)n * (long)sizeof(Integer);
        Integer*   marks   = static_cast<Integer*>(allocate_temporary(bytes));
        Integer*   offsets = static_cast<Integer*>(allocate_temporary(bytes));

        parallel_for(n, [=] HD(long i) { marks[i] = condition((Integer)i) ? 1 : 0; });
        const Integer n_found = offsets_from_counts(n, marks, offsets);
        parallel_for(n, [=] HD(long i) {
            if (marks[i] == 1) indices[offsets[i]] = (Integer)i;
        });

        free_temporary(offsets, bytes);
        free_temporary(marks, bytes);
        return n_found;
    }

    // on the GPU the fences make earlier writes visible to every thread that sees the addition
    HD inline int atomic_add(int* counter, int value) {
#ifdef __CUDA_ARCH__
        __threadfence();
        const int before = atomicAdd(counter, value);
        __threadfence();
        return before;
#else
        return __atomic_fetch_add(counter, value, __ATOMIC_ACQ_REL);
#endif
    }

    // a volatile read goes past the cache to the memory itself
    template <typename Value> HD inline Value read_written_by_other_thread(const Value* address) {
        static_assert(sizeof(Value) % sizeof(double) == 0, "the value is read as doubles");
        Value                  value;
        const volatile double* from = reinterpret_cast<const volatile double*>(address);
        double*                to   = reinterpret_cast<double*>(&value);
        for (int i = 0; i < (int)(sizeof(Value) / sizeof(double)); i++)
            to[i] = from[i];
        return value;
    }

    HD inline int count_leading_zero_bits(uint64_t x) {
#ifdef __CUDA_ARCH__
        return __clzll((long long)x);
#else
        return (x == 0) ? 64 : __builtin_clzll(x);
#endif
    }

    HD inline int count_leading_zero_bits(uint32_t x) {
#ifdef __CUDA_ARCH__
        return __clz((int)x);
#else
        return (x == 0) ? 32 : __builtin_clz(x);
#endif
    }

} // namespace device
