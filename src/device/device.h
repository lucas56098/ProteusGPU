#pragma once

// Memory and loops on CPU and GPU; the only module with CUDA code (besides HD in config).

#include "config/config.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace device {

    // picks GPU, sets memory limit (this rank)
    void init(int node_rank, int ranks_on_node, double memory_limit_in_gb);

    int n_threads();

    void* allocate_bytes(const char* name, long bytes); // stops run if rank goes above memory limit
    void free_bytes(void* memory, long bytes);
    void copy_bytes(void* to, const void* from, long bytes);

    long peak_memory_in_bytes();
    void print_max_memory_usage();

    // array usable on CPU and GPU, in CUDA managed memory on the GPU; owns its memory, can be moved but not copied
    template <typename ElementType> class Array {

        static_assert(std::is_trivially_copyable<ElementType>::value,
                      "an Array holds plain values that can be copied as bytes");

      public:
        Array() = default;
        Array(const char* name, long size); // name shows up when memory runs out
        ~Array();

        Array(const Array&)            = delete;
        Array& operator=(const Array&) = delete;
        Array(Array&& other) noexcept { swap(other); }
        Array& operator=(Array&& other) noexcept {
            swap(other);
            return *this;
        }

        long size() const { return length; }
        ElementType* data() { return elements; }
        const ElementType* data() const { return elements; }

        // element access on the CPU
        ElementType& operator[](long i) { return elements[i]; }
        const ElementType& operator[](long i) const { return elements[i]; }

        // grows to at least size elements, keeping the old ones
        void grow_to_at_least(long size);

        void fill(const ElementType& value);

      private:
        const char*  name     = "";
        long         length   = 0;
        ElementType* elements = nullptr;

        void swap(Array& other) {
            std::swap(name, other.name);
            std::swap(length, other.length);
            std::swap(elements, other.elements);
        }
    };

    // copies the first n elements of from into to
    template <typename ElementType> void copy(const Array<ElementType>& from, Array<ElementType>& to, long n);

    // calls function(i) for every i in [0, n), on GPU or CPU
    template <typename Function> void parallel_for(long n, Function function);

    template <int THREADS_PER_BLOCK, int MIN_BLOCKS_PER_MULTIPROCESSOR, typename Function>
    void parallel_for(long n, Function function); // same, but with tunable block size

    template <typename Function> void parallel_for_on_cpu(long n, Function function); // also in GPU builds

    // sum, min, max or a general reduction of value_of_index(i) over every i in [0, n)
    template <typename ValueFunction> auto sum(long n, ValueFunction value_of_index);
    template <typename ValueFunction> auto min(long n, ValueFunction value_of_index);
    template <typename ValueFunction> auto max(long n, ValueFunction value_of_index);
    template <typename Result, typename ValueFunction, typename CombineFunction>
    Result reduce(long n, Result start, ValueFunction value_of_index, CombineFunction combine);

    // writes every i in [0, n) where condition(i) is true into indices; returns how many in total
    template <typename Integer, typename ConditionFunction>
    Integer indices_where(Integer n, ConditionFunction condition, Integer* indices);

    // offsets[i] = sum of counts[0, i); returns the total
    int offsets_from_counts(long n, const int* counts, int* offsets);
    long offsets_from_counts(long n, const long* counts, long* offsets);

    // sorts keys[0, n) and moves values along
    void sort_by_key(long n, uint64_t* keys, int* values);
    void sort_by_key(long n, uint64_t* keys, long* values);

    // atomic *counter += value; returns the old value
    HD inline int atomic_add(int* counter, int value);

    // reads a value another thread just wrote, past the GPU cache
    template <typename Value> HD inline Value read_written_by_other_thread(const Value* address);

    HD inline int count_leading_zero_bits(uint64_t x);
    HD inline int count_leading_zero_bits(uint32_t x);

    // packs field[index[i]] into item i; an item is a fixed block of bytes sent as one, e.g. the fields of one cell
    template <typename ElementType>
    void pack_into_items(const Array<ElementType>& field,
                         const int*                index,
                         long                      n,
                         int                       item_bytes,
                         int                       offset,
                         Array<char>&              items);

    // unpacks item i into field[first + i]
    template <typename ElementType>
    void unpack_from_items(const Array<char>&  items,
                           long                n,
                           int                 item_bytes,
                           int                 offset,
                           long                first,
                           Array<ElementType>& field);

    // unpacks item i into field[index[i]]
    template <typename ElementType>
    void unpack_from_items_at(const Array<char>&  items,
                              long                n,
                              int                 item_bytes,
                              int                 offset,
                              const int*          index,
                              Array<ElementType>& field);

    // mpi::exchange_items for items in array memory
    void exchange_items(const void* send,
                        const int*  send_count,
                        void*       receive,
                        const int*  receive_count,
                        int         item_bytes);

    // named ranges for Nsight Systems (only with CUDA_PROFILING)
    void begin_named_range(const char* name);
    void end_named_range();

} // namespace device

// the definitions of the functions above
#include "array.h"
#include "loops.h"
