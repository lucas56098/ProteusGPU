// implements offsets_from_counts and sort_by_key (device.h); integers, so CUB and the standard library agree

#include "device/device.h"

#include <algorithm>
#include <numeric>

#ifdef CUDA
#include <cub/cub.cuh>
#endif

namespace device {

    namespace {

        template <typename Integer> Integer offsets_and_total(long n, const Integer* counts, Integer* offsets);
        template <typename Integer> void sort_pairs(long n, uint64_t* keys, Integer* values);
        template <typename Integer>
        void sort_into(long            n,
                       const uint64_t* keys,
                       const Integer*  values,
                       uint64_t*       sorted_keys,
                       Integer*        sorted_values);

    } // namespace

    int offsets_from_counts(long n, const int* counts, int* offsets) {
        return offsets_and_total(n, counts, offsets);
    }
    long offsets_from_counts(long n, const long* counts, long* offsets) {
        return offsets_and_total(n, counts, offsets);
    }

    void sort_by_key(long n, uint64_t* keys, int* values) {
        sort_pairs(n, keys, values);
    }
    void sort_by_key(long n, uint64_t* keys, long* values) {
        sort_pairs(n, keys, values);
    }

    namespace {

        template <typename Integer> Integer offsets_and_total(long n, const Integer* counts, Integer* offsets) {
            if (n <= 0) return 0;
#ifdef CUDA
            // CUB first says how much temporary memory it needs
            size_t temporary_bytes = 0;
            stop_on_cuda_error(cub::DeviceScan::ExclusiveSum(nullptr, temporary_bytes, counts, offsets, n),
                               "offsets_from_counts");
            void* temporary = allocate_temporary((long)temporary_bytes);
            stop_on_cuda_error(cub::DeviceScan::ExclusiveSum(temporary, temporary_bytes, counts, offsets, n),
                               "offsets_from_counts");
            wait_for_gpu("offsets_from_counts");
            free_temporary(temporary, (long)temporary_bytes);
#else
            std::exclusive_scan(counts, counts + n, offsets, Integer(0));
#endif
            Integer last_offset, last_count;
            copy_bytes(&last_offset, offsets + n - 1, (long)sizeof(Integer));
            copy_bytes(&last_count, counts + n - 1, (long)sizeof(Integer));
            return last_offset + last_count;
        }

        template <typename Integer> void sort_pairs(long n, uint64_t* keys, Integer* values) {
            if (n <= 1) return;
            const long key_bytes     = n * (long)sizeof(uint64_t);
            const long value_bytes   = n * (long)sizeof(Integer);
            uint64_t*  sorted_keys   = static_cast<uint64_t*>(allocate_temporary(key_bytes));
            Integer*   sorted_values = static_cast<Integer*>(allocate_temporary(value_bytes));

            sort_into(n, keys, values, sorted_keys, sorted_values);
            copy_bytes(keys, sorted_keys, key_bytes);
            copy_bytes(values, sorted_values, value_bytes);

            free_temporary(sorted_values, value_bytes);
            free_temporary(sorted_keys, key_bytes);
        }

        // both keep equal keys in their order
        template <typename Integer>
        void sort_into(long            n,
                       const uint64_t* keys,
                       const Integer*  values,
                       uint64_t*       sorted_keys,
                       Integer*        sorted_values) {
#ifdef CUDA
            size_t temporary_bytes = 0;
            stop_on_cuda_error(
                cub::DeviceRadixSort::SortPairs(nullptr, temporary_bytes, keys, sorted_keys, values, sorted_values, n),
                "sort_by_key");
            void* temporary = allocate_temporary((long)temporary_bytes);
            stop_on_cuda_error(cub::DeviceRadixSort::SortPairs(
                                   temporary, temporary_bytes, keys, sorted_keys, values, sorted_values, n),
                               "sort_by_key");
            wait_for_gpu("sort_by_key");
            free_temporary(temporary, (long)temporary_bytes);
#else
            const long order_bytes = n * (long)sizeof(long);
            long*      order       = static_cast<long*>(allocate_temporary(order_bytes));
            std::iota(order, order + n, 0L);
            std::stable_sort(order, order + n, [&](long a, long b) { return keys[a] < keys[b]; });
            for (long i = 0; i < n; i++) {
                sorted_keys[i]   = keys[order[i]];
                sorted_values[i] = values[order[i]];
            }
            free_temporary(order, order_bytes);
#endif
        }

    } // namespace

} // namespace device
