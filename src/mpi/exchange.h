#ifndef MPI_EXCHANGE_H
#define MPI_EXCHANGE_H
#pragma once

// Items between ranks: first how many, then the items, every partner's block in one gpu buffer.

#include "global/gpu_compat.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace proteus_mpi {

    // one block of items per partner rank in one buffer, partners in ascending rank order
    struct Blocks {
        std::vector<int>    ranks;
        std::vector<size_t> counts;
        std::vector<size_t> offsets;
        size_t              total = 0;

        void clear() {
            ranks.clear();
            counts.clear();
            offsets.clear();
            total = 0;
        }
        // appends the block of rank r; ranks must come in ascending order
        void add(int r, size_t count) {
            ranks.push_back(r);
            counts.push_back(count);
            offsets.push_back(total);
            total += count;
        }
    };

    // the blocks of a buffer whose item i goes to rank keys[i]; keys sorted, in gpu memory, the work arrays
    // in scratch
    Blocks blocks_of_sorted_ranks(const uint64_t* keys, size_t n, GpuArray<unsigned int>* scratch);

    // the same blocks, plus an empty one for every partner that has none
    Blocks with_partners(const Blocks& b, const std::vector<int>& partners);

    // tells every rank in out how many items it gets, and learns who sends here. Collective. The
    // blocks that arrive are sorted by source rank, so the result does not depend on timing
    void sparse_counts(const Blocks& out, Blocks* in);

    // the same when both sides know their partners: every rank in recv_from sends exactly one count here
    void partner_counts(const Blocks& out, const std::vector<int>& recv_from, Blocks* in);

    // the items themselves, both sides know the blocks; both buffers in gpu memory
    void exchange_items(const void* sendbuf, const Blocks& out, void* recvbuf, const Blocks& in, size_t item_bytes);

} // namespace proteus_mpi

#endif
