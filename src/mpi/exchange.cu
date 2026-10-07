// implements the exchanges of counts and items (exchange.h)

#include "exchange.h"

#include "decomp.h"
#include "global/allvars.h"
#include "mpi_compat.h"
#include "profiler/profiler.h"

#include <algorithm>
#include <climits>
#include <utility>

namespace proteus_mpi {

    // a run of equal keys is one block; only the first index of every run goes to the host
    Blocks blocks_of_sorted_ranks(const uint64_t* keys, size_t n, GpuArray<unsigned int>* scratch) {
        Blocks b;
        if (n == 0) return b;
        unsigned int* flag  = scratch->fit(2 * n + scan_scratch_size(n, _MPI_PACK_BLOCK_SIZE_));
        unsigned int* first = flag + n;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "RUN_FLAG", n, [=] HD(size_t i) { flag[i] = (i == 0 || keys[i] != keys[i - 1]) ? 1u : 0u; });
        const unsigned int last_starts = flag[n - 1];
        parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_>("RUN_SCAN", n, flag, flag, first + n);
        const size_t runs = (size_t)flag[n - 1] + last_starts;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("RUN_FIRST", n, [=] HD(size_t i) {
            if (i == 0 || keys[i] != keys[i - 1]) first[flag[i]] = (unsigned int)i;
        });
        for (size_t r = 0; r < runs; r++) {
            const size_t lo = first[r];
            const size_t hi = (r + 1 < runs) ? first[r + 1] : n;
            b.add((int)keys[lo], hi - lo);
        }
        return b;
    }

    Blocks with_partners(const Blocks& b, const std::vector<int>& partners) {
        std::vector<std::pair<int, size_t>> all;
        for (size_t i = 0; i < b.ranks.size(); i++)
            all.emplace_back(b.ranks[i], b.counts[i]);
        for (int p : partners) {
            if (!std::binary_search(b.ranks.begin(), b.ranks.end(), p)) all.emplace_back(p, 0);
        }
        std::sort(all.begin(), all.end());
        Blocks out;
        for (const auto& rc : all)
            out.add(rc.first, rc.second);
        return out;
    }

#ifdef USE_MPI
    // the counts of known partners and all items go point to point: between two ranks MPI keeps the
    // order of messages with one tag, so a later exchange cannot be taken for this one
    enum ExchangeTag { TAG_COUNT = 6001, TAG_ITEMS = 6002 };

    // without GPU aware MPI the buffer has to sit on the host before it is sent
    static void sync_before_send(const void* buf, size_t bytes) {
#ifdef CPU_DEBUG
        (void)buf;
        (void)bytes;
#else
        CUDA_CHECK(cudaDeviceSynchronize());
#ifndef GPU_AWARE_MPI
        if (bytes > 0 && buf != nullptr) {
            gpu_prefetch_to_cpu(const_cast<void*>(buf), bytes);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
#else
        (void)buf;
        (void)bytes;
#endif
#endif
    }

    // and goes back to the device after it arrived
    static void sync_after_recv(void* buf, size_t bytes) {
#if defined(CPU_DEBUG) || defined(GPU_AWARE_MPI)
        (void)buf;
        (void)bytes;
#else
        if (bytes > 0 && buf != nullptr) gpu_prefetch_to_gpu(buf, bytes);
#endif
    }

    // a sparse exchange probes for any source, so each one needs a tag of its own
    static int next_sparse_tag() {
        static int s_count = 0;
        s_count            = (s_count + 1) % 4096;
        return 7000 + s_count;
    }

    // the blocks of the received counts, in source rank order
    static void blocks_from_counts(std::vector<std::pair<int, long long>>* got, Blocks* in) {
        std::sort(got->begin(), got->end());
        in->clear();
        for (const auto& rc : *got)
            in->add(rc.first, (size_t)rc.second);
    }
#endif

    // synchronous sends of the counts, probing for whatever comes, then a barrier that ends when every
    // count has arrived
    void sparse_counts(const Blocks& out, Blocks* in) {
#ifndef USE_MPI
        *in = out;
#else
        PROFILE_MPI("SPARSE_COUNTS");
        const int                              tag = next_sparse_tag();
        std::vector<std::pair<int, long long>> got;

        std::vector<long long>   sent(out.ranks.size());
        std::vector<MPI_Request> sends;
        for (size_t i = 0; i < out.ranks.size(); i++) {
            sent[i] = (long long)out.counts[i];
            if (out.ranks[i] == decomp.rank) {
                got.emplace_back(decomp.rank, sent[i]);
                continue;
            }
            sends.emplace_back();
            MPI_Issend(&sent[i], 1, MPI_LONG_LONG, out.ranks[i], tag, decomp.comm, &sends.back());
        }

        MPI_Request barrier      = MPI_REQUEST_NULL;
        bool        barrier_on   = false;
        int         barrier_done = 0;
        while (!barrier_done) {
            int        flag = 0;
            MPI_Status st;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, decomp.comm, &flag, &st);
            if (flag) {
                long long c = 0;
                MPI_Recv(&c, 1, MPI_LONG_LONG, st.MPI_SOURCE, tag, decomp.comm, MPI_STATUS_IGNORE);
                got.emplace_back(st.MPI_SOURCE, c);
            }
            if (!barrier_on) {
                int all_sent = 1;
                if (!sends.empty()) MPI_Testall((int)sends.size(), sends.data(), &all_sent, MPI_STATUSES_IGNORE);
                if (all_sent) {
                    MPI_Ibarrier(decomp.comm, &barrier);
                    barrier_on = true;
                }
            } else {
                MPI_Test(&barrier, &barrier_done, MPI_STATUS_IGNORE);
            }
        }
        blocks_from_counts(&got, in);
#endif
    }

    void partner_counts(const Blocks& out, const std::vector<int>& recv_from, Blocks* in) {
#ifndef USE_MPI
        (void)recv_from;
        *in = out;
#else
        PROFILE_MPI("PARTNER_COUNTS");
        std::vector<std::pair<int, long long>> got;
        std::vector<long long>                 sent(out.ranks.size());
        std::vector<long long>                 recvd(recv_from.size());
        std::vector<MPI_Request>               reqs;
        for (size_t j = 0; j < recv_from.size(); j++) {
            if (recv_from[j] == decomp.rank) continue;
            reqs.emplace_back();
            MPI_Irecv(&recvd[j], 1, MPI_LONG_LONG, recv_from[j], TAG_COUNT, decomp.comm, &reqs.back());
        }
        for (size_t i = 0; i < out.ranks.size(); i++) {
            sent[i] = (long long)out.counts[i];
            if (out.ranks[i] == decomp.rank) {
                got.emplace_back(decomp.rank, sent[i]);
                continue;
            }
            reqs.emplace_back();
            MPI_Isend(&sent[i], 1, MPI_LONG_LONG, out.ranks[i], TAG_COUNT, decomp.comm, &reqs.back());
        }
        if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        for (size_t j = 0; j < recv_from.size(); j++) {
            if (recv_from[j] != decomp.rank) got.emplace_back(recv_from[j], recvd[j]);
        }
        blocks_from_counts(&got, in);
#endif
    }

    // the block a rank sends itself is a copy; MPI counts whole items, so a block may pass 2 GB
    void exchange_items(const void* sendbuf, const Blocks& out, void* recvbuf, const Blocks& in, size_t item_bytes) {
        const char* s  = (const char*)sendbuf;
        char*       r  = (char*)recvbuf;
        const int   me = decomp.rank;

        // the block to this rank, if there is one
        for (size_t i = 0; i < out.ranks.size(); i++) {
            if (out.ranks[i] != me || out.counts[i] == 0) continue;
            for (size_t j = 0; j < in.ranks.size(); j++) {
                if (in.ranks[j] != me) continue;
                if (in.counts[j] != out.counts[i]) {
                    exit_failure(
                        "EXCHANGE: rank %d sends itself %zu items but expects %zu\n", me, out.counts[i], in.counts[j]);
                }
                gpu_memcpy(r + in.offsets[j] * item_bytes, s + out.offsets[i] * item_bytes, out.counts[i] * item_bytes);
            }
        }
#ifndef USE_MPI
        (void)me;
#else
        PROFILE_MPI("ITEMS");
        MPI_Datatype item;
        MPI_Type_contiguous((int)item_bytes, MPI_BYTE, &item);
        MPI_Type_commit(&item);

        sync_before_send(sendbuf, item_bytes * out.total);
        std::vector<MPI_Request> reqs;
        for (size_t j = 0; j < in.ranks.size(); j++) {
            if (in.ranks[j] == me || in.counts[j] == 0) continue;
            if (in.counts[j] > (size_t)INT_MAX) exit_failure("EXCHANGE: %zu items from one rank\n", in.counts[j]);
            reqs.emplace_back();
            MPI_Irecv(r + in.offsets[j] * item_bytes,
                      (int)in.counts[j],
                      item,
                      in.ranks[j],
                      TAG_ITEMS,
                      decomp.comm,
                      &reqs.back());
        }
        for (size_t i = 0; i < out.ranks.size(); i++) {
            if (out.ranks[i] == me || out.counts[i] == 0) continue;
            if (out.counts[i] > (size_t)INT_MAX) exit_failure("EXCHANGE: %zu items to one rank\n", out.counts[i]);
            reqs.emplace_back();
            MPI_Isend(s + out.offsets[i] * item_bytes,
                      (int)out.counts[i],
                      item,
                      out.ranks[i],
                      TAG_ITEMS,
                      decomp.comm,
                      &reqs.back());
        }
        if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        sync_after_recv(recvbuf, item_bytes * in.total);
        MPI_Type_free(&item);
#endif
    }

} // namespace proteus_mpi
