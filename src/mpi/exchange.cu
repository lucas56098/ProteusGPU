// implements the byte message exchanges (exchange.h)

#include "exchange.h"

#include "decomp.h"
#include "mpi_compat.h"
#include "profiler/profiler.h"

#include <algorithm>
#include <cstring>

namespace proteus_mpi {

    std::vector<char>& Messages::to(int r) {
        if (ranks.empty() || ranks.back() != r) {
            ranks.push_back(r);
            data.emplace_back();
        }
        return data.back();
    }

#ifdef USE_MPI
    // every exchange gets its own tag, so a fast rank's next exchange cannot be taken for this one
    static int next_tag() {
        static int s_count = 0;
        s_count            = (s_count + 1) % 4096;
        return 7000 + s_count;
    }

    // keeps the self message out of MPI
    static void take_self(const Messages& out, Messages* in) {
        for (size_t i = 0; i < out.ranks.size(); i++) {
            if (out.ranks[i] == decomp.rank) {
                in->ranks.push_back(decomp.rank);
                in->data.push_back(out.data[i]);
            }
        }
    }

    static void sort_by_rank(Messages* in) {
        std::vector<size_t> order(in->ranks.size());
        for (size_t i = 0; i < order.size(); i++)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return in->ranks[a] < in->ranks[b]; });
        Messages sorted;
        for (size_t i : order) {
            sorted.ranks.push_back(in->ranks[i]);
            sorted.data.push_back(std::move(in->data[i]));
        }
        *in = std::move(sorted);
    }
#endif

    // synchronous sends, probing for whatever comes, then a barrier that ends when every send has arrived
    void sparse_exchange(const Messages& out, Messages* in) {
        in->clear();
#ifndef USE_MPI
        for (size_t i = 0; i < out.ranks.size(); i++) {
            in->ranks.push_back(out.ranks[i]);
            in->data.push_back(out.data[i]);
        }
#else
        PROFILE_MPI("SPARSE_EXCHANGE");
        const int tag = next_tag();
        take_self(out, in);

        std::vector<MPI_Request> sends;
        for (size_t i = 0; i < out.ranks.size(); i++) {
            if (out.ranks[i] == decomp.rank) continue;
            sends.emplace_back();
            MPI_Issend(
                out.data[i].data(), (int)out.data[i].size(), MPI_BYTE, out.ranks[i], tag, decomp.comm, &sends.back());
        }

        MPI_Request barrier      = MPI_REQUEST_NULL;
        bool        barrier_on   = false;
        int         barrier_done = 0;
        while (!barrier_done) {
            int        flag = 0;
            MPI_Status st;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, decomp.comm, &flag, &st);
            if (flag) {
                int bytes = 0;
                MPI_Get_count(&st, MPI_BYTE, &bytes);
                in->ranks.push_back(st.MPI_SOURCE);
                in->data.emplace_back((size_t)bytes);
                MPI_Recv(in->data.back().data(), bytes, MPI_BYTE, st.MPI_SOURCE, tag, decomp.comm, MPI_STATUS_IGNORE);
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
        sort_by_rank(in);
#endif
    }

    // sends first, then one probe and receive per partner, in rank order
    void partner_exchange(const Messages& out, const std::vector<int>& recv_from, Messages* in) {
        in->clear();
#ifndef USE_MPI
        (void)recv_from;
        for (size_t i = 0; i < out.ranks.size(); i++) {
            in->ranks.push_back(out.ranks[i]);
            in->data.push_back(out.data[i]);
        }
#else
        PROFILE_MPI("PARTNER_EXCHANGE");
        const int tag = next_tag();
        take_self(out, in);

        std::vector<MPI_Request> sends;
        for (size_t i = 0; i < out.ranks.size(); i++) {
            if (out.ranks[i] == decomp.rank) continue;
            sends.emplace_back();
            MPI_Isend(
                out.data[i].data(), (int)out.data[i].size(), MPI_BYTE, out.ranks[i], tag, decomp.comm, &sends.back());
        }
        for (int src : recv_from) {
            if (src == decomp.rank) continue;
            MPI_Status st;
            MPI_Probe(src, tag, decomp.comm, &st);
            int bytes = 0;
            MPI_Get_count(&st, MPI_BYTE, &bytes);
            in->ranks.push_back(src);
            in->data.emplace_back((size_t)bytes);
            MPI_Recv(in->data.back().data(), bytes, MPI_BYTE, src, tag, decomp.comm, MPI_STATUS_IGNORE);
        }
        if (!sends.empty()) MPI_Waitall((int)sends.size(), sends.data(), MPI_STATUSES_IGNORE);
        sort_by_rank(in);
#endif
    }

} // namespace proteus_mpi
