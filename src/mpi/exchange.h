#ifndef MPI_EXCHANGE_H
#define MPI_EXCHANGE_H
#pragma once

// Byte messages between ranks: to ranks the receiver does not know about, or between known partners.

#include <cstddef>
#include <vector>

namespace proteus_mpi {

    // one message per partner rank, partners sorted by rank
    struct Messages {
        std::vector<int>               ranks;
        std::vector<std::vector<char>> data;

        void clear() {
            ranks.clear();
            data.clear();
        }
        // the message to rank r, a new empty one if there is none yet; ranks must come in ascending order
        std::vector<char>& to(int r);
    };

    // sends to the given ranks and receives from whoever sends; the receiver does not need to know who.
    // Collective. What arrives is sorted by source rank, so the result does not depend on timing.
    void sparse_exchange(const Messages& out, Messages* in);

    // both sides know their partners: every rank in recv_from sends exactly one message here
    void partner_exchange(const Messages& out, const std::vector<int>& recv_from, Messages* in);

    // the typed view of a message
    template <typename T> inline void append(std::vector<char>& msg, const T& v) {
        const char* p = (const char*)&v;
        msg.insert(msg.end(), p, p + sizeof(T));
    }
    template <typename T> inline size_t count_of(const std::vector<char>& msg) {
        return msg.size() / sizeof(T);
    }
    template <typename T> inline const T* items_of(const std::vector<char>& msg) {
        return (const T*)msg.data();
    }

} // namespace proteus_mpi

#endif
