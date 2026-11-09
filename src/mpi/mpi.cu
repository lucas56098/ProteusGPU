// implements the communication between ranks (mpi.h)

#include "mpi/mpi.h"

#include "timers/timers.h"

#ifdef USE_MPI
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace mpi {

    namespace {

        // the comm world, set once at init
        struct World {
            int rank      = 0;
            int nranks    = 1;
            int node_rank = 0;
            int node_size = 1;
#ifdef USE_MPI
            MPI_Comm comm = MPI_COMM_NULL;
#endif
        };
        World world;

        enum class Operation { min, max, sum };

        // between two ranks MPI keeps the order of the messages of one tag, so these two need no more than that
        constexpr int TAG_OF_KNOWN_COUNTS = 6001;
        constexpr int TAG_OF_ITEMS        = 6002;

        // a rank can already be in the next counts_from_any_rank while another one still is in this one
        constexpr int FIRST_TAG_OF_COUNTS_FROM_ANY_RANK = 7000;
        constexpr int TAGS_OF_COUNTS_FROM_ANY_RANK      = 4096;

        template <typename Number> Number combined_over_ranks(Number value, Operation operation);
#ifdef USE_MPI
        int tag_of_next_counts_from_any_rank();
        void take_a_count_that_came(int tag, std::vector<std::pair<int, int>>& came);
        bool all_are_taken(std::vector<MPI_Request>& sends);
        void blocks_of_counts_that_came(std::vector<std::pair<int, int>>& came, Blocks& in);
#endif

    } // namespace

    void init(int* argc, char*** argv) {
#ifdef USE_MPI
        MPI_Init(argc, argv);
        MPI_Comm_dup(MPI_COMM_WORLD, &world.comm);
        MPI_Comm_rank(world.comm, &world.rank);
        MPI_Comm_size(world.comm, &world.nranks);

        // the ranks that share this node, and the place of this rank among them
        MPI_Comm node;
        MPI_Comm_split_type(world.comm, MPI_COMM_TYPE_SHARED, world.rank, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &world.node_rank);
        MPI_Comm_size(node, &world.node_size);
        MPI_Comm_free(&node);
#else
        (void)argc;
        (void)argv;
#endif
    }

    void finalize() {
#ifdef USE_MPI
        MPI_Comm_free(&world.comm);
        MPI_Finalize();
#endif
    }

#ifdef USE_MPI
    MPI_Comm communicator() {
        return world.comm;
    }
#endif

    int rank() {
        return world.rank;
    }
    int nranks() {
        return world.nranks;
    }
    bool is_root() {
        return world.rank == 0;
    }
    int node_rank() {
        return world.node_rank;
    }
    int node_size() {
        return world.node_size;
    }

    std::vector<std::string> about_library() {
        std::vector<std::string> lines;
#ifdef USE_MPI
#ifdef CUDA
#ifdef GPU_AWARE_MPI
        lines.push_back("GPU_AWARE_MPI is set");
#else
        lines.push_back("GPU_AWARE_MPI not set");
#endif
#ifdef MPIX_CUDA_AWARE_SUPPORT
        lines.push_back("MPI MPIX_Query_cuda_support() = " + std::to_string(MPIX_Query_cuda_support()));
#endif
        if (const char* enabled = std::getenv("MPICH_GPU_SUPPORT_ENABLED")) {
            lines.push_back(std::string("MPI MPICH_GPU_SUPPORT_ENABLED = ") + enabled);
        }
#endif
        char version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int  length                                  = 0;
        if (MPI_Get_library_version(version, &length) == MPI_SUCCESS) {
            lines.push_back(std::string("MPI library = ") + version);
        }
#endif
        return lines;
    }

    void stop_all_ranks() {
        std::fflush(nullptr);
#ifdef USE_MPI
        int started = 0;
        MPI_Initialized(&started);
        if (started) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        std::exit(1);
    }

    int min_over_ranks(int value) {
        return combined_over_ranks(value, Operation::min);
    }
    long long min_over_ranks(long long value) {
        return combined_over_ranks(value, Operation::min);
    }
    double min_over_ranks(double value) {
        return combined_over_ranks(value, Operation::min);
    }
    int max_over_ranks(int value) {
        return combined_over_ranks(value, Operation::max);
    }
    long long max_over_ranks(long long value) {
        return combined_over_ranks(value, Operation::max);
    }
    double max_over_ranks(double value) {
        return combined_over_ranks(value, Operation::max);
    }
    long long sum_over_ranks(long long value) {
        return combined_over_ranks(value, Operation::sum);
    }

    void sum_over_ranks(long long* values, int n) {
#ifdef USE_MPI
        const timers::Scope timer("MPI_REDUCE", timers::Type::mpi);
        MPI_Allreduce(MPI_IN_PLACE, values, n, MPI_LONG_LONG, MPI_SUM, world.comm);
#else
        (void)values;
        (void)n;
#endif
    }

    bool on_any_rank(bool value) {
        return max_over_ranks(value ? 1 : 0) == 1;
    }

    void gather_from_all_ranks(long long value, long long* value_of_rank) {
#ifdef USE_MPI
        const timers::Scope timer("MPI_GATHER", timers::Type::mpi);
        MPI_Allgather(&value, 1, MPI_LONG_LONG, value_of_rank, 1, MPI_LONG_LONG, world.comm);
#else
        value_of_rank[0] = value;
#endif
    }

    std::vector<char> gather_items_from_all_ranks(const void* items, int n, int item_bytes) {
#ifdef USE_MPI
        const timers::Scope timer("MPI_GATHER", timers::Type::mpi);

        std::vector<int> bytes_of_rank(world.nranks), first_byte(world.nranks, 0);

        const int bytes = n * item_bytes;

        MPI_Allgather(&bytes, 1, MPI_INT, bytes_of_rank.data(), 1, MPI_INT, world.comm);

        for (int r = 1; r < world.nranks; r++)
            first_byte[r] = first_byte[r - 1] + bytes_of_rank[r - 1];

        std::vector<char> all(first_byte[world.nranks - 1] + bytes_of_rank[world.nranks - 1]);

        MPI_Allgatherv(
            items, bytes, MPI_BYTE, all.data(), bytes_of_rank.data(), first_byte.data(), MPI_BYTE, world.comm);

        return all;
#else
        const char* first = static_cast<const char*>(items);
        return std::vector<char>(first, first + (size_t)n * item_bytes);
#endif
    }

    void gather_one_item_from_all_ranks(const void* item, int item_bytes, void* all_items) {
#ifdef USE_MPI
        const timers::Scope timer("MPI_GATHER", timers::Type::mpi);
        MPI_Allgather(item, item_bytes, MPI_BYTE, all_items, item_bytes, MPI_BYTE, world.comm);
#else
        std::memcpy(all_items, item, (size_t)item_bytes);
#endif
    }

    void Blocks::clear() {
        ranks.clear();
        counts.clear();
        first.clear();
        total = 0;
    }

    void Blocks::add(int rank, int count) {
        ranks.push_back(rank);
        counts.push_back(count);
        first.push_back(total);
        total += count;
    }

    Blocks blocks_of_counts(const int* count, int n_ranks) {
        Blocks blocks;
        for (int r = 0; r < n_ranks; r++) {
            if (count[r] > 0) blocks.add(r, count[r]);
        }
        return blocks;
    }

    // the counts go out as synchronous sends, which end when they are taken; a rank joins the reduction once all its
    // sends have ended, so when the reduction ends every count of every rank has been taken
    long long counts_from_any_rank(const Blocks& out, long long to_sum, Blocks& in) {
#ifdef USE_MPI
        const timers::Scope              timer("MPI_COUNTS", timers::Type::mpi);
        const int                        tag = tag_of_next_counts_from_any_rank();
        std::vector<std::pair<int, int>> came;
        std::vector<MPI_Request>         sends;
        for (size_t b = 0; b < out.ranks.size(); b++) {
            if (out.ranks[b] == world.rank) {
                came.emplace_back(world.rank, out.counts[b]);
                continue;
            }
            sends.emplace_back();
            MPI_Issend(&out.counts[b], 1, MPI_INT, out.ranks[b], tag, world.comm, &sends.back());
        }

        long long   sum        = 0;
        MPI_Request reduction  = MPI_REQUEST_NULL;
        bool        has_joined = false;
        int         all_joined = 0;
        while (!all_joined) {
            take_a_count_that_came(tag, came);
            if (has_joined) {
                MPI_Test(&reduction, &all_joined, MPI_STATUS_IGNORE);
            } else if (all_are_taken(sends)) {
                MPI_Iallreduce(&to_sum, &sum, 1, MPI_LONG_LONG, MPI_SUM, world.comm, &reduction);
                has_joined = true;
            }
        }
        blocks_of_counts_that_came(came, in);
        return sum;
#else
        in = out;
        return to_sum;
#endif
    }

    void counts_from_known_ranks(const Blocks& out, const std::vector<int>& senders, Blocks& in) {
#ifdef USE_MPI
        const timers::Scope      timer("MPI_COUNTS", timers::Type::mpi);
        std::vector<int>         came(senders.size(), 0);
        std::vector<MPI_Request> requests;
        for (size_t j = 0; j < senders.size(); j++) {
            if (senders[j] == world.rank) continue;
            requests.emplace_back();
            MPI_Irecv(&came[j], 1, MPI_INT, senders[j], TAG_OF_KNOWN_COUNTS, world.comm, &requests.back());
        }
        for (size_t b = 0; b < out.ranks.size(); b++) {
            if (out.ranks[b] == world.rank) {
                const size_t j = std::lower_bound(senders.begin(), senders.end(), world.rank) - senders.begin();
                came[j]        = out.counts[b];
                continue;
            }
            requests.emplace_back();
            MPI_Isend(&out.counts[b], 1, MPI_INT, out.ranks[b], TAG_OF_KNOWN_COUNTS, world.comm, &requests.back());
        }
        if (!requests.empty()) MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        in.clear();
        for (size_t j = 0; j < senders.size(); j++)
            in.add(senders[j], came[j]);
#else
        (void)senders;
        in = out;
#endif
    }

    void exchange_items(const void* send, const Blocks& out, void* receive, const Blocks& in, int item_bytes) {
#ifdef USE_MPI
        const timers::Scope timer("MPI_ITEMS", timers::Type::mpi);
        MPI_Datatype        item;
        MPI_Type_contiguous(item_bytes, MPI_BYTE, &item);
        MPI_Type_commit(&item);

        std::vector<MPI_Request> requests;
        for (size_t b = 0; b < in.ranks.size(); b++) {
            if (in.ranks[b] == world.rank || in.counts[b] == 0) continue;
            requests.emplace_back();
            MPI_Irecv(static_cast<char*>(receive) + (long)in.first[b] * item_bytes,
                      in.counts[b],
                      item,
                      in.ranks[b],
                      TAG_OF_ITEMS,
                      world.comm,
                      &requests.back());
        }
        for (size_t b = 0; b < out.ranks.size(); b++) {
            if (out.ranks[b] == world.rank || out.counts[b] == 0) continue;
            requests.emplace_back();
            MPI_Isend(static_cast<const char*>(send) + (long)out.first[b] * item_bytes,
                      out.counts[b],
                      item,
                      out.ranks[b],
                      TAG_OF_ITEMS,
                      world.comm,
                      &requests.back());
        }
        if (!requests.empty()) MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        MPI_Type_free(&item);
#else
        (void)send;
        (void)out;
        (void)receive;
        (void)in;
        (void)item_bytes;
#endif
    }

    namespace {

#ifdef USE_MPI
        template <typename Number> MPI_Datatype mpi_type_of();
        template <> MPI_Datatype mpi_type_of<int>() {
            return MPI_INT;
        }
        template <> MPI_Datatype mpi_type_of<long long>() {
            return MPI_LONG_LONG;
        }
        template <> MPI_Datatype mpi_type_of<double>() {
            return MPI_DOUBLE;
        }

        MPI_Op mpi_operation_of(Operation operation) {
            switch (operation) {
            case Operation::min:
                return MPI_MIN;
            case Operation::max:
                return MPI_MAX;
            case Operation::sum:
                return MPI_SUM;
            }
            return MPI_OP_NULL;
        }
#endif

        template <typename Number> Number combined_over_ranks(Number value, Operation operation) {
#ifdef USE_MPI
            const timers::Scope timer("MPI_REDUCE", timers::Type::mpi);
            Number              result;
            MPI_Allreduce(&value, &result, 1, mpi_type_of<Number>(), mpi_operation_of(operation), world.comm);
            return result;
#else
            (void)operation;
            return value;
#endif
        }

#ifdef USE_MPI
        // every rank calls counts_from_any_rank equally often, so all agree on the tag
        int tag_of_next_counts_from_any_rank() {
            static int calls = 0;
            calls            = (calls + 1) % TAGS_OF_COUNTS_FROM_ANY_RANK;
            return FIRST_TAG_OF_COUNTS_FROM_ANY_RANK + calls;
        }

        void take_a_count_that_came(int tag, std::vector<std::pair<int, int>>& came) {
            int        arrived = 0;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, world.comm, &arrived, &status);
            if (!arrived) return;
            int count = 0;
            MPI_Recv(&count, 1, MPI_INT, status.MPI_SOURCE, tag, world.comm, MPI_STATUS_IGNORE);
            came.emplace_back(status.MPI_SOURCE, count);
        }

        bool all_are_taken(std::vector<MPI_Request>& sends) {
            int all_taken = 1;
            if (!sends.empty()) MPI_Testall((int)sends.size(), sends.data(), &all_taken, MPI_STATUSES_IGNORE);
            return all_taken != 0;
        }

        void blocks_of_counts_that_came(std::vector<std::pair<int, int>>& came, Blocks& in) {
            std::sort(came.begin(), came.end());
            in.clear();
            for (const auto& [rank, count] : came)
                in.add(rank, count);
        }
#endif

    } // namespace

} // namespace mpi
