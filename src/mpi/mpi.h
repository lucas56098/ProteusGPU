#pragma once

// Communication between ranks; the only module that calls MPI.

#include <string>
#include <vector>
#ifdef USE_MPI
// only the C interface of MPI
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include <mpi.h>
#endif

namespace mpi {

#ifdef USE_MPI
    MPI_Comm communicator();
#endif

    // start and end of MPI; collective
    void init(int* argc, char*** argv);
    void finalize();

    int rank();
    int nranks();
    bool is_root();
    int node_rank();
    int node_size(); // ranks on this node

    // the value of every rank on this node, by node rank; collective on the node
    std::vector<long long> gather_on_node(long long value);

    // what the MPI library says about itself and about GPU memory, as lines for the log; none without MPI
    std::vector<std::string> about_library();

    // flushes the output and stops every rank at once
    [[noreturn]] void stop_all_ranks();

    // reductions over all ranks; collective
    int min_over_ranks(int value);
    long long min_over_ranks(long long value);
    double min_over_ranks(double value);
    int max_over_ranks(int value);
    long long max_over_ranks(long long value);
    double max_over_ranks(double value);
    long long sum_over_ranks(long long value);
    double sum_over_ranks(double value);
    void sum_over_ranks(long long* values, int n); // in place
    bool on_any_rank(bool value);

    // gathers value of all ranks into value_of_rank[nranks]; collective
    void gather_from_all_ranks(long long value, long long* value_of_rank);

    // every rank gets the items of all ranks, as bytes; collective
    std::vector<char> gather_items_from_all_ranks(const void* items, int n, int item_bytes);

    // every rank gets one item of every rank, that of rank r at all_items + r * item_bytes; collective
    void gather_one_item_from_all_ranks(const void* item, int item_bytes, void* all_items);

    // blocks of items for or from other ranks, by rising rank: block b holds counts[b] items from first[b] on
    struct Blocks {
        std::vector<int> ranks;
        std::vector<int> counts;
        std::vector<int> first;
        int              total = 0;

        void clear();
        void add(int rank, int count);
    };

    // the blocks of the ranks r with count[r] > 0
    Blocks blocks_of_counts(const int* count, int n_ranks);

    // every rank learns how many items come, from ranks it does not know; returns the sum of to_sum; collective
    long long counts_from_any_rank(const Blocks& out, long long to_sum, Blocks& in);

    // the same when the senders are known, by rising rank: each of them sends a count, also 0, and each rank of out
    // gets one; no other rank takes part
    void counts_from_known_ranks(const Blocks& out, const std::vector<int>& senders, Blocks& in);

    // the items of out go to their ranks and those of in come in; only ranks with items take part; leaves out the
    // block of this rank (device::exchange_items copies that one)
    void exchange_items(const void* send, const Blocks& out, void* receive, const Blocks& in, int item_bytes);

} // namespace mpi
