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
    void sum_over_ranks(long long* values, int n); // in place
    bool on_any_rank(bool value);

    // gathers value of all ranks into value_of_rank[nranks]; collective
    void gather_from_all_ranks(long long value, long long* value_of_rank);

    // every rank gets the items of all ranks, as bytes; collective
    std::vector<char> gather_items_from_all_ranks(const void* items, int n, int item_bytes);

    // every rank tells every other rank how many items it will send; collective
    void exchange_counts(const int* send_count, int* receive_count);

    // exchanges the items between the ranks; collective
    void exchange_items(const void* send,
                        const int*  send_count,
                        void*       receive,
                        const int*  receive_count,
                        int         item_bytes);

} // namespace mpi
