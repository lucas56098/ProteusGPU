#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H
#pragma once

// MPI startup, rank numbers and the fatal error path; all of it works without USE_MPI.

#include <cstddef>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace mpi {

    // starts MPI and picks the GPU of this rank
    void init(int* argc, char*** argv);
    void finalize();

    int rank();
    int nranks();
    int node_local_size();
    int gpus_per_node();

    inline bool is_root() {
        return rank() == 0;
    }

    // prints the message and takes the whole run down
    void exit_failure(const char* fmt, ...) __attribute__((format(printf, 1, 2), noreturn));

    void report_gpu_aware_mpi();

    // one number reduced over all ranks
    double min_over_ranks(double v);
    double sum_over_ranks(double v);

} // namespace mpi

#endif
