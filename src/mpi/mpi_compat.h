#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H
#pragma once

// MPI startup, rank numbers and the fatal error path; all of it works without USE_MPI.

#include <cstddef>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace proteus_mpi {

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

    // managed memory around an MPI call: to the host before a send, back to the device after a receive
    void mpi_sync_before_send(const void* buf, size_t bytes);

    void mpi_sync_after_recv(void* buf, size_t bytes);

} // namespace proteus_mpi

#endif
