#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H
#pragma once

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace proteus_mpi {

    // init/finalize MPI communication
    void init(int* argc, char*** argv);
    void finalize();

    // getter functions
    int rank();
    int nranks();
    int node_local_size();
    int gpus_per_node();

    inline bool is_root() {
        return rank() == 0;
    }

    // exit MPI if failure
    void exit_failure(const char* fmt, ...) __attribute__((format(printf, 1, 2), noreturn));

} // namespace proteus_mpi

#endif // MPI_COMPAT_H
