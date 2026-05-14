#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H
#pragma once

// single-node fallback: when USE_MPI is undefined the program runs as one
// rank with no mpi.h dependency; when set, mpi.h is included here so the
// rest of the codebase only needs this header.

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace proteus_mpi {

// initialize MPI (no-op single-node) and pin one GPU per rank when CUDA is enabled
void init(int* argc, char*** argv);

// tear down MPI (no-op single-node)
void finalize();

int rank();
int nranks();

// ranks sharing this node (from MPI_COMM_TYPE_SHARED split)
int node_local_size();

// CUDA devices visible to this process; 0 in CPU_DEBUG / no-CUDA builds
int gpus_per_node();

inline bool is_root() { return rank() == 0; }

}  // namespace proteus_mpi

#endif  // MPI_COMPAT_H
