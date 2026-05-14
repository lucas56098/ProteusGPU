#ifndef MPI_EXTENSION_H
#define MPI_EXTENSION_H
#pragma once

// Per-cell array sizing for MPI ghost slots and migration headroom. Kept
// dependency-free so it can be included from structs.h / gpu_compat.h
// without pulling in mpi.h.

namespace proteus_mpi {

// total MPI-ghost capacity (set by halo_init; 0 single-rank or no neighbors)
extern int g_n_mpi_capacity;

// migration headroom on n_local; tune up if a conservation abort hits overflow
constexpr double ALLOC_GROWTH = 1.5;

// max n_local this rank tolerates after migration; allocation sizes per-cell
// arrays so that arr[k] is valid for k in [0, max_n_local + g_n_mpi_capacity)
inline int max_n_local(int n_initial) {
    return (int)((double)n_initial * ALLOC_GROWTH);
}

// runtime live size: current n_local + reserved MPI ghost slots
inline int extended_size(int n_local) {
    return n_local + g_n_mpi_capacity;
}

// allocation-time size: max-growth n_local + MPI ghost slots
inline int alloc_per_cell_size(int n_initial) {
    return max_n_local(n_initial) + g_n_mpi_capacity;
}

}  // namespace proteus_mpi

#endif  // MPI_EXTENSION_H
