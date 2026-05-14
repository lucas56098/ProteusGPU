#ifndef LOG_H
#define LOG_H
#pragma once

#include <ostream>

namespace logging {

// std::cout on rank 0, a no-op sink elsewhere
std::ostream& root();

// MPI sum-reduce across MPI_COMM_WORLD; all ranks must call. Single-rank builds return `local`.
int       sum_global(int local);
long long sum_global(long long local);

// OpenMP threads per rank (omp_get_max_threads, or 1 without OpenMP)
int omp_threads();

}  // namespace logging

#endif  // LOG_H
