#ifndef LOG_H
#define LOG_H
#pragma once

// Rank-aware printing and the small cross-rank reductions.

#include <ostream>
#include <string>

namespace logging {

    // prints on rank 0, the other ranks write into a sink
    std::ostream& root();

    // reduce over all ranks; without USE_MPI they hand back the local value
    int       sum_global(int local);
    long long sum_global(long long local);
    int       max_global(int local);
    int       omp_threads(); // threads per rank, 1 without USE_OPENMP

} // namespace logging

#endif
