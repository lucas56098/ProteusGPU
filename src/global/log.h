#ifndef LOG_H
#define LOG_H
#pragma once

#include <ostream>
#include <string>

namespace logging {

    // std::cout on rank 0, a no-op sink elsewhere
    std::ostream& root();

    int       sum_global(int local);
    long long sum_global(long long local);
    int       max_global(int local);
    int       omp_threads();

} // namespace logging

#endif // LOG_H
