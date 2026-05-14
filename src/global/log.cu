#include "log.h"
#include "../mpi/mpi_compat.h"

#include <iostream>
#include <streambuf>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace logging {

namespace {
    class NullBuf : public std::streambuf {
      protected:
        int overflow(int c) override { return c; }
    };

    std::ostream& null_stream() {
        static NullBuf      buf;
        static std::ostream s(&buf);
        return s;
    }
}

std::ostream& root() {
    return proteus_mpi::is_root() ? std::cout : null_stream();
}

int sum_global(int local) {
#ifdef USE_MPI
    int g = local;
    MPI_Allreduce(&local, &g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return g;
#else
    return local;
#endif
}

long long sum_global(long long local) {
#ifdef USE_MPI
    long long g = local;
    MPI_Allreduce(&local, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return g;
#else
    return local;
#endif
}

int omp_threads() {
#ifdef USE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace logging
