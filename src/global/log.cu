#include "../mpi/decomp.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "log.h"

#include <iostream>
#include <streambuf>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace logging {

    namespace {

        // streambuf that discards every byte written to it
        class NullBuf : public std::streambuf {
          protected:
            int overflow(int c) override { return traits_type::not_eof(c); }
        };

        // singleton null ostream shared by all silent loggers
        std::ostream& null_stream() {
            static NullBuf      buf;
            static std::ostream s(&buf);
            return s;
        }

    } // namespace

    // std::cout on rank 0, null sink elsewhere
    std::ostream& root() {
        return proteus_mpi::is_root() ? std::cout : null_stream();
    }

#ifdef USE_MPI
    namespace {

        // allreduce wrapper
        template <typename T> T reduce_global(T local, MPI_Datatype dtype, MPI_Op op) {
            PROFILE_MPI("ALLREDUCE");
            T g = local;
            MPI_Allreduce(&local, &g, 1, dtype, op, proteus_mpi::decomp.cart_comm);
            return g;
        }

    } // namespace
#endif

    int sum_global(int local) {
#ifdef USE_MPI
        return reduce_global(local, MPI_INT, MPI_SUM);
#else
        return local;
#endif
    }
    long long sum_global(long long local) {
#ifdef USE_MPI
        return reduce_global(local, MPI_LONG_LONG, MPI_SUM);
#else
        return local;
#endif
    }
    int max_global(int local) {
#ifdef USE_MPI
        return reduce_global(local, MPI_INT, MPI_MAX);
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

} // namespace logging
