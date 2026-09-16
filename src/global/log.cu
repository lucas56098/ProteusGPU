// implements logging (log.h)

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

        // stream buffer that discards everything written to it
        class NullBuf : public std::streambuf {
          protected:
            int overflow(int c) override { return traits_type::not_eof(c); }
        };

        // one sink shared by all non-root ranks
        std::ostream& null_stream() {
            static NullBuf      buf;
            static std::ostream s(&buf);
            return s;
        }

    } // namespace

    // rank 0 gets std::cout, every other rank the sink
    std::ostream& root() {
        return proteus_mpi::is_root() ? std::cout : null_stream();
    }

#ifdef USE_MPI
    namespace {

        // one Allreduce over cart_comm
        template <typename T> T reduce_global(T local, MPI_Datatype dtype, MPI_Op op) {
            PROFILE_MPI("ALLREDUCE");
            T g = local;
            MPI_Allreduce(&local, &g, 1, dtype, op, proteus_mpi::decomp.cart_comm);
            return g;
        }

    } // namespace
#endif

    // sum and max over all ranks, local value without USE_MPI
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

    // threads this rank may use
    int omp_threads() {
#ifdef USE_OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

} // namespace logging
