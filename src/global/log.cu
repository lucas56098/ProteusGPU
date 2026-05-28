#include "../mpi/decomp.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "log.h"

#include <fstream>
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

    FileLogger::FileLogger(const std::string& path) {
        if (!proteus_mpi::is_root()) { return; } // non-root: leave file closed, root() returns null_stream

        std::ios::openmode mode = std::ios::out | std::ios::app;
        file.open(path, mode);
        if (!file.is_open()) { std::cerr << "LOG: Error! Could not open log file: " << path << std::endl; }
    }

    FileLogger::~FileLogger() {
        if (file.is_open()) {
            file.flush();
            file.close();
        }
    }

    void FileLogger::flush() {
        if (file.is_open()) { file.flush(); }
    }

    std::ostream& FileLogger::root() {
        return file.is_open() ? static_cast<std::ostream&>(file) : null_stream();
    }

    // std::cout on rank 0, null sink elsewhere
    std::ostream& root() {
        return proteus_mpi::is_root() ? std::cout : null_stream();
    }

#ifdef USE_MPI
    namespace {

        // allreduce wrapper
        template <typename T> T reduce_global(T local, MPI_Datatype dtype, MPI_Op op) {
            Profiler::StartTimer("MPI_COMM");
            Profiler::StartTimer("MPI_REDUCE");
            T g = local;
            MPI_Allreduce(&local, &g, 1, dtype, op, proteus_mpi::decomp.cart_comm);
            Profiler::EndTimer("MPI_REDUCE");
            Profiler::EndTimer("MPI_COMM");
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
