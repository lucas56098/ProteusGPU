#include "../mpi/mpi_compat.h"
#include "log.h"

#include <fstream>
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

    } // namespace

    FileLogger::FileLogger(const std::string& path) {
        if (!proteus_mpi::is_root()) { return; }

        std::ios::openmode mode = std::ios::out | std::ios::app; // append

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

} // namespace logging
