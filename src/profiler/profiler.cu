// implements Profiler (profiler.h)

#include "../global/allvars.h"
#include "../io/h5.h"
#include "../mpi/mpi_compat.h"
#include "hdf5.h"
#include "profiler.h"
#include <algorithm>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>

#ifdef ENABLE_PROFILING

namespace {

    // timer state, shared by the three parts below
    thread_local std::vector<std::string> s_path_stack; // scope stack of this thread

    std::unordered_map<std::string, long long> s_cum_us; // microseconds per path, closed scopes only

    std::unordered_map<std::string, char> s_kind; // cpu, mpi or gpu per path

    std::unordered_map<std::string, long long> s_restart_baseline; // totals a restart started from

    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point>
        s_live_start; // start time of the open scopes

} // namespace

// clang-format off
// one translation unit, so the include order matters
#include "timers.cu"
#include "report.cu"
#include "profile_log.cu"
// clang-format on

#endif

// peak memory of rank 0, GPU bytes in a CUDA build
void print_max_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {

        double rss_bytes = 0.0;
// ru_maxrss is bytes on macOS and kilobytes on linux
#if defined(__APPLE__) && defined(__MACH__)
        rss_bytes = static_cast<double>(usage.ru_maxrss);
#elif defined(__linux__)
        rss_bytes = static_cast<double>(usage.ru_maxrss) * 1024.0;
#else
        rss_bytes = static_cast<double>(usage.ru_maxrss);
#endif

        const long   pages     = sysconf(_SC_PHYS_PAGES);
        const long   page_size = sysconf(_SC_PAGE_SIZE);
        const double total_ram = (pages > 0 && page_size > 0) ? (double)pages * (double)page_size : 0.0;

        constexpr double MiB     = 1024.0 * 1024.0;
        const double     rss_mib = rss_bytes / MiB;
        const char*      tag     = proteus_mpi::nranks() > 1 ? " (rank 0)" : "";
        logging::root() << "MAIN: maximum CPU memory used" << tag << ": " << rss_mib << " MiB (" << total_ram / MiB
                        << " MiB total)" << std::endl;
    } else {
        std::cerr << "Error getting resource usage." << std::endl;
    }

#ifndef CPU_DEBUG
    size_t gpu_free  = 0;
    size_t gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    constexpr double MiB      = 1024.0 * 1024.0;
    const double     peak_mib = (double)g_gpu_bytes_peak() / MiB;
    const char*      tag      = proteus_mpi::nranks() > 1 ? " (rank 0)" : "";
    logging::root() << "MAIN: maximum GPU memory used" << tag << ": " << peak_mib << " MiB (" << (double)gpu_total / MiB
                    << " MiB total)" << std::endl;
#endif
}
