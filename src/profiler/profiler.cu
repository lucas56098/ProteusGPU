#include "../global/gpu_compat.h"
#include "../global/log.h"
#include "../mpi/mpi_compat.h"
#include "profiler.h"
#include <algorithm>
#include <sys/resource.h>
#include <unistd.h>

// CPU wall-clock profiling (chrono-based, works in all modes)

std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> Profiler::m_StartTimes;
std::unordered_map<std::string, long long>                                      Profiler::m_Timings;

// GPU event timing storage
std::unordered_map<std::string, double> Profiler::m_GpuTimings;
std::unordered_map<std::string, int>    Profiler::m_GpuCounts;

#if !defined(CPU_DEBUG) && defined(CUDA)
std::unordered_map<std::string, Profiler::GpuEventPair> Profiler::m_GpuEvents;
#endif

void Profiler::StartTimer(const std::string& name) {
    NVTX_PUSH(name.c_str());
    m_StartTimes[name] = std::chrono::high_resolution_clock::now();
}

void Profiler::EndTimer(const std::string& name) {
    auto endTime   = std::chrono::high_resolution_clock::now();
    auto startTime = m_StartTimes[name];
    m_Timings[name] += std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    NVTX_POP();
}

// GPU event-based profiling (CUDA mode only)

void Profiler::StartGPU(const std::string& name) {
#if !defined(CPU_DEBUG) && defined(CUDA)
    NVTX_PUSH(name.c_str());
    auto& ev = m_GpuEvents[name];
    if (!ev.start) {
        cudaEventCreate((cudaEvent_t*)&ev.start);
        cudaEventCreate((cudaEvent_t*)&ev.stop);
    }
    cudaEventRecord((cudaEvent_t)ev.start, 0);
#else
    (void)name;
#endif
}

void Profiler::EndGPU(const std::string& name) {
#if !defined(CPU_DEBUG) && defined(CUDA)
    auto& ev = m_GpuEvents[name];
    cudaEventRecord((cudaEvent_t)ev.stop, 0);
    cudaEventSynchronize((cudaEvent_t)ev.stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, (cudaEvent_t)ev.start, (cudaEvent_t)ev.stop);
    m_GpuTimings[name] += (double)ms;
    m_GpuCounts[name]++;
    NVTX_POP();
#else
    (void)name;
#endif
}

// print combined results — per-rank timings but only rank 0's view is surfaced
void Profiler::PrintResults() {
    std::ostream& out = logging::root();
    out << "\n=== Profiling Results (Wall Clock Time, rank 0) ===\n";
    long long totalRuntime     = 0;
    long long parallelizedTime = 0;

    for (const auto& entry : m_Timings) {
        double timeInSeconds = entry.second / 1e6;
        out << "[PROFILE] " << entry.first << " took " << timeInSeconds << "s\n";

        if (entry.first.find("(par)") != std::string::npos) { parallelizedTime += entry.second; }

        if (entry.first == "TOTAL_RUNTIME") { totalRuntime = entry.second; }
    }

    double parallelFraction = 0.0;
    if (totalRuntime > 0) { parallelFraction = static_cast<double>(parallelizedTime) / totalRuntime; }
    out << "\nTOTAL_RUNTIME: " << (totalRuntime / 1e6) << "s\n";
    out << "PARALLELIZED_TIME: " << (parallelizedTime / 1e6) << "s\n";
    out << "PARALLEL_FRACTION: " << parallelFraction * 100.0 << " %\n";

    // GPU kernel timing breakdown
    if (!m_GpuTimings.empty()) {
        out << "\n=== GPU Kernel Profiling (CUDA Events, rank 0) ===\n";

        // sort by cumulative time for reading
        std::vector<std::pair<std::string, double>> sorted(m_GpuTimings.begin(), m_GpuTimings.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        double gpu_total_ms = 0.0;
        for (const auto& entry : sorted) {
            gpu_total_ms += entry.second;
        }

        for (const auto& entry : sorted) {
            int    calls = m_GpuCounts[entry.first];
            double pct   = (gpu_total_ms > 0.0) ? (entry.second / gpu_total_ms * 100.0) : 0.0;
            out << "[GPU] " << entry.first << ": " << entry.second << " ms"
                << " (" << calls << " calls, " << (entry.second / calls) << " ms/call"
                << ", " << pct << "%)\n";
        }
        out << "[GPU] TOTAL: " << gpu_total_ms << " ms\n";
    }

    out << "=========================\n";
}

// peak CPU RSS (high-water mark from the OS) and peak GPU memory we ever held
void print_max_memory_usage() {

    // CPU side: peak resident set size from getrusage, total system RAM from sysconf
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {

        double rssBytes = 0.0;
#if defined(__APPLE__) && defined(__MACH__)
        rssBytes = static_cast<double>(usage.ru_maxrss); // macOS reports bytes
#elif defined(__linux__)
        rssBytes = static_cast<double>(usage.ru_maxrss) * 1024.0; // Linux reports KiB
#else
        rssBytes = static_cast<double>(usage.ru_maxrss); // fallback: assume bytes
#endif

        const long   pages    = sysconf(_SC_PHYS_PAGES);
        const long   pageSize = sysconf(_SC_PAGE_SIZE);
        const double totalRam = (pages > 0 && pageSize > 0) ? (double)pages * (double)pageSize : 0.0;

        constexpr double MiB     = 1024.0 * 1024.0;
        const double     rssMiB  = rssBytes / MiB;
        const char*      tag     = proteus_mpi::nranks() > 1 ? " (rank 0)" : "";
        logging::root() << "MAIN: maximum CPU memory used" << tag << ": " << rssMiB << " MiB ("
                        << totalRam / MiB << " MiB total)" << std::endl;
    } else {

        std::cerr << "Error getting resource usage." << std::endl;
    }

    // GPU side: high-water mark of bytes we requested through gpu_alloc, total from cudaMemGetInfo
#ifndef CPU_DEBUG
    size_t gpu_free  = 0;
    size_t gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    constexpr double MiB     = 1024.0 * 1024.0;
    const double     peakMiB = (double)g_gpu_bytes_peak() / MiB;
    const char*      tag     = proteus_mpi::nranks() > 1 ? " (rank 0)" : "";
    logging::root() << "MAIN: maximum GPU memory used" << tag << ": " << peakMiB << " MiB ("
                    << (double)gpu_total / MiB << " MiB total)" << std::endl;
#endif
}