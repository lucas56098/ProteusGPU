#include "profiler.h"
#include "../global/gpu_compat.h"
#include <sys/resource.h>

std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> Profiler::m_StartTimes;
std::unordered_map<std::string, long long>                                      Profiler::m_Timings;

void Profiler::StartTimer(const std::string& name) {
    m_StartTimes[name] = std::chrono::high_resolution_clock::now();
}

void Profiler::EndTimer(const std::string& name) {
    auto endTime   = std::chrono::high_resolution_clock::now();
    auto startTime = m_StartTimes[name];
    m_Timings[name] += std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

void Profiler::PrintResults() {
    std::cout << "\n=== Profiling Results (Wall Clock Time) ===\n";
    long long totalRuntime     = 0;
    long long parallelizedTime = 0;

    for (const auto& entry : m_Timings) {
        double timeInSeconds = entry.second / 1e6;
        std::cout << "[PROFILE] " << entry.first << " took " << timeInSeconds << "s\n";

        if (entry.first.find("(par)") != std::string::npos) { parallelizedTime += entry.second; }

        if (entry.first == "TOTAL_RUNTIME") { totalRuntime = entry.second; }
    }

    double parallelFraction = 0.0;
    if (totalRuntime > 0) { parallelFraction = static_cast<double>(parallelizedTime) / totalRuntime; }
    std::cout << "\nTOTAL_RUNTIME: " << (totalRuntime / 1e6) << "s\n";
    std::cout << "PARALLELIZED_TIME: " << (parallelizedTime / 1e6) << "s\n";
    std::cout << "PARALLEL_FRACTION: " << parallelFraction * 100.0 << " %\n";
    std::cout << "=========================\n";
}

// MEMORY: function to print out maximum memory usage
void print_max_memory_usage() {

    struct rusage usage;

    // get resource usage statistics for the current process
    if (getrusage(RUSAGE_SELF, &usage) == 0) {

        double rssBytes = 0.0;
#if defined(__APPLE__) && defined(__MACH__)
        rssBytes = static_cast<double>(usage.ru_maxrss); // macOS reports bytes
#elif defined(__linux__)
        rssBytes = static_cast<double>(usage.ru_maxrss) * 1024.0; // Linux reports KiB
#else
        rssBytes = static_cast<double>(usage.ru_maxrss); // fallback: assume bytes
#endif

        std::cout << "MAIN: maximum memory used: " << rssBytes / 1000000.0 << " MB" << std::endl;
    } else {

        std::cerr << "Error getting resource usage." << std::endl;
    }
}