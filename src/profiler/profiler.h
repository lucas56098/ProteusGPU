#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// NVTX range annotations for Nsight Systems timeline
#ifdef CUDA_PROFILING
#include "nvtx3/nvToolsExt.h"
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif

class Profiler {
  public:
    static void StartTimer(const std::string& name);
    static void EndTimer(const std::string& name);
    static void PrintResults();
    static void LogTimestep(int step, std::ostream& out);

    static void StartGPU(const std::string& name);
    static void EndGPU(const std::string& name);

  private:
    // CPU wall-clock timing
    static std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> m_StartTimes;
    static std::unordered_map<std::string, long long>                                      m_Timings;
    static std::unordered_map<std::string, long long>                                      m_PrevStepCum; // cumulative at the end of the previous LogTimestep

    // GPU event timing (accumulated ms per region)
    static std::unordered_map<std::string, double> m_GpuTimings;     // cumulative ms
    static std::unordered_map<std::string, int>    m_GpuCounts;      // call counts
    static std::unordered_map<std::string, double> m_GpuPrevStepCum; // cumulative ms at the end of the previous LogTimestep

    // NVTX events for GPU timing
    struct GpuEventPair {
        void* start; // cudaEvent_t
        void* stop;  // cudaEvent_t
    };
    static std::unordered_map<std::string, GpuEventPair> m_GpuEvents;
};

inline std::string format_hms(double seconds) {
    if (seconds < 0.0) { seconds = 0.0; }
    long long total = static_cast<long long>(seconds + 0.5);
    long long h     = total / 3600;
    long long m     = (total % 3600) / 60;
    long long s     = total % 60;

    std::ostringstream os;
    os << std::setfill('0') << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
    return os.str();
}

void print_max_memory_usage();

#endif // PROFILER_H
