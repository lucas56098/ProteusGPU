#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#ifdef ENABLE_PROFILING
#define PROFILE_START(name) Profiler::StartTimer(name)
#define PROFILE_END(name) Profiler::EndTimer(name)
#define PROFILE_PRINT_RESULTS() Profiler::PrintResults()
#else
#define PROFILE_START(name)
#define PROFILE_END(name)
#define PROFILE_PRINT_RESULTS()
#endif

class Profiler {
  public:
    static void StartTimer(const std::string& name);
    static void EndTimer(const std::string& name);
    static void PrintResults();

  private:
    static std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> m_StartTimes;
    static std::unordered_map<std::string, long long>                                      m_Timings;
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
