#ifndef PROFILER_H
#define PROFILER_H

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>
using namespace std;

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
    static void StartTimer(const string& name);
    static void EndTimer(const string& name);
    static void PrintResults();

private:
    static unordered_map<string, chrono::high_resolution_clock::time_point> m_StartTimes;
    static unordered_map<string, long long> m_Timings;
};

#endif // PROFILER_H
