#ifndef PROFILER_H
#define PROFILER_H

// Scoped timers, printed at the end of the run and written to profile.hdf5.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef CUDA_PROFILING
#include "nvtx3/nvToolsExt.h"
#endif

// all state is static, one per process
class Profiler {
  public:
#ifdef ENABLE_PROFILING
    // times its scope, path = parent path + name
    class Scope {
      public:
        explicit Scope(const char* short_name);
        ~Scope();
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;

      private:
        std::string m_path;
    };

    // same, counted as mpi time
    class MpiScope {
      public:
        explicit MpiScope(const char* short_name);
        ~MpiScope();
        MpiScope(const MpiScope&)            = delete;
        MpiScope& operator=(const MpiScope&) = delete;

      private:
        std::string m_path;
    };

    // GPU time from CUDA events, CUDA builds only
    class KernelScope {
      public:
        explicit KernelScope(const char* short_name);
        ~KernelScope();
        KernelScope(const KernelScope&)            = delete;
        KernelScope& operator=(const KernelScope&) = delete;

      private:
        std::string m_path;
#ifdef CUDA_PROFILING
        void* m_start_event;
#endif
    };

    // TOTAL timer around the whole run
    static void   start_total_timer();
    static void   stop_total_timer();
    static double total_seconds();

    // timer tree at the end of the run
    static void print_results();

    // profile.hdf5: open, one row per step, close
    static void open_profile_log(const std::string& path, int restart_step);
    static void close_profile_log();
    static void log_timestep(int step);

    // close path of exit_failure
    static void abort_profile_log();

    // restart: start from the totals in the snapshot
    static void seed_from_cumulative(const std::unordered_map<std::string, double>& cum_sec);

    // totals per path, stored in every snapshot
    static std::unordered_map<std::string, double> current_cumulative();

  private:
    static void drain_gpu_events(bool force_sync);

    static std::vector<std::pair<std::string, long long>> collect_current();
#else
    // without ENABLE_PROFILING everything here compiles to nothing
    struct Scope {
        explicit Scope(const char*) {}
    };
    struct MpiScope {
        explicit MpiScope(const char*) {}
    };
    struct KernelScope {
        explicit KernelScope(const char*) {}
    };

    static inline void   start_total_timer() {}
    static inline void   stop_total_timer() {}
    static inline double total_seconds() { return 0.0; }
    static inline void   print_results() {}
    static inline void   open_profile_log(const std::string&, int) {}
    static inline void   close_profile_log() {}
    static inline void   log_timestep(int) {}
    static inline void   abort_profile_log() {}
    static inline void   seed_from_cumulative(const std::unordered_map<std::string, double>&) {}
    static inline std::unordered_map<std::string, double> current_cumulative() { return {}; }
#endif
};

#ifdef ENABLE_PROFILING
#define PROFILE_CAT_(a, b) a##b
#define PROFILE_CAT(a, b) PROFILE_CAT_(a, b)
// one scope object per use, __LINE__ makes the name unique
#define PROFILE(name) Profiler::Scope PROFILE_CAT(_prof_scope_, __LINE__)(name)
#define PROFILE_MPI(name) Profiler::MpiScope PROFILE_CAT(_prof_mscope_, __LINE__)(name)
#define PROFILE_KERNEL(name) Profiler::KernelScope PROFILE_CAT(_prof_kscope_, __LINE__)(name)
#else
#define PROFILE(name) ((void)0)
#define PROFILE_MPI(name) ((void)0)
#define PROFILE_KERNEL(name) ((void)0)
#endif

// seconds as hh:mm:ss
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

// peak memory line at the end of the run
void print_max_memory_usage();

#endif
