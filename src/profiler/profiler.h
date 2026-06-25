#ifndef PROFILER_H
#define PROFILER_H

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

// Profiler — hierarchical CPU + GPU + MPI timing.
//
// Three RAII scope kinds; each one pushes a node onto a path stack so the
// runtime parent of every timer is its enclosing scope. The full path
// (DOMAIN.SUB.LEAF) is what gets stored / printed / written to HDF5.
//
//   PROFILE("MESH.CELLS")          // CPU work, kind=c
//   PROFILE_MPI("WAIT")            // MPI call site, kind=m
//   PROFILE_KERNEL("FLUX_KERNEL")  // CUDA kernel, kind=g — GPU events
//                                  //   queried lazily at LogTimestep
//
// The macros are RAII over a Scope/MpiScope/KernelScope; the destructor pops
// the stack and accumulates elapsed time. Don't mix with manual Start/End.
class Profiler {
  public:
    // CPU host wall-clock scope.
    class Scope {
      public:
        explicit Scope(const char* short_name);
        ~Scope();
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;

      private:
        std::string m_path;
    };

    // CPU wall-clock scope, marked as MPI work in the tree.
    class MpiScope {
      public:
        explicit MpiScope(const char* short_name);
        ~MpiScope();
        MpiScope(const MpiScope&)            = delete;
        MpiScope& operator=(const MpiScope&) = delete;

      private:
        std::string m_path;
    };

    // CUDA-kernel scope: records start/stop events on the default stream.
    // CPU launch time is folded into the parent (host scope still ticks, but
    // the leaf's stored value is GPU device time). Events drain lazily.
    class KernelScope {
      public:
        explicit KernelScope(const char* short_name);
        ~KernelScope();
        KernelScope(const KernelScope&)            = delete;
        KernelScope& operator=(const KernelScope&) = delete;

      private:
        std::string m_path;
#ifdef CUDA_PROFILING
        void* m_start_event; // cudaEvent_t — owned during this scope's lifetime
#endif
    };

    static void   StartTotalTimer();
    static void   StopTotalTimer();
    static double TotalSeconds();

    // End-of-run summary on rank 0 (with cross-rank min/avg/max).
    static void PrintResults();

    // Shared profile.hdf5 with one /rank_<R>/per_step/<TIMER> + /rank_<R>/cumulative/<TIMER>
    // dataset per timer (full hierarchical path). Each dataset gets a @kind attribute
    // ("cpu" | "mpi" | "gpu"). restart_step >= 0 opens for append and truncates rows
    // past restart_step+1.
    static void OpenProfileLog(const std::string& path, int restart_step);
    static void CloseProfileLog();
    static void LogTimestep(int step);

    // Seed in-memory cumulative timings from a snapshot's /header/profiler group.
    // Must run before any new Start so subsequent diffs are computed from the
    // restored baseline. TOTAL is rewound by adjusting its live start time so
    // CollectCurrent's live offset includes the resumed runtime.
    static void SeedFromCumulative(const std::unordered_map<std::string, double>& cum_sec);

    // Current cumulative seconds per full-path timer (live values for any
    // currently-open scopes are folded in). Used by output.cu to snapshot
    // profiler state for restart-on-snapshot.
    static std::unordered_map<std::string, double> CurrentCumulative();

  private:
    // Non-blocking drain of completed GPU events into the cumulative GPU map.
    // Called from LogTimestep (every step) and PrintResults (force-sync).
    static void DrainGpuEvents(bool force_sync);

    // Build the (name, cum_us) view this rank currently has, with live timers
    // (TOTAL, HYDRO) extended to "now". One row per full-path timer, regardless
    // of kind — for cpu/mpi rows the unit is CPU µs, for gpu rows it's GPU µs.
    static std::vector<std::pair<std::string, long long>> CollectCurrent();
};

// Macros — each one declares a uniquely-named RAII object so multiple PROFILE
// lines can live in the same scope. __COUNTER__ would also work; __LINE__ is
// enough and the diagnostics are kinder.
#define PROFILE_CAT_(a, b) a##b
#define PROFILE_CAT(a, b) PROFILE_CAT_(a, b)
#define PROFILE(name) Profiler::Scope PROFILE_CAT(_prof_scope_, __LINE__)(name)
#define PROFILE_MPI(name) Profiler::MpiScope PROFILE_CAT(_prof_mscope_, __LINE__)(name)
#define PROFILE_KERNEL(name) Profiler::KernelScope PROFILE_CAT(_prof_kscope_, __LINE__)(name)

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
