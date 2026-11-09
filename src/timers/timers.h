#pragma once

// Timers for named phases of a run, per rank (only with ENABLE_PROFILING); the profile module adds them up.

#include "config/config.h"

#include <string>
#include <vector>

namespace timers {

    // a phase of the run, one MPI call, or one GPU loop
    enum class Type { work, mpi, gpu };

    // what Scope calls
    void open_scope(const char* name, Type type);
    void close_scope();

    // times its scope into the timer of its name, inside the scope around it (e.g. TOTAL.START.FLUX)
    class Scope {
      public:
        explicit Scope(const char* name, Type type = Type::work) {
            if constexpr (config::profiling) open_scope(name, type);
        }
        ~Scope() {
            if constexpr (config::profiling) close_scope();
        }
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
    };

    // a timer of this rank so far
    struct Total {
        std::string path;
        double      seconds;
        long long   calls;
        Type        type;
    };

    // every timer of this rank, sorted by path; open scopes count up to now
    std::vector<Total> totals_of_this_rank();

    // keeps the totals of this rank as a row of the profile (at the end of a top step)
    void record_top_step(int step, double time);

    // a row of the profile; it holds the paths known when it was recorded
    struct RowOfTopStep {
        int                 step;
        double              time;
        std::vector<double> seconds;
    };

    const std::vector<RowOfTopStep>& rows_of_top_steps();

    // where the seconds of a path are in the rows that know it; -1 for a path no row knows
    int place_in_rows_of_top_steps(const std::string& path);

} // namespace timers
