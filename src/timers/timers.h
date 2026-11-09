#pragma once

// Timers for named phases of a run, per rank (only with ENABLE_PROFILING).

#include "config/config.h"

#include <string>
#include <utility>
#include <vector>

namespace timers {

    // work of this rank, or waiting for other ranks
    enum class Type { work, communication };

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

    // seconds of every timer of this rank so far, sorted by path
    std::vector<std::pair<std::string, double>> totals_of_this_rank();

    // keeps the totals of this rank as a row of the profile (at the end of a top step)
    void record_top_step(int step, double time);

    // timers of all ranks, and one row of seconds per recorded top step
    struct Profile {
        std::vector<std::string> paths;
        std::vector<int>         communication; // 1 for a timer that waits for other ranks, else 0
        std::vector<int>         steps;
        std::vector<double>      times;
        std::vector<double>      seconds; // seconds[row * paths.size() + t]
    };

    // the profile of the run so far; collective
    Profile profile_of_run();

    // prints the tree of timers over all ranks, from rank 0; collective
    void print_summary();

} // namespace timers
