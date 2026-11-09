// implements the timers (timers.h): a total per path and a stack of open scopes, global per rank

#include "timers/timers.h"

#ifdef CUDA_PROFILING
#include <nvtx3/nvToolsExt.h>
#endif

#include <chrono>
#include <map>

namespace timers {

    namespace {

        using Clock = std::chrono::steady_clock;

        struct Timer {
            double    seconds = 0.0;
            long long calls   = 0;
            Type      type    = Type::work;
        };

        struct OpenScope {
            std::string       path;
            Clock::time_point start;
        };

        // timers of this rank by path, and its open scopes (innermost last)
        std::map<std::string, Timer> timers_of_rank;
        std::vector<OpenScope>       open_scopes;

        // profile rows: totals per top step, in the order of paths_of_rows
        std::vector<RowOfTopStep>  rows_of_profile;
        std::vector<std::string>   paths_of_rows;
        std::map<std::string, int> place_in_rows;

    } // namespace

    // with CUDA_PROFILING the timers show up as named ranges in Nsight Systems
    void open_scope(const char* name, Type type) {
        const std::string path  = open_scopes.empty() ? std::string(name) : open_scopes.back().path + "." + name;
        Timer&            timer = timers_of_rank[path];
        timer.type              = type;
        timer.calls++;
#ifdef CUDA_PROFILING
        nvtxRangePushA(path.c_str());
#endif
        open_scopes.push_back(OpenScope{path, Clock::now()});
    }

    void close_scope() {
        const OpenScope& scope = open_scopes.back();
        timers_of_rank[scope.path].seconds += std::chrono::duration<double>(Clock::now() - scope.start).count();
#ifdef CUDA_PROFILING
        nvtxRangePop();
#endif
        open_scopes.pop_back();
    }

    std::vector<Total> totals_of_this_rank() {
        std::map<std::string, Timer> timers = timers_of_rank;
        const Clock::time_point      now    = Clock::now();
        for (const OpenScope& scope : open_scopes)
            timers[scope.path].seconds += std::chrono::duration<double>(now - scope.start).count();

        std::vector<Total> totals;
        for (const auto& [path, timer] : timers)
            totals.push_back(Total{path, timer.seconds, timer.calls, timer.type});
        return totals;
    }

    void record_top_step(int step, double time) {
        const std::vector<Total> totals = totals_of_this_rank();
        for (const Total& total : totals) {
            if (place_in_rows.count(total.path) == 0) {
                place_in_rows[total.path] = (int)paths_of_rows.size();
                paths_of_rows.push_back(total.path);
            }
        }
        RowOfTopStep row{step, time, std::vector<double>(paths_of_rows.size(), 0.0)};
        for (const Total& total : totals)
            row.seconds[place_in_rows[total.path]] = total.seconds;
        rows_of_profile.push_back(row);
    }

    const std::vector<RowOfTopStep>& rows_of_top_steps() {
        return rows_of_profile;
    }

    int place_in_rows_of_top_steps(const std::string& path) {
        const auto found = place_in_rows.find(path);
        return (found == place_in_rows.end()) ? -1 : found->second;
    }

} // namespace timers
