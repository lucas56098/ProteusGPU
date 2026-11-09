// implements the timers (timers.h): a total per path and a stack of open scopes, global per rank

#include "timers/timers.h"

#include "device/device.h"
#include "logging/logging.h"
#include "mpi/mpi.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>

namespace timers {

    namespace {

        using Clock = std::chrono::steady_clock;

        constexpr size_t LINE_WIDTH = 70;

        struct Timer {
            double seconds = 0.0;
            Type   type    = Type::work;
        };

        struct OpenScope {
            std::string       path;
            Clock::time_point start;
        };

        // timers of this rank by path, and its open scopes (innermost last)
        std::map<std::string, Timer> timers_of_rank;
        std::vector<OpenScope>       open_scopes;

        // profile rows: totals per top step, in the order of paths_of_rows; a row holds the paths known then
        struct Row {
            int                 step;
            double              time;
            std::vector<double> seconds;
        };
        std::vector<Row>           rows_of_profile;
        std::vector<std::string>   paths_of_rows;
        std::map<std::string, int> place_in_rows;

        struct TimerOfRank {
            long long microseconds;
            int       communication;
        };

        struct TimerOfRun {
            std::string path;
            double      seconds_summed;
            double      imbalance;
            bool        communication;
        };

        std::vector<std::string> paths_of_every_rank(const std::vector<std::pair<std::string, double>>& totals);
        std::vector<TimerOfRun> timers_of_run(const std::vector<std::string>&                    paths,
                                              const std::vector<std::pair<std::string, double>>& totals);
        void print_tree(const std::vector<TimerOfRun>& timers, const std::string& parent, int depth, double total);
        std::string parent_of(const std::string& path);

    } // namespace

    // a timer counts as communication if any of its scopes does
    void open_scope(const char* name, Type type) {
        const std::string path  = open_scopes.empty() ? std::string(name) : open_scopes.back().path + "." + name;
        Timer&            timer = timers_of_rank[path];
        if (type == Type::communication) timer.type = Type::communication;
        if constexpr (config::cuda_profiling) device::begin_named_range(path.c_str());
        open_scopes.push_back(OpenScope{path, Clock::now()});
    }

    void close_scope() {
        const OpenScope& scope = open_scopes.back();
        timers_of_rank[scope.path].seconds += std::chrono::duration<double>(Clock::now() - scope.start).count();
        if constexpr (config::cuda_profiling) device::end_named_range();
        open_scopes.pop_back();
    }

    // open scopes count up to now
    std::vector<std::pair<std::string, double>> totals_of_this_rank() {
        std::map<std::string, double> totals;
        for (const auto& [path, timer] : timers_of_rank)
            totals[path] = timer.seconds;
        const Clock::time_point now = Clock::now();
        for (const OpenScope& scope : open_scopes)
            totals[scope.path] += std::chrono::duration<double>(now - scope.start).count();
        return std::vector<std::pair<std::string, double>>(totals.begin(), totals.end());
    }

    void record_top_step(int step, double time) {
        const std::vector<std::pair<std::string, double>> totals = totals_of_this_rank();
        for (const auto& [path, seconds] : totals) {
            if (place_in_rows.count(path) == 0) {
                place_in_rows[path] = (int)paths_of_rows.size();
                paths_of_rows.push_back(path);
            }
        }
        Row row{step, time, std::vector<double>(paths_of_rows.size(), 0.0)};
        for (const auto& [path, seconds] : totals)
            row.seconds[place_in_rows[path]] = seconds;
        rows_of_profile.push_back(row);
    }

    // paths of all ranks; 0 in a row for a timer that did not exist yet
    Profile profile_of_run() {
        const std::vector<std::pair<std::string, double>> totals = totals_of_this_rank();
        Profile                                           profile;
        profile.paths = paths_of_every_rank(totals);
        for (const TimerOfRun& timer : timers_of_run(profile.paths, totals))
            profile.communication.push_back(timer.communication ? 1 : 0);
        for (const Row& row : rows_of_profile) {
            profile.steps.push_back(row.step);
            profile.times.push_back(row.time);
            for (const std::string& path : profile.paths) {
                const auto found = place_in_rows.find(path);
                const bool there = found != place_in_rows.end() && found->second < (int)row.seconds.size();
                profile.seconds.push_back(there ? row.seconds[found->second] : 0.0);
            }
        }
        return profile;
    }

    // shares are of the largest timer, TOTAL
    void print_summary() {
        if constexpr (!config::profiling) return;

        const std::vector<std::pair<std::string, double>> totals = totals_of_this_rank();
        const std::vector<TimerOfRun>                     timers = timers_of_run(paths_of_every_rank(totals), totals);

        double total = 0.0, waiting = 0.0;
        for (const TimerOfRun& timer : timers) {
            total = std::max(total, timer.seconds_summed);
            if (timer.communication) waiting += timer.seconds_summed;
        }

        const std::string title = " Profiling Results ";
        const size_t      side  = (LINE_WIDTH - title.size()) / 2;
        const std::string thin(LINE_WIDTH, '-');
        const std::string thick(LINE_WIDTH, '=');
        logging::print(
            "\n%s%s%s", thick.substr(0, side).c_str(), title.c_str(), thick.substr(side + title.size()).c_str());

        const int ranks = mpi::nranks();
        char      who[80];
        std::snprintf(who,
                      sizeof who,
                      "sum of %d ranks (%d threads, %d GPUs)",
                      ranks,
                      ranks * device::n_threads(),
                      config::use_gpu ? ranks : 0);
        logging::print("%-37s  %8s  %10s  %9s", who, "time", "percentage", "imbalance");
        logging::print("%s", thin.c_str());
        print_tree(timers, "", 0, total);
        logging::print("%s", thin.c_str());
        logging::print("%-30s  %-5s  %7.3fs  %9.1f%%",
                       "MPI total",
                       "[mpi]",
                       waiting,
                       (total > 0.0) ? 100.0 * waiting / total : 0.0);
        logging::print("%s\n", thick.c_str());
    }

    namespace {

        // paths of all ranks, sorted, each once; sent as names ending in a zero byte; collective
        std::vector<std::string> paths_of_every_rank(const std::vector<std::pair<std::string, double>>& totals) {
            std::string names;
            for (const auto& [path, seconds] : totals)
                names += path + '\0';
            const std::vector<char> all = mpi::gather_items_from_all_ranks(names.data(), (int)names.size(), 1);

            std::set<std::string> paths;
            for (size_t start = 0; start < all.size();) {
                const std::string path(&all[start]);
                paths.insert(path);
                start += path.size() + 1;
            }
            return std::vector<std::string>(paths.begin(), paths.end());
        }

        // sum and imbalance of every path over all ranks; sent in microseconds, 0 where a rank has none; collective
        std::vector<TimerOfRun> timers_of_run(const std::vector<std::string>&                    paths,
                                              const std::vector<std::pair<std::string, double>>& totals) {
            const std::map<std::string, double> mine(totals.begin(), totals.end());
            std::vector<TimerOfRank>            of_rank(paths.size());
            for (size_t i = 0; i < paths.size(); i++) {
                const auto found         = mine.find(paths[i]);
                const auto timer         = timers_of_rank.find(paths[i]);
                of_rank[i].microseconds  = (found == mine.end()) ? 0 : (long long)(found->second * 1e6 + 0.5);
                of_rank[i].communication = (timer != timers_of_rank.end() && timer->second.type == Type::communication);
            }
            const std::vector<char> all =
                mpi::gather_items_from_all_ranks(of_rank.data(), (int)of_rank.size(), sizeof(TimerOfRank));

            const int               n_ranks = mpi::nranks();
            std::vector<TimerOfRun> timers;
            for (size_t i = 0; i < paths.size(); i++) {
                long long summed = 0, slowest = 0;
                bool      communication = false;
                for (int rank = 0; rank < n_ranks; rank++) {
                    TimerOfRank timer;
                    std::memcpy(&timer, &all[(rank * paths.size() + i) * sizeof(TimerOfRank)], sizeof(TimerOfRank));
                    summed += timer.microseconds;
                    slowest       = std::max(slowest, timer.microseconds);
                    communication = communication || timer.communication;
                }
                const double mean = (double)summed / n_ranks;
                timers.push_back(
                    TimerOfRun{paths[i], summed / 1e6, (mean > 0.0) ? slowest / mean : 1.0, communication});
            }
            return timers;
        }

        // prints the children of parent, largest first, each followed by its own children
        void print_tree(const std::vector<TimerOfRun>& timers, const std::string& parent, int depth, double total) {
            std::vector<const TimerOfRun*> children;
            for (const TimerOfRun& timer : timers)
                if (parent_of(timer.path) == parent) children.push_back(&timer);
            std::sort(children.begin(), children.end(), [](const TimerOfRun* a, const TimerOfRun* b) {
                return a->seconds_summed > b->seconds_summed;
            });

            for (const TimerOfRun* timer : children) {
                const std::string name =
                    std::string(2 * depth, ' ') + timer->path.substr(parent.empty() ? 0 : parent.size() + 1);
                logging::print("%-30s  %-5s  %7.3fs  %9.1f%%  %9.3f",
                               name.c_str(),
                               timer->communication ? "[mpi]" : "     ",
                               timer->seconds_summed,
                               (total > 0.0) ? 100.0 * timer->seconds_summed / total : 0.0,
                               timer->imbalance);
                print_tree(timers, timer->path, depth + 1, total);
            }
        }

        std::string parent_of(const std::string& path) {
            const size_t dot = path.rfind('.');
            return (dot == std::string::npos) ? std::string() : path.substr(0, dot);
        }

    } // namespace

} // namespace timers
