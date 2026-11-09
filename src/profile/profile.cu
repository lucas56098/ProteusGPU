// implements what the timers of all ranks add up to (profile.h)

#include "profile/profile.h"

#include "device/device.h"
#include "logging/logging.h"
#include "mpi/mpi.h"
#include "timers/timers.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>

namespace profile {

    namespace {

        constexpr size_t LINE_WIDTH = 82;

        struct TimerOfRank {
            long long microseconds;
            long long calls;
            int       type;
        };

        struct TimerOfRun {
            std::string  path;
            double       seconds_summed;
            double       imbalance;
            long long    calls;
            timers::Type type;
        };

        std::vector<std::string> paths_of_every_rank(const std::vector<timers::Total>& totals);
        std::vector<TimerOfRun> timers_of_run(const std::vector<std::string>&   paths,
                                              const std::vector<timers::Total>& totals);
        void print_tree(const std::vector<TimerOfRun>& timers, const std::string& parent, int depth, double total);
        void print_share(const char* name, const char* tag, double seconds, double total);
        const char* tag_of(timers::Type type);
        std::string parent_of(const std::string& path);

    } // namespace

    // paths of all ranks; 0 in a row for a timer that did not exist yet
    Profile profile_of_run() {
        const std::vector<timers::Total> totals = timers::totals_of_this_rank();
        Profile                          of_run;
        of_run.paths = paths_of_every_rank(totals);
        for (const TimerOfRun& timer : timers_of_run(of_run.paths, totals))
            of_run.communication.push_back((timer.type == timers::Type::mpi) ? 1 : 0);
        for (const timers::RowOfTopStep& row : timers::rows_of_top_steps()) {
            of_run.steps.push_back(row.step);
            of_run.times.push_back(row.time);
            for (const std::string& path : of_run.paths) {
                const int  place = timers::place_in_rows_of_top_steps(path);
                const bool there = place >= 0 && place < (int)row.seconds.size();
                of_run.seconds.push_back(there ? row.seconds[place] : 0.0);
            }
        }
        return of_run;
    }

    // shares are of the largest timer, TOTAL; an MPI call or a GPU loop never holds another one, so none counts twice
    void print_summary() {
        if constexpr (!config::profiling) return;

        const std::vector<timers::Total> totals = timers::totals_of_this_rank();
        const std::vector<TimerOfRun>    of_run = timers_of_run(paths_of_every_rank(totals), totals);

        double total = 0.0, in_mpi = 0.0, in_gpu = 0.0;
        for (const TimerOfRun& timer : of_run) {
            total = std::max(total, timer.seconds_summed);
            if (timer.type == timers::Type::mpi) in_mpi += timer.seconds_summed;
            if (timer.type == timers::Type::gpu) in_gpu += timer.seconds_summed;
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
        logging::print("%-37s  %8s  %10s  %9s  %10s", who, "time", "percentage", "imbalance", "calls");
        logging::print("%s", thin.c_str());
        print_tree(of_run, "", 0, total);
        logging::print("%s", thin.c_str());
        print_share("MPI total", "[mpi]", in_mpi, total);
        print_share("GPU total", "[gpu]", in_gpu, total);
        print_share("Everything else", "", total - in_mpi - in_gpu, total);
        logging::print("%s\n", thick.c_str());
    }

    namespace {

        // paths of all ranks, sorted, each once; sent as names ending in a zero byte; collective
        std::vector<std::string> paths_of_every_rank(const std::vector<timers::Total>& totals) {
            std::string names;
            for (const timers::Total& total : totals)
                names += total.path + '\0';
            const std::vector<char> all = mpi::gather_items_from_all_ranks(names.data(), (int)names.size(), 1);

            std::set<std::string> paths;
            for (size_t start = 0; start < all.size();) {
                const std::string path(&all[start]);
                paths.insert(path);
                start += path.size() + 1;
            }
            return std::vector<std::string>(paths.begin(), paths.end());
        }

        // sum, imbalance and calls of every path over all ranks; sent in microseconds, 0 where a rank has none;
        // collective
        std::vector<TimerOfRun> timers_of_run(const std::vector<std::string>&   paths,
                                              const std::vector<timers::Total>& totals) {
            std::map<std::string, timers::Total> mine;
            for (const timers::Total& total : totals)
                mine.emplace(total.path, total);

            std::vector<TimerOfRank> of_rank(paths.size(), TimerOfRank{0, 0, 0});
            for (size_t i = 0; i < paths.size(); i++) {
                const auto found = mine.find(paths[i]);
                if (found == mine.end()) continue;
                of_rank[i].microseconds = (long long)(found->second.seconds * 1e6 + 0.5);
                of_rank[i].calls        = found->second.calls;
                of_rank[i].type         = (int)found->second.type;
            }
            const std::vector<char> all =
                mpi::gather_items_from_all_ranks(of_rank.data(), (int)of_rank.size(), sizeof(TimerOfRank));

            const int               n_ranks = mpi::nranks();
            std::vector<TimerOfRun> timers;
            for (size_t i = 0; i < paths.size(); i++) {
                long long summed = 0, slowest = 0, calls = 0;
                int       type = 0;
                for (int rank = 0; rank < n_ranks; rank++) {
                    TimerOfRank timer;
                    std::memcpy(&timer, &all[(rank * paths.size() + i) * sizeof(TimerOfRank)], sizeof(TimerOfRank));
                    summed += timer.microseconds;
                    slowest = std::max(slowest, timer.microseconds);
                    calls += timer.calls;
                    type = std::max(type, timer.type);
                }
                const double mean = (double)summed / n_ranks;
                timers.push_back(TimerOfRun{paths[i],
                                            summed / 1e6,
                                            (mean > 0.0) ? slowest / mean : 1.0,
                                            calls,
                                            static_cast<timers::Type>(type)});
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
                logging::print("%-30s  %-5s  %7.3fs  %9.1f%%  %9.3f  %10lld",
                               name.c_str(),
                               tag_of(timer->type),
                               timer->seconds_summed,
                               (total > 0.0) ? 100.0 * timer->seconds_summed / total : 0.0,
                               timer->imbalance,
                               timer->calls);
                print_tree(timers, timer->path, depth + 1, total);
            }
        }

        void print_share(const char* name, const char* tag, double seconds, double total) {
            logging::print(
                "%-30s  %-5s  %7.3fs  %9.1f%%", name, tag, seconds, (total > 0.0) ? 100.0 * seconds / total : 0.0);
        }

        const char* tag_of(timers::Type type) {
            switch (type) {
            case timers::Type::mpi:
                return "[mpi]";
            case timers::Type::gpu:
                return "[gpu]";
            case timers::Type::work:
                return "";
            }
            return "";
        }

        std::string parent_of(const std::string& path) {
            const size_t dot = path.rfind('.');
            return (dot == std::string::npos) ? std::string() : path.substr(0, dot);
        }

    } // namespace

} // namespace profile
