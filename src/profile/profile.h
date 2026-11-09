#pragma once

// What the timers of all ranks add up to: the summary at the end of a run and the rows of the profile file.

#include <string>
#include <vector>

namespace profile {

    // timers of all ranks, and one row of seconds per recorded top step
    struct Profile {
        std::vector<std::string> paths;
        std::vector<int>         communication; // 1 for a timer of one MPI call, else 0
        std::vector<int>         steps;
        std::vector<double>      times;
        std::vector<double>      seconds; // seconds[row * paths.size() + t]
    };

    // the profile of the run so far; collective
    Profile profile_of_run();

    // prints the tree of timers over all ranks, from rank 0; collective
    void print_summary();

} // namespace profile
