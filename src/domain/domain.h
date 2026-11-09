#pragma once

// Splits the box among the ranks along the Hilbert curve, and gives the owner of a point and the ranks a ball touches.

#include "device/device.h"
#include "math/math.h"

#include <cstdint>

namespace domain {

    struct DomainPointers;

    // the cuts, the same on every rank
    class Domain {
      public:
        // cut r is the smallest key with at least r N / n_ranks of the N cells of all ranks below it; collective
        void cut_by_number_of_cells(const Vec* positions, int n_cells);

        // the same by work[c]: cut r has at least r W / n_ranks of the work W of all ranks below it; collective
        void cut_by_work(const Vec* positions, const long* work, int n_cells);

        DomainPointers pointers() const;

      private:
        int                     n_ranks = 0;
        device::Array<uint64_t> cuts{"domain: cuts", 0};
    };

    // the cuts are made again when the busiest rank holds more than this times the mean number of cells
    constexpr double MOST_CELLS_OVER_MEAN = 1.1;

    // whether the busiest rank holds more than MOST_CELLS_OVER_MEAN times the mean number of cells; collective
    bool is_unbalanced(int n_cells);

    // the same for the work of the cells; collective
    bool is_unbalanced_by_work(const long* work, int n_cells);

    // how many times the mean number of cells the busiest rank holds; collective
    double imbalance(int n_cells);

    // how many times the mean work the busiest rank has; collective
    double imbalance_by_work(const long* work, int n_cells);

    HD inline int owner_of(const DomainPointers& domain, const Vec& position);

    // calls function(rank) for every rank whose part of [0, 1) the ball touches; a rank may come more than once
    template <typename Function>
    HD void for_each_rank_touched_by_ball(const DomainPointers& domain,
                                          const Vec&            center,
                                          double                radius,
                                          Function              function);

    // whether the ball lies inside [0, 1) and touches no other rank; then this rank holds every cell inside it
    HD inline bool ball_is_inside_this_rank(const DomainPointers& domain, const Vec& center, double radius);

    // plain pointers to the cuts, for kernels
    struct DomainPointers {
        int             n_ranks;
        int             this_rank;
        const uint64_t* cuts; // n_ranks + 1 keys: 0 first, the end of the curve last
    };

} // namespace domain

#include "owners.h"
