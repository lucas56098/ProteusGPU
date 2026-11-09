#pragma once

// Moves cells with all their fields to other ranks; the caller gives every destination and the per-cell arrays.

#include "device/device.h"

namespace migration {

    // which cells leave and in what order, how many go to and come from every rank, and which staying cells fill in
    class Plan {
      public:
        // the plan for cell c to go to rank destination[c]; a rank may keep a cell; collective
        void make(const int* destination, int n_cells);

        // cells after the moves: the staying ones, then those that came, by source rank in their order there
        int n_cells_after() const;

        // moves every field in one exchange; for_each_field(function) calls function on every array; collective
        template <typename FieldList> void move_cells(FieldList for_each_field) const;

      private:
        int                n_cells    = 0;
        int                n_leaving  = 0;
        int                n_arriving = 0;
        device::Array<int> leaving_order{"migration: cells in the order they leave", 0};
        device::Array<int> send_count{"migration: cells to every rank", 0};
        device::Array<int> receive_count{"migration: cells from every rank", 0};
        int                n_places = 0; // of leaving cells before n_cells - n_leaving
        device::Array<int> place{"migration: places of leaving cells to fill", 0};
        device::Array<int> filler{"migration: staying cells that fill them", 0};
    };

} // namespace migration

#include "moves.h"
