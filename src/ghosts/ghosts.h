#pragma once

// Copies of the cells that the balls of this rank reach: cells of other ranks, and periodic copies of any cell.

#include "device/device.h"
#include "domain/domain.h"
#include "ngbtree/ngbtree.h"

#include <cstdint>

namespace ghosts {

    // plain pointers to the ghosts, for kernels; ghost seed g is point n_cells + g of the search tree
    struct GhostPointers {
        int        n_ghost_seeds;
        const Vec* position; // seen from this rank; outside [0, 1) for a periodic copy
        const int* owner;
        const int* cell_on_owner;
        const int* shift; // periodic shift from its cell to the ghost seed
        int        n_ghost_cells; // their values sit at n_cells + ghost_cell in the arrays with ghost cells
        const int* ghost_cell; // of ghost seed g; -1 if it has none
        const int* ghost_seed; // of a ghost cell
    };

    // a ball one rank asks another one (or itself) for, in the frame of the owner, and the shift its answers get
    struct Query {
        Vec    center;
        double radius;
        int    shift;
    };

    // a cell inside a query, as its owner sends it back: moved by the shift into the frame of the asker
    struct Answer {
        Vec position;
        int cell;
        int shift;
    };

    // items for every rank, or from every rank: the block of rank r holds count[r] items from items[first[r]] on
    template <typename Item> struct ItemsByRank {
        int                 total = 0;
        device::Array<Item> items{"ghosts: items to or from the ranks", 0};
        device::Array<int>  count{"ghosts: items per rank", 0};
        device::Array<int>  first{"ghosts: first item per rank", 0};
    };

    // the cells inside the queries that came: a hit is the key (asker, cell, shift) and the tree index of the cell
    struct Hits {
        int                     n = 0;
        device::Array<uint64_t> key{"ghosts: keys of the hits", 0};
        device::Array<int>      tree_index{"ghosts: tree indices of the hits", 0};
        device::Array<int>      count_of_query{"ghosts: hits per query", 0};
        device::Array<int>      first_of_query{"ghosts: first hit per query", 0};

        // the hits not sent before, in key order
        int                     n_new = 0;
        device::Array<int>      new_hit{"ghosts: new hits", 0};
        device::Array<uint64_t> new_key{"ghosts: keys of the new hits", 0};
        device::Array<int>      new_tree_index{"ghosts: tree indices of the new hits", 0};
    };

    // the ghosts, and what this rank sent
    class Ghosts {
      public:
        // forgets the ghost seeds, the ghost cells and what was sent
        void start_new_build();

        // the cells in the balls not here yet become ghost seeds; drift_margin widens a ball by how far a seed may have
        // left the part of its rank; tree has the n_cells cells as input indices 0 to n_cells - 1; collective
        void add_cells_in_balls(const Vec*            center,
                                const double*         radius,
                                int                   n_balls,
                                double                drift_margin,
                                const domain::Domain& domain,
                                const ngbtree::Tree&  tree,
                                int                   n_cells);

        // every used ghost seed without a ghost cell gets one, behind those so far; collective
        void give_ghost_cells_to(const bool* used, int n_cells);

        // every field to field[n_cells + ghost_cell] on every rank that holds the cell as a ghost cell, in one
        // exchange; for_each_field(function) calls function(field) for every array; collective
        template <typename FieldList> void copy_to_ghost_cells(int n_cells, FieldList for_each_field);

        // the same for the sorted cells changed[i] only; nullptr, on every rank at once, is every cell; collective
        template <typename FieldList>
        void copy_changed_to_ghost_cells(int n_cells, const int* changed, int n_changed, FieldList for_each_field);

        // every ghost seed to the seed of its cell plus its shift; seeds[n_cells + ghost_cell] for other ranks' cells
        void move_to(const Vec* seeds, int n_cells);

        // marked_cell[c] = true for every cell c that a rank holds as a marked ghost cell; collective
        void mark_cells_of_ghost_cells(int n_cells, const bool* marked_ghost_cell, bool* marked_cell);

        GhostPointers pointers() const;

        // the parts of copy_changed_to_ghost_cells; public only as nvcc wants GPU loops in public member functions
        int start_items_of_changed_cells(const int* changed, int n_changed, int item_bytes);
        void find_ghost_cells_of_items_that_came(int item_bytes);

      private:
        int                n_ghost_seeds = 0;
        device::Array<Vec> position{"ghosts: positions", 0};
        device::Array<int> owner{"ghosts: owners", 0};
        device::Array<int> cell_on_owner{"ghosts: cells on their owners", 0};
        device::Array<int> shift{"ghosts: periodic shifts", 0};

        // the sorted keys (asker, cell, shift) of the cells sent in this build, so that none goes twice
        int                     n_sent = 0;
        device::Array<uint64_t> sent{"ghosts: sent keys", 0};
        device::Array<uint64_t> merged_sent{"ghosts: sent keys", 0};

        // ghost_cell is filled up to n_seeds_with_ghost_cell; a ghost cell keeps its number until the next build
        int                n_ghost_cells           = 0;
        int                n_seeds_with_ghost_cell = 0;
        device::Array<int> ghost_cell{"ghosts: ghost cell of every ghost seed", 0};
        device::Array<int> ghost_seed{"ghosts: ghost seed of every ghost cell", 0};

        // n_cells + ghost_cell of the ghost cells by owner, in the order the values from the owners come
        device::Array<int> ghost_cell_place_by_owner{"ghosts: places of the ghost cells by owner", 0};

        // what the copies send: ghost cells per owner here, cells sent per rank in the order of its ghost cells
        device::Array<int> ghost_cells_of_rank{"ghosts: ghost cells of every owner", 0};
        device::Array<int> cells_for_rank{"ghosts: cells sent to every rank", 0};
        int                n_cells_sent = 0;
        device::Array<int> cell_sent{"ghosts: cells sent, by asking rank", 0};
        device::Array<int> first_ghost_cell_of_rank{"ghosts: first ghost cell of every owner", 0};
        device::Array<int> first_cell_for_rank{"ghosts: first cell sent to every rank", 0};

        // the work of a copy of changed cells
        device::Array<int> changed_sent{"ghosts: places of the changed cells sent", 0};
        device::Array<int> changed_cell_sent{"ghosts: changed cells sent", 0};
        device::Array<int> changed_for_rank{"ghosts: changed cells sent to every rank", 0};
        device::Array<int> changed_from_rank{"ghosts: changed cells from every rank", 0};
        device::Array<int> first_changed_from_rank{"ghosts: first changed cell from every rank", 0};
        int                n_changed_received = 0;
        device::Array<int> ghost_cell_of_changed{"ghosts: ghost cells of the changed cells that came", 0};

        // the work of one call, kept so that the next call reuses the memory
        device::Array<char> values_sent{"ghosts: values sent to the ghost cells", 0};
        device::Array<char> values_received{"ghosts: values received for the ghost cells", 0};
        ItemsByRank<Query>  queries_out, queries_in;
        ItemsByRank<Answer> answers_out, answers_in;
        device::Array<int>  next_place{"ghosts: next place in the block of a rank", 0};
        Hits                hits;
    };

} // namespace ghosts

#include "ghost_cells.h"
