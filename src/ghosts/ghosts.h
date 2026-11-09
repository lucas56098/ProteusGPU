#pragma once

// Copies of the cells that the balls of this rank reach: cells of other ranks, and periodic copies of any cell.

#include "device/device.h"
#include "domain/domain.h"
#include "mpi/mpi.h"
#include "ngbtree/ngbtree.h"

#include <cstdint>
#include <vector>

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

    // a new ghost cell, as its owner learns of it: its cell there, and the rank that holds the ghost cell
    struct NewGhostCell {
        int cell;
        int asker;
    };

    // the ghosts, and what this rank sent
    class Ghosts {
      public:
        // forgets the ghost seeds, the ghost cells and what was sent
        void start_new_build();

        // the cells in the balls not here yet become ghost seeds; drift_margin widens a ball by how far a seed may have
        // left the part of its rank; tree has the n_cells cells as input indices 0 to n_cells - 1; returns how many
        // balls all ranks asked for; collective
        long long add_cells_in_balls(const Vec*            center,
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

        // the same, but only into the ghost cells of the last give_ghost_cells_to; collective
        template <typename FieldList> void copy_to_new_ghost_cells(FieldList for_each_field);

        // at the start of a sync that is not the end of a top step: which cells go to which ghost cells at this sync,
        // those whose step ends at tick; end also holds the ghost cells, and both sides of a copy read the same value
        void find_active_cells(const int* end, int tick);

        // into the ghost cells of the active cells of find_active_cells; of every cell if all are active; collective
        template <typename FieldList>
        void copy_active_to_ghost_cells(bool every_cell_is_active, int n_cells, FieldList for_each_field);

        // every ghost seed to the seed of its cell plus its shift; seeds[n_cells + ghost_cell] for other ranks' cells
        void move_to(const Vec* seeds, int n_cells);

        // marked_cell[c] = true for every cell c that a rank holds as a marked ghost cell; collective
        void mark_cells_of_ghost_cells(int n_cells, const bool* marked_ghost_cell, bool* marked_cell);

        GhostPointers pointers() const;

        // the parts of give_ghost_cells_to; public only as nvcc wants GPU loops in public member functions
        void number_new_ghost_cells(const bool* used);
        void order_ghost_cells_by_owner(int n_cells);
        void tell_owners_of_new_ghost_cells(int old_ghost_cells);

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

        // the ranks that hold cells of this rank as ghost seeds, and the ranks whose cells this rank holds, in this
        // build
        std::vector<int> askers;
        std::vector<int> owners;

        // n_cells + ghost_cell of the ghost cells by owner, in the order the values from the owners come
        device::Array<int>      ghost_cell_place_by_owner{"ghosts: places of the ghost cells by owner", 0};
        device::Array<int>      ghost_cell_by_owner{"ghosts: ghost cells by owner", 0};
        device::Array<uint64_t> owner_of_ghost_cell_by_owner{"ghosts: owners of the ghost cells, sorted", 0};

        // what the copies send: the ghost cells by owner, and the cells by asking rank in the order of its ghost cells
        mpi::Blocks             ghost_cells_from_owners;
        mpi::Blocks             cells_to_askers;
        int                     n_cells_sent = 0;
        device::Array<int>      cell_sent{"ghosts: cells sent, by asking rank", 0};
        device::Array<uint64_t> asker_of_cell_sent{"ghosts: asking ranks of the cells sent, sorted", 0};

        // the last give_ghost_cells_to: its new ghost cells by owner, and the new ones of other ranks by asking rank
        mpi::Blocks                 new_from_owners;
        mpi::Blocks                 new_to_askers;
        device::Array<int>          new_by_owner{"ghosts: new ghost cells, in the order by owner", 0};
        device::Array<int>          place_of_new_ghost_cell{"ghosts: places of the new ghost cells by owner", 0};
        device::Array<NewGhostCell> new_asked_here{"ghosts: new ghost cells to tell their owners", 0};
        device::Array<NewGhostCell> new_asked_of_this_rank{"ghosts: new ghost cells of other ranks", 0};
        device::Array<int>          cell_of_new_asked{"ghosts: cells of the new ghost cells of other ranks", 0};

        // the copies of the active cells at a sync
        mpi::Blocks        active_to_askers;
        mpi::Blocks        active_from_owners;
        int                n_active_cells_sent  = 0;
        int                n_active_ghost_cells = 0;
        device::Array<int> active_place_sent{"ghosts: places of the active cells sent", 0};
        device::Array<int> active_cell_sent{"ghosts: active cells sent", 0};
        device::Array<int> active_place_by_owner{"ghosts: places of the active ghost cells by owner", 0};
        device::Array<int> active_ghost_cell{"ghosts: active ghost cells", 0};

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
