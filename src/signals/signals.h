#pragma once

// Shortest signal time to every cell, and the CFL steps it shortens (Springel 2010).

#include "gas/gas.h"
#include "mesh/mesh.h"

namespace signals {

    // the cells below a node of the tree: largest sound speed and velocity range per axis; sound below 0 for no cell
    struct Bounds {
        double sound;
        Vec    slowest;
        Vec    fastest;
    };

    struct BoundsOfRank {
        Vec    box_min;
        Vec    box_max;
        Bounds bounds;
    };

    // a cell whose signal is searched for, and the shortest time found so far
    struct Query {
        Vec    position;
        Vec    velocity;
        double sound;
        double radius;
        double shortest;
    };

    // with several ranks: a pair is a cell of the list and a rank it is searched on; a tree over the ranks finds the
    // pairs: node i has the children 2 i and 2 i + 1, and node n_ranks + r is rank r
    struct BetweenRanks {
        device::Array<BoundsOfRank> of_node{"signals: bounds of the ranks below every node", 0};
        device::Array<int>          n_pairs_of_entry{"signals: ranks of every cell of the list", 0};
        device::Array<int>          first_pair{"signals: first pair of every cell of the list", 0};
        device::Array<int>          entry_of_pair{"signals: cell of the list of every pair", 0};
        device::Array<uint64_t>     key_of_pair{"signals: rank, then pair", 0};
        device::Array<int>          pair_in_order{"signals: the pairs by rank", 0};
        device::Array<double>       time_of_pair{"signals: the time found for every pair", 0};
        device::Array<int>          send_count{"signals: queries for every rank", 0};
        device::Array<Query>        sent{"signals: queries sent", 0};
        device::Array<Query>        received{"signals: queries received", 0};
    };

    // the work of a search, kept for the next one
    class Signals {
      public:
        // limits the CFL steps of the list to cfl_fraction times the shortest signal time from any other cell
        // time from cell j: (d + R) / (c + c_j - (v_j - v) . d / |d|); same bits on any number of ranks; collective
        void shorten_cfl_steps(const mesh::CellList&         cells,
                               const mesh::Mesh&             mesh,
                               const gas::PrimitivePointers& state,
                               double                        cfl_fraction,
                               double                        longest_step_that_matters,
                               double*                       cfl_step);

      private:
        device::Array<Vec>    velocity_in_tree{"signals: velocity of every point of the tree", 0};
        device::Array<double> sound_in_tree{"signals: sound speed of every point of the tree, below 0 for a ghost seed",
                                            0};
        device::Array<Bounds> bounds_of_node{"signals: bounds below the nodes of the tree", 0};
        device::Array<int>    arrivals{"signals: arrivals at the nodes", 0};
        device::Array<Query>  query{"signals: the cells of the list", 0};
        BetweenRanks          between_ranks;
    };

} // namespace signals
