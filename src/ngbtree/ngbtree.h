#pragma once

// Search tree over points: nearest neighbours, every point in a ball, and walks the caller steers box by box.

#include "device/device.h"
#include "math/math.h"

namespace ngbtree {

    // one internal node: its two children, each an internal node or a leaf of up to config::tree_leaf_points points,
    // and the box around each child
    struct Node {
        Vec child_box_min[2];
        Vec child_box_max[2];
        int child[2];
    };

    struct TreePointers {
        int         n_points;
        const Vec*  positions; // by tree index
        const Node* nodes;
        const int*  parent_of_node; // NO_NODE for the root
        const int*  parent_of_point; // by tree index
    };

    // the points sorted along the Morton curve, and the binary tree (Karras 2012) over them
    // tree index: place of a point in that list; input index: its place in the points given to build
    class Tree {
      public:
        // builds the tree over the n points, in [-0.5, 1.5) on every axis; the memory is reused by the next build
        void build(const Vec* points, int n);

        // the points of the last build at new positions: the tree keeps its order, only its boxes are made anew
        void move_points(const Vec* points);

        TreePointers pointers() const;

        const int* input_indices() const;

      private:
        int                     n_points = 0;
        device::Array<Vec>      positions{"ngbtree: positions", 0};
        device::Array<uint64_t> keys{"ngbtree: Morton keys", 0};
        device::Array<int>      input_index_of_point{"ngbtree: input indices", 0};
        device::Array<Node>     nodes{"ngbtree: nodes", 0};
        device::Array<int>      parent_of_node{"ngbtree: parents of nodes", 0};
        device::Array<int>      parent_of_point{"ngbtree: parents of points", 0};
        device::Array<int>      arrivals{"ngbtree: arrivals", 0}; // children that reached a node while its box was made
    };

    // the K nearest neighbours of a point as tree indices, without itself, nearest first; -1 where there are fewer
    template <int K> HD void nearest_neighbours(const TreePointers& tree, int tree_index, int* neighbour_tree_indices);

    // calls function(tree_index) for every point at most radius from center
    template <typename Function>
    HD void for_each_point_in_ball(const TreePointers& tree, const Vec& center, double radius, Function function);

    // depth first, nearer child first; a box is tested when its turn comes; function(tree_index) on the points reached
    template <typename BoxTest, typename Function>
    HD void walk_nearest_first(const TreePointers& tree,
                               const Vec&          position,
                               BoxTest             box_is_worth_entering,
                               Function            function);

    // depth first, lower value first; enters a child while its value is below limit(), which may fall during the walk;
    // a leaf goes by the value of its node
    template <typename BoxValue, typename Limit, typename Function>
    HD void walk_lowest_value_first(const TreePointers& tree, BoxValue value_of_box, Limit limit, Function function);

    // value_of_node[n] = combine(child 0, child 1), leaves first; Value is made of doubles; arrivals: n_points - 1 ints
    template <typename Value, typename PointValue, typename Combine>
    void fill_node_values(const TreePointers& tree,
                          PointValue          point_value,
                          Combine             combine,
                          int*                arrivals,
                          Value*              value_of_node);

    // squared distance to the nearest point of the box; after rounding never larger than to a point in the box
    HD inline double squared_distance_to_box(const Vec& position, const Vec& box_min, const Vec& box_max);

    // squared distance to the farthest point of the box; rounded, so a caller keeps a margin
    HD inline double farthest_squared_distance_to_box(const Vec& position, const Vec& box_min, const Vec& box_max);

} // namespace ngbtree

#include "search.h"

// uses what search.h defines
#include "node_values.h"
