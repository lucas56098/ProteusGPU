#pragma once

// the searches of ngbtree.h, and how a node stores its children (included by ngbtree.h)

namespace ngbtree {

    constexpr int ROOT    = 0;
    constexpr int NO_NODE = -1;

    // a child is one int: an internal node is its index, the leaf of count points from tree index first on is
    // -(16 first + count), so up to 134 million points
    static_assert(config::tree_leaf_points >= 1 && config::tree_leaf_points < 16, "a leaf count has to fit 4 bits");
    HD inline bool child_is_leaf(int child) {
        return child < 0;
    }
    HD inline int first_point_of_leaf(int child) {
        return (-child) / 16;
    }
    HD inline int points_in_leaf(int child) {
        return (-child) % 16;
    }
    HD inline int child_of_leaf(int first, int count) {
        return -(16 * first + count);
    }

    // every node on a path shares more leading bits than its parent: at most 63 of the key plus 32 of the tree index
    constexpr int MAX_TREE_DEPTH = 96;

    template <int K> struct NearestSoFar;
    template <int K> struct WaitingNodes;
    template <int K>
    HD void offer_points_among_children(const TreePointers& tree, int node, int tree_index, NearestSoFar<K>& nearest);
    template <int K>
    HD int next_node_to_enter(const TreePointers&    tree,
                              int                    node,
                              const Vec&             position,
                              const NearestSoFar<K>& nearest,
                              WaitingNodes<K>&       waiting);
    HD inline void add_children_nearer_on_top(const TreePointers& tree,
                                              int                 node,
                                              const Vec&          position,
                                              int*                waiting,
                                              int&                n_waiting);
    HD inline double squared_distance(const Vec& a, const Vec& b);
    HD inline bool is_first_point_of_leaf(const Node& node, int point);
    HD inline bool comes_before(const TreePointers& tree, int a, double distance_a, int b, double distance_b);

    // from the root down: offers the points at every node, enters the nearer child worth it, the other one waits
    template <int K> HD void nearest_neighbours(const TreePointers& tree, int tree_index, int* neighbour_tree_indices) {
        NearestSoFar<K> nearest;
        WaitingNodes<K> waiting;
        const Vec       position = tree.positions[tree_index];

        int node = (tree.n_points >= 2) ? ROOT : NO_NODE;
        while (node != NO_NODE) {
            offer_points_among_children(tree, node, tree_index, nearest);
            node = next_node_to_enter(tree, node, position, nearest, waiting);
        }
        nearest.write_nearest_first(tree, neighbour_tree_indices);
    }

    // the children still to enter wait on a list of at most one node per level plus one
    template <typename Function>
    HD void for_each_point_in_ball(const TreePointers& tree, const Vec& center, double radius, Function function) {
        const double radius2 = radius * radius;

        if (tree.n_points == 1 && squared_distance(center, tree.positions[0]) <= radius2) function(0);
        if (tree.n_points < 2) return;

        int waiting[MAX_TREE_DEPTH + 1];
        int n_waiting        = 0;
        waiting[n_waiting++] = ROOT;
        while (n_waiting > 0) {
            const Node& current = tree.nodes[waiting[--n_waiting]];
            for (int h = 0; h < 2; h++) {
                const int child = current.child[h];
                if (child_is_leaf(child)) {
                    const int first = first_point_of_leaf(child);
                    for (int t = first; t < first + points_in_leaf(child); t++) {
                        if (squared_distance(center, tree.positions[t]) <= radius2) function(t);
                    }
                } else if (squared_distance_to_box(center, current.child_box_min[h], current.child_box_max[h]) <=
                           radius2) {
                    waiting[n_waiting++] = child;
                }
            }
        }
    }

    // a child waits as 2 node + h; its box sits in node; the list holds one child per level, two for the deepest
    template <typename BoxTest, typename Function>
    HD void walk_nearest_first(const TreePointers& tree,
                               const Vec&          position,
                               BoxTest             box_is_worth_entering,
                               Function            function) {
        if (tree.n_points == 1 && box_is_worth_entering(tree.positions[0], tree.positions[0])) function(0);
        if (tree.n_points < 2) return;

        int waiting[MAX_TREE_DEPTH + 1];
        int n_waiting = 0;
        add_children_nearer_on_top(tree, ROOT, position, waiting, n_waiting);
        while (n_waiting > 0) {
            const int   entry  = waiting[--n_waiting];
            const Node& parent = tree.nodes[entry / 2];
            const int   h      = entry % 2;
            if (!box_is_worth_entering(parent.child_box_min[h], parent.child_box_max[h])) continue;

            const int child = parent.child[h];
            if (child_is_leaf(child)) {
                const int first = first_point_of_leaf(child);
                for (int t = first; t < first + points_in_leaf(child); t++) {
                    if (box_is_worth_entering(tree.positions[t], tree.positions[t])) function(t);
                }
            } else {
                add_children_nearer_on_top(tree, child, position, waiting, n_waiting);
            }
        }
    }

    // the lower child is entered at once, the other waits with its value; the list holds one child per level
    template <typename BoxValue, typename Limit, typename Function>
    HD void walk_lowest_value_first(const TreePointers& tree, BoxValue value_of_box, Limit limit, Function function) {
        if (tree.n_points == 1) function(0);
        if (tree.n_points < 2) return;

        int    waiting[MAX_TREE_DEPTH];
        double value_of_waiting[MAX_TREE_DEPTH];
        int    n_waiting = 0;
        int    node      = ROOT;
        while (node != NO_NODE) {
            const Node& current = tree.nodes[node];
            double      value[2];
            for (int h = 0; h < 2; h++) {
                if (child_is_leaf(current.child[h])) {
                    // a leaf has no value of its own and goes by that of its node
                    if (value_of_box(current.child_box_min[h], current.child_box_max[h], node) < limit()) {
                        const int first = first_point_of_leaf(current.child[h]);
                        for (int t = first; t < first + points_in_leaf(current.child[h]); t++)
                            function(t);
                    }
                    value[h] = INFINITY;
                } else {
                    value[h] = value_of_box(current.child_box_min[h], current.child_box_max[h], current.child[h]);
                }
            }

            const int    lower = (value[1] < value[0]) ? 1 : 0;
            const double below = limit();
            if (value[1 - lower] < below) {
                waiting[n_waiting]          = current.child[1 - lower];
                value_of_waiting[n_waiting] = value[1 - lower];
                n_waiting++;
            }
            node = (value[lower] < below) ? current.child[lower] : NO_NODE;
            while (node == NO_NODE && n_waiting > 0) {
                n_waiting--;
                if (value_of_waiting[n_waiting] < limit()) node = waiting[n_waiting];
            }
        }
    }

    // offers the points in the leaves among the children of node whose box could still hold a kept point, except the
    // point itself
    template <int K>
    HD void offer_points_among_children(const TreePointers& tree, int node, int tree_index, NearestSoFar<K>& nearest) {
        const Node& current  = tree.nodes[node];
        const Vec   position = tree.positions[tree_index];
        for (int h = 0; h < 2; h++) {
            const int child = current.child[h];
            if (!child_is_leaf(child)) continue;
            if (!nearest.could_keep_a_point_at(
                    squared_distance_to_box(position, current.child_box_min[h], current.child_box_max[h]))) {
                continue;
            }
            const int first = first_point_of_leaf(child);
            for (int t = first; t < first + points_in_leaf(child); t++) {
                if (t != tree_index) nearest.offer(tree, t, squared_distance(position, tree.positions[t]));
            }
        }
    }

    // the nearer child worth entering, the other one waits; with none, the last waiting node worth it, or NO_NODE
    template <int K>
    HD int next_node_to_enter(const TreePointers&    tree,
                              int                    node,
                              const Vec&             position,
                              const NearestSoFar<K>& nearest,
                              WaitingNodes<K>&       waiting) {
        const Node& current = tree.nodes[node];
        int         worth[2];
        double      distance_of_worth[2];
        int         n_worth = 0;
        for (int h = 0; h < 2; h++) {
            if (child_is_leaf(current.child[h])) continue;
            const double distance =
                squared_distance_to_box(position, current.child_box_min[h], current.child_box_max[h]);
            if (!nearest.could_keep_a_point_at(distance)) continue;
            worth[n_worth]             = current.child[h];
            distance_of_worth[n_worth] = distance;
            n_worth++;
        }

        if (n_worth == 2) {
            const int nearer = (distance_of_worth[1] < distance_of_worth[0]) ? 1 : 0;
            waiting.add(worth[1 - nearer], distance_of_worth[1 - nearer]);
            return worth[nearer];
        }
        if (n_worth == 1) return worth[0];
        return waiting.take_last_still_worth_entering(nearest);
    }

    // the K nearest points so far, as a heap; entry 0 comes last, so a better point can replace it at once
    template <int K> struct NearestSoFar {
        int    tree_index[K];
        double distance[K]; // squared
        int    count = 0;

        HD bool could_keep_a_point_at(double squared) const { return count < K || squared <= distance[0]; }

        HD void offer(const TreePointers& tree, int candidate, double distance_of_candidate) {
            if (count < K) {
                tree_index[count] = candidate;
                distance[count]   = distance_of_candidate;
                count++;
                if (count == K) make_heap(tree);
            } else if (comes_before(tree, candidate, distance_of_candidate, tree_index[0], distance[0])) {
                move_down(tree, 0, K, candidate, distance_of_candidate);
            }
        }

        // the kept points, nearest first, and -1 behind them; sorts the heap in place
        HD void write_nearest_first(const TreePointers& tree, int* neighbour_tree_indices) {
            if (count < K) make_heap(tree);
            for (int end = count - 1; end > 0; end--) {
                const int    last          = tree_index[end];
                const double distance_last = distance[end];
                tree_index[end]            = tree_index[0];
                distance[end]              = distance[0];
                sink_and_rise(tree, end, last, distance_last);
            }
            for (int i = 0; i < K; i++)
                neighbour_tree_indices[i] = (i < count) ? tree_index[i] : -1;
        }

        // the points so far become a heap, from the last entry with a child up to the top
        HD void make_heap(const TreePointers& tree) {
            for (int i = count / 2 - 1; i >= 0; i--)
                move_down(tree, i, count, tree_index[i], distance[i]);
        }

        // the entry from the end goes in at the top of the first n: the later child moves up all the way down, then
        // the entry rises to its place, as it belongs near the bottom
        HD void sink_and_rise(const TreePointers& tree, int n, int entry, double distance_of_entry) {
            int i = 0;
            while (2 * i + 1 < n) {
                int child = 2 * i + 1;
                if (child + 1 < n &&
                    comes_before(
                        tree, tree_index[child], distance[child], tree_index[child + 1], distance[child + 1])) {
                    child++;
                }
                tree_index[i] = tree_index[child];
                distance[i]   = distance[child];
                i             = child;
            }
            while (i > 0) {
                const int parent = (i - 1) / 2;
                if (!comes_before(tree, tree_index[parent], distance[parent], entry, distance_of_entry)) break;
                tree_index[i] = tree_index[parent];
                distance[i]   = distance[parent];
                i             = parent;
            }
            tree_index[i] = entry;
            distance[i]   = distance_of_entry;
        }

        // the entry goes in at i of the first n: children that come after it move up, and it is written once
        HD void move_down(const TreePointers& tree, int i, int n, int entry, double distance_of_entry) {
            while (2 * i + 1 < n) {
                int child = 2 * i + 1;
                if (child + 1 < n &&
                    comes_before(
                        tree, tree_index[child], distance[child], tree_index[child + 1], distance[child + 1])) {
                    child++;
                }
                if (!comes_before(tree, entry, distance_of_entry, tree_index[child], distance[child])) break;
                tree_index[i] = tree_index[child];
                distance[i]   = distance[child];
                i             = child;
            }
            tree_index[i] = entry;
            distance[i]   = distance_of_entry;
        }
    };

    // the nodes waiting to be entered, each with the squared distance to its box; one per level at most
    template <int K> struct WaitingNodes {
        int    node[MAX_TREE_DEPTH];
        double distance[MAX_TREE_DEPTH];
        int    count = 0;

        HD void add(int waiting_node, double distance_to_box) {
            node[count]     = waiting_node;
            distance[count] = distance_to_box;
            count++;
        }

        HD int take_last_still_worth_entering(const NearestSoFar<K>& nearest) {
            while (count > 0) {
                count--;
                if (nearest.could_keep_a_point_at(distance[count])) return node[count];
            }
            return NO_NODE;
        }
    };

    // at equal distance child 0 is nearer
    HD inline void add_children_nearer_on_top(const TreePointers& tree,
                                              int                 node,
                                              const Vec&          position,
                                              int*                waiting,
                                              int&                n_waiting) {
        const Node&  current    = tree.nodes[node];
        const double distance_0 = squared_distance_to_box(position, current.child_box_min[0], current.child_box_max[0]);
        const double distance_1 = squared_distance_to_box(position, current.child_box_min[1], current.child_box_max[1]);
        const int    nearer     = (distance_1 < distance_0) ? 1 : 0;
        waiting[n_waiting++]    = 2 * node + (1 - nearer);
        waiting[n_waiting++]    = 2 * node + nearer;
    }

    HD inline double squared_distance(const Vec& a, const Vec& b) {
        return norm2(a - b);
    }

    // only the first point of a leaf goes up the tree for its leaf
    HD inline bool is_first_point_of_leaf(const Node& node, int point) {
        for (int h = 0; h < 2; h++) {
            if (child_is_leaf(node.child[h]) && first_point_of_leaf(node.child[h]) == point) return true;
        }
        return false;
    }

    // computed like squared_distance, so after rounding a box is only skipped when it is really farther
    HD inline double squared_distance_to_box(const Vec& position, const Vec& box_min, const Vec& box_max) {
        Vec gap{};
        for (int d = 0; d < config::dim; d++) {
            if (position[d] < box_min[d]) gap[d] = box_min[d] - position[d];
            if (position[d] > box_max[d]) gap[d] = position[d] - box_max[d];
        }
        return norm2(gap);
    }

    HD inline double farthest_squared_distance_to_box(const Vec& position, const Vec& box_min, const Vec& box_max) {
        Vec gap{};
        for (int d = 0; d < config::dim; d++)
            gap[d] = fmax(fabs(position[d] - box_min[d]), fabs(box_max[d] - position[d]));
        return norm2(gap);
    }

    // nearer first; at equal distance the smaller position (x, y, z), so the order does not depend on the tree
    HD inline bool comes_before(const TreePointers& tree, int a, double distance_a, int b, double distance_b) {
        if (distance_a != distance_b) return distance_a < distance_b;
        for (int d = 0; d < config::dim; d++) {
            if (tree.positions[a][d] != tree.positions[b][d]) return tree.positions[a][d] < tree.positions[b][d];
        }
        return a < b;
    }

} // namespace ngbtree
