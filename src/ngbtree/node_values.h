#pragma once

// how fill_node_values works (included by ngbtree.h)

namespace ngbtree {

    // from the first point of every leaf up to the root: the first child to arrive at a node stops, the second
    // combines both and goes on
    template <typename Value, typename PointValue, typename Combine>
    void fill_node_values(const TreePointers& tree,
                          PointValue          point_value,
                          Combine             combine,
                          int*                arrivals,
                          Value*              value_of_node) {
        const int n = tree.n_points;
        if (n < 2) return; // no internal node
        device::parallel_for(n - 1, [=] HD(long i) { arrivals[i] = 0; });
        device::parallel_for(n, [=] HD(long i) {
            int node = tree.parent_of_point[i];
            if (!is_first_point_of_leaf(tree.nodes[node], (int)i)) return;
            while (node != NO_NODE) {
                if (device::atomic_add(&arrivals[node], 1) == 0) return;
                Value of_child[2];
                for (int h = 0; h < 2; h++) {
                    const int child = tree.nodes[node].child[h];
                    if (child_is_leaf(child)) {
                        const int first = first_point_of_leaf(child);
                        of_child[h]     = point_value(first);
                        for (int t = first + 1; t < first + points_in_leaf(child); t++)
                            of_child[h] = combine(of_child[h], point_value(t));
                    } else {
                        of_child[h] = device::read_written_by_other_thread(&value_of_node[child]);
                    }
                }
                value_of_node[node] = combine(of_child[0], of_child[1]);
                node                = tree.parent_of_node[node];
            }
        });
    }

} // namespace ngbtree
