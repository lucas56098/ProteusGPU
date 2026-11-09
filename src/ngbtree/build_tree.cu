// building the search tree: Morton sort, the internal nodes of Karras (2012), boxes from the leaves up (ngbtree.h)

#include "ngbtree/ngbtree.h"

#include "keys/keys.h"

namespace ngbtree {

    namespace {

        void sort_along_morton_curve(const Vec* points,
                                     int        n,
                                     uint64_t*  keys,
                                     int*       input_index_of_point,
                                     Vec*       positions);
        void link_nodes(int n, const uint64_t* keys, Node* nodes, int* parent_of_node, int* parent_of_point);
        void compute_boxes(int        n,
                           const Vec* positions,
                           Node*      nodes,
                           const int* parent_of_node,
                           const int* parent_of_point,
                           int*       arrivals);

    } // namespace

    void Tree::build(const Vec* points, int n) {
        n_points = n;
        positions.grow_to_at_least(n);
        keys.grow_to_at_least(n);
        input_index_of_point.grow_to_at_least(n);
        parent_of_point.grow_to_at_least(n);
        nodes.grow_to_at_least(n - 1);
        parent_of_node.grow_to_at_least(n - 1);
        arrivals.grow_to_at_least(n - 1);

        sort_along_morton_curve(points, n, keys.data(), input_index_of_point.data(), positions.data());
        if (n < 2) return; // no internal node

        link_nodes(n, keys.data(), nodes.data(), parent_of_node.data(), parent_of_point.data());
        compute_boxes(
            n, positions.data(), nodes.data(), parent_of_node.data(), parent_of_point.data(), arrivals.data());
    }

    void Tree::move_points(const Vec* points) {
        const int* input_index = input_index_of_point.data();
        Vec*       moved       = positions.data();
        device::parallel_for(n_points, [=] HD(long t) { moved[t] = points[input_index[t]]; });
        if (n_points < 2) return; // no internal node

        compute_boxes(
            n_points, positions.data(), nodes.data(), parent_of_node.data(), parent_of_point.data(), arrivals.data());
    }

    TreePointers Tree::pointers() const {
        return TreePointers{n_points, positions.data(), nodes.data(), parent_of_node.data(), parent_of_point.data()};
    }

    const int* Tree::input_indices() const {
        return input_index_of_point.data();
    }

    namespace {

        // the first and last tree index below an internal node
        struct Range {
            int first;
            int last;
        };

        HD Range range_of_node(const uint64_t* keys, int n, int node);
        HD int split_of_range(const uint64_t* keys, int n, Range range);
        HD int common_prefix(const uint64_t* keys, int n, long a, long b);
        HD void box_of_child(const Node* nodes, const Vec* positions, int child, Vec* box_min, Vec* box_max);

        // equal keys keep their input order
        void sort_along_morton_curve(const Vec* points,
                                     int        n,
                                     uint64_t*  keys,
                                     int*       input_index_of_point,
                                     Vec*       positions) {
            device::parallel_for(n, [=] HD(long i) {
                keys[i]                 = keys::morton_key(points[i]);
                input_index_of_point[i] = (int)i;
            });
            device::sort_by_key(n, keys, input_index_of_point);
            device::parallel_for(n, [=] HD(long i) { positions[i] = points[input_index_of_point[i]]; });
        }

        // every internal node finds its range and split; a child that holds one point is that point
        void link_nodes(int n, const uint64_t* keys, Node* nodes, int* parent_of_node, int* parent_of_point) {
            device::parallel_for(n - 1, [=] HD(long i) {
                const int   node  = (int)i;
                const Range range = range_of_node(keys, n, node);
                const int   split = split_of_range(keys, n, range);
                const int   left  = (split == range.first) ? child_of_tree_index(split) : split;
                const int   right = (split + 1 == range.last) ? child_of_tree_index(split + 1) : split + 1;

                nodes[node].child[0] = left;
                nodes[node].child[1] = right;
                for (int h = 0; h < 2; h++) {
                    const int child = nodes[node].child[h];
                    if (child_is_point(child)) {
                        parent_of_point[tree_index_of_child(child)] = node;
                    } else {
                        parent_of_node[child] = node;
                    }
                }
                if (node == ROOT) parent_of_node[ROOT] = NO_NODE;
            });
        }

        // one end of the range is the point with the node's index; the other is found along the side sharing more bits
        HD Range range_of_node(const uint64_t* keys, int n, int node) {
            const int direction =
                (common_prefix(keys, n, node, node + 1) > common_prefix(keys, n, node, node - 1)) ? 1 : -1;
            const int shared_with_outside = common_prefix(keys, n, node, node - direction);

            long max_length = 2;
            while (common_prefix(keys, n, node, node + max_length * direction) > shared_with_outside)
                max_length *= 2;
            long length = 0;
            for (long step = max_length / 2; step >= 1; step /= 2) {
                if (common_prefix(keys, n, node, node + (length + step) * direction) > shared_with_outside)
                    length += step;
            }

            const int other_end = (int)(node + length * direction);
            return (other_end < node) ? Range{other_end, node} : Range{node, other_end};
        }

        // last tree index of the left child: the farthest key sharing more bits with first than the whole range does
        HD int split_of_range(const uint64_t* keys, int n, Range range) {
            const int shared_by_range = common_prefix(keys, n, range.first, range.last);
            int       split           = range.first;
            int       step            = range.last - range.first;
            do {
                step                = (step + 1) / 2;
                const int candidate = split + step;
                if (candidate < range.last && common_prefix(keys, n, range.first, candidate) > shared_by_range) {
                    split = candidate;
                }
            } while (step > 1);
            return split;
        }

        // leading bits the keys of a and b share, then those of a and b for equal keys; -1 if b is outside the list
        HD int common_prefix(const uint64_t* keys, int n, long a, long b) {
            if (b < 0 || b >= n) return -1;
            if (keys[a] != keys[b]) return device::count_leading_zero_bits(keys[a] ^ keys[b]);
            return 64 + device::count_leading_zero_bits((uint32_t)a ^ (uint32_t)b);
        }

        // from every point up to the root: the first child to arrive stops, the second makes both boxes and goes on up
        void compute_boxes(int        n,
                           const Vec* positions,
                           Node*      nodes,
                           const int* parent_of_node,
                           const int* parent_of_point,
                           int*       arrivals) {
            device::parallel_for(n - 1, [=] HD(long i) { arrivals[i] = 0; });
            device::parallel_for(n, [=] HD(long i) {
                int node = parent_of_point[i];
                while (node != NO_NODE) {
                    if (device::atomic_add(&arrivals[node], 1) == 0) return;
                    for (int h = 0; h < 2; h++) {
                        box_of_child(nodes,
                                     positions,
                                     nodes[node].child[h],
                                     &nodes[node].child_box_min[h],
                                     &nodes[node].child_box_max[h]);
                    }
                    node = parent_of_node[node];
                }
            });
        }

        // a point is its own box; a node's box holds those of its children, which another thread may have just written
        HD void box_of_child(const Node* nodes, const Vec* positions, int child, Vec* box_min, Vec* box_max) {
            if (child_is_point(child)) {
                *box_min = positions[tree_index_of_child(child)];
                *box_max = positions[tree_index_of_child(child)];
                return;
            }
            const Node& node  = nodes[child];
            const Vec   min_0 = device::read_written_by_other_thread(&node.child_box_min[0]);
            const Vec   min_1 = device::read_written_by_other_thread(&node.child_box_min[1]);
            const Vec   max_0 = device::read_written_by_other_thread(&node.child_box_max[0]);
            const Vec   max_1 = device::read_written_by_other_thread(&node.child_box_max[1]);
            for (int d = 0; d < config::dim; d++) {
                (*box_min)[d] = (min_1[d] < min_0[d]) ? min_1[d] : min_0[d];
                (*box_max)[d] = (max_1[d] > max_0[d]) ? max_1[d] : max_0[d];
            }
        }

    } // namespace

} // namespace ngbtree
