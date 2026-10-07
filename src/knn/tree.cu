// the tree over the sorted points: its build and its searches on the host (included by knn.cu)

namespace knn {

    HD inline int count_leading_zeros(uint64_t x) {
#if defined(__CUDA_ARCH__)
        return __clzll((long long)x);
#else
        return __builtin_clzll(x);
#endif
    }

    HD inline int count_leading_zeros(unsigned int x) {
#if defined(__CUDA_ARCH__)
        return __clz((int)x);
#else
        return __builtin_clz(x);
#endif
    }

    // length of the common prefix of the keys of sorted points i and j, -1 outside the list;
    // equal keys go on with the indices, so no two points look the same
    HD inline int common_prefix(const uint64_t* keys, int n, int i, int j) {
        if (j < 0 || j >= n) return -1;
        const uint64_t a = keys[i];
        const uint64_t b = keys[j];
        if (a != b) return count_leading_zeros(a ^ b);
        return 64 + count_leading_zeros((unsigned int)i ^ (unsigned int)j);
    }

    // counts the children that reached a node, the first gets 0; what it wrote before, the second one sees
    HD inline unsigned int arrive_at_node(unsigned int* counter) {
#if defined(__CUDA_ARCH__)
        __threadfence();
        const unsigned int old = atomicAdd(counter, 1u);
        __threadfence();
        return old;
#else
        return __atomic_fetch_add(counter, 1u, __ATOMIC_ACQ_REL);
#endif
    }

    HD inline double min_of(double a, double b) {
        return (b < a) ? b : a;
    }
    HD inline double max_of(double a, double b) {
        return (b > a) ? b : a;
    }

    // box of a child of node; an internal child is read past the cache, another thread has just written it
    HD inline void
    child_box(const TreeNode* nodes, int c, int leaf0, const POINT_TYPE* pts, POINT_TYPE* lo, POINT_TYPE* hi) {
        if (c >= leaf0) {
            *lo = pts[c - leaf0];
            *hi = *lo;
            return;
        }
        const volatile double* l = (const volatile double*)nodes[c].lo;
        const volatile double* h = (const volatile double*)nodes[c].hi;
        lo->x                    = min_of(l[0], l[DIMENSION]);
        lo->y                    = min_of(l[1], l[DIMENSION + 1]);
        hi->x                    = max_of(h[0], h[DIMENSION]);
        hi->y                    = max_of(h[1], h[DIMENSION + 1]);
#ifdef dim_3D
        lo->z = min_of(l[2], l[DIMENSION + 2]);
        hi->z = max_of(h[2], h[DIMENSION + 2]);
#endif
    }

    // radix tree over the sorted keys (Karras 2012), then the child boxes from the leaves up
    static void build_tree(knn_problem* knn) {
        const int n = knn->len_pts;
        if (n < 2) {
            if (n == 1) knn->d_parent[0] = -1;
            return;
        }

        const uint64_t*   keys   = knn->d_keys;
        const POINT_TYPE* pts    = knn->d_stored_points;
        TreeNode*         nodes  = knn->d_nodes;
        int*              parent = knn->d_parent;
        unsigned int*     visits = knn->d_visits;
        const int         leaf0  = n - 1;

        // each internal node finds its range of sorted points and where it splits
        parallel_for<_KNN_BLOCK_SIZE_>("TREE_NODES", n - 1, [=] HD(int i) {
            const int d    = (common_prefix(keys, n, i, i + 1) > common_prefix(keys, n, i, i - 1)) ? 1 : -1;
            const int dmin = common_prefix(keys, n, i, i - d);

            int lmax = 2;
            while (common_prefix(keys, n, i, i + lmax * d) > dmin)
                lmax <<= 1;
            int l = 0;
            for (int t = lmax >> 1; t >= 1; t >>= 1) {
                if (common_prefix(keys, n, i, i + (l + t) * d) > dmin) l += t;
            }
            const int j     = i + l * d;
            const int first = imin(i, j);
            const int last  = imax(i, j);

            const int dnode = common_prefix(keys, n, first, last);
            int       split = first;
            int       step  = last - first;
            do {
                step         = (step + 1) >> 1;
                const int ns = split + step;
                if (ns < last && common_prefix(keys, n, first, ns) > dnode) split = ns;
            } while (step > 1);

            const int lc      = (split == first) ? leaf0 + split : split;
            const int rc      = (split + 1 == last) ? leaf0 + split + 1 : split + 1;
            nodes[i].child[0] = lc;
            nodes[i].child[1] = rc;
            parent[lc]        = i;
            parent[rc]        = i;
            if (i == 0) parent[0] = -1;
        });

        // the second child to reach a node writes both child boxes, then goes on up
        gpu_memset(visits, 0, (size_t)(n - 1) * sizeof(unsigned int));
        parallel_for<_KNN_BLOCK_SIZE_>("TREE_BOXES", n, [=] HD(int s) {
            int node = parent[leaf0 + s];
            while (node >= 0) {
                if (arrive_at_node(visits + node) == 0) return;
                for (int h = 0; h < 2; h++) {
                    POINT_TYPE lo, hi;
                    child_box(nodes, nodes[node].child[h], leaf0, pts, &lo, &hi);
                    nodes[node].lo[h] = lo;
                    nodes[node].hi[h] = hi;
                }
                node = parent[node];
            }
        });
    }

    // nearest max_k points, sorted
    std::vector<std::pair<double, int>> nearest_on_host(const knn_problem* knn, int sid, int max_k) {
        std::vector<std::pair<double, int>> best; // max heap, front is the last of them
        const int                           n = knn->len_pts;
        if (n < 2 || max_k <= 0) return best;

        const TreeNode*  nodes = knn->d_nodes;
        const int        leaf0 = n - 1;
        const POINT_TYPE p     = knn->d_stored_points[sid];

        auto worth = [&](double d) { return (int)best.size() < max_k || d <= best.front().first * TREE_PRUNE_SLACK; };
        auto offer = [&](int s, const POINT_TYPE& q) {
            if (s == sid) return;
            const std::pair<double, int> c(dist2_point(p, q), s);
            if ((int)best.size() < max_k) {
                best.push_back(c);
                std::push_heap(best.begin(), best.end());
            } else if (c < best.front()) {
                std::pop_heap(best.begin(), best.end());
                best.back() = c;
                std::push_heap(best.begin(), best.end());
            }
        };

        // pending internal nodes with the distance to their box, the nearer child comes off first
        std::vector<std::pair<int, double>> stack(1, std::make_pair(0, 0.0));
        while (!stack.empty()) {
            const int    node = stack.back().first;
            const double dn   = stack.back().second;
            stack.pop_back();
            if (!worth(dn)) continue;
            const TreeNode& nd   = nodes[node];
            double          d[2] = {-1.0, -1.0};
            for (int h = 0; h < 2; h++) {
                if (nd.child[h] >= leaf0) {
                    offer(nd.child[h] - leaf0, nd.lo[h]);
                } else {
                    d[h] = dist2_box(nd.lo[h], nd.hi[h], p);
                }
            }
            const int near = (d[1] >= 0.0 && (d[0] < 0.0 || d[1] < d[0])) ? 1 : 0;
            if (d[1 - near] >= 0.0) stack.push_back(std::make_pair(nd.child[1 - near], d[1 - near]));
            if (d[near] >= 0.0) stack.push_back(std::make_pair(nd.child[near], d[near]));
        }

        std::sort_heap(best.begin(), best.end());
        return best;
    }

} // namespace knn
