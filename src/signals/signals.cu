// implements the signal search (signals.h), as Arepo's tree-based timesteps (Springel 2010)

#include "signals/signals.h"

#include "mpi/mpi.h"
#include "timers/timers.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace signals {

    namespace {

        constexpr int THREADS_PER_BLOCK             = 64;
        constexpr int MIN_BLOCKS_PER_MULTIPROCESSOR = 16;

        // margin on the bound of a box, so that rounding never skips a cell that would reach the cell sooner
        constexpr double BOUND_SAFETY = 1e-10;

        constexpr int ROOT_OF_RANK_TREE      = 1;
        constexpr int MAX_DEPTH_OF_RANK_TREE = 64;

        void put_cells_in_tree_order(const ngbtree::TreePointers&  tree,
                                     const int*                    point,
                                     int                           n_cells,
                                     const gas::PrimitivePointers& state,
                                     Vec*                          velocity_in_tree,
                                     double*                       sound_in_tree);
        HD Query query_of_cell(int                           cell,
                               const mesh::MeshPointers&     mesh,
                               const gas::PrimitivePointers& state,
                               double                        shortest);
        HD double shortest_time(const Query&                 query,
                                const ngbtree::TreePointers& tree,
                                const Vec*                   velocity_in_tree,
                                const double*                sound_in_tree,
                                const Bounds*                bounds_of_node);
        void search_other_ranks(const mesh::CellList&         cells,
                                const ngbtree::TreePointers&  tree,
                                const mesh::MeshPointers&     mesh,
                                const gas::PrimitivePointers& state,
                                const Vec*                    velocity_in_tree,
                                const double*                 sound_in_tree,
                                const Bounds*                 bounds_of_node,
                                Query*                        query,
                                BetweenRanks&                 between);
        void fill_tree_over_ranks(const BoundsOfRank& mine, int n_ranks, device::Array<BoundsOfRank>& of_node);
        template <typename Function>
        HD void for_each_rank_that_might_reach(const Query&        query,
                                               const BoundsOfRank* of_node,
                                               int                 n_ranks,
                                               int                 here,
                                               Function            function);
        HD BoundsOfRank both_ranks(const BoundsOfRank& a, const BoundsOfRank& b);
        HD Bounds bounds_of_point(const Vec& velocity, double sound);
        HD Bounds combined(const Bounds& a, const Bounds& b);
        HD double sound_speed_of(const gas::Primitive& w);
        HD double time_at_least(const Query& query, const Vec& box_min, const Vec& box_max, const Bounds& bounds);
        HD double signal_time(const Query& query, const Vec& seed, const Vec& velocity, double sound);

    } // namespace

    void Signals::shorten_cfl_steps(const mesh::CellList&         cells,
                                    const mesh::Mesh&             mesh,
                                    const gas::PrimitivePointers& state,
                                    double                        cfl_fraction,
                                    double                        longest_step_that_matters,
                                    double*                       cfl_step) {
        const mesh::MeshPointers    on_mesh = mesh.pointers();
        const ngbtree::TreePointers tree    = mesh.tree();

        {
            const timers::Scope timer("BOUNDS");
            this->velocity_in_tree.grow_to_at_least(tree.n_points);
            this->sound_in_tree.grow_to_at_least(tree.n_points);
            this->bounds_of_node.grow_to_at_least(tree.n_points);
            arrivals.grow_to_at_least(tree.n_points);
            Vec*    velocity_in_tree = this->velocity_in_tree.data();
            double* sound_in_tree    = this->sound_in_tree.data();
            put_cells_in_tree_order(
                tree, mesh.point_of_tree_index(), on_mesh.n_cells, state, velocity_in_tree, sound_in_tree);
            ngbtree::fill_node_values(
                tree,
                [=] HD(int tree_index) {
                    return bounds_of_point(velocity_in_tree[tree_index], sound_in_tree[tree_index]);
                },
                [=] HD(const Bounds& a, const Bounds& b) { return combined(a, b); },
                arrivals.data(),
                this->bounds_of_node.data());
        }
        const Vec*    velocity_in_tree = this->velocity_in_tree.data();
        const double* sound_in_tree    = this->sound_in_tree.data();
        const Bounds* bounds_of_node   = this->bounds_of_node.data();

        this->query.grow_to_at_least(cells.n);
        Query* query = this->query.data();
        device::parallel_for(cells.n, [=] HD(long i) {
            const int cell = cells[(int)i];
            query[i] =
                query_of_cell(cell, on_mesh, state, fmin(cfl_step[cell], longest_step_that_matters) / cfl_fraction);
        });
        {
            const timers::Scope timer("WALK");
            device::parallel_for<THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR>(cells.n, [=] HD(long i) {
                query[i].shortest = shortest_time(query[i], tree, velocity_in_tree, sound_in_tree, bounds_of_node);
            });
        }
        if (mpi::nranks() > 1) {
            const timers::Scope timer("OTHER_RANKS");
            search_other_ranks(
                cells, tree, on_mesh, state, velocity_in_tree, sound_in_tree, bounds_of_node, query, between_ranks);
        }

        device::parallel_for(cells.n, [=] HD(long i) {
            const int cell = cells[(int)i];
            if (query[i].shortest < fmin(cfl_step[cell], longest_step_that_matters) / cfl_fraction)
                cfl_step[cell] = cfl_fraction * query[i].shortest;
        });
    }

    namespace {

        // a ghost seed is a cell of another rank, or a periodic copy; it reaches no cell here and bounds nothing
        void put_cells_in_tree_order(const ngbtree::TreePointers&  tree,
                                     const int*                    point,
                                     int                           n_cells,
                                     const gas::PrimitivePointers& state,
                                     Vec*                          velocity_in_tree,
                                     double*                       sound_in_tree) {
            device::parallel_for(tree.n_points, [=] HD(long t) {
                const int cell = point[t];
                if (cell >= n_cells) {
                    velocity_in_tree[t] = Vec{};
                    sound_in_tree[t]    = -1.0;
                    return;
                }
                const gas::Primitive w = gas::load(state, cell);
                velocity_in_tree[t]    = w.velocity;
                sound_in_tree[t]       = sound_speed_of(w);
            });
        }

        HD Query query_of_cell(int                           cell,
                               const mesh::MeshPointers&     mesh,
                               const gas::PrimitivePointers& state,
                               double                        shortest) {
            const gas::Primitive w = gas::load(state, cell);
            return Query{mesh.seed[cell], w.velocity, sound_speed_of(w), radius_of_ball(mesh.volume[cell]), shortest};
        }

        // into every box that might reach the cell sooner than found so far, the soonest box first
        HD double shortest_time(const Query&                 query,
                                const ngbtree::TreePointers& tree,
                                const Vec*                   velocity_in_tree,
                                const double*                sound_in_tree,
                                const Bounds*                bounds_of_node) {
            double     shortest    = query.shortest;
            const auto time_of_box = [&](const Vec& box_min, const Vec& box_max, int node) {
                return time_at_least(query, box_min, box_max, bounds_of_node[node]);
            };
            const auto offer = [&](int tree_index) {
                const double sound = sound_in_tree[tree_index];
                if (sound < 0.0) return;
                shortest =
                    fmin(shortest, signal_time(query, tree.positions[tree_index], velocity_in_tree[tree_index], sound));
            };
            ngbtree::walk_lowest_value_first(tree, time_of_box, [&] { return shortest; }, offer);
            return shortest;
        }

        // queries go to the ranks that might reach a cell sooner and come back with the shortest time; collective
        void search_other_ranks(const mesh::CellList&         cells,
                                const ngbtree::TreePointers&  tree,
                                const mesh::MeshPointers&     mesh,
                                const gas::PrimitivePointers& state,
                                const Vec*                    velocity_in_tree,
                                const double*                 sound_in_tree,
                                const Bounds*                 bounds_of_node,
                                Query*                        query,
                                BetweenRanks&                 between) {
            const int n_ranks = mpi::nranks();
            const int here    = mpi::rank();

            // the bounds of every rank
            const Vec*   seed  = mesh.seed;
            BoundsOfRank empty = {Vec{}, Vec{}, Bounds{-1.0, Vec{}, Vec{}}};
            for (int d = 0; d < config::dim; d++) {
                empty.box_min[d] = INFINITY;
                empty.box_max[d] = -INFINITY;
            }
            const BoundsOfRank mine = device::reduce(
                mesh.n_cells,
                empty,
                [=] HD(long i) {
                    const gas::Primitive w = gas::load(state, (int)i);
                    return BoundsOfRank{seed[i], seed[i], bounds_of_point(w.velocity, sound_speed_of(w))};
                },
                [=] HD(const BoundsOfRank& a, const BoundsOfRank& b) { return both_ranks(a, b); });
            fill_tree_over_ranks(mine, n_ranks, between.of_node);
            const BoundsOfRank* of_node = between.of_node.data();

            // the pairs of a cell of the list and a rank; sent in the order of the ranks
            between.n_pairs_of_entry.grow_to_at_least(cells.n);
            between.first_pair.grow_to_at_least(cells.n);
            int* n_pairs_of_entry = between.n_pairs_of_entry.data();
            device::parallel_for(cells.n, [=] HD(long i) {
                int n = 0;
                for_each_rank_that_might_reach(query[i], of_node, n_ranks, here, [&](int) { n++; });
                n_pairs_of_entry[i] = n;
            });
            const int n_pairs = device::offsets_from_counts(cells.n, n_pairs_of_entry, between.first_pair.data());
            between.entry_of_pair.grow_to_at_least(n_pairs);
            between.key_of_pair.grow_to_at_least(n_pairs);
            between.pair_in_order.grow_to_at_least(n_pairs);
            between.time_of_pair.grow_to_at_least(n_pairs);
            int*       entry_of_pair = between.entry_of_pair.data();
            uint64_t*  key_of_pair   = between.key_of_pair.data();
            int*       pair_in_order = between.pair_in_order.data();
            const int* first_pair    = between.first_pair.data();
            device::parallel_for(cells.n, [=] HD(long i) {
                int pair = first_pair[i];
                for_each_rank_that_might_reach(query[i], of_node, n_ranks, here, [&](int rank) {
                    entry_of_pair[pair] = (int)i;
                    key_of_pair[pair]   = (uint64_t)rank * (uint64_t)n_pairs + (uint64_t)pair;
                    pair_in_order[pair] = pair;
                    pair++;
                });
            });
            device::sort_by_key(n_pairs, key_of_pair, pair_in_order);

            // the queries out, the searches, and back
            between.send_count.grow_to_at_least(n_ranks);
            between.sent.grow_to_at_least(n_pairs);
            int* send_count = between.send_count.data();
            for (int rank = 0; rank < n_ranks; rank++)
                send_count[rank] = 0;
            Query* sent = between.sent.data();
            device::parallel_for(n_pairs, [=] HD(long k) {
                sent[k] = query[entry_of_pair[pair_in_order[k]]];
                device::atomic_add(&send_count[key_of_pair[k] / (uint64_t)n_pairs], 1);
            });
            const mpi::Blocks to = mpi::blocks_of_counts(send_count, n_ranks);
            mpi::Blocks       from;
            mpi::counts_from_any_rank(to, 0, from);
            between.received.grow_to_at_least(from.total);
            Query* received = between.received.data();
            device::exchange_items(sent, to, received, from, sizeof(Query));
            device::parallel_for<THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR>(from.total, [=] HD(long k) {
                received[k].shortest =
                    shortest_time(received[k], tree, velocity_in_tree, sound_in_tree, bounds_of_node);
            });
            device::exchange_items(received, from, sent, to, sizeof(Query));

            // the shortest time of every cell of the list over its pairs
            double* time_of_pair = between.time_of_pair.data();
            device::parallel_for(n_pairs, [=] HD(long k) { time_of_pair[pair_in_order[k]] = sent[k].shortest; });
            device::parallel_for(cells.n, [=] HD(long i) {
                for (int pair = first_pair[i]; pair < first_pair[i] + n_pairs_of_entry[i]; pair++)
                    query[i].shortest = fmin(query[i].shortest, time_of_pair[pair]);
            });
        }

        // on every rank the same numbers, from the ranks up to the root; ranks next to each other are neighbours on the
        // Hilbert curve, so the nodes stay small; collective
        void fill_tree_over_ranks(const BoundsOfRank& mine, int n_ranks, device::Array<BoundsOfRank>& of_node) {
            std::vector<BoundsOfRank> nodes(2 * n_ranks);
            mpi::gather_one_item_from_all_ranks(&mine, (int)sizeof(BoundsOfRank), nodes.data() + n_ranks);
            for (int node = n_ranks - 1; node >= ROOT_OF_RANK_TREE; node--)
                nodes[node] = both_ranks(nodes[2 * node], nodes[2 * node + 1]);

            of_node.grow_to_at_least(2 * n_ranks);
            std::memcpy(of_node.data(), nodes.data(), sizeof(BoundsOfRank) * 2 * n_ranks);
        }

        // a node holds its ranks, so no rank below a node that cannot reach the cell sooner can reach it sooner either
        template <typename Function>
        HD void for_each_rank_that_might_reach(const Query&        query,
                                               const BoundsOfRank* of_node,
                                               int                 n_ranks,
                                               int                 here,
                                               Function            function) {
            int waiting[MAX_DEPTH_OF_RANK_TREE];
            int n_waiting        = 0;
            waiting[n_waiting++] = ROOT_OF_RANK_TREE;
            while (n_waiting > 0) {
                const int           node = waiting[--n_waiting];
                const BoundsOfRank& r    = of_node[node];
                if (!(time_at_least(query, r.box_min, r.box_max, r.bounds) < query.shortest)) continue;
                if (node >= n_ranks) {
                    if (node - n_ranks != here) function(node - n_ranks);
                    continue;
                }
                waiting[n_waiting++] = 2 * node + 1;
                waiting[n_waiting++] = 2 * node;
            }
        }

        HD BoundsOfRank both_ranks(const BoundsOfRank& a, const BoundsOfRank& b) {
            BoundsOfRank both{a.box_min, a.box_max, combined(a.bounds, b.bounds)};
            for (int d = 0; d < config::dim; d++) {
                both.box_min[d] = fmin(a.box_min[d], b.box_min[d]);
                both.box_max[d] = fmax(a.box_max[d], b.box_max[d]);
            }
            return both;
        }

        HD Bounds bounds_of_point(const Vec& velocity, double sound) {
            return Bounds{sound, velocity, velocity};
        }

        HD Bounds combined(const Bounds& a, const Bounds& b) {
            if (a.sound < 0.0) return b;
            if (b.sound < 0.0) return a;
            Bounds both{fmax(a.sound, b.sound), a.slowest, a.fastest};
            for (int d = 0; d < config::dim; d++) {
                both.slowest[d] = fmin(a.slowest[d], b.slowest[d]);
                both.fastest[d] = fmax(a.fastest[d], b.fastest[d]);
            }
            return both;
        }

        HD double sound_speed_of(const gas::Primitive& w) {
            return (w.density > 0.0 && w.pressure > 0.0) ? gas::sound_speed(w) : 0.0;
        }

        // never longer than the time of any cell in the box: the nearest offsets over the fastest possible approach
        HD double time_at_least(const Query& query, const Vec& box_min, const Vec& box_max, const Bounds& bounds) {
            if (bounds.sound < 0.0) return INFINITY;
            double squared_distance = 0.0, squared_towards = 0.0, away = 0.0;
            for (int d = 0; d < config::dim; d++) {
                // the offsets to the box, wrapped as signal_time wraps them; no shift if all lie in [-0.5, 0.5]
                const double low = box_min[d] - query.position[d], high = box_max[d] - query.position[d];
                double       gap   = INFINITY;
                bool         below = false, above = false;
                if (low >= -0.5 && high <= 0.5) {
                    gap   = fmax(fmax(low, -high), 0.0);
                    below = low < 0.0;
                    above = high > 0.0;
                } else {
                    for (int shift = -1; shift <= 1; shift++) {
                        const double from = fmax(low + shift, -0.5), to = fmin(high + shift, 0.5);
                        if (from > to) continue;
                        gap = fmin(gap, fmax(fmax(from, -to), 0.0));
                        below |= from < 0.0;
                        above |= to > 0.0;
                    }
                }
                squared_distance += gap * gap;

                const double from_below = bounds.fastest[d] - query.velocity[d];
                const double from_above = query.velocity[d] - bounds.slowest[d];
                const double towards    = fmax(fmax(below ? from_below : 0.0, above ? from_above : 0.0), 0.0);
                squared_towards += towards * towards;
                away += fmax(fabs(from_below), fabs(from_above));
            }
            const double speed = query.sound + bounds.sound + sqrt(squared_towards) + BOUND_SAFETY * away;
            if (!(speed > 0.0)) return INFINITY;
            return (1.0 - BOUND_SAFETY) * (sqrt(squared_distance) + query.radius) / speed;
        }

        // no time where the two part faster than their sound speeds
        HD double signal_time(const Query& query, const Vec& seed, const Vec& velocity, double sound) {
            const Vec    to_other = periodic_difference(seed, query.position);
            const double distance = sqrt(norm2(to_other));
            if (distance == 0.0) return INFINITY;
            const double speed = query.sound + sound - dot(velocity - query.velocity, to_other) / distance;
            if (!(speed > 0.0)) return INFINITY;
            return (distance + query.radius) / speed;
        }

    } // namespace

} // namespace signals
