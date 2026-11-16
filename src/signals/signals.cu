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

        // a ball rarely reaches many ranks; remembering a few keeps most ranks from getting the same query twice
        constexpr int RANKS_REMEMBERED = 16;

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
                                const domain::DomainPointers& domain,
                                double                        drift_margin,
                                BetweenRanks&                 between);
        Bounds gather_bounds_of_ranks(const Bounds& mine, int n_ranks, device::Array<Bounds>& of_rank);
        template <typename Function>
        HD void for_each_rank_that_might_reach(const Query&                  query,
                                               const domain::DomainPointers& domain,
                                               const Bounds*                 of_rank,
                                               const Bounds&                 of_all,
                                               double                        drift_margin,
                                               Function                      function);
        HD Bounds bounds_of_point(const Vec& velocity, double sound);
        HD Bounds combined(const Bounds& a, const Bounds& b);
        HD double sound_speed_of(const gas::Primitive& w);
        HD double time_at_least(const Query& query, const Vec& box_min, const Vec& box_max, const Bounds& bounds);
        HD double time_at_least_from_beyond(const Query& query, double distance, const Bounds& bounds);
        HD double signal_time(const Query& query, const Vec& seed, const Vec& velocity, double sound);

    } // namespace

    void Signals::shorten_cfl_steps(const mesh::CellList&         cells,
                                    const mesh::Mesh&             mesh,
                                    const gas::PrimitivePointers& state,
                                    double                        cfl_fraction,
                                    double                        longest_step_that_matters,
                                    const domain::Domain&         domain,
                                    double                        drift_margin,
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
            search_other_ranks(cells,
                               tree,
                               on_mesh,
                               state,
                               velocity_in_tree,
                               sound_in_tree,
                               bounds_of_node,
                               query,
                               domain.pointers(),
                               drift_margin,
                               between_ranks);
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
                                const domain::DomainPointers& domain,
                                double                        drift_margin,
                                BetweenRanks&                 between) {
            const int n_ranks = mpi::nranks();

            // the speeds of every rank; its cells lie in its part of the curve, widened by the drift margin
            const Bounds mine = device::reduce(
                mesh.n_cells,
                Bounds{-1.0, Vec{}, Vec{}},
                [=] HD(long i) {
                    const gas::Primitive w = gas::load(state, (int)i);
                    return bounds_of_point(w.velocity, sound_speed_of(w));
                },
                [=] HD(const Bounds& a, const Bounds& b) { return combined(a, b); });
            const Bounds  of_all  = gather_bounds_of_ranks(mine, n_ranks, between.of_rank);
            const Bounds* of_rank = between.of_rank.data();

            // the pairs of a cell of the list and a rank; sent in the order of the ranks
            between.n_pairs_of_entry.grow_to_at_least(cells.n);
            between.first_pair.grow_to_at_least(cells.n);
            int* n_pairs_of_entry = between.n_pairs_of_entry.data();
            device::parallel_for(cells.n, [=] HD(long i) {
                int n = 0;
                for_each_rank_that_might_reach(query[i], domain, of_rank, of_all, drift_margin, [&](int) { n++; });
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
                if (n_pairs_of_entry[i] == 0) return;
                int pair = first_pair[i];
                for_each_rank_that_might_reach(query[i], domain, of_rank, of_all, drift_margin, [&](int rank) {
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

        // the same numbers on every rank; returns the bounds of all of them; collective
        Bounds gather_bounds_of_ranks(const Bounds& mine, int n_ranks, device::Array<Bounds>& of_rank) {
            std::vector<Bounds> all(n_ranks);
            mpi::gather_one_item_from_all_ranks(&mine, (int)sizeof(Bounds), all.data());
            of_rank.grow_to_at_least(n_ranks);
            Bounds of_all{-1.0, Vec{}, Vec{}};
            for (int r = 0; r < n_ranks; r++) {
                of_rank[r] = all[r];
                of_all     = combined(of_all, all[r]);
            }
            return of_all;
        }

        // the cubes of the Hilbert curve from the whole box down, as domain::for_each_rank_touched_by_ball: a cube of
        // one other rank gives the rank if its cells might reach the cell sooner, a cube of several ranks splits if
        // any cell might; every cube widened by the drift margin, as a seed may have left the part of its rank. Most
        // cubes are too far for a cell of any rank, which is tested first, as it needs no keys. A cell deep inside a
        // cube of this rank needs no walk: every cube of another rank lies outside that cube, so at least the room in
        // it less the drift margin away
        template <typename Function>
        HD void for_each_rank_that_might_reach(const Query&                  query,
                                               const domain::DomainPointers& domain,
                                               const Bounds*                 of_rank,
                                               const Bounds&                 of_all,
                                               double                        drift_margin,
                                               Function                      function) {
            const double room = domain::room_in_own_cube(domain, query.position) - drift_margin;
            if (room > 0.0 && !(time_at_least_from_beyond(query, room, of_all) < query.shortest)) return;

            domain::Cube waiting[domain::MAX_WAITING_CUBES];
            int          n_waiting = 0;
            int          remembered[RANKS_REMEMBERED];
            int          n_remembered = 0;
            waiting[n_waiting++]      = domain::Cube{0, {}};
            while (n_waiting > 0) {
                const domain::Cube cube = waiting[--n_waiting];
                const double       side = 1.0 / (double)(1ull << cube.level);
                Vec                box_min{}, box_max{};
                for (int d = 0; d < config::dim; d++) {
                    box_min[d] = cube.corner[d] * side - drift_margin;
                    box_max[d] = (cube.corner[d] + 1) * side + drift_margin;
                }
                if (!(time_at_least(query, box_min, box_max, of_all) < query.shortest)) continue;
                uint64_t first, last;
                keys::hilbert_keys_of_cube(cube.level, cube.corner, &first, &last);
                const int owner = domain::owner_of_key(domain, first);
                if (owner == domain::owner_of_key(domain, last)) {
                    if (owner == domain.this_rank) continue;
                    if (!(time_at_least(query, box_min, box_max, of_rank[owner]) < query.shortest)) continue;
                    bool known = false;
                    for (int i = 0; i < n_remembered; i++)
                        known = known || remembered[i] == owner;
                    if (known) continue;
                    if (n_remembered < RANKS_REMEMBERED) remembered[n_remembered++] = owner;
                    function(owner);
                    continue;
                }
                for (int child = 0; child < domain::CHILDREN_OF_CUBE; child++)
                    waiting[n_waiting++] = domain::child_of(cube, child);
            }
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

        // never longer than time_at_least for any box at least distance away: the distance is the nearest it can be,
        // and every approach the fastest of the two sides
        HD double time_at_least_from_beyond(const Query& query, double distance, const Bounds& bounds) {
            if (bounds.sound < 0.0) return INFINITY;
            double squared_towards = 0.0, away = 0.0;
            for (int d = 0; d < config::dim; d++) {
                const double from_below = bounds.fastest[d] - query.velocity[d];
                const double from_above = query.velocity[d] - bounds.slowest[d];
                const double towards    = fmax(fmax(from_below, from_above), 0.0);
                squared_towards += towards * towards;
                away += fmax(fabs(from_below), fabs(from_above));
            }
            const double speed = query.sound + bounds.sound + sqrt(squared_towards) + BOUND_SAFETY * away;
            if (!(speed > 0.0)) return INFINITY;
            return (1.0 - BOUND_SAFETY) * ((1.0 - BOUND_SAFETY) * distance + query.radius) / speed;
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
