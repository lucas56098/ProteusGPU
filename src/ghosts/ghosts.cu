// implements the ghosts: the balls go to the owners as queries, and the cells inside them come back (ghosts.h)

#include "ghosts/ghosts.h"

#include "mpi/mpi.h"

#include <algorithm>
#include <utility>

namespace ghosts {

    namespace {

        // shifts are the 3^dim vectors with components -1, 0 and 1, stored as (s[0] + 1) + 3 (s[1] + 1) + 9 (s[2] + 1)
        constexpr int N_SHIFTS = (config::dim == 2) ? 9 : 27;
        constexpr int NO_SHIFT = (N_SHIFTS - 1) / 2;

        // a ball rarely touches many ranks; remembering a few keeps most ranks from getting the same query twice
        constexpr int RANKS_REMEMBERED = 16;

        // a hit as one number: the asking rank, then the cell, then the shift, so that sorting groups the hits by asker
        constexpr int SHIFT_BITS = 5;
        constexpr int CELL_BITS  = 31;
        static_assert(N_SHIFTS <= (1 << SHIFT_BITS), "every shift fits into its bits");

        void write_queries_by_rank(const Vec*                    center,
                                   const double*                 radius,
                                   int                           n_balls,
                                   double                        drift_margin,
                                   const domain::DomainPointers& domain,
                                   ItemsByRank<Query>&           queries,
                                   device::Array<int>&           next_place);
        template <typename Item> void send_to_their_ranks(const ItemsByRank<Item>& out, ItemsByRank<Item>& in);
        void find_hits(const ngbtree::Tree& tree, int n_cells, const ItemsByRank<Query>& queries, Hits& hits);
        void keep_new_hits(const device::Array<uint64_t>& sent, int n_sent, Hits& hits);
        void write_answers(const ngbtree::Tree& tree, const Hits& hits, ItemsByRank<Answer>& answers);
        void remember_as_sent(const Hits&              hits,
                              device::Array<uint64_t>& sent,
                              int&                     n_sent,
                              device::Array<uint64_t>& merged_sent);
        void append_ghost_seeds(const ItemsByRank<Answer>& answers,
                                int&                       n_ghost_seeds,
                                device::Array<Vec>&        position,
                                device::Array<int>&        owner,
                                device::Array<int>&        cell_on_owner,
                                device::Array<int>&        shift);

        template <typename Function>
        HD void for_each_query_of_ball(const domain::DomainPointers& domain,
                                       const Vec&                    center,
                                       double                        radius,
                                       double                        drift_margin,
                                       Function                      function);
        HD inline Vec shift_vector(int shift);
        HD inline uint64_t key_of_hit(int asker, int cell, int shift);
        HD inline int asker_of_key(uint64_t key);
        HD inline int cell_of_key(uint64_t key);
        HD inline int shift_of_key(uint64_t key);
        HD inline int rank_of_item(const int* first, int n_ranks, int item);
        template <typename Key> HD inline int number_below(const Key* sorted, int n, Key key);

    } // namespace

    void Ghosts::start_new_build() {
        n_ghost_seeds           = 0;
        n_sent                  = 0;
        n_ghost_cells           = 0;
        n_seeds_with_ghost_cell = 0;
        n_cells_sent            = 0;
    }

    // every rank takes part, also with no balls of its own
    void Ghosts::add_cells_in_balls(const Vec*            center,
                                    const double*         radius,
                                    int                   n_balls,
                                    double                drift_margin,
                                    const domain::Domain& domain,
                                    const ngbtree::Tree&  tree,
                                    int                   n_cells) {
        write_queries_by_rank(center, radius, n_balls, drift_margin, domain.pointers(), queries_out, next_place);
        send_to_their_ranks(queries_out, queries_in);

        find_hits(tree, n_cells, queries_in, hits);
        keep_new_hits(sent, n_sent, hits);
        write_answers(tree, hits, answers_out);
        remember_as_sent(hits, sent, n_sent, merged_sent);
        send_to_their_ranks(answers_out, answers_in);

        append_ghost_seeds(answers_in, n_ghost_seeds, position, owner, cell_on_owner, shift);
    }

    void Ghosts::give_ghost_cells_to(const bool* used, int n_cells) {
        const int n_ranks       = mpi::nranks();
        const int n_ghost_seeds = this->n_ghost_seeds;
        this->ghost_cell.grow_to_at_least(n_ghost_seeds);
        this->ghost_seed.grow_to_at_least(n_ghost_seeds);
        ghost_cells_of_rank.grow_to_at_least(n_ranks);
        cells_for_rank.grow_to_at_least(n_ranks);

        // the new ghost cells, behind the old ones
        int*      ghost_cell      = this->ghost_cell.data();
        int*      ghost_seed      = this->ghost_seed.data();
        const int old_ghost_cells = this->n_ghost_cells;
        const int known           = n_seeds_with_ghost_cell;
        device::parallel_for(n_ghost_seeds - known, [=] HD(long i) { ghost_cell[known + i] = -1; });
        const int n_new = device::indices_where(
            n_ghost_seeds, [=] HD(int g) { return used[g] && ghost_cell[g] < 0; }, ghost_seed + old_ghost_cells);
        device::parallel_for(
            n_new, [=] HD(long i) { ghost_cell[ghost_seed[old_ghost_cells + i]] = old_ghost_cells + (int)i; });
        this->n_ghost_cells     = old_ghost_cells + n_new;
        n_seeds_with_ghost_cell = n_ghost_seeds;
        const int n_ghost_cells = this->n_ghost_cells;

        // the ghost cells by owner; the sort keeps the order within an owner, which is the order the owners send in
        device::Array<uint64_t> owner_of_ghost_cell("ghosts: owners of the ghost cells", n_ghost_cells);
        device::Array<int>      by_owner("ghosts: ghost cells by owner", n_ghost_cells);
        device::Array<int>      cell_of_ghost_cell("ghosts: cells of the ghost cells on their owners", n_ghost_cells);
        ghost_cell_place_by_owner.grow_to_at_least(n_ghost_cells);
        const int* owner         = this->owner.data();
        const int* cell_on_owner = this->cell_on_owner.data();
        uint64_t*  key           = owner_of_ghost_cell.data();
        int*       order         = by_owner.data();
        device::parallel_for(n_ghost_cells, [=] HD(long i) {
            key[i]   = (uint64_t)owner[ghost_seed[i]];
            order[i] = (int)i;
        });
        device::sort_by_key(n_ghost_cells, owner_of_ghost_cell.data(), order);

        int* place = ghost_cell_place_by_owner.data();
        int* cell  = cell_of_ghost_cell.data();
        device::parallel_for(n_ghost_cells, [=] HD(long i) {
            place[i] = n_cells + order[i];
            cell[i]  = cell_on_owner[ghost_seed[order[i]]];
        });

        const uint64_t* first = owner_of_ghost_cell.data();
        const uint64_t* end   = first + n_ghost_cells;
        for (int r = 0; r < n_ranks; r++)
            ghost_cells_of_rank[r] =
                (int)(std::lower_bound(first, end, (uint64_t)r + 1) - std::lower_bound(first, end, (uint64_t)r));

        mpi::exchange_counts(ghost_cells_of_rank.data(), cells_for_rank.data());
        first_ghost_cell_of_rank.grow_to_at_least(n_ranks + 1);
        first_cell_for_rank.grow_to_at_least(n_ranks + 1);
        first_ghost_cell_of_rank[0] = 0;
        first_cell_for_rank[0]      = 0;
        for (int r = 0; r < n_ranks; r++) {
            first_ghost_cell_of_rank[r + 1] = first_ghost_cell_of_rank[r] + ghost_cells_of_rank[r];
            first_cell_for_rank[r + 1]      = first_cell_for_rank[r] + cells_for_rank[r];
        }

        // the owners learn which cells they send
        n_cells_sent = first_cell_for_rank[n_ranks];
        cell_sent.grow_to_at_least(n_cells_sent);
        device::exchange_items(cell_of_ghost_cell.data(),
                               ghost_cells_of_rank.data(),
                               cell_sent.data(),
                               cells_for_rank.data(),
                               (int)sizeof(int));
    }

    // the cells sent are by asking rank, in the order of the ghost cells there, so the items are too
    int Ghosts::start_items_of_changed_cells(const int* changed, int n_changed, int item_bytes) {
        const int n_ranks = mpi::nranks();
        changed_sent.grow_to_at_least(n_cells_sent);
        changed_cell_sent.grow_to_at_least(n_cells_sent);
        changed_for_rank.grow_to_at_least(n_ranks);
        changed_from_rank.grow_to_at_least(n_ranks);
        first_changed_from_rank.grow_to_at_least(n_ranks + 1);

        const int* cell_sent  = this->cell_sent.data();
        int*       sent       = changed_sent.data();
        const auto is_changed = [=] HD(int place) {
            const int below = number_below(changed, n_changed, cell_sent[place]);
            return below < n_changed && changed[below] == cell_sent[place];
        };
        const int n_items = device::indices_where(n_cells_sent, is_changed, sent);

        const int* first_for_rank = first_cell_for_rank.data();
        for (int r = 0; r < n_ranks; r++) {
            changed_for_rank[r] =
                number_below(sent, n_items, first_for_rank[r + 1]) - number_below(sent, n_items, first_for_rank[r]);
        }
        mpi::exchange_counts(changed_for_rank.data(), changed_from_rank.data());
        first_changed_from_rank[0] = 0;
        for (int r = 0; r < n_ranks; r++)
            first_changed_from_rank[r + 1] = first_changed_from_rank[r] + changed_from_rank[r];
        n_changed_received = first_changed_from_rank[n_ranks];

        // every item starts with its place in the block of its rank
        values_sent.grow_to_at_least((long)n_items * item_bytes);
        values_received.grow_to_at_least((long)n_changed_received * item_bytes);
        int*  cell  = changed_cell_sent.data();
        char* items = values_sent.data();
        device::parallel_for(n_items, [=] HD(long i) {
            const int place_in_block = sent[i] - first_for_rank[rank_of_item(first_for_rank, n_ranks, sent[i])];
            cell[i]                  = cell_sent[sent[i]];
            memcpy(items + i * item_bytes, &place_in_block, sizeof(int));
        });
        return n_items;
    }

    // the ghost cell of an item is its place in the block of its owner, in the order by owner
    void Ghosts::find_ghost_cells_of_items_that_came(int item_bytes) {
        const int n_ranks = mpi::nranks();
        ghost_cell_of_changed.grow_to_at_least(n_changed_received);
        const int*  first_from_rank  = first_changed_from_rank.data();
        const int*  first_ghost_cell = first_ghost_cell_of_rank.data();
        const int*  place_by_owner   = ghost_cell_place_by_owner.data();
        const char* items            = values_received.data();
        int*        ghost_cell       = ghost_cell_of_changed.data();
        device::parallel_for(n_changed_received, [=] HD(long i) {
            int place_in_block;
            memcpy(&place_in_block, items + i * item_bytes, sizeof(int));
            ghost_cell[i] =
                place_by_owner[first_ghost_cell[rank_of_item(first_from_rank, n_ranks, (int)i)] + place_in_block];
        });
    }

    // the seeds are rounded to the box precision, so a seed and its shifted copy differ by exactly the shift
    void Ghosts::move_to(const Vec* seeds, int n_cells) {
        const int  here          = mpi::rank();
        Vec*       position      = this->position.data();
        const int* owner         = this->owner.data();
        const int* cell_on_owner = this->cell_on_owner.data();
        const int* shift         = this->shift.data();
        const int* ghost_cell    = this->ghost_cell.data();
        device::parallel_for(n_ghost_seeds, [=] HD(long g) {
            const Vec seed = (owner[g] == here) ? seeds[cell_on_owner[g]] : seeds[n_cells + ghost_cell[g]];
            position[g]    = seed + shift_vector(shift[g]);
        });
    }

    // one flag per ghost cell by owner, so that an owner reads them in the order of the cells it sends there
    void Ghosts::mark_cells_of_ghost_cells(int n_cells, const bool* marked_ghost_cell, bool* marked_cell) {
        device::Array<char> flag_out("ghosts: marks of the ghost cells by owner", n_ghost_cells);
        device::Array<char> flag_in("ghosts: marks of the cells sent", n_cells_sent);
        const int*          place = ghost_cell_place_by_owner.data();
        char*               out   = flag_out.data();
        device::parallel_for(n_ghost_cells, [=] HD(long i) { out[i] = marked_ghost_cell[place[i] - n_cells] ? 1 : 0; });
        device::exchange_items(
            flag_out.data(), ghost_cells_of_rank.data(), flag_in.data(), cells_for_rank.data(), (int)sizeof(char));

        const char* in   = flag_in.data();
        const int*  cell = cell_sent.data();
        device::parallel_for(n_cells_sent, [=] HD(long e) {
            if (in[e]) marked_cell[cell[e]] = true;
        });
    }

    GhostPointers Ghosts::pointers() const {
        return GhostPointers{n_ghost_seeds,
                             position.data(),
                             owner.data(),
                             cell_on_owner.data(),
                             shift.data(),
                             n_ghost_cells,
                             ghost_cell.data(),
                             ghost_seed.data()};
    }

    namespace {

        // ==========================================================
        // the parts of add_cells_in_balls
        // ==========================================================

        // counts the queries of every rank, then writes them; the order in a block does not matter, the owner sorts
        void write_queries_by_rank(const Vec*                    center,
                                   const double*                 radius,
                                   int                           n_balls,
                                   double                        drift_margin,
                                   const domain::DomainPointers& domain,
                                   ItemsByRank<Query>&           queries,
                                   device::Array<int>&           next_place) {
            const int n_ranks = domain.n_ranks;
            queries.count.grow_to_at_least(n_ranks);
            queries.first.grow_to_at_least(n_ranks);
            next_place.grow_to_at_least(n_ranks);
            queries.count.fill(0);

            int* count = queries.count.data();
            device::parallel_for(n_balls, [=] HD(long i) {
                for_each_query_of_ball(domain, center[i], radius[i], drift_margin, [&](int rank, const Query&) {
                    device::atomic_add(&count[rank], 1);
                });
            });
            queries.total = device::offsets_from_counts(n_ranks, count, queries.first.data());

            queries.items.grow_to_at_least(queries.total);
            device::copy(queries.first, next_place, n_ranks);
            Query* items = queries.items.data();
            int*   place = next_place.data();
            device::parallel_for(n_balls, [=] HD(long i) {
                for_each_query_of_ball(domain, center[i], radius[i], drift_margin, [&](int rank, const Query& query) {
                    items[device::atomic_add(&place[rank], 1)] = query;
                });
            });
        }

        template <typename Item> void send_to_their_ranks(const ItemsByRank<Item>& out, ItemsByRank<Item>& in) {
            const int n_ranks = mpi::nranks();
            in.count.grow_to_at_least(n_ranks);
            in.first.grow_to_at_least(n_ranks);
            mpi::exchange_counts(out.count.data(), in.count.data());
            in.total = device::offsets_from_counts(n_ranks, in.count.data(), in.first.data());

            in.items.grow_to_at_least(in.total);
            device::exchange_items(
                out.items.data(), out.count.data(), in.items.data(), in.count.data(), (int)sizeof(Item));
        }

        // counts the hits of every query, then writes them; tree points from input index n_cells on are ghost seeds
        void find_hits(const ngbtree::Tree& tree, int n_cells, const ItemsByRank<Query>& queries, Hits& hits) {
            const ngbtree::TreePointers tree_pointers = tree.pointers();
            const int*                  point_of      = tree.input_indices();
            const Query*                query         = queries.items.data();
            hits.count_of_query.grow_to_at_least(queries.total);
            hits.first_of_query.grow_to_at_least(queries.total);

            int* count = hits.count_of_query.data();
            device::parallel_for(queries.total, [=] HD(long q) {
                int n = 0;
                ngbtree::for_each_point_in_ball(tree_pointers, query[q].center, query[q].radius, [&](int hit) {
                    if (point_of[hit] < n_cells) n++;
                });
                count[q] = n;
            });
            hits.n = device::offsets_from_counts(queries.total, count, hits.first_of_query.data());

            hits.key.grow_to_at_least(hits.n);
            hits.tree_index.grow_to_at_least(hits.n);
            const int* first_of_rank = queries.first.data();
            const int  n_ranks       = mpi::nranks();
            const int* first_hit     = hits.first_of_query.data();
            uint64_t*  key           = hits.key.data();
            int*       tree_index    = hits.tree_index.data();
            device::parallel_for(queries.total, [=] HD(long q) {
                const int asker = rank_of_item(first_of_rank, n_ranks, (int)q);
                int       at    = first_hit[q];
                ngbtree::for_each_point_in_ball(tree_pointers, query[q].center, query[q].radius, [&](int hit) {
                    if (point_of[hit] >= n_cells) return;
                    key[at]        = key_of_hit(asker, point_of[hit], query[q].shift);
                    tree_index[at] = hit;
                    at++;
                });
            });
        }

        // sorted by key, a hit is new if its key differs from the one before it and was not sent before in this build
        void keep_new_hits(const device::Array<uint64_t>& sent, int n_sent, Hits& hits) {
            device::sort_by_key(hits.n, hits.key.data(), hits.tree_index.data());

            const uint64_t* key      = hits.key.data();
            const uint64_t* sent_key = sent.data();
            hits.new_hit.grow_to_at_least(hits.n);
            hits.n_new = device::indices_where(
                hits.n,
                [=] HD(int h) {
                    if (h > 0 && key[h] == key[h - 1]) return false;
                    const int place = number_below(sent_key, n_sent, key[h]);
                    return place == n_sent || sent_key[place] != key[h];
                },
                hits.new_hit.data());

            hits.new_key.grow_to_at_least(hits.n_new);
            hits.new_tree_index.grow_to_at_least(hits.n_new);
            const int* new_hit        = hits.new_hit.data();
            const int* tree_index     = hits.tree_index.data();
            uint64_t*  new_key        = hits.new_key.data();
            int*       new_tree_index = hits.new_tree_index.data();
            device::parallel_for(hits.n_new, [=] HD(long k) {
                new_key[k]        = key[new_hit[k]];
                new_tree_index[k] = tree_index[new_hit[k]];
            });
        }

        // the new hits in key order are grouped by asker already
        void write_answers(const ngbtree::Tree& tree, const Hits& hits, ItemsByRank<Answer>& answers) {
            const int n_ranks = mpi::nranks();
            answers.count.grow_to_at_least(n_ranks);
            answers.first.grow_to_at_least(n_ranks);
            answers.count.fill(0);

            const uint64_t* key   = hits.new_key.data();
            int*            count = answers.count.data();
            device::parallel_for(hits.n_new, [=] HD(long k) { device::atomic_add(&count[asker_of_key(key[k])], 1); });
            answers.total = device::offsets_from_counts(n_ranks, count, answers.first.data());

            answers.items.grow_to_at_least(answers.total);
            const ngbtree::TreePointers tree_pointers = tree.pointers();
            const int*                  tree_index    = hits.new_tree_index.data();
            Answer*                     items         = answers.items.data();
            device::parallel_for(hits.n_new, [=] HD(long k) {
                const Vec position = tree_pointers.positions[tree_index[k]] + shift_vector(shift_of_key(key[k]));
                items[k]           = Answer{position, cell_of_key(key[k]), shift_of_key(key[k])};
            });
        }

        // both lists are sorted and share no key, so a key's place follows from the keys of the other list below it
        void remember_as_sent(const Hits&              hits,
                              device::Array<uint64_t>& sent,
                              int&                     n_sent,
                              device::Array<uint64_t>& merged_sent) {
            const int n_old = n_sent;
            const int n_new = hits.n_new;
            merged_sent.grow_to_at_least(n_old + n_new);

            const uint64_t* old_key = sent.data();
            const uint64_t* new_key = hits.new_key.data();
            uint64_t*       merged  = merged_sent.data();
            device::parallel_for(n_old,
                                 [=] HD(long i) { merged[i + number_below(new_key, n_new, old_key[i])] = old_key[i]; });
            device::parallel_for(n_new,
                                 [=] HD(long k) { merged[k + number_below(old_key, n_old, new_key[k])] = new_key[k]; });

            std::swap(sent, merged_sent);
            n_sent = n_old + n_new;
        }

        // by owner rank, and within one owner in the order it sent them
        void append_ghost_seeds(const ItemsByRank<Answer>& answers,
                                int&                       n_ghost_seeds,
                                device::Array<Vec>&        position,
                                device::Array<int>&        owner,
                                device::Array<int>&        cell_on_owner,
                                device::Array<int>&        shift) {
            const int first_new = n_ghost_seeds;
            n_ghost_seeds += answers.total;
            position.grow_to_at_least(n_ghost_seeds);
            owner.grow_to_at_least(n_ghost_seeds);
            cell_on_owner.grow_to_at_least(n_ghost_seeds);
            shift.grow_to_at_least(n_ghost_seeds);

            const Answer* answer        = answers.items.data();
            const int*    first_of_rank = answers.first.data();
            const int     n_ranks       = mpi::nranks();
            Vec*          new_position  = position.data() + first_new;
            int*          new_owner     = owner.data() + first_new;
            int*          new_cell      = cell_on_owner.data() + first_new;
            int*          new_shift     = shift.data() + first_new;
            device::parallel_for(answers.total, [=] HD(long a) {
                new_position[a] = answer[a].position;
                new_owner[a]    = rank_of_item(first_of_rank, n_ranks, (int)a);
                new_cell[a]     = answer[a].cell;
                new_shift[a]    = answer[a].shift;
            });
        }

        // ==========================================================
        // queries, keys and blocks
        // ==========================================================

        // every rank that a periodic copy of the ball, widened by the margin, touches; the unshifted ball skips this
        // rank; the center moves back by the shift, so a cell is in the query exactly when its copy is in the ball
        template <typename Function>
        HD void for_each_query_of_ball(const domain::DomainPointers& domain,
                                       const Vec&                    center,
                                       double                        radius,
                                       double                        drift_margin,
                                       Function                      function) {
            const double reach = radius + drift_margin;
            for (int shift = 0; shift < N_SHIFTS; shift++) {
                const Vec center_seen_by_owner = center - shift_vector(shift);
                int       remembered[RANKS_REMEMBERED];
                int       n_remembered = 0;
                domain::for_each_rank_touched_by_ball(domain, center_seen_by_owner, reach, [&](int rank) {
                    if (rank == domain.this_rank && shift == NO_SHIFT) return;
                    for (int i = 0; i < n_remembered; i++) {
                        if (remembered[i] == rank) return;
                    }
                    if (n_remembered < RANKS_REMEMBERED) remembered[n_remembered++] = rank;
                    function(rank, Query{center_seen_by_owner, radius, shift});
                });
            }
        }

        HD inline Vec shift_vector(int shift) {
            Vec result{};
            for (int d = 0; d < config::dim; d++) {
                result[d] = (double)(shift % 3 - 1);
                shift /= 3;
            }
            return result;
        }

        HD inline uint64_t key_of_hit(int asker, int cell, int shift) {
            return ((uint64_t)asker << (CELL_BITS + SHIFT_BITS)) | ((uint64_t)cell << SHIFT_BITS) | (uint64_t)shift;
        }
        HD inline int asker_of_key(uint64_t key) {
            return (int)(key >> (CELL_BITS + SHIFT_BITS));
        }
        HD inline int cell_of_key(uint64_t key) {
            return (int)((key >> SHIFT_BITS) & ((1ull << CELL_BITS) - 1));
        }
        HD inline int shift_of_key(uint64_t key) {
            return (int)(key & ((1ull << SHIFT_BITS) - 1));
        }

        // the last rank whose block starts at or before item; the empty blocks before it start at the same place
        HD inline int rank_of_item(const int* first, int n_ranks, int item) {
            int low  = 0;
            int high = n_ranks;
            while (high - low > 1) {
                const int middle = (low + high) / 2;
                if (first[middle] <= item) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            return low;
        }

        template <typename Key> HD inline int number_below(const Key* sorted, int n, Key key) {
            int low  = 0;
            int high = n;
            while (low < high) {
                const int middle = (low + high) / 2;
                if (sorted[middle] < key) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            return low;
        }

    } // namespace

} // namespace ghosts
