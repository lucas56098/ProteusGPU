// the mesh of one build, in rounds of building the open cells and asking for more points (mesh.h)

#include "mesh/mesh.h"

#include "logging/logging.h"
#include "mesh/tiers.h"
#include "mpi/mpi.h"
#include "timers/timers.h"

#include <cmath>
#include <cstdio>

namespace mesh {

    namespace {

        // the face arrays start with room for a little more than the faces of an average cell
        constexpr int FACES_PER_CELL_AT_START = (config::dim == 2) ? 8 : 17;

        // a cell whose ball of this many mean cell spacings lies inside this rank asks for nothing at first
        constexpr double SPACINGS_WITHOUT_ASKING = 4.0;

        // every round doubles the balls that are still open, so a build needs only a few
        constexpr int MAX_ROUNDS = 24;

        void start_build(const Vec*             positions,
                         int                    n_cells,
                         device::Array<Vec>&    seed,
                         device::Array<double>& volume,
                         device::Array<Vec>&    centroid_from_seed,
                         Faces&                 faces,
                         BuildWork&             work);
        void put_cells_and_ghost_seeds_into_tree(int                       n_cells,
                                                 const device::Array<Vec>& seed,
                                                 const ghosts::Ghosts&     ghosts,
                                                 bool                      keep_order,
                                                 BuildWork&                work);
        CellsPerTier build_cells_in_rounds(const CellList&        cells,
                                           int                    n_cells,
                                           const domain::Domain&  domain,
                                           double                 drift_margin,
                                           bool                   keep_tree_order,
                                           ghosts::Ghosts&        ghosts,
                                           device::Array<Vec>&    seed,
                                           device::Array<double>& volume,
                                           device::Array<Vec>&    centroid_from_seed,
                                           Faces&                 faces,
                                           BuildWork&             work);
        void make_first_guess(const CellList&       cells,
                              int                   n_cells,
                              double                drift_margin,
                              const domain::Domain& domain,
                              BuildWork&            work);
        long long ask_for_balls(int                       n_cells,
                                const domain::Domain&     domain,
                                double                    drift_margin,
                                const device::Array<Vec>& seed,
                                ghosts::Ghosts&           ghosts,
                                BuildWork&                work);
        void stop_if_a_ball_cannot_grow(int n_asking, const BuildWork& work);
        int give_ghost_cells_to_ghost_seeds_of_faces(const CellList&        cells,
                                                     int                    n_cells,
                                                     ghosts::Ghosts&        ghosts,
                                                     Faces&                 faces,
                                                     device::Array<Vec>&    seed,
                                                     device::Array<double>& volume,
                                                     device::Array<Vec>&    centroid_from_seed);
        void list_cells_next_to_ghost_cells(int                 n_cells,
                                            int                 n_ghost_cells,
                                            const Faces&        faces,
                                            device::Array<int>& first_cell_next_to_ghost_cell,
                                            device::Array<int>& cell_next_to_ghost_cell);
        void count_for_log(int n_cells, const CellsPerTier& built, BuildsSinceLog& since_log);
        void print_request_round(long long n_asked, const ghosts::Ghosts& ghosts, BuildsSinceLog& since_log);
        double mean_cell_spacing(int n_cells, const domain::DomainPointers& domain);

    } // namespace

    void Mesh::build(const Vec* positions, int n_cells, const domain::Domain& domain, ghosts::Ghosts& ghosts) {
        this->n_cells = n_cells;
        start_build(positions, n_cells, seed, volume, centroid_from_seed, faces, work);
        ghosts.start_new_build();
        const CellList     every = CellList{n_cells, nullptr};
        const CellsPerTier built = build_cells_in_rounds(
            every, n_cells, domain, 0.0, false, ghosts, seed, volume, centroid_from_seed, faces, work);

        {
            const timers::Scope timer("GHOST_CELLS", timers::Type::communication);
            n_ghost_cells = give_ghost_cells_to_ghost_seeds_of_faces(
                every, n_cells, ghosts, faces, seed, volume, centroid_from_seed);
        }
        if constexpr (config::moving_mesh && config::individual_timesteps) {
            built_at_sync.grow_to_at_least(n_cells);
            built_at_sync.fill(true);
        }
        if constexpr (config::individual_timesteps && !config::moving_mesh) {
            list_cells_next_to_ghost_cells(
                n_cells, n_ghost_cells, faces, first_cell_next_to_ghost_cell, cell_next_to_ghost_cell);
        }

        count_for_log(n_cells, built, work.since_log);
    }

    void Mesh::forget_faces() {
        work.face_counter[0] = 0;
        faces.n_faces.fill(0);
        built_at_sync.grow_to_at_least(n_cells);
        built_at_sync.fill(false);
    }

    // as a whole build, but the ghost seeds stay and move with their cells, and only the cells of the list are built
    void Mesh::build_list(const Vec*            seeds,
                          int                   n_cells,
                          const CellList&       cells,
                          const domain::Domain& domain,
                          double                drift_margin,
                          ghosts::Ghosts&       ghosts) {
        if (!mpi::on_any_rank(cells.n > 0)) return;

        Vec* seed = this->seed.data();
        device::parallel_for(n_cells, [=] HD(long c) { seed[c] = seeds[c]; });
        ghosts.move_to(seeds, n_cells);

        work.faces_before = work.face_counter[0];
        work.progress.fill(Progress::not_in_this_build);
        const CellsPerTier per_tier = build_cells_in_rounds(
            cells, n_cells, domain, drift_margin, true, ghosts, this->seed, volume, centroid_from_seed, faces, work);
        {
            const timers::Scope timer("GHOST_CELLS", timers::Type::communication);
            n_ghost_cells = give_ghost_cells_to_ghost_seeds_of_faces(
                cells, n_cells, ghosts, faces, this->seed, volume, centroid_from_seed);
        }

        bool* built = built_at_sync.data();
        device::parallel_for(cells.n, [=] HD(long i) { built[cells[(int)i]] = true; });
        count_for_log(cells.n, per_tier, work.since_log);
    }

    void Mesh::print_builds(const ghosts::Ghosts& ghosts) {
        const ghosts::GhostPointers ghost = ghosts.pointers();
        const BuildsSinceLog&       since = work.since_log;
        long long number[5] = {since.n_cells, since.slow, since.exact, ghost.n_ghost_seeds, ghost.n_ghost_cells};
        mpi::sum_over_ranks(number, 5);
        work.since_log = BuildsSinceLog{};

        char details[256];
        int  written =
            std::snprintf(details, sizeof details, "%g%% slow tier", 100.0 * (double)number[1] / (double)number[0]);
        if (mpi::nranks() > 1) {
            written += std::snprintf(
                details + written, sizeof details - written, ", ghosts=%lld, used=%lld", number[3], number[4]);
        }
        if (number[2] > 0) {
            std::snprintf(details + written, sizeof details - written, ", cpu_fallback=%lld", number[2]);
        }
        logging::print_labelled("VORONOI", "Generated %lld cells (%s)", number[0], details);
    }

    CellList Mesh::neighbours_not_built(const CellList& cells, ghosts::Ghosts& ghosts) {
        const int n_cells = this->n_cells;
        this->wanted.grow_to_at_least(n_cells);
        this->wanted_ghost_cell.grow_to_at_least(n_ghost_cells);
        neighbours.grow_to_at_least(n_cells);
        this->wanted.fill(false);
        this->wanted_ghost_cell.fill(false);

        // the cells and ghost cells behind the faces of the list
        bool*      wanted            = this->wanted.data();
        bool*      wanted_ghost_cell = this->wanted_ghost_cell.data();
        const int* first_face        = faces.first_face.data();
        const int* n_faces           = faces.n_faces.data();
        const int* neighbour         = faces.neighbour.data();
        device::parallel_for(cells.n, [=] HD(long i) {
            const int cell = cells[(int)i];
            for (int f = first_face[cell]; f < first_face[cell] + n_faces[cell]; f++) {
                if (neighbour[f] < n_cells) {
                    wanted[neighbour[f]] = true;
                } else {
                    wanted_ghost_cell[neighbour[f] - n_cells] = true;
                }
            }
        });

        ghosts.mark_cells_of_ghost_cells(n_cells, wanted_ghost_cell, wanted);

        const bool* built = built_at_sync.data();
        const int   n =
            device::indices_where(n_cells, [=] HD(int c) { return wanted[c] && !built[c]; }, neighbours.data());
        return CellList{n, neighbours.data()};
    }

    MeshPointers Mesh::pointers() const {
        return MeshPointers{n_cells,
                            n_ghost_cells,
                            seed.data(),
                            volume.data(),
                            centroid_from_seed.data(),
                            faces.first_face.data(),
                            faces.n_faces.data(),
                            faces.neighbour.data(),
                            faces.area.data(),
                            faces.centroid_along_face.data(),
                            first_cell_next_to_ghost_cell.data(),
                            cell_next_to_ghost_cell.data(),
                            reference_radius};
    }

    ngbtree::TreePointers Mesh::tree() const {
        return work.tree.pointers();
    }

    const int* Mesh::point_of_tree_index() const {
        return work.tree.input_indices();
    }

    namespace {

        // copies the seeds in and makes room for the arrays; the face counter starts at zero
        void start_build(const Vec*             positions,
                         int                    n_cells,
                         device::Array<Vec>&    seed,
                         device::Array<double>& volume,
                         device::Array<Vec>&    centroid_from_seed,
                         Faces&                 faces,
                         BuildWork&             work) {
            seed.grow_to_at_least(n_cells);
            volume.grow_to_at_least(n_cells);
            centroid_from_seed.grow_to_at_least(n_cells);
            faces.first_face.grow_to_at_least(n_cells);
            faces.n_faces.grow_to_at_least(n_cells);

            const long room = (long)FACES_PER_CELL_AT_START * n_cells;
            faces.neighbour.grow_to_at_least(room);
            faces.area.grow_to_at_least(room);
            faces.centroid_along_face.grow_to_at_least((config::dim - 1) * room);

            work.tree_index_of_cell.grow_to_at_least(n_cells);
            work.progress.grow_to_at_least(n_cells);
            work.status.grow_to_at_least(n_cells);
            work.handover.grow_to_at_least(n_cells);
            work.known_radius.grow_to_at_least(n_cells);
            work.next_radius.grow_to_at_least(n_cells);
            work.list.grow_to_at_least(n_cells);
            work.face_counter.grow_to_at_least(1);
            work.face_counter[0] = 0;
            work.faces_before    = 0;

            Vec* to = seed.data();
            device::parallel_for(n_cells, [=] HD(long c) { to[c] = positions[c]; });
        }

        // with keep_order the ghost seeds only come behind, so the same number of points means the same points
        void put_cells_and_ghost_seeds_into_tree(int                       n_cells,
                                                 const device::Array<Vec>& seed,
                                                 const ghosts::Ghosts&     ghosts,
                                                 bool                      keep_order,
                                                 BuildWork&                work) {
            const ghosts::GhostPointers ghost    = ghosts.pointers();
            const int                   n_points = n_cells + ghost.n_ghost_seeds;
            work.points.grow_to_at_least(n_points);

            const Vec* cell_seed = seed.data();
            Vec*       points    = work.points.data();
            device::parallel_for(
                n_points, [=] HD(long i) { points[i] = (i < n_cells) ? cell_seed[i] : ghost.position[i - n_cells]; });
            if (keep_order && work.tree.pointers().n_points == n_points) {
                work.tree.move_points(points);
                return;
            }
            work.tree.build(points, n_points);

            const int* point_of           = work.tree.input_indices();
            int*       tree_index_of_cell = work.tree_index_of_cell.data();
            device::parallel_for(n_points, [=] HD(long t) {
                if (point_of[t] < n_cells) tree_index_of_cell[point_of[t]] = (int)t;
            });
        }

        // builds the open cells of the list in rounds, until no cell on any rank asks for more points; collective
        CellsPerTier build_cells_in_rounds(const CellList&        cells,
                                           int                    n_cells,
                                           const domain::Domain&  domain,
                                           double                 drift_margin,
                                           bool                   keep_tree_order,
                                           ghosts::Ghosts&        ghosts,
                                           device::Array<Vec>&    seed,
                                           device::Array<double>& volume,
                                           device::Array<Vec>&    centroid_from_seed,
                                           Faces&                 faces,
                                           BuildWork&             work) {
            {
                const timers::Scope timer("TREE");
                put_cells_and_ghost_seeds_into_tree(n_cells, seed, ghosts, keep_tree_order, work);
            }
            {
                const timers::Scope timer("FIRST_GUESS");
                make_first_guess(cells, n_cells, drift_margin, domain, work);
            }
            {
                const timers::Scope timer("GHOST_SEEDS", timers::Type::communication);
                ask_for_balls(n_cells, domain, drift_margin, seed, ghosts, work);
            }

            CellsPerTier built;
            int          n_rounds = 0;
            bool         asked    = true;
            while (asked) {
                if (n_rounds == MAX_ROUNDS)
                    logging::stop_run("VORONOI: the cells still ask for points after %d rounds", n_rounds);
                // the seeds stay during the rounds, so only new ghost seeds change the tree
                if (n_cells + ghosts.pointers().n_ghost_seeds != work.tree.pointers().n_points) {
                    const timers::Scope timer("TREE");
                    put_cells_and_ghost_seeds_into_tree(n_cells, seed, ghosts, false, work);
                }
                {
                    const timers::Scope timer("CELLS");
                    built.add(build_open_cells(
                        n_cells, domain, drift_margin, ghosts, seed, volume, centroid_from_seed, faces, work));
                }
                n_rounds++;
                const timers::Scope timer("GHOST_SEEDS", timers::Type::communication);
                const long long     n_asked = ask_for_balls(n_cells, domain, drift_margin, seed, ghosts, work);
                asked                       = n_asked > 0;
                if (asked) print_request_round(n_asked, ghosts, work.since_log);
            }
            return built;
        }

        // a cell whose ball lies inside this rank is open; any other asks out to the farthest of its nearest points
        void make_first_guess(const CellList&       cells,
                              int                   n_cells,
                              double                drift_margin,
                              const domain::Domain& domain,
                              BuildWork&            work) {
            const domain::DomainPointers rank_part = domain.pointers();
            const double                 without_asking =
                fmin(SPACINGS_WITHOUT_ASKING * mean_cell_spacing(n_cells, rank_part), MAX_BALL_RADIUS);

            const ngbtree::TreePointers tree       = work.tree.pointers();
            const int*                  tree_index = work.tree_index_of_cell.data();
            double*                     known      = work.known_radius.data();
            double*                     next       = work.next_radius.data();
            Progress*                   progress   = work.progress.data();
            device::parallel_for(cells.n, [=] HD(long i) {
                const int cell = cells[(int)i];
                const int t    = tree_index[cell];
                const Vec seed = tree.positions[t];
                if (domain::ball_is_inside_this_rank(rank_part, seed, without_asking + drift_margin)) {
                    known[cell]    = without_asking;
                    progress[cell] = Progress::open;
                    return;
                }

                int neighbours[FAST_TIER_K];
                ngbtree::nearest_neighbours<FAST_TIER_K>(tree, (int)t, neighbours);
                double reach = 0.0;
                for (int k = 0; k < FAST_TIER_K && neighbours[k] >= 0; k++)
                    reach = sqrt(norm2(tree.positions[neighbours[k]] - seed));
                known[cell]    = 0.0;
                next[cell]     = (reach > 0.0 && reach < MAX_BALL_RADIUS) ? reach : MAX_BALL_RADIUS;
                progress[cell] = Progress::asks_for_more_points;
            });
        }

        // cells that ask send their balls to ghosts and are open again; returns how many asked on all ranks; collective
        long long ask_for_balls(int                       n_cells,
                                const domain::Domain&     domain,
                                double                    drift_margin,
                                const device::Array<Vec>& seed,
                                ghosts::Ghosts&           ghosts,
                                BuildWork&                work) {
            Progress* progress = work.progress.data();
            int*      asking   = work.list.data();
            const int n_asking = device::indices_where(
                n_cells, [=] HD(int c) { return progress[c] == Progress::asks_for_more_points; }, asking);
            const long long n_asked = mpi::sum_over_ranks((long long)n_asking);
            if (n_asked == 0) return 0;
            stop_if_a_ball_cannot_grow(n_asking, work);

            work.ball_center.grow_to_at_least(n_asking);
            work.ball_radius.grow_to_at_least(n_asking);
            const Vec*    cell_seed = seed.data();
            double*       known     = work.known_radius.data();
            const double* next      = work.next_radius.data();
            Vec*          center    = work.ball_center.data();
            double*       radius    = work.ball_radius.data();
            device::parallel_for(n_asking, [=] HD(long i) {
                center[i] = cell_seed[asking[i]];
                radius[i] = next[asking[i]];
            });
            ghosts.add_cells_in_balls(center, radius, n_asking, drift_margin, domain, work.tree, n_cells);

            // now they know every point in their balls
            device::parallel_for(n_asking, [=] HD(long i) {
                known[asking[i]]    = next[asking[i]];
                progress[asking[i]] = Progress::open;
            });
            return n_asked;
        }

        // a cell that asks for no more than it knows already has reached MAX_BALL_RADIUS: the box has too few cells
        void stop_if_a_ball_cannot_grow(int n_asking, const BuildWork& work) {
            const int*    asking = work.list.data();
            const double* known  = work.known_radius.data();
            const double* next   = work.next_radius.data();
            const int     n_stuck =
                device::sum(n_asking, [=] HD(long i) { return (next[asking[i]] <= known[asking[i]]) ? 1 : 0; });
            if (n_stuck > 0) {
                logging::stop_run(
                    "VORONOI: %d cells need points from farther than %g of the box; the box has too few cells",
                    n_stuck,
                    MAX_BALL_RADIUS);
            }
        }

        // gives a ghost cell to every ghost seed of another rank a face names; returns how many; collective
        int give_ghost_cells_to_ghost_seeds_of_faces(const CellList&        cells,
                                                     int                    n_cells,
                                                     ghosts::Ghosts&        ghosts,
                                                     Faces&                 faces,
                                                     device::Array<Vec>&    seed,
                                                     device::Array<double>& volume,
                                                     device::Array<Vec>&    centroid_from_seed) {
            const ghosts::GhostPointers before        = ghosts.pointers();
            const int                   n_ghost_seeds = before.n_ghost_seeds;
            const int                   here          = mpi::rank();
            const int*                  first_face    = faces.first_face.data();
            const int*                  n_faces       = faces.n_faces.data();
            int*                        neighbour     = faces.neighbour.data();

            // with both switches every ghost seed of another rank moves with its cell, whose path its ghost cell holds
            device::Array<int>  faces_naming("mesh: faces naming every ghost seed", n_ghost_seeds);
            device::Array<bool> used("mesh: ghost seeds a face names", n_ghost_seeds);
            faces_naming.fill(0);
            int*  naming  = faces_naming.data();
            bool* is_used = used.data();
            device::parallel_for(cells.n, [=] HD(long i) {
                const int c = cells[(int)i];
                for (int f = first_face[c]; f < first_face[c] + n_faces[c]; f++) {
                    if (neighbour[f] >= n_cells) device::atomic_add(&naming[neighbour[f] - n_cells], 1);
                }
            });
            constexpr bool every_ghost_seed = config::moving_mesh && config::individual_timesteps;
            device::parallel_for(n_ghost_seeds, [=] HD(long g) {
                is_used[g] = every_ghost_seed ? before.owner[g] != here : naming[g] > 0;
            });
            ghosts.give_ghost_cells_to(used.data(), n_cells);

            // the faces name the ghost cells; the ghost cells get their seeds and centroids
            const ghosts::GhostPointers ghost = ghosts.pointers();
            seed.grow_to_at_least(n_cells + ghost.n_ghost_cells);
            centroid_from_seed.grow_to_at_least(n_cells + ghost.n_ghost_cells);
            device::parallel_for(cells.n, [=] HD(long i) {
                const int c = cells[(int)i];
                for (int f = first_face[c]; f < first_face[c] + n_faces[c]; f++) {
                    if (neighbour[f] >= n_cells) neighbour[f] = n_cells + ghost.ghost_cell[neighbour[f] - n_cells];
                }
            });
            Vec* ghost_cell_seed = seed.data() + n_cells;
            device::parallel_for(ghost.n_ghost_cells,
                                 [=] HD(long i) { ghost_cell_seed[i] = ghost.position[ghost.ghost_seed[i]]; });
            if constexpr (config::regularization_speed_by_volume > 0.0) {
                volume.grow_to_at_least(n_cells + ghost.n_ghost_cells);
                ghosts.copy_to_ghost_cells(n_cells, [&](auto function) {
                    function(centroid_from_seed);
                    function(volume);
                });
            } else {
                ghosts.copy_to_ghost_cells(n_cells, [&](auto function) { function(centroid_from_seed); });
            }
            return ghost.n_ghost_cells;
        }

        // pairs (ghost cell, cell), one per face to a ghost cell, sorted by ghost cell, then in cell order
        void list_cells_next_to_ghost_cells(int                 n_cells,
                                            int                 n_ghost_cells,
                                            const Faces&        faces,
                                            device::Array<int>& first_cell_next_to_ghost_cell,
                                            device::Array<int>& cell_next_to_ghost_cell) {
            const int* first_face = faces.first_face.data();
            const int* n_faces    = faces.n_faces.data();
            const int* neighbour  = faces.neighbour.data();

            device::Array<int> faces_to_ghost_cells("mesh: faces of every cell to ghost cells", n_cells);
            device::Array<int> first_pair("mesh: first pair of every cell", n_cells);
            int*               to_ghost_cells = faces_to_ghost_cells.data();
            device::parallel_for(n_cells, [=] HD(long c) {
                int n = 0;
                for (int f = first_face[c]; f < first_face[c] + n_faces[c]; f++)
                    n += (neighbour[f] >= n_cells) ? 1 : 0;
                to_ghost_cells[c] = n;
            });
            const int n_pairs = device::offsets_from_counts(n_cells, to_ghost_cells, first_pair.data());

            device::Array<uint64_t> ghost_cell_of_pair("mesh: ghost cell of every pair", n_pairs);
            cell_next_to_ghost_cell.grow_to_at_least(n_pairs);
            first_cell_next_to_ghost_cell.grow_to_at_least(n_ghost_cells + 1);
            uint64_t*  ghost_cell = ghost_cell_of_pair.data();
            int*       cell       = cell_next_to_ghost_cell.data();
            const int* first      = first_pair.data();
            device::parallel_for(n_cells, [=] HD(long c) {
                int pair = first[c];
                for (int f = first_face[c]; f < first_face[c] + n_faces[c]; f++) {
                    if (neighbour[f] < n_cells) continue;
                    ghost_cell[pair] = (uint64_t)(neighbour[f] - n_cells);
                    cell[pair]       = (int)c;
                    pair++;
                }
            });
            device::sort_by_key(n_pairs, ghost_cell, cell);

            // where the cells of every ghost cell begin; every ghost cell has a face
            int* first_cell = first_cell_next_to_ghost_cell.data();
            device::parallel_for(n_pairs, [=] HD(long i) {
                if (i == 0 || ghost_cell[i] != ghost_cell[i - 1]) first_cell[ghost_cell[i]] = (int)i;
            });
            first_cell[n_ghost_cells] = n_pairs;
        }

        void count_for_log(int n_cells, const CellsPerTier& built, BuildsSinceLog& since_log) {
            since_log.n_cells += n_cells;
            since_log.slow += built.slow;
            since_log.exact += built.exact;
        }

        // the rounds count on over all builds of one line in the log; collective
        void print_request_round(long long n_asked, const ghosts::Ghosts& ghosts, BuildsSinceLog& since_log) {
            since_log.n_request_rounds++;
            const long long n_ghost_seeds = mpi::sum_over_ranks((long long)ghosts.pointers().n_ghost_seeds);
            if (mpi::nranks() > 1) {
                logging::print_labelled("VORONOI",
                                        "Round %d: %lld cells asked for more points (ghosts=%lld)",
                                        since_log.n_request_rounds,
                                        n_asked,
                                        n_ghost_seeds);
            } else {
                logging::print_labelled(
                    "VORONOI", "Round %d: %lld cells asked for more points", since_log.n_request_rounds, n_asked);
            }
        }

        // from the part of the curve this rank holds; only for the first guess, so the rounding of pow does not matter
        double mean_cell_spacing(int n_cells, const domain::DomainPointers& domain) {
            const double share = (double)(domain.cuts[domain.this_rank + 1] - domain.cuts[domain.this_rank]) /
                                 (double)domain.cuts[domain.n_ranks];
            return std::pow(share / (double)((n_cells > 0) ? n_cells : 1), 1.0 / config::dim);
        }

    } // namespace

} // namespace mesh
