// the three tiers that build the open cells of a round: fast and slow on the GPU, exact on the CPU (tiers.h)

#include "mesh/tiers.h"

#include "logging/logging.h"

namespace mesh {

    namespace {

        // how a tier builds a cell; THREADS_PER_BLOCK and MIN_BLOCKS_PER_MULTIPROCESSOR are tuned for the GH200
        struct FastTier {
            static constexpr int  K           = FAST_TIER_K;
            static constexpr int  MAX_PLANES  = (config::dim == 2) ? 20 : 30;
            static constexpr int  MAX_CORNERS = (config::dim == 2) ? 20 : 60;
            static constexpr bool WALK        = false;
            static constexpr bool EXACT       = false;

            static constexpr int THREADS_PER_BLOCK             = 64;
            static constexpr int MIN_BLOCKS_PER_MULTIPROCESSOR = 16;
        };
        struct SlowTier {
            static constexpr int  K           = (config::dim == 2) ? 35 : 64;
            static constexpr int  MAX_PLANES  = (config::dim == 2) ? 30 : 50;
            static constexpr int  MAX_CORNERS = (config::dim == 2) ? 60 : 96;
            static constexpr bool WALK        = true;
            static constexpr bool EXACT       = false;

            static constexpr int THREADS_PER_BLOCK             = 64;
            static constexpr int MIN_BLOCKS_PER_MULTIPROCESSOR = 8;
        };
        static_assert(SlowTier::K >= FAST_TIER_K, "the slow tier goes on after the neighbours of the fast tier");
        struct ExactTier {
            static constexpr int  K           = (config::dim == 2) ? 35 : 64;
            static constexpr int  MAX_PLANES  = 1024;
            static constexpr int  MAX_CORNERS = 2048;
            static constexpr bool WALK        = true;
            static constexpr bool EXACT       = true;
        };

        struct TierPointers {
            int                    n_cells;
            ngbtree::TreePointers  tree;
            const int*             point_of_tree_index; // a cell, or n_cells + g for ghost seed g
            const int*             tree_index_of_cell;
            ghosts::GhostPointers  ghosts;
            domain::DomainPointers domain;
            double                 drift_margin;
            const Vec*             seed;
            const double*          known_radius;
            double*                next_radius;
            Progress*              progress;
            voronoi_cell::Status*  status;
            uint64_t*              handover; // the faces among the fast tier's neighbours, for the slow tier
            double*                volume;
            Vec*                   centroid_from_seed;
            int*                   face_counter;
            int                    room_for_faces;
            int*                   first_face;
            int*                   n_faces;
            int*                   neighbour;
            double*                area;
            double*                centroid_along_face;
        };

        TierPointers pointers_for_tiers(int                       n_cells,
                                        const domain::Domain&     domain,
                                        double                    drift_margin,
                                        const ghosts::Ghosts&     ghosts,
                                        const device::Array<Vec>& seed,
                                        device::Array<double>&    volume,
                                        device::Array<Vec>&       centroid_from_seed,
                                        Faces&                    faces,
                                        BuildWork&                work);
        void point_at_faces(Faces& faces, TierPointers& p);

        template <typename Tier, typename Condition>
        int build_cells_where(Condition condition, TierPointers p, int n_cells, Faces& faces, BuildWork& work);
        void make_room_for_waiting_faces(int n_cells, int n_waiting, Faces& faces, BuildWork& work);

        template <typename Tier> HD void build_and_judge(int cell, const TierPointers& p);
        template <typename VoronoiCell> HD void store(const VoronoiCell& voronoi, int cell, const TierPointers& p);
        HD inline int neighbour_of_point(int tree_index, const TierPointers& p);

        HD inline void write_centroid_along_face(int                 tree_index,
                                                 const Vec&          seed,
                                                 const Vec&          centroid,
                                                 const TierPointers& p,
                                                 double*             along);

        [[noreturn]] void stop_for_cell(int cell, voronoi_cell::Status status, const Vec& seed);
        const char* words_for(voronoi_cell::Status status);

    } // namespace

    // fast tier on every open cell, slow tier on what is left except cells needing exact tests, exact tier on the rest
    CellsPerTier build_open_cells(int                       n_cells,
                                  const domain::Domain&     domain,
                                  double                    drift_margin,
                                  const ghosts::Ghosts&     ghosts,
                                  const device::Array<Vec>& seed,
                                  device::Array<double>&    volume,
                                  device::Array<Vec>&       centroid_from_seed,
                                  Faces&                    faces,
                                  BuildWork&                work) {
        const TierPointers p =
            pointers_for_tiers(n_cells, domain, drift_margin, ghosts, seed, volume, centroid_from_seed, faces, work);
        const Progress* progress = work.progress.data();
        const auto*     status   = work.status.data();

        CellsPerTier built;
        built.fast = build_cells_where<FastTier>(
            [=] HD(int c) { return progress[c] == Progress::open; }, p, n_cells, faces, work);
        built.slow = build_cells_where<SlowTier>(
            [=] HD(int c) {
                return progress[c] == Progress::open && status[c] != voronoi_cell::Status::needs_exact_tests;
            },
            p,
            n_cells,
            faces,
            work);
        built.exact = build_cells_where<ExactTier>(
            [=] HD(int c) { return progress[c] == Progress::open; }, p, n_cells, faces, work);

        const int n_failed =
            device::indices_where(n_cells, [=] HD(int c) { return progress[c] == Progress::open; }, work.list.data());
        if (n_failed > 0) stop_for_cell(work.list[0], work.status[work.list[0]], seed[work.list[0]]);
        return built;
    }

    namespace {

        TierPointers pointers_for_tiers(int                       n_cells,
                                        const domain::Domain&     domain,
                                        double                    drift_margin,
                                        const ghosts::Ghosts&     ghosts,
                                        const device::Array<Vec>& seed,
                                        device::Array<double>&    volume,
                                        device::Array<Vec>&       centroid_from_seed,
                                        Faces&                    faces,
                                        BuildWork&                work) {
            TierPointers p;
            p.n_cells             = n_cells;
            p.tree                = work.tree.pointers();
            p.point_of_tree_index = work.tree.input_indices();
            p.tree_index_of_cell  = work.tree_index_of_cell.data();
            p.ghosts              = ghosts.pointers();
            p.domain              = domain.pointers();
            p.drift_margin        = drift_margin;
            p.seed                = seed.data();
            p.known_radius        = work.known_radius.data();
            p.next_radius         = work.next_radius.data();
            p.progress            = work.progress.data();
            p.status              = work.status.data();
            p.handover            = work.handover.data();
            p.volume              = volume.data();
            p.centroid_from_seed  = centroid_from_seed.data();
            p.face_counter        = work.face_counter.data();
            point_at_faces(faces, p);
            return p;
        }

        void point_at_faces(Faces& faces, TierPointers& p) {
            p.room_for_faces      = (int)faces.neighbour.size();
            p.first_face          = faces.first_face.data();
            p.n_faces             = faces.n_faces.data();
            p.neighbour           = faces.neighbour.data();
            p.area                = faces.area.data();
            p.centroid_along_face = faces.centroid_along_face.data();
        }

        // builds the cells where condition holds, returns how many; those without room for faces go again with more
        template <typename Tier, typename Condition>
        int build_cells_where(Condition condition, TierPointers p, int n_cells, Faces& faces, BuildWork& work) {
            int*      list     = work.list.data();
            Progress* progress = p.progress;
            const int n_built  = device::indices_where(n_cells, condition, list);
            int       n_listed = n_built;
            while (n_listed > 0) {
                if constexpr (Tier::EXACT) {
                    device::parallel_for_on_cpu(n_listed, [=](long i) { build_and_judge<Tier>(list[i], p); });
                } else {
                    device::parallel_for<Tier::THREADS_PER_BLOCK, Tier::MIN_BLOCKS_PER_MULTIPROCESSOR>(
                        n_listed, [=] HD(long i) { build_and_judge<Tier>(list[i], p); });
                }

                n_listed = device::indices_where(
                    n_cells, [=] HD(int c) { return progress[c] == Progress::no_room_for_faces; }, list);
                if (n_listed == 0) break;
                make_room_for_waiting_faces(n_cells, n_listed, faces, work);
                point_at_faces(faces, p);
                device::parallel_for(n_listed, [=] HD(long i) { progress[list[i]] = Progress::open; });
            }
            return n_built;
        }

        // cells take room from the counter in turn, so the stored faces have no gap and after the first miss none fit;
        // the counter goes back to their end, and the arrays grow by the faces of the waiting cells
        void make_room_for_waiting_faces(int n_cells, int n_waiting, Faces& faces, BuildWork& work) {
            const Progress* progress     = work.progress.data();
            const int*      first_face   = faces.first_face.data();
            const int*      n_faces      = faces.n_faces.data();
            const int*      waiting      = work.list.data();
            const int       faces_before = work.faces_before;
            const int       stored_end   = device::max(n_cells, [=] HD(long c) {
                return (progress[c] == Progress::certain) ? first_face[c] + n_faces[c] : faces_before;
            });
            const long      needed       = device::sum(n_waiting, [=] HD(long i) { return (long)n_faces[waiting[i]]; });

            work.face_counter[0] = stored_end;
            faces.neighbour.grow_to_at_least(stored_end + needed);
            faces.area.grow_to_at_least(stored_end + needed);
            faces.centroid_along_face.grow_to_at_least((config::dim - 1) * (stored_end + needed));
        }

        // certain when the security ball lies inside the known ball or this rank; else it asks for a larger ball
        template <typename Tier> HD void build_and_judge(int cell, const TierPointers& p) {
            const int seed = p.tree_index_of_cell[cell];

            // the slow tier goes on from the faces of a cell that ran out of the neighbours of the fast tier
            voronoi_cell::Handover handover{0, 0};
            if constexpr (Tier::WALK && !Tier::EXACT) {
                if (p.status[cell] == voronoi_cell::Status::needs_more_points)
                    handover = voronoi_cell::Handover{p.handover[cell], FAST_TIER_K};
            }

            voronoi_cell::Cell<Tier::MAX_PLANES, Tier::MAX_CORNERS, Tier::EXACT> voronoi(seed, p.tree.positions);
            voronoi_cell::build_cell<Tier::K, Tier::WALK>(p.tree, seed, voronoi, handover);
            p.status[cell] = voronoi.status();
            if constexpr (!Tier::WALK) {
                if (voronoi.status() == voronoi_cell::Status::needs_more_points) p.handover[cell] = handover.faces;
            }
            if (voronoi.status() != voronoi_cell::Status::ok) return;

            const double security_radius = voronoi.security_radius();
            const bool   certain =
                security_radius <= p.known_radius[cell] ||
                domain::ball_is_inside_this_rank(p.domain, p.seed[cell], security_radius + p.drift_margin);
            if (!certain) {
                p.next_radius[cell] = fmin(fmin(security_radius, 2.0 * p.known_radius[cell]), MAX_BALL_RADIUS);
                p.progress[cell]    = Progress::asks_for_more_points;
                return;
            }
            store(voronoi, cell, p);
        }

        // the room for the faces comes first, from the counter; compute_geometry then writes straight into it
        template <typename VoronoiCell> HD void store(const VoronoiCell& voronoi, int cell, const TierPointers& p) {
            const int n_faces = voronoi.number_of_faces();
            const int first   = device::atomic_add(p.face_counter, n_faces);
            if (first + n_faces > p.room_for_faces) {
                p.n_faces[cell]  = n_faces;
                p.progress[cell] = Progress::no_room_for_faces;
                return;
            }

            const Vec  seed            = p.seed[cell];
            int        face            = first;
            const auto geometry        = voronoi.compute_geometry([&](int neighbour, double area, const Vec& centroid) {
                p.neighbour[face] = neighbour_of_point(neighbour, p);
                p.area[face]      = area;
                write_centroid_along_face(
                    neighbour, seed, centroid, p, &p.centroid_along_face[(config::dim - 1) * (long)face]);
                face++;
            });
            p.first_face[cell]         = first;
            p.n_faces[cell]            = n_faces;
            p.volume[cell]             = geometry.volume;
            p.centroid_from_seed[cell] = geometry.centroid_from_seed;
            p.progress[cell]           = Progress::certain;
        }

        // the cell behind a face, also through a periodic shift; n_cells + g for another rank; -1 for a wall
        HD inline int neighbour_of_point(int tree_index, const TierPointers& p) {
            if (tree_index < 0) return -1;
            const int point = p.point_of_tree_index[tree_index];
            if (point < p.n_cells) return point;
            const int g = point - p.n_cells;
            return (p.ghosts.owner[g] == p.domain.this_rank) ? p.ghosts.cell_on_owner[g] : p.n_cells + g;
        }

        // the centroid from the middle of the seed and the other point, along the frame tangents; zeros for a wall
        HD inline void write_centroid_along_face(int                 tree_index,
                                                 const Vec&          seed,
                                                 const Vec&          centroid,
                                                 const TierPointers& p,
                                                 double*             along) {
            for (int t = 0; t < config::dim - 1; t++)
                along[t] = 0.0;
            if (tree_index < 0) return;

            const Vec   other            = p.tree.positions[tree_index];
            const Frame frame            = frame_around(other - seed);
            const Vec   seen_from_middle = centroid - 0.5 * (seed + other);
            for (int t = 0; t < config::dim - 1; t++)
                along[t] = dot(seen_from_middle, frame.tangent[t]);
        }

        void stop_for_cell(int cell, voronoi_cell::Status status, const Vec& seed) {
            const double z = (config::dim == 3) ? seed[config::dim - 1] : 0.0;
            logging::stop_run("VORONOI: cell %d at (%g, %g, %g) cannot be built, even with exact tests: %s",
                              cell,
                              seed[0],
                              seed[1],
                              z,
                              words_for(status));
        }

        const char* words_for(voronoi_cell::Status status) {
            switch (status) {
            case voronoi_cell::Status::ok:
                return "it is fine";
            case voronoi_cell::Status::needs_more_points:
                return "the points ran out";
            case voronoi_cell::Status::too_many_planes:
                return "more planes than it has room for";
            case voronoi_cell::Status::too_many_corners:
                return "more corners than it has room for";
            case voronoi_cell::Status::broken_ring:
                return "the ring of planes around the cut away corners does not close";
            case voronoi_cell::Status::needs_exact_tests:
                return "a cut too close to call";
            case voronoi_cell::Status::two_points_at_one_place:
                return "two seeds at one place";
            }
            return "";
        }

    } // namespace

} // namespace mesh
