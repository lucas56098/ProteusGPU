#pragma once

// The Voronoi mesh of this rank's cells: the volume, centroid and faces of every cell for given seeds.

#include "device/device.h"
#include "domain/domain.h"
#include "ghosts/ghosts.h"
#include "ngbtree/ngbtree.h"
#include "voronoi_cell/voronoi_cell.h"

namespace mesh {

    // plain pointers to the mesh, for kernels; seed and centroid_from_seed hold the ghost cells behind the cells
    struct MeshPointers {
        int           n_cells;
        int           n_ghost_cells;
        const Vec*    seed; // in the box; a ghost cell's seed as seen from this rank
        const double* volume; // an area in 2D; with REGULARIZE_MESH_VOLUME also of the ghost cells
        const Vec*    centroid_from_seed;
        const int*    first_face;
        const int*    n_faces;
        // the cell on the other side: of this rank, also through a periodic shift, or n_cells + ghost_cell
        const int*    neighbour;
        const double* area; // a length in 2D
        // [(config::dim - 1) f + t]: centroid from the seeds' middle, along tangent t of the frame to the neighbour
        const double* centroid_along_face;
        // with individual timesteps on a static mesh: the cells next to each ghost cell, in cell order; else null
        const int* first_cell_next_to_ghost_cell;
        const int* cell_next_to_ghost_cell;
        double     reference_radius; // with REGULARIZE_MESH_VOLUME: the radius a cell is compared with
    };

    // the cells a kernel works on: cell[i] for i < n, or every cell 0 .. n - 1 when cell is nullptr
    struct CellList {
        int        n;
        const int* cell;
        HD int operator[](int i) const { return cell ? cell[i] : i; }
        bool is_every_cell() const { return cell == nullptr; }
    };

    inline CellList every_cell(const MeshPointers& mesh) {
        return CellList{mesh.n_cells, nullptr};
    }

    // the faces of every cell, in the order the cells were finished
    struct Faces {
        device::Array<int>    first_face{"mesh: first face of every cell", 0};
        device::Array<int>    n_faces{"mesh: faces of every cell", 0};
        device::Array<int>    neighbour{"mesh: neighbours", 0};
        device::Array<double> area{"mesh: face areas", 0};
        device::Array<double> centroid_along_face{"mesh: face centroids along the faces", 0};
    };

    // where a cell stands in a build
    enum class Progress {
        not_in_this_build, // a build of a list leaves it out
        open, // to be built in this round
        asks_for_more_points, // built, but a point that could cut it may be missing
        no_room_for_faces, // certain, but its faces did not fit; built again with more room
        certain, // built with every point that could cut it; its faces are stored
    };

    struct BuildsSinceLog {
        long long n_cells          = 0;
        long long slow             = 0;
        long long exact            = 0;
        int       n_request_rounds = 0;
    };

    // the arrays of a build, kept for the next one
    struct BuildWork {
        // the seeds of the cells, then the ghost seeds; ghosts answers its requests from the same tree
        ngbtree::Tree      tree;
        bool               tree_holds_these_seeds = false; // no seed moved since the tree was made or refitted
        device::Array<Vec> points{"mesh: seeds and ghost seeds", 0};
        device::Array<int> tree_index_of_cell{"mesh: tree index of every cell", 0};

        device::Array<Progress>             progress{"mesh: progress of every cell", 0};
        device::Array<voronoi_cell::Status> status{"mesh: status of the last build of every cell", 0};
        device::Array<double>               known_radius{"mesh: radius around every seed with all points here", 0};
        device::Array<double>               next_radius{"mesh: radius every cell asks for next", 0};

        // the faces taken so far, one int, and where it stood when the build began
        device::Array<int> face_counter{"mesh: counter of faces", 0};
        int                faces_before = 0;

        device::Array<int>    list{"mesh: list of cells", 0};
        device::Array<Vec>    ball_center{"mesh: centers of the balls", 0};
        device::Array<double> ball_radius{"mesh: radii of the balls", 0};

        BuildsSinceLog since_log;
    };

    class Mesh {
      public:
        // the cells at positions, in the box and rounded to its precision, and their ghost cells; collective
        void build(const Vec* positions, int n_cells, const domain::Domain& domain, ghosts::Ghosts& ghosts);

        // at a sync inside a top step (moving mesh, individual timesteps); the builds of lists store the faces anew
        void forget_faces();

        // builds the cells of the list anew at seeds (cells::seeds_at_sync), also moving the ghost seeds; collective
        void build_list(const Vec*            seeds,
                        int                   n_cells,
                        const CellList&       cells,
                        const domain::Domain& domain,
                        double                drift_margin,
                        ghosts::Ghosts&       ghosts);

        // cells not built at this sync with a face to the list, also through a ghost cell; in cell order; collective
        CellList neighbours_not_built(const CellList& cells, ghosts::Ghosts& ghosts);

        // one line in the log for the builds after the last call; collective
        void print_builds(const ghosts::Ghosts& ghosts);

        void set_reference_radius(double radius) { reference_radius = radius; }

        MeshPointers pointers() const;

        // the tree of the last build; point_of_tree_index gives a cell c < n_cells, or n_cells + g for ghost seed g
        ngbtree::TreePointers tree() const;
        const int* point_of_tree_index() const;

      private:
        int                   n_cells          = 0;
        int                   n_ghost_cells    = 0;
        double                reference_radius = 0.0;
        device::Array<Vec>    seed{"mesh: seeds", 0};
        device::Array<double> volume{"mesh: volumes", 0};
        device::Array<Vec>    centroid_from_seed{"mesh: centroids relative to the seeds", 0};
        Faces                 faces;
        BuildWork             work;

        device::Array<int> first_cell_next_to_ghost_cell{"mesh: first cell next to every ghost cell", 0};
        device::Array<int> cell_next_to_ghost_cell{"mesh: cells next to the ghost cells", 0};

        // moving mesh with individual timesteps: the cells built at this sync, and the work of neighbours_not_built
        device::Array<bool> built_at_sync{"mesh: cells built at this sync", 0};
        device::Array<bool> wanted{"mesh: cells next to a list", 0};
        device::Array<bool> wanted_ghost_cell{"mesh: ghost cells next to a list", 0};
        device::Array<int>  neighbours{"mesh: cells next to a list, not built yet", 0};
    };

    // the centroid of a face of this cell, relative to its seed
    HD inline Vec face_centroid_from_seed(const MeshPointers& mesh, int cell, int face);

    // the same, for a caller that already has the direction to the neighbour and the frame around it
    HD inline Vec face_centroid_from_seed(const MeshPointers& mesh,
                                          int                 face,
                                          const Vec&          to_neighbour,
                                          const Frame&        frame);

    // a face with at most this fraction of the surface of its cell counts as no face in the gradients and the bins
    constexpr double NEGLIGIBLE_FACE = 1e-10;
    HD inline double negligible_area_of(const MeshPointers& mesh, int cell);

} // namespace mesh

#include "face_geometry.h"
