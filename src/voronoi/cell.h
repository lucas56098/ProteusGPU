#ifndef CELL_H
#define CELL_H

// One cell while it is built, and the helpers that read a finished one.

#include "../global/allvars.h"
#include "../knn/knn.h"
#include "geometry.h"
#include "voronoi.h"

namespace voronoi {

    template <typename T> HD constexpr long long idx_max() {
        return (T)(-1) > 0 ? (long long)(T)(-1) : (long long)((1ULL << (8 * sizeof(T) - 1)) - 1ULL);
    }

    template <typename VERT> HD inline VERT make_vert(int i, int j, int k = 0) {
        VERT v;
        v.x = (decltype(v.x))i;
        v.y = (decltype(v.y))j;
#ifdef dim_3D
        v.z = (decltype(v.z))k;
#else
        (void)k;
#endif
        return v;
    }

    // the starting cell is the box plus this margin on every side; no cell reaches half a box from its seed
    constexpr double CELL_BOX_MARGIN = 0.5;

    // its walls, a hair past the margin: at -CELL_WALL_LO and CELL_WALL_HI on every axis
    constexpr double CELL_WALL_LO = CELL_BOX_MARGIN + 1e-14;
    constexpr double CELL_WALL_HI = 1.0 + CELL_BOX_MARGIN + 1e-14;

    // a cell is a list of planes, a vertex is where DIMENSION planes meet
    // the first 2 x DIMENSION planes are the box walls, every later one is a bisector to a point
    // an exact cell (CPU only) decides every cut with exact tests on the points and a fixed rule for ties
    template <int MAX_P, int MAX_T, typename IDX, typename VERT> struct BasicConvexCell {
        HD BasicConvexCell(int p_seed, double* p_pts, Status* p_status, bool p_exact = false);

        static_assert(sizeof(decltype(VERT::x)) == sizeof(IDX), "VERT component width must equal IDX");
        static_assert((long long)MAX_P <= (long long)idx_max<IDX>(), "MAX_P does not fit in IDX");
        static_assert((long long)MAX_T <= (long long)idx_max<IDX>(), "MAX_T does not fit in IDX");

        static constexpr IDX END_OF_LIST = (IDX)(-1);

        double*   pts; // sorted point list, a plane is named by its point
        double4_t voro_seed;
        Status*   status;
        bool      exact;

        IDX nb_v; // planes
        IDX nb_t; // vertices
        IDX nb_r; // vertices the current plane cuts away, parked behind nb_t

        // farthest vertex of the last security check, still right while no clip changed the cell
        double far_num;
        double far_denom;
        bool   far_valid;

        int plane_vid[MAX_P]; // point of each plane, -1 for a box wall

        VERT triangle[MAX_T]; // the planes that meet in each vertex

        IDX first_boundary; // ring of planes around the cut away vertices
        IDX boundary_next[MAX_P];

        // plane p as (n, w) with n . x + w >= 0 inside; scale, if given, bounds the size of the row
        HD double4_t plane_for(int p, double* scale = nullptr) const;

        HD void clip_by_plane(int vid);

        HD int new_halfplane(int vid);

        // whether plane (with equation eqn and scale eqn_scale) cuts vertex v away
        HD bool vert_is_in_conflict(VERT v, int plane, double4_t eqn, double eqn_scale) const;

        HD void compute_boundary();

        HD void new_vertex(IDX i, IDX j, IDX k = 0);

        // true if last_neig is farther than 2 x the farthest vertex
        HD bool is_security_radius_reached(double4_t last_neig);

        // farthest vertex distance as num / denom
        HD void max_vertex_r2_ratio(double* out_num, double* out_denom) const;

        // where the planes of v meet, as (x, y, z, w)
        HD double4_t compute_vertex_point(VERT v, bool persp_divide = true) const;

        // the point whose bisector with the seed is plane p; for a wall the mirror image of the seed
        HD void plane_point(int p, double* out) const;

        // the exact versions, host only
        bool      exact_conflict(VERT v, int plane) const;
        double4_t exact_vertex_point(VERT v) const;
#ifdef dim_2D
        bool exact_vertex_turns_left(IDX i, IDX j) const;
#endif
    };

    using ConvexCell = BasicConvexCell<_MAX_P_, _MAX_T_, uchar, VERT_TYPE>; // slow tier

    using BigConvexCell = BasicConvexCell<_BIG_MAX_P_, _BIG_MAX_T_, int, BIG_VERT_TYPE>; // wide tier

    void ensure_face_capacity(VMesh* mesh, uint64_t needed);

    // read a finished cell
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD bool collect_face_vertices(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell,
                                  int                                             p,
                                  const double4_t*                                vertices,
                                  double4_t*                                      face_verts,
                                  int*                                            n_face_verts);

    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD int count_cell_faces(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell);

    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD uint64_t extract_cell_all(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell,
                                 VMesh*                                          mesh,
                                 uint64_t                                        cell_index);

    // adds every point that can still cut the cell, found on the tree; the cell is exact afterwards
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void
    close_cell_by_tree_walk(BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell, int seed_id, const knn_problem* knn);

    // build cell k and write it into the mesh; WALK finishes a cell its K points did not close
    template <int K, int MAX_P, int MAX_T, typename IDX, typename VERT, bool WALK = false>
    HD void compute_single_voronoi_cell(int                 k,
                                        int                 seed_id,
                                        double*             d_stored_points,
                                        const knn_problem*  knn,
                                        Status*             stat,
                                        VMesh*              mesh,
                                        unsigned long long* face_offset,
                                        int*                overflow_flag);

} // namespace voronoi

#endif
