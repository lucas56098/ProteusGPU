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

    // a cell is a list of planes, a vertex is where DIMENSION planes meet
    // the first 2 x DIMENSION planes are the box walls, every later one is a bisector to a point
    template <int MAX_P, int MAX_T, typename IDX, typename VERT> struct BasicConvexCell {
        HD BasicConvexCell(int p_seed, double* p_pts, Status* p_status, double p_buff);

        static_assert(sizeof(decltype(VERT::x)) == sizeof(IDX), "VERT component width must equal IDX");
        static_assert((long long)MAX_P <= (long long)idx_max<IDX>(), "MAX_P does not fit in IDX");
        static_assert((long long)MAX_T <= (long long)idx_max<IDX>(), "MAX_T does not fit in IDX");

        static constexpr IDX END_OF_LIST = (IDX)(-1);

        double*   pts; // sorted point list, a plane is named by its point
        double4_t voro_seed;
        Status*   status;
        double    buff;

        IDX nb_v; // planes
        IDX nb_t; // vertices
        IDX nb_r; // vertices the current plane cuts away, parked behind nb_t

        int plane_vid[MAX_P]; // point of each plane, -1 for a box wall

        VERT triangle[MAX_T]; // the planes that meet in each vertex

        IDX first_boundary; // ring of planes around the cut away vertices
        IDX boundary_next[MAX_P];

        HD double4_t plane_for(int p) const;

        HD void clip_by_plane(int vid);

        HD int new_halfplane(int vid);

        HD bool vert_is_in_conflict(VERT v, double4_t eqn) const;

        HD void compute_boundary();

        HD void new_vertex(IDX i, IDX j, IDX k = 0);

        // true if last_neig is farther than 2 x the farthest vertex
        HD bool is_security_radius_reached(double4_t last_neig) const;

        // farthest vertex distance as num / denom
        HD void max_vertex_r2_ratio(double* out_num, double* out_denom) const;

        // where the planes of v meet, as (x, y, z, w)
        HD double4_t compute_vertex_point(VERT v, bool persp_divide = true) const;
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

    // true if no point outside the rank data can still cut the cell
    HD bool cell_certified_within_data(
        double4_t seed, double r2_num, double r2_denom, const double* data_lo, const double* data_hi);

    // build cell k and write it into the mesh
    template <int K, int MAX_P, int MAX_T, typename IDX, typename VERT>
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
