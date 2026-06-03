#ifndef CELL_H
#define CELL_H

#include "../global/allvars.h"
#include "../knn/knn.h"
#include "geometry.h"
#include "voronoi.h"

namespace voronoi {

    // per-cell convex-polytope state used during mesh generation. Size limits are template
    // parameters so a small "fast" tier and a full "slow" tier can coexist in the same TU.
    template <int MAX_P, int MAX_T> struct BasicConvexCell {
        HD BasicConvexCell(int p_seed, double* p_pts, Status* p_status, double p_buff);

        // input arrays + per-cell context
        double* pts;
        double4 voro_seed;
        Status* status;
        double  buff; // bounding box covers [-buff, 1+buff]^d (set per-cell so plane_for can read it)

        // current plane / triangle counts
        uchar nb_v; // number of planes (= bounding box + clip planes added so far)
        uchar nb_t; // number of triangles (dual-graph vertices currently in the cell)
        uchar nb_r; // number of triangles tagged for removal during the current clip

        // plane -> source: -1 for bounding-box planes, otherwise the point id behind the bisector
        int plane_vid[MAX_P];

        // dual-graph vertices: each triangle is a corner where DIMENSION planes intersect
        VERT_TYPE triangle[MAX_T];

        // boundary loop (linked list head + next-pointers, used during compute_boundary)
        uchar first_boundary;
        uchar boundary_next[MAX_P];

        // plane equation for slot p, rebuilt on demand from plane_vid[p] (or fixed bounding-box
        // constants for p < 2*DIMENSION). Saves the 960 B/thread that used to live in a stored
        // half_plane[] array.
        HD double4 plane_for(int p) const;

        // clip the cell by the perpendicular bisector of (voro_seed, pts[vid])
        HD void clip_by_plane(int vid);

        // append a new plane slot for vid; returns the new slot index (or -1 on overflow)
        HD int new_halfplane(int vid);

        // is this triangle on the "wrong" side of eqn (and so should be removed by the clip)?
        HD bool vert_is_in_conflict(VERT_TYPE v, double4 eqn) const;

        // build the boundary loop separating kept triangles from removed ones
        HD void compute_boundary();

        // create a new triangle from planes i, j (and k in 3D); writes into triangle[nb_t]
        HD void new_vertex(uchar i, uchar j, uchar k = 0);

        // every cell vertex fits in the sphere of radius ||last_neig - voro_seed|| / 2
        HD bool is_security_radius_reached(double4 last_neig) const;

        // primal point of a dual-graph vertex (intersection of DIMENSION planes)
        HD double4 compute_vertex_point(VERT_TYPE v, bool persp_divide = true) const;
    };

    // ConvexCell uses the slow-tier capacity. The fast-tier kernel instantiates
    // BasicConvexCell<_FAST_MAX_P_, _FAST_MAX_T_> directly.
    using ConvexCell = BasicConvexCell<_MAX_P_, _MAX_T_>;

    // grow mesh's face buffer if `needed` exceeds the current face_capacity
    void ensure_face_capacity(VMesh* mesh, hsize_t needed);

    // walk the boundary of face p in dual-graph order, writing primal vertices into face_verts[].
    // Returns false if the face has fewer than DIMENSION vertices (= not a real face).
    template <int MAX_P, int MAX_T>
    HD bool collect_face_vertices(const BasicConvexCell<MAX_P, MAX_T>& cell,
                                  int                                  p,
                                  const double4*                       vertices,
                                  double4*                             face_verts,
                                  int*                                 n_face_verts);

    // number of planes that contribute a face (at least DIMENSION triangles reference them)
    template <int MAX_P, int MAX_T> HD int count_cell_faces(const BasicConvexCell<MAX_P, MAX_T>& cell);

    // emit cell volume + centroid + per-face data into the global mesh SoA arrays
    template <int MAX_P, int MAX_T>
    HD void extract_cell_all(const BasicConvexCell<MAX_P, MAX_T>& cell, VMesh* mesh, hsize_t cell_index);

    // build the Voronoi cell for seed `k` by clipping the bounding box against its K nearest
    // neighbours; on success atomically reserves face storage and emits geometry into mesh
    template <int K, int MAX_P, int MAX_T>
    HD void compute_single_voronoi_cell(int                 k,
                                        int                 seed_id,
                                        double*             d_stored_points,
                                        const knn_problem*  knn,
                                        Status*             stat,
                                        VMesh*              mesh,
                                        unsigned long long* face_offset,
                                        int*                overflow_flag);

} // namespace voronoi

#endif // CELL_H
