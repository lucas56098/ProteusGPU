#ifndef CELL_H
#define CELL_H

#include "../global/allvars.h"
#include "../knn/knn.h"
#include "geometry.h"
#include "voronoi.h"

namespace voronoi {

    // largest value representable in T, as a compile-time constant. Hand-rolled rather than
    // std::numeric_limits so it is usable from device code without pulling in <limits>.
    template <typename T> HD constexpr long long idx_max() {
        return (T)(-1) > 0 ? (long long)(T)(-1)                                 // unsigned: all bits set
                           : (long long)((1ULL << (8 * sizeof(T) - 1)) - 1ULL); // signed: 2^(n-1) - 1
    }

    // build a dual-graph vertex from DIMENSION plane indices. Replaces the make_uchar2 /
    // make_uchar3 calls so the same code serves both the uchar and int tiers.
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

    // per-cell convex-polytope state used during mesh generation. Size limits are template
    // parameters so a small "fast" tier and a full "slow" tier can coexist in the same TU.
    //
    // IDX/VERT select the index width. The GPU tiers instantiate <..., uchar, VERT_TYPE>,
    // which is the layout this struct has always had; the CPU fallback additionally
    // instantiates <..., int, BIG_VERT_TYPE> so a pathological cell can use more than 255
    // plane slots (see _BIG_MAX_P_ in global/globals.h for why that happens).
    //   IDX  = type of a plane/triangle index and of the counters
    //   VERT = DIMENSION-component vector of IDX; one dual-graph vertex = DIMENSION planes
    template <int MAX_P, int MAX_T, typename IDX, typename VERT> struct BasicConvexCell {
        HD BasicConvexCell(int p_seed, double* p_pts, Status* p_status, double p_buff);

        // VERT's components must be exactly IDX, otherwise plane indices silently truncate
        // on the triangle[] <-> boundary_next[] round trip (see cell.cu, first_boundary).
        static_assert(sizeof(decltype(VERT::x)) == sizeof(IDX), "VERT component width must equal IDX");
        // every plane index in [0, MAX_P) must be representable in IDX *and* must not
        // collide with END_OF_LIST. For unsigned IDX the sentinel is the max value, so
        // MAX_P <= max means the largest valid index (MAX_P - 1) stays below it.
        static_assert((long long)MAX_P <= (long long)idx_max<IDX>(), "MAX_P does not fit in IDX");
        static_assert((long long)MAX_T <= (long long)idx_max<IDX>(), "MAX_T does not fit in IDX");

        // "no such plane" marker for the boundary linked list. (IDX)(-1) is 255 for uchar --
        // bit-identical to the hardcoded 255 this used to be -- and -1 for int, which can
        // never alias a real plane index.
        static constexpr IDX END_OF_LIST = (IDX)(-1);

        // input arrays + per-cell context
        double* pts;
        double4 voro_seed;
        Status* status;
        double  buff; // bounding box covers [-buff, 1+buff]^d (set per-cell so plane_for can read it)

        // current plane / triangle counts
        IDX nb_v; // number of planes (= bounding box + clip planes added so far)
        IDX nb_t; // number of triangles (dual-graph vertices currently in the cell)
        IDX nb_r; // number of triangles tagged for removal during the current clip

        // plane -> source: -1 for bounding-box planes, otherwise the point id behind the
        // bisector. Holds point ids, NOT plane indices, so it stays int at every tier.
        int plane_vid[MAX_P];

        // dual-graph vertices: each triangle is a corner where DIMENSION planes intersect
        VERT triangle[MAX_T];

        // boundary loop (linked list head + next-pointers, used during compute_boundary)
        IDX first_boundary;
        IDX boundary_next[MAX_P];

        // plane equation for slot p, rebuilt on demand from plane_vid[p] (or fixed bounding-box
        // constants for p < 2*DIMENSION). Saves the 960 B/thread that used to live in a stored
        // half_plane[] array.
        HD double4 plane_for(int p) const;

        // clip the cell by the perpendicular bisector of (voro_seed, pts[vid])
        HD void clip_by_plane(int vid);

        // append a new plane slot for vid; returns the new slot index (or -1 on overflow)
        HD int new_halfplane(int vid);

        // is this triangle on the "wrong" side of eqn (and so should be removed by the clip)?
        HD bool vert_is_in_conflict(VERT v, double4 eqn) const;

        // build the boundary loop separating kept triangles from removed ones
        HD void compute_boundary();

        // create a new triangle from planes i, j (and k in 3D); writes into triangle[nb_t]
        HD void new_vertex(IDX i, IDX j, IDX k = 0);

        // every cell vertex fits in the sphere of radius ||last_neig - voro_seed|| / 2
        HD bool is_security_radius_reached(double4 last_neig) const;

        // squared distance from voro_seed to the farthest cell vertex, as num/denom (vertices are
        // homogeneous, so the ratio is only formed by callers that need a plain value)
        HD void max_vertex_r2_ratio(double* out_num, double* out_denom) const;

        // primal point of a dual-graph vertex (intersection of DIMENSION planes)
        HD double4 compute_vertex_point(VERT v, bool persp_divide = true) const;
    };

    // ConvexCell uses the slow-tier capacity. The fast-tier kernel instantiates
    // BasicConvexCell<_FAST_MAX_P_, _FAST_MAX_T_, uchar, VERT_TYPE> directly.
    // Both GPU tiers keep the 8-bit indices this struct has always used, so their layout
    // and generated code are unchanged by the addition of the IDX/VERT parameters.
    using ConvexCell = BasicConvexCell<_MAX_P_, _MAX_T_, uchar, VERT_TYPE>;

    // Wide tier: 32-bit indices, CPU fallback only. Its single odr-use is a plain host function
    // (try_build_cell_from_neighbours_as<BigConvexCell> in fallback.cu), so no kernel can reach
    // it and it never occupies a GPU thread's stack.
    //
    // Be precise about what that does and does not guarantee: nvcc's device pass parses the whole
    // TU and instantiates HD templates regardless of whether the odr-use came from a kernel or
    // from host code, so a device copy of these members IS emitted; being unreferenced, it is
    // dead-stripped by nvlink under -dc. The cost is device compile time and object size, not
    // occupancy -- an unreached function's ~82 KB frame constrains nothing. What this DOES mean
    // is that anything device-hostile must stay out of the wide path, and that raising
    // _BIG_MAX_P_/_BIG_MAX_T_ inflates emitted-then-discarded device code.
    using BigConvexCell = BasicConvexCell<_BIG_MAX_P_, _BIG_MAX_T_, int, BIG_VERT_TYPE>;

    // grow mesh's face buffer if `needed` exceeds the current face_capacity
    void ensure_face_capacity(VMesh* mesh, hsize_t needed);

    // walk the boundary of face p in dual-graph order, writing primal vertices into face_verts[].
    // Returns false if the face has fewer than DIMENSION vertices (= not a real face).
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD bool collect_face_vertices(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell,
                                  int                                             p,
                                  const double4*                                  vertices,
                                  double4*                                        face_verts,
                                  int*                                            n_face_verts);

    // number of planes that contribute a face (at least DIMENSION triangles reference them)
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD int count_cell_faces(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell);

    // emit cell volume + centroid + per-face data into the global mesh SoA arrays
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    // returns the number of faces actually written at mesh->face_ptr[cell_index]; may be less
    // than count_cell_faces(cell) on degenerate cells. Callers must store this in
    // face_counts[cell_index] -- see the definition in cell.cu for why.
    HD hsize_t extract_cell_all(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell, VMesh* mesh, hsize_t cell_index);

    // True iff no seed outside [data_lo, data_hi] can clip a cell centred on `seed` whose
    // bounding-sphere radius squared is r2_num / r2_denom. See the definition in cell.cu for
    // the 2R <= safe argument. data_hi == data_lo disables the check (single rank).
    HD bool cell_certified_within_data(
        double4 seed, double r2_num, double r2_denom, const double* data_lo, const double* data_hi);

    // build the Voronoi cell for seed `k` by clipping the bounding box against its K nearest
    // neighbours; on success atomically reserves face storage and emits geometry into mesh
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

#endif // CELL_H
