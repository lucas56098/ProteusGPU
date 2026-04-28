#ifndef CELL_H
#define CELL_H

#include "../global/allvars.h"
#include "../knn/knn.h"
#include "geometry.h"
#include "voronoi.h"

namespace voronoi {

    // struct used for mesh generation. Size limits are template parameters so
    // a small "fast" tier and a full "slow" tier can coexist in the same TU.
    template <int MAX_P, int MAX_T> struct BasicConvexCell {
        HD BasicConvexCell(int p_seed, double* p_pts, Status* p_status);

        double* pts;
        double4 voro_seed;
        uchar   first_boundary;
        Status* status;
        uchar   nb_v;
        uchar   nb_t;
        uchar   nb_r;
        int     plane_vid[MAX_P]; // maps plane index to global point id (-1 for boundary planes)

        VERT_TYPE triangle[MAX_T];
        uchar     boundary_next[MAX_P];
        double4   half_plane[MAX_P];

        // clipping functions
        HD void clip_by_plane(int vid);
        HD int  new_halfplane(int vid);
        HD bool vert_is_in_conflict(VERT_TYPE v, double4 eqn) const;
        HD void compute_boundary();
        HD void new_vertex(uchar i, uchar j, uchar k = 0);

        // security radius check
        HD bool    is_security_radius_reached(double4 last_neig) const;
        HD double4 compute_vertex_point(VERT_TYPE v, bool persp_divide = true) const;
    };

    // ConvexCell uses the slow-tier capacity. The fast-tier kernel instantiates
    // BasicConvexCell<_FAST_MAX_P_, _FAST_MAX_T_> directly.
    using ConvexCell = BasicConvexCell<_MAX_P_, _MAX_T_>;

    // put convex cell into VMesh struct
    void ensure_face_capacity(VMesh* mesh, hsize_t needed);

    template <int MAX_P, int MAX_T>
    HD bool collect_face_vertices(const BasicConvexCell<MAX_P, MAX_T>& cell,
                                  int                                  p,
                                  const double4*                       vertices,
                                  double4*                             face_verts,
                                  int*                                 n_face_verts);

    template <int MAX_P, int MAX_T> HD int count_cell_faces(const BasicConvexCell<MAX_P, MAX_T>& cell);

    template <int MAX_P, int MAX_T>
    HD void extract_cell_all(const BasicConvexCell<MAX_P, MAX_T>& cell, VMesh* mesh, hsize_t cell_index);

    HD void write_face(VMesh*         mesh,
                       hsize_t        fi,
                       int            neighbor_id,
                       double         face_measure,
                       const double4* face_verts,
                       int            n_face_verts,
                       double4        seed,
                       double4        neighbor);

} // namespace voronoi

#endif // CELL_H
