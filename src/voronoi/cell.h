#ifndef CELL_H
#define CELL_H

#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "geometry.h"
#include "voronoi.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

namespace voronoi {

    // struct used for mesh generation
    struct ConvexCell {
        HD ConvexCell(int p_seed, double* p_pts, Status* p_status);

        double* pts;
        double4 voro_seed;
        uchar   first_boundary;
        Status* status;
        uchar   nb_v;
        uchar   nb_t;
        uchar   nb_r;
        int     plane_vid[_MAX_P_]; // maps plane index to global point id (-1 for boundary planes)

        VERT_TYPE triangle[_MAX_T_];
        uchar     boundary_next[_MAX_P_];
        double4   half_plane[_MAX_P_];

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

    // put convex cell into VMesh struct
    void    ensure_face_capacity(VMesh* mesh, hsize_t needed);
    HD bool collect_face_vertices(
        const ConvexCell& cell, int p, const double4* vertices, double4* face_verts, int* n_face_verts);

    HD int  count_cell_faces(const ConvexCell& cell);
    HD void extract_cell_all(const ConvexCell& cell, VMesh* mesh, hsize_t cell_index);
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