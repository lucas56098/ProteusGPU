#ifndef CELL_H
#define CELL_H

#include "../global/allvars.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "voronoi.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

namespace voronoi {

    // struct used for mesh generation
    struct ConvexCell {
        ConvexCell(int p_seed, double* p_pts, Status* p_status);

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
        void clip_by_plane(int vid);
        int  new_halfplane(int vid);
        bool vert_is_in_conflict(VERT_TYPE v, double4 eqn) const;
        void compute_boundary();
        void new_vertex(uchar i, uchar j, uchar k = 0);

        // security radius check
        bool    is_security_radius_reached(double4 last_neig);
        double4 compute_vertex_point(VERT_TYPE v, bool persp_divide = true) const;
    };

    // put convex cell into VMesh struct
    void extract_cell_to_vmesh(ConvexCell& cell, VMesh* mesh, hsize_t cell_index);
    // helper to compute additional quantities needed in hydro
    double compute_cell_area_centroid_2d(const double4* vertices, int nb_t, double& cx, double& cy);
    double
         compute_cell_volume_centroid_3d(ConvexCell& cell, const double4* vertices, double& cx, double& cy, double& cz);
    void ensure_face_capacity(VMesh* mesh, hsize_t needed);
    bool
    collect_face_vertices(ConvexCell& cell, int p, const double4* vertices, double4* face_verts, int* n_face_verts);
    double compute_face_measure(double4* face_verts, int n_face_verts, double4 seed, double* cell_volume);
    void   store_face_data(VMesh*         mesh,
                           const double4* face_verts,
                           int            n_face_verts,
                           int            neighbor_id,
                           double         face_measure,
                           double4        seed,
                           double4        neighbor);

#ifdef CPU_DEBUG
    // two-pass parallel extraction:
    // pass 1: extract per-cell data (seeds/com/volumes) and count faces
    int extract_cell_percell_count(ConvexCell& cell, VMesh* mesh, hsize_t cell_index);
    // pass 2: extract face data directly into VMesh at known face_ptr offset
    void extract_cell_percell_faces(ConvexCell& cell, VMesh* mesh, hsize_t cell_index);
#endif

#ifdef DEBUG_MODE
    void ensure_edge_coords_capacity(VMesh* mesh, hsize_t needed_verts);
#endif

} // namespace voronoi

#endif // CELL_H