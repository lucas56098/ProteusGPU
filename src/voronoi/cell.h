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
        bool    is_security_radius_reached(double4 last_neig) const;
        double4 compute_vertex_point(VERT_TYPE v, bool persp_divide = true) const;
    };

    // put convex cell into VMesh struct
    void ensure_face_capacity(VMesh* mesh, hsize_t needed);
    bool collect_face_vertices(
        const ConvexCell& cell, int p, const double4* vertices, double4* face_verts, int* n_face_verts);

    // two-pass extraction (caller pre-computes vertices once, passes to both)
    // pass 1: extract per-cell data (seeds/com/volumes) and return face count
    int extract_cell_data(const ConvexCell& cell, const double4* vertices, VMesh* mesh, hsize_t cell_index);
    // pass 2: write face data into VMesh at face_ptr[cell_index] offset
    void extract_cell_faces(const ConvexCell& cell, const double4* vertices, VMesh* mesh, hsize_t cell_index);
    // write a single face at index fi (no num_faces increment)
    void write_face(VMesh*         mesh,
                    hsize_t        fi,
                    int            neighbor_id,
                    double         face_measure,
                    const double4* face_verts,
                    int            n_face_verts,
                    double4        seed,
                    double4        neighbor);

} // namespace voronoi

#endif // CELL_H