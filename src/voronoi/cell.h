#ifndef CELL_H
#define CELL_H

#include <string>
#include "../global/allvars.h"
#include "../knn/knn.h"
#include "../io/input.h"
#include "../io/output.h"
#include "voronoi.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace voronoi {

    // struct used for mesh generation
    struct ConvexCell {
        ConvexCell(int p_seed, double* p_pts, Status* p_status);

        double *pts;
        int voro_id;
        double4 voro_seed;
        uchar first_boundary;
        Status* status;
        uchar nb_v;
        uchar nb_t;
        uchar nb_r;
        int plane_vid[_MAX_P_]; // maps plane index to global point id (-1 for boundary planes)

        // clipping functions
        void clip_by_plane(int vid);
            int new_halfplane(int vid);
            bool vert_is_in_conflict(VERT_TYPE v, double4 eqn) const;
            void compute_boundary();
            void new_vertex(uchar i, uchar j, uchar k = 0);
        
        // security radius check
        bool is_security_radius_reached(double4 last_neig);
            double4 compute_vertex_point(VERT_TYPE v, bool persp_divide=true) const;
        
    };

    // put convex cell into VMesh struct
    void extract_cell_to_vmesh(ConvexCell& cell, VMesh* mesh, hsize_t cell_index, hsize_t& face_capacity);
        // helper to compute additional quantities needed in hydro
        double compute_cell_area_2d(const std::vector<double4>& vertices, int nb_t);
        void ensure_face_capacity(VMesh* mesh, hsize_t& face_capacity, hsize_t needed);
        bool collect_face_vertices(ConvexCell& cell, int p, const std::vector<double4>& vertices, std::vector<double4>& face_verts);
        double3 compute_face_normal(int p);
        double compute_face_measure(std::vector<double4>& face_verts, double4 seed, double* cell_volume);
        void store_face_data(VMesh* mesh, const std::vector<double4>& face_verts, int neighbor_id, double3 normal, double face_measure);

} // namespace voronoi

#endif // CELL_H