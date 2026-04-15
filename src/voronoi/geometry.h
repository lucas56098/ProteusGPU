#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "../global/allvars.h"
#include <cmath>

namespace voronoi {

    struct ConvexCell; // forward declaration (defined in cell.h)

    // --- Cell volume/area and centroid ---
    double compute_cell_area_centroid_2d(const double4* vertices, int nb_t, double& cx, double& cy);
    double compute_cell_volume_centroid_3d(
        const ConvexCell& cell, const double4* vertices, double& cx, double& cy, double& cz);

    // --- Face geometry ---
    double compute_face_measure(double4* face_verts, int n_face_verts, double4 seed, double* cell_volume);

    // Compute face centroid from face vertex coordinates.
    // 2D: midpoint of the 2 edge endpoints. 3D: area-weighted centroid via fan triangulation.
    inline void
    compute_face_centroid(const double4* face_verts, int n_face_verts, double& fmx, double& fmy, double& fmz) {
#ifdef dim_2D
        (void)n_face_verts;
        (void)fmz;
        fmx = 0.5 * (face_verts[0].x + face_verts[1].x);
        fmy = 0.5 * (face_verts[0].y + face_verts[1].y);
#else
        double total_area = 0.0;
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (int i = 1; i + 1 < n_face_verts; i++) {
            double4 e1     = minus4(face_verts[i], face_verts[0]);
            double4 e2     = minus4(face_verts[i + 1], face_verts[0]);
            double4 cr     = cross3(e1, e2);
            double  t_area = 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
            cx += t_area * (face_verts[0].x + face_verts[i].x + face_verts[i + 1].x) / 3.0;
            cy += t_area * (face_verts[0].y + face_verts[i].y + face_verts[i + 1].y) / 3.0;
            cz += t_area * (face_verts[0].z + face_verts[i].z + face_verts[i + 1].z) / 3.0;
            total_area += t_area;
        }
        if (total_area > 0.0) {
            fmx = cx / total_area;
            fmy = cy / total_area;
            fmz = cz / total_area;
        } else {
            fmx = face_verts[0].x;
            fmy = face_verts[0].y;
            fmz = face_verts[0].z;
        }
#endif
    }

} // namespace voronoi

#endif // GEOMETRY_H
