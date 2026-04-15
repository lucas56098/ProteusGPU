#include "geometry.h"
#include "cell.h"
#include <cmath>
#include <iostream>

namespace voronoi {

#ifdef dim_2D
    // compute cell area using shoelace formula (2D only)
    double compute_cell_area_centroid_2d(const double4* vertices, int nb_t, double& cx, double& cy) {
        double mx = 0.0, my = 0.0;
        for (int i = 0; i < nb_t; i++) {
            mx += vertices[i].x;
            my += vertices[i].y;
        }
        mx /= nb_t;
        my /= nb_t;

        // sort by angle using cross-product comparison (insertion sort — GPU safe)
        int order[_MAX_T_];
        for (int i = 0; i < nb_t; i++) {
            order[i] = i;
        }
        for (int i = 1; i < nb_t; i++) {
            int    key = order[i];
            double kx = vertices[key].x - mx, ky = vertices[key].y - my;
            int    hk = (ky > 0.0) || (ky == 0.0 && kx > 0.0) ? 0 : 1;
            int    j  = i - 1;
            while (j >= 0) {
                int    oj  = order[j];
                double ojx = vertices[oj].x - mx, ojy = vertices[oj].y - my;
                int    hj = (ojy > 0.0) || (ojy == 0.0 && ojx > 0.0) ? 0 : 1;
                bool   should_swap;
                if (hj != hk) {
                    should_swap = hj > hk;
                } else {
                    double cross = ojx * ky - ojy * kx;
                    if (cross != 0.0)
                        should_swap = cross < 0.0;
                    else
                        should_swap = (ojx * ojx + ojy * ojy) > (kx * kx + ky * ky);
                }
                if (!should_swap) break;
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        double area2  = 0.0;
        double Cx_num = 0.0;
        double Cy_num = 0.0;
        for (int i = 0; i < nb_t; i++) {
            int          j     = (i + 1) % nb_t;
            const double xi    = vertices[order[i]].x;
            const double yi    = vertices[order[i]].y;
            const double xj    = vertices[order[j]].x;
            const double yj    = vertices[order[j]].y;
            const double cross = xi * yj - xj * yi;

            area2 += cross;
            Cx_num += (xi + xj) * cross;
            Cy_num += (yi + yj) * cross;
        }

        if (fabs(area2) > 1e-14) {
            cx = Cx_num / (3.0 * area2);
            cy = Cy_num / (3.0 * area2);
        }

        return 0.5 * fabs(area2);
    }
#endif

    // compute exact volume and centroid of a 3D Voronoi cell via tetrahedral decomposition
    double compute_cell_volume_centroid_3d(
        const ConvexCell& cell, const double4* vertices, double& cx, double& cy, double& cz) {
        double total_volume = 0.0;
        double wx = 0.0, wy = 0.0, wz = 0.0;

        double4 face_verts[_MAX_T_];
        int     n_fv;

        for (int p = 0; p < cell.nb_v; p++) {
            if (!collect_face_vertices(cell, p, vertices, face_verts, &n_fv)) continue;
            if (n_fv < DIMENSION) continue;

            // ensure outward-facing orientation
            double4 edge1      = minus4(face_verts[1], face_verts[0]);
            double4 edge2      = minus4(face_verts[2], face_verts[0]);
            double4 face_cross = cross3(edge1, edge2);
            double4 fc         = make_double4(0, 0, 0, 0);
            for (int i = 0; i < n_fv; i++) {
                fc.x += face_verts[i].x;
                fc.y += face_verts[i].y;
                fc.z += face_verts[i].z;
            }
            fc.x /= n_fv;
            fc.y /= n_fv;
            fc.z /= n_fv;
            double4 outward = minus4(fc, cell.voro_seed);
            if (dot3(face_cross, outward) < 0) {
                for (int lo = 0, hi = n_fv - 1; lo < hi; lo++, hi--) {
                    double4 tmp    = face_verts[lo];
                    face_verts[lo] = face_verts[hi];
                    face_verts[hi] = tmp;
                }
            }

            // decompose face into triangles, form tets with seed as apex
            for (int i = 1; i + 1 < n_fv; i++) {
                double4 a   = minus4(face_verts[0], cell.voro_seed);
                double4 b   = minus4(face_verts[i], cell.voro_seed);
                double4 c   = minus4(face_verts[i + 1], cell.voro_seed);
                double4 bxc = cross3(b, c);
                double  tv  = dot3(a, bxc) / 6.0;

                // tet centroid = (seed + v0 + vi + vi+1) / 4
                wx += tv * 0.25 * (cell.voro_seed.x + face_verts[0].x + face_verts[i].x + face_verts[i + 1].x);
                wy += tv * 0.25 * (cell.voro_seed.y + face_verts[0].y + face_verts[i].y + face_verts[i + 1].y);
                wz += tv * 0.25 * (cell.voro_seed.z + face_verts[0].z + face_verts[i].z + face_verts[i + 1].z);
                total_volume += tv;
            }
        }

        if (fabs(total_volume) > 1e-30) {
            cx = wx / total_volume;
            cy = wy / total_volume;
            cz = wz / total_volume;
        } else {
            cx = cell.voro_seed.x;
            cy = cell.voro_seed.y;
            cz = cell.voro_seed.z;
        }

        return fabs(total_volume);
    }

    // compute face measure (length in 2D, area in 3D)
    double compute_face_measure(double4* face_verts, int n_face_verts, double4 seed, double* cell_volume) {
        double face_measure = 0.0;

#ifdef dim_2D
        (void)seed;
        (void)cell_volume;
        double dx    = face_verts[1].x - face_verts[0].x;
        double dy    = face_verts[1].y - face_verts[0].y;
        face_measure = sqrt(dx * dx + dy * dy);
#else
        // ensure face vertices are oriented consistently (outward from seed)
        {
            double4 edge1      = minus4(face_verts[1], face_verts[0]);
            double4 edge2      = minus4(face_verts[2], face_verts[0]);
            double4 face_cross = cross3(edge1, edge2);
            double4 centroid   = make_double4(0, 0, 0, 0);
            for (int k = 0; k < n_face_verts; k++) {
                centroid.x += face_verts[k].x;
                centroid.y += face_verts[k].y;
                centroid.z += face_verts[k].z;
            }
            centroid.x /= n_face_verts;
            centroid.y /= n_face_verts;
            centroid.z /= n_face_verts;
            double4 outward = minus4(centroid, seed);
            if (dot3(face_cross, outward) < 0) {
                for (int lo = 0, hi = n_face_verts - 1; lo < hi; lo++, hi--) {
                    double4 tmp    = face_verts[lo];
                    face_verts[lo] = face_verts[hi];
                    face_verts[hi] = tmp;
                }
            }
        }

        // face area via fan triangulation from vertex 0
        double4 v0 = face_verts[0];
        for (int i = 1; i + 1 < n_face_verts; i++) {
            double4 edge1 = minus4(face_verts[i], v0);
            double4 edge2 = minus4(face_verts[i + 1], v0);
            double4 cr    = cross3(edge1, edge2);
            face_measure += 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
        }

        // contribute to cell volume using divergence theorem
        if (cell_volume) {
            for (int i = 1; i + 1 < n_face_verts; i++) {
                double4 a   = minus4(face_verts[0], seed);
                double4 b   = minus4(face_verts[i], seed);
                double4 c   = minus4(face_verts[i + 1], seed);
                double4 bxc = cross3(b, c);
                *cell_volume += dot3(a, bxc) / 6.0;
            }
        }
#endif

        (void)n_face_verts; // used only in 3D
        return face_measure;
    }

} // namespace voronoi
