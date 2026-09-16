
// cell geometry (geometry.h)

#include "cell.h"
#include "geometry.h"
#include <cmath>

namespace voronoi {

#ifdef dim_2D
    // walks the 2D cell corner by corner and sums the triangles
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD double compute_cell_area_centroid_2d(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell,
                                            const double4_t*                                vertices,
                                            double&                                         cx,
                                            double&                                         cy) {
        const int nb_t = cell.nb_t;

        if (nb_t < 3) {
            cx = cell.voro_seed.x;
            cy = cell.voro_seed.y;
            return 0.0;
        }

        bool visited[MAX_T];
        for (int i = 0; i < nb_t; i++)
            visited[i] = false;
        visited[0] = true;
        int cur    = 0;

        double4_t first_pt = vertices[0];
        double4_t prev_pt  = first_pt;

        double area2 = 0.0, Cx_num = 0.0, Cy_num = 0.0;

        // next corner shares a line with this one
        for (int step = 1; step < nb_t; step++) {
            const VERT t_cur = cell.triangle[cur];
            int        next  = -1;
            for (int j = 0; j < nb_t; j++) {
                if (visited[j]) continue;
                const VERT t_j = cell.triangle[j];
                if (t_cur.x == t_j.x || t_cur.x == t_j.y || t_cur.y == t_j.x || t_cur.y == t_j.y) {
                    next = j;
                    break;
                }
            }
            if (next < 0) break;

            visited[next] = true;
            cur           = next;

            const double4_t cur_pt = vertices[next];
            const double    cross  = prev_pt.x * cur_pt.y - cur_pt.x * prev_pt.y;
            area2 += cross;
            Cx_num += (prev_pt.x + cur_pt.x) * cross;
            Cy_num += (prev_pt.y + cur_pt.y) * cross;
            prev_pt = cur_pt;
        }

        // close the loop
        const double cross = prev_pt.x * first_pt.y - first_pt.x * prev_pt.y;
        area2 += cross;
        Cx_num += (prev_pt.x + first_pt.x) * cross;
        Cy_num += (prev_pt.y + first_pt.y) * cross;

        if (fabs(area2) > 1e-14) {
            cx = Cx_num / (3.0 * area2);
            cy = Cy_num / (3.0 * area2);
        }

        return 0.5 * fabs(area2);
    }
#endif

    // turns the corner order around if the normal points at the seed
    HD void orient_face_outward(double4_t* face_verts, int n_fv, double4_t seed) {

        const double4_t edge1      = minus4(face_verts[1], face_verts[0]);
        const double4_t edge2      = minus4(face_verts[2], face_verts[0]);
        const double4_t face_cross = cross3(edge1, edge2);

        double4_t fc = make_double4_t(0, 0, 0, 0);
        for (int i = 0; i < n_fv; i++) {
            fc.x += face_verts[i].x;
            fc.y += face_verts[i].y;
            fc.z += face_verts[i].z;
        }
        const double inv_nfv = 1.0 / n_fv;
        fc.x *= inv_nfv;
        fc.y *= inv_nfv;
        fc.z *= inv_nfv;

        const double4_t outward = minus4(fc, seed);
        if (dot3(face_cross, outward) < 0) {
            for (int lo = 0, hi = n_fv - 1; lo < hi; lo++, hi--) {
                double4_t tmp  = face_verts[lo];
                face_verts[lo] = face_verts[hi];
                face_verts[hi] = tmp;
            }
        }
    }

    // face as a fan of triangles, each with the seed a tetrahedron
    HD void compute_face_area_and_volume_centroid(const double4_t* face_verts,
                                                  int              n_fv,
                                                  double4_t        seed,
                                                  double&          face_area,
                                                  double&          vol_accum,
                                                  double&          wx_accum,
                                                  double&          wy_accum,
                                                  double&          wz_accum) {
        face_area          = 0.0;
        const double4_t v0 = face_verts[0];

        for (int i = 1; i + 1 < n_fv; i++) {
            const double4_t e1 = minus4(face_verts[i], v0);
            const double4_t e2 = minus4(face_verts[i + 1], v0);
            const double4_t cr = cross3(e1, e2);
            face_area += 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);

            const double4_t a   = minus4(v0, seed);
            const double4_t b   = minus4(face_verts[i], seed);
            const double4_t c   = minus4(face_verts[i + 1], seed);
            const double4_t bxc = cross3(b, c);
            const double    tv  = dot3(a, bxc) * (1.0 / 6.0);

            wx_accum += tv * 0.25 * (seed.x + v0.x + face_verts[i].x + face_verts[i + 1].x);
            wy_accum += tv * 0.25 * (seed.y + v0.y + face_verts[i].y + face_verts[i + 1].y);
            wz_accum += tv * 0.25 * (seed.z + v0.z + face_verts[i].z + face_verts[i + 1].z);
            vol_accum += tv;
        }
    }

    // face area, length in 2D; adds the tetrahedra if cell_volume is given
    HD double compute_face_measure(double4_t* face_verts, int n_face_verts, double4_t seed, double* cell_volume) {
        double face_measure = 0.0;

#ifdef dim_2D
        (void)seed;
        (void)cell_volume;
        const double dx = face_verts[1].x - face_verts[0].x;
        const double dy = face_verts[1].y - face_verts[0].y;
        face_measure    = sqrt(dx * dx + dy * dy);
#else
        {
            const double4_t edge1      = minus4(face_verts[1], face_verts[0]);
            const double4_t edge2      = minus4(face_verts[2], face_verts[0]);
            const double4_t face_cross = cross3(edge1, edge2);

            double4_t centroid = make_double4_t(0, 0, 0, 0);
            for (int k = 0; k < n_face_verts; k++) {
                centroid.x += face_verts[k].x;
                centroid.y += face_verts[k].y;
                centroid.z += face_verts[k].z;
            }
            centroid.x /= n_face_verts;
            centroid.y /= n_face_verts;
            centroid.z /= n_face_verts;

            const double4_t outward = minus4(centroid, seed);
            if (dot3(face_cross, outward) < 0) {
                for (int lo = 0, hi = n_face_verts - 1; lo < hi; lo++, hi--) {
                    double4_t tmp  = face_verts[lo];
                    face_verts[lo] = face_verts[hi];
                    face_verts[hi] = tmp;
                }
            }
        }

        const double4_t v0 = face_verts[0];
        for (int i = 1; i + 1 < n_face_verts; i++) {
            const double4_t edge1 = minus4(face_verts[i], v0);
            const double4_t edge2 = minus4(face_verts[i + 1], v0);
            const double4_t cr    = cross3(edge1, edge2);
            face_measure += 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
        }

        if (cell_volume) {
            for (int i = 1; i + 1 < n_face_verts; i++) {
                const double4_t a   = minus4(face_verts[0], seed);
                const double4_t b   = minus4(face_verts[i], seed);
                const double4_t c   = minus4(face_verts[i + 1], seed);
                const double4_t bxc = cross3(b, c);
                *cell_volume += dot3(a, bxc) / 6.0;
            }
        }
#endif

        (void)n_face_verts;
        return face_measure;
    }

} // namespace voronoi
