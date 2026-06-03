#include "cell.h"
#include "geometry.h"
#include <cmath>

namespace voronoi {

#ifdef dim_2D
    // 2D: walk the cell's polygon boundary (adjacency via shared plane index) and accumulate
    // area + centroid via shoelace
    template <int MAX_P, int MAX_T>
    HD double compute_cell_area_centroid_2d(const BasicConvexCell<MAX_P, MAX_T>& cell,
                                            const double4*                       vertices,
                                            double&                              cx,
                                            double&                              cy) {
        const int nb_t = cell.nb_t;

        // degenerate cell: return seed as centroid, zero area
        if (nb_t < 3) {
            cx = cell.voro_seed.x;
            cy = cell.voro_seed.y;
            return 0.0;
        }

        // walk the polygon boundary: two vertices (uchar2) are adjacent if they share a plane index
        bool visited[MAX_P]; // nb_t <= MAX_P in 2D
        for (int i = 0; i < nb_t; i++)
            visited[i] = false;
        visited[0] = true;
        int cur    = 0;

        double4 first_pt = vertices[0];
        double4 prev_pt  = first_pt;

        // shoelace accumulators
        double area2 = 0.0, Cx_num = 0.0, Cy_num = 0.0;

        for (int step = 1; step < nb_t; step++) {
            // find the next unvisited vertex that shares a plane with cur
            const VERT_TYPE t_cur = cell.triangle[cur];
            int             next  = -1;
            for (int j = 0; j < nb_t; j++) {
                if (visited[j]) continue;
                const VERT_TYPE t_j = cell.triangle[j];
                if (t_cur.x == t_j.x || t_cur.x == t_j.y || t_cur.y == t_j.x || t_cur.y == t_j.y) {
                    next = j;
                    break;
                }
            }
            if (next < 0) break;

            visited[next] = true;
            cur           = next;

            // accumulate shoelace edge contribution prev -> cur
            const double4 cur_pt = vertices[next];
            const double  cross  = prev_pt.x * cur_pt.y - cur_pt.x * prev_pt.y;
            area2 += cross;
            Cx_num += (prev_pt.x + cur_pt.x) * cross;
            Cy_num += (prev_pt.y + cur_pt.y) * cross;
            prev_pt = cur_pt;
        }

        // close the polygon (last vertex -> first vertex)
        const double cross = prev_pt.x * first_pt.y - first_pt.x * prev_pt.y;
        area2 += cross;
        Cx_num += (prev_pt.x + first_pt.x) * cross;
        Cy_num += (prev_pt.y + first_pt.y) * cross;

        // normalise centroid; leave seed-as-fallback if area is degenerate
        if (fabs(area2) > 1e-14) {
            cx = Cx_num / (3.0 * area2);
            cy = Cy_num / (3.0 * area2);
        }

        return 0.5 * fabs(area2);
    }
#endif

    // reverse face_verts[] in place if its winding faces inward (toward the seed)
    HD void orient_face_outward(double4* face_verts, int n_fv, double4 seed) {

        // signed normal from the first triangle (edge1 x edge2)
        const double4 edge1      = minus4(face_verts[1], face_verts[0]);
        const double4 edge2      = minus4(face_verts[2], face_verts[0]);
        const double4 face_cross = cross3(edge1, edge2);

        // face centroid (unweighted average of the n_fv vertices)
        double4 fc = make_double4(0, 0, 0, 0);
        for (int i = 0; i < n_fv; i++) {
            fc.x += face_verts[i].x;
            fc.y += face_verts[i].y;
            fc.z += face_verts[i].z;
        }
        const double inv_nfv = 1.0 / n_fv;
        fc.x *= inv_nfv;
        fc.y *= inv_nfv;
        fc.z *= inv_nfv;

        // outward direction = centroid - seed; reverse winding if normal points inward
        const double4 outward = minus4(fc, seed);
        if (dot3(face_cross, outward) < 0) {
            for (int lo = 0, hi = n_fv - 1; lo < hi; lo++, hi--) {
                double4 tmp    = face_verts[lo];
                face_verts[lo] = face_verts[hi];
                face_verts[hi] = tmp;
            }
        }
    }

    // 3D: fan-triangulate from v0; accumulate face area, cell volume, and weighted centroid.
    // Volume is computed via the divergence theorem on (seed, v0, v_i, v_{i+1}) tetrahedra.
    HD void compute_face_area_and_volume_centroid(const double4* face_verts,
                                                  int            n_fv,
                                                  double4        seed,
                                                  double&        face_area,
                                                  double&        vol_accum,
                                                  double&        wx_accum,
                                                  double&        wy_accum,
                                                  double&        wz_accum) {
        face_area              = 0.0;
        const double4 v0       = face_verts[0];

        for (int i = 1; i + 1 < n_fv; i++) {
            // triangle area = 0.5 * |edge1 x edge2|
            const double4 e1 = minus4(face_verts[i], v0);
            const double4 e2 = minus4(face_verts[i + 1], v0);
            const double4 cr = cross3(e1, e2);
            face_area += 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);

            // signed tetrahedron volume V = (a . (b x c)) / 6
            const double4 a   = minus4(v0, seed);
            const double4 b   = minus4(face_verts[i], seed);
            const double4 c   = minus4(face_verts[i + 1], seed);
            const double4 bxc = cross3(b, c);
            const double  tv  = dot3(a, bxc) * (1.0 / 6.0);

            // weighted centroid contribution: tv * (average of the 4 tet vertices)
            wx_accum += tv * 0.25 * (seed.x + v0.x + face_verts[i].x + face_verts[i + 1].x);
            wy_accum += tv * 0.25 * (seed.y + v0.y + face_verts[i].y + face_verts[i + 1].y);
            wz_accum += tv * 0.25 * (seed.z + v0.z + face_verts[i].z + face_verts[i + 1].z);
            vol_accum += tv;
        }
    }

    // face measure = edge length in 2D, face area in 3D; in 3D also adds the face's
    // contribution to the cell volume via the divergence theorem (if cell_volume != nullptr)
    HD double compute_face_measure(double4* face_verts, int n_face_verts, double4 seed, double* cell_volume) {
        double face_measure = 0.0;

#ifdef dim_2D
        // 2D: face is a single edge, measure is its length
        (void)seed;
        (void)cell_volume;
        const double dx = face_verts[1].x - face_verts[0].x;
        const double dy = face_verts[1].y - face_verts[0].y;
        face_measure    = sqrt(dx * dx + dy * dy);
#else
        // 3D: ensure face vertices are oriented outward from seed before triangulating
        {
            const double4 edge1      = minus4(face_verts[1], face_verts[0]);
            const double4 edge2      = minus4(face_verts[2], face_verts[0]);
            const double4 face_cross = cross3(edge1, edge2);

            double4 centroid = make_double4(0, 0, 0, 0);
            for (int k = 0; k < n_face_verts; k++) {
                centroid.x += face_verts[k].x;
                centroid.y += face_verts[k].y;
                centroid.z += face_verts[k].z;
            }
            centroid.x /= n_face_verts;
            centroid.y /= n_face_verts;
            centroid.z /= n_face_verts;

            const double4 outward = minus4(centroid, seed);
            if (dot3(face_cross, outward) < 0) {
                for (int lo = 0, hi = n_face_verts - 1; lo < hi; lo++, hi--) {
                    double4 tmp    = face_verts[lo];
                    face_verts[lo] = face_verts[hi];
                    face_verts[hi] = tmp;
                }
            }
        }

        // 3D: face area via fan triangulation from vertex 0
        const double4 v0 = face_verts[0];
        for (int i = 1; i + 1 < n_face_verts; i++) {
            const double4 edge1 = minus4(face_verts[i], v0);
            const double4 edge2 = minus4(face_verts[i + 1], v0);
            const double4 cr    = cross3(edge1, edge2);
            face_measure += 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
        }

        // 3D: contribute this face to the cell volume via the divergence theorem
        if (cell_volume) {
            for (int i = 1; i + 1 < n_face_verts; i++) {
                const double4 a   = minus4(face_verts[0], seed);
                const double4 b   = minus4(face_verts[i], seed);
                const double4 c   = minus4(face_verts[i + 1], seed);
                const double4 bxc = cross3(b, c);
                *cell_volume += dot3(a, bxc) / 6.0;
            }
        }
#endif

        (void)n_face_verts; // used only in 3D
        return face_measure;
    }

} // namespace voronoi
