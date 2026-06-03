#include "cell.h"
#include "geometry.h"
#include "voronoi.h"
#include <cmath>
#include <iostream>

namespace voronoi {

    // forward declarations
    HD static void write_face(VMesh*         mesh,
                              hsize_t        fi,
                              int            neighbor_id,
                              double         face_measure,
                              const double4* face_verts,
                              int            n_face_verts,
                              double4        seed,
                              double4        neighbor);

    static const uchar END_OF_LIST = 255;

    HD inline uchar& ith_plane(VERT_TYPE* triangles, uchar t, int i) {
        return reinterpret_cast<uchar*>(&(triangles[t]))[i];
    }

    HD inline uchar ith_plane(const VERT_TYPE* triangles, uchar t, int i) {
        return reinterpret_cast<const uchar*>(&(triangles[t]))[i];
    }

    HD static inline bool vert_references_plane(const VERT_TYPE* triangles, int t_idx, uchar p) {
        for (int d = 0; d < DIMENSION; d++) {
            if (ith_plane(triangles, (uchar)t_idx, d) == p) return true;
        }
        return false;
    }

    // ============================================================
    // Constructor: initialize convex cell as bounding box
    // ============================================================

    // initialize the convex cell to the unit-box bounding cell (eps margin against degeneracies).
    // The bounding-box plane equations are no longer stored — plane_for() returns them on demand
    // for the first 2*DIMENSION slots.
    template <int MAX_P, int MAX_T>
    HD BasicConvexCell<MAX_P, MAX_T>::BasicConvexCell(int p_seed, double* p_pts, Status* p_status, double p_buff) {

        pts  = p_pts;
        buff = p_buff;

        // boundary linked list and plane->point-id map start empty
        first_boundary = END_OF_LIST;
        for (int i = 0; i < MAX_P; i++) {
            boundary_next[i] = END_OF_LIST;
            plane_vid[i]     = -1;
        }

        status  = p_status;
        *status = success;

        voro_seed = point_from_ptr(pts + DIMENSION * p_seed);

        // dual-graph vertices: each "triangle" is a corner of the box (intersection of D planes)
#ifdef dim_2D
        triangle[0] = make_uchar2(2, 0);
        triangle[1] = make_uchar2(1, 2);
        triangle[2] = make_uchar2(3, 1);
        triangle[3] = make_uchar2(0, 3);
        nb_v        = 4;
        nb_t        = 4;
#else
        triangle[0] = make_uchar3(2, 5, 0);
        triangle[1] = make_uchar3(5, 3, 0);
        triangle[2] = make_uchar3(1, 5, 2);
        triangle[3] = make_uchar3(5, 1, 3);
        triangle[4] = make_uchar3(4, 2, 0);
        triangle[5] = make_uchar3(4, 0, 3);
        triangle[6] = make_uchar3(2, 4, 1);
        triangle[7] = make_uchar3(4, 3, 1);
        nb_v        = 6;
        nb_t        = 8;
#endif
    }

    // ============================================================
    // Main operations (clip, security radius, extraction)
    // ============================================================

    // clip the cell by the perpendicular bisector of (seed, point[vid])
    template <int MAX_P, int MAX_T> HD void BasicConvexCell<MAX_P, MAX_T>::clip_by_plane(int vid) {

        // append the new bisector plane; bail if we're out of plane slots
        int cur_v = new_halfplane(vid);
        if (*status == vertex_overflow) { return; }

        // partition: move triangles on the wrong side of the new plane to the tail of triangle[]
        double4 eqn = plane_for(cur_v);
        nb_r        = 0;

        int i = 0;
        while (i < nb_t) {
            if (vert_is_in_conflict(triangle[i], eqn)) {
                nb_t--;
                VERT_TYPE tmp  = triangle[i];
                triangle[i]    = triangle[nb_t];
                triangle[nb_t] = tmp;
                nb_r++;
            } else {
                i++;
            }
        }
        if (*status == needs_exact_predicates) { return; }

        // no conflict -> plane doesn't cut the cell, roll back the append
        if (nb_r == 0) {
            nb_v--;
            return;
        }

        // build the boundary loop separating kept triangles from removed ones
        compute_boundary();
        if (*status != success) { return; }
        if (first_boundary == END_OF_LIST) { return; }

        // sew new triangles along the boundary loop, anchored on the new plane
        uchar cir = first_boundary;
        do {
#ifdef dim_2D
            new_vertex(cur_v, cir);
#else
            new_vertex(cur_v, cir, boundary_next[cir]);
#endif
            if (*status != success) return;
            cir = boundary_next[cir];
        } while (cir != first_boundary);
    }

    // returns true iff every cell vertex fits in the sphere of radius ||last_neig - seed||/2
    template <int MAX_P, int MAX_T>
    HD bool BasicConvexCell<MAX_P, MAX_T>::is_security_radius_reached(double4 last_neig) const {
        double max_num   = 0.0;
        double max_denom = 1.0;

        // max squared distance from seed to any vertex (kept in projective num/denom form)
        for (int i = 0; i < nb_t; i++) {
            double4 pc = compute_vertex_point(triangle[i], false);
            double  dx = pc.x - voro_seed.x * pc.w;
            double  dy = pc.y - voro_seed.y * pc.w;
#ifdef dim_3D
            double dz  = pc.z - voro_seed.z * pc.w;
            double num = dx * dx + dy * dy + dz * dz;
#else
            double num = dx * dx + dy * dy;
#endif
            double denom = pc.w * pc.w;
            if (num * max_denom > max_num * denom) {
                max_num   = num;
                max_denom = denom;
            }
        }

        // d2/4 > max_vertex_d2, rearranged to avoid division
        double4 diff = minus4(last_neig, voro_seed);
        double  d2   = dot3(diff, diff);
        return (d2 * max_denom > 4.0 * max_num);
    }

    // ============================================================
    // Cell-to-mesh extraction
    // ============================================================

    // a plane contributes a face only if at least DIMENSION triangles reference it
    template <int MAX_P, int MAX_T> HD int count_cell_faces(const BasicConvexCell<MAX_P, MAX_T>& cell) {
        int count = 0;
        for (int p = 0; p < cell.nb_v; p++) {
            int refs = 0;
            for (int i = 0; i < cell.nb_t; i++) {
                if (vert_references_plane(cell.triangle, i, (uchar)p)) {
                    refs++;
                    if (refs >= DIMENSION) {
                        count++;
                        break;
                    }
                }
            }
        }
        return count;
    }

    // emit cell volume, centroid, and per-face data into the global mesh arrays
    template <int MAX_P, int MAX_T>
    HD void extract_cell_all(const BasicConvexCell<MAX_P, MAX_T>& cell, VMesh* mesh, hsize_t cell_index) {
        double3 seed            = {cell.voro_seed.x, cell.voro_seed.y, cell.voro_seed.z};
        mesh->seeds[cell_index] = seed;

#ifdef dim_2D
        // resolve dual triangles into primal polygon vertices
        double4 vertices_2d[MAX_P];
        for (int vi = 0; vi < cell.nb_t; vi++)
            vertices_2d[vi] = cell.compute_vertex_point(cell.triangle[vi], true);

        // area + centroid via shoelace
        double cx                 = cell.voro_seed.x;
        double cy                 = cell.voro_seed.y;
        mesh->volumes[cell_index] = compute_cell_area_centroid_2d(cell, vertices_2d, cx, cy);
        mesh->com[cell_index]     = {cx, cy, 0.0};

        // emit one edge per plane that reached the polygon
        hsize_t fi = mesh->face_ptr[cell_index];
        double4 face_verts[2];
        int     n_fv;
        for (int p = 0; p < cell.nb_v; p++) {
            if (!collect_face_vertices(cell, p, vertices_2d, face_verts, &n_fv)) continue;
            double  face_measure = compute_face_measure(face_verts, n_fv, cell.voro_seed, nullptr);
            int     neighbor_id  = cell.plane_vid[p];
            double4 neighbor     = make_double4(0.0, 0.0, 0.0, 0.0);
            if (neighbor_id >= 0) { neighbor = point_from_ptr(cell.pts + DIMENSION * neighbor_id); }
            write_face(mesh, fi, neighbor_id, face_measure, face_verts, n_fv, cell.voro_seed, neighbor);
            fi++;
        }
#else
        // 3D: fan-triangulate each face; volume via divergence theorem on (seed,v0,vi,vi+1) tets
        double  total_volume = 0.0;
        double  wx = 0.0, wy = 0.0, wz = 0.0;
        hsize_t fi = mesh->face_ptr[cell_index];

        double4 face_verts[MAX_T];

        for (int p = 0; p < cell.nb_v; p++) {
            // gather triangles touching plane p — these are the corners of face p
            int face_vert_indices[MAX_T];
            int n_fvi = 0;
            for (int i = 0; i < cell.nb_t; i++) {
                if (vert_references_plane(cell.triangle, i, (uchar)p)) { face_vert_indices[n_fvi++] = i; }
            }
            if (n_fvi < DIMENSION) continue;

            // walk vertices around the face: each adjacent pair shares a non-p plane
            int  ordered[MAX_T];
            bool used[MAX_T];
            for (int k = 0; k < n_fvi; k++)
                used[k] = false;
            ordered[0]    = face_vert_indices[0];
            used[0]       = true;
            int n_ordered = 1;

            for (int step = 1; step < n_fvi; step++) {
                int   last = ordered[n_ordered - 1];
                uchar others_last[DIMENSION - 1];
                int   cnt = 0;
                for (int d = 0; d < DIMENSION; d++) {
                    uchar pl = ith_plane(cell.triangle, (uchar)last, d);
                    if (pl != (uchar)p) others_last[cnt++] = pl;
                }
                bool found = false;
                for (int j = 0; j < n_fvi; j++) {
                    if (used[j]) continue;
                    int candidate = face_vert_indices[j];
                    for (int o = 0; o < DIMENSION - 1; o++) {
                        if (vert_references_plane(cell.triangle, candidate, others_last[o])) {
                            ordered[n_ordered++] = candidate;
                            used[j]              = true;
                            found                = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                if (!found) break;
            }
            if (n_ordered < DIMENSION) continue;

            int n_fv = n_ordered;

            // dual triangles -> primal coordinates
            for (int k = 0; k < n_fv; k++) {
                face_verts[k] = cell.compute_vertex_point(cell.triangle[ordered[k]], true);
            }

            orient_face_outward(face_verts, n_fv, cell.voro_seed);

            double face_measure = 0.0;
            compute_face_area_and_volume_centroid(
                face_verts, n_fv, cell.voro_seed, face_measure, total_volume, wx, wy, wz);

            int     neighbor_id = cell.plane_vid[p];
            double4 neighbor    = make_double4(0.0, 0.0, 0.0, 0.0);
            if (neighbor_id >= 0) { neighbor = point_from_ptr(cell.pts + DIMENSION * neighbor_id); }
            write_face(mesh, fi, neighbor_id, face_measure, face_verts, n_fv, cell.voro_seed, neighbor);
            fi++;
        }

        if (fabs(total_volume) > 1e-30) {
            double inv_vol        = 1.0 / total_volume;
            mesh->com[cell_index] = {wx * inv_vol, wy * inv_vol, wz * inv_vol};
        } else {
            mesh->com[cell_index] = seed;
        }
        mesh->volumes[cell_index] = fabs(total_volume);
#endif
    }

    void ensure_face_capacity(VMesh* mesh, hsize_t needed) {
        if (needed <= mesh->face_capacity) return;
        std::cerr << "VORONOI: Error! face count " << needed << " exceeds pre-allocated face capacity "
                  << mesh->face_capacity << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
        exit(EXIT_FAILURE);
    }

    // ============================================================
    // Clipping helpers
    // ============================================================

    template <int MAX_P, int MAX_T> HD int BasicConvexCell<MAX_P, MAX_T>::new_halfplane(int vid) {
        if (nb_v >= MAX_P) {
            *status = vertex_overflow;
            return -1;
        }
        plane_vid[nb_v] = vid;
        nb_v++;
        return nb_v - 1;
    }

    // rebuild plane equation for slot p — bounding-box constants for p < 2*DIMENSION,
    // otherwise the perpendicular bisector of (voro_seed, pts[plane_vid[p]]).
    template <int MAX_P, int MAX_T>
    HD double4 BasicConvexCell<MAX_P, MAX_T>::plane_for(int p) const {
        if (p < 2 * DIMENSION) {
            // bounding box [-buff, 1+buff]^d (the periodic ghost copies extend out to those limits,
            // so the bounding planes have to enclose them). plane n.x*x+n.y*y+n.z*z+w >= 0 form:
            //   min plane:  x >= -buff  -> ( 1, 0, 0,  buff + eps)
            //   max plane:  x <=  1+buff -> (-1, 0, 0, 1 + buff + eps)
            constexpr double eps    = 1e-14;
            const double     w_min  = buff + eps;
            const double     w_max  = 1.0 + buff + eps;
            switch (p) {
                case 0: return make_double4(1.0, 0.0, 0.0, w_min);          // -xmin
                case 1: return make_double4(-1.0, 0.0, 0.0, w_max);         //  xmax
                case 2: return make_double4(0.0, 1.0, 0.0, w_min);          // -ymin
                case 3: return make_double4(0.0, -1.0, 0.0, w_max);         //  ymax
#ifdef dim_3D
                case 4: return make_double4(0.0, 0.0, 1.0, w_min);          // -zmin
                case 5: return make_double4(0.0, 0.0, -1.0, w_max);         //  zmax
#endif
            }
        }
        double4 B    = point_from_ptr(pts + DIMENSION * plane_vid[p]);
        double4 dir  = minus4(voro_seed, B);
        double4 ave2 = plus4(voro_seed, B);
        double  dot  = dot3(ave2, dir);
        return make_double4(dir.x, dir.y, dir.z, -dot * 0.5);
    }

    template <int MAX_P, int MAX_T>
    HD bool BasicConvexCell<MAX_P, MAX_T>::vert_is_in_conflict(VERT_TYPE v, double4 eqn) const {

        double4 pi1 = plane_for(v.x);
        double4 pi2 = plane_for(v.y);

#ifdef dim_2D
        double det = det3x3(pi1.x, pi2.x, eqn.x, pi1.y, pi2.y, eqn.y, pi1.w, pi2.w, eqn.w);

        double maxx = fmax(fmax(fabs(pi1.x), fabs(pi2.x)), fabs(eqn.x));
        double maxy = fmax(fmax(fabs(pi1.y), fabs(pi2.y)), fabs(eqn.y));
        double maxw = fmax(fmax(fabs(pi1.w), fabs(pi2.w)), fabs(eqn.w));

        double max_max = fmax(fmax(maxx, maxy), maxw);
        double eps     = 1e-14 * maxx * maxy * maxw;
        eps *= max_max;
#else
        double4 pi3 = plane_for(v.z);

        double det = det4x4(pi1.x,
                            pi2.x,
                            pi3.x,
                            eqn.x,
                            pi1.y,
                            pi2.y,
                            pi3.y,
                            eqn.y,
                            pi1.z,
                            pi2.z,
                            pi3.z,
                            eqn.z,
                            pi1.w,
                            pi2.w,
                            pi3.w,
                            eqn.w);

        double maxx = fmax(fmax(fabs(pi1.x), fabs(pi2.x)), fmax(fabs(pi3.x), fabs(eqn.x)));
        double maxy = fmax(fmax(fabs(pi1.y), fabs(pi2.y)), fmax(fabs(pi3.y), fabs(eqn.y)));
        double maxz = fmax(fmax(fabs(pi1.z), fabs(pi2.z)), fmax(fabs(pi3.z), fabs(eqn.z)));

        double eps = 1e-14 * maxx * maxy * maxz;
        double min_max, max_max;
        get_minmax3(min_max, max_max, maxx, maxy, maxz);
        eps *= (max_max * max_max);
#endif

        if (fabs(det) < eps) { *status = needs_exact_predicates; }

        return (det > 0.0);
    }

    // build the boundary loop separating removed triangles from kept ones (linked list in boundary_next[])
    template <int MAX_P, int MAX_T> HD void BasicConvexCell<MAX_P, MAX_T>::compute_boundary() {

#ifdef dim_2D
        // reset boundary linked list
        for (int i = 0; i < MAX_P; i++) {
            boundary_next[i] = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        // count how many removed edges reference each plane
        uchar line_count[MAX_P];
        for (int i = 0; i < MAX_P; i++) {
            line_count[i] = 0;
        }
        for (int r = 0; r < nb_r; r++) {
            uchar2 e = triangle[nb_t + r];
            line_count[e.x]++;
            line_count[e.y]++;
        }

        // boundary endpoints are the two planes referenced by exactly one removed edge (parity)
        uchar boundary_lines[2];
        int   nb_boundary = 0;
        for (int p = 0; p < nb_v; p++) {
            if (line_count[p] == 1) {
                if (nb_boundary < 2) { boundary_lines[nb_boundary++] = (uchar)p; }
            }
        }

        if (nb_boundary != 2) {
            *status = inconsistent_boundary;
            return;
        }

        first_boundary                   = boundary_lines[0];
        boundary_next[boundary_lines[0]] = boundary_lines[1];
        boundary_next[boundary_lines[1]] = boundary_lines[0];

#else
        // reset boundary linked list
        for (int i = 0; i < MAX_P; i++) {
            boundary_next[i] = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        // absorb removed triangles into the boundary loop one at a time, skipping any that
        // would make the loop non-simple; bail if we loop forever
        int   nb_iter = 0;
        uchar t       = nb_t;

        while (nb_r > 0) {
            if (nb_iter++ > 10000) {
                *status = inconsistent_boundary;
                return;
            }

            bool is_in_border[3];
            bool next_is_opp[3];

            for (int e = 0; e < 3; e++) {
                is_in_border[e] = (boundary_next[ith_plane(triangle, t, e)] != END_OF_LIST);
            }
            for (int e = 0; e < 3; e++) {
                next_is_opp[e] = (boundary_next[ith_plane(triangle, t, (e + 1) % 3)] == ith_plane(triangle, t, e));
            }

            bool new_border_is_simple = true;

            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) {
                    new_border_is_simple = false;
                }
            }

            if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
                if (first_boundary == END_OF_LIST) {
                    for (int e = 0; e < 3; e++) {
                        boundary_next[ith_plane(triangle, t, e)] = ith_plane(triangle, t, (e + 1) % 3);
                    }
                    first_boundary = triangle[t].x;
                } else {
                    new_border_is_simple = false;
                }
            }

            if (!new_border_is_simple) {
                t++;
                if (t == nb_t + nb_r) { t = nb_t; }
                continue;
            }

            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e]) { boundary_next[ith_plane(triangle, t, e)] = ith_plane(triangle, t, (e + 1) % 3); }
            }

            for (int e = 0; e < 3; e++) {
                if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
                    if (first_boundary == ith_plane(triangle, t, (e + 1) % 3)) {
                        first_boundary = boundary_next[ith_plane(triangle, t, (e + 1) % 3)];
                    }
                    boundary_next[ith_plane(triangle, t, (e + 1) % 3)] = END_OF_LIST;
                }
            }

            VERT_TYPE tmp             = triangle[t];
            triangle[t]               = triangle[nb_t + nb_r - 1];
            triangle[nb_t + nb_r - 1] = tmp;
            t                         = nb_t;
            nb_r--;
        }
#endif
    }

    template <int MAX_P, int MAX_T> HD void BasicConvexCell<MAX_P, MAX_T>::new_vertex(uchar i, uchar j, uchar k) {
        if (nb_t + 1 >= MAX_T) {
            *status = triangle_overflow;
            return;
        }
#ifdef dim_2D
        (void)k;
        double4 hi = plane_for(i);
        double4 hj = plane_for(j);
        double  rw = det2x2(hi.x, hi.y, hj.x, hj.y);
        if (rw > 0) {
            triangle[nb_t] = make_uchar2(j, i);
        } else {
            triangle[nb_t] = make_uchar2(i, j);
        }
#else
        triangle[nb_t] = make_uchar3(i, j, k);
#endif
        nb_t++;
    }

    // ============================================================
    // Extraction helpers
    // ============================================================

    template <int MAX_P, int MAX_T>
    HD double4 BasicConvexCell<MAX_P, MAX_T>::compute_vertex_point(VERT_TYPE v, bool persp_divide) const {
        double4 pi1 = plane_for(v.x);
        double4 pi2 = plane_for(v.y);
        double4 result;
#ifdef dim_2D
        result.x = -det2x2(pi1.w, pi1.y, pi2.w, pi2.y);
        result.y = -det2x2(pi1.x, pi1.w, pi2.x, pi2.w);
        result.z = 0;
        result.w = det2x2(pi1.x, pi1.y, pi2.x, pi2.y);
        if (persp_divide) { return make_double4(result.x / result.w, result.y / result.w, 0, 1); }
#else
        double4 pi3 = plane_for(v.z);
        result.x    = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y    = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z    = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w    = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
        if (persp_divide) {
            double inv_w = 1.0 / result.w;
            return make_double4(result.x * inv_w, result.y * inv_w, result.z * inv_w, 1);
        }
#endif
        return result;
    }

    template <int MAX_P, int MAX_T>
    HD bool collect_face_vertices(const BasicConvexCell<MAX_P, MAX_T>& cell,
                                  int                                  p,
                                  const double4*                       vertices,
                                  double4*                             face_verts,
                                  int*                                 n_face_verts) {
#ifdef dim_2D
        int n_fvi = 0;
        for (int i = 0; i < cell.nb_t; i++) {
            if (vert_references_plane(cell.triangle, i, (uchar)p)) {
                face_verts[n_fvi] = vertices[i];
                n_fvi++;
                if (n_fvi == 2) break;
            }
        }
        if (n_fvi < 2) return false;
        *n_face_verts = 2;
#else
        int face_vert_indices[MAX_T];
        int n_fvi = 0;
        for (int i = 0; i < cell.nb_t; i++) {
            if (vert_references_plane(cell.triangle, i, (uchar)p)) { face_vert_indices[n_fvi++] = i; }
        }
        if (n_fvi < DIMENSION) return false;

        int  ordered[MAX_T];
        bool used[MAX_T];
        for (int k = 0; k < n_fvi; k++)
            used[k] = false;
        ordered[0]    = face_vert_indices[0];
        used[0]       = true;
        int n_ordered = 1;

        for (int step = 1; step < n_fvi; step++) {
            int last = ordered[n_ordered - 1];

            uchar others_last[DIMENSION - 1];
            int   cnt = 0;
            for (int d = 0; d < DIMENSION; d++) {
                uchar pl = ith_plane(cell.triangle, (uchar)last, d);
                if (pl != (uchar)p) others_last[cnt++] = pl;
            }

            bool found = false;
            for (int j = 0; j < n_fvi; j++) {
                if (used[j]) continue;
                int candidate = face_vert_indices[j];

                for (int o = 0; o < DIMENSION - 1; o++) {
                    if (vert_references_plane(cell.triangle, candidate, others_last[o])) {
                        ordered[n_ordered++] = candidate;
                        used[j]              = true;
                        found                = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (!found) break;
        }

        if (n_ordered < DIMENSION) return false;

        *n_face_verts = n_ordered;
        for (int k = 0; k < n_ordered; k++) {
            face_verts[k] = vertices[ordered[k]];
        }
#endif
        return true;
    }

    HD static void write_face(VMesh*         mesh,
                       hsize_t        fi,
                       int            neighbor_id,
                       double         face_measure,
                       const double4* face_verts,
                       int            n_face_verts,
                       double4        seed,
                       double4        neighbor) {
        (void)face_verts;
        (void)n_face_verts;
        // neighbor_id arrives as a sorted-mixed sid (or -1 for bounding-box planes).
        // Remap to a real-sorted index via mesh->sid_to_neighbor before storing.
        int remapped = (neighbor_id >= 0) ? (int)mesh->sid_to_neighbor[neighbor_id] : neighbor_id;
        mesh->neighbor_cell[fi] = remapped;
        mesh->face_area[fi]     = face_measure;

#ifdef MOVING_MESH
        double fmx = 0.0, fmy = 0.0, fmz = 0.0;
        compute_face_centroid(face_verts, n_face_verts, fmx, fmy, fmz);

        if (neighbor_id >= 0) {
            double3 raw_normal = {neighbor.x - seed.x, neighbor.y - seed.y, neighbor.z - seed.z};
            geom    g_local    = compute_geom(raw_normal);
            double  ox         = fmx - 0.5 * (seed.x + neighbor.x);
            double  oy         = fmy - 0.5 * (seed.y + neighbor.y);
#ifdef dim_2D
            mesh->f_mid_local[fi] = ox * g_local.m.x + oy * g_local.m.y;
#else
            double oz                     = fmz - 0.5 * (seed.z + neighbor.z);
            mesh->f_mid_local[2 * fi]     = ox * g_local.m.x + oy * g_local.m.y + oz * g_local.m.z;
            mesh->f_mid_local[2 * fi + 1] = ox * g_local.p.x + oy * g_local.p.y + oz * g_local.p.z;
#endif
        } else {
#ifdef dim_2D
            mesh->f_mid_local[fi] = 0.0;
#else
            mesh->f_mid_local[2 * fi]     = 0.0;
            mesh->f_mid_local[2 * fi + 1] = 0.0;
#endif
        }
#endif
    }

} // namespace voronoi
