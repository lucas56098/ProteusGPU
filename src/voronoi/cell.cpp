#include "cell.h"
#include "geometry.h"
#include "voronoi.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#if defined(USE_OPENMP)
#include <omp.h>
#endif

namespace voronoi {

    static const uchar END_OF_LIST = 255;

    inline uchar& ith_plane(VERT_TYPE* triangles, uchar t, int i) {
        return reinterpret_cast<uchar*>(&(triangles[t]))[i];
    }

    inline uchar ith_plane(const VERT_TYPE* triangles, uchar t, int i) {
        return reinterpret_cast<const uchar*>(&(triangles[t]))[i];
    }

    // returns true if vertex at index t_idx references plane p
    static inline bool vert_references_plane(const VERT_TYPE* triangles, int t_idx, uchar p) {
        for (int d = 0; d < DIMENSION; d++) {
            if (ith_plane(triangles, (uchar)t_idx, d) == p) return true;
        }
        return false;
    }

    // init a convex cell (set it to bounding box)
    ConvexCell::ConvexCell(int p_seed, double* p_pts, Status* p_status) {

        // define bounding box (assume boxsize =1.0)
        double eps  = 1e-14;
        double xmin = -eps;
        double xmax = 1.0 + eps;
        double ymin = -eps;
        double ymax = 1.0 + eps;
#ifdef dim_3D
        double zmin = -eps;
        double zmax = 1.0 + eps;
#endif

        // store pointer to pts
        pts = p_pts;

        // set boundaries to END_OF_LIST
        first_boundary = END_OF_LIST;
        for (int i = 0; i < _MAX_P_; i++) {
            boundary_next[i] = END_OF_LIST;
        }

        // initialize plane_vid: boundary planes (-1), rest unset
        for (int i = 0; i < _MAX_P_; i++) {
            plane_vid[i] = -1;
        }

        // status set to success for now
        status  = p_status;
        *status = success;

        voro_seed = point_from_ptr(pts + DIMENSION * p_seed);

        // create 6/4 bounding planes for the initial bounding box
        half_plane[0] = make_double4(1.0, 0.0, 0.0, -xmin); // x >= xmin (left face)
        half_plane[1] = make_double4(-1.0, 0.0, 0.0, xmax); // x <= xmax (right face)
        half_plane[2] = make_double4(0.0, 1.0, 0.0, -ymin); // y >= ymin (front face)
        half_plane[3] = make_double4(0.0, -1.0, 0.0, ymax); // y <= ymax (back face)
#ifdef dim_3D
        half_plane[4] = make_double4(0.0, 0.0, 1.0, -zmin); // z >= zmin (bottom face)
        half_plane[5] = make_double4(0.0, 0.0, -1.0, zmax); // z <= zmax (top face)
#endif

        // store initaial planes delunay triangles
#ifdef dim_2D
        triangle[0] = make_uchar2(2, 0); // bottom-left
        triangle[1] = make_uchar2(1, 2); // bottom-right
        triangle[2] = make_uchar2(3, 1); // top-right
        triangle[3] = make_uchar2(0, 3); // top-left
        nb_v        = 4;                 // 4 initial planes
        nb_t        = 4;                 // 4 initial triangles
#else
        triangle[0] = make_uchar3(2, 5, 0); // (top front left)
        triangle[1] = make_uchar3(5, 3, 0); // (top back left)
        triangle[2] = make_uchar3(1, 5, 2); // (top front right)
        triangle[3] = make_uchar3(5, 1, 3); // (top back right)
        triangle[4] = make_uchar3(4, 2, 0); // (bottom front left)
        triangle[5] = make_uchar3(4, 0, 3); // (bottom back left)
        triangle[6] = make_uchar3(2, 4, 1); // (bottom front right)
        triangle[7] = make_uchar3(4, 3, 1); // (bottom back right)
        nb_v        = 6;                    // 6 initial planes
        nb_t        = 8;                    // 8 initial triangles
#endif
    }

    // --------------------------------------------
    // ------------ clipping functions ------------
    // --------------------------------------------
    // clip convex cell by a plane
    void ConvexCell::clip_by_plane(int vid) {

        // add new plane/line equation to memory pool
        int cur_v = new_halfplane(vid);
        if (*status == vertex_overflow) { return; }

        // get that half plane
        double4 eqn = half_plane[cur_v];
        nb_r        = 0;

        int i = 0;
        while (i < nb_t) { // for all vertices of the cell
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

        // if no clips, then remove the plane equation
        if (nb_r == 0) {
            nb_v--;
            return;
        }

        // compute cavity boundary
        compute_boundary();
        if (*status != success) { return; }
        if (first_boundary == END_OF_LIST) { return; }

        // triangulate cavity using boundary cycle
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

    // -------- helper functions for this --------
    // add new halfplane to memory pool and return its index
    int ConvexCell::new_halfplane(int vid) {
        if (nb_v >= _MAX_P_) {
            *status = vertex_overflow;
            return -1;
        }

        double4 B        = point_from_ptr(pts + DIMENSION * vid);
        double4 dir      = minus4(voro_seed, B);
        double4 ave2     = plus4(voro_seed, B);
        double  dot      = dot3(ave2, dir); // works for 2D since z=0
        half_plane[nb_v] = make_double4(dir.x, dir.y, dir.z, -dot / 2.0);
        plane_vid[nb_v]  = vid;
        nb_v++;
        return nb_v - 1;
    }

    // check if vertex is on the wrong side of half plane, i.e. if it needs to be removed
    bool ConvexCell::vert_is_in_conflict(VERT_TYPE v, double4 eqn) const {

        double4 pi1 = half_plane[v.x];
        double4 pi2 = half_plane[v.y];

#ifdef dim_2D
        double det = det3x3(pi1.x, pi2.x, eqn.x, pi1.y, pi2.y, eqn.y, pi1.w, pi2.w, eqn.w);

        double maxx = fmax(fmax(fabs(pi1.x), fabs(pi2.x)), fabs(eqn.x));
        double maxy = fmax(fmax(fabs(pi1.y), fabs(pi2.y)), fabs(eqn.y));
        double maxw = fmax(fmax(fabs(pi1.w), fabs(pi2.w)), fabs(eqn.w));

        // bound for 3x3 determinant with entries from rows (x, y, w)
        double max_max = fmax(fmax(maxx, maxy), maxw);
        double eps     = 1e-14 * maxx * maxy * maxw;
        eps *= max_max;
#else
        double4 pi3 = half_plane[v.z];

        // 4x4 determinant: rows are (x, y, z, w) of pi1, pi2, pi3, eqn
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

    // compute cavity boundary after clipping by plane
    void ConvexCell::compute_boundary() {

#ifdef dim_2D
        // 2D boundary computation: find exactly 2 boundary lines
        // A boundary line appears in exactly one removed vertex and one surviving vertex
        for (int i = 0; i < _MAX_P_; i++) {
            boundary_next[i] = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        // count how many times each line appears in removed vertices
        int line_count[_MAX_P_];
        for (int i = 0; i < _MAX_P_; i++) {
            line_count[i] = 0;
        }

        for (int r = 0; r < nb_r; r++) {
            uchar2 e = triangle[nb_t + r];
            line_count[e.x]++;
            line_count[e.y]++;
        }

        // boundary lines are those appearing exactly once in removed vertices
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

        // build circular list: B0 -> B1 -> B0
        first_boundary                   = boundary_lines[0];
        boundary_next[boundary_lines[0]] = boundary_lines[1];
        boundary_next[boundary_lines[1]] = boundary_lines[0];

#else
        // 3D boundary computation
        // clean circular list of the boundary
        for (int i = 0; i < _MAX_P_; i++) {
            boundary_next[i] = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        int   nb_iter = 0;
        uchar t       = nb_t;

        while (nb_r > 0) {
            if (nb_iter++ > 100) {
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

            // check for non manifoldness
            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) {
                    new_border_is_simple = false;
                }
            }

            // check for more than one boundary ... or first triangle
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

            // link next
            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e]) { boundary_next[ith_plane(triangle, t, e)] = ith_plane(triangle, t, (e + 1) % 3); }
            }

            // destroy link from removed vertices
            for (int e = 0; e < 3; e++) {
                if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
                    if (first_boundary == ith_plane(triangle, t, (e + 1) % 3)) {
                        first_boundary = boundary_next[ith_plane(triangle, t, (e + 1) % 3)];
                    }
                    boundary_next[ith_plane(triangle, t, (e + 1) % 3)] = END_OF_LIST;
                }
            }

            // remove triangle from R, and restart iterating on R
            VERT_TYPE tmp             = triangle[t];
            triangle[t]               = triangle[nb_t + nb_r - 1];
            triangle[nb_t + nb_r - 1] = tmp;
            t                         = nb_t;
            nb_r--;
        }
#endif
    }

    // add new vertex to convex cell
    void ConvexCell::new_vertex(uchar i, uchar j, uchar k) {
        if (nb_t + 1 >= _MAX_T_) {
            *status = triangle_overflow;
            return;
        }
#ifdef dim_2D
        (void)k; // unused in 2D
        // ensure consistent orientation: result.w < 0 (same convention as 3D)
        double rw = det2x2(half_plane[i].x, half_plane[i].y, half_plane[j].x, half_plane[j].y);
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

    // --------------------------------------------
    // ---------- security radius check -----------
    // --------------------------------------------
    bool ConvexCell::is_security_radius_reached(double4 last_neig) const {
        // finds furthest voro vertex distance2
        double v_dist = 0;

        for (int i = 0; i < nb_t; i++) {
            double4 pc   = compute_vertex_point(triangle[i]);
            double4 diff = minus4(pc, voro_seed);
            double  d2   = dot3(diff, diff); // works for 2D since z=0
            v_dist       = fmax(d2, v_dist);
        }

        // compare to new neighbors distance2
        double4 diff = minus4(last_neig, voro_seed);
        double  d2   = dot3(diff, diff);
        return (d2 > 4 * v_dist);
    }

    // -------- helper functions for this --------
    // compute vertex position from intersecting planes
    double4 ConvexCell::compute_vertex_point(VERT_TYPE v, bool persp_divide) const {
        double4 pi1 = half_plane[v.x];
        double4 pi2 = half_plane[v.y];
        double4 result;
#ifdef dim_2D
        result.x = -det2x2(pi1.w, pi1.y, pi2.w, pi2.y);
        result.y = -det2x2(pi1.x, pi1.w, pi2.x, pi2.w);
        result.z = 0;
        result.w = det2x2(pi1.x, pi1.y, pi2.x, pi2.y);
        if (persp_divide) { return make_double4(result.x / result.w, result.y / result.w, 0, 1); }
#else
        double4 pi3 = half_plane[v.z];
        result.x    = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y    = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z    = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w    = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
        if (persp_divide) { return make_double4(result.x / result.w, result.y / result.w, result.z / result.w, 1); }
#endif
        return result;
    }

    // --------------------------------------------
    // -------- put convex cell into VMesh --------
    // --------------------------------------------

    // pass 1: extract per-cell data (seeds/com/volumes) and return face count
    int extract_cell_data(const ConvexCell& cell, const double4* vertices, VMesh* mesh, hsize_t cell_index) {
        double3 seed            = {cell.voro_seed.x, cell.voro_seed.y, cell.voro_seed.z};
        mesh->seeds[cell_index] = seed;

#ifdef dim_2D
        double cx                 = cell.voro_seed.x;
        double cy                 = cell.voro_seed.y;
        mesh->volumes[cell_index] = compute_cell_area_centroid_2d(vertices, cell.nb_t, cx, cy);
        mesh->com[cell_index]     = {cx, cy, 0.0};
#else
        double cx, cy, cz;
        mesh->volumes[cell_index] = compute_cell_volume_centroid_3d(cell, vertices, cx, cy, cz);
        mesh->com[cell_index]     = {cx, cy, cz};
#endif

        int     face_count = 0;
        double4 face_verts[_MAX_T_];
        int     n_fv;
        for (int p = 0; p < cell.nb_v; p++) {
            if (!collect_face_vertices(cell, p, vertices, face_verts, &n_fv)) continue;
            face_count++;
        }

        return face_count;
    }

    // pass 2: write face data into VMesh at face_ptr offset
    void extract_cell_faces(const ConvexCell& cell, const double4* vertices, VMesh* mesh, hsize_t cell_index) {
        hsize_t fi = mesh->face_ptr[cell_index];

        double4 face_verts[_MAX_T_];
        int     n_fv;
        for (int p = 0; p < cell.nb_v; p++) {
            if (!collect_face_vertices(cell, p, vertices, face_verts, &n_fv)) continue;

            double  face_measure = compute_face_measure(face_verts, n_fv, cell.voro_seed, nullptr);
            double4 neighbor     = point_from_ptr(cell.pts + DIMENSION * cell.plane_vid[p]);
            write_face(mesh, fi, cell.plane_vid[p], face_measure, face_verts, n_fv, cell.voro_seed, neighbor);
            fi++;
        }
    }

    // check that face arrays have sufficient pre-allocated capacity (no realloc for GPU compatibility)
    void ensure_face_capacity(VMesh* mesh, hsize_t needed) {
        if (needed <= mesh->face_capacity) return;
        std::cerr << "VORONOI: Error! face count " << needed << " exceeds pre-allocated face capacity "
                  << mesh->face_capacity << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
        exit(EXIT_FAILURE);
    }

    bool collect_face_vertices(
        const ConvexCell& cell, int p, const double4* vertices, double4* face_verts, int* n_face_verts) {
        // find vertices that reference plane p (stack array)
        int face_vert_indices[_MAX_T_];
        int n_fvi = 0;
        for (int i = 0; i < cell.nb_t; i++) {
            if (vert_references_plane(cell.triangle, i, (uchar)p)) { face_vert_indices[n_fvi++] = i; }
        }
        if (n_fvi < DIMENSION) return false;

#ifdef dim_2D
        *n_face_verts = n_fvi;
        for (int k = 0; k < n_fvi; k++) {
            face_verts[k] = vertices[face_vert_indices[k]];
        }
#else
        // order vertices by adjacency (adjacent triangles on a face share an edge)
        int  ordered[_MAX_T_];
        bool used[_MAX_T_];
        for (int k = 0; k < n_fvi; k++)
            used[k] = false;
        ordered[0]    = face_vert_indices[0];
        used[0]       = true;
        int n_ordered = 1;

        for (int step = 1; step < n_fvi; step++) {
            int last = ordered[n_ordered - 1];

            // get the DIMENSION-1 other planes in this vertex that are not p
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

    // write a single face at index fi
    void write_face(VMesh*         mesh,
                    hsize_t        fi,
                    int            neighbor_id,
                    double         face_measure,
                    const double4* face_verts,
                    int            n_face_verts,
                    double4        seed,
                    double4        neighbor) {
        (void)face_verts; // unused unless MOVING_MESH
        (void)n_face_verts;
        mesh->neighbor_cell[fi] = neighbor_id;
        mesh->face_area[fi]     = (compact_t)face_measure;

#ifdef MOVING_MESH
        double fmx = 0.0, fmy = 0.0, fmz = 0.0;
        compute_face_centroid(face_verts, n_face_verts, fmx, fmy, fmz);

        if (neighbor_id >= 0) {
            double3 raw_normal = {neighbor.x - seed.x, neighbor.y - seed.y, neighbor.z - seed.z};
            geom    g_local    = compute_geom(raw_normal);
            double  ox         = fmx - 0.5 * (seed.x + neighbor.x);
            double  oy         = fmy - 0.5 * (seed.y + neighbor.y);
#ifdef dim_2D
            mesh->f_mid_local[fi] = (compact_t)(ox * g_local.m.x + oy * g_local.m.y);
#else
            double oz                     = fmz - 0.5 * (seed.z + neighbor.z);
            mesh->f_mid_local[2 * fi]     = (compact_t)(ox * g_local.m.x + oy * g_local.m.y + oz * g_local.m.z);
            mesh->f_mid_local[2 * fi + 1] = (compact_t)(ox * g_local.p.x + oy * g_local.p.y + oz * g_local.p.z);
#endif
        } else {
#ifdef dim_2D
            mesh->f_mid_local[fi] = 0;
#else
            mesh->f_mid_local[2 * fi]     = 0;
            mesh->f_mid_local[2 * fi + 1] = 0;
#endif
        }
#endif
    }

} // namespace voronoi