#include "cell.h"
#include <iostream>
#include "../global/allvars.h"
#include "../knn/knn.h"
#include "../io/input.h"
#include "../io/output.h"
#include "cell.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include "voronoi.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace voronoi {

    static const uchar END_OF_LIST = 255;

    // memory pools for voronoi mesh generation
    static VERT_TYPE triangle_data[_VORO_BLOCK_SIZE_ * _MAX_T_];
    static uchar boundary_next_data[_VORO_BLOCK_SIZE_ * _MAX_P_];
    static double4 half_plane_data[_VORO_BLOCK_SIZE_ * _MAX_P_];

    // helper functions to access memory pools
    static inline VERT_TYPE& triangle(int t) { return triangle_data[threadId.x * _MAX_T_ + t]; }
    static inline uchar& boundary_next(int v) { return boundary_next_data[threadId.x * _MAX_P_ + v]; }
    static inline double4& half_plane(int v) { return half_plane_data[threadId.x * _MAX_P_ + v]; }
    inline  uchar& ith_plane(uchar t, int i) {return reinterpret_cast<uchar *>(&(triangle(t)))[i];}

    // returns true if vertex at index t_idx references plane p
    static inline bool vert_references_plane(int t_idx, uchar p) {
        for (int d = 0; d < DIMENSION; d++) {
            if (ith_plane((uchar)t_idx, d) == p) return true;
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
        for (int i = 0; i < _MAX_P_; i++) {boundary_next(i) = END_OF_LIST;}

        // initialize plane_vid: boundary planes (-1), rest unset
        for (int i = 0; i < _MAX_P_; i++) {plane_vid[i] = -1;}

        // store seed point info
        voro_id = p_seed;

        // status set to success for now
        status = p_status;
        *status = success;

        voro_seed = point_from_ptr(pts + DIMENSION * voro_id);

        // create 6/4 bounding planes for the initial bounding box
        half_plane(0) = make_double4( 1.0,  0.0,  0.0, -xmin);  // x >= xmin (left face)
        half_plane(1) = make_double4(-1.0,  0.0,  0.0,  xmax);  // x <= xmax (right face)
        half_plane(2) = make_double4( 0.0,  1.0,  0.0, -ymin);  // y >= ymin (front face)
        half_plane(3) = make_double4( 0.0, -1.0,  0.0,  ymax);  // y <= ymax (back face)
        #ifdef dim_3D
        half_plane(4) = make_double4( 0.0,  0.0,  1.0, -zmin);  // z >= zmin (bottom face)
        half_plane(5) = make_double4( 0.0,  0.0, -1.0,  zmax);  // z <= zmax (top face)
        #endif

        // store initaial planes delunay triangles
    #ifdef dim_2D    
        triangle(0) = make_uchar2(2, 0);  // bottom-left
        triangle(1) = make_uchar2(1, 2);  // bottom-right
        triangle(2) = make_uchar2(3, 1);  // top-right
        triangle(3) = make_uchar2(0, 3);  // top-left
        nb_v = 4;  // 4 initial planes
        nb_t = 4;  // 4 initial triangles
    #else
        triangle(0) = make_uchar3(2, 5, 0); // (top front left)
        triangle(1) = make_uchar3(5, 3, 0); // (top back left)
        triangle(2) = make_uchar3(1, 5, 2); // (top front right)
        triangle(3) = make_uchar3(5, 1, 3); // (top back right)
        triangle(4) = make_uchar3(4, 2, 0); // (bottom front left)
        triangle(5) = make_uchar3(4, 0, 3); // (bottom back left)
        triangle(6) = make_uchar3(2, 4, 1); // (bottom front right)
        triangle(7) = make_uchar3(4, 3, 1); // (bottom back right)
        nb_v = 6;  // 6 initial planes
        nb_t = 8;  // 8 initial triangles
    #endif
    }

// --------------------------------------------
// ------------ clipping functions ------------
// --------------------------------------------
    // clip convex cell by a plane
    void ConvexCell::clip_by_plane(int vid) {
        
        // add new plane/line equation to memory pool
        int cur_v = new_halfplane(vid); 
        if (*status == vertex_overflow) {return;}

        // get that half plane
        double4 eqn = half_plane(cur_v);
        nb_r = 0;

        int i = 0;
        while (i < nb_t) { // for all vertices of the cell
            if(vert_is_in_conflict(triangle(i), eqn)) {
                nb_t--;
                std::swap(triangle(i), triangle(nb_t));
                nb_r++;
            } else {
            i++;
            }
        }
        if (*status == needs_exact_predicates) {return;}

        // if no clips, then remove the plane equation
        if (nb_r == 0) {
            nb_v--;
            return;
        }

        // compute cavity boundary
        compute_boundary();
        if (*status != success) {return;}
        if (first_boundary == END_OF_LIST) {return;}

        // triangulate cavity using boundary cycle
        uchar cir = first_boundary;
        do {
        #ifdef dim_2D
            new_vertex(cur_v, cir);
        #else
            new_vertex(cur_v, cir, boundary_next(cir));
        #endif
            if (*status != success) return;
            cir = boundary_next(cir);
        } while (cir != first_boundary);
    }

    // -------- helper functions for this --------
    // add new halfplane to memory pool and return its index
    int ConvexCell::new_halfplane(int vid) {
        if (nb_v >= _MAX_P_) { 
            *status = vertex_overflow; 
            return -1; 
        }

        double4 B = point_from_ptr(pts + DIMENSION * vid);
        double4 dir = minus4(voro_seed, B);
        double4 ave2 = plus4(voro_seed, B);
        double dot = dot3(ave2, dir); // works for 2D since z=0
        half_plane(nb_v) = make_double4(dir.x, dir.y, dir.z, -dot / 2.0);
        plane_vid[nb_v] = vid;
        nb_v++;
        return nb_v - 1;
    }

    // check if vertex is on the wrong side of half plane, i.e. if it needs to be removed
    bool ConvexCell::vert_is_in_conflict(VERT_TYPE v, double4 eqn) const {
    
    double4 pi1 = half_plane(v.x);
    double4 pi2 = half_plane(v.y);

    #ifdef dim_2D
    double det = det3x3(
	pi1.x, pi2.x, eqn.x,
	pi1.y, pi2.y, eqn.y,
	pi1.w, pi2.w, eqn.w
    );

    double maxx = std::max({std::fabs(pi1.x), std::fabs(pi2.x), std::fabs(eqn.x)});
    double maxy = std::max({std::fabs(pi1.y), std::fabs(pi2.y), std::fabs(eqn.y)});
    double maxw = std::max({std::fabs(pi1.w), std::fabs(pi2.w), std::fabs(eqn.w)});

    // bound for 3x3 determinant with entries from rows (x, y, w)
    double max_max = std::max({maxx, maxy, maxw});
    double eps = 1e-14 * maxx * maxy * maxw;
    eps *= max_max;
    #else
    double4 pi3 = half_plane(v.z);

    // 4x4 determinant: rows are (x, y, z, w) of pi1, pi2, pi3, eqn
    double det = det4x4(
	pi1.x, pi2.x, pi3.x, eqn.x,
	pi1.y, pi2.y, pi3.y, eqn.y,
	pi1.z, pi2.z, pi3.z, eqn.z,
	pi1.w, pi2.w, pi3.w, eqn.w
    );

    double maxx = std::max({std::fabs(pi1.x), std::fabs(pi2.x), std::fabs(pi3.x), std::fabs(eqn.x)});
    double maxy = std::max({std::fabs(pi1.y), std::fabs(pi2.y), std::fabs(pi3.y), std::fabs(eqn.y)});
    double maxz = std::max({std::fabs(pi1.z), std::fabs(pi2.z), std::fabs(pi3.z), std::fabs(eqn.z)});

    double eps = 1e-14 * maxx * maxy * maxz;
    double min_max, max_max;
    get_minmax3(min_max, max_max, maxx, maxy, maxz);
    eps *= (max_max * max_max);
    #endif

    if(std::fabs(det) < eps) {
	*status = needs_exact_predicates;
    }

    return (det > 0.0);
    }

    // compute cavity boundary after clipping by plane
    void ConvexCell::compute_boundary() {

        #ifdef dim_2D
        // 2D boundary computation: find exactly 2 boundary lines
        // A boundary line appears in exactly one removed vertex and one surviving vertex
        for (int i = 0; i < _MAX_P_; i++) {
            boundary_next(i) = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        // count how many times each line appears in removed vertices
        int line_count[_MAX_P_];
        for (int i = 0; i < _MAX_P_; i++) { line_count[i] = 0; }

        for (int r = 0; r < nb_r; r++) {
            uchar2 e = triangle(nb_t + r);
            line_count[e.x]++;
            line_count[e.y]++;
        }

        // boundary lines are those appearing exactly once in removed vertices
        uchar boundary_lines[2];
        int nb_boundary = 0;

        for (int p = 0; p < nb_v; p++) {
            if (line_count[p] == 1) {
                if (nb_boundary < 2) {
                    boundary_lines[nb_boundary++] = (uchar)p;
                }
            }
        }

        if (nb_boundary != 2) {
            *status = inconsistent_boundary;
            return;
        }

        // build circular list: B0 → B1 → B0
        first_boundary = boundary_lines[0];
        boundary_next(boundary_lines[0]) = boundary_lines[1];
        boundary_next(boundary_lines[1]) = boundary_lines[0];

    #else
        // 3D boundary computation
        // clean circular list of the boundary
        for (int i = 0; i < _MAX_P_; i++) {
            boundary_next(i) = END_OF_LIST;
        } 
        first_boundary = END_OF_LIST;
    
        int nb_iter =0;
        uchar t = nb_t;

        while (nb_r > 0) {
            if (nb_iter++>100) {
                *status = inconsistent_boundary;
                return;
            }

            bool is_in_border[3];
            bool next_is_opp[3];

            for (int e = 0; e < 3; e++) {
                is_in_border[e] = (boundary_next(ith_plane(t, e)) != END_OF_LIST);
            }
            for (int e = 0; e < 3; e++) {
                next_is_opp[e] = (boundary_next(ith_plane(t, (e + 1)%3)) == ith_plane(t, e));
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
                        boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
                    }
                    first_boundary = triangle(t).x;   
                } else {
                    new_border_is_simple = false;
                }
            }

            if (!new_border_is_simple) {
                t++;
                if (t == nb_t + nb_r) {t = nb_t;}
                continue;
            }

            // link next
            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e]) {
                    boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
                }
            }

            // destroy link from removed vertices
            for (int e = 0; e < 3; e++) {
                if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
                    if (first_boundary == ith_plane(t, (e + 1) % 3)) {
                        first_boundary = boundary_next(ith_plane(t, (e + 1) % 3));
                    }
                    boundary_next(ith_plane(t, (e + 1) % 3)) = END_OF_LIST;
                }
            }
            
            //remove triangle from R, and restart iterating on R
            std::swap(triangle(t), triangle(nb_t+nb_r-1));
            t = nb_t;
            nb_r--;
        }
    #endif
    }

    // add new vertex to convex cell
    void ConvexCell::new_vertex(uchar i, uchar j, uchar k) {
        if (nb_t+1 >= _MAX_T_) { 
            *status = triangle_overflow; 
            return; 
        }
    #ifdef dim_2D
        (void)k; // unused in 2D
        // ensure consistent orientation: result.w < 0 (same convention as 3D)
        double rw = det2x2(half_plane(i).x, half_plane(i).y, half_plane(j).x, half_plane(j).y);
        if (rw > 0) {
            triangle(nb_t) = make_uchar2(j, i);
        } else {
            triangle(nb_t) = make_uchar2(i, j);
        }
    #else
        triangle(nb_t) = make_uchar3(i, j, k);
    #endif
        nb_t++;
    }

// --------------------------------------------
// ---------- security radius check -----------
// --------------------------------------------
    bool ConvexCell::is_security_radius_reached(double4 last_neig) {
        // finds furthest voro vertex distance2
        double v_dist = 0;
    
        for (int i = 0; i < nb_t; i++) {
            double4 pc = compute_vertex_point(triangle(i));
            double4 diff = minus4(pc, voro_seed);
            double d2 = dot3(diff, diff); // works for 2D since z=0
            v_dist = std::max(d2, v_dist);
        }
    
        //compare to new neighbors distance2
        double4 diff = minus4(last_neig, voro_seed);
        double d2 = dot3(diff, diff);
        return (d2 > 4*v_dist);
    }

    // -------- helper functions for this --------
    // compute vertex position from intersecting planes
    double4 ConvexCell::compute_vertex_point(VERT_TYPE v, bool persp_divide) const {
        double4 pi1 = half_plane(v.x);
        double4 pi2 = half_plane(v.y);
        double4 result;
    #ifdef dim_2D
        result.x = -det2x2(pi1.w, pi1.y, pi2.w, pi2.y);
        result.y = -det2x2(pi1.x, pi1.w, pi2.x, pi2.w);
        result.z = 0;
        result.w =  det2x2(pi1.x, pi1.y, pi2.x, pi2.y);
        if (persp_divide) {
            return make_double4(result.x / result.w, result.y / result.w, 0, 1);
        }
    #else
        double4 pi3 = half_plane(v.z);
        result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w =  det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
        if (persp_divide) {
            return make_double4(result.x / result.w, result.y / result.w, result.z / result.w, 1);
        }
    #endif
        return result;
    }

// --------------------------------------------
// -------- put convex cell into VMesh --------
// --------------------------------------------
    void extract_cell_to_vmesh(ConvexCell& cell, VMesh* mesh, hsize_t cell_index, hsize_t& face_capacity) {
        // store cell id and seed position (z=0 in 2D via point_from_ptr)
        mesh->cell_ids[cell_index] = (hsize_t)cell.voro_id;
        double3 seed = {cell.voro_seed.x, cell.voro_seed.y, cell.voro_seed.z};
        mesh->seeds[cell_index] = seed;

        // compute all vertex positions
        std::vector<double4> vertices(cell.nb_t);
        for (int i = 0; i < cell.nb_t; i++) {
            vertices[i] = cell.compute_vertex_point(triangle(i), true);
        }

        // compute cell volume/area
    #ifdef dim_2D
        mesh->volumes[cell_index] = compute_cell_area_2d(vertices, cell.nb_t);
    #else
        double cell_volume = 0.0;
    #endif

        // count valid faces to ensure capacity (need DIMENSION vertices per face)
        int face_count = 0;
        for (int p = 0; p < cell.nb_v; p++) {
            int cnt = 0;
            for (int i = 0; i < cell.nb_t; i++) {
                if (vert_references_plane(i, (uchar)p)) cnt++;
            }
            if (cnt >= DIMENSION) face_count++;
        }

        ensure_face_capacity(mesh, face_capacity, mesh->num_faces + face_count);
        mesh->face_ptr[cell_index] = mesh->num_faces;

        // extract faces: iterate over each plane and collect referencing vertices
        int actual_count = 0;
        for (int p = 0; p < cell.nb_v; p++) {
            std::vector<double4> face_verts;
            if (!collect_face_vertices(cell, p, vertices, face_verts)) continue;

            double3 normal = compute_face_normal(p);

        #ifdef dim_2D
            double face_measure = compute_face_measure(face_verts, cell.voro_seed, nullptr);
        #else
            double face_measure = compute_face_measure(face_verts, cell.voro_seed, &cell_volume);
        #endif

            store_face_data(mesh, face_verts, cell.plane_vid[p], normal, face_measure);
            actual_count++;
        }

    #ifdef dim_3D
        mesh->volumes[cell_index] = std::fabs(cell_volume);
    #endif
        mesh->face_counts[cell_index] = (hsize_t)actual_count;
    }

    // -------- helper functions for this --------
#ifdef dim_2D
    // compute cell area using shoelace formula (2D only)
    static double compute_cell_area_2d(const std::vector<double4>& vertices, int nb_t) {
        double cx = 0, cy = 0;
        for (int i = 0; i < nb_t; i++) {
            cx += vertices[i].x;
            cy += vertices[i].y;
        }
        cx /= nb_t;
        cy /= nb_t;

        std::vector<int> order(nb_t);
        for (int i = 0; i < nb_t; i++) { order[i] = i; }
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::atan2(vertices[a].y - cy, vertices[a].x - cx) <
                   std::atan2(vertices[b].y - cy, vertices[b].x - cx);
        });

        double cell_area = 0;
        for (int i = 0; i < nb_t; i++) {
            int j = (i + 1) % nb_t;
            cell_area += vertices[order[i]].x * vertices[order[j]].y;
            cell_area -= vertices[order[j]].x * vertices[order[i]].y;
        }
        return std::fabs(cell_area) / 2.0;
    }
#endif

    // helper to grow face arrays when needed
    void ensure_face_capacity(VMesh* mesh, hsize_t& face_capacity, hsize_t needed) {
        if (needed <= face_capacity) return;
        hsize_t new_capacity = face_capacity * 2;
        if (new_capacity < needed) new_capacity = needed;
        mesh->neighbor_cell = (int*)realloc(mesh->neighbor_cell, new_capacity * sizeof(int));
        mesh->face_normal = (double3*)realloc(mesh->face_normal, new_capacity * sizeof(double3));
        mesh->face_area = (double*)realloc(mesh->face_area, new_capacity * sizeof(double));
        #ifdef DEBUG_MODE
        mesh->edge_coords_offsets = (hsize_t*)realloc(mesh->edge_coords_offsets, new_capacity * sizeof(hsize_t));
        #endif
        face_capacity = new_capacity;
    }

    bool collect_face_vertices(ConvexCell& cell, int p, const std::vector<double4>& vertices, std::vector<double4>& face_verts) {
        // find vertices that reference plane p
        std::vector<int> face_vert_indices;
        for (int i = 0; i < cell.nb_t; i++) {
            if (vert_references_plane(i, (uchar)p)) {
                face_vert_indices.push_back(i);
            }
        }
        if ((int)face_vert_indices.size() < DIMENSION) return false;

    #ifdef dim_2D
        for (int idx : face_vert_indices) {
            face_verts.push_back(vertices[idx]);
        }
    #else
        // order vertices by adjacency (adjacent triangles on a face share an edge)
        std::vector<int> ordered;
        ordered.push_back(face_vert_indices[0]);
        std::vector<bool> used(face_vert_indices.size(), false);
        used[0] = true;

        for (size_t step = 1; step < face_vert_indices.size(); step++) {
            int last = ordered.back();

            // get the DIMENSION-1 other planes in this vertex that are not p
            uchar others_last[DIMENSION - 1];
            int cnt = 0;
            for (int d = 0; d < DIMENSION; d++) {
                uchar pl = ith_plane((uchar)last, d);
                if (pl != (uchar)p) others_last[cnt++] = pl;
            }

            bool found = false;
            for (size_t j = 0; j < face_vert_indices.size(); j++) {
                if (used[j]) continue;
                int candidate = face_vert_indices[j];

                for (int o = 0; o < DIMENSION - 1; o++) {
                    if (vert_references_plane(candidate, others_last[o])) {
                        ordered.push_back(candidate);
                        used[j] = true;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (!found) break;
        }

        if ((int)ordered.size() < DIMENSION) return false;

        for (int idx : ordered) {
            face_verts.push_back(vertices[idx]);
        }
    #endif
        return true;
    }

    // compute face normal from clip plane equation (z=0 in 2D, so dot3 gives ||(a,b)||)
    double3 compute_face_normal(int p) {
        double4 plane_eq = half_plane(p);
        double nlen = std::sqrt(dot3(plane_eq, plane_eq));
        return {plane_eq.x / nlen, plane_eq.y / nlen, plane_eq.z / nlen};
    }

    // compute face measure (length/area)
    double compute_face_measure(std::vector<double4>& face_verts, double4 seed, double* cell_volume) {
        double face_measure = 0.0;

    #ifdef dim_2D
        (void)seed; (void)cell_volume;
        double dx = face_verts[1].x - face_verts[0].x;
        double dy = face_verts[1].y - face_verts[0].y;
        face_measure = std::sqrt(dx * dx + dy * dy);
    #else
        // ensure face vertices are oriented consistently (outward from seed)
        {
            double4 edge1 = minus4(face_verts[1], face_verts[0]);
            double4 edge2 = minus4(face_verts[2], face_verts[0]);
            double4 face_cross = cross3(edge1, edge2);
            double4 centroid = make_double4(0, 0, 0, 0);
            for (const auto& fv : face_verts) {
                centroid.x += fv.x;
                centroid.y += fv.y;
                centroid.z += fv.z;
            }
            centroid.x /= face_verts.size();
            centroid.y /= face_verts.size();
            centroid.z /= face_verts.size();
            double4 outward = minus4(centroid, seed);
            if (dot3(face_cross, outward) < 0) {
                std::reverse(face_verts.begin(), face_verts.end());
            }
        }

        // face area via fan triangulation from vertex 0
        double4 v0 = face_verts[0];
        for (size_t i = 1; i + 1 < face_verts.size(); i++) {
            double4 edge1 = minus4(face_verts[i], v0);
            double4 edge2 = minus4(face_verts[i + 1], v0);
            double4 cr = cross3(edge1, edge2);
            face_measure += 0.5 * std::sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
        }

        // contribute to cell volume using divergence theorem
        for (size_t i = 1; i + 1 < face_verts.size(); i++) {
            double4 a = minus4(face_verts[0], seed);
            double4 b = minus4(face_verts[i], seed);
            double4 c = minus4(face_verts[i + 1], seed);
            double4 bxc = cross3(b, c);
            *cell_volume += dot3(a, bxc) / 6.0;
        }
    #endif

        return face_measure;
    }

#ifdef DEBUG_MODE
    // helper needed in store_face_data
    static hsize_t edge_coords_capacity_global = 0;
    static void ensure_edge_coords_capacity(VMesh* mesh, hsize_t needed_verts) {
        if (needed_verts <= edge_coords_capacity_global) return;
        hsize_t new_capacity = edge_coords_capacity_global * 2;
        if (new_capacity < needed_verts) new_capacity = needed_verts;
        mesh->edge_coords = (double*)realloc(mesh->edge_coords, new_capacity * DIMENSION * sizeof(double));
        edge_coords_capacity_global = new_capacity;
    }
#endif

    // store face data into VMesh arrays
    void store_face_data(VMesh* mesh, const std::vector<double4>& face_verts, int neighbor_id, double3 normal, double face_measure) {
        (void)face_verts;  // unused unless DEBUG_MODE is enabled
        hsize_t fi = mesh->num_faces;
        mesh->neighbor_cell[fi] = neighbor_id;
        mesh->face_normal[fi] = normal;
        mesh->face_area[fi] = face_measure;

        #ifdef DEBUG_MODE
        // store face vertex coordinates
        mesh->edge_coords_offsets[fi] = (hsize_t)face_verts.size();
        ensure_edge_coords_capacity(mesh, mesh->num_edge_coord_verts + face_verts.size());
        hsize_t ec = mesh->num_edge_coord_verts;
        for (size_t vi = 0; vi < face_verts.size(); vi++) {
            mesh->edge_coords[(ec + vi) * DIMENSION + 0] = face_verts[vi].x;
            mesh->edge_coords[(ec + vi) * DIMENSION + 1] = face_verts[vi].y;
            #ifdef dim_3D
            mesh->edge_coords[(ec + vi) * DIMENSION + 2] = face_verts[vi].z;
            #endif
        }
        mesh->num_edge_coord_verts += face_verts.size();
        #endif

        mesh->num_faces++;
    }

} // namespace voronoi