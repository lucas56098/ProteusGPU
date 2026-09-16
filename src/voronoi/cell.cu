
// builds a single cell (cell.h)

#include "cell.h"
#include "geometry.h"
#include "voronoi.h"
#include <cmath>
#include <iostream>

namespace voronoi {

    template <typename VERT>
    HD static inline auto ith_plane(const VERT* triangles, int t, int i) -> decltype(triangles[0].x);
    template <typename VERT, typename IDXP>
    HD static inline bool vert_references_plane(const VERT* triangles, int t_idx, IDXP p);
    HD static void        write_face(VMesh*           mesh,
                                     uint64_t         fi,
                                     int              neighbor_id,
                                     double           face_measure,
                                     const double4_t* face_verts,
                                     int              n_face_verts,
                                     double4_t        seed,
                                     double4_t        neighbor);

#ifdef USE_MPI
    // (2 x farthest vertex)^2, capped at the size of the box
    HD inline void store_security_d2(VMesh* mesh, uint64_t k, double r2_num, double r2_denom) {
        const double s      = 1.0 + 2.0 * mesh->buff;
        const double d2_cap = 12.0 * s * s;
        double       d2     = (r2_denom > 0.0) ? 4.0 * r2_num / r2_denom : d2_cap;
        if (!(d2 <= d2_cap)) d2 = d2_cap;
        mesh->security_d2[k] = d2;
    }
#endif

    // builds cell k: start from the box, cut with one neighbour after the other
    template <int K, int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void compute_single_voronoi_cell(int                 k,
                                        int                 seed_id,
                                        double*             d_stored_points,
                                        const knn_problem*  knn,
                                        Status*             stat,
                                        VMesh*              mesh,
                                        unsigned long long* face_offset,
                                        int*                overflow_flag) {

        // K nearest points, nearest first
        unsigned int local_knn[K];
        knn::knn_for_point<K>(seed_id, knn, local_knn);

        BasicConvexCell<MAX_P, MAX_T, IDX, VERT> cell(seed_id, d_stored_points, &(stat[k]), mesh->buff);

        int __attribute__((unused))  v_terminate = K - 1;
        bool __attribute__((unused)) early_break = false;
        for (int v = 0; v < K; v++) {
            const unsigned int z = local_knn[v];
            cell.clip_by_plane(z);
            if (stat[k] != success) {
                v_terminate = v;
                break;
            }

            // cell is final, farther points cannot reach it
            if (v >= 2 * DIMENSION &&
                cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * z))) {
                v_terminate = v;
                early_break = true;
                break;
            }
        }

        // K neighbours were not enough
        if (!cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * local_knn[K - 1]))) {
            stat[k] = security_radius_not_reached;
        }

#ifdef USE_MPI
        if (stat[k] == success) {
            double r2_num, r2_denom;
            cell.max_vertex_r2_ratio(&r2_num, &r2_denom);
            store_security_d2(mesh, (uint64_t)k, r2_num, r2_denom);
            // cell may reach into data this rank does not have
            if (!cell_certified_within_data(cell.voro_seed, r2_num, r2_denom, mesh->data_lo, mesh->data_hi)) {
                stat[k] = security_radius_beyond_data;
            }
        }
        (void)early_break;
        (void)v_terminate;
#endif

        if (stat[k] == success) {
            // take a block of face slots and write the cell
            const int      fc        = count_cell_faces(cell);
            const uint64_t my_offset = (uint64_t)portable_atomicAdd(face_offset, (unsigned long long)fc);
            if (my_offset + (uint64_t)fc > mesh->face_capacity) {
                portable_atomicExch(overflow_flag, 1);
                return;
            }
            mesh->face_ptr[k]    = my_offset;
            mesh->face_counts[k] = extract_cell_all(cell, mesh, (uint64_t)k);
        }
    }

    // a plane with DIMENSION vertices on it is a face
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD int count_cell_faces(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell) {
        int count = 0;
        for (int p = 0; p < cell.nb_v; p++) {
            int refs = 0;
            for (int i = 0; i < cell.nb_t; i++) {
                if (vert_references_plane(cell.triangle, i, (IDX)p)) {
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

    // writes seed, volume, centre and the faces; returns the face count
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD uint64_t extract_cell_all(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell,
                                 VMesh*                                          mesh,
                                 uint64_t                                        cell_index) {
        const double3 seed      = {cell.voro_seed.x, cell.voro_seed.y, cell.voro_seed.z};
        mesh->seeds[cell_index] = seed;

#ifdef dim_2D
        // corner points
        double4_t vertices_2d[MAX_T];
        for (int vi = 0; vi < cell.nb_t; vi++)
            vertices_2d[vi] = cell.compute_vertex_point(cell.triangle[vi], true);

        double cx                 = cell.voro_seed.x;
        double cy                 = cell.voro_seed.y;
        mesh->volumes[cell_index] = compute_cell_area_centroid_2d(cell, vertices_2d, cx, cy);
        mesh->com[cell_index]     = {cx, cy, 0.0};

        uint64_t  fi = mesh->face_ptr[cell_index];
        double4_t face_verts[2];
        int       n_fv;
        for (int p = 0; p < cell.nb_v; p++) {
            if (!collect_face_vertices(cell, p, vertices_2d, face_verts, &n_fv)) continue;
            const double face_measure = compute_face_measure(face_verts, n_fv, cell.voro_seed, nullptr);
            const int    neighbor_id  = cell.plane_vid[p];
            double4_t    neighbor     = make_double4_t(0.0, 0.0, 0.0, 0.0);
            if (neighbor_id >= 0) { neighbor = point_from_ptr(cell.pts + DIMENSION * neighbor_id); }
            write_face(mesh, fi, neighbor_id, face_measure, face_verts, n_fv, cell.voro_seed, neighbor);
            fi++;
        }
        return fi - mesh->face_ptr[cell_index];
#else
        double    total_volume = 0.0;
        double    wx = 0.0, wy = 0.0, wz = 0.0;
        uint64_t  fi = mesh->face_ptr[cell_index];
        double4_t face_verts[MAX_T];

        // vertices on plane p
        for (int p = 0; p < cell.nb_v; p++) {
            int face_vert_indices[MAX_T];
            int n_fvi = 0;
            for (int i = 0; i < cell.nb_t; i++) {
                if (vert_references_plane(cell.triangle, i, (IDX)p)) { face_vert_indices[n_fvi++] = i; }
            }
            if (n_fvi < DIMENSION) continue;

            int  ordered[MAX_T];
            bool used[MAX_T];
            for (int k = 0; k < n_fvi; k++)
                used[k] = false;
            ordered[0]    = face_vert_indices[0];
            used[0]       = true;
            int n_ordered = 1;

            // walk around the face: the next vertex shares the other planes of the last one
            for (int step = 1; step < n_fvi; step++) {
                const int last = ordered[n_ordered - 1];
                IDX       others_last[DIMENSION - 1];
                int       cnt = 0;
                for (int d = 0; d < DIMENSION; d++) {
                    const IDX pl = ith_plane(cell.triangle, last, d);
                    if (pl != (IDX)p) others_last[cnt++] = pl;
                }
                bool found = false;
                for (int j = 0; j < n_fvi; j++) {
                    if (used[j]) continue;
                    const int candidate = face_vert_indices[j];
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

            const int n_fv = n_ordered;

            for (int k = 0; k < n_fv; k++) {
                face_verts[k] = cell.compute_vertex_point(cell.triangle[ordered[k]], true);
            }

            // face and seed give a fan of tetrahedra: volume and centre
            orient_face_outward(face_verts, n_fv, cell.voro_seed);
            double face_measure = 0.0;
            compute_face_area_and_volume_centroid(
                face_verts, n_fv, cell.voro_seed, face_measure, total_volume, wx, wy, wz);

            const int neighbor_id = cell.plane_vid[p];
            double4_t neighbor    = make_double4_t(0.0, 0.0, 0.0, 0.0);
            if (neighbor_id >= 0) { neighbor = point_from_ptr(cell.pts + DIMENSION * neighbor_id); }
            write_face(mesh, fi, neighbor_id, face_measure, face_verts, n_fv, cell.voro_seed, neighbor);
            fi++;
        }

        if (fabs(total_volume) > 1e-30) {
            const double inv_vol  = 1.0 / total_volume;
            mesh->com[cell_index] = {wx * inv_vol, wy * inv_vol, wz * inv_vol};
        } else {
            mesh->com[cell_index] = seed;
        }
        mesh->volumes[cell_index] = fabs(total_volume);
        return fi - mesh->face_ptr[cell_index];
#endif
    }

    // ball of 2 x the farthest vertex inside the extent
    HD bool cell_certified_within_data(
        double4_t seed, double r2_num, double r2_denom, const double* data_lo, const double* data_hi) {
        if (!(data_hi[0] > data_lo[0])) return true; // no extent, nothing to check

        double safe = fmin(seed.x - data_lo[0], data_hi[0] - seed.x);
        safe        = fmin(safe, fmin(seed.y - data_lo[1], data_hi[1] - seed.y));
#ifdef dim_3D
        safe = fmin(safe, fmin(seed.z - data_lo[2], data_hi[2] - seed.z));
#endif
        if (safe <= 0.0) return false;

        return (4.0 * r2_num <= safe * safe * r2_denom);
    }

    // the face arrays do not grow
    void ensure_face_capacity(VMesh* mesh, uint64_t needed) {
        if (needed <= mesh->face_capacity) return;
        proteus_mpi::exit_failure("VORONOI: Error! face count %llu exceeds pre-allocated face capacity %llu. "
                                  "Increase _FACE_CAPACITY_MULT_ in Config.sh.\n",
                                  (unsigned long long)needed,
                                  (unsigned long long)mesh->face_capacity);
    }

    // starts as the box plus band: the wall planes and their corners
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::BasicConvexCell(int     p_seed,
                                                                 double* p_pts,
                                                                 Status* p_status,
                                                                 double  p_buff) {
        pts       = p_pts;
        buff      = p_buff;
        status    = p_status;
        *status   = success;
        voro_seed = point_from_ptr(pts + DIMENSION * p_seed);

        first_boundary = END_OF_LIST;
        for (int i = 0; i < MAX_P; i++) {
            boundary_next[i] = END_OF_LIST;
            plane_vid[i]     = -1;
        }

#ifdef dim_2D
        triangle[0] = make_vert<VERT>(2, 0);
        triangle[1] = make_vert<VERT>(1, 2);
        triangle[2] = make_vert<VERT>(3, 1);
        triangle[3] = make_vert<VERT>(0, 3);
        nb_v        = 4;
        nb_t        = 4;
#else
        triangle[0] = make_vert<VERT>(2, 5, 0);
        triangle[1] = make_vert<VERT>(5, 3, 0);
        triangle[2] = make_vert<VERT>(1, 5, 2);
        triangle[3] = make_vert<VERT>(5, 1, 3);
        triangle[4] = make_vert<VERT>(4, 2, 0);
        triangle[5] = make_vert<VERT>(4, 0, 3);
        triangle[6] = make_vert<VERT>(2, 4, 1);
        triangle[7] = make_vert<VERT>(4, 3, 1);
        nb_v        = 6;
        nb_t        = 8;
#endif
    }

    // cuts the cell with the plane halfway to point vid
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::clip_by_plane(int vid) {

        const int cur_v = new_halfplane(vid);
        if (*status == vertex_overflow) { return; }

        const double4_t eqn = plane_for(cur_v);
        // park the vertices on the far side behind nb_t
        nb_r  = 0;
        int i = 0;
        while (i < nb_t) {
            if (vert_is_in_conflict(triangle[i], eqn)) {
                nb_t--;
                VERT tmp       = triangle[i];
                triangle[i]    = triangle[nb_t];
                triangle[nb_t] = tmp;
                nb_r++;
            } else {
                i++;
            }
        }
        if (*status == needs_exact_predicates) { return; }

        // plane cut nothing, drop it again
        if (nb_r == 0) {
            nb_v--;
            return;
        }

        compute_boundary();
        if (*status != success) { return; }
        if (first_boundary == END_OF_LIST) { return; }

        // the ring around them carries the new vertices
        IDX cir = first_boundary;
        do {
            const IDX nxt = boundary_next[cir];
            if (nxt == END_OF_LIST) {
                *status = inconsistent_boundary;
                return;
            }
#ifdef dim_2D
            new_vertex((IDX)cur_v, cir);
#else
            new_vertex((IDX)cur_v, cir, nxt);
#endif
            if (*status != success) return;
            cir = nxt;
        } while (cir != first_boundary);
    }

    // a point farther than 2 x the farthest vertex cannot cut the cell
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD bool BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::is_security_radius_reached(double4_t last_neig) const {
        double max_num, max_denom;
        max_vertex_r2_ratio(&max_num, &max_denom);

        const double4_t diff = minus4(last_neig, voro_seed);
        const double    d2   = dot3(diff, diff);
        return (d2 * max_denom > 4.0 * max_num);
    }

    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::max_vertex_r2_ratio(double* out_num, double* out_denom) const {
        double max_num   = 0.0;
        double max_denom = 1.0;
        for (int i = 0; i < nb_t; i++) {
            const double4_t pc = compute_vertex_point(triangle[i], false);
            const double    dx = pc.x - voro_seed.x * pc.w;
            const double    dy = pc.y - voro_seed.y * pc.w;
#ifdef dim_3D
            const double dz  = pc.z - voro_seed.z * pc.w;
            const double num = dx * dx + dy * dy + dz * dz;
#else
            const double num = dx * dx + dy * dy;
#endif
            const double denom = pc.w * pc.w;
            if (num * max_denom > max_num * denom) {
                max_num   = num;
                max_denom = denom;
            }
        }
        *out_num   = max_num;
        *out_denom = max_denom;
    }

    // plane equation, computed on every use
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD double4_t BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::plane_for(int p) const {

        // box walls, from -buff to 1 + buff
        if (p < 2 * DIMENSION) {
            constexpr double eps   = 1e-14;
            const double     w_min = buff + eps;
            const double     w_max = 1.0 + buff + eps;
            switch (p) {
            case 0:
                return make_double4_t(1.0, 0.0, 0.0, w_min);
            case 1:
                return make_double4_t(-1.0, 0.0, 0.0, w_max);
            case 2:
                return make_double4_t(0.0, 1.0, 0.0, w_min);
            case 3:
                return make_double4_t(0.0, -1.0, 0.0, w_max);
#ifdef dim_3D
            case 4:
                return make_double4_t(0.0, 0.0, 1.0, w_min);
            case 5:
                return make_double4_t(0.0, 0.0, -1.0, w_max);
#endif
            }
        }

        // bisector: halfway to the point
        const double4_t B    = point_from_ptr(pts + DIMENSION * plane_vid[p]);
        const double4_t dir  = minus4(voro_seed, B);
        const double4_t ave2 = plus4(voro_seed, B);
        const double    dot  = dot3(ave2, dir);
        return make_double4_t(dir.x, dir.y, dir.z, -dot * 0.5);
    }

    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD int BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::new_halfplane(int vid) {
        if (nb_v >= MAX_P) {
            *status = vertex_overflow;
            return -1;
        }
        plane_vid[nb_v] = vid;
        nb_v++;
        return nb_v - 1;
    }

    // which side of eqn the vertex is on
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD bool BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::vert_is_in_conflict(VERT v, double4_t eqn) const {

        const double4_t pi1 = plane_for(v.x);
        const double4_t pi2 = plane_for(v.y);
#ifdef dim_2D
        const double det = det3x3(pi1.x, pi2.x, eqn.x, pi1.y, pi2.y, eqn.y, pi1.w, pi2.w, eqn.w);

        const double maxx    = fmax(fmax(fabs(pi1.x), fabs(pi2.x)), fabs(eqn.x));
        const double maxy    = fmax(fmax(fabs(pi1.y), fabs(pi2.y)), fabs(eqn.y));
        const double maxw    = fmax(fmax(fabs(pi1.w), fabs(pi2.w)), fabs(eqn.w));
        const double max_max = fmax(fmax(maxx, maxy), maxw);
        double       eps     = 1e-14 * maxx * maxy * maxw;
        eps *= max_max;
#else
        const double4_t pi3 = plane_for(v.z);

        const double det = det4x4(pi1.x,
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

        const double maxx = fmax(fmax(fabs(pi1.x), fabs(pi2.x)), fmax(fabs(pi3.x), fabs(eqn.x)));
        const double maxy = fmax(fmax(fabs(pi1.y), fabs(pi2.y)), fmax(fabs(pi3.y), fabs(eqn.y)));
        const double maxz = fmax(fmax(fabs(pi1.z), fabs(pi2.z)), fmax(fabs(pi3.z), fabs(eqn.z)));
        double       eps  = 1e-12 * maxx * maxy * maxz;
        double       min_max, max_max;
        get_minmax3(min_max, max_max, maxx, maxy, maxz);
        eps *= (max_max * max_max);
#endif

        // eps grows with the size of the numbers, below it the sign means nothing
        if (fabs(det) < eps) { *status = needs_exact_predicates; }
        return (det > 0.0);
    }

    // the cut away vertices form one piece, this links the planes around it into a ring
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::compute_boundary() {

#ifdef dim_2D
        for (int i = 0; i < MAX_P; i++) {
            boundary_next[i] = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        // a plane seen once is an end of the hole
        int line_count[MAX_P];
        for (int i = 0; i < MAX_P; i++) {
            line_count[i] = 0;
        }
        for (int r = 0; r < nb_r; r++) {
            const VERT e = triangle[nb_t + r];
            line_count[e.x]++;
            line_count[e.y]++;
        }

        IDX boundary_lines[2];
        int nb_boundary = 0;
        for (int p = 0; p < nb_v; p++) {
            if (line_count[p] == 1) {
                if (nb_boundary < 2) { boundary_lines[nb_boundary++] = (IDX)p; }
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
        for (int i = 0; i < MAX_P; i++) {
            boundary_next[i] = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        int nb_iter = 0;
        IDX t       = nb_t;

        constexpr int max_iter = (MAX_T > 255) ? (100 * MAX_T) : 10000;

        // take one cut away vertex at a time and grow the ring
        // an edge already in the ring the other way round is inside the hole and drops out
        // a vertex that would split the ring waits for a later turn
        while (nb_r > 0) {
            if (nb_iter++ > max_iter) {
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

            VERT tmp                  = triangle[t];
            triangle[t]               = triangle[nb_t + nb_r - 1];
            triangle[nb_t + nb_r - 1] = tmp;
            t                         = nb_t;
            nb_r--;
        }
#endif
    }

    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::new_vertex(IDX i, IDX j, IDX k) {
        if (nb_t + 1 >= MAX_T) {
            *status = triangle_overflow;
            return;
        }
#ifdef dim_2D
        (void)k;
        const double4_t hi = plane_for(i);
        const double4_t hj = plane_for(j);
        const double    rw = det2x2(hi.x, hi.y, hj.x, hj.y);
        if (rw > 0) {
            triangle[nb_t] = make_vert<VERT>(j, i);
        } else {
            triangle[nb_t] = make_vert<VERT>(i, j);
        }
#else
        triangle[nb_t] = make_vert<VERT>(i, j, k);
#endif
        nb_t++;
    }

    // solves the DIMENSION plane equations, w is the determinant
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD double4_t BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::compute_vertex_point(VERT v, bool persp_divide) const {
        const double4_t pi1 = plane_for(v.x);
        const double4_t pi2 = plane_for(v.y);
        double4_t       result;
#ifdef dim_2D
        result.x = -det2x2(pi1.w, pi1.y, pi2.w, pi2.y);
        result.y = -det2x2(pi1.x, pi1.w, pi2.x, pi2.w);
        result.z = 0;
        result.w = det2x2(pi1.x, pi1.y, pi2.x, pi2.y);
        if (persp_divide) { return make_double4_t(result.x / result.w, result.y / result.w, 0, 1); }
#else
        const double4_t pi3 = plane_for(v.z);
        result.x            = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y            = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z            = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w            = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
        if (persp_divide) {
            const double inv_w = 1.0 / result.w;
            return make_double4_t(result.x * inv_w, result.y * inv_w, result.z * inv_w, 1);
        }
#endif
        return result;
    }

    // vertices on plane p, ordered around the face
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD bool collect_face_vertices(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell,
                                  int                                             p,
                                  const double4_t*                                vertices,
                                  double4_t*                                      face_verts,
                                  int*                                            n_face_verts) {
#ifdef dim_2D
        int n_fvi = 0;
        for (int i = 0; i < cell.nb_t; i++) {
            if (vert_references_plane(cell.triangle, i, (IDX)p)) {
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
            if (vert_references_plane(cell.triangle, i, (IDX)p)) { face_vert_indices[n_fvi++] = i; }
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
            const int last = ordered[n_ordered - 1];

            IDX others_last[DIMENSION - 1];
            int cnt = 0;
            for (int d = 0; d < DIMENSION; d++) {
                const IDX pl = ith_plane(cell.triangle, last, d);
                if (pl != (IDX)p) others_last[cnt++] = pl;
            }

            bool found = false;
            for (int j = 0; j < n_fvi; j++) {
                if (used[j]) continue;
                const int candidate = face_vert_indices[j];
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

    // one face: neighbour, area, centre offset
    HD static void write_face(VMesh*           mesh,
                              uint64_t         fi,
                              int              neighbor_id,
                              double           face_measure,
                              const double4_t* face_verts,
                              int              n_face_verts,
                              double4_t        seed,
                              double4_t        neighbor) {
        (void)face_verts;
        (void)n_face_verts;
        (void)seed;
        (void)neighbor;

        // faces name the neighbour cell, not the point
        const int remapped      = (neighbor_id >= 0) ? (int)mesh->sid_to_neighbor[neighbor_id] : neighbor_id;
        mesh->neighbor_cell[fi] = remapped;
        mesh->face_area[fi]     = face_measure;

#ifdef MOVING_MESH
        double fmx = 0.0, fmy = 0.0, fmz = 0.0;
        compute_face_centroid(face_verts, n_face_verts, fmx, fmy, fmz);

        if (neighbor_id >= 0) {
            const double3 raw_normal = {neighbor.x - seed.x, neighbor.y - seed.y, neighbor.z - seed.z};
            const geom    g_local    = compute_geom(raw_normal);
            const double  ox         = fmx - 0.5 * (seed.x + neighbor.x);
            const double  oy         = fmy - 0.5 * (seed.y + neighbor.y);
#ifdef dim_2D
            mesh->f_mid_local[fi] = ox * g_local.m.x + oy * g_local.m.y;
#else
            const double oz               = fmz - 0.5 * (seed.z + neighbor.z);
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

    template <typename VERT>
    HD static inline auto ith_plane(const VERT* triangles, int t, int i) -> decltype(triangles[0].x) {
        const VERT& v = triangles[t];
#ifdef dim_2D
        return (i == 0) ? v.x : v.y;
#else
        return (i == 0) ? v.x : ((i == 1) ? v.y : v.z);
#endif
    }

    template <typename VERT, typename IDXP>
    HD static inline bool vert_references_plane(const VERT* triangles, int t_idx, IDXP p) {
        for (int d = 0; d < DIMENSION; d++) {
            if (ith_plane(triangles, t_idx, d) == p) return true;
        }
        return false;
    }

} // namespace voronoi
