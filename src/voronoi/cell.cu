
// builds a single cell (cell.h)

#include "cell.h"
#include "geometry.h"
#include "predicates.h"
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

    // the determinant of the plane rows is off by at most about 20 roundings times the product of the row
    // scales, the rounding of the planes included; below this bound, with a wide margin, its sign means nothing
    constexpr double DET_REL_EPS = 1e-13;

    // a point at exactly twice the distance of the farthest corner can still cut it in a tie, and corners are
    // rounded; every test of how far a cell reaches keeps this margin
    constexpr double REACH_SLACK = 1.0 + 1e-6;

    // planes with w^2 above this times the product of their squared normals meet in a vertex the plane
    // equations give to about 1e-12; an exact cell computes the others from the points
    constexpr double WELL_POSED = 1e-8;

    HD inline void copy_point(const double4_t& a, double* out) {
        out[0] = a.x;
        out[1] = a.y;
#ifdef dim_3D
        out[2] = a.z;
#endif
    }

    // (2 x farthest vertex)^2 with the margin, capped well past any cell that fits the box
    HD inline double security_d2_of(double r2_num, double r2_denom) {
        constexpr double d2_cap = 1e30;
        double           d2     = (r2_denom > 0.0) ? 4.0 * REACH_SLACK * r2_num / r2_denom : d2_cap;
        if (!(d2 <= d2_cap)) d2 = d2_cap;
        return d2;
    }

    HD inline void store_security_d2(VMesh* mesh, uint64_t k, double r2_num, double r2_denom) {
        mesh->security_d2[k] = security_d2_of(r2_num, r2_denom);
    }

    // builds cell k: start from the box, cut with one neighbour after the other
    template <int K, int MAX_P, int MAX_T, typename IDX, typename VERT, bool WALK>
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

        BasicConvexCell<MAX_P, MAX_T, IDX, VERT> cell(seed_id, d_stored_points, &(stat[k]));

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

        // K neighbours were not enough; the walk finds the rest
        if (!cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * local_knn[K - 1]))) {
            if (WALK && stat[k] == success) {
                close_cell_by_tree_walk(cell, seed_id, knn);
            } else {
                stat[k] = security_radius_not_reached;
            }
        }

        // whether this rank has every point inside it is checked after the build
        if (stat[k] == success) {
            double r2_num, r2_denom;
            cell.max_vertex_r2_ratio(&r2_num, &r2_denom);
            store_security_d2(mesh, (uint64_t)k, r2_num, r2_denom);
        }

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

    // corners of the cell and their squared distance to the seed; false if a corner is not reliable
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD bool cell_corners(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell,
                         POINT_TYPE*                                     corner,
                         double*                                         corner_r2,
                         double*                                         r2_max) {
        *r2_max = 0.0;
        for (int i = 0; i < cell.nb_t; i++) {
            const VERT      v = cell.triangle[i];
            const double4_t h = cell.compute_vertex_point(v, false);

            // nearly parallel planes put the corner far off and inexact
            const double4_t p1 = cell.plane_for(v.x);
            const double4_t p2 = cell.plane_for(v.y);
            double          n2 = dot3(p1, p1) * dot3(p2, p2);
#ifdef dim_3D
            const double4_t p3 = cell.plane_for(v.z);
            n2 *= dot3(p3, p3);
#endif
            if (!(h.w * h.w > 1e-16 * n2)) return false;

            corner[i].x     = h.x / h.w;
            corner[i].y     = h.y / h.w;
            const double dx = corner[i].x - cell.voro_seed.x;
            const double dy = corner[i].y - cell.voro_seed.y;
#ifdef dim_3D
            corner[i].z     = h.z / h.w;
            const double dz = corner[i].z - cell.voro_seed.z;
            corner_r2[i]    = dx * dx + dy * dy + dz * dz;
#else
            corner_r2[i] = dx * dx + dy * dy;
#endif
            if (!(corner_r2[i] < 1e300)) return false;
            if (corner_r2[i] > *r2_max) *r2_max = corner_r2[i];
        }
        return true;
    }

    // true if some point of the box may be closer to a corner than the seed is
    HD inline bool box_may_cut(const POINT_TYPE& lo,
                               const POINT_TYPE& hi,
                               const POINT_TYPE& seed,
                               const POINT_TYPE* corner,
                               const double*     corner_r2,
                               int               n_corner,
                               double            r2_max) {
        // every corner sphere lies inside the ball of twice the farthest corner
        if (knn::dist2_box(lo, hi, seed) > 4.0 * r2_max * REACH_SLACK) return false;
        for (int i = 0; i < n_corner; i++) {
            if (!(knn::dist2_box(lo, hi, corner[i]) > corner_r2[i] * REACH_SLACK)) return true;
        }
        return false;
    }

    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD bool cell_has_plane(const BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell, int q) {
        for (int p = 2 * DIMENSION; p < cell.nb_v; p++) {
            if (cell.plane_vid[p] == q) return true;
        }
        return false;
    }

    // walks the tree, nearer child first, and clips with every point a corner sphere may hold;
    // a cut shrinks the spheres, so a node skipped once stays skipped
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void
    close_cell_by_tree_walk(BasicConvexCell<MAX_P, MAX_T, IDX, VERT>& cell, int seed_id, const knn_problem* knn) {
        const int n = knn->len_pts;
        if (n < 2) return;
        const TreeNode* nodes = knn->d_nodes;
        const int       leaf0 = n - 1;

        POINT_TYPE seed;
        seed.x = cell.voro_seed.x;
        seed.y = cell.voro_seed.y;
#ifdef dim_3D
        seed.z = cell.voro_seed.z;
#endif

        POINT_TYPE corner[MAX_T];
        double     corner_r2[MAX_T];
        double     r2_max;
        if (!cell_corners(cell, corner, corner_r2, &r2_max)) {
            *cell.status = security_radius_not_reached;
            return;
        }

        // a stack entry is 2 x parent + side, the box of a node sits in its parent
        int stack[knn::TREE_STACK];
        int sp      = 0;
        int near    = (knn::dist2_box(nodes[0].lo[1], nodes[0].hi[1], seed) <
                    knn::dist2_box(nodes[0].lo[0], nodes[0].hi[0], seed))
                          ? 1
                          : 0;
        stack[sp++] = 1 - near;
        stack[sp++] = near;
        while (sp > 0) {
            const int       e      = stack[--sp];
            const TreeNode& parent = nodes[e >> 1];
            const int       side   = e & 1;
            if (!box_may_cut(parent.lo[side], parent.hi[side], seed, corner, corner_r2, cell.nb_t, r2_max)) continue;

            const int c = parent.child[side];
            if (c >= leaf0) {
                const int q = c - leaf0;
                if (q == seed_id || cell_has_plane(cell, q)) continue;
                const int planes_before = cell.nb_v;
                cell.clip_by_plane(q);
                if (*cell.status != success) return;
                if (cell.nb_v != planes_before && !cell_corners(cell, corner, corner_r2, &r2_max)) {
                    *cell.status = security_radius_not_reached;
                    return;
                }
                continue;
            }

            // the nearer child comes off the stack first
            const TreeNode& nd = nodes[c];
            near = (knn::dist2_box(nd.lo[1], nd.hi[1], seed) < knn::dist2_box(nd.lo[0], nd.hi[0], seed)) ? 1 : 0;
            if (sp + 2 > knn::TREE_STACK) {
                *cell.status = security_radius_not_reached;
                return;
            }
            stack[sp++] = 2 * c + 1 - near;
            stack[sp++] = 2 * c + near;
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

    // the face arrays do not grow
    void ensure_face_capacity(VMesh* mesh, uint64_t needed) {
        if (needed <= mesh->face_capacity) return;
        mpi::exit_failure("VORONOI: Error! face count %llu exceeds pre-allocated face capacity %llu. "
                          "Increase _FACE_CAPACITY_MULT_ in Config.sh.\n",
                          (unsigned long long)needed,
                          (unsigned long long)mesh->face_capacity);
    }

    // starts as the box plus margin: the wall planes and their corners
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::BasicConvexCell(int     p_seed,
                                                                 double* p_pts,
                                                                 Status* p_status,
                                                                 bool    p_exact) {
        pts       = p_pts;
        status    = p_status;
        *status   = success;
        exact     = p_exact;
        voro_seed = point_from_ptr(pts + DIMENSION * p_seed);
        far_num   = 0.0;
        far_denom = 1.0;
        far_valid = false;

        // a later plane gets its point when it is added, the ring is set up by compute_boundary
        first_boundary = END_OF_LIST;
        for (int i = 0; i < 2 * DIMENSION; i++) {
            plane_vid[i] = -1;
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
        const bool far_was_valid = far_valid;
        far_valid                = false;

        const int cur_v = new_halfplane(vid);
        if (*status == vertex_overflow) { return; }

        double          eqn_scale;
        const double4_t eqn = plane_for(cur_v, &eqn_scale);
        // park the vertices on the far side behind nb_t
        nb_r  = 0;
        int i = 0;
        while (i < nb_t) {
            if (vert_is_in_conflict(triangle[i], cur_v, eqn, eqn_scale)) {
                nb_t--;
                VERT tmp       = triangle[i];
                triangle[i]    = triangle[nb_t];
                triangle[nb_t] = tmp;
                nb_r++;
            } else {
                i++;
            }
        }
        if (*status != success) { return; }

        // plane cut nothing, drop it again; the vertices did not change
        if (nb_r == 0) {
            nb_v--;
            far_valid = far_was_valid;
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

    // a point farther than 2 x the farthest vertex, past the margin, cannot cut the cell
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD bool BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::is_security_radius_reached(double4_t last_neig) {
        if (!far_valid) {
            max_vertex_r2_ratio(&far_num, &far_denom);
            far_valid = true;
        }

        const double4_t diff = minus4(last_neig, voro_seed);
        const double    d2   = dot3(diff, diff);
        return (d2 * far_denom > 4.0 * REACH_SLACK * far_num);
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

    // plane equation, computed on every use; scale is |n|_1 plus a bound on |w| and its rounding
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD double4_t BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::plane_for(int p, double* scale) const {

        // box walls, from -margin to 1 + margin
        if (p < 2 * DIMENSION) {
            constexpr double w_min = CELL_WALL_LO;
            constexpr double w_max = CELL_WALL_HI;
            if (scale) *scale = 1.0 + ((p % 2 == 0) ? w_min : w_max);
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
        if (scale) {
            *scale = fabs(dir.x) + fabs(dir.y) + fabs(dir.z) +
                     0.5 * (fabs(ave2.x * dir.x) + fabs(ave2.y * dir.y) + fabs(ave2.z * dir.z));
        }
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
    HD bool BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::vert_is_in_conflict(VERT      v,
                                                                          int       plane,
                                                                          double4_t eqn,
                                                                          double    eqn_scale) const {
#ifndef __CUDA_ARCH__
        if (exact) return exact_conflict(v, plane);
#else
        (void)plane;
#endif

        double          s1, s2;
        const double4_t pi1 = plane_for(v.x, &s1);
        const double4_t pi2 = plane_for(v.y, &s2);
#ifdef dim_2D
        const double det   = det3x3(pi1.x, pi2.x, eqn.x, pi1.y, pi2.y, eqn.y, pi1.w, pi2.w, eqn.w);
        const double bound = DET_REL_EPS * s1 * s2 * eqn_scale;
#else
        double          s3;
        const double4_t pi3 = plane_for(v.z, &s3);

        const double det   = det4x4(pi1.x,
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
        const double bound = DET_REL_EPS * s1 * s2 * s3 * eqn_scale;
#endif

        // too close to a tie for doubles: the CPU decides it exactly
        if (!(fabs(det) > bound)) { *status = needs_exact_predicates; }
        return (det > 0.0);
    }

    // the cut away vertices form one piece, this links the planes around it into a ring
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::compute_boundary() {

#ifdef dim_2D
        for (int i = 0; i < nb_v; i++) {
            boundary_next[i] = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        // a plane seen once is an end of the hole
        int line_count[MAX_P];
        for (int i = 0; i < nb_v; i++) {
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
        for (int i = 0; i < nb_v; i++) {
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
        double          rw = det2x2(hi.x, hi.y, hj.x, hj.y);
#ifndef __CUDA_ARCH__
        if (exact) rw = exact_vertex_turns_left(i, j) ? 1.0 : -1.0;
#endif
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
#ifndef __CUDA_ARCH__
        if (exact && !(result.w * result.w > WELL_POSED * dot3(pi1, pi1) * dot3(pi2, pi2))) {
            return exact_vertex_point(v);
        }
#endif
        if (persp_divide) { return make_double4_t(result.x / result.w, result.y / result.w, 0, 1); }
#else
        const double4_t pi3 = plane_for(v.z);
        result.x            = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y            = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z            = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w            = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
#ifndef __CUDA_ARCH__
        if (exact && !(result.w * result.w > WELL_POSED * dot3(pi1, pi1) * dot3(pi2, pi2) * dot3(pi3, pi3))) {
            return exact_vertex_point(v);
        }
#endif
        if (persp_divide) {
            const double inv_w = 1.0 / result.w;
            return make_double4_t(result.x * inv_w, result.y * inv_w, result.z * inv_w, 1);
        }
#endif
        return result;
    }

    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    HD void BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::plane_point(int p, double* out) const {
        copy_point(voro_seed, out);
        if (p >= 2 * DIMENSION) {
            const double* q = pts + DIMENSION * plane_vid[p];
            for (int d = 0; d < DIMENSION; d++)
                out[d] = q[d];
            return;
        }

        // the mirror image across the wall: the low wall of an axis first, then the high one
        const int axis = p / 2;
        out[axis]      = (p % 2 == 0) ? -2.0 * CELL_WALL_LO - out[axis] : 2.0 * CELL_WALL_HI - out[axis];
    }

    // the vertex is the centre of the sphere through the seed and the points of its planes; the plane cuts
    // it away if its point lies inside
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    bool BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::exact_conflict(VERT v, int plane) const {
        double pt[DIMENSION + 2][DIMENSION];
        copy_point(voro_seed, pt[0]);
        plane_point(v.x, pt[1]);
        plane_point(v.y, pt[2]);
#ifdef dim_3D
        plane_point(v.z, pt[3]);
#endif
        plane_point(plane, pt[DIMENSION + 1]);
#ifdef dim_2D
        const double* p[3] = {pt[0], pt[1], pt[2]};
#else
        const double* p[4] = {pt[0], pt[1], pt[2], pt[3]};
#endif
        const int side = exact::in_circumsphere(p, pt[DIMENSION + 1]);
        if (side < 0) *status = coincident_points;
        return side == 1;
    }

    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    double4_t BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::exact_vertex_point(VERT v) const {
        double pt[DIMENSION + 1][DIMENSION];
        copy_point(voro_seed, pt[0]);
        plane_point(v.x, pt[1]);
        plane_point(v.y, pt[2]);
#ifdef dim_2D
        const double* p[3] = {pt[0], pt[1], pt[2]};
#else
        plane_point(v.z, pt[3]);
        const double* p[4] = {pt[0], pt[1], pt[2], pt[3]};
#endif
        double c[3] = {0.0, 0.0, 0.0};
        exact::circumcentre(p, c);
        return make_double4_t(c[0], c[1], c[2], 1.0);
    }

#ifdef dim_2D
    // the sign of det(n_i, n_j) of the plane normals n = seed - point: the orientation of the two points
    // and the seed
    template <int MAX_P, int MAX_T, typename IDX, typename VERT>
    bool BasicConvexCell<MAX_P, MAX_T, IDX, VERT>::exact_vertex_turns_left(IDX i, IDX j) const {
        double pt[3][DIMENSION];
        plane_point(i, pt[0]);
        plane_point(j, pt[1]);
        copy_point(voro_seed, pt[2]);
        const double* p[3] = {pt[0], pt[1], pt[2]};
        return exact::orientation(p) > 0;
    }
#endif

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
        // faces name the neighbour cell, not the point
        const int remapped      = (neighbor_id >= 0) ? (int)mesh->sid_to_neighbor[neighbor_id] : neighbor_id;
        mesh->neighbor_cell[fi] = remapped;
        mesh->face_area[fi]     = face_measure;

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
