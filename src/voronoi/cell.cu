#include "cell.h"
#include "geometry.h"
#include "voronoi.h"
#include <cmath>
#include <iostream>

namespace voronoi {

    // ---- forward declarations ----
    static const uchar END_OF_LIST = 255;

    HD static inline uchar& ith_plane(VERT_TYPE* triangles, uchar t, int i);
    HD static inline uchar  ith_plane(const VERT_TYPE* triangles, uchar t, int i);
    HD static inline bool   vert_references_plane(const VERT_TYPE* triangles, int t_idx, uchar p);
    HD static void          write_face(VMesh*         mesh,
                                       hsize_t        fi,
                                       int            neighbor_id,
                                       double         face_measure,
                                       const double4* face_verts,
                                       int            n_face_verts,
                                       double4        seed,
                                       double4        neighbor);

    // ============================================================
    // Main routines
    // ============================================================

    // build the Voronoi cell for seed `k` by clipping the bounding box against the K nearest
    // neighbours; on success atomically reserves face storage and emits geometry into mesh.
    template <int K, int MAX_P, int MAX_T>
    HD void compute_single_voronoi_cell(int                 k,
                                        int                 seed_id,
                                        double*             d_stored_points,
                                        const knn_problem*  knn,
                                        Status*             stat,
                                        VMesh*              mesh,
                                        unsigned long long* face_offset,
                                        int*                overflow_flag) {

        // gather the K nearest neighbour sids for this seed
        unsigned int local_knn[K];
        knn::knn_for_point<K>(seed_id, knn, local_knn);

        // start from the bounding-box cell, then clip plane-by-plane
        BasicConvexCell<MAX_P, MAX_T> cell(seed_id, d_stored_points, &(stat[k]), mesh->buff);

        // v_terminate / early_break are only read by the MPI completeness piggyback below;
        // attributes silence unused-warnings when built without USE_MPI or on the device side
        int __attribute__((unused))  v_terminate = K - 1;
        bool __attribute__((unused)) early_break = false;
        for (int v = 0; v < K; v++) {
            const unsigned int z = local_knn[v];
            cell.clip_by_plane(z);
            if (stat[k] != success) {
                v_terminate = v;
                break;
            }

            // early exit once the security radius is comfortably reached
            if (v >= 2 * DIMENSION &&
                cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * z))) {
                v_terminate = v;
                early_break = true;
                break;
            }
        }

        // final security check: if even the K-th neighbour doesn't enclose the cell, mark a failure
        if (!cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * local_knn[K - 1]))) {
            stat[k] = security_radius_not_reached;
        }

#ifdef USE_MPI
        // Data-extent soundness guard. is_security_radius_reached only proves the cell fits
        // inside the local K-th's circumscribed sphere; under multi-rank decomposition that
        // sphere can poke past the edge of the data we actually have (own brick + halo W),
        // meaning a true global K-nearest could be sitting outside, unseen. The safe radius
        // for this seed is its smallest distance to any face of [data_lo, data_hi]; if the
        // Voronoi cell's bounding sphere (radius = sqrt(d2)/2 from the deciding K-th) reaches
        // past that, flip to security_radius_not_reached so the widen-W loop iterates.
        // A cell deep inside the brick keeps a large safe radius — no false widens.
        // Fast-tier only (K <= 50): the slow tier in sparse regions routinely needs farther
        // K-nearest by design, and re-flagging it just feeds the perturb / CPU-fallback twice.
        if (stat[k] == success && K <= 50 && mesh->data_hi[0] > mesh->data_lo[0]) {
            const double4 last = point_from_ptr(d_stored_points + DIMENSION * local_knn[v_terminate]);
            const double  dx   = last.x - cell.voro_seed.x;
            const double  dy   = last.y - cell.voro_seed.y;
#ifdef dim_3D
            const double dz = last.z - cell.voro_seed.z;
            const double d2 = dx * dx + dy * dy + dz * dz;
#else
            const double d2 = dx * dx + dy * dy;
#endif
            const double sx   = cell.voro_seed.x;
            const double sy   = cell.voro_seed.y;
            double       safe = sx - mesh->data_lo[0];
            safe              = fmin(safe, mesh->data_hi[0] - sx);
            safe              = fmin(safe, sy - mesh->data_lo[1]);
            safe              = fmin(safe, mesh->data_hi[1] - sy);
#ifdef dim_3D
            const double sz = cell.voro_seed.z;
            safe            = fmin(safe, sz - mesh->data_lo[2]);
            safe            = fmin(safe, mesh->data_hi[2] - sz);
#endif
            if (safe < 0.0 || d2 > 4.0 * safe * safe) { stat[k] = security_radius_not_reached; }
        }
#endif

#ifdef USE_MPI
        // MPI halo-completeness piggyback. Fires only when (a) the cell succeeded via an
        // early break before v=K-1, (b) the deciding clip plane was an outermost-layer MPI ghost,
        // and (c) this is a fast-tier build (K small). The slow tier with K=190 routinely touches
        // the outer halo on geometrically-correct cells, so flagging it triggers spurious widening.
        // The single shared mesh->outer_halo_hit flag is what SENTINEL_OUTER reads — no separate
        // KNN-walk kernel needed. portable_atomicExch is idempotent: many threads writing 1 is fine.
        // Reads halo metadata from mesh->{n_mpi_ghosts,is_outer_layer,pts_mpi_base} (snapshotted
        // host-side from proteus_mpi::halo before the kernel launch) — the proteus_mpi::halo
        // global is host-only and isn't visible to device code.
        if (stat[k] == success && early_break && v_terminate < K - 1 && K <= 50) {
            const int                  n_mpi_ghosts = mesh->n_mpi_ghosts;
            const unsigned char* const is_outer     = mesh->is_outer_layer;
            const int                  pts_mpi_base = mesh->pts_mpi_base;
            if (n_mpi_ghosts > 0 && is_outer != nullptr) {
                const unsigned int orig = knn->d_permutation[local_knn[v_terminate]];
                if ((int)orig >= pts_mpi_base) {
                    const int slot = (int)orig - pts_mpi_base;
                    if (slot >= 0 && slot < n_mpi_ghosts && is_outer[slot]) {
                        portable_atomicExch(mesh->outer_halo_hit, 1);
                    }
                }
            }
        }
#endif

        // on success, reserve face slots atomically and emit the cell geometry
        if (stat[k] == success) {
            const int     fc        = count_cell_faces(cell);
            const hsize_t my_offset = (hsize_t)portable_atomicAdd(face_offset, (unsigned long long)fc);
            if (my_offset + (hsize_t)fc > mesh->face_capacity) {
                portable_atomicExch(overflow_flag, 1);
                return;
            }
            mesh->face_counts[k] = (hsize_t)fc;
            mesh->face_ptr[k]    = my_offset;
            extract_cell_all(cell, mesh, (hsize_t)k);
        }
    }

    // count how many of the cell's planes contribute a real face
    // (a plane contributes iff at least DIMENSION triangles reference it)
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

    // emit cell volume + centroid + per-face data into the global mesh arrays
    template <int MAX_P, int MAX_T>
    HD void extract_cell_all(const BasicConvexCell<MAX_P, MAX_T>& cell, VMesh* mesh, hsize_t cell_index) {
        const double3 seed      = {cell.voro_seed.x, cell.voro_seed.y, cell.voro_seed.z};
        mesh->seeds[cell_index] = seed;

#ifdef dim_2D
        // resolve dual triangles -> primal polygon vertices
        double4 vertices_2d[MAX_P];
        for (int vi = 0; vi < cell.nb_t; vi++)
            vertices_2d[vi] = cell.compute_vertex_point(cell.triangle[vi], true);

        // area + centroid via shoelace; fallback centroid is the seed
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
            const double face_measure = compute_face_measure(face_verts, n_fv, cell.voro_seed, nullptr);
            const int    neighbor_id  = cell.plane_vid[p];
            double4      neighbor     = make_double4(0.0, 0.0, 0.0, 0.0);
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
                const int last = ordered[n_ordered - 1];
                uchar     others_last[DIMENSION - 1];
                int       cnt = 0;
                for (int d = 0; d < DIMENSION; d++) {
                    const uchar pl = ith_plane(cell.triangle, (uchar)last, d);
                    if (pl != (uchar)p) others_last[cnt++] = pl;
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

            // dual triangles -> primal coordinates
            for (int k = 0; k < n_fv; k++) {
                face_verts[k] = cell.compute_vertex_point(cell.triangle[ordered[k]], true);
            }

            // outward-orient the face, then accumulate area + volume + weighted centroid
            orient_face_outward(face_verts, n_fv, cell.voro_seed);
            double face_measure = 0.0;
            compute_face_area_and_volume_centroid(
                face_verts, n_fv, cell.voro_seed, face_measure, total_volume, wx, wy, wz);

            // write the face into mesh's SoA arrays
            const int neighbor_id = cell.plane_vid[p];
            double4   neighbor    = make_double4(0.0, 0.0, 0.0, 0.0);
            if (neighbor_id >= 0) { neighbor = point_from_ptr(cell.pts + DIMENSION * neighbor_id); }
            write_face(mesh, fi, neighbor_id, face_measure, face_verts, n_fv, cell.voro_seed, neighbor);
            fi++;
        }

        // normalise centroid; fall back to seed if volume is degenerate
        if (fabs(total_volume) > 1e-30) {
            const double inv_vol  = 1.0 / total_volume;
            mesh->com[cell_index] = {wx * inv_vol, wy * inv_vol, wz * inv_vol};
        } else {
            mesh->com[cell_index] = seed;
        }
        mesh->volumes[cell_index] = fabs(total_volume);
#endif
    }

    // abort if `needed` exceeds the pre-allocated face buffer capacity
    void ensure_face_capacity(VMesh* mesh, hsize_t needed) {
        if (needed <= mesh->face_capacity) return;
        std::cerr << "VORONOI: Error! face count " << needed << " exceeds pre-allocated face capacity "
                  << mesh->face_capacity << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
        exit(EXIT_FAILURE);
    }

    // ============================================================
    // Helpers
    // ============================================================

    // initialise the convex cell to the unit-box bounding cell (eps margin against degeneracies).
    // Bounding-box plane equations are not stored — plane_for() returns them on demand for the
    // first 2*DIMENSION slots.
    template <int MAX_P, int MAX_T>
    HD BasicConvexCell<MAX_P, MAX_T>::BasicConvexCell(int p_seed, double* p_pts, Status* p_status, double p_buff) {
        pts       = p_pts;
        buff      = p_buff;
        status    = p_status;
        *status   = success;
        voro_seed = point_from_ptr(pts + DIMENSION * p_seed);

        // boundary linked list + plane->point-id map start empty
        first_boundary = END_OF_LIST;
        for (int i = 0; i < MAX_P; i++) {
            boundary_next[i] = END_OF_LIST;
            plane_vid[i]     = -1;
        }

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

    // clip the cell by the perpendicular bisector of (voro_seed, pts[vid])
    template <int MAX_P, int MAX_T> HD void BasicConvexCell<MAX_P, MAX_T>::clip_by_plane(int vid) {

        // append the new bisector plane; bail if we are out of plane slots
        const int cur_v = new_halfplane(vid);
        if (*status == vertex_overflow) { return; }

        // partition: move conflicting triangles to the tail (kept = [0, nb_t), removed = [nb_t, nb_t + nb_r))
        const double4 eqn = plane_for(cur_v);
        nb_r              = 0;
        int i             = 0;
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

        // build the boundary loop separating kept from removed triangles
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

    // returns true iff every cell vertex fits inside the sphere of radius ||last_neig - voro_seed||/2
    // (kept in projective num/denom form to avoid divisions in the inner loop)
    template <int MAX_P, int MAX_T>
    HD bool BasicConvexCell<MAX_P, MAX_T>::is_security_radius_reached(double4 last_neig) const {

        // scan for the vertex with the largest squared distance from voro_seed
        double max_num   = 0.0;
        double max_denom = 1.0;
        for (int i = 0; i < nb_t; i++) {
            const double4 pc = compute_vertex_point(triangle[i], false);
            const double  dx = pc.x - voro_seed.x * pc.w;
            const double  dy = pc.y - voro_seed.y * pc.w;
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

        // check d^2/4 > max_vertex_d2 (rearranged to avoid a division)
        const double4 diff = minus4(last_neig, voro_seed);
        const double  d2   = dot3(diff, diff);
        return (d2 * max_denom > 4.0 * max_num);
    }

    // rebuild plane equation for slot p — bounding-box constants for p < 2*DIMENSION,
    // otherwise the perpendicular bisector of (voro_seed, pts[plane_vid[p]])
    template <int MAX_P, int MAX_T> HD double4 BasicConvexCell<MAX_P, MAX_T>::plane_for(int p) const {

        // bounding box [-buff, 1+buff]^d (periodic ghost copies extend out to those limits,
        // so the bounding planes have to enclose them). plane form n.(x,y,z) + w >= 0:
        //   min plane:  x >= -buff   ->  ( 1, 0, 0,   buff + eps)
        //   max plane:  x <=  1+buff ->  (-1, 0, 0, 1+buff + eps)
        if (p < 2 * DIMENSION) {
            constexpr double eps   = 1e-14;
            const double     w_min = buff + eps;
            const double     w_max = 1.0 + buff + eps;
            switch (p) {
            case 0:
                return make_double4(1.0, 0.0, 0.0, w_min); // -xmin
            case 1:
                return make_double4(-1.0, 0.0, 0.0, w_max); //  xmax
            case 2:
                return make_double4(0.0, 1.0, 0.0, w_min); // -ymin
            case 3:
                return make_double4(0.0, -1.0, 0.0, w_max); //  ymax
#ifdef dim_3D
            case 4:
                return make_double4(0.0, 0.0, 1.0, w_min); // -zmin
            case 5:
                return make_double4(0.0, 0.0, -1.0, w_max); //  zmax
#endif
            }
        }

        // bisector of (voro_seed, B): normal = (voro_seed - B), offset = -((voro_seed + B)/2) . normal
        const double4 B    = point_from_ptr(pts + DIMENSION * plane_vid[p]);
        const double4 dir  = minus4(voro_seed, B);
        const double4 ave2 = plus4(voro_seed, B);
        const double  dot  = dot3(ave2, dir);
        return make_double4(dir.x, dir.y, dir.z, -dot * 0.5);
    }

    // append a new plane slot for vid; returns the new slot index (or -1 on overflow)
    template <int MAX_P, int MAX_T> HD int BasicConvexCell<MAX_P, MAX_T>::new_halfplane(int vid) {
        if (nb_v >= MAX_P) {
            *status = vertex_overflow;
            return -1;
        }
        plane_vid[nb_v] = vid;
        nb_v++;
        return nb_v - 1;
    }

    // is this triangle on the side of eqn that should be removed by the clip?
    // Uses interval-arithmetic guards to flag near-degenerate determinants as needs_exact_predicates.
    template <int MAX_P, int MAX_T>
    HD bool BasicConvexCell<MAX_P, MAX_T>::vert_is_in_conflict(VERT_TYPE v, double4 eqn) const {

        // gather the DIMENSION planes that define this triangle
        const double4 pi1 = plane_for(v.x);
        const double4 pi2 = plane_for(v.y);
#ifdef dim_2D
        // 2D: 3x3 determinant decides on which side the dual point of (pi1, pi2) lies relative to eqn
        const double det = det3x3(pi1.x, pi2.x, eqn.x, pi1.y, pi2.y, eqn.y, pi1.w, pi2.w, eqn.w);

        // interval bound on rounding error in det (Shewchuk-style)
        const double maxx    = fmax(fmax(fabs(pi1.x), fabs(pi2.x)), fabs(eqn.x));
        const double maxy    = fmax(fmax(fabs(pi1.y), fabs(pi2.y)), fabs(eqn.y));
        const double maxw    = fmax(fmax(fabs(pi1.w), fabs(pi2.w)), fabs(eqn.w));
        const double max_max = fmax(fmax(maxx, maxy), maxw);
        double       eps     = 1e-14 * maxx * maxy * maxw;
        eps *= max_max;
#else
        const double4 pi3 = plane_for(v.z);

        // 3D: 4x4 determinant decides on which side the dual point of (pi1, pi2, pi3) lies relative to eqn
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

        // interval bound on rounding error in det (Shewchuk-style)
        const double maxx = fmax(fmax(fabs(pi1.x), fabs(pi2.x)), fmax(fabs(pi3.x), fabs(eqn.x)));
        const double maxy = fmax(fmax(fabs(pi1.y), fabs(pi2.y)), fmax(fabs(pi3.y), fabs(eqn.y)));
        const double maxz = fmax(fmax(fabs(pi1.z), fabs(pi2.z)), fmax(fabs(pi3.z), fabs(eqn.z)));
        double       eps  = 1e-12 * maxx * maxy * maxz;
        double       min_max, max_max;
        get_minmax3(min_max, max_max, maxx, maxy, maxz);
        eps *= (max_max * max_max);
#endif

        // det within eps of zero: the sign decision is unreliable -> request exact predicates
        if (fabs(det) < eps) { *status = needs_exact_predicates; }
        return (det > 0.0);
    }

    // build the boundary loop separating removed from kept triangles (linked list in boundary_next[])
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
            const uchar2 e = triangle[nb_t + r];
            line_count[e.x]++;
            line_count[e.y]++;
        }

        // boundary endpoints = the two planes referenced by exactly one removed edge (parity)
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

        // wire up the 2-element boundary loop
        first_boundary                   = boundary_lines[0];
        boundary_next[boundary_lines[0]] = boundary_lines[1];
        boundary_next[boundary_lines[1]] = boundary_lines[0];
#else
        // reset boundary linked list
        for (int i = 0; i < MAX_P; i++) {
            boundary_next[i] = END_OF_LIST;
        }
        first_boundary = END_OF_LIST;

        // absorb removed triangles into the boundary loop one at a time, skipping any that would
        // make the loop non-simple; bail out if we loop forever
        int   nb_iter = 0;
        uchar t       = nb_t;

        while (nb_r > 0) {
            if (nb_iter++ > 10000) {
                *status = inconsistent_boundary;
                return;
            }

            // classify this triangle's 3 edges against the current boundary
            bool is_in_border[3];
            bool next_is_opp[3];
            for (int e = 0; e < 3; e++) {
                is_in_border[e] = (boundary_next[ith_plane(triangle, t, e)] != END_OF_LIST);
            }
            for (int e = 0; e < 3; e++) {
                next_is_opp[e] = (boundary_next[ith_plane(triangle, t, (e + 1) % 3)] == ith_plane(triangle, t, e));
            }

            // would adding this triangle leave the boundary simple (no self-crossings)?
            bool new_border_is_simple = true;
            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) {
                    new_border_is_simple = false;
                }
            }

            // bootstrap the empty loop with this triangle (or reject if loop already started)
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

            // not simple yet: skip to the next removed triangle and try again
            if (!new_border_is_simple) {
                t++;
                if (t == nb_t + nb_r) { t = nb_t; }
                continue;
            }

            // add the triangle's non-opposite edges to the boundary
            for (int e = 0; e < 3; e++) {
                if (!next_is_opp[e]) { boundary_next[ith_plane(triangle, t, e)] = ith_plane(triangle, t, (e + 1) % 3); }
            }

            // remove edges that became fully cancelled (pairs of opposite-direction edges)
            for (int e = 0; e < 3; e++) {
                if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
                    if (first_boundary == ith_plane(triangle, t, (e + 1) % 3)) {
                        first_boundary = boundary_next[ith_plane(triangle, t, (e + 1) % 3)];
                    }
                    boundary_next[ith_plane(triangle, t, (e + 1) % 3)] = END_OF_LIST;
                }
            }

            // swap the processed triangle to the tail and shrink the removed range
            VERT_TYPE tmp             = triangle[t];
            triangle[t]               = triangle[nb_t + nb_r - 1];
            triangle[nb_t + nb_r - 1] = tmp;
            t                         = nb_t;
            nb_r--;
        }
#endif
    }

    // create a new triangle from planes i, j (and k in 3D); writes into triangle[nb_t]
    template <int MAX_P, int MAX_T> HD void BasicConvexCell<MAX_P, MAX_T>::new_vertex(uchar i, uchar j, uchar k) {
        if (nb_t + 1 >= MAX_T) {
            *status = triangle_overflow;
            return;
        }
#ifdef dim_2D
        // 2D: orient (i, j) so the resulting boundary winds consistently
        (void)k;
        const double4 hi = plane_for(i);
        const double4 hj = plane_for(j);
        const double  rw = det2x2(hi.x, hi.y, hj.x, hj.y);
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

    // primal coordinates of a dual-graph vertex (intersection of DIMENSION planes); set
    // persp_divide=false to return the projective representation (used by security-radius check)
    template <int MAX_P, int MAX_T>
    HD double4 BasicConvexCell<MAX_P, MAX_T>::compute_vertex_point(VERT_TYPE v, bool persp_divide) const {
        const double4 pi1 = plane_for(v.x);
        const double4 pi2 = plane_for(v.y);
        double4       result;
#ifdef dim_2D
        // 2D: 2x2 determinants in (x, y, w) coordinates
        result.x = -det2x2(pi1.w, pi1.y, pi2.w, pi2.y);
        result.y = -det2x2(pi1.x, pi1.w, pi2.x, pi2.w);
        result.z = 0;
        result.w = det2x2(pi1.x, pi1.y, pi2.x, pi2.y);
        if (persp_divide) { return make_double4(result.x / result.w, result.y / result.w, 0, 1); }
#else
        // 3D: 3x3 determinants in (x, y, z, w) coordinates
        const double4 pi3 = plane_for(v.z);
        result.x          = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
        result.y          = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
        result.z          = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
        result.w          = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
        if (persp_divide) {
            const double inv_w = 1.0 / result.w;
            return make_double4(result.x * inv_w, result.y * inv_w, result.z * inv_w, 1);
        }
#endif
        return result;
    }

    // walk the boundary of face p in dual-graph order, writing primal vertices into face_verts[].
    // Returns false if the face has fewer than DIMENSION vertices (= not a real face).
    template <int MAX_P, int MAX_T>
    HD bool collect_face_vertices(const BasicConvexCell<MAX_P, MAX_T>& cell,
                                  int                                  p,
                                  const double4*                       vertices,
                                  double4*                             face_verts,
                                  int*                                 n_face_verts) {
#ifdef dim_2D
        // 2D: a face has exactly 2 vertices (an edge); grab them from the dual-graph triangles
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
        // 3D: gather triangles touching plane p; these are the corners of face p
        int face_vert_indices[MAX_T];
        int n_fvi = 0;
        for (int i = 0; i < cell.nb_t; i++) {
            if (vert_references_plane(cell.triangle, i, (uchar)p)) { face_vert_indices[n_fvi++] = i; }
        }
        if (n_fvi < DIMENSION) return false;

        // walk vertices around the face: each adjacent pair shares a non-p plane
        int  ordered[MAX_T];
        bool used[MAX_T];
        for (int k = 0; k < n_fvi; k++)
            used[k] = false;
        ordered[0]    = face_vert_indices[0];
        used[0]       = true;
        int n_ordered = 1;

        for (int step = 1; step < n_fvi; step++) {
            const int last = ordered[n_ordered - 1];

            // collect the non-p planes of the current triangle
            uchar others_last[DIMENSION - 1];
            int   cnt = 0;
            for (int d = 0; d < DIMENSION; d++) {
                const uchar pl = ith_plane(cell.triangle, (uchar)last, d);
                if (pl != (uchar)p) others_last[cnt++] = pl;
            }

            // find an unused candidate that shares one of those planes
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

        // resolve dual triangles -> primal coordinates in walk order
        *n_face_verts = n_ordered;
        for (int k = 0; k < n_ordered; k++) {
            face_verts[k] = vertices[ordered[k]];
        }
#endif
        return true;
    }

    // write a single face into mesh's SoA arrays (neighbour id, area, and moving-mesh face midpoint
    // expressed in the local rotated frame of the face)
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
        const int remapped      = (neighbor_id >= 0) ? (int)mesh->sid_to_neighbor[neighbor_id] : neighbor_id;
        mesh->neighbor_cell[fi] = remapped;
        mesh->face_area[fi]     = face_measure;

#ifdef MOVING_MESH
        // face midpoint relative to the seed-to-neighbour midpoint, projected into the local frame
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
            // bounding-box face: no neighbour, no local-frame offset
#ifdef dim_2D
            mesh->f_mid_local[fi] = 0.0;
#else
            mesh->f_mid_local[2 * fi]     = 0.0;
            mesh->f_mid_local[2 * fi + 1] = 0.0;
#endif
        }
#endif
    }

    // ---- utility predicates (file-local) ----

    // index into a triangle's plane indices as bytes (VERT_TYPE is uchar2 or uchar3)
    HD static inline uchar& ith_plane(VERT_TYPE* triangles, uchar t, int i) {
        return reinterpret_cast<uchar*>(&(triangles[t]))[i];
    }

    HD static inline uchar ith_plane(const VERT_TYPE* triangles, uchar t, int i) {
        return reinterpret_cast<const uchar*>(&(triangles[t]))[i];
    }

    // does triangle t_idx have plane p as one of its DIMENSION planes?
    HD static inline bool vert_references_plane(const VERT_TYPE* triangles, int t_idx, uchar p) {
        for (int d = 0; d < DIMENSION; d++) {
            if (ith_plane(triangles, (uchar)t_idx, d) == p) return true;
        }
        return false;
    }

} // namespace voronoi
