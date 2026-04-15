#include "periodic_mesh.h"
#include "../begrun/begrun.h"
#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../hydro/riemann.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include <climits>
#include <cmath>
#include <iostream>

namespace voronoi {

    constexpr double PI = 3.14159265358979323846;

    // checks if pt is in box given by xa, xb, ya, ...
    inline bool is_in(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za, double zb) {
#ifdef dim_2D
        (void)za;
        (void)zb;
        return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb);
#else
        return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb) && (pt.z > za && pt.z < zb);
#endif
    }

    // add ghost point to shifted position
    inline void add_ghost(POINT_TYPE*    pts,
                          hsize_t        index,
                          hsize_t*       n_ghosts,
                          const hsize_t* n_hydro,
                          hsize_t*       original_ids,
                          double         shift_x,
                          double         shift_y,
                          double         shift_z) {
        // create shifted pt
        POINT_TYPE pt;
        pt.x = pts[index].x + shift_x;
        pt.y = pts[index].y + shift_y;
#ifdef dim_3D
        pt.z = pts[index].z + shift_z;
#else
        (void)shift_z;
#endif

        // add pt to pts
        pts[(*n_hydro) + (*n_ghosts)] = pt;
        original_ids[*n_ghosts]       = index;
        (*n_ghosts)++;
    }

    VMesh* compute_periodic_mesh(POINT_TYPE* pts_data, hsize_t num_points, VMesh* reuse) {
        PROFILE_START("MESH_TOTAL");

#ifdef DEBUG_MODE
        std::cout << "VORONOI: set up periodic mesh" << std::endl;
#endif

        // pre-allocate VMesh with worst-case capacity on first call (no realloc for GPU compatibility)
        // ghost estimate: for uniform points in [0,1]^D with buffer width buff,
        // the ghost fraction is (1+2*buff)^D - 1. Apply 2x fudge factor for safety.
        if (!reuse) {
            double  ghost_frac     = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
            hsize_t max_ghosts     = (hsize_t)(2.0 * ghost_frac * num_points) + 1;
            hsize_t max_n_total    = num_points + max_ghosts;
            hsize_t max_face_count = max_n_total * _FACE_CAPACITY_MULT_;
            reuse                  = allocate_vmesh(max_n_total, max_face_count);
        }

        // allocate temporary pts (hydro + ghosts) and ghost id mapping
        // same geometric estimate with 2x fudge factor
        double      ghost_frac       = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        hsize_t     max_ghost_points = (hsize_t)(2.0 * ghost_frac * num_points) + 1;
        POINT_TYPE* pts;
        pts = (POINT_TYPE*)malloc((num_points + max_ghost_points) * sizeof(POINT_TYPE));

        hsize_t  n_ghosts = 0;
        hsize_t  n_hydro  = num_points;
        hsize_t* original_ids;
        original_ids = (hsize_t*)malloc(max_ghost_points * sizeof(hsize_t));

        // select points that get ghosts
        for (hsize_t i = 0; i < n_hydro; i++) {

            // copy original point to pts
            pts[i] = pts_data[i];

#ifdef dim_2D
            // check if point is in any of those regions... if so add the corresponding ghost
            // edges
            if (is_in(pts[i], 0., 0. + buff, 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 0.);
            } // region 1
            if (is_in(pts[i], 0., 1., 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., -1.);
            } // region 2
            if (is_in(pts[i], 1. - buff, 1., 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 0.);
            } // region 3
            if (is_in(pts[i], 0., 1., 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 1.);
            } // region 4
            // corners
            if (is_in(pts[i], 0., buff, 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 1.);
            } // region 5
            if (is_in(pts[i], 0., buff, 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., -1.);
            } // region 6
            if (is_in(pts[i], 1. - buff, 1., 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., -1.);
            } // region 7
            if (is_in(pts[i], 1. - buff, 1., 0, buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 1.);
            } // region 8
#else
            // check if point is in any of those regions... if so add the corresponding ghost
            // faces
            if (is_in(pts[i], 0., 1., 0., 1., 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 0., 1.);
            } // 1
            if (is_in(pts[i], 0., buff, 0., 1., 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 0., 0.);
            } // 2
            if (is_in(pts[i], 0., 1., 1. - buff, 1., 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., -1., 0.);
            } // 3
            if (is_in(pts[i], 1. - buff, 1., 0., 1., 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 0., 0.);
            } // 4
            if (is_in(pts[i], 0., 1., 0., buff, 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 1., 0.);
            } // 5
            if (is_in(pts[i], 0., 1., 0., 1., 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 0., -1.);
            } // 6
            // edges
            if (is_in(pts[i], 0., 1., 0., buff, 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 1., 1.);
            } // 1
            if (is_in(pts[i], 0., buff, 0., 1, 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 0., 1.);
            } // 2
            if (is_in(pts[i], 0., 1., 1. - buff, 1., 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., -1., 1.);
            } // 3
            if (is_in(pts[i], 1. - buff, 1., 0., 1., 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 0., 1.);
            } // 4
            if (is_in(pts[i], 0., buff, 0., buff, 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 1., 0.);
            } // 5
            if (is_in(pts[i], 0., buff, 1. - buff, 1., 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., -1., 0.);
            } // 6
            if (is_in(pts[i], 1. - buff, 1., 1. - buff, 1., 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., -1., 0.);
            } // 7
            if (is_in(pts[i], 1. - buff, 1., 0., buff, 0., 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 1., 0.);
            } // 8
            if (is_in(pts[i], 0., buff, 0., 1., 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 0., -1.);
            } // 9
            if (is_in(pts[i], 0., 1., 1. - buff, 1., 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., -1., -1.);
            } // 10
            if (is_in(pts[i], 1. - buff, 1., 0., 1., 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 0., -1.);
            } // 11
            if (is_in(pts[i], 0., 1., 0., buff, 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 0., 1., -1.);
            } // 12
            // corners
            if (is_in(pts[i], 0., buff, 0., buff, 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 1., 1.);
            } // 1
            if (is_in(pts[i], 0., buff, 1. - buff, 1., 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., -1., 1.);
            } // 2
            if (is_in(pts[i], 1. - buff, 1., 1. - buff, 1., 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., -1., 1.);
            } // 3
            if (is_in(pts[i], 1. - buff, 1., 0., buff, 0., buff)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 1., 1.);
            } // 4
            if (is_in(pts[i], 0., buff, 0., buff, 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., 1., -1.);
            } // 5
            if (is_in(pts[i], 0., buff, 1. - buff, 1., 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, 1., -1., -1.);
            } // 6
            if (is_in(pts[i], 1. - buff, 1., 1. - buff, 1., 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., -1., -1.);
            } // 7
            if (is_in(pts[i], 1. - buff, 1., 0., buff, 1. - buff, 1.)) {
                add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, -1., 1., -1.);
            } // 8
#endif
        }

        // verify ghost count fits in pre-allocated arrays
        if (n_ghosts > max_ghost_points) {
            std::cerr << "VORONOI: Error! ghost count " << n_ghosts << " exceeds estimated max " << max_ghost_points
                      << ". Distribution is highly non-uniform." << std::endl;
            exit(EXIT_FAILURE);
        }

        // scale down... to [0,1]^2
        double scale = 1. / (1. + (2 * buff));
        for (hsize_t i = 0; i < n_hydro + n_ghosts; i++) {
            pts[i].x = scale * (pts[i].x - 0.5) + 0.5;
            pts[i].y = scale * (pts[i].y - 0.5) + 0.5;
#ifdef dim_3D
            pts[i].z = scale * (pts[i].z - 0.5) + 0.5;
#endif
        }

        // compute mesh (reuse old mesh buffers if available to avoid fragmentation)
        VMesh* mesh = compute_mesh(pts, n_hydro + n_ghosts, reuse);
        free(pts);

        // set mesh ghost quantities (free old ghost_ids if reusing an existing mesh)
        mesh->n_hydro = n_hydro;
        free(mesh->ghost_ids);
        mesh->ghost_ids = original_ids;

        // scale mesh up
        scale = 1. + (2 * buff);
#ifdef dim_2D
        double vscale = scale * scale;
        double ascale = scale;
#else
        double vscale = scale * scale * scale;
        double ascale = scale * scale;
#endif

        for (hsize_t i = 0; i < n_hydro + n_ghosts; i++) {
            mesh->seeds[i].x = (mesh->seeds[i].x - 0.5) * scale + 0.5;
            mesh->seeds[i].y = (mesh->seeds[i].y - 0.5) * scale + 0.5;
            mesh->com[i].x   = (mesh->com[i].x - 0.5) * scale + 0.5;
            mesh->com[i].y   = (mesh->com[i].y - 0.5) * scale + 0.5;
#ifdef dim_3D
            mesh->seeds[i].z = (mesh->seeds[i].z - 0.5) * scale + 0.5;
            mesh->com[i].z   = (mesh->com[i].z - 0.5) * scale + 0.5;
#endif
            mesh->volumes[i] = vscale * mesh->volumes[i];
        }

#ifdef MOVING_MESH
        for (hsize_t i = 0; i < mesh->num_faces * (DIMENSION - 1); i++) {
            mesh->f_mid_local[i] = (compact_t)((double)mesh->f_mid_local[i] * scale);
        }
#endif

        for (hsize_t i = 0; i < mesh->num_faces; i++) {
            mesh->face_area[i] = (compact_t)(ascale * (double)mesh->face_area[i]);
        }

#ifdef DEBUG_MODE
        for (hsize_t i = 0; i < mesh->num_edge_coord_verts; i++) {
            mesh->edge_coords[DIMENSION * i]     = (mesh->edge_coords[DIMENSION * i] - 0.5) * scale + 0.5;
            mesh->edge_coords[DIMENSION * i + 1] = (mesh->edge_coords[DIMENSION * i + 1] - 0.5) * scale + 0.5;
#ifdef dim_3D
            mesh->edge_coords[DIMENSION * i + 2] = (mesh->edge_coords[DIMENSION * i + 2] - 0.5) * scale + 0.5;
#endif
        }
#endif

#ifdef DEBUG_MODE
        std::cout << "VORONOI: periodic mesh should be created" << std::endl;
#endif
        PROFILE_END("MESH_TOTAL");

        // return that mesh :D
        return mesh;
    }

    // compute mesh-point velocities (gas velocity + CM drift regularization) to roughly preserve mass
    void compute_mesh_velocities(const VMesh*                    mesh,
                                 const hydro::primvars*          primvar,
                                 const gradients::PrimGradients* grads,
                                 POINT_TYPE*                     v_mesh) {

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            double vx_mesh = primvar->v[i].x;
            double vy_mesh = primvar->v[i].y;
#ifdef dim_3D
            double vz_mesh = primvar->v[i].z;
#endif

            // effective cell radius
#ifdef dim_2D
            const double Ri = std::sqrt(std::max(mesh->volumes[i], 0.0) / PI);
#else
            const double Ri = std::cbrt(3.0 * std::max(mesh->volumes[i], 0.0) / (4.0 * PI));
#endif

            // displacement from seed to COM
            double dx = wrap_periodic_delta(mesh->com[i].x - mesh->seeds[i].x);
            double dy = wrap_periodic_delta(mesh->com[i].y - mesh->seeds[i].y);
#ifdef dim_3D
            double dz = wrap_periodic_delta(mesh->com[i].z - mesh->seeds[i].z);
#endif

            // mesh aims for roughly equal-mass cells
            if (grads != nullptr && Ri > 0.0) {
#ifdef dim_3D
                const double dgrad = std::sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y +
                                               grads->rho[i].z * grads->rho[i].z);
#else
                const double dgrad = std::sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y);
#endif
                if (dgrad > 0.0) {
                    const double scale = primvar->rho[i] / dgrad;
                    const double tmp   = 3.0 * Ri + scale;
                    const double disc  = tmp * tmp - 8.0 * Ri * Ri;
                    if (disc > 0.0) {
                        const double x_off = (tmp - std::sqrt(disc)) / 4.0;
                        if (x_off < 0.25 * Ri) {
                            dx += x_off * grads->rho[i].x / dgrad;
                            dy += x_off * grads->rho[i].y / dgrad;
#ifdef dim_3D
                            dz += x_off * grads->rho[i].z / dgrad;
#endif
                        }
                    }
                }
            }

            // distance to target
#ifdef dim_3D
            const double di = std::sqrt(dx * dx + dy * dy + dz * dz);
#else
            const double di = std::sqrt(dx * dx + dy * dy);
#endif

            // ramp: kicks in at 0.75 * F * R, full strength at F * R
            if (di > 0.0 && Ri > 0.0) {
                const double threshold = CellShapingFactor * Ri;
                double       fraction  = 0.0;
                if (di > 0.75 * threshold) {
                    if (di > threshold)
                        fraction = CellShapingSpeed;
                    else
                        fraction = CellShapingSpeed * (di - 0.75 * threshold) / (0.25 * threshold);
                }

                if (fraction > 0.0) {
                    const double rho     = primvar->rho[i];
                    hydro::prim  state_i = get_state(i, mesh, primvar);
                    const double p       = std::max(0.0, hydro::get_P_ideal_gas(&state_i));
                    if (rho > 0.0 && p > 0.0) {
                        const double ci = std::sqrt(gamma_eos * p / rho);
                        vx_mesh += fraction * ci * dx / di;
                        vy_mesh += fraction * ci * dy / di;
#ifdef dim_3D
                        vz_mesh += fraction * ci * dz / di;
#endif
                    }
                }
            }

            v_mesh[i].x = vx_mesh;
            v_mesh[i].y = vy_mesh;
#ifdef dim_3D
            v_mesh[i].z = vz_mesh;
#endif
        }
    }

    // move the mesh with the given mesh point velocities
    VMesh* move_mesh(VMesh* mesh, const POINT_TYPE* v_mesh, double dt) {

        POINT_TYPE* pts = static_cast<POINT_TYPE*>(malloc(mesh->n_hydro * sizeof(POINT_TYPE)));

        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            pts[i].x = fmod((mesh->seeds[i].x + dt * v_mesh[i].x) + 1.0, 1.0);
            pts[i].y = fmod((mesh->seeds[i].y + dt * v_mesh[i].y) + 1.0, 1.0);
#ifdef dim_3D
            pts[i].z = fmod((mesh->seeds[i].z + dt * v_mesh[i].z) + 1.0, 1.0);
#endif
        }

        hsize_t n_hydro = mesh->n_hydro;

        // pass old mesh for buffer reuse instead of freeing and reallocating
        VMesh* new_mesh = compute_periodic_mesh(pts, n_hydro, mesh);

        free(pts);

        return new_mesh;
    }

} // namespace voronoi