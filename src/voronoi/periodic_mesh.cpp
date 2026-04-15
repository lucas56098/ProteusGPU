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
    inline bool is_in(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za = 0.0, double zb = 1.0) {
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
                          double         shift_z = 0.0) {
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

    // ============================================================================
    // Periodic mesh computation
    // ============================================================================

    void compute_periodic_mesh(VMesh* mesh, POINT_TYPE* pts_data, hsize_t num_points) {
        PROFILE_START("MESH_TOTAL");

#ifdef DEBUG_MODE
        std::cout << "VORONOI: set up periodic mesh" << std::endl;
#endif

        // ghost estimate for capacity check
        double  ghost_frac       = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        hsize_t max_ghost_points = (hsize_t)(2.0 * ghost_frac * num_points) + 1;

        POINT_TYPE* pts      = mesh->scratch_pts;
        hsize_t     n_ghosts = 0;
        hsize_t     n_hydro  = num_points;

        // write ghost IDs directly into pre-allocated VMesh buffer
        hsize_t* original_ids = mesh->ghost_ids;

        // select points that get ghosts
        for (hsize_t i = 0; i < n_hydro; i++) {

            // copy original point to pts
            pts[i] = pts_data[i];

            // table-driven ghost generation: iterate over all 3^D - 1 shift combinations
            for (int sx = -1; sx <= 1; sx++) {
                for (int sy = -1; sy <= 1; sy++) {
#ifdef dim_3D
                    for (int sz = -1; sz <= 1; sz++) {
#else
                    {
                        int sz = 0;
#endif
                        if (sx == 0 && sy == 0 && sz == 0) continue;
                        // shift → region: +1 means near low wall [0,buff], -1 near high [1-buff,1], 0 full [0,1]
                        double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff : 0.0;
                        double xb = (sx == 1) ? buff : 1.0;
                        double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff : 0.0;
                        double yb = (sy == 1) ? buff : 1.0;
                        double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff : 0.0;
                        double zb = (sz == 1) ? buff : 1.0;
                        if (is_in(pts[i], xa, xb, ya, yb, za, zb))
                            add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, (double)sx, (double)sy, (double)sz);
                    }
                }
            }
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

        // compute mesh
        compute_mesh(mesh, pts, n_hydro + n_ghosts);

        // ghost_ids were written directly into mesh->ghost_ids above
        mesh->n_hydro = n_hydro;

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

        PROFILE_END("MESH_TOTAL");
    }

    // compute mesh-point velocities (gas velocity + CM drift regularization) to roughly preserve mass
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads) {

        POINT_TYPE* v_mesh = mesh->v_mesh;

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
            const double Ri = sqrt(fmax(mesh->volumes[i], 0.0) / PI);
#else
            const double Ri = cbrt(3.0 * fmax(mesh->volumes[i], 0.0) / (4.0 * PI));
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
                const double dgrad = sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y +
                                          grads->rho[i].z * grads->rho[i].z);
#else
                const double dgrad = sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y);
#endif
                if (dgrad > 0.0) {
                    const double scale = primvar->rho[i] / dgrad;
                    const double tmp   = 3.0 * Ri + scale;
                    const double disc  = tmp * tmp - 8.0 * Ri * Ri;
                    if (disc > 0.0) {
                        const double x_off = (tmp - sqrt(disc)) / 4.0;
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
            const double di = sqrt(dx * dx + dy * dy + dz * dz);
#else
            const double di = sqrt(dx * dx + dy * dy);
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
                    const double p       = fmax(0.0, hydro::get_P_ideal_gas(&state_i));
                    if (rho > 0.0 && p > 0.0) {
                        const double ci = sqrt(gamma_eos * p / rho);
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
    void move_mesh(VMesh* mesh, double dt) {

        POINT_TYPE* pts     = mesh->scratch_move;
        hsize_t     n_hydro = mesh->n_hydro;

        for (hsize_t i = 0; i < n_hydro; i++) {
            pts[i].x = fmod((mesh->seeds[i].x + dt * mesh->v_mesh[i].x) + 1.0, 1.0);
            pts[i].y = fmod((mesh->seeds[i].y + dt * mesh->v_mesh[i].y) + 1.0, 1.0);
#ifdef dim_3D
            pts[i].z = fmod((mesh->seeds[i].z + dt * mesh->v_mesh[i].z) + 1.0, 1.0);
#endif
        }

        // rebuild periodic mesh in-place using moved seed positions
        compute_periodic_mesh(mesh, pts, n_hydro);
    }

} // namespace voronoi