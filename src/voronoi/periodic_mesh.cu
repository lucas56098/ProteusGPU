#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../hydro/riemann.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "periodic_mesh.h"
#include <cmath>
#include <iostream>

namespace voronoi {

    constexpr double PI = 3.14159265358979323846;

    // forward declarations
#ifdef MOVING_MESH
    HD void compute_mesh_velocity_for_cell(hsize_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    HD void move_mesh_for_cell(hsize_t, const VMesh*, double, POINT_TYPE*);
#endif

#ifndef CPU_DEBUG
#ifdef MOVING_MESH
    // kernels
    GLOBAL void kernel_mesh_velocities(hsize_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    GLOBAL void kernel_move_mesh(hsize_t, const VMesh*, double, POINT_TYPE*);
#endif
    GLOBAL void kernel_generate_ghosts(hsize_t,
                                       const POINT_TYPE* __restrict__,
                                       POINT_TYPE* __restrict__,
                                       hsize_t* __restrict__,
                                       int* __restrict__,
                                       double);
#endif

    // Inline helpers
    HD inline bool is_in(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za = 0.0, double zb = 1.0) {
#ifdef dim_2D
        (void)za;
        (void)zb;
        return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb);
#else
        return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb) && (pt.z > za && pt.z < zb);
#endif
    }

    inline void add_ghost(POINT_TYPE*    pts,
                          hsize_t        index,
                          hsize_t*       n_ghosts,
                          const hsize_t* n_hydro,
                          hsize_t*       original_ids,
                          double         shift_x,
                          double         shift_y,
                          double         shift_z = 0.0) {
        POINT_TYPE pt;
        pt.x = pts[index].x + shift_x;
        pt.y = pts[index].y + shift_y;
#ifdef dim_3D
        pt.z = pts[index].z + shift_z;
#else
        (void)shift_z;
#endif

        pts[(*n_hydro) + (*n_ghosts)] = pt;
        original_ids[*n_ghosts]       = index;
        (*n_ghosts)++;
    }

    // ============================================================
    // periodic mesh computation, velocity, motion
    // ============================================================

    // wrap seeds around unit-box boundaries with ghost copies, then build the mesh
    // on the enlarged [-buff, 1+buff]^d domain (no rescaling step)
    void compute_periodic_mesh(VMesh* mesh, POINT_TYPE* pts_data, hsize_t num_points) {
        PROFILE_START("MESH_TOTAL");

        double  ghost_frac       = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        hsize_t max_ghost_points = (hsize_t)(2.0 * ghost_frac * num_points) + 1;

        POINT_TYPE* pts      = mesh->scratch_pts;
        hsize_t     n_ghosts = 0;
        hsize_t     n_hydro  = num_points;

        hsize_t* original_ids = mesh->ghost_ids;

        // emit ghost copies of every hydro point that falls within `buff` of a face
        PROFILE_START("GHOST_GEN (cpu)");
#ifndef CPU_DEBUG
        {
            int* d_ghost_count = (int*)gpu_malloc(sizeof(int));
            gpu_memset(d_ghost_count, 0, sizeof(int));

            int tpb    = _MESH_BLOCK_SIZE_;
            int blocks = ((int)n_hydro + tpb - 1) / tpb;
            PROFILE_GPU_START("kernel_generate_ghosts");
            kernel_generate_ghosts<<<blocks, tpb>>>(n_hydro, pts_data, pts, original_ids, d_ghost_count, buff);
            PROFILE_GPU_END("kernel_generate_ghosts");

            GPU_SYNC();
            n_ghosts = (hsize_t)(*d_ghost_count);
            gpu_free(d_ghost_count);
        }
#else
        for (hsize_t i = 0; i < n_hydro; i++) {
            pts[i] = pts_data[i];

            for (int sx = -1; sx <= 1; sx++) {
                for (int sy = -1; sy <= 1; sy++) {
#ifdef dim_3D
                    for (int sz = -1; sz <= 1; sz++) {
#else
                    {
                        int sz = 0;
#endif
                        if (sx == 0 && sy == 0 && sz == 0) continue;
                        double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff : 0.0;
                        double xb = (sx == 1) ? buff : 1.0;
                        double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff : 0.0;
                        double yb = (sy == 1) ? buff : 1.0;
                        double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff : 0.0;
                        double zb = (sz == 1) ? buff : 1.0;
                        if (is_in(pts[i], xa, xb, ya, yb, za, zb)) {
                            add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, (double)sx, (double)sy, (double)sz);
                        }
                    }
                }
            }
        }
#endif

        PROFILE_END("GHOST_GEN (cpu)");
        if (n_ghosts > max_ghost_points) {
            std::cerr << "VORONOI: Error! ghost count " << n_ghosts << " exceeds estimated max " << max_ghost_points
                      << ". Distribution is highly non-uniform." << std::endl;
            exit(EXIT_FAILURE);
        }

        hsize_t n_total = n_hydro + n_ghosts;

        // build the Voronoi mesh directly on the [-buff, 1+buff]^d (hydro + ghost) point set.
        // KNN's bucket grid and BasicConvexCell's bounding planes both pick up mesh->buff /
        // knn->buff so they enclose the ghost ring without any explicit rescaling step.
        mesh->n_hydro = n_hydro;
        compute_mesh(mesh, pts, n_total);

        PROFILE_END("MESH_TOTAL");
    }

#ifdef MOVING_MESH
    // gas-velocity + Lloyd-style regularization, per hydro cell
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads) {

#ifndef CPU_DEBUG
        int tpb    = _MESH_BLOCK_SIZE_;
        int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
        PROFILE_GPU_START("kernel_mesh_velocities");
        kernel_mesh_velocities<<<blocks, tpb>>>(mesh->n_hydro, mesh, primvar, grads);
        PROFILE_GPU_END("kernel_mesh_velocities");
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            compute_mesh_velocity_for_cell(i, mesh, primvar, grads);
        }
#endif
    }

    // advance seeds by v_mesh*dt (with periodic wrap), then rebuild the mesh
    void move_mesh(VMesh* mesh, double dt) {

        POINT_TYPE* pts     = mesh->scratch_move;
        hsize_t     n_hydro = mesh->n_hydro;

#ifndef CPU_DEBUG
        int tpb    = _MESH_BLOCK_SIZE_;
        int blocks = ((int)n_hydro + tpb - 1) / tpb;
        kernel_move_mesh<<<blocks, tpb>>>(n_hydro, mesh, dt, pts);
        GPU_LAUNCH_CHECK();
#else
        for (hsize_t i = 0; i < n_hydro; i++) {
            move_mesh_for_cell(i, mesh, dt, pts);
        }
#endif

        compute_periodic_mesh(mesh, pts, n_hydro);
    }
#endif // MOVING_MESH

    // ============================================================
    // CUDA kernel wrappers
    // ============================================================
#ifndef CPU_DEBUG

#ifdef MOVING_MESH
    GLOBAL void kernel_mesh_velocities(hsize_t                         n_hydro,
                                       VMesh*                          mesh,
                                       const hydro::primvars*          primvar,
                                       const gradients::PrimGradients* grads) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        compute_mesh_velocity_for_cell(i, mesh, primvar, grads);
    }

    GLOBAL void kernel_move_mesh(hsize_t n_hydro, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        move_mesh_for_cell(i, mesh, dt, pts);
    }
#endif // MOVING_MESH

    GLOBAL void kernel_generate_ghosts(hsize_t n_hydro,
                                       const POINT_TYPE* __restrict__ pts_data,
                                       POINT_TYPE* __restrict__ pts,
                                       hsize_t* __restrict__ original_ids,
                                       int* __restrict__ d_ghost_count,
                                       double buff_val) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;

        pts[i] = pts_data[i];

        POINT_TYPE pi = pts[i];

        for (int sx = -1; sx <= 1; sx++) {
            for (int sy = -1; sy <= 1; sy++) {
#ifdef dim_3D
                for (int sz = -1; sz <= 1; sz++) {
#else
                {
                    int sz = 0;
#endif
                    if (sx == 0 && sy == 0 && sz == 0) continue;

                    double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff_val : 0.0;
                    double xb = (sx == 1) ? buff_val : 1.0;
                    double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff_val : 0.0;
                    double yb = (sy == 1) ? buff_val : 1.0;
                    double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff_val : 0.0;
                    double zb = (sz == 1) ? buff_val : 1.0;

                    if (is_in(pi, xa, xb, ya, yb, za, zb)) {
                        int        slot = atomicAdd(d_ghost_count, 1);
                        POINT_TYPE gpt;
                        gpt.x = pi.x + (double)sx;
                        gpt.y = pi.y + (double)sy;
#ifdef dim_3D
                        gpt.z = pi.z + (double)sz;
#endif
                        pts[n_hydro + slot] = gpt;
                        original_ids[slot]  = i;
                    }
                }
            }
        }
    }

#endif // !CPU_DEBUG

    // ============================================================
    // Per-cell work (called by kernels and CPU loops)
    // ============================================================
#ifdef MOVING_MESH

    HD void compute_mesh_velocity_for_cell(hsize_t                         i,
                                           VMesh*                          mesh,
                                           const hydro::primvars*          primvar,
                                           const gradients::PrimGradients* grads) {
        // mesh velocity starts as the gas velocity
        double vx_mesh = primvar->v[i].x;
        double vy_mesh = primvar->v[i].y;
#ifdef dim_3D
        double vz_mesh = primvar->v[i].z;
#endif

        // effective cell radius (sphere/disk equivalent)
#ifdef dim_2D
        const double Ri = sqrt(fmax(mesh->volumes[i], 0.0) / PI);
#else
        const double Ri = cbrt(3.0 * fmax(mesh->volumes[i], 0.0) / (4.0 * PI));
#endif

        // displacement of seed from cell centroid (Lloyd target)
        double dx = wrap_periodic_delta(mesh->com[i].x - mesh->seeds[i].x);
        double dy = wrap_periodic_delta(mesh->com[i].y - mesh->seeds[i].y);
#ifdef dim_3D
        double dz = wrap_periodic_delta(mesh->com[i].z - mesh->seeds[i].z);
#endif

        // density-gradient correction: bias the target toward the steeper side, capped at Ri/4.
        // The cap is a smooth clamp (not a binary skip) so the offset stays continuous across
        // steps — otherwise small fluctuations near the cap flip the regularization on/off.
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
                    const double x_off  = (tmp - sqrt(disc)) / 4.0;
                    const double offset = fmin(x_off, 0.25 * Ri);
                    dx += offset * grads->rho[i].x / dgrad;
                    dy += offset * grads->rho[i].y / dgrad;
#ifdef dim_3D
                    dz += offset * grads->rho[i].z / dgrad;
#endif
                }
            }
        }

#ifdef dim_3D
        const double di = sqrt(dx * dx + dy * dy + dz * dz);
#else
        const double di = sqrt(dx * dx + dy * dy);
#endif

        // ramp regularization velocity from 0 (well-shaped) to CellShapingSpeed*c_s (very distorted)
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
                // scale by sound speed so regularization respects local timescales
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

        mesh->v_mesh[i].x = vx_mesh;
        mesh->v_mesh[i].y = vy_mesh;
#ifdef dim_3D
        mesh->v_mesh[i].z = vz_mesh;
#endif
    }

    // advance one seed by v_mesh*dt with periodic wrap into [0,1)
    HD void move_mesh_for_cell(hsize_t i, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        pts[i].x = fmod((mesh->seeds[i].x + dt * mesh->v_mesh[i].x) + 1.0, 1.0);
        pts[i].y = fmod((mesh->seeds[i].y + dt * mesh->v_mesh[i].y) + 1.0, 1.0);
#ifdef dim_3D
        pts[i].z = fmod((mesh->seeds[i].z + dt * mesh->v_mesh[i].z) + 1.0, 1.0);
#endif
    }

#endif // MOVING_MESH

} // namespace voronoi
