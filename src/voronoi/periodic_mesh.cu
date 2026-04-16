#include "../begrun/begrun.h"
#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../hydro/riemann.h"
#include "../io/input.h"
#include "../io/output.h"
#include "../knn/knn.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "periodic_mesh.h"
#include <climits>
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
    GLOBAL void kernel_scale_down_points(hsize_t, POINT_TYPE*, double);
    GLOBAL void kernel_scale_up_mesh(hsize_t, double3*, double3*, double*, double, double);
    GLOBAL void kernel_scale_face_area(hsize_t, compact_t*, double);
#ifdef MOVING_MESH
    GLOBAL void kernel_scale_f_mid(hsize_t, compact_t*, double);
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

    void compute_periodic_mesh(VMesh* mesh, POINT_TYPE* pts_data, hsize_t num_points) {
        PROFILE_START("MESH_TOTAL");

#ifdef DEBUG_MODE
        std::cout << "VORONOI: set up periodic mesh" << std::endl;
#endif

        double  ghost_frac       = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        hsize_t max_ghost_points = (hsize_t)(2.0 * ghost_frac * num_points) + 1;

        POINT_TYPE* pts      = mesh->scratch_pts;
        hsize_t     n_ghosts = 0;
        hsize_t     n_hydro  = num_points;

        hsize_t* original_ids = mesh->ghost_ids;

        PROFILE_START("GHOST_GEN (cpu)");
#ifndef CPU_DEBUG
        {
            int* d_ghost_count = (int*)gpu_malloc(sizeof(int));
            gpu_memset(d_ghost_count, 0, sizeof(int));

            int tpb    = 256;
            int blocks = ((int)n_hydro + tpb - 1) / tpb;
            PROFILE_GPU_START("kernel_generate_ghosts");
            kernel_generate_ghosts<<<blocks, tpb>>>(n_hydro, pts_data, pts, original_ids, d_ghost_count, buff);
            PROFILE_GPU_END("kernel_generate_ghosts");

            int h_ghost_count = 0;
            gpu_memcpy(&h_ghost_count, d_ghost_count, sizeof(int));
            n_ghosts = (hsize_t)h_ghost_count;
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

        double  scale   = 1. / (1. + (2 * buff));
        hsize_t n_total = n_hydro + n_ghosts;
#ifndef CPU_DEBUG
        {
            int tpb    = 256;
            int blocks = ((int)n_total + tpb - 1) / tpb;
            kernel_scale_down_points<<<blocks, tpb>>>(n_total, pts, scale);
            GPU_LAUNCH_CHECK();
        }
#else
        for (hsize_t i = 0; i < n_total; i++) {
            pts[i].x = scale * (pts[i].x - 0.5) + 0.5;
            pts[i].y = scale * (pts[i].y - 0.5) + 0.5;
#ifdef dim_3D
            pts[i].z = scale * (pts[i].z - 0.5) + 0.5;
#endif
        }
#endif

        compute_mesh(mesh, pts, n_total);

        mesh->n_hydro = n_hydro;

        scale = 1. + (2 * buff);
#ifdef dim_2D
        double vscale = scale * scale;
        double ascale = scale;
#else
        double vscale = scale * scale * scale;
        double ascale = scale * scale;
#endif

#ifndef CPU_DEBUG
        {
            int tpb    = 256;
            int blocks = ((int)n_total + tpb - 1) / tpb;
            kernel_scale_up_mesh<<<blocks, tpb>>>(n_total, mesh->seeds, mesh->com, mesh->volumes, scale, vscale);
            GPU_LAUNCH_CHECK();
        }

#ifdef MOVING_MESH
        {
            hsize_t n_fmid = mesh->num_faces * (DIMENSION - 1);
            int     tpb    = 256;
            int     blocks = ((int)n_fmid + tpb - 1) / tpb;
            kernel_scale_f_mid<<<blocks, tpb>>>(n_fmid, mesh->f_mid_local, scale);
            GPU_LAUNCH_CHECK();
        }
#endif

        {
            int tpb    = 256;
            int blocks = ((int)mesh->num_faces + tpb - 1) / tpb;
            kernel_scale_face_area<<<blocks, tpb>>>(mesh->num_faces, mesh->face_area, ascale);
            GPU_LAUNCH_CHECK();
        }
#else
        for (hsize_t i = 0; i < n_total; i++) {
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
#endif

        PROFILE_END("MESH_TOTAL");
    }

#ifdef MOVING_MESH
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads) {

#ifndef CPU_DEBUG
        int tpb    = 256;
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

    void move_mesh(VMesh* mesh, double dt) {

        POINT_TYPE* pts     = mesh->scratch_move;
        hsize_t     n_hydro = mesh->n_hydro;

#ifndef CPU_DEBUG
        int tpb    = 256;
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

    GLOBAL void kernel_scale_down_points(hsize_t n, POINT_TYPE* pts, double scale) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        pts[i].x = scale * (pts[i].x - 0.5) + 0.5;
        pts[i].y = scale * (pts[i].y - 0.5) + 0.5;
#ifdef dim_3D
        pts[i].z = scale * (pts[i].z - 0.5) + 0.5;
#endif
    }

    GLOBAL void
    kernel_scale_up_mesh(hsize_t n, double3* seeds, double3* com, double* volumes, double scale, double vscale) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        seeds[i].x = (seeds[i].x - 0.5) * scale + 0.5;
        seeds[i].y = (seeds[i].y - 0.5) * scale + 0.5;
        com[i].x   = (com[i].x - 0.5) * scale + 0.5;
        com[i].y   = (com[i].y - 0.5) * scale + 0.5;
#ifdef dim_3D
        seeds[i].z = (seeds[i].z - 0.5) * scale + 0.5;
        com[i].z   = (com[i].z - 0.5) * scale + 0.5;
#endif
        volumes[i] = vscale * volumes[i];
    }

    GLOBAL void kernel_scale_face_area(hsize_t n, compact_t* face_area, double ascale) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        face_area[i] = (compact_t)(ascale * (double)face_area[i]);
    }

#ifdef MOVING_MESH
    GLOBAL void kernel_scale_f_mid(hsize_t n, compact_t* f_mid_local, double scale) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        f_mid_local[i] = (compact_t)((double)f_mid_local[i] * scale);
    }
#endif

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
        double vx_mesh = primvar->v[i].x;
        double vy_mesh = primvar->v[i].y;
#ifdef dim_3D
        double vz_mesh = primvar->v[i].z;
#endif

#ifdef dim_2D
        const double Ri = sqrt(fmax(mesh->volumes[i], 0.0) / PI);
#else
        const double Ri = cbrt(3.0 * fmax(mesh->volumes[i], 0.0) / (4.0 * PI));
#endif

        double dx = wrap_periodic_delta(mesh->com[i].x - mesh->seeds[i].x);
        double dy = wrap_periodic_delta(mesh->com[i].y - mesh->seeds[i].y);
#ifdef dim_3D
        double dz = wrap_periodic_delta(mesh->com[i].z - mesh->seeds[i].z);
#endif

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

#ifdef dim_3D
        const double di = sqrt(dx * dx + dy * dy + dz * dz);
#else
        const double di = sqrt(dx * dx + dy * dy);
#endif

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

        mesh->v_mesh[i].x = vx_mesh;
        mesh->v_mesh[i].y = vy_mesh;
#ifdef dim_3D
        mesh->v_mesh[i].z = vz_mesh;
#endif
    }

    HD void move_mesh_for_cell(hsize_t i, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        pts[i].x = fmod((mesh->seeds[i].x + dt * mesh->v_mesh[i].x) + 1.0, 1.0);
        pts[i].y = fmod((mesh->seeds[i].y + dt * mesh->v_mesh[i].y) + 1.0, 1.0);
#ifdef dim_3D
        pts[i].z = fmod((mesh->seeds[i].z + dt * mesh->v_mesh[i].z) + 1.0, 1.0);
#endif
    }

#endif // MOVING_MESH

} // namespace voronoi
