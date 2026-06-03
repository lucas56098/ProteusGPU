#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../hydro/riemann.h"
#include "../mpi/migrate.h"
#include "../profiler/profiler.h"
#include "../voronoi/periodic.h"
#include "../voronoi/voronoi.h"
#include "moving.h"
#include <cmath>

namespace voronoi {

#ifdef MOVING_MESH

    // forward declarations
    HD void compute_mesh_velocity_for_cell(hsize_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    HD void move_mesh_for_cell(hsize_t, const VMesh*, double, POINT_TYPE*);
#ifndef CPU_DEBUG
    GLOBAL void kernel_mesh_velocities(hsize_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    GLOBAL void kernel_move_mesh(hsize_t, const VMesh*, double, POINT_TYPE*);
#endif

    // ============================================================
    // Public entry points
    // ============================================================

    // gas-velocity + Lloyd-style regularization, per hydro cell
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads) {

#ifndef CPU_DEBUG
        int tpb    = _MESH_BLOCK_SIZE_;
        int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
        Profiler::StartGPU("kernel_mesh_velocities");
        kernel_mesh_velocities<<<blocks, tpb>>>(mesh->n_hydro, mesh, primvar, grads);
        Profiler::EndGPU("kernel_mesh_velocities");
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
    void move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::primvars* primvar_aux) {

        // store old volume (needed for volume correct)
        gpu_memcpy(mesh->old_volumes, mesh->volumes, mesh->n_hydro * sizeof(double));

        POINT_TYPE* pts     = mesh->scratch_move;
        hsize_t     n_hydro = mesh->n_hydro;

#ifndef CPU_DEBUG
        int tpb    = _MESH_BLOCK_SIZE_;
        int blocks = ((int)n_hydro + tpb - 1) / tpb;
        kernel_move_mesh<<<blocks, tpb>>>(n_hydro, mesh, dt, pts);
        GPU_LAUNCH_CHECK();
        GPU_SYNC(); // migrate_seeds reads pts on the host below
#else
        for (hsize_t i = 0; i < n_hydro; i++) {
            move_mesh_for_cell(i, mesh, dt, pts);
        }
#endif

        // ship cells whose new bucket is owned by another rank; updates mesh->n_hydro
        // and compacts/extends pts (== mesh->scratch_move) to match
        proteus_mpi::migrate_seeds(mesh, primvar, primvar_aux);
        n_hydro = mesh->n_hydro;

        compute_periodic_mesh(mesh, pts, n_hydro, primvar, primvar_aux);
    }

    // ============================================================
    // CUDA kernels
    // ============================================================
#ifndef CPU_DEBUG

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

#endif // !CPU_DEBUG

    // ============================================================
    // Per-cell work (called by kernels and CPU loops)
    // ============================================================

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
                hydro::prim  state_i = get_state(i, primvar);
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
