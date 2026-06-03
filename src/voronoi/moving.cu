#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../hydro/riemann.h"
#include "../mpi/migrate.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include <cmath>

namespace voronoi {

#ifdef MOVING_MESH

    // ---- file-local types ----
    namespace {
        // Lloyd-regularization displacement target: how far (and which way) the seed should
        // move to reach its cell's centroid, optionally biased by the local density gradient.
        struct LloydDisplacement {
            double dx; // x-component of target offset
            double dy; // y-component
            double dz; // z-component (unused in 2D)
            double di; // magnitude
            double Ri; // effective cell radius (sphere / disk equivalent)
        };
    } // namespace

    // ---- forward declarations ----
    HD void compute_mesh_velocity_for_cell(hsize_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    HD void move_mesh_for_cell(hsize_t, const VMesh*, double, POINT_TYPE*);
    HD void
    volume_correct_for_cell(hsize_t i, const double* old_volumes, const double* new_volumes, double* rho, double* E);

    static void                 advance_seeds_by_dt(VMesh* mesh, double dt, POINT_TYPE* pts);
    static void                 correct_for_volume_change(VMesh* mesh, hydro::primvars* primvar);
    static HD POINT_TYPE        gas_velocity_for_cell(hsize_t i, const hydro::primvars* primvar);
    static HD LloydDisplacement lloyd_correction_for_cell(hsize_t                         i,
                                                          const VMesh*                    mesh,
                                                          const hydro::primvars*          primvar,
                                                          const gradients::PrimGradients* grads);
    static HD void              blend_into_mesh_velocity(
        hsize_t i, VMesh* mesh, POINT_TYPE v_gas, LloydDisplacement L, const hydro::primvars* primvar);

#ifndef CPU_DEBUG
    GLOBAL void kernel_mesh_velocities(hsize_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    GLOBAL void kernel_move_mesh(hsize_t, const VMesh*, double, POINT_TYPE*);
    GLOBAL void kernel_volume_correct(
        hsize_t n_hydro, const double* old_volumes, const double* new_volumes, double* rho, double* E);
#endif

    // ============================================================
    // Main routines
    // ============================================================

    // compute the mesh-point velocity (gas velocity + Lloyd regularization) for every cell
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads) {
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
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

    // advance seeds by v_mesh * dt, migrate across ranks, rebuild mesh,
    // and correct primvar_aux for the cell-volume change.
    // mesh->scratch_move is the seed-position buffer through all three voronoi-side calls:
    // advance_seeds_by_dt writes into it, migrate_seeds compacts/extends it in place,
    // compute_periodic_mesh reads from it.
    void move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::primvars* primvar_aux) {

        // store old volumes for volume correction afterwards
        gpu_memcpy(mesh->old_volumes, mesh->volumes, mesh->n_hydro * sizeof(double));

        // advance seed positions by v_mesh * dt into mesh->scratch_move
        advance_seeds_by_dt(mesh, dt, mesh->scratch_move);

        // migrate cells whose new bucket is owned by another rank;
        // updates mesh->n_hydro and rewrites mesh->scratch_move in place
        proteus_mpi::migrate_seeds(mesh, primvar, primvar_aux);

        // rebuild the Voronoi mesh from the new seed positions
        compute_periodic_mesh(mesh, mesh->scratch_move, mesh->n_hydro, primvar, primvar_aux);

        // correct primvar_aux for the cell-volume change (conservation: rho, E scale with old/new ratio)
        correct_for_volume_change(mesh, primvar_aux);
    }

    // ============================================================
    // Helpers
    // ============================================================

    // dispatch the per-cell move kernel; CPU branch loops via OpenMP
    static void advance_seeds_by_dt(VMesh* mesh, double dt, POINT_TYPE* pts) {
        const hsize_t n_hydro = mesh->n_hydro;
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = ((int)n_hydro + tpb - 1) / tpb;
        kernel_move_mesh<<<blocks, tpb>>>(n_hydro, mesh, dt, pts);
        GPU_LAUNCH_CHECK();
        GPU_SYNC(); // migrate_seeds reads pts on the host below
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t i = 0; i < n_hydro; i++) {
            move_mesh_for_cell(i, mesh, dt, pts);
        }
#endif
    }

    // dispatch the per-cell volume-correction kernel; CPU branch loops via OpenMP
    static void correct_for_volume_change(VMesh* mesh, hydro::primvars* primvar) {
        const hsize_t n_hydro = mesh->n_hydro;
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = ((int)n_hydro + tpb - 1) / tpb;
        Profiler::StartGPU("kernel_volume_correct");
        kernel_volume_correct<<<blocks, tpb>>>(n_hydro, mesh->old_volumes, mesh->volumes, primvar->rho, primvar->E);
        Profiler::EndGPU("kernel_volume_correct");
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t i = 0; i < n_hydro; i++) {
            volume_correct_for_cell(i, mesh->old_volumes, mesh->volumes, primvar->rho, primvar->E);
        }
#endif
    }

    // ============================================================
    // Per-cell work (called by kernels and CPU loops)
    // ============================================================

    // mesh-point velocity = gas velocity + Lloyd regularization, both scaled by sound speed
    HD void compute_mesh_velocity_for_cell(hsize_t                         i,
                                           VMesh*                          mesh,
                                           const hydro::primvars*          primvar,
                                           const gradients::PrimGradients* grads) {
        const POINT_TYPE        v_gas = gas_velocity_for_cell(i, primvar);
        const LloydDisplacement L     = lloyd_correction_for_cell(i, mesh, primvar, grads);
        blend_into_mesh_velocity(i, mesh, v_gas, L, primvar);
    }

    // advance one seed by v_mesh * dt with periodic wrap into [0, 1)
    HD void move_mesh_for_cell(hsize_t i, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        pts[i].x = fmod((mesh->seeds[i].x + dt * mesh->v_mesh[i].x) + 1.0, 1.0);
        pts[i].y = fmod((mesh->seeds[i].y + dt * mesh->v_mesh[i].y) + 1.0, 1.0);
#ifdef dim_3D
        pts[i].z = fmod((mesh->seeds[i].z + dt * mesh->v_mesh[i].z) + 1.0, 1.0);
#endif
    }

    // scale rho and E by old/new cell-volume ratio so total mass / energy stay conserved
    // when cell volume changes during the mesh move
    HD void
    volume_correct_for_cell(hsize_t i, const double* old_volumes, const double* new_volumes, double* rho, double* E) {
        const double ratio = old_volumes[i] / new_volumes[i];
        rho[i] *= ratio;
        E[i] *= ratio;
    }

    // read primvar->v[i] into a POINT_TYPE; the seed's gas velocity component
    HD static POINT_TYPE gas_velocity_for_cell(hsize_t i, const hydro::primvars* primvar) {
        POINT_TYPE v;
        v.x = primvar->v[i].x;
        v.y = primvar->v[i].y;
#ifdef dim_3D
        v.z = primvar->v[i].z;
#endif
        return v;
    }

    // seed-to-centroid offset + density-gradient bias toward the steeper side (capped at
    // Ri/4 and smoothly clamped so small fluctuations near the cap don't flip the bias)
    HD static LloydDisplacement lloyd_correction_for_cell(hsize_t                         i,
                                                          const VMesh*                    mesh,
                                                          const hydro::primvars*          primvar,
                                                          const gradients::PrimGradients* grads) {
        // effective cell radius from the volume
        LloydDisplacement L{};
#ifdef dim_2D
        L.Ri = sqrt(fmax(mesh->volumes[i], 0.0) / PI);
#else
        L.Ri = cbrt(3.0 * fmax(mesh->volumes[i], 0.0) / (4.0 * PI));
#endif

        // base offset: seed -> centroid, with periodic wrap on the deltas
        L.dx = wrap_periodic_delta(mesh->com[i].x - mesh->seeds[i].x);
        L.dy = wrap_periodic_delta(mesh->com[i].y - mesh->seeds[i].y);
#ifdef dim_3D
        L.dz = wrap_periodic_delta(mesh->com[i].z - mesh->seeds[i].z);
#endif

        // density-gradient bias: push toward the steeper side of the gradient
        if (grads != nullptr && L.Ri > 0.0) {
#ifdef dim_3D
            const double dgrad = sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y +
                                      grads->rho[i].z * grads->rho[i].z);
#else
            const double dgrad = sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y);
#endif
            if (dgrad > 0.0) {
                const double scale = primvar->rho[i] / dgrad;
                const double tmp   = 3.0 * L.Ri + scale;
                const double disc  = tmp * tmp - 8.0 * L.Ri * L.Ri;
                if (disc > 0.0) {
                    const double x_off  = (tmp - sqrt(disc)) / 4.0;
                    const double offset = fmin(x_off, 0.25 * L.Ri);
                    L.dx += offset * grads->rho[i].x / dgrad;
                    L.dy += offset * grads->rho[i].y / dgrad;
#ifdef dim_3D
                    L.dz += offset * grads->rho[i].z / dgrad;
#endif
                }
            }
        }

        // magnitude of the full target offset
#ifdef dim_3D
        L.di = sqrt(L.dx * L.dx + L.dy * L.dy + L.dz * L.dz);
#else
        L.di = sqrt(L.dx * L.dx + L.dy * L.dy);
#endif
        return L;
    }

    // ramp regularisation speed from 0 (well-shaped) to CellShapingSpeed * c_s (very
    // distorted), scaled by local sound speed so the correction respects local time scales.
    // Writes the final mesh velocity for cell i.
    HD static void blend_into_mesh_velocity(
        hsize_t i, VMesh* mesh, POINT_TYPE v_gas, LloydDisplacement L, const hydro::primvars* primvar) {
        if (L.di > 0.0 && L.Ri > 0.0) {
            // ramp factor: 0 below 0.75 * threshold, up to CellShapingSpeed at threshold
            const double threshold = CellShapingFactor * L.Ri;
            double       fraction  = 0.0;
            if (L.di > 0.75 * threshold) {
                fraction = (L.di > threshold) ? CellShapingSpeed
                                              : CellShapingSpeed * (L.di - 0.75 * threshold) / (0.25 * threshold);
            }

            // add fraction * c_s along the displacement direction
            if (fraction > 0.0) {
                const double rho     = primvar->rho[i];
                hydro::prim  state_i = get_state(i, primvar);
                const double p       = fmax(0.0, hydro::get_P_ideal_gas(&state_i));
                if (rho > 0.0 && p > 0.0) {
                    const double ci = sqrt(gamma_eos * p / rho);
                    v_gas.x += fraction * ci * L.dx / L.di;
                    v_gas.y += fraction * ci * L.dy / L.di;
#ifdef dim_3D
                    v_gas.z += fraction * ci * L.dz / L.di;
#endif
                }
            }
        }

        // commit the final velocity
        mesh->v_mesh[i].x = v_gas.x;
        mesh->v_mesh[i].y = v_gas.y;
#ifdef dim_3D
        mesh->v_mesh[i].z = v_gas.z;
#endif
    }

    // ============================================================
    // CUDA kernels
    // ============================================================
#ifndef CPU_DEBUG

    GLOBAL void kernel_mesh_velocities(hsize_t                         n_hydro,
                                       VMesh*                          mesh,
                                       const hydro::primvars*          primvar,
                                       const gradients::PrimGradients* grads) {
        const hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        compute_mesh_velocity_for_cell(i, mesh, primvar, grads);
    }

    GLOBAL void kernel_move_mesh(hsize_t n_hydro, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        const hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        move_mesh_for_cell(i, mesh, dt, pts);
    }

    GLOBAL void kernel_volume_correct(
        hsize_t n_hydro, const double* old_volumes, const double* new_volumes, double* rho, double* E) {
        const hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        volume_correct_for_cell(i, old_volumes, new_volumes, rho, E);
    }

#endif // !CPU_DEBUG

#endif // MOVING_MESH

} // namespace voronoi
