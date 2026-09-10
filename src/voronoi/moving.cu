#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../hydro/riemann.h"
#include "../mpi/migrate.h"
#include "../mpi/rebalance.h"
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
    HD void compute_mesh_velocity_for_cell(uint64_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    HD void move_mesh_for_cell(uint64_t, const VMesh*, double, POINT_TYPE*);
    HD void
    volume_correct_for_cell(uint64_t i, const double* old_volumes, const double* new_volumes, double* rho, double* E);

    static void                 advance_seeds_by_dt(VMesh* mesh, double dt, POINT_TYPE* pts);
    static void                 correct_for_volume_change(VMesh* mesh, hydro::primvars* primvar);
    static HD POINT_TYPE        gas_velocity_for_cell(uint64_t i, const hydro::primvars* primvar);
    static HD LloydDisplacement lloyd_correction_for_cell(uint64_t                        i,
                                                          const VMesh*                    mesh,
                                                          const hydro::primvars*          primvar,
                                                          const gradients::PrimGradients* grads);
    static HD void              blend_into_mesh_velocity(
                     uint64_t i, VMesh* mesh, POINT_TYPE v_gas, LloydDisplacement L, const hydro::primvars* primvar);

    // ============================================================
    // Main routines
    // ============================================================

    // compute the mesh-point velocity (gas velocity + Lloyd regularization) for every cell
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads) {
        parallel_for<_MESH_BLOCK_SIZE_>(
            "V_MESH", mesh->n_hydro, [=] HD(size_t i) { compute_mesh_velocity_for_cell(i, mesh, primvar, grads); });
    }

    // advance seeds by v_mesh * dt, migrate across ranks, rebuild mesh,
    // and correct primvar_aux for the cell-volume change.
    // mesh->scratch_move is the seed-position buffer through all three voronoi-side calls:
    // advance_seeds_by_dt writes into it, the migrate path compacts/extends it in place,
    // compute_periodic_mesh reads from it.
    //
    // On rebalance steps the per-step Cart-neighbor migrate is replaced by the full
    // Alltoallv migrate_for_rebalance. Either way there is exactly one mesh build per
    // step — the rebalance does not trigger a separate compute_periodic_mesh.
    void move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::primvars* primvar_aux) {

        // store old volumes for volume correction afterwards
        gpu_memcpy(mesh->old_volumes, mesh->volumes, mesh->n_hydro * sizeof(double));

        // advance seed positions by v_mesh * dt into mesh->scratch_move
        advance_seeds_by_dt(mesh, dt, mesh->scratch_move);

        // migrate cells whose new bucket is owned by another rank;
        // updates mesh->n_hydro and rewrites mesh->scratch_move in place
        if (proteus_mpi::rebalance_decide(sim.step, mesh, mesh->scratch_move)) {
            proteus_mpi::migrate_for_rebalance(mesh, primvar, primvar_aux);
            proteus_mpi::rebalance_log_after_migration(mesh);
        } else {
            proteus_mpi::migrate_seeds(mesh, primvar, primvar_aux);
        }

        // rebuild the Voronoi mesh from the new seed positions; dt lets the CPU fallback
        // fold each cell's perturbation delta into v_mesh as delta/dt
        compute_periodic_mesh(mesh, mesh->scratch_move, mesh->n_hydro, primvar, primvar_aux, dt);

        // correct primvar_aux for the cell-volume change (conservation: rho, E scale with old/new ratio)
        correct_for_volume_change(mesh, primvar_aux);
    }

    // ============================================================
    // Helpers
    // ============================================================

    static void advance_seeds_by_dt(VMesh* mesh, double dt, POINT_TYPE* pts) {
        const uint64_t n_hydro = mesh->n_hydro;
        parallel_for<_MESH_BLOCK_SIZE_>(
            "MOVE_MESH", n_hydro, [=] HD(size_t i) { move_mesh_for_cell(i, mesh, dt, pts); });
        GPU_SYNC(); // migrate_seeds reads pts on the host below
    }

    static void correct_for_volume_change(VMesh* mesh, hydro::primvars* primvar) {
        const uint64_t n_hydro     = mesh->n_hydro;
        const double*  old_volumes = mesh->old_volumes;
        const double*  new_volumes = mesh->volumes;
        double*        rho         = primvar->rho;
        double*        E           = primvar->E;

        parallel_for<_MESH_BLOCK_SIZE_>(
            "VOL_CORRECT", n_hydro, [=] HD(size_t i) { volume_correct_for_cell(i, old_volumes, new_volumes, rho, E); });
    }

    // ============================================================
    // Per-cell work (parallel_for bodies)
    // ============================================================

    // mesh-point velocity = gas velocity + Lloyd regularization, both scaled by sound speed
    HD void compute_mesh_velocity_for_cell(uint64_t                        i,
                                           VMesh*                          mesh,
                                           const hydro::primvars*          primvar,
                                           const gradients::PrimGradients* grads) {
        const POINT_TYPE        v_gas = gas_velocity_for_cell(i, primvar);
        const LloydDisplacement L     = lloyd_correction_for_cell(i, mesh, primvar, grads);
        blend_into_mesh_velocity(i, mesh, v_gas, L, primvar);
    }

    // advance one seed by v_mesh * dt with periodic wrap into [0, 1)
    HD void move_mesh_for_cell(uint64_t i, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        pts[i].x = fmod((mesh->seeds[i].x + dt * mesh->v_mesh[i].x) + 1.0, 1.0);
        pts[i].y = fmod((mesh->seeds[i].y + dt * mesh->v_mesh[i].y) + 1.0, 1.0);
#ifdef dim_3D
        pts[i].z = fmod((mesh->seeds[i].z + dt * mesh->v_mesh[i].z) + 1.0, 1.0);
#endif
    }

    // scale rho and E by old/new cell-volume ratio so total mass / energy stay conserved
    // when cell volume changes during the mesh move
    HD void
    volume_correct_for_cell(uint64_t i, const double* old_volumes, const double* new_volumes, double* rho, double* E) {
        const double ratio = old_volumes[i] / new_volumes[i];
        rho[i] *= ratio;
        E[i] *= ratio;
    }

    // read primvar->v[i] into a POINT_TYPE; the seed's gas velocity component
    HD static POINT_TYPE gas_velocity_for_cell(uint64_t i, const hydro::primvars* primvar) {
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
    HD static LloydDisplacement lloyd_correction_for_cell(uint64_t                        i,
                                                          const VMesh*                    mesh,
                                                          const hydro::primvars*          primvar,
                                                          const gradients::PrimGradients* grads) {
        // effective cell radius from the volume
        LloydDisplacement L{};
#ifdef dim_2D
        L.Ri = sqrt(fmax(mesh->volumes[i], 0.0) / PI);
#else
        L.Ri = portable_cbrt(3.0 * fmax(mesh->volumes[i], 0.0) / (4.0 * PI));
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
        uint64_t i, VMesh* mesh, POINT_TYPE v_gas, LloydDisplacement L, const hydro::primvars* primvar) {
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

#ifdef VOL_REGULARIZE
        // size-equalizing drift: nudge small cells toward larger neighbours (soft de-refinement).
        // engages only once smallness Ri_ref/Ri exceeds the VOL_REGULARIZE threshold, ramping to
        // the full speed cap at twice the threshold.
        if (L.Ri > 0.0 && mesh->Ri_ref > 0.0) {
            const double thresh     = (double)VOL_REGULARIZE;
            const double size_ratio = mesh->Ri_ref / L.Ri;
            double       vf         = 0.0;
            if (size_ratio > thresh) vf = VolShapingSpeed * fmin((size_ratio - thresh) / thresh, 1.0);

            if (vf > 0.0) {
                const double rho     = primvar->rho[i];
                hydro::prim  state_i = get_state(i, primvar);
                const double p       = fmax(0.0, hydro::get_P_ideal_gas(&state_i));
                if (rho > 0.0 && p > 0.0) {
                    // discrete size-gradient: sum_j area_j * (V_j - V_i) * unit(seed_i -> seed_j)
                    const int      n_hydro_int = (int)mesh->n_hydro;
                    const double   Vi          = mesh->volumes[i];
                    const uint64_t fp          = mesh->face_ptr[i];
                    const uint64_t fc          = mesh->face_counts[i];
                    double         gx = 0.0, gy = 0.0;
#ifdef dim_3D
                    double gz = 0.0;
#endif
                    for (uint64_t fj = 0; fj < fc; fj++) {
                        const int nb = mesh->neighbor_cell[fp + fj];
                        if (nb < 0) continue; // box boundary
                        const double3 sj = get_seed_at(nb, n_hydro_int, mesh);
                        const double  rx = wrap_periodic_delta(sj.x - mesh->seeds[i].x);
                        const double  ry = wrap_periodic_delta(sj.y - mesh->seeds[i].y);
#ifdef dim_3D
                        const double rz   = wrap_periodic_delta(sj.z - mesh->seeds[i].z);
                        const double rlen = sqrt(rx * rx + ry * ry + rz * rz);
#else
                        const double rlen = sqrt(rx * rx + ry * ry);
#endif
                        if (rlen < 1e-30) continue;
                        const double w = mesh->face_area[fp + fj] * (get_volume_at(nb, n_hydro_int, mesh) - Vi) / rlen;
                        gx += w * rx;
                        gy += w * ry;
#ifdef dim_3D
                        gz += w * rz;
#endif
                    }
#ifdef dim_3D
                    const double glen = sqrt(gx * gx + gy * gy + gz * gz);
#else
                    const double glen = sqrt(gx * gx + gy * gy);
#endif
                    if (glen > 0.0) {
                        // scale by max(c_s, |v_gas|): in cold condensing gas c_s collapses, so
                        // the inflow speed is the signal that must be matched to hold the mesh.
                        const double ci = sqrt(gamma_eos * p / rho);
#ifdef dim_3D
                        const double vmag = sqrt(primvar->v[i].x * primvar->v[i].x + primvar->v[i].y * primvar->v[i].y +
                                                 primvar->v[i].z * primvar->v[i].z);
#else
                        const double vmag = sqrt(primvar->v[i].x * primvar->v[i].x + primvar->v[i].y * primvar->v[i].y);
#endif
                        const double speed = fmax(ci, vmag);
                        v_gas.x += vf * speed * gx / glen;
                        v_gas.y += vf * speed * gy / glen;
#ifdef dim_3D
                        v_gas.z += vf * speed * gz / glen;
#endif
                    }
                }
            }
        }
#endif

        // commit the final velocity
        mesh->v_mesh[i].x = v_gas.x;
        mesh->v_mesh[i].y = v_gas.y;
#ifdef dim_3D
        mesh->v_mesh[i].z = v_gas.z;
#endif
    }

#endif // MOVING_MESH

} // namespace voronoi
