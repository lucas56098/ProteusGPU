
// mesh motion (voronoi.h)

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

    namespace {
        // how far the seed sits from the centre of its cell
        struct LloydDisplacement {
            double dx;
            double dy;
            double dz;
            double di; // length of dx, dy, dz
            double Ri; // radius of a ball of the cell volume
        };
    } // namespace

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

    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads) {
        parallel_for<_MESH_BLOCK_SIZE_>(
            "V_MESH", mesh->n_hydro, [=] HD(size_t i) { compute_mesh_velocity_for_cell(i, mesh, primvar, grads); });
    }

    // moves the seeds, migrates cells, builds the mesh again
    void move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::primvars* primvar_aux) {

        // the state is per volume, keep the old one
        gpu_memcpy(mesh->old_volumes, mesh->volumes, mesh->n_hydro * sizeof(double));

        advance_seeds_by_dt(mesh, dt, mesh->scratch_move);

        // a rebalance moves the brick borders, so more cells change rank
        if (proteus_mpi::rebalance_decide(sim.step, mesh, mesh->scratch_move)) {
            proteus_mpi::migrate_for_rebalance(mesh, primvar, primvar_aux);
            proteus_mpi::rebalance_log_after_migration(mesh);
        } else {
            proteus_mpi::migrate_seeds(mesh, primvar, primvar_aux);
        }

        compute_periodic_mesh(mesh, mesh->scratch_move, mesh->n_hydro, primvar, primvar_aux, dt);

        correct_for_volume_change(mesh, primvar_aux);
    }

    // writes the moved positions into pts, mesh->seeds stays
    static void advance_seeds_by_dt(VMesh* mesh, double dt, POINT_TYPE* pts) {
        const uint64_t n_hydro = mesh->n_hydro;
        parallel_for<_MESH_BLOCK_SIZE_>(
            "MOVE_MESH", n_hydro, [=] HD(size_t i) { move_mesh_for_cell(i, mesh, dt, pts); });
        GPU_SYNC();
    }

    // the cell volume changed, so rho and E follow
    static void correct_for_volume_change(VMesh* mesh, hydro::primvars* primvar) {
        const uint64_t n_hydro     = mesh->n_hydro;
        const double*  old_volumes = mesh->old_volumes;
        const double*  new_volumes = mesh->volumes;
        double*        rho         = primvar->rho;
        double*        E           = primvar->E;

        parallel_for<_MESH_BLOCK_SIZE_>(
            "VOL_CORRECT", n_hydro, [=] HD(size_t i) { volume_correct_for_cell(i, old_volumes, new_volumes, rho, E); });
    }

    HD void compute_mesh_velocity_for_cell(uint64_t                        i,
                                           VMesh*                          mesh,
                                           const hydro::primvars*          primvar,
                                           const gradients::PrimGradients* grads) {
        const POINT_TYPE        v_gas = gas_velocity_for_cell(i, primvar);
        const LloydDisplacement L     = lloyd_correction_for_cell(i, mesh, primvar, grads);
        blend_into_mesh_velocity(i, mesh, v_gas, L, primvar);
    }

    // wrapped back into the box
    HD void move_mesh_for_cell(uint64_t i, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        pts[i].x = fmod((mesh->seeds[i].x + dt * mesh->v_mesh[i].x) + 1.0, 1.0);
        pts[i].y = fmod((mesh->seeds[i].y + dt * mesh->v_mesh[i].y) + 1.0, 1.0);
#ifdef dim_3D
        pts[i].z = fmod((mesh->seeds[i].z + dt * mesh->v_mesh[i].z) + 1.0, 1.0);
#endif
    }

    HD void
    volume_correct_for_cell(uint64_t i, const double* old_volumes, const double* new_volumes, double* rho, double* E) {
        const double ratio = old_volumes[i] / new_volumes[i];
        rho[i] *= ratio;
        E[i] *= ratio;
    }

    HD static POINT_TYPE gas_velocity_for_cell(uint64_t i, const hydro::primvars* primvar) {
        POINT_TYPE v;
        v.x = primvar->v[i].x;
        v.y = primvar->v[i].y;
#ifdef dim_3D
        v.z = primvar->v[i].z;
#endif
        return v;
    }

    // seed to centre of mass, plus a step up the density gradient
    HD static LloydDisplacement lloyd_correction_for_cell(uint64_t                        i,
                                                          const VMesh*                    mesh,
                                                          const hydro::primvars*          primvar,
                                                          const gradients::PrimGradients* grads) {
        LloydDisplacement L{};
#ifdef dim_2D
        L.Ri = sqrt(fmax(mesh->volumes[i], 0.0) / PI);
#else
        L.Ri = portable_cbrt(3.0 * fmax(mesh->volumes[i], 0.0) / (4.0 * PI));
#endif

        L.dx = wrap_periodic_delta(mesh->com[i].x - mesh->seeds[i].x);
        L.dy = wrap_periodic_delta(mesh->com[i].y - mesh->seeds[i].y);
#ifdef dim_3D
        L.dz = wrap_periodic_delta(mesh->com[i].z - mesh->seeds[i].z);
#endif

        if (grads != nullptr && L.Ri > 0.0) {
#ifdef dim_3D
            const double dgrad = sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y +
                                      grads->rho[i].z * grads->rho[i].z);
#else
            const double dgrad = sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y);
#endif
            if (dgrad > 0.0) {
                // length over which the density changes by its own value; the step is at most Ri / 4
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

#ifdef dim_3D
        L.di = sqrt(L.dx * L.dx + L.dy * L.dy + L.dz * L.dz);
#else
        L.di = sqrt(L.dx * L.dx + L.dy * L.dy);
#endif
        return L;
    }

    // gas velocity plus the terms that keep the cell in shape
    HD static void blend_into_mesh_velocity(
        uint64_t i, VMesh* mesh, POINT_TYPE v_gas, LloydDisplacement L, const hydro::primvars* primvar) {
        if (L.di > 0.0 && L.Ri > 0.0) {
            // how far off centre a seed may sit
            const double threshold = CellShapingFactor * L.Ri;
            double       fraction  = 0.0;
            // nothing below three quarters of it, full speed above
            if (L.di > 0.75 * threshold) {
                fraction = (L.di > threshold) ? CellShapingSpeed
                                              : CellShapingSpeed * (L.di - 0.75 * threshold) / (0.25 * threshold);
            }

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
        // cells far below the reference size push towards their larger neighbours
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
                    const int      n_hydro_int = (int)mesh->n_hydro;
                    const double   Vi          = mesh->volumes[i];
                    const uint64_t fp          = mesh->face_ptr[i];
                    const uint64_t fc          = mesh->face_counts[i];
                    double         gx = 0.0, gy = 0.0;
#ifdef dim_3D
                    double gz = 0.0;
#endif
                    // which direction the larger neighbours are in
                    for (uint64_t fj = 0; fj < fc; fj++) {
                        const int nb = mesh->neighbor_cell[fp + fj];
                        if (nb < 0) continue;
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

        mesh->v_mesh[i].x = v_gas.x;
        mesh->v_mesh[i].y = v_gas.y;
#ifdef dim_3D
        mesh->v_mesh[i].z = v_gas.z;
#endif
    }

#endif

} // namespace voronoi
