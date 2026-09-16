// implements the gradient estimate (gradients.h)

#include "../profiler/profiler.h"
#include "gradients.h"
#include <cmath>

namespace gradients {

    HD void                 compute_gradient_for_cell(uint64_t, const VMesh*, const hydro::primvars*, PrimGradients*);
    HD static inline double limit_single_gradient(const double      value,
                                                  const double      min_value,
                                                  const double      max_value,
                                                  const POINT_TYPE& d,
                                                  const POINT_TYPE& grad);
    HD static inline double
    recon_pressure(const hydro::prim& state_i, const PrimGradient& grad_i, const POINT_TYPE& d, double s);
    HD static inline double
    pressure_safe_scale(const hydro::prim& state_i, const PrimGradient& grad_i, const POINT_TYPE& d, double p_floor);

    // one gradient per cell
    void compute_prim_gradients(const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads) {
        PROFILE("GRAD");

        parallel_for<_GRAD_BLOCK_SIZE_, 2>(
            "GRAD_KERNEL", mesh->n_hydro, [=] HD(size_t i) { compute_gradient_for_cell(i, mesh, primvar, grads); });
    }

    // time derivative of rho, v and E from the Euler equations
    HD void time_gradient(hydro::prim state_i, PrimGradient grad_i, hydro::prim* dWdt) {

        // divergence of v, and v times its gradient
        double v2   = point_dot(state_i.v, state_i.v);
        double divv = grad_i.vx.x + grad_i.vy.y;
        double kinx = state_i.v.x * grad_i.vx.x + state_i.v.y * grad_i.vy.x;
        double kiny = state_i.v.x * grad_i.vx.y + state_i.v.y * grad_i.vy.y;
#ifdef dim_3D
        divv += grad_i.vz.z;
        kinx += state_i.v.z * grad_i.vz.x;
        kiny += state_i.v.z * grad_i.vz.y;
        const double kinz = state_i.v.x * grad_i.vx.z + state_i.v.y * grad_i.vy.z + state_i.v.z * grad_i.vz.z;
#endif

        // pressure and its gradient
        const double P     = (gamma_eos - 1.0) * (state_i.E - 0.5 * state_i.rho * v2);
        const double dP_dx = (gamma_eos - 1.0) * (grad_i.E.x - 0.5 * (v2 * grad_i.rho.x + 2.0 * state_i.rho * kinx));
        const double dP_dy = (gamma_eos - 1.0) * (grad_i.E.y - 0.5 * (v2 * grad_i.rho.y + 2.0 * state_i.rho * kiny));
#ifdef dim_3D
        const double dP_dz = (gamma_eos - 1.0) * (grad_i.E.z - 0.5 * (v2 * grad_i.rho.z + 2.0 * state_i.rho * kinz));
#endif

        // continuity
        dWdt->rho = -(state_i.v.x * grad_i.rho.x + state_i.v.y * grad_i.rho.y + state_i.rho * divv);
#ifdef dim_3D
        dWdt->rho -= state_i.v.z * grad_i.rho.z;
#endif

        // momentum
        double inv_rho = 1.0 / state_i.rho;
        dWdt->v.x      = -(state_i.v.x * grad_i.vx.x + state_i.v.y * grad_i.vx.y) - dP_dx * inv_rho;
        dWdt->v.y      = -(state_i.v.x * grad_i.vy.x + state_i.v.y * grad_i.vy.y) - dP_dy * inv_rho;
#ifdef dim_3D
        dWdt->v.x -= state_i.v.z * grad_i.vx.z;
        dWdt->v.y -= state_i.v.z * grad_i.vy.z;
        dWdt->v.z =
            -(state_i.v.x * grad_i.vz.x + state_i.v.y * grad_i.vz.y + state_i.v.z * grad_i.vz.z) - dP_dz * inv_rho;
#endif

        // energy
        dWdt->E = -(state_i.v.x * (grad_i.E.x + dP_dx) + state_i.v.y * (grad_i.E.y + dP_dy) + (state_i.E + P) * divv);
#ifdef dim_3D
        dWdt->E -= state_i.v.z * (grad_i.E.z + dP_dz);
#endif
    }

    // least squares fit over the faces of cell i, then the limiters
    HD void
    compute_gradient_for_cell(uint64_t i, const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads) {

        hydro::prim state_i = get_state(i, primvar);

// normal equations: one matrix for all variables, one right side each
#ifdef dim_2D
        double m00 = 0.0, m01 = 0.0, m11 = 0.0;
        double b_rho_0 = 0.0, b_rho_1 = 0.0;
        double b_vx_0 = 0.0, b_vx_1 = 0.0;
        double b_vy_0 = 0.0, b_vy_1 = 0.0;
        double b_E_0 = 0.0, b_E_1 = 0.0;
#else
        double m00 = 0.0, m01 = 0.0, m02 = 0.0, m11 = 0.0, m12 = 0.0, m22 = 0.0;
        double b_rho_0 = 0.0, b_rho_1 = 0.0, b_rho_2 = 0.0;
        double b_vx_0 = 0.0, b_vx_1 = 0.0, b_vx_2 = 0.0;
        double b_vy_0 = 0.0, b_vy_1 = 0.0, b_vy_2 = 0.0;
        double b_vz_0 = 0.0, b_vz_1 = 0.0, b_vz_2 = 0.0;
        double b_E_0 = 0.0, b_E_1 = 0.0, b_E_2 = 0.0;
#endif

        // range of the cell and its neighbours, for the limiter
        double min_rho = state_i.rho, max_rho = state_i.rho;
        double min_vx = state_i.v.x, max_vx = state_i.v.x;
        double min_vy = state_i.v.y, max_vy = state_i.v.y;
#ifdef dim_3D
        double min_vz = state_i.v.z, max_vz = state_i.v.z;
#endif
        double min_E = state_i.E, max_E = state_i.E;

        uint64_t  face_count  = mesh->face_counts[i];
        uint64_t  face_start  = mesh->face_ptr[i];
        const int n_hydro_int = (int)mesh->n_hydro;

        // every face adds its neighbour, weighted by face area over distance squared
        for (uint64_t fj = 0; fj < face_count; fj++) {
            uint64_t face_idx = face_start + fj;
            int      neighbor = mesh->neighbor_cell[face_idx];

            POINT_TYPE dx    = point_diff_periodic(get_seed_at(neighbor, n_hydro_int, mesh), mesh->seeds[i]);
            double     dist2 = point_dot(dx, dx);
            // a neighbour sitting on the seed would blow the weight up
            if (dist2 < 1e-24) continue;

            double face_area = mesh->face_area[face_idx];
            double weight    = face_area / dist2;

            m00 += weight * dx.x * dx.x;
            m01 += weight * dx.x * dx.y;
            m11 += weight * dx.y * dx.y;
#ifdef dim_3D
            m02 += weight * dx.x * dx.z;
            m12 += weight * dx.y * dx.z;
            m22 += weight * dx.z * dx.z;
#endif

            // right side: the difference to the neighbour
            hydro::prim state_j = get_state_at(neighbor, n_hydro_int, primvar);
            hydro::prim d_state;
            d_state.rho = state_j.rho - state_i.rho;
            d_state.v.x = state_j.v.x - state_i.v.x;
            d_state.v.y = state_j.v.y - state_i.v.y;
#ifdef dim_3D
            d_state.v.z = state_j.v.z - state_i.v.z;
#endif
            d_state.E = state_j.E - state_i.E;

            b_rho_0 += weight * dx.x * d_state.rho;
            b_rho_1 += weight * dx.y * d_state.rho;
            b_vx_0 += weight * dx.x * d_state.v.x;
            b_vx_1 += weight * dx.y * d_state.v.x;
            b_vy_0 += weight * dx.x * d_state.v.y;
            b_vy_1 += weight * dx.y * d_state.v.y;
            b_E_0 += weight * dx.x * d_state.E;
            b_E_1 += weight * dx.y * d_state.E;
#ifdef dim_3D
            b_rho_2 += weight * dx.z * d_state.rho;
            b_vx_2 += weight * dx.z * d_state.v.x;
            b_vy_2 += weight * dx.z * d_state.v.y;
            b_vz_0 += weight * dx.x * d_state.v.z;
            b_vz_1 += weight * dx.y * d_state.v.z;
            b_vz_2 += weight * dx.z * d_state.v.z;
            b_E_2 += weight * dx.z * d_state.E;
#endif

            min_rho = fmin(min_rho, state_j.rho);
            max_rho = fmax(max_rho, state_j.rho);
            min_vx  = fmin(min_vx, state_j.v.x);
            max_vx  = fmax(max_vx, state_j.v.x);
            min_vy  = fmin(min_vy, state_j.v.y);
            max_vy  = fmax(max_vy, state_j.v.y);
#ifdef dim_3D
            min_vz = fmin(min_vz, state_j.v.z);
            max_vz = fmax(max_vz, state_j.v.z);
#endif
            min_E = fmin(min_E, state_j.E);
            max_E = fmax(max_E, state_j.E);
        }

#ifdef dim_2D
        // same matrix, one solve per variable
        solve_weighted_lsq_2d(m00, m01, m11, b_rho_0, b_rho_1, &grads->rho[i]);
        solve_weighted_lsq_2d(m00, m01, m11, b_vx_0, b_vx_1, &grads->vx[i]);
        solve_weighted_lsq_2d(m00, m01, m11, b_vy_0, b_vy_1, &grads->vy[i]);
        solve_weighted_lsq_2d(m00, m01, m11, b_E_0, b_E_1, &grads->E[i]);
#else
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_rho_0, b_rho_1, b_rho_2, &grads->rho[i]);
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vx_0, b_vx_1, b_vx_2, &grads->vx[i]);
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vy_0, b_vy_1, b_vy_2, &grads->vy[i]);
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vz_0, b_vz_1, b_vz_2, &grads->vz[i]);
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_E_0, b_E_1, b_E_2, &grads->E[i]);
#endif

        // limiter: scale the gradient down until no face value leaves the neighbour range
        double alpha_rho = 1.0, alpha_vx = 1.0, alpha_vy = 1.0, alpha_E = 1.0;
#ifdef dim_3D
        double alpha_vz = 1.0;
#endif
        for (uint64_t fj = 0; fj < face_count; fj++) {
            uint64_t   face_idx = face_start + fj;
            int        neighbor = mesh->neighbor_cell[face_idx];
            POINT_TYPE dx       = point_diff_periodic(get_seed_at(neighbor, n_hydro_int, mesh), mesh->seeds[i]);
            // halfway to the neighbour seed
            POINT_TYPE d = point_mul(0.5, dx);

            alpha_rho = fmin(alpha_rho, limit_single_gradient(state_i.rho, min_rho, max_rho, d, grads->rho[i]));
            alpha_vx  = fmin(alpha_vx, limit_single_gradient(state_i.v.x, min_vx, max_vx, d, grads->vx[i]));
            alpha_vy  = fmin(alpha_vy, limit_single_gradient(state_i.v.y, min_vy, max_vy, d, grads->vy[i]));
#ifdef dim_3D
            alpha_vz = fmin(alpha_vz, limit_single_gradient(state_i.v.z, min_vz, max_vz, d, grads->vz[i]));
#endif
            alpha_E = fmin(alpha_E, limit_single_gradient(state_i.E, min_E, max_E, d, grads->E[i]));
        }

        grads->rho[i] = point_mul(alpha_rho, grads->rho[i]);
        grads->vx[i]  = point_mul(alpha_vx, grads->vx[i]);
        grads->vy[i]  = point_mul(alpha_vy, grads->vy[i]);
#ifdef dim_3D
        grads->vz[i] = point_mul(alpha_vz, grads->vz[i]);
#endif
        grads->E[i] = point_mul(alpha_E, grads->E[i]);

        // second limiter: keep the reconstructed pressure above the floor
        const double p_floor       = 1e-12;
        PrimGradient grad_i_scaled = grads->load(i);
        double       alpha_p       = 1.0;
        for (uint64_t fj = 0; fj < face_count; fj++) {
            uint64_t   face_idx = face_start + fj;
            int        neighbor = mesh->neighbor_cell[face_idx];
            POINT_TYPE dx       = point_diff_periodic(get_seed_at(neighbor, n_hydro_int, mesh), mesh->seeds[i]);
            POINT_TYPE d        = point_mul(0.5, dx);
            alpha_p             = fmin(alpha_p, pressure_safe_scale(state_i, grad_i_scaled, d, p_floor));
        }
        if (alpha_p < 1.0) {
            grads->rho[i] = point_mul(alpha_p, grads->rho[i]);
            grads->vx[i]  = point_mul(alpha_p, grads->vx[i]);
            grads->vy[i]  = point_mul(alpha_p, grads->vy[i]);
#ifdef dim_3D
            grads->vz[i] = point_mul(alpha_p, grads->vz[i]);
#endif
            grads->E[i] = point_mul(alpha_p, grads->E[i]);
        }
    }

    // factor that keeps value + grad . d between min_value and max_value
    HD static inline double limit_single_gradient(const double      value,
                                                  const double      min_value,
                                                  const double      max_value,
                                                  const POINT_TYPE& d,
                                                  const POINT_TYPE& grad) {
        double dp  = point_dot(grad, d);
        double fac = 1.0;

        if (dp > 0.0) {
            if (value + dp > max_value) {
                if (max_value > value) {
                    fac = (max_value - value) / dp;
                } else {
                    fac = 0.0;
                }
            }
        } else if (dp < 0.0) {
            if (value + dp < min_value) {
                if (min_value < value) {
                    fac = (min_value - value) / dp;
                } else {
                    fac = 0.0;
                }
            }
        }

        if (fac < 0.0) { fac = 0.0; }
        if (fac > 1.0) { fac = 1.0; }

        return fac;
    }

    // pressure at d with the gradient scaled by s
    HD static inline double
    recon_pressure(const hydro::prim& state_i, const PrimGradient& grad_i, const POINT_TYPE& d, double s) {
        double rho = state_i.rho + s * point_dot(grad_i.rho, d);
        double vx  = state_i.v.x + s * point_dot(grad_i.vx, d);
        double vy  = state_i.v.y + s * point_dot(grad_i.vy, d);
#ifdef dim_3D
        double vz = state_i.v.z + s * point_dot(grad_i.vz, d);
        double v2 = vx * vx + vy * vy + vz * vz;
#else
        double v2 = vx * vx + vy * vy;
#endif
        double E = state_i.E + s * point_dot(grad_i.E, d);
        return (gamma_eos - 1.0) * (E - 0.5 * rho * v2);
    }

    // largest scale that holds the pressure at the floor, by bisection
    HD static inline double
    pressure_safe_scale(const hydro::prim& state_i, const PrimGradient& grad_i, const POINT_TYPE& d, double p_floor) {
        if (recon_pressure(state_i, grad_i, d, 1.0) >= p_floor) return 1.0;

        double s_lo = 0.0;
        double s_hi = 1.0;
        for (int it = 0; it < 16; ++it) {
            double s_mid = 0.5 * (s_lo + s_hi);
            if (recon_pressure(state_i, grad_i, d, s_mid) >= p_floor)
                s_lo = s_mid;
            else
                s_hi = s_mid;
        }
        return s_lo;
    }

} // namespace gradients
