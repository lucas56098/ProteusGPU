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
    HD static inline double pressure(const hydro::prim& state);
    HD static inline POINT_TYPE
    face_point(uint64_t i, uint64_t face_idx, const VMesh* mesh, int n_hydro_int, const POINT_TYPE& com_off_i);

    // one gradient per cell
    void compute_prim_gradients(const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads) {
        PROFILE("GRAD");

        parallel_for<_GRAD_BLOCK_SIZE_, 2>(
            "GRAD_KERNEL", mesh->n_hydro, [=] HD(size_t i) { compute_gradient_for_cell(i, mesh, primvar, grads); });
    }

    // time derivative of rho, v and P from the Euler equations; along the flow with a moving mesh
    HD void time_gradient(hydro::prim state_i, double P_i, PrimGradient grad_i, PrimRates* dWdt) {

        double divv = grad_i.vx.x + grad_i.vy.y;
#ifdef dim_3D
        divv += grad_i.vz.z;
#endif
        const double inv_rho = 1.0 / state_i.rho;

#ifdef MOVING_MESH
        // along the flow: the extrapolation vector carries the motion of the gas instead
        dWdt->rho = -state_i.rho * divv;
        dWdt->v.x = -grad_i.P.x * inv_rho;
        dWdt->v.y = -grad_i.P.y * inv_rho;
#ifdef dim_3D
        dWdt->v.z = -grad_i.P.z * inv_rho;
#endif
        dWdt->P = -gamma_eos * P_i * divv;
#else
        // continuity
        dWdt->rho = -(point_dot(state_i.v, grad_i.rho) + state_i.rho * divv);

        // momentum
        dWdt->v.x = -point_dot(state_i.v, grad_i.vx) - grad_i.P.x * inv_rho;
        dWdt->v.y = -point_dot(state_i.v, grad_i.vy) - grad_i.P.y * inv_rho;
#ifdef dim_3D
        dWdt->v.z = -point_dot(state_i.v, grad_i.vz) - grad_i.P.z * inv_rho;
#endif

        // pressure
        dWdt->P = -(point_dot(state_i.v, grad_i.P) + gamma_eos * P_i * divv);
#endif
    }

    // least squares fit over the faces of cell i, then the limiters
    HD void
    compute_gradient_for_cell(uint64_t i, const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads) {

        hydro::prim  state_i = get_state(i, primvar);
        const double P_i     = pressure(state_i);

// normal equations: one matrix for all variables, one right side each
#ifdef dim_2D
        double m00 = 0.0, m01 = 0.0, m11 = 0.0;
        double b_rho_0 = 0.0, b_rho_1 = 0.0;
        double b_vx_0 = 0.0, b_vx_1 = 0.0;
        double b_vy_0 = 0.0, b_vy_1 = 0.0;
        double b_P_0 = 0.0, b_P_1 = 0.0;
#else
        double m00 = 0.0, m01 = 0.0, m02 = 0.0, m11 = 0.0, m12 = 0.0, m22 = 0.0;
        double b_rho_0 = 0.0, b_rho_1 = 0.0, b_rho_2 = 0.0;
        double b_vx_0 = 0.0, b_vx_1 = 0.0, b_vx_2 = 0.0;
        double b_vy_0 = 0.0, b_vy_1 = 0.0, b_vy_2 = 0.0;
        double b_vz_0 = 0.0, b_vz_1 = 0.0, b_vz_2 = 0.0;
        double b_P_0 = 0.0, b_P_1 = 0.0, b_P_2 = 0.0;
#endif

        // range of the cell and its neighbours, for the limiter
        double min_rho = state_i.rho, max_rho = state_i.rho;
        double min_vx = state_i.v.x, max_vx = state_i.v.x;
        double min_vy = state_i.v.y, max_vy = state_i.v.y;
#ifdef dim_3D
        double min_vz = state_i.v.z, max_vz = state_i.v.z;
#endif
        double min_P = P_i, max_P = P_i;

        uint64_t  face_count  = mesh->face_counts[i];
        uint64_t  face_start  = mesh->face_ptr[i];
        const int n_hydro_int = (int)mesh->n_hydro;

        // the cell values belong to the centroids, so the fit and the limiters work from there
        const POINT_TYPE com_off_i = get_com_off_at((int)i, n_hydro_int, mesh);
        grads->anchor[i]           = com_off_i;

        // every face adds its neighbour, weighted by face area over distance squared
        for (uint64_t fj = 0; fj < face_count; fj++) {
            uint64_t face_idx = face_start + fj;
            int      neighbor = mesh->neighbor_cell[face_idx];

            // centroid of the neighbour - centroid of the cell
            const POINT_TYPE seed_dx = point_diff_periodic(get_seed_at(neighbor, n_hydro_int, mesh), mesh->seeds[i]);
            POINT_TYPE       dx = point_add(seed_dx, point_sub(get_com_off_at(neighbor, n_hydro_int, mesh), com_off_i));
            double           dist2 = point_dot(dx, dx);
            // a neighbour sitting on the centroid would blow the weight up
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
            const double P_j = pressure(state_j);
            const double d_P = P_j - P_i;

            b_rho_0 += weight * dx.x * d_state.rho;
            b_rho_1 += weight * dx.y * d_state.rho;
            b_vx_0 += weight * dx.x * d_state.v.x;
            b_vx_1 += weight * dx.y * d_state.v.x;
            b_vy_0 += weight * dx.x * d_state.v.y;
            b_vy_1 += weight * dx.y * d_state.v.y;
            b_P_0 += weight * dx.x * d_P;
            b_P_1 += weight * dx.y * d_P;
#ifdef dim_3D
            b_rho_2 += weight * dx.z * d_state.rho;
            b_vx_2 += weight * dx.z * d_state.v.x;
            b_vy_2 += weight * dx.z * d_state.v.y;
            b_vz_0 += weight * dx.x * d_state.v.z;
            b_vz_1 += weight * dx.y * d_state.v.z;
            b_vz_2 += weight * dx.z * d_state.v.z;
            b_P_2 += weight * dx.z * d_P;
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
            min_P = fmin(min_P, P_j);
            max_P = fmax(max_P, P_j);
        }

#ifdef dim_2D
        // same matrix, one solve per variable
        solve_weighted_lsq_2d(m00, m01, m11, b_rho_0, b_rho_1, &grads->rho[i]);
        solve_weighted_lsq_2d(m00, m01, m11, b_vx_0, b_vx_1, &grads->vx[i]);
        solve_weighted_lsq_2d(m00, m01, m11, b_vy_0, b_vy_1, &grads->vy[i]);
        solve_weighted_lsq_2d(m00, m01, m11, b_P_0, b_P_1, &grads->P[i]);
#else
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_rho_0, b_rho_1, b_rho_2, &grads->rho[i]);
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vx_0, b_vx_1, b_vx_2, &grads->vx[i]);
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vy_0, b_vy_1, b_vy_2, &grads->vy[i]);
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vz_0, b_vz_1, b_vz_2, &grads->vz[i]);
        solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_P_0, b_P_1, b_P_2, &grads->P[i]);
#endif

        // limiter: scale the gradient down until no face value leaves the neighbour range
        double alpha_rho = 1.0, alpha_vx = 1.0, alpha_vy = 1.0, alpha_P = 1.0;
#ifdef dim_3D
        double alpha_vz = 1.0;
#endif
        for (uint64_t fj = 0; fj < face_count; fj++) {
            const POINT_TYPE d = face_point(i, face_start + fj, mesh, n_hydro_int, com_off_i);

            alpha_rho = fmin(alpha_rho, limit_single_gradient(state_i.rho, min_rho, max_rho, d, grads->rho[i]));
            alpha_vx  = fmin(alpha_vx, limit_single_gradient(state_i.v.x, min_vx, max_vx, d, grads->vx[i]));
            alpha_vy  = fmin(alpha_vy, limit_single_gradient(state_i.v.y, min_vy, max_vy, d, grads->vy[i]));
#ifdef dim_3D
            alpha_vz = fmin(alpha_vz, limit_single_gradient(state_i.v.z, min_vz, max_vz, d, grads->vz[i]));
#endif
            alpha_P = fmin(alpha_P, limit_single_gradient(P_i, min_P, max_P, d, grads->P[i]));
        }

        grads->rho[i] = point_mul(alpha_rho, grads->rho[i]);
        grads->vx[i]  = point_mul(alpha_vx, grads->vx[i]);
        grads->vy[i]  = point_mul(alpha_vy, grads->vy[i]);
#ifdef dim_3D
        grads->vz[i] = point_mul(alpha_vz, grads->vz[i]);
#endif
        grads->P[i] = point_mul(alpha_P, grads->P[i]);
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

    // pressure of an ideal gas with total energy E
    HD static inline double pressure(const hydro::prim& state) {
        return (gamma_eos - 1.0) * (state.E - 0.5 * state.rho * point_dot(state.v, state.v));
    }

    // face centroid seen from the centroid of cell i, the same point the first flux extrapolates to
    HD static inline POINT_TYPE
    face_point(uint64_t i, uint64_t face_idx, const VMesh* mesh, int n_hydro_int, const POINT_TYPE& com_off_i) {
        const double3    seed_j = get_seed_at(mesh->neighbor_cell[face_idx], n_hydro_int, mesh);
        const double3    delta  = {wrap_periodic_delta(seed_j.x - mesh->seeds[i].x),
                                   wrap_periodic_delta(seed_j.y - mesh->seeds[i].y),
                                   wrap_periodic_delta(seed_j.z - mesh->seeds[i].z)};
        const POINT_TYPE dx     = point_diff_periodic(seed_j, mesh->seeds[i]);
        const POINT_TYPE r      = face_centroid_from_seed(
            point_mul(0.5, dx), compute_geom(delta), &mesh->f_mid_local[face_idx * (DIMENSION - 1)]);
        return point_sub(r, com_off_i);
    }

} // namespace gradients
