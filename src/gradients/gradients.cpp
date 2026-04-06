#include "gradients.h"
#include "../profiler/profiler.h"
#include <cmath>
#include <cstdlib>

namespace gradients {

    // compute spatial gradients
    PrimGradients* compute_prim_gradients(const VMesh* mesh, const primvars* primvar) {
        PROFILE_START("GRADIENTS (par)");
        PrimGradients* grads = (PrimGradients*)calloc(mesh->n_hydro, sizeof(PrimGradients));

        // loop over all hydro cells
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(_OMP_HYDRO_THREADS_)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {

            // load state (previously qi_rho...)
            prim state_i = get_state(i, mesh, primvar);

            // weighted least-squares matrix and rhs vectors
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

            // min/max bounds for slope limiting
            double min_rho = state_i.rho, max_rho = state_i.rho;
            double min_vx = state_i.v.x, max_vx = state_i.v.x;
            double min_vy = state_i.v.y, max_vy = state_i.v.y;
#ifdef dim_3D
            double min_vz = state_i.v.z, max_vz = state_i.v.z;
#endif
            double min_E = state_i.E, max_E = state_i.E;

            hsize_t face_count = mesh->face_counts[i];
            hsize_t face_start = mesh->face_ptr[i];

            // build weighted least-squares system from neighbors (Mg = b)
            for (hsize_t fj = 0; fj < face_count; fj++) {
                hsize_t face_idx     = face_start + fj;
                hsize_t neighbor_raw = (hsize_t)mesh->neighbor_cell[face_idx];
                hsize_t neighbor_h   = hydro_index(neighbor_raw, mesh);

                POINT_TYPE dx    = point_diff(mesh->seeds[neighbor_raw], mesh->seeds[i]);
                double     dist2 = point_dot(dx, dx);
                if (dist2 < 1e-24) { continue; }

                // build M
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

                // get neighbor state
                prim state_j = get_state(neighbor_h, mesh, primvar);
                prim d_state;
                d_state.rho = state_j.rho - state_i.rho;
                d_state.v.x = state_j.v.x - state_i.v.x;
                d_state.v.y = state_j.v.y - state_i.v.y;
#ifdef dim_3D
                d_state.v.z = state_j.v.z - state_i.v.z;
#endif
                d_state.E = state_j.E - state_i.E;

                // build b
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

                // set min, max values
                min_rho = std::min(min_rho, state_j.rho);
                max_rho = std::max(max_rho, state_j.rho);
                min_vx  = std::min(min_vx, state_j.v.x);
                max_vx  = std::max(max_vx, state_j.v.x);
                min_vy  = std::min(min_vy, state_j.v.y);
                max_vy  = std::max(max_vy, state_j.v.y);
#ifdef dim_3D
                min_vz = std::min(min_vz, state_j.v.z);
                max_vz = std::max(max_vz, state_j.v.z);
#endif
                min_E = std::min(min_E, state_j.E);
                max_E = std::max(max_E, state_j.E);
            }

            // solve gradient system (g = M^-1 b)
#ifdef dim_2D
            solve_weighted_lsq_2d(m00, m01, m11, b_rho_0, b_rho_1, &grads[i].rho);
            solve_weighted_lsq_2d(m00, m01, m11, b_vx_0, b_vx_1, &grads[i].vx);
            solve_weighted_lsq_2d(m00, m01, m11, b_vy_0, b_vy_1, &grads[i].vy);
            solve_weighted_lsq_2d(m00, m01, m11, b_E_0, b_E_1, &grads[i].E);
#else
            solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_rho_0, b_rho_1, b_rho_2, &grads[i].rho);
            solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vx_0, b_vx_1, b_vx_2, &grads[i].vx);
            solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vy_0, b_vy_1, b_vy_2, &grads[i].vy);
            solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_vz_0, b_vz_1, b_vz_2, &grads[i].vz);
            solve_weighted_lsq_3d(m00, m01, m02, m11, m12, m22, b_E_0, b_E_1, b_E_2, &grads[i].E);
#endif

            // limit gradients using all local faces
            for (hsize_t fj = 0; fj < face_count; fj++) {
                hsize_t    face_idx     = face_start + fj;
                hsize_t    neighbor_raw = (hsize_t)mesh->neighbor_cell[face_idx];
                POINT_TYPE dx           = point_diff(mesh->seeds[neighbor_raw], mesh->seeds[i]);
                POINT_TYPE d            = point_mul(0.5, dx);

                limit_single_gradient(state_i.rho, min_rho, max_rho, d, &grads[i].rho);
                limit_single_gradient(state_i.v.x, min_vx, max_vx, d, &grads[i].vx);
                limit_single_gradient(state_i.v.y, min_vy, max_vy, d, &grads[i].vy);
#ifdef dim_3D
                limit_single_gradient(state_i.v.z, min_vz, max_vz, d, &grads[i].vz);
#endif
                limit_single_gradient(state_i.E, min_E, max_E, d, &grads[i].E);
            }
        }

        PROFILE_END("GRADIENTS (par)");
        return grads;
    }

    // free gradients buffer
    void free_prim_gradients(PrimGradients* grads) {
        free(grads);
    }

    // compute dW/dt ("time gradients") based on states and gradients
    void time_gradient(prim state_i, PrimGradients grad_i, prim* dWdt) {

        // precomputed helpers
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

        // pressure and its spatial derivatives
        const double P     = (_gamma_ - 1.0) * (state_i.E - 0.5 * state_i.rho * v2);
        const double dP_dx = (_gamma_ - 1.0) * (grad_i.E.x - 0.5 * (v2 * grad_i.rho.x + 2.0 * state_i.rho * kinx));
        const double dP_dy = (_gamma_ - 1.0) * (grad_i.E.y - 0.5 * (v2 * grad_i.rho.y + 2.0 * state_i.rho * kiny));
#ifdef dim_3D
        const double dP_dz = (_gamma_ - 1.0) * (grad_i.E.z - 0.5 * (v2 * grad_i.rho.z + 2.0 * state_i.rho * kinz));
#endif

        // compute drho/dt
        dWdt->rho = -(state_i.v.x * grad_i.rho.x + state_i.v.y * grad_i.rho.y + state_i.rho * divv);
#ifdef dim_3D
        dWdt->rho -= state_i.v.z * grad_i.rho.z;
#endif

        // compute dv/dt
        dWdt->v.x = -(state_i.v.x * grad_i.vx.x + state_i.v.y * grad_i.vx.y) - dP_dx / state_i.rho;
        dWdt->v.y = -(state_i.v.x * grad_i.vy.x + state_i.v.y * grad_i.vy.y) - dP_dy / state_i.rho;
#ifdef dim_3D
        dWdt->v.x -= state_i.v.z * grad_i.vx.z;
        dWdt->v.y -= state_i.v.z * grad_i.vy.z;
        dWdt->v.z =
            -(state_i.v.x * grad_i.vz.x + state_i.v.y * grad_i.vz.y + state_i.v.z * grad_i.vz.z) - dP_dz / state_i.rho;
#endif

        // compute dE/dt
        dWdt->E = -(state_i.v.x * (grad_i.E.x + dP_dx) + state_i.v.y * (grad_i.E.y + dP_dy) + (state_i.E + P) * divv);
#ifdef dim_3D
        dWdt->E -= state_i.v.z * (grad_i.E.z + dP_dz);
#endif
    }

    // arepo-like face limiter for one scalar gradient
    inline void limit_single_gradient(
        const double value, const double min_value, const double max_value, const POINT_TYPE& d, POINT_TYPE* grad) {
        double dp  = point_dot(*grad, d);
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

        *grad = point_mul(fac, *grad);
    }

} // namespace gradients
