#include "../profiler/profiler.h"
#include "gradients.h"
#include <cmath>

namespace gradients {

    // forward declarations
    HD void compute_gradient_for_cell(hsize_t, const VMesh*, const hydro::primvars*, PrimGradients*);

#ifndef CPU_DEBUG
    // kernel
    GLOBAL void kernel_compute_gradients(hsize_t, const VMesh*, const hydro::primvars*, PrimGradients*);
#endif

    // ============================================================
    // Main routine
    // ============================================================

    void compute_prim_gradients(const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads) {
        PROFILE_START("GRADIENTS (par)");
        zero_grad(grads);

#ifndef CPU_DEBUG
        int tpb    = _GRAD_BLOCK_SIZE_;
        int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
        PROFILE_GPU_START("kernel_compute_gradients");
        kernel_compute_gradients<<<blocks, tpb>>>(mesh->n_hydro, mesh, primvar, grads);
        PROFILE_GPU_END("kernel_compute_gradients");
#else
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(_OMP_HYDRO_THREADS_)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            compute_gradient_for_cell(i, mesh, primvar, grads);
        }
#endif

        PROFILE_END("GRADIENTS (par)");
    }

    // ============================================================
    // CUDA kernel wrapper
    // ============================================================
#ifndef CPU_DEBUG

    GLOBAL void __launch_bounds__(_GRAD_BLOCK_SIZE_, 2) kernel_compute_gradients(hsize_t                n_hydro,
                                                                                 const VMesh*           mesh,
                                                                                 const hydro::primvars* primvar,
                                                                                 PrimGradients*         grads) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        compute_gradient_for_cell(i, mesh, primvar, grads);
    }

#endif // !CPU_DEBUG

    // ============================================================
    // Per-cell gradient computation (called by kernel and CPU loop)
    // ============================================================

    HD void
    compute_gradient_for_cell(hsize_t i, const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads) {

        hydro::prim state_i = get_state(i, mesh, primvar);

        // weighted least-squares accumulators (M and b for each primitive variable)
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

        // running min/max over neighbours (used by the slope limiter below)
        double min_rho = state_i.rho, max_rho = state_i.rho;
        double min_vx = state_i.v.x, max_vx = state_i.v.x;
        double min_vy = state_i.v.y, max_vy = state_i.v.y;
#ifdef dim_3D
        double min_vz = state_i.v.z, max_vz = state_i.v.z;
#endif
        double min_E = state_i.E, max_E = state_i.E;

        // accumulate over each face/neighbour
        hsize_t face_count = mesh->face_counts[i];
        hsize_t face_start = mesh->face_ptr[i];

        for (hsize_t fj = 0; fj < face_count; fj++) {
            hsize_t face_idx     = face_start + fj;
            hsize_t neighbor_raw = (hsize_t)mesh->neighbor_cell[face_idx];
            hsize_t neighbor_h   = hydro_index(neighbor_raw, mesh);

            // separation vector and inverse-distance weighting
            POINT_TYPE dx    = point_diff(mesh->seeds[neighbor_raw], mesh->seeds[i]);
            double     dist2 = point_dot(dx, dx);
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

            hydro::prim state_j = get_state(neighbor_h, mesh, primvar);
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

        // solve M * grad = b for each primitive (one shared M, separate RHS per variable)
#ifdef dim_2D
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

        // shrink each gradient so reconstructed face values
        // stay between the cell's neighbour min/max
        double alpha_rho = 1.0, alpha_vx = 1.0, alpha_vy = 1.0, alpha_E = 1.0;
#ifdef dim_3D
        double alpha_vz = 1.0;
#endif
        for (hsize_t fj = 0; fj < face_count; fj++) {
            hsize_t    face_idx     = face_start + fj;
            hsize_t    neighbor_raw = (hsize_t)mesh->neighbor_cell[face_idx];
            POINT_TYPE dx           = point_diff(mesh->seeds[neighbor_raw], mesh->seeds[i]);
            POINT_TYPE d            = point_mul(0.5, dx);

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
    }

    // largest fac in [0,1] such that value + fac*dp stays in [min,max]
    HD inline double limit_single_gradient(const double      value,
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

} // namespace gradients
