#include "finite_volume_solver.h"
#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../profiler/profiler.h"
#include "riemann.h"
#include <cstring>

namespace hydro {

    // init hydrostruct from IC data
    primvars* init(int n_hydro) {

        // allocate prim struct
        primvars* hydro_data = new primvars();
        hydro_data->rho      = (double*)malloc(n_hydro * sizeof(double));
        hydro_data->v        = (POINT_TYPE*)malloc(n_hydro * sizeof(POINT_TYPE));
        hydro_data->E        = (double*)malloc(n_hydro * sizeof(double));

        // fill hydro_data from icData
        for (int i = 0; i < n_hydro; i++) {
            hydro_data->rho[i] = icData.rho[i];
            hydro_data->E[i]   = icData.Energy[i];

            hydro_data->v[i].x = icData.vel[DIMENSION * i];
            hydro_data->v[i].y = icData.vel[DIMENSION * i + 1];
#ifdef dim_3D
            hydro_data->v[i].z = icData.vel[DIMENSION * i + 2];
#endif
        }

        std::cout << "HYDRO: Initialized primitive variables for " << n_hydro << " particles" << std::endl;

        return hydro_data;
    }

    // free the primvars again
    void free_prim(primvars** primvar) {
        free((*primvar)->rho);
        free((*primvar)->v);
        free((*primvar)->E);
        free(*primvar);
        *primvar = NULL;
    }

    // main hydro routine (computes fluxes and updates states)
    void hydro_step(double dt, const VMesh* mesh, primvars* primvar) {

        // allocate primitive accumulator for new state
        primvars prim_new;
        allocate_prim_buffer(mesh->n_hydro, &prim_new);

        // initialize new state from old primitive variables
        std::memcpy(prim_new.rho, primvar->rho, mesh->n_hydro * sizeof(double));
        std::memcpy(prim_new.v, primvar->v, mesh->n_hydro * sizeof(POINT_TYPE));
        std::memcpy(prim_new.E, primvar->E, mesh->n_hydro * sizeof(double));

        // compute gradients once from old state
        PrimGradients* grads = gradients::compute_prim_gradients(mesh, primvar);

        // first half update (no time extrapolation)
        apply_flux_update(0.5 * dt, 0.0, mesh, primvar, grads, &prim_new);

        // second half update (full dt primitive time extrapolation)
        apply_flux_update(0.5 * dt, dt, mesh, primvar, grads, &prim_new);

        gradients::free_prim_gradients(grads);

        // final copy: primvar = prim_new
        std::memcpy(primvar->rho, prim_new.rho, mesh->n_hydro * sizeof(double));
        std::memcpy(primvar->v, prim_new.v, mesh->n_hydro * sizeof(POINT_TYPE));
        std::memcpy(primvar->E, prim_new.E, mesh->n_hydro * sizeof(double));

        free_prim_buffer(&prim_new);
    }

    // apply one part of RK2 flux update (either with dt_extrap = 0 or dt)
    void apply_flux_update(double               dt_update,
                           double               dt_extrap,
                           const VMesh*         mesh,
                           const primvars*      prim_old,
                           const PrimGradients* grads,
                           primvars*            prim_new) {

        PROFILE_START("HYDRO_STEP (par)");
        const bool do_time_extrap = (dt_extrap != 0.0);

// loop over all active cells to calc new primvars
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(_OMP_HYDRO_THREADS_)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {

            const hsize_t face_base = mesh->face_ptr[i];

            // get state of cell i
            prim          state_i = get_state(i, mesh, prim_old);
            PrimGradients grad_i  = grads[i];

            prim total_flux;

            // calculate total_flux by summing over edge flux * edge_length
            for (hsize_t j = 0; j < mesh->face_counts[i]; j++) {

                // get state of cell j
                int           face_idx = face_base + j;
                hsize_t       index_j  = mesh->neighbor_cell[face_idx];
                prim          state_j  = get_state(index_j, mesh, prim_old);
                PrimGradients grad_j   = grads[hydro_index(index_j, mesh)];

                // second-order reconstruction at face center
                prim       state_l;
                prim       state_r;
                POINT_TYPE dx = point_diff(mesh->seeds[index_j], mesh->seeds[i]);

                // apply the gradients
                apply_spatial_extrapolation(state_i, grad_i, point_mul(0.5, dx), &state_l);
                apply_spatial_extrapolation(state_j, grad_j, point_mul(-0.5, dx), &state_r);

                // only in second half of RK2
                if (do_time_extrap) {
                    apply_time_extrapolation(state_i, grad_i, dt_extrap, &state_l);
                    apply_time_extrapolation(state_j, grad_j, dt_extrap, &state_l);
                }

                // ensure rho > rho_min, P > P_min
                keep_state_physical(&state_l);
                keep_state_physical(&state_r);

                // calc flux using riemann solver
#ifdef RIEMANN_HLL
                prim flux_ij = riemann_hll(i, j, state_l, state_r, mesh);
#elif RIEMANN_HLLC
                prim flux_ij = riemann_hllc(i, j, state_l, state_r, mesh);
#else
#error "No Riemann solver specified in Config.sh: choose RIEMANN_HLL or RIEMANN_HLLC"
#endif

                // get face area/length
                double face_area = mesh->face_area[face_idx];

                // add to total flux * area
                total_flux.rho += flux_ij.rho * face_area;
                total_flux.v.x += flux_ij.v.x * face_area;
                total_flux.v.y += flux_ij.v.y * face_area;
#ifdef dim_3D
                total_flux.v.z += flux_ij.v.z * face_area;
#endif
                total_flux.E += flux_ij.E * face_area;
            }

            double frac    = dt_update / mesh->volumes[i];
            double rho_old = prim_new->rho[i];
            double rho_new = rho_old - frac * total_flux.rho;
            double rho_inv = 1.0 / rho_new;

            prim_new->rho[i] = rho_new;
            prim_new->v[i].x = (rho_old * prim_new->v[i].x - frac * total_flux.v.x) * rho_inv;
            prim_new->v[i].y = (rho_old * prim_new->v[i].y - frac * total_flux.v.y) * rho_inv;
#ifdef dim_3D
            prim_new->v[i].z = (rho_old * prim_new->v[i].z - frac * total_flux.v.z) * rho_inv;
#endif
            prim_new->E[i] -= frac * total_flux.E;
        }

        PROFILE_END("HYDRO_STEP (par)");
    }

    // apply linear spatial extrapolation
    void apply_spatial_extrapolation(const prim state, const PrimGradients gradient, POINT_TYPE dx, prim* st_extrap) {

        st_extrap->rho = state.rho + point_dot(gradient.rho, dx);
        st_extrap->v.x = state.v.x + point_dot(gradient.vx, dx);
        st_extrap->v.y = state.v.y + point_dot(gradient.vy, dx);
#ifdef dim_3D
        st_extrap->v.z = state.v.z + point_dot(gradient.vz, dx);
#endif
        st_extrap->E = state.E + point_dot(gradient.E, dx);
    }

    // apply primitive time extrapolation: W -> W + dt_extrap * dW/dt(cell_idx)
    void apply_time_extrapolation(prim state_i, PrimGradients grad_i, double dt_extrap, prim* st_extrap) {

        // first compute time derivatives dW/dt
        prim dWdt;
        gradients::time_gradient(state_i, grad_i, &dWdt);

        // do time extrapolation
        st_extrap->rho += dt_extrap * dWdt.rho;
        st_extrap->v.x += dt_extrap * dWdt.v.x;
        st_extrap->v.y += dt_extrap * dWdt.v.y;
#ifdef dim_3D
        st_extrap->v.z += dt_extrap * dWdt.v.z;
#endif
        st_extrap->E += dt_extrap * dWdt.E;
    }

    // keep states physical
    void keep_state_physical(prim* state) {
        const double rho_floor = 1e-12;
        const double p_floor   = 1e-12;

        if (state->rho < rho_floor) { state->rho = rho_floor; }

        double v2 = state->v.x * state->v.x + state->v.y * state->v.y;
#ifdef dim_3D
        v2 += state->v.z * state->v.z;
#endif
        double ekin = 0.5 * state->rho * v2;
        double emin = ekin + p_floor / (_gamma_ - 1.0);
        if (state->E < emin) { state->E = emin; }
    }

    // calc timestep using CFL condition for euler equations
    double dt_CFL(double CFL, const VMesh* mesh, const primvars* primvar) {
        PROFILE_START("CFL (par)");

        double min_dt = 1e100;

#ifdef USE_OPENMP
#pragma omp parallel for reduction(min : min_dt)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            // build prim state for cell i to get pressure
            prim state_i;
            state_i.rho = primvar->rho[i];
            state_i.E   = primvar->E[i];
            state_i.v.x = primvar->v[i].x;
            state_i.v.y = primvar->v[i].y;
#ifdef dim_3D
            state_i.v.z = primvar->v[i].z;
#endif

            double P   = get_P_ideal_gas(&state_i);
            double c_i = sqrt(_gamma_ * P / state_i.rho);

#ifdef dim_2D
            double R_i   = sqrt(mesh->volumes[i] / M_PI);
            double v_abs = sqrt(state_i.v.x * state_i.v.x + state_i.v.y * state_i.v.y);
#else
            double R_i   = cbrt(3.0 * mesh->volumes[i] / (4.0 * M_PI));
            double v_abs = sqrt(state_i.v.x * state_i.v.x + state_i.v.y * state_i.v.y + state_i.v.z * state_i.v.z);
#endif

            double dt_i = CFL * (R_i / (c_i + v_abs));

            if (dt_i < min_dt) { min_dt = dt_i; }
        }

        PROFILE_END("CFL (par)");
        return min_dt;
    }

} // namespace hydro