// implements the finite volume step (finite_volume_solver.h)

#include "../astro/agn.h"
#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../io/input.h"
#include "../mpi/decomp.h"
#include "../mpi/halo.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "finite_volume_solver.h"
#include "riemann.cu"
#include <utility>

namespace hydro {

    HD void flux_update_for_cell(
        uint64_t, double, bool, double, const VMesh*, const primvars*, const gradients::PrimGradients*, primvars*);
#ifdef AGN_ENABLED
    HD double dt_CFL_for_cell(uint64_t, double, const VMesh*, const primvars*, bool, const astro::AgnParams&);
#else
    HD double dt_CFL_for_cell(uint64_t, double, const VMesh*, const primvars*);
#endif
    static void check_unphysical_state(VMesh*, const primvars*);
    static void reset_prim_new(VMesh* mesh, primvars* primvar, primvars* prim_new);
    static void swap_primvars(primvars* primvar, primvars* prim_new);

    // allocates the state arrays and fills them from the IC
    void init_hydro() {
        const int n_hydro = (int)sim.n_hydro;

        sim.primvar = gpu_alloc<primvars>(1);
        allocate_prim_buffer(sim.n_hydro, sim.primvar, true);

        for (int i = 0; i < n_hydro; i++) {
            sim.primvar->rho[i] = ic_data.rho[i];
            sim.primvar->E[i]   = ic_data.energy[i];
            sim.primvar->v[i].x = ic_data.vel[DIMENSION * i];
            sim.primvar->v[i].y = ic_data.vel[DIMENSION * i + 1];
#ifdef dim_3D
            sim.primvar->v[i].z = ic_data.vel[DIMENSION * i + 2];
#endif
        }

        sim.prim_new = gpu_alloc<primvars>(1);
        allocate_prim_buffer(sim.n_hydro, sim.prim_new, false);

        sim.grads = gpu_alloc<gradients::PrimGradients>(1);
        gradients::allocate_grad(sim.n_hydro, sim.grads);

        sim.dt = gpu_alloc<double>(1);

        const int n_hydro_global = logging::sum_global(n_hydro);
        logging::root() << "HYDRO: Initialized hydro for " << n_hydro_global << " particles" << std::endl;
    }

    // gives them back at the end of the run
    void free_hydro() {
        free_prim_buffer(sim.primvar);
        free_prim_buffer(sim.prim_new);
        gradients::free_grad(sim.grads);

        gpu_free(sim.primvar);
        gpu_free(sim.prim_new);
        gpu_free(sim.grads);
        gpu_free(sim.dt);

        sim.primvar  = nullptr;
        sim.prim_new = nullptr;
        sim.grads    = nullptr;
        sim.dt       = nullptr;
    }

    // one step: half the fluxes, move the mesh, the other half
    void hydro_step(double dt, VMesh* mesh, primvars* primvar) {

        primvars*                 prim_new = sim.prim_new;
        gradients::PrimGradients* grads    = sim.grads;

        // the ghosts get the state of their own rank
        proteus_mpi::halo_exchange_primvars(mesh, primvar);

        // prim_new collects the update, primvar stays as it is
        reset_prim_new(mesh, primvar, prim_new);

        // gradients on the mesh as it is now
        gradients::compute_prim_gradients(mesh, primvar, grads);
        proteus_mpi::halo_exchange_gradients(mesh, grads);

#ifdef MOVING_MESH
        voronoi::compute_mesh_velocities(mesh, primvar, grads);
        proteus_mpi::halo_exchange_v_mesh(mesh);
#endif

        // first half step, states taken at the current time
        apply_flux_update(0.5 * dt, 0.0, mesh, primvar, grads, prim_new);
        logging::root() << "HYDRO: Computed " << logging::sum_global((int)mesh->num_faces) << " fluxes (1/2)"
                        << std::endl;

#ifdef MOVING_MESH

        // move the mesh, then gradients again on the new one
        voronoi::move_mesh(mesh, dt, primvar, prim_new);

        gradients::compute_prim_gradients(mesh, primvar, grads);
        proteus_mpi::halo_exchange_gradients(mesh, grads);
#endif

        // second half step, states extrapolated to the end of the step
        apply_flux_update(0.5 * dt, dt, mesh, primvar, grads, prim_new);
        logging::root() << "HYDRO: Computed " << logging::sum_global((int)mesh->num_faces) << " fluxes (2/2)"
                        << std::endl;

        // prim_new is the state of the run from here
        swap_primvars(primvar, prim_new);

        check_unphysical_state(mesh, primvar);
    }

    // one flux update over all cells; dt_extrap > 0 also extrapolates the states in time
    void apply_flux_update(double                          dt_update,
                           double                          dt_extrap,
                           const VMesh*                    mesh,
                           const primvars*                 prim_old,
                           const gradients::PrimGradients* grads,
                           primvars*                       prim_new) {

        PROFILE("FLUX");

        const bool do_time_extrap = (dt_extrap != 0.0);

        parallel_for<_HYDRO_BLOCK_SIZE_, 2>("FLUX_KERNEL", mesh->n_hydro, [=] HD(size_t i) {
            flux_update_for_cell(i, dt_update, do_time_extrap, dt_extrap, mesh, prim_old, grads, prim_new);
        });
    }

    // smallest CFL step of all cells, over all ranks
    double calc_timestep(double CFL, const VMesh* mesh, const primvars* primvar) {

        {
            PROFILE("CFL");

#ifdef AGN_ENABLED
            const bool             agn_firing = astro::agn_is_firing();
            const astro::AgnParams p_agn      = astro::agn_params();
#endif
            *sim.dt = parallel_reduce<_HYDRO_BLOCK_SIZE_, double>(
                "DT_CFL",
                mesh->n_hydro,
                1e100,
                [] HD(double a, double b) { return a < b ? a : b; },
                [=] HD(size_t i) {
#ifdef AGN_ENABLED
                    return dt_CFL_for_cell(i, CFL, mesh, primvar, agn_firing, p_agn);
#else
                    return dt_CFL_for_cell(i, CFL, mesh, primvar);
#endif
                });
        }

        proteus_mpi::halo_dt_allreduce(sim.dt);

        // do not step over the next output or the end of the run
        if (sim.t_sim + *sim.dt > sim.t_nextoutput) { *sim.dt = sim.t_nextoutput - sim.t_sim; }
        if (sim.t_sim + *sim.dt > sim.t_end) { *sim.dt = sim.t_end - sim.t_sim; }

        return *sim.dt;
    }

    // prim_new = primvar
    static void reset_prim_new(VMesh* mesh, primvars* primvar, primvars* prim_new) {

        PROFILE("COPY_PRIMVAR");
        gpu_memcpy(prim_new->rho, primvar->rho, mesh->n_hydro * sizeof(double));
        gpu_memcpy(prim_new->v, primvar->v, mesh->n_hydro * sizeof(POINT_TYPE));
        gpu_memcpy(prim_new->E, primvar->E, mesh->n_hydro * sizeof(double));
    }

    static void swap_primvars(primvars* primvar, primvars* prim_new) {
        GPU_SYNC();
        std::swap(primvar->rho, prim_new->rho);
        std::swap(primvar->v, prim_new->v);
        std::swap(primvar->E, prim_new->E);
    }

    // bad cells of each kind
    struct UnphysCounts {
        int rho_bad, E_bad, nan_bad;
    };

    // stops the run if any cell has rho <= 0, E <= 0 or a NaN
    static void check_unphysical_state(VMesh* mesh, const primvars* primvar) {
        PROFILE("UNPHYS_CHECK");

        const UnphysCounts counts = parallel_reduce<_HYDRO_BLOCK_SIZE_, UnphysCounts>(
            "UNPHYS_KERNEL",
            mesh->n_hydro,
            UnphysCounts{0, 0, 0},
            [] HD(UnphysCounts a, UnphysCounts b) {
                return UnphysCounts{a.rho_bad + b.rho_bad, a.E_bad + b.E_bad, a.nan_bad + b.nan_bad};
            },
            [=] HD(size_t i) {
                const double rho     = primvar->rho[i];
                const double E       = primvar->E[i];
                const bool   has_nan = !(rho == rho) || !(E == E) || !(primvar->v[i].x == primvar->v[i].x) ||
                                     !(primvar->v[i].y == primvar->v[i].y)
#ifdef dim_3D
                                     || !(primvar->v[i].z == primvar->v[i].z)
#endif
                    ;
                return UnphysCounts{rho <= 0.0 ? 1 : 0, E <= 0.0 ? 1 : 0, has_nan ? 1 : 0};
            });

        const int rho_bad = logging::sum_global(counts.rho_bad);
        const int E_bad   = logging::sum_global(counts.E_bad);
        const int nan_bad = logging::sum_global(counts.nan_bad);
        if (rho_bad == 0 && E_bad == 0 && nan_bad == 0) return;

        if (rho_bad > 0) logging::root() << "HYDRO: WARNING: " << rho_bad << " cells with rho<=0" << std::endl;
        if (E_bad > 0) logging::root() << "HYDRO: WARNING: " << E_bad << " cells with E<=0" << std::endl;
        if (nan_bad > 0) logging::root() << "HYDRO: WARNING: " << nan_bad << " cells with NaN" << std::endl;
        proteus_mpi::exit_failure("HYDRO: ABORT: unphysical state detected — terminating run.\n");
    }

    // sums the fluxes over the faces of cell i and updates its state
    HD void flux_update_for_cell(uint64_t                        i,
                                 double                          dt_update,
                                 bool                            do_time_extrap,
                                 double                          dt_extrap,
                                 const VMesh*                    mesh,
                                 const primvars*                 prim_old,
                                 const gradients::PrimGradients* grads,
                                 primvars*                       prim_new) {

        const uint64_t face_base = mesh->face_ptr[i];

        prim                    state_i = get_state(i, prim_old);
        gradients::PrimGradient grad_i  = grads->load(i);

        prim      total_flux;
        const int n_hydro_int = (int)mesh->n_hydro;

        // one face after the other
        for (uint64_t j = 0; j < mesh->face_counts[i]; j++) {
            uint64_t                face_idx = face_base + j;
            int                     index_j  = mesh->neighbor_cell[face_idx];
            prim                    state_j  = get_state_at(index_j, n_hydro_int, prim_old);
            gradients::PrimGradient grad_j   = grads->load_at(index_j, n_hydro_int);
            double3                 seed_j   = get_seed_at(index_j, n_hydro_int, mesh);

            // face frame: n along the line between the two seeds, m and p across it
            double3 delta = {wrap_periodic_delta(seed_j.x - mesh->seeds[i].x),
                             wrap_periodic_delta(seed_j.y - mesh->seeds[i].y),
                             wrap_periodic_delta(seed_j.z - mesh->seeds[i].z)};
            geom    g     = compute_geom(delta);

#ifdef MOVING_MESH
            // the face itself moves
            POINT_TYPE vel_face, vel_face_turned;
            POINT_TYPE vm_i = mesh->v_mesh[i];
            POINT_TYPE vm_j = get_vmesh_at(index_j, n_hydro_int, mesh);
            get_vel_face(i,
                         (uint64_t)index_j,
                         vm_i,
                         vm_j,
                         &mesh->f_mid_local[face_idx * (DIMENSION - 1)],
                         mesh,
                         g,
                         &vel_face,
                         &vel_face_turned);
#endif

            prim       state_l, state_r;
            POINT_TYPE dx = point_diff_periodic(seed_j, mesh->seeds[i]);

            // both states, extrapolated from the seed to the middle of the face
            apply_spatial_extrapolation(state_i, grad_i, point_mul(0.5, dx), &state_l);
            apply_spatial_extrapolation(state_j, grad_j, point_mul(-0.5, dx), &state_r);

            // and to the end of the step
            if (do_time_extrap) {
                apply_time_extrapolation(state_i, grad_i, dt_extrap, &state_l);
                apply_time_extrapolation(state_j, grad_j, dt_extrap, &state_r);
            }

#ifdef MOVING_MESH
            // into the frame that moves with the face
            convert_state_to_local_frame(&state_l, vel_face);
            convert_state_to_local_frame(&state_r, vel_face);
#endif

            // the extrapolation can leave the physical range
            keep_state_physical(&state_l, mesh->min_egy_spec);
            keep_state_physical(&state_r, mesh->min_egy_spec);

            // x along the normal, so the solver works in 1D
            rotate_to_face(&state_l, &g);
            rotate_to_face(&state_r, &g);

            flux_t flux_ij = riemann_hllc(state_l, state_r);

#ifdef MOVING_MESH
            convert_flux_to_lab_frame(&flux_ij, vel_face_turned);
#endif
            rotate_from_face(&flux_ij, &g);

            // the solver gives a flux per area
            double face_area = mesh->face_area[face_idx];

            total_flux.rho += flux_ij.rho * face_area;
            total_flux.v.x += flux_ij.v.x * face_area;
            total_flux.v.y += flux_ij.v.y * face_area;
#ifdef dim_3D
            total_flux.v.z += flux_ij.v.z * face_area;
#endif
            total_flux.E += flux_ij.E * face_area;
        }

        // what flows out in dt_update, spread over the cell volume
        double           frac           = dt_update / mesh->volumes[i];
        double           rho_old        = prim_new->rho[i];
        double           rho_new        = rho_old - frac * total_flux.rho;
        constexpr double RHO_FLOOR_CELL = 1e-13;

        // with a temperature floor the cell is held at the floor instead of going empty
        if (mesh->min_egy_spec > 0.0 && rho_new < RHO_FLOOR_CELL) {
            rho_new          = RHO_FLOOR_CELL;
            prim_new->rho[i] = rho_new;
            double v2        = prim_new->v[i].x * prim_new->v[i].x + prim_new->v[i].y * prim_new->v[i].y;
#ifdef dim_3D
            v2 += prim_new->v[i].z * prim_new->v[i].z;
#endif
            prim_new->E[i] = 0.5 * rho_new * v2 + rho_new * mesh->min_egy_spec;
        } else {
            // v is a velocity, so the update runs over the momentum and divides by the new density
            double rho_inv = 1.0 / rho_new;

            prim_new->rho[i] = rho_new;
            prim_new->v[i].x = (rho_old * prim_new->v[i].x - frac * total_flux.v.x) * rho_inv;
            prim_new->v[i].y = (rho_old * prim_new->v[i].y - frac * total_flux.v.y) * rho_inv;
#ifdef dim_3D
            prim_new->v[i].z = (rho_old * prim_new->v[i].z - frac * total_flux.v.z) * rho_inv;
#endif
            prim_new->E[i] -= frac * total_flux.E;

            // keep the internal energy at the floor
            if (mesh->min_egy_spec > 0.0) {
                double v2 = prim_new->v[i].x * prim_new->v[i].x + prim_new->v[i].y * prim_new->v[i].y;
#ifdef dim_3D
                v2 += prim_new->v[i].z * prim_new->v[i].z;
#endif
                double e_wall = 0.5 * rho_new * v2 + rho_new * mesh->min_egy_spec;
                if (prim_new->E[i] < e_wall) { prim_new->E[i] = e_wall; }
            }
        }
    }

// CFL step of one cell: its radius over sound speed plus velocity
#ifdef AGN_ENABLED
    HD double dt_CFL_for_cell(uint64_t                i,
                              double                  CFL,
                              const VMesh*            mesh,
                              const primvars*         primvar,
                              bool                    agn_firing,
                              const astro::AgnParams& p_agn) {
#else
    HD double dt_CFL_for_cell(uint64_t i, double CFL, const VMesh* mesh, const primvars* primvar) {
#endif

        prim state_i;
        state_i.rho = primvar->rho[i];
        state_i.E   = primvar->E[i];
        state_i.v.x = primvar->v[i].x;
        state_i.v.y = primvar->v[i].y;
#ifdef dim_3D
        state_i.v.z = primvar->v[i].z;
#endif

        double P   = get_P_ideal_gas(&state_i);
        double c_i = (state_i.rho > 0.0 && P > 0.0) ? sqrt(gamma_eos * P / state_i.rho) : 0.0;
#ifdef AGN_ENABLED
        if (agn_firing) {
            const double3 sd = mesh->seeds[i];
            const double  dx = sd.x - p_agn.cx, dy = sd.y - p_agn.cy;
#ifdef dim_3D
            const double dz = sd.z - p_agn.cz;
            const double r2 = dx * dx + dy * dy + dz * dz;
#else
            const double r2 = dx * dx + dy * dy;
#endif
            // the AGN is about to heat this gas, so use the sound speed it will have
            if (r2 < p_agn.r_T2) {
                const double c_ceiling = sqrt(p_agn.cs2_max);
                if (c_ceiling > c_i) c_i = c_ceiling;
            }
        }
#endif

#ifdef dim_2D
        // radius of a disc or ball of the cell volume
        double R_i = sqrt(mesh->volumes[i] / M_PI);
#else
        double R_i = portable_cbrt(3.0 * mesh->volumes[i] / (4.0 * M_PI));
#endif
#ifdef MOVING_MESH
        // the mesh moves along, only the rest matters
        double dvx = state_i.v.x - mesh->v_mesh[i].x;
        double dvy = state_i.v.y - mesh->v_mesh[i].y;
#ifdef dim_3D
        double dvz   = state_i.v.z - mesh->v_mesh[i].z;
        double v_sig = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
#else
        double v_sig = sqrt(dvx * dvx + dvy * dvy);
#endif
#else
#ifdef dim_2D
        double v_sig = sqrt(state_i.v.x * state_i.v.x + state_i.v.y * state_i.v.y);
#else
        double v_sig = sqrt(state_i.v.x * state_i.v.x + state_i.v.y * state_i.v.y + state_i.v.z * state_i.v.z);
#endif
#endif
#if defined(AGN_ENABLED) && defined(AGN_KINETIC)
        if (agn_firing) {
            const double3 sd  = mesh->seeds[i];
            const double  dx  = sd.x - p_agn.cx;
            const double  ady = fabs(sd.y - p_agn.cy);
#ifdef dim_3D
            const double dz    = sd.z - p_agn.cz;
            const double perp2 = dx * dx + dz * dz;
#else
            const double perp2 = dx * dx;
#endif
            // same for the velocity the jet is about to add
            if (perp2 < p_agn.r_jet2 && ady > p_agn.L_jet && ady < p_agn.L_jet + p_agn.h_jet) {
                if (p_agn.v_cap > v_sig) v_sig = p_agn.v_cap;
            }
        }
#endif

        return CFL * (R_i / (c_i + v_sig));
    }

    // floors for density and internal energy
    HD void keep_state_physical(prim* state, double min_egy_spec) {
        const double rho_floor = 1e-12;
        const double p_floor   = 1e-12;

        if (state->rho < rho_floor) { state->rho = rho_floor; }

        double v2 = state->v.x * state->v.x + state->v.y * state->v.y;
#ifdef dim_3D
        v2 += state->v.z * state->v.z;
#endif
        double ekin      = 0.5 * state->rho * v2;
        double e_int_min = (min_egy_spec > 0.0) ? state->rho * min_egy_spec : p_floor / (gamma_eos - 1.0);
        double emin      = ekin + e_int_min;
        if (state->E < emin) { state->E = emin; }
    }

    // velocity into the face frame
    HD void rotate_to_face(prim* state, geom* g) {
        double velx = state->v.x;
        double vely = state->v.y;
#ifdef dim_2D
        state->v.x = velx * g->n.x + vely * g->n.y;
        state->v.y = velx * g->m.x + vely * g->m.y;
#else
        double velz = state->v.z;
        state->v.x  = velx * g->n.x + vely * g->n.y + velz * g->n.z;
        state->v.y  = velx * g->m.x + vely * g->m.y + velz * g->m.z;
        state->v.z  = velx * g->p.x + vely * g->p.y + velz * g->p.z;
#endif
    }

    // and back
    HD void rotate_from_face(prim* state, geom* g) {
        double velx = state->v.x;
        double vely = state->v.y;
#ifdef dim_2D
        state->v.x = velx * g->n.x + vely * g->m.x;
        state->v.y = velx * g->n.y + vely * g->m.y;
#else
        double velz = state->v.z;
        state->v.x  = velx * g->n.x + vely * g->m.x + velz * g->p.x;
        state->v.y  = velx * g->n.y + vely * g->m.y + velz * g->p.y;
        state->v.z  = velx * g->n.z + vely * g->m.z + velz * g->p.z;
#endif
    }

    // state at seed + dx, from the gradient
    HD void apply_spatial_extrapolation(const prim                    state,
                                        const gradients::PrimGradient gradient,
                                        POINT_TYPE                    dx,
                                        prim*                         st_extrap) {
        st_extrap->rho = state.rho + point_dot(gradient.rho, dx);
        st_extrap->v.x = state.v.x + point_dot(gradient.vx, dx);
        st_extrap->v.y = state.v.y + point_dot(gradient.vy, dx);
#ifdef dim_3D
        st_extrap->v.z = state.v.z + point_dot(gradient.vz, dx);
#endif
        st_extrap->E = state.E + point_dot(gradient.E, dx);
    }

    // state dt_extrap later, from the Euler equations
    HD void apply_time_extrapolation(prim state_i, gradients::PrimGradient grad_i, double dt_extrap, prim* st_extrap) {
        prim dWdt;
        gradients::time_gradient(state_i, grad_i, &dWdt);

        constexpr double RHO_FLOOR_FACE = 1e-12;
        constexpr double P_FLOOR_FACE   = 1e-12;
        double           beta           = 1.0;

        // beta shortens the step so that the density stays above its floor
        if (dWdt.rho < 0.0) {
            const double denom = -dt_extrap * dWdt.rho;
            if (denom > 0.0) {
                const double beta_rho = (st_extrap->rho - RHO_FLOOR_FACE) / denom;
                if (beta_rho < beta) beta = fmax(0.0, beta_rho);
            }
        }

        {
            const double rho_b = st_extrap->rho + beta * dt_extrap * dWdt.rho;
            const double vx_b  = st_extrap->v.x + beta * dt_extrap * dWdt.v.x;
            const double vy_b  = st_extrap->v.y + beta * dt_extrap * dWdt.v.y;
#ifdef dim_3D
            const double vz_b = st_extrap->v.z + beta * dt_extrap * dWdt.v.z;
            const double v2_b = vx_b * vx_b + vy_b * vy_b + vz_b * vz_b;
#else
            const double v2_b = vx_b * vx_b + vy_b * vy_b;
#endif
            const double E_b = st_extrap->E + beta * dt_extrap * dWdt.E;
            const double P_b = (gamma_eos - 1.0) * (E_b - 0.5 * rho_b * v2_b);

            // and the pressure above its own, found by bisection
            if (P_b < P_FLOOR_FACE) {
                double lo = 0.0, hi = beta;
                for (int it = 0; it < 16; ++it) {
                    const double mid   = 0.5 * (lo + hi);
                    const double rho_m = st_extrap->rho + mid * dt_extrap * dWdt.rho;
                    const double vx_m  = st_extrap->v.x + mid * dt_extrap * dWdt.v.x;
                    const double vy_m  = st_extrap->v.y + mid * dt_extrap * dWdt.v.y;
#ifdef dim_3D
                    const double vz_m = st_extrap->v.z + mid * dt_extrap * dWdt.v.z;
                    const double v2_m = vx_m * vx_m + vy_m * vy_m + vz_m * vz_m;
#else
                    const double v2_m = vx_m * vx_m + vy_m * vy_m;
#endif
                    const double E_m = st_extrap->E + mid * dt_extrap * dWdt.E;
                    const double P_m = (gamma_eos - 1.0) * (E_m - 0.5 * rho_m * v2_m);
                    if (P_m >= P_FLOOR_FACE)
                        lo = mid;
                    else
                        hi = mid;
                }
                beta = lo;
            }
        }

        const double bdt = beta * dt_extrap;
        st_extrap->rho += bdt * dWdt.rho;
        st_extrap->v.x += bdt * dWdt.v.x;
        st_extrap->v.y += bdt * dWdt.v.y;
#ifdef dim_3D
        st_extrap->v.z += bdt * dWdt.v.z;
#endif
        st_extrap->E += bdt * dWdt.E;
    }

#ifdef MOVING_MESH
    // velocity of the face: mean of the two mesh velocities, plus the turn of the face
    HD void get_vel_face(uint64_t      i,
                         uint64_t      index_j,
                         POINT_TYPE    v_mesh_i,
                         POINT_TYPE    v_mesh_j,
                         const double* f_mid_local,
                         const VMesh*  mesh,
                         geom          g,
                         POINT_TYPE*   vel_face,
                         POINT_TYPE*   vel_face_turned) {

        double facv;

        const double3 seed_j = get_seed_at((int)index_j, (int)mesh->n_hydro, mesh);
        double        nnx    = wrap_periodic_delta(seed_j.x - mesh->seeds[i].x);
        double        nny    = wrap_periodic_delta(seed_j.y - mesh->seeds[i].y);
#ifdef dim_3D
        double nnz = wrap_periodic_delta(seed_j.z - mesh->seeds[i].z);
        double nn  = sqrt(nnx * nnx + nny * nny + nnz * nnz);
#else
        double nn = sqrt(nnx * nnx + nny * nny);
#endif

        // mean of the two seeds
        vel_face->x = 0.5 * (v_mesh_i.x + v_mesh_j.x);
        vel_face->y = 0.5 * (v_mesh_i.y + v_mesh_j.y);

// the face centre is off the middle of the seeds, so the face also turns around it
#ifdef dim_2D
        double alpha = f_mid_local[0];
        double cx    = alpha * g.m.x;
        double cy    = alpha * g.m.y;
#else
        vel_face->z  = 0.5 * (v_mesh_i.z + v_mesh_j.z);
        double alpha = f_mid_local[0];
        double beta  = f_mid_local[1];
        double cx    = alpha * g.m.x + beta * g.p.x;
        double cy    = alpha * g.m.y + beta * g.p.y;
        double cz    = alpha * g.m.z + beta * g.p.z;

        facv = (cx * (v_mesh_i.x - v_mesh_j.x) + cy * (v_mesh_i.y - v_mesh_j.y) + cz * (v_mesh_i.z - v_mesh_j.z)) / nn;

        double cc = sqrt(cx * cx + cy * cy + cz * cz);
#endif

#ifdef dim_2D
        facv      = (cx * (v_mesh_i.x - v_mesh_j.x) + cy * (v_mesh_i.y - v_mesh_j.y)) / nn;
        double cc = sqrt(cx * cx + cy * cy);
#endif

        // cap it before the face outruns the seeds
        if (cc > 0.9 * nn) facv *= (0.9 * nn) / cc;

        vel_face->x += facv * g.n.x;
        vel_face->y += facv * g.n.y;
#ifdef dim_3D
        vel_face->z += facv * g.n.z;
#endif

#ifdef dim_2D
        vel_face_turned->x = vel_face->x * g.n.x + vel_face->y * g.n.y;
        vel_face_turned->y = vel_face->x * g.m.x + vel_face->y * g.m.y;
#else
        vel_face_turned->x = vel_face->x * g.n.x + vel_face->y * g.n.y + vel_face->z * g.n.z;
        vel_face_turned->y = vel_face->x * g.m.x + vel_face->y * g.m.y + vel_face->z * g.m.z;
        vel_face_turned->z = vel_face->x * g.p.x + vel_face->y * g.p.y + vel_face->z * g.p.z;
#endif
    }

    // into the frame of the face; the pressure stays, the energy follows the new velocity
    HD void convert_state_to_local_frame(prim* st, POINT_TYPE vel_face) {
        double v2_old = st->v.x * st->v.x + st->v.y * st->v.y;
#ifdef dim_3D
        v2_old += st->v.z * st->v.z;
#endif
        double P = (gamma_eos - 1.0) * (st->E - 0.5 * st->rho * v2_old);
        if (P < 0.0) P = 0.0;

        st->v.x -= vel_face.x;
        st->v.y -= vel_face.y;
#ifdef dim_3D
        st->v.z -= vel_face.z;
#endif

        double v2_new = st->v.x * st->v.x + st->v.y * st->v.y;
#ifdef dim_3D
        v2_new += st->v.z * st->v.z;
#endif
        st->E = P / (gamma_eos - 1.0) + 0.5 * st->rho * v2_new;
    }

    // and the flux back into the lab frame
    HD void convert_flux_to_lab_frame(flux_t* flux, POINT_TYPE vel_face_turned) {
        double momx = flux->v.x;
        double momy = flux->v.y;

        flux->v.x += vel_face_turned.x * flux->rho;
        flux->v.y += vel_face_turned.y * flux->rho;

#ifdef dim_3D
        double momz = flux->v.z;
        flux->v.z += vel_face_turned.z * flux->rho;

        flux->E += momx * vel_face_turned.x + momy * vel_face_turned.y + momz * vel_face_turned.z +
                   0.5 * flux->rho *
                       (vel_face_turned.x * vel_face_turned.x + vel_face_turned.y * vel_face_turned.y +
                        vel_face_turned.z * vel_face_turned.z);
#else
        flux->E += momx * vel_face_turned.x + momy * vel_face_turned.y +
                   0.5 * flux->rho * (vel_face_turned.x * vel_face_turned.x + vel_face_turned.y * vel_face_turned.y);
#endif
    }
#endif

} // namespace hydro
