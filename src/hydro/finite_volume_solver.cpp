#include "finite_volume_solver.h"
#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../profiler/profiler.h"
#include "riemann.h"
#include <cstring>
#include <utility>

namespace hydro {

    // init hydrostruct from IC data
    primvars* init(int n_hydro) {

        // allocate prim struct
        primvars* hydro_data = (primvars*)malloc(sizeof(primvars));
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

    // persistent buffers for hydro_step
    static primvars                 s_prim_new = {};
    static gradients::PrimGradients s_grads    = {};

    void allocate_hydro_buffers(hsize_t n_hydro) {
        allocate_prim_buffer(n_hydro, &s_prim_new);
        gradients::allocate_grad(n_hydro, &s_grads);
    }

    void free_hydro_buffers() {
        free_prim_buffer(&s_prim_new);
        gradients::free_grad(&s_grads);
    }

    // main hydro routine (RK2 step including mesh movement and updated states)
    void hydro_step(double dt, VMesh* mesh, primvars* primvar) {

        primvars&                 prim_new = s_prim_new;
        gradients::PrimGradients& grads    = s_grads;

        // initialize new state from old primitive variables
        std::memcpy(prim_new.rho, primvar->rho, mesh->n_hydro * sizeof(double));
        std::memcpy(prim_new.v, primvar->v, mesh->n_hydro * sizeof(POINT_TYPE));
        std::memcpy(prim_new.E, primvar->E, mesh->n_hydro * sizeof(double));

        // compute gradients from old state on old mesh
        gradients::compute_prim_gradients(mesh, primvar, &grads);

#ifdef MOVING_MESH
        voronoi::compute_mesh_velocities(mesh, primvar, &grads);
#endif

        // first half update (no time extrapolation)
#ifdef MOVING_MESH
        apply_flux_update(0.5 * dt, 0.0, mesh, primvar, &grads, mesh->v_mesh, &prim_new);
#else
        apply_flux_update(0.5 * dt, 0.0, mesh, primvar, &grads, nullptr, &prim_new);
#endif

#ifdef MOVING_MESH
        // store old volume
        std::memcpy(mesh->old_volumes, mesh->volumes, mesh->n_hydro * sizeof(double));

        // move mesh
        voronoi::move_mesh(mesh, dt);

        // correct new primitive variables for volume change
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            double volume_ratio = mesh->old_volumes[i] / mesh->volumes[i];
            prim_new.rho[i] *= volume_ratio;
            prim_new.E[i] *= volume_ratio;
        }

        // compute gradients for second half
        gradients::compute_prim_gradients(mesh, primvar, &grads);
#endif

        // second half update (full dt time extrapolation)
#ifdef MOVING_MESH
        apply_flux_update(0.5 * dt, dt, mesh, primvar, &grads, mesh->v_mesh, &prim_new);
#else
        apply_flux_update(0.5 * dt, dt, mesh, primvar, &grads, nullptr, &prim_new);
#endif

        // swap primvar old <-> new
        std::swap(primvar->rho, prim_new.rho);
        std::swap(primvar->v, prim_new.v);
        std::swap(primvar->E, prim_new.E);
    }

    // apply one part of RK2 flux update (either with dt_extrap = 0 or dt)
    void apply_flux_update(double                          dt_update,
                           double                          dt_extrap,
                           const VMesh*                    mesh,
                           const primvars*                 prim_old,
                           const gradients::PrimGradients* grads,
                           const POINT_TYPE*               v_mesh,
                           primvars*                       prim_new) {

        PROFILE_START("HYDRO_STEP (par)");
        const bool do_time_extrap = (dt_extrap != 0.0);

// loop over all active cells to calc new primvars
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(_OMP_HYDRO_THREADS_)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {

            const hsize_t face_base = mesh->face_ptr[i];

            // get state of cell i
            prim                    state_i = get_state(i, mesh, prim_old);
            gradients::PrimGradient grad_i  = grads->load(i);

            prim total_flux;

            // calculate total_flux by summing over edge flux * edge_length
            for (hsize_t j = 0; j < mesh->face_counts[i]; j++) {

                // get state of cell j
                int                     face_idx = face_base + j;
                hsize_t                 index_j  = mesh->neighbor_cell[face_idx];
                prim                    state_j  = get_state(index_j, mesh, prim_old);
                gradients::PrimGradient grad_j   = grads->load(hydro_index(index_j, mesh));

                // get normal and geometry
                double3 delta = {wrap_periodic_delta(mesh->seeds[index_j].x - mesh->seeds[i].x),
                                 wrap_periodic_delta(mesh->seeds[index_j].y - mesh->seeds[i].y),
                                 wrap_periodic_delta(mesh->seeds[index_j].z - mesh->seeds[i].z)};
                geom    g     = compute_geom(delta);

#ifdef MOVING_MESH
                // compute face velocity
                POINT_TYPE vel_face, vel_face_turned;
                POINT_TYPE vm_i = v_mesh[i];
                POINT_TYPE vm_j = v_mesh[hydro_index(index_j, mesh)];
                get_vel_face(i,
                             index_j,
                             vm_i,
                             vm_j,
                             &mesh->f_mid_local[face_idx * (DIMENSION - 1)],
                             mesh,
                             g,
                             &vel_face,
                             &vel_face_turned);
#else
                (void)v_mesh;
#endif

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
                    apply_time_extrapolation(state_j, grad_j, dt_extrap, &state_r);
                }

#ifdef MOVING_MESH
                // transform into frame of face
                convert_state_to_local_frame(&state_l, vel_face);
                convert_state_to_local_frame(&state_r, vel_face);
#endif

                // ensure rho > rho_min, P > P_min
                keep_state_physical(&state_l);
                keep_state_physical(&state_r);

                // rotate state into face direction
                rotate_to_face(&state_l, &g);
                rotate_to_face(&state_r, &g);

                // calc flux using riemann solver
#ifdef RIEMANN_HLL
                flux_t flux_ij = riemann_hll(state_l, state_r);
#else
                flux_t flux_ij = riemann_hllc(state_l, state_r);
#endif

#ifdef MOVING_MESH
                // transform flux back to lab frame
                convert_flux_to_lab_frame(&flux_ij, vel_face_turned);
#endif
                // rotate flux back to lab frame
                rotate_from_face(&flux_ij, &g);

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

            // finite volume update of primvar
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
    void apply_spatial_extrapolation(const prim                    state,
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

    // apply primitive time extrapolation: W -> W + dt_extrap * dW/dt(cell_idx)
    void apply_time_extrapolation(prim state_i, gradients::PrimGradient grad_i, double dt_extrap, prim* st_extrap) {

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

#ifdef MOVING_MESH
    void get_vel_face(hsize_t          i,
                      hsize_t          index_j,
                      POINT_TYPE       v_mesh_i,
                      POINT_TYPE       v_mesh_j,
                      const compact_t* f_mid_local,
                      const VMesh*     mesh,
                      geom             g,
                      POINT_TYPE*      vel_face,
                      POINT_TYPE*      vel_face_turned) {

        double facv;

        // rough motion of mid-point of edge
        // compute distance between generators (nn = |r_ij|)
        double nnx = wrap_periodic_delta(mesh->seeds[index_j].x - mesh->seeds[i].x);
        double nny = wrap_periodic_delta(mesh->seeds[index_j].y - mesh->seeds[i].y);
#ifdef dim_3D
        double nnz = wrap_periodic_delta(mesh->seeds[index_j].z - mesh->seeds[i].z);
        double nn  = sqrt(nnx * nnx + nny * nny + nnz * nnz);
#else
        double nn = sqrt(nnx * nnx + nny * nny);
#endif

        vel_face->x = 0.5 * (v_mesh_i.x + v_mesh_j.x);
        vel_face->y = 0.5 * (v_mesh_i.y + v_mesh_j.y);

        // reconstruct offset from seed midpoint using local tangent-space coords
#ifdef dim_2D
        double alpha = (double)f_mid_local[0];
        double cx    = alpha * g.m.x;
        double cy    = alpha * g.m.y;
#else
        vel_face->z  = 0.5 * (v_mesh_i.z + v_mesh_j.z);
        double alpha = (double)f_mid_local[0];
        double beta  = (double)f_mid_local[1];
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

        // put in a limiter for highly distorted cells
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

    void convert_state_to_local_frame(prim* st, POINT_TYPE vel_face) {

        // compute pressure P from energy (P is conserved during transformation)
        double v2_old = st->v.x * st->v.x + st->v.y * st->v.y;
#ifdef dim_3D
        v2_old += st->v.z * st->v.z;
#endif
        double P = (gamma_eos - 1.0) * (st->E - 0.5 * st->rho * v2_old);
        if (P < 0.0) P = 0.0;

        // transform velocities
        st->v.x -= vel_face.x;
        st->v.y -= vel_face.y;
#ifdef dim_3D
        st->v.z -= vel_face.z;
#endif

        // from new vel and pressure P compute transformed energy E
        double v2_new = st->v.x * st->v.x + st->v.y * st->v.y;
#ifdef dim_3D
        v2_new += st->v.z * st->v.z;
#endif
        st->E = P / (gamma_eos - 1.0) + 0.5 * st->rho * v2_new;
    }

    void convert_flux_to_lab_frame(flux_t* flux, POINT_TYPE vel_face_turned) {
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
        double emin = ekin + p_floor / (gamma_eos - 1.0);
        if (state->E < emin) { state->E = emin; }
    }

    void rotate_to_face(prim* state, geom* g) {
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

    void rotate_from_face(prim* state, geom* g) {
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
            double c_i = sqrt(gamma_eos * P / state_i.rho);

#ifdef dim_2D
            double R_i = sqrt(mesh->volumes[i] / M_PI);
#else
            double R_i = cbrt(3.0 * mesh->volumes[i] / (4.0 * M_PI));
#endif

#ifdef MOVING_MESH
            // moving mesh: use relative velocity between fluid and mesh point
            double dvx = state_i.v.x - mesh->v_mesh[i].x;
            double dvy = state_i.v.y - mesh->v_mesh[i].y;
#ifdef dim_3D
            double dvz   = state_i.v.z - mesh->v_mesh[i].z;
            double v_sig = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
#else
            double v_sig = sqrt(dvx * dvx + dvy * dvy);
#endif
#else
            // static mesh: use absolute fluid velocity
#ifdef dim_2D
            double v_sig = sqrt(state_i.v.x * state_i.v.x + state_i.v.y * state_i.v.y);
#else
            double v_sig = sqrt(state_i.v.x * state_i.v.x + state_i.v.y * state_i.v.y + state_i.v.z * state_i.v.z);
#endif
#endif

            double dt_i = CFL * (R_i / (c_i + v_sig));

            if (dt_i < min_dt) { min_dt = dt_i; }
        }

        PROFILE_END("CFL (par)");
        return min_dt;
    }

} // namespace hydro