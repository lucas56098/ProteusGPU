#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../mpi/decomp.h"
#include "../mpi/halo.h"
#include "../profiler/profiler.h"
#include "finite_volume_solver.h"
#include "riemann.cu" // include directly for single translation unit compilation
#include <utility>

namespace hydro {

    // forward declarations
    HD void flux_update_for_cell(
        hsize_t, double, bool, double, const VMesh*, const primvars*, const gradients::PrimGradients*, primvars*);
    HD double dt_CFL_for_cell(hsize_t, double, const VMesh*, const primvars*);

#ifndef CPU_DEBUG
    // kernels
    GLOBAL void
    kernel_flux_update(double, int, double, const VMesh*, const primvars*, const gradients::PrimGradients*, primvars*);
    GLOBAL void
    kernel_copy_primvars(hsize_t, const double*, const POINT_TYPE*, const double*, double*, POINT_TYPE*, double*);
    GLOBAL void kernel_dt_CFL(double, hsize_t, const VMesh*, const primvars*, double*);
    GLOBAL void kernel_volume_correct(hsize_t, const double*, const double*, double*, double*);
#endif

    // ============================================================
    // Allocation and initialization
    // ============================================================

    // step-scratch primvars buffer + gradient buffer (reused every hydro step)
    static primvars*                 s_prim_new = nullptr;
    static gradients::PrimGradients* s_grads    = nullptr;

    // allocate primvars from IC data and copy into managed memory.
    // ext sizes the array for both MPI ghost slots and migration growth headroom.
    primvars* init(int n_hydro) {
        const int ext = proteus_mpi::alloc_per_cell_size(n_hydro);

        primvars* hydro_data = gpu_alloc<primvars>(1);
        hydro_data->rho      = gpu_alloc<double>(ext);
        hydro_data->v        = gpu_alloc<POINT_TYPE>(ext);
        hydro_data->E        = gpu_alloc<double>(ext);

        gpu_advise_gpu_preferred(hydro_data->rho, ext * sizeof(double));
        gpu_advise_gpu_preferred(hydro_data->v, ext * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(hydro_data->E, ext * sizeof(double));

        for (int i = 0; i < n_hydro; i++) {
            hydro_data->rho[i] = icData.rho[i];
            hydro_data->E[i]   = icData.energy[i];
            hydro_data->v[i].x = icData.vel[DIMENSION * i];
            hydro_data->v[i].y = icData.vel[DIMENSION * i + 1];
#ifdef dim_3D
            hydro_data->v[i].z = icData.vel[DIMENSION * i + 2];
#endif
        }

        const int n_hydro_global = logging::sum_global((int)n_hydro);
        logging::root() << "HYDRO: Initialized primitive variables for " << n_hydro_global << " particles" << std::endl;
        return hydro_data;
    }

    void free_prim(primvars** primvar) {
        gpu_free((*primvar)->rho);
        gpu_free((*primvar)->v);
        gpu_free((*primvar)->E);
        gpu_free(*primvar);
        *primvar = NULL;
    }

    // allocate the per-step scratch buffers used by hydro_step (s_prim_new, s_grads)
    void allocate_hydro_buffers(hsize_t n_hydro) {
        s_prim_new = gpu_alloc<primvars>(1);
        allocate_prim_buffer(n_hydro, s_prim_new);
        s_grads = gpu_alloc<gradients::PrimGradients>(1);
        gradients::allocate_grad(n_hydro, s_grads);
    }

    void free_hydro_buffers() {
        free_prim_buffer(s_prim_new);
        gradients::free_grad(s_grads);
        gpu_free(s_prim_new);
        gpu_free(s_grads);
        s_prim_new = nullptr;
        s_grads    = nullptr;
    }

    primvars* prim_new_buffer() {
        return s_prim_new;
    }

    // ============================================================
    // Main routines
    // ============================================================

    void hydro_step(double dt, VMesh* mesh, primvars* primvar) {

        primvars*                 prim_new = s_prim_new;
        gradients::PrimGradients* grads    = s_grads;

        // refresh MPI ghost primvars (stale after the previous step's prim_new↔primvar swap)
        proteus_mpi::halo_exchange_primvars(mesh, primvar);

        // initialize new state from old primitive variables
#ifndef CPU_DEBUG
        {
            int tpb    = _HYDRO_BLOCK_SIZE_;
            int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
            kernel_copy_primvars<<<blocks, tpb>>>(
                mesh->n_hydro, primvar->rho, primvar->v, primvar->E, prim_new->rho, prim_new->v, prim_new->E);
            GPU_LAUNCH_CHECK();
        }
#else
        gpu_memcpy(prim_new->rho, primvar->rho, mesh->n_hydro * sizeof(double));
        gpu_memcpy(prim_new->v, primvar->v, mesh->n_hydro * sizeof(POINT_TYPE));
        gpu_memcpy(prim_new->E, primvar->E, mesh->n_hydro * sizeof(double));
#endif

        // compute gradients from old state on old mesh
        gradients::compute_prim_gradients(mesh, primvar, grads);
        proteus_mpi::halo_exchange_gradients(mesh, grads);

#ifdef MOVING_MESH
        voronoi::compute_mesh_velocities(mesh, primvar, grads);
        // refresh MPI ghost v_mesh — the upcoming flux reads it at neighbor indices
        proteus_mpi::halo_exchange_vmesh(mesh);
#endif

        // first half update (no time extrapolation)
        apply_flux_update(0.5 * dt, 0.0, mesh, primvar, grads, prim_new);

#ifdef MOVING_MESH
        // store old volume
        gpu_memcpy(mesh->old_volumes, mesh->volumes, mesh->n_hydro * sizeof(double));

        // move mesh
        voronoi::move_mesh(mesh, dt, primvar, prim_new);

        // correct new primitive variables for volume change
#ifndef CPU_DEBUG
        {
            int tpb    = _HYDRO_BLOCK_SIZE_;
            int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
            Profiler::StartGPU("kernel_volume_correct");
            kernel_volume_correct<<<blocks, tpb>>>(
                mesh->n_hydro, mesh->old_volumes, mesh->volumes, prim_new->rho, prim_new->E);
            Profiler::EndGPU("kernel_volume_correct");
        }
#else
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            double volume_ratio = mesh->old_volumes[i] / mesh->volumes[i];
            prim_new->rho[i] *= volume_ratio;
            prim_new->E[i] *= volume_ratio;
        }
#endif

        // recompute gradients on moved mesh for second half
        gradients::compute_prim_gradients(mesh, primvar, grads);
        proteus_mpi::halo_exchange_gradients(mesh, grads);
#endif

        // second half update (with time extrapolation)
        apply_flux_update(0.5 * dt, dt, mesh, primvar, grads, prim_new);

        // swap primvar pointers
#ifndef CPU_DEBUG
        GPU_SYNC();
#endif
        {
            double* tmp_rho = primvar->rho;
            primvar->rho    = prim_new->rho;
            prim_new->rho   = tmp_rho;

            POINT_TYPE* tmp_v = primvar->v;
            primvar->v        = prim_new->v;
            prim_new->v       = tmp_v;

            double* tmp_E = primvar->E;
            primvar->E    = prim_new->E;
            prim_new->E   = tmp_E;
        }
    }

    // dispatch the per-cell flux update; dt_extrap > 0 enables MUSCL-Hancock time extrapolation
    void apply_flux_update(double                          dt_update,
                           double                          dt_extrap,
                           const VMesh*                    mesh,
                           const primvars*                 prim_old,
                           const gradients::PrimGradients* grads,
                           primvars*                       prim_new) {

        Profiler::StartTimer("HYDRO_STEP (par)");

#ifndef CPU_DEBUG
        int tpb                = _HYDRO_BLOCK_SIZE_;
        int blocks             = ((int)mesh->n_hydro + tpb - 1) / tpb;
        int do_time_extrap_int = (dt_extrap != 0.0) ? 1 : 0;
        Profiler::StartGPU("kernel_flux_update");
        kernel_flux_update<<<blocks, tpb>>>(dt_update, do_time_extrap_int, dt_extrap, mesh, prim_old, grads, prim_new);
        Profiler::EndGPU("kernel_flux_update");
#else
        const bool do_time_extrap = (dt_extrap != 0.0);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            flux_update_for_cell(i, dt_update, do_time_extrap, dt_extrap, mesh, prim_old, grads, prim_new);
        }
#endif

        Profiler::EndTimer("HYDRO_STEP (par)");
    }

    // global CFL timestep: min over all hydro cells
    double dt_CFL(double CFL, const VMesh* mesh, const primvars* primvar) {
        Profiler::StartTimer("CFL (par)");

        double min_dt = 1e100;

#ifndef CPU_DEBUG
        static double* d_min_dt = nullptr;
        if (!d_min_dt) d_min_dt = gpu_alloc<double>(1);

        *d_min_dt = 1e100;

        int tpb    = _HYDRO_BLOCK_SIZE_;
        int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
        Profiler::StartGPU("kernel_dt_CFL");
        kernel_dt_CFL<<<blocks, tpb>>>(CFL, mesh->n_hydro, mesh, primvar, d_min_dt);
        Profiler::EndGPU("kernel_dt_CFL");

        GPU_SYNC();
        min_dt = *d_min_dt;
#else
#ifdef USE_OPENMP
#pragma omp parallel for reduction(min : min_dt)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            double dt_i = dt_CFL_for_cell(i, CFL, mesh, primvar);
            if (dt_i < min_dt) { min_dt = dt_i; }
        }
#endif

        Profiler::EndTimer("CFL (par)");
        return min_dt;
    }

    // ============================================================
    // CUDA kernel wrappers
    // ============================================================
#ifndef CPU_DEBUG

    GLOBAL void __launch_bounds__(_HYDRO_BLOCK_SIZE_, 2) kernel_flux_update(double          dt_update,
                                                                            int             do_time_extrap_int,
                                                                            double          dt_extrap,
                                                                            const VMesh*    mesh,
                                                                            const primvars* prim_old,
                                                                            const gradients::PrimGradients* grads,
                                                                            primvars*                       prim_new) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= mesh->n_hydro) return;
        flux_update_for_cell(i, dt_update, (do_time_extrap_int != 0), dt_extrap, mesh, prim_old, grads, prim_new);
    }

    GLOBAL void kernel_copy_primvars(hsize_t           n_hydro,
                                     const double*     rho_src,
                                     const POINT_TYPE* v_src,
                                     const double*     E_src,
                                     double*           rho_dst,
                                     POINT_TYPE*       v_dst,
                                     double*           E_dst) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        rho_dst[i] = rho_src[i];
        v_dst[i]   = v_src[i];
        E_dst[i]   = E_src[i];
    }

    GLOBAL void
    kernel_dt_CFL(double CFL, hsize_t n_hydro, const VMesh* mesh, const primvars* primvar, double* d_min_dt) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;

        double dt_local = 1e100;
        if (i < n_hydro) { dt_local = dt_CFL_for_cell(i, CFL, mesh, primvar); }

        // warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_down_sync(0xFFFFFFFF, dt_local, offset);
            dt_local     = fmin(dt_local, other);
        }

        // lane 0 of each warp does atomicMin via bit reinterpretation
        if ((threadIdx.x & 31) == 0) {
            unsigned long long val = __double_as_longlong(dt_local);
            atomicMin((unsigned long long*)d_min_dt, val);
        }
    }

    GLOBAL void kernel_volume_correct(
        hsize_t n_hydro, const double* old_volumes, const double* new_volumes, double* rho, double* E) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        double ratio = old_volumes[i] / new_volumes[i];
        rho[i] *= ratio;
        E[i] *= ratio;
    }

#endif // !CPU_DEBUG

    // ============================================================
    // Per-cell work functions (called by kernels and CPU loops)
    // ============================================================

    // sum face fluxes around cell i and apply the conservative update to prim_new[i]
    HD void flux_update_for_cell(hsize_t                         i,
                                 double                          dt_update,
                                 bool                            do_time_extrap,
                                 double                          dt_extrap,
                                 const VMesh*                    mesh,
                                 const primvars*                 prim_old,
                                 const gradients::PrimGradients* grads,
                                 primvars*                       prim_new) {

        const hsize_t face_base = mesh->face_ptr[i];

        // own state and gradient
        prim                    state_i = get_state(i, prim_old);
        gradients::PrimGradient grad_i  = grads->load(i);

        prim total_flux;

        // accumulate flux contribution from each face
        for (hsize_t j = 0; j < mesh->face_counts[i]; j++) {
            int                     face_idx = face_base + j;
            hsize_t                 index_j  = mesh->neighbor_cell[face_idx];
            prim                    state_j  = get_state(index_j, prim_old);
            gradients::PrimGradient grad_j   = grads->load(index_j);

            // local face frame (n, m, p) along the seed-to-seed direction
            double3 delta = {wrap_periodic_delta(mesh->seeds[index_j].x - mesh->seeds[i].x),
                             wrap_periodic_delta(mesh->seeds[index_j].y - mesh->seeds[i].y),
                             wrap_periodic_delta(mesh->seeds[index_j].z - mesh->seeds[i].z)};
            geom    g     = compute_geom(delta);

#ifdef MOVING_MESH
            // face velocity (lab + face-frame) for the moving-mesh transformation
            POINT_TYPE vel_face, vel_face_turned;
            POINT_TYPE vm_i = mesh->v_mesh[i];
            POINT_TYPE vm_j = mesh->v_mesh[index_j];
            get_vel_face(i,
                         index_j,
                         vm_i,
                         vm_j,
                         &mesh->f_mid_local[face_idx * (DIMENSION - 1)],
                         mesh,
                         g,
                         &vel_face,
                         &vel_face_turned);
#endif

            // reconstruct left/right face states by spatial (and optionally temporal) extrapolation
            prim       state_l, state_r;
            POINT_TYPE dx = point_diff(mesh->seeds[index_j], mesh->seeds[i]);

            apply_spatial_extrapolation(state_i, grad_i, point_mul(0.5, dx), &state_l);
            apply_spatial_extrapolation(state_j, grad_j, point_mul(-0.5, dx), &state_r);

            if (do_time_extrap) {
                apply_time_extrapolation(state_i, grad_i, dt_extrap, &state_l);
                apply_time_extrapolation(state_j, grad_j, dt_extrap, &state_r);
            }

#ifdef MOVING_MESH
            // boost into the face-comoving frame so the Riemann solver sees a stationary face
            convert_state_to_local_frame(&state_l, vel_face);
            convert_state_to_local_frame(&state_r, vel_face);
#endif

            // floor density and pressure to keep the Riemann solver well-defined
            keep_state_physical(&state_l);
            keep_state_physical(&state_r);

            // rotate so the face normal aligns with x; solve 1D Riemann; rotate flux back
            rotate_to_face(&state_l, &g);
            rotate_to_face(&state_r, &g);

            flux_t flux_ij = riemann_hllc(state_l, state_r);

#ifdef MOVING_MESH
            // boost flux back to the lab frame
            convert_flux_to_lab_frame(&flux_ij, vel_face_turned);
#endif
            rotate_from_face(&flux_ij, &g);

            // accumulate area-weighted flux
            double face_area = mesh->face_area[face_idx];

            total_flux.rho += flux_ij.rho * face_area;
            total_flux.v.x += flux_ij.v.x * face_area;
            total_flux.v.y += flux_ij.v.y * face_area;
#ifdef dim_3D
            total_flux.v.z += flux_ij.v.z * face_area;
#endif
            total_flux.E += flux_ij.E * face_area;
        }

        // conservative update: state_new = state_old - (dt/V) * sum(F * A)
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

    // CFL timestep for cell i: CFL * R_cell / (sound_speed + signal_velocity)
    HD double dt_CFL_for_cell(hsize_t i, double CFL, const VMesh* mesh, const primvars* primvar) {
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

        return CFL * (R_i / (c_i + v_sig));
    }

    // ============================================================
    // helper functions
    // ============================================================

    // floor density and pressure to small positive values (numerical safety net)
    HD void keep_state_physical(prim* state) {
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

    // rotate velocity from lab axes into the face frame {n, m, p}
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

    // rotate velocity from the face frame back to lab axes
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

    // st_extrap = state + dx . gradient (linear reconstruction from cell to face)
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

    // st_extrap += dt * dW/dt (MUSCL-Hancock half-step time advance)
    HD void apply_time_extrapolation(prim state_i, gradients::PrimGradient grad_i, double dt_extrap, prim* st_extrap) {
        prim dWdt;
        gradients::time_gradient(state_i, grad_i, &dWdt);

        st_extrap->rho += dt_extrap * dWdt.rho;
        st_extrap->v.x += dt_extrap * dWdt.v.x;
        st_extrap->v.y += dt_extrap * dWdt.v.y;
#ifdef dim_3D
        st_extrap->v.z += dt_extrap * dWdt.v.z;
#endif
        st_extrap->E += dt_extrap * dWdt.E;
    }

#ifdef MOVING_MESH
    // face velocity for moving-mesh boost: midpoint of generator velocities + offset correction.
    // vel_face is in lab axes, vel_face_turned in the face frame {n, m, p}.
    HD void get_vel_face(hsize_t       i,
                         hsize_t       index_j,
                         POINT_TYPE    v_mesh_i,
                         POINT_TYPE    v_mesh_j,
                         const double* f_mid_local,
                         const VMesh*  mesh,
                         geom          g,
                         POINT_TYPE*   vel_face,
                         POINT_TYPE*   vel_face_turned) {

        double facv;

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

        // limiter for highly distorted cells
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

    // boost the primitive state into the face-comoving frame (subtract vel_face from velocity)
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

    // boost the flux back from the face frame to the lab frame (add advected mass/momentum/energy)
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
#endif // MOVING_MESH

} // namespace hydro
