/* hydro handling: init/free; hydro stepping; flux calc; CFL timestep */
#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../mpi/decomp.h"
#include "../mpi/halo.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "../astro/agn.h"
#include "finite_volume_solver.h"
#include "riemann.cu"
#include <utility>

namespace hydro {

    // forward declarations
    HD void flux_update_for_cell(
        hsize_t, double, bool, double, const VMesh*, const primvars*, const gradients::PrimGradients*, primvars*);
#ifdef AGN_ENABLED
    HD double dt_CFL_for_cell(hsize_t, double, const VMesh*, const primvars*, bool, const astro::AgnParams&);
#else
    HD double   dt_CFL_for_cell(hsize_t, double, const VMesh*, const primvars*);
#endif
    static void check_unphysical_state(VMesh*, const primvars*);
    static void reset_prim_new(VMesh* mesh, primvars* primvar, primvars* prim_new);
    static void swap_primvars(primvars* primvar, primvars* prim_new);

#ifndef CPU_DEBUG
    // kernels
    GLOBAL void
    kernel_flux_update(double, int, double, const VMesh*, const primvars*, const gradients::PrimGradients*, primvars*);
    GLOBAL void
    kernel_copy_primvars(hsize_t, const double*, const POINT_TYPE*, const double*, double*, POINT_TYPE*, double*);
#ifdef AGN_ENABLED
    GLOBAL void kernel_dt_CFL(double, hsize_t, const VMesh*, const primvars*, double*, bool, astro::AgnParams);
#else
    GLOBAL void kernel_dt_CFL(double, hsize_t, const VMesh*, const primvars*, double*);
#endif
    GLOBAL void kernel_check_unphysical(hsize_t, const primvars*, int*);
#endif

    // ============================================================
    // Allocation and initialization
    // ============================================================

    void init_hydro() {
        const int n_hydro = (int)sim.n_hydro;

        // allocate primvar
        sim.primvar = gpu_alloc<primvars>(1);
        allocate_prim_buffer(sim.n_hydro, sim.primvar, /*with_ghosts=*/true);

        // fill primvar with icData
        for (int i = 0; i < n_hydro; i++) {
            sim.primvar->rho[i] = icData.rho[i];
            sim.primvar->E[i]   = icData.energy[i];
            sim.primvar->v[i].x = icData.vel[DIMENSION * i];
            sim.primvar->v[i].y = icData.vel[DIMENSION * i + 1];
#ifdef dim_3D
            sim.primvar->v[i].z = icData.vel[DIMENSION * i + 2];
#endif
        }

        // allocate prim_new
        sim.prim_new = gpu_alloc<primvars>(1);
        allocate_prim_buffer(sim.n_hydro, sim.prim_new, /*with_ghosts=*/false);

        // allocate gradients
        sim.grads = gpu_alloc<gradients::PrimGradients>(1);
        gradients::allocate_grad(sim.n_hydro, sim.grads);

        // allocate dt
        sim.dt = gpu_alloc<double>(1);

        const int n_hydro_global = logging::sum_global(n_hydro);
        logging::root() << "HYDRO: Initialized hydro for " << n_hydro_global << " particles" << std::endl;
    }

    void free_hydro() {
        // free arrays
        free_prim_buffer(sim.primvar);
        free_prim_buffer(sim.prim_new);
        gradients::free_grad(sim.grads);

        // free structures
        gpu_free(sim.primvar);
        gpu_free(sim.prim_new);
        gpu_free(sim.grads);
        gpu_free(sim.dt);

        // set to nullptr
        sim.primvar  = nullptr;
        sim.prim_new = nullptr;
        sim.grads    = nullptr;
        sim.dt       = nullptr;
    }

    // ============================================================
    // Main routines
    // ============================================================

    void hydro_step(double dt, VMesh* mesh, primvars* primvar) {

        primvars*                 prim_new = sim.prim_new;
        gradients::PrimGradients* grads    = sim.grads;

        // MPI exchange primvars
        proteus_mpi::halo_exchange_primvars(mesh, primvar);

        // set prim_new equal to primvar
        reset_prim_new(mesh, primvar, prim_new);

        // compute gradients from old state on old mesh
        gradients::compute_prim_gradients(mesh, primvar, grads);
        proteus_mpi::halo_exchange_gradients(mesh, grads);

#ifdef MOVING_MESH
        // compute v_mesh
        voronoi::compute_mesh_velocities(mesh, primvar, grads);
        proteus_mpi::halo_exchange_v_mesh(mesh);
#endif

        // first half update (no time extrapolation)
        apply_flux_update(0.5 * dt, 0.0, mesh, primvar, grads, prim_new);
        logging::root() << "HYDRO: Computed " << logging::sum_global((int)mesh->num_faces) << " fluxes (1/2)"
                        << std::endl;

#ifdef MOVING_MESH

        // move mesh
        voronoi::move_mesh(mesh, dt, primvar, prim_new);

        // recompute gradients on moved mesh for second half
        gradients::compute_prim_gradients(mesh, primvar, grads);
        proteus_mpi::halo_exchange_gradients(mesh, grads);
#endif

        // second half update (with time extrapolation)
        apply_flux_update(0.5 * dt, dt, mesh, primvar, grads, prim_new);
        logging::root() << "HYDRO: Computed " << logging::sum_global((int)mesh->num_faces) << " fluxes (2/2)"
                        << std::endl;

        // set prim_new as the new primvar
        swap_primvars(primvar, prim_new);

        check_unphysical_state(mesh, primvar);
    }

    // per cell flux update (used in both RK2 steps)
    void apply_flux_update(double                          dt_update,
                           double                          dt_extrap,
                           const VMesh*                    mesh,
                           const primvars*                 prim_old,
                           const gradients::PrimGradients* grads,
                           primvars*                       prim_new) {

        PROFILE("FLUX");

#ifndef CPU_DEBUG
        int tpb                = _HYDRO_BLOCK_SIZE_;
        int blocks             = ((int)mesh->n_hydro + tpb - 1) / tpb;
        int do_time_extrap_int = (dt_extrap != 0.0) ? 1 : 0;
        {
            PROFILE_KERNEL("FLUX_KERNEL");
            kernel_flux_update<<<blocks, tpb>>>(
                dt_update, do_time_extrap_int, dt_extrap, mesh, prim_old, grads, prim_new);
            GPU_SYNC();
        }
#else
        const bool do_time_extrap = (dt_extrap != 0.0);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            flux_update_for_cell(i, dt_update, do_time_extrap, dt_extrap, mesh, prim_old, grads, prim_new);
        }
#endif
    }

    double calc_timestep(double CFL, const VMesh* mesh, const primvars* primvar) {

        {
            // per rank CFL timestep (min over all cells)
            PROFILE("CFL");

            double* min_dt = sim.dt;
            *min_dt        = 1e100;
#ifdef AGN_ENABLED
            // hoisted out of the per-cell loop: these read agn.cu statics on the host, and
            // calling them per cell inside the OpenMP loop is a silent per-cell cost
            const bool             agn_firing = astro::agn_is_firing();
            const astro::AgnParams p_agn      = astro::agn_params();
#endif

#ifndef CPU_DEBUG
            int tpb    = _HYDRO_BLOCK_SIZE_;
            int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
            {
                PROFILE_KERNEL("DT_CFL");
#ifdef AGN_ENABLED
                kernel_dt_CFL<<<blocks, tpb>>>(CFL, mesh->n_hydro, mesh, primvar, min_dt, agn_firing, p_agn);
#else
                kernel_dt_CFL<<<blocks, tpb>>>(CFL, mesh->n_hydro, mesh, primvar, min_dt);
#endif
            }
            GPU_SYNC();
#else
#ifdef USE_OPENMP
#pragma omp parallel for reduction(min : min_dt[0])
#endif
            for (hsize_t i = 0; i < mesh->n_hydro; i++) {
#ifdef AGN_ENABLED
                double dt_i = dt_CFL_for_cell(i, CFL, mesh, primvar, agn_firing, p_agn);
#else
                double dt_i = dt_CFL_for_cell(i, CFL, mesh, primvar);
#endif
                if (dt_i < *min_dt) { *min_dt = dt_i; }
            }
#endif
        }

        // global all rank minimum dt
        proteus_mpi::halo_dt_allreduce(sim.dt);

        // limit to snapshot/end of simulation
        if (sim.t_sim + *sim.dt > sim.t_nextoutput) { *sim.dt = sim.t_nextoutput - sim.t_sim; }
        if (sim.t_sim + *sim.dt > sim.t_end) { *sim.dt = sim.t_end - sim.t_sim; }

        return *sim.dt;
    }

    // ============================================================
    // Host functions
    // ============================================================

    // set prim_new equal to prim
    static void reset_prim_new(VMesh* mesh, primvars* primvar, primvars* prim_new) {
#ifndef CPU_DEBUG
        {
            PROFILE_KERNEL("COPY_PRIMVAR");
            int tpb    = _HYDRO_BLOCK_SIZE_;
            int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
            kernel_copy_primvars<<<blocks, tpb>>>(
                mesh->n_hydro, primvar->rho, primvar->v, primvar->E, prim_new->rho, prim_new->v, prim_new->E);
            GPU_SYNC();
        }
#else
        gpu_memcpy(prim_new->rho, primvar->rho, mesh->n_hydro * sizeof(double));
        gpu_memcpy(prim_new->v, primvar->v, mesh->n_hydro * sizeof(POINT_TYPE));
        gpu_memcpy(prim_new->E, primvar->E, mesh->n_hydro * sizeof(double));
#endif
    }

    // swap the rho / v / E SoA pointers between primvar and prim_new so primvar holds
    // the newly-computed state for the next step
    static void swap_primvars(primvars* primvar, primvars* prim_new) {
#ifndef CPU_DEBUG
        GPU_SYNC(); // ensure all kernel writes to prim_new have landed before the swap
#endif
        std::swap(primvar->rho, prim_new->rho);
        std::swap(primvar->v, prim_new->v);
        std::swap(primvar->E, prim_new->E);
    }

    // scan primvar for unphysical values
    static void check_unphysical_state(VMesh* mesh, const primvars* primvar) {
        PROFILE("UNPHYS_CHECK");
        int counts[UNPHYS_N] = {0, 0, 0};
#ifndef CPU_DEBUG
        static int* d_counts = nullptr;
        if (!d_counts) d_counts = gpu_alloc<int>(UNPHYS_N);
        for (int k = 0; k < UNPHYS_N; k++)
            d_counts[k] = 0;
        const int tpb    = _HYDRO_BLOCK_SIZE_;
        const int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("UNPHYS_KERNEL");
            kernel_check_unphysical<<<blocks, tpb>>>(mesh->n_hydro, primvar, d_counts);
        }
        GPU_SYNC();
        for (int k = 0; k < UNPHYS_N; k++)
            counts[k] = d_counts[k];
#else
        int n_rho_bad = 0, n_E_bad = 0, n_nan = 0;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+ : n_rho_bad, n_E_bad, n_nan)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            const double rho = primvar->rho[i];
            const double E   = primvar->E[i];
            if (rho <= 0.0) n_rho_bad++; // NaN <= 0 is false, so NaN doesn't double-count here
            if (E <= 0.0) n_E_bad++;
            const bool has_nan = std::isnan(rho) || std::isnan(E) || std::isnan(primvar->v[i].x) ||
                                 std::isnan(primvar->v[i].y)
#ifdef dim_3D
                                 || std::isnan(primvar->v[i].z)
#endif
                ;
            if (has_nan) n_nan++;
        }
        counts[UNPHYS_RHO] = n_rho_bad;
        counts[UNPHYS_E]   = n_E_bad;
        counts[UNPHYS_NAN] = n_nan;
#endif
        // global reduce; abort the run if anything fired
        const int rho_bad = logging::sum_global(counts[UNPHYS_RHO]);
        const int E_bad   = logging::sum_global(counts[UNPHYS_E]);
        const int nan_bad = logging::sum_global(counts[UNPHYS_NAN]);
        if (rho_bad == 0 && E_bad == 0 && nan_bad == 0) return;

        if (rho_bad > 0) logging::root() << "HYDRO: WARNING: " << rho_bad << " cells with rho<=0" << std::endl;
        if (E_bad > 0) logging::root() << "HYDRO: WARNING: " << E_bad << " cells with E<=0" << std::endl;
        if (nan_bad > 0) logging::root() << "HYDRO: WARNING: " << nan_bad << " cells with NaN" << std::endl;
        proteus_mpi::exit_failure("HYDRO: ABORT: unphysical state detected — terminating run.\n");
    }

    // ============================================================
    // CUDA kernel wrappers
    // ============================================================
#ifndef CPU_DEBUG

    // calls flux update for cell
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

    // copy primvars from one to another
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

    // for each cell calc CFL and then do warp-level reduction and lane 0 atomicMin
    GLOBAL void
#ifdef AGN_ENABLED
    kernel_dt_CFL(double CFL, hsize_t n_hydro, const VMesh* mesh, const primvars* primvar, double* d_min_dt,
                  bool agn_firing, astro::AgnParams p_agn) {
#else
    kernel_dt_CFL(double CFL, hsize_t n_hydro, const VMesh* mesh, const primvars* primvar, double* d_min_dt) {
#endif
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;

        double dt_local = 1e100;
#ifdef AGN_ENABLED
        if (i < n_hydro) { dt_local = dt_CFL_for_cell(i, CFL, mesh, primvar, agn_firing, p_agn); }
#else
        if (i < n_hydro) { dt_local = dt_CFL_for_cell(i, CFL, mesh, primvar); }
#endif

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

    // per-cell sanity check
    // counters: [n_rho<=0, n_E<=0, n_NaN]
    GLOBAL void kernel_check_unphysical(hsize_t n_hydro, const primvars* p, int* counters) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;

        const double rho = p->rho[i];
        const double E   = p->E[i];

        // sanity checks
        if (rho <= 0.0) portable_atomicAdd(&counters[UNPHYS_RHO], 1); // NaN<=0 is false → no double-count
        if (E <= 0.0) portable_atomicAdd(&counters[UNPHYS_E], 1);
        if (isnan(rho) || isnan(E) || isnan(p->v[i].x) || isnan(p->v[i].y)
#ifdef dim_3D
            || isnan(p->v[i].z)
#endif
        ) {
            portable_atomicAdd(&counters[UNPHYS_NAN], 1);
        }
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

        prim      total_flux;
        const int n_hydro_int = (int)mesh->n_hydro;

        // accumulate flux contribution from each face. face_idx must be hsize_t — at
        // 2e8 cells/rank with _FACE_CAPACITY_MULT_=17, max_faces ~ 5e9 and an int
        // would silently wrap, reading garbage from neighbor_cell / face_area.
        for (hsize_t j = 0; j < mesh->face_counts[i]; j++) {
            hsize_t                 face_idx = face_base + j;
            int                     index_j  = mesh->neighbor_cell[face_idx];
            prim                    state_j  = get_state_at(index_j, n_hydro_int, prim_old);
            gradients::PrimGradient grad_j   = grads->load_at(index_j, n_hydro_int);
            double3                 seed_j   = get_seed_at(index_j, n_hydro_int, mesh);

            // local face frame (n, m, p) along the seed-to-seed direction
            double3 delta = {wrap_periodic_delta(seed_j.x - mesh->seeds[i].x),
                             wrap_periodic_delta(seed_j.y - mesh->seeds[i].y),
                             wrap_periodic_delta(seed_j.z - mesh->seeds[i].z)};
            geom    g     = compute_geom(delta);

#ifdef MOVING_MESH
            // face velocity (lab + face-frame) for the moving-mesh transformation
            POINT_TYPE vel_face, vel_face_turned;
            POINT_TYPE vm_i = mesh->v_mesh[i];
            POINT_TYPE vm_j = get_vmesh_at(index_j, n_hydro_int, mesh);
            get_vel_face(i,
                         (hsize_t)index_j,
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
            POINT_TYPE dx = point_diff_periodic(seed_j, mesh->seeds[i]);

            apply_spatial_extrapolation(state_i, grad_i, point_mul(0.5, dx), &state_l);
            apply_spatial_extrapolation(state_j, grad_j, point_mul(-0.5, dx), &state_r);

            if (do_time_extrap) {
                apply_time_extrapolation(state_i, grad_i, dt_extrap, &state_l);
                apply_time_extrapolation(state_j, grad_j, dt_extrap, &state_r);
            }

#ifdef MOVING_MESH
            // boost into the face-comoving frame
            convert_state_to_local_frame(&state_l, vel_face);
            convert_state_to_local_frame(&state_r, vel_face);
#endif

            // floor density and pressure
            keep_state_physical(&state_l, mesh->min_egy_spec);
            keep_state_physical(&state_r, mesh->min_egy_spec);

            // rotate into face frame
            rotate_to_face(&state_l, &g);
            rotate_to_face(&state_r, &g);

            // solve flux
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
        double           frac           = dt_update / mesh->volumes[i];
        double           rho_old        = prim_new->rho[i];
        double           rho_new        = rho_old - frac * total_flux.rho;
        constexpr double RHO_FLOOR_CELL = 1e-13;

        if (mesh->min_egy_spec > 0.0 && rho_new < RHO_FLOOR_CELL) {
            // Density-floor "soft landing". The cell would go negative-mass under a normal
            // momentum-conserving update, and conserving momentum against the floored rho
            // amplifies v by ~1/floor_ratio, producing cells that NaN the flux calc
            // downstream. Instead floor rho, freeze v at its pre-flux value (drop the
            // momentum flux this step), and reset E onto the temperature wall for the new
            // rho. Locally non-conservative, but the mass was already essentially gone.
            // The floor is absolute, not relative: rho_new = 1e-10 * rho_old would ratchet
            // 10x smaller on every firing and make 1/rho in time_gradient explode.
            rho_new          = RHO_FLOOR_CELL;
            prim_new->rho[i] = rho_new;
            // prim_new->v[i] already holds the pre-flux value; leave it untouched
            double v2 = prim_new->v[i].x * prim_new->v[i].x + prim_new->v[i].y * prim_new->v[i].y;
#ifdef dim_3D
            v2 += prim_new->v[i].z * prim_new->v[i].z;
#endif
            prim_new->E[i] = 0.5 * rho_new * v2 + rho_new * mesh->min_egy_spec;
        } else {
            double rho_inv = 1.0 / rho_new;

            prim_new->rho[i] = rho_new;
            prim_new->v[i].x = (rho_old * prim_new->v[i].x - frac * total_flux.v.x) * rho_inv;
            prim_new->v[i].y = (rho_old * prim_new->v[i].y - frac * total_flux.v.y) * rho_inv;
#ifdef dim_3D
            prim_new->v[i].z = (rho_old * prim_new->v[i].z - frac * total_flux.v.z) * rho_inv;
#endif
            prim_new->E[i] -= frac * total_flux.E;

            // temperature floor: keep E above kinetic + e_int(T_floor) = rho * min_egy_spec
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

    // CFL timestep for cell i
#ifdef AGN_ENABLED
    HD double dt_CFL_for_cell(hsize_t i, double CFL, const VMesh* mesh, const primvars* primvar, bool agn_firing,
                              const astro::AgnParams& p_agn) {
#else
    HD double dt_CFL_for_cell(hsize_t i, double CFL, const VMesh* mesh, const primvars* primvar) {
#endif

        // get state
        prim state_i;
        state_i.rho = primvar->rho[i];
        state_i.E   = primvar->E[i];
        state_i.v.x = primvar->v[i].x;
        state_i.v.y = primvar->v[i].y;
#ifdef dim_3D
        state_i.v.z = primvar->v[i].z;
#endif

        // sound speed
        double P   = get_P_ideal_gas(&state_i);
        // guard the sqrt: a negative P (unphysical cell) would make c_i NaN, and NaN survives the
        // atomicMin/fmin reduction to poison the GLOBAL dt
        double c_i = (state_i.rho > 0.0 && P > 0.0) ? sqrt(gamma_eos * P / state_i.rho) : 0.0;
#ifdef AGN_ENABLED
        // cells inside the thermal deposit sphere can be pushed to T_max by the next injection,
        // so bound dt by that sound speed rather than the current one
        if (agn_firing) {
            const double3 sd = mesh->seeds[i];
            const double  dx = sd.x - p_agn.cx, dy = sd.y - p_agn.cy;
#ifdef dim_3D
            const double dz = sd.z - p_agn.cz;
            const double r2 = dx * dx + dy * dy + dz * dz;
#else
            const double r2 = dx * dx + dy * dy;
#endif
            if (r2 < p_agn.r_T2) {
                const double c_ceiling = sqrt(p_agn.cs2_max);
                if (c_ceiling > c_i) c_i = c_ceiling;
            }
        }
#endif

        // radius
#ifdef dim_2D
        double R_i = sqrt(mesh->volumes[i] / M_PI);
#else
        double R_i = cbrt(3.0 * mesh->volumes[i] / (4.0 * M_PI));
#endif
        // fluid speed
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
#if defined(AGN_ENABLED) && defined(AGN_KINETIC)
        // jet launch zones can push |v| up to v_cap in one step
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
            if (perp2 < p_agn.r_jet2 && ady > p_agn.L_jet && ady < p_agn.L_jet + p_agn.h_jet) {
                if (p_agn.v_cap > v_sig) v_sig = p_agn.v_cap;
            }
        }
#endif

        // calc CFL dt
        return CFL * (R_i / (c_i + v_sig));
    }

    // ============================================================
    // helper functions
    // ============================================================

    // floor density and pressure to small positive values
    HD void keep_state_physical(prim* state, double min_egy_spec) {
        const double rho_floor = 1e-12;
        const double p_floor   = 1e-12;

        if (state->rho < rho_floor) { state->rho = rho_floor; }

        double v2 = state->v.x * state->v.x + state->v.y * state->v.y;
#ifdef dim_3D
        v2 += state->v.z * state->v.z;
#endif
        // keep ekin a separate temporary: folding it into the sum below lets the compiler
        // contract it to an FMA, which changes the rounding of every floor-free run
        double ekin      = 0.5 * state->rho * v2;
        double e_int_min = (min_egy_spec > 0.0) ? state->rho * min_egy_spec : p_floor / (gamma_eos - 1.0);
        double emin      = ekin + e_int_min;
        if (state->E < emin) { state->E = emin; }
    }

    // rotate velocity from lab into the face frame
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

    // rotate velocity from the face frame back to lab frame
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

    // st_extrap = state + dx * gradient
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

    // st_extrap += dt * dW/dt
    // st_extrap += beta * dt * dW/dt, where beta in [0, 1] is the largest scale that keeps
    // the reconstructed face state above the rho and pressure floors. Analogous to
    // gradients::pressure_safe_scale for the spatial term (per-cell), but applied here
    // per-face on top of the already spatially-limited state. Without it, dt * dWdt is
    // unbounded and can drive face states to arbitrary rho/E, producing pathological
    // Riemann inputs at sharp interfaces (e.g. cells adjacent to a floored neighbour, or
    // cells whose neighbour topology just changed after a symmetry-cascade rebuild).
    HD void apply_time_extrapolation(prim state_i, gradients::PrimGradient grad_i, double dt_extrap, prim* st_extrap) {
        prim dWdt;
        gradients::time_gradient(state_i, grad_i, &dWdt);

        constexpr double RHO_FLOOR_FACE = 1e-12;
        constexpr double P_FLOOR_FACE   = 1e-12;
        double           beta           = 1.0;

        // rho constraint: linear in beta. Only binds when dWdt.rho < 0.
        if (dWdt.rho < 0.0) {
            const double denom = -dt_extrap * dWdt.rho;
            if (denom > 0.0) {
                const double beta_rho = (st_extrap->rho - RHO_FLOOR_FACE) / denom;
                if (beta_rho < beta) beta = fmax(0.0, beta_rho);
            }
        }

        // pressure constraint: P = (gamma-1)*(E - 0.5*rho*v^2) is cubic in beta. Bisect on
        // [0, beta] if the current beta already violates P >= P_FLOOR_FACE. Reuses the
        // same 16-iteration bisection depth as gradients::pressure_safe_scale.
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
    // face velocity
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

        // compute distance between generators (nn = |r_ij|) — index_j may be an MPI ghost,
        // so route the seed read through the ghost-aware accessor.
        const double3 seed_j = get_seed_at((int)index_j, (int)mesh->n_hydro, mesh);
        double        nnx    = wrap_periodic_delta(seed_j.x - mesh->seeds[i].x);
        double        nny    = wrap_periodic_delta(seed_j.y - mesh->seeds[i].y);
#ifdef dim_3D
        double nnz = wrap_periodic_delta(seed_j.z - mesh->seeds[i].z);
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
