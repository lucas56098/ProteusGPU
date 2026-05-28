#include "../global/allvars.h"
#include "../gradients/gradients.h"
#include "../hydro/riemann.h"
#include "../mpi/decomp.h"
#include "../mpi/halo.h"
#include "../mpi/migrate.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "periodic_mesh.h"
#include <cmath>
#include <cstring>
#include <iostream>

namespace voronoi {

    constexpr double PI = 3.14159265358979323846;

    // forward declarations
#ifdef MOVING_MESH
    HD void compute_mesh_velocity_for_cell(hsize_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    HD void move_mesh_for_cell(hsize_t, const VMesh*, double, POINT_TYPE*);
#endif

#ifndef CPU_DEBUG
#ifdef MOVING_MESH
    // kernels
    GLOBAL void kernel_mesh_velocities(hsize_t, VMesh*, const hydro::primvars*, const gradients::PrimGradients*);
    GLOBAL void kernel_move_mesh(hsize_t, const VMesh*, double, POINT_TYPE*);
#endif
    GLOBAL void kernel_generate_ghosts(hsize_t,
                                       const POINT_TYPE* __restrict__,
                                       POINT_TYPE* __restrict__,
                                       hsize_t* __restrict__,
                                       int* __restrict__,
                                       double);
#endif

    // Inline helpers
    HD inline bool is_in(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za = 0.0, double zb = 1.0) {
#ifdef dim_2D
        (void)za;
        (void)zb;
        return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb);
#else
        return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb) && (pt.z > za && pt.z < zb);
#endif
    }

    inline void add_ghost(POINT_TYPE*    pts,
                          hsize_t        index,
                          hsize_t*       n_ghosts,
                          const hsize_t* n_hydro,
                          hsize_t*       original_ids,
                          double         shift_x,
                          double         shift_y,
                          double         shift_z = 0.0) {
        POINT_TYPE pt;
        pt.x = pts[index].x + shift_x;
        pt.y = pts[index].y + shift_y;
#ifdef dim_3D
        pt.z = pts[index].z + shift_z;
#else
        (void)shift_z;
#endif

        pts[(*n_hydro) + (*n_ghosts)] = pt;
        original_ids[*n_ghosts]       = index;
        (*n_ghosts)++;
    }

    // ============================================================
    // periodic mesh computation, velocity, motion
    // ============================================================

    // regenerate periodic ghosts and copy pts_data → pts for the real-cell range;
    // returns the number of ghosts emitted
    static hsize_t regenerate_periodic_ghosts(
        hsize_t n_hydro, const POINT_TYPE* pts_data, POINT_TYPE* pts, hsize_t* original_ids, double buff_val) {
        hsize_t n_ghosts = 0;
#ifndef CPU_DEBUG
        int* d_ghost_count = (int*)gpu_malloc(sizeof(int));
        gpu_memset(d_ghost_count, 0, sizeof(int));
        int tpb    = _MESH_BLOCK_SIZE_;
        int blocks = ((int)n_hydro + tpb - 1) / tpb;
        kernel_generate_ghosts<<<blocks, tpb>>>(n_hydro, pts_data, pts, original_ids, d_ghost_count, buff_val);
        GPU_SYNC();
        n_ghosts = (hsize_t)(*d_ghost_count);
        gpu_free(d_ghost_count);
#else
        for (hsize_t i = 0; i < n_hydro; i++) {
            pts[i] = pts_data[i];

            for (int sx = -1; sx <= 1; sx++) {
                for (int sy = -1; sy <= 1; sy++) {
#ifdef dim_3D
                    for (int sz = -1; sz <= 1; sz++) {
#else
                    {
                        int sz = 0;
#endif
                        if (sx == 0 && sy == 0 && sz == 0) continue;
                        double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff_val : 0.0;
                        double xb = (sx == 1) ? buff_val : 1.0;
                        double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff_val : 0.0;
                        double yb = (sy == 1) ? buff_val : 1.0;
                        double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff_val : 0.0;
                        double zb = (sz == 1) ? buff_val : 1.0;
                        if (is_in(pts[i], xa, xb, ya, yb, za, zb)) {
                            add_ghost(pts, i, &n_ghosts, &n_hydro, original_ids, (double)sx, (double)sy, (double)sz);
                        }
                    }
                }
            }
        }
#endif
        return n_ghosts;
    }

    // wrap seeds around unit-box boundaries with ghost copies, then build the mesh
    // on the enlarged [-buff, 1+buff]^d domain (no rescaling step)
    void compute_periodic_mesh(
        VMesh* mesh, POINT_TYPE* pts_data, hsize_t num_points, hydro::primvars* primvar, hydro::primvars* primvar_aux) {
        Profiler::StartTimer("MESH_TOTAL");

        double  ghost_frac       = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        hsize_t max_ghost_points = (hsize_t)(2.0 * ghost_frac * num_points) + 1;

        POINT_TYPE* pts     = mesh->scratch_pts;
        hsize_t     n_hydro = num_points;

        hsize_t* original_ids = mesh->ghost_ids;

        constexpr int MAX_WIDEN_ITERS   = 4;
        constexpr int MAX_CASCADE_ITERS = 4;

        // Empirical startup margin on top of the periodic-buff-derived W_base.
        // Avoids the typical first-step widening iteration on non-trivial ICs
        // (e.g. KH at moderate density) by starting wide enough that the
        // completeness check passes on the first build.
        constexpr int W_STARTUP_MARGIN  = 2;
        const bool have_mpi_neighbors = proteus_mpi::halo.n_neighbors > 0;
        const int  W_base             = have_mpi_neighbors
                                            ? proteus_mpi::halo_default_width(buff) + W_STARTUP_MARGIN
                                            : 0;
        // sticky across steps: start from whatever the last build converged at. If widening
        // didn't fire last step we'll try shrinking by 1 below to track changing density.
        static int s_last_W           = 0;
        hsize_t    n_ghosts           = 0;
        hsize_t    n_mpi_ghosts       = 0;
        int        W_iter             = have_mpi_neighbors ? std::max(W_base, s_last_W) : 0;
        int        last_iter          = 0;

        // halo-widening loop: keep expanding the halo width until no cell fails security_radius.
        // Lockstep Allreduce(SUM) on the per-rank failure count keeps all ranks in sync.
        // cpu_fallback runs outside this loop — widening the halo addresses the root cause.
        for (int iter = 0; iter < MAX_WIDEN_ITERS; iter++) {
            last_iter = iter;

            Profiler::StartTimer("GHOST_GEN (cpu)");
            n_ghosts = regenerate_periodic_ghosts(n_hydro, pts_data, pts, original_ids, buff);
            Profiler::EndTimer("GHOST_GEN (cpu)");
            if (n_ghosts > max_ghost_points) {
                std::cerr << "VORONOI: Error! ghost count " << n_ghosts << " exceeds estimated max " << max_ghost_points
                          << ". Distribution is highly non-uniform." << std::endl;
                exit(EXIT_FAILURE);
            }

            n_mpi_ghosts = 0;
            if (have_mpi_neighbors) {
                proteus_mpi::halo_build_exports(pts_data, (int)n_hydro, buff, W_iter);
                proteus_mpi::halo_exchange_seeds(mesh, pts, (int)(n_hydro + n_ghosts));
                n_mpi_ghosts = (hsize_t)proteus_mpi::halo.n_mpi_ghosts;
                for (int n = 0; n < proteus_mpi::halo.n_neighbors; n++) {
                    int ghost_off = proteus_mpi::halo.ghost_offset[n];
                    for (int j = 0; j < proteus_mpi::halo.recv_count[n]; j++) {
                        int slot                                        = ghost_off + j;
                        int ext_k                                       = (int)n_hydro + slot;
                        original_ids[(hsize_t)n_ghosts + (hsize_t)slot] = (hsize_t)ext_k;
                    }
                }
            }
            const hsize_t n_total = n_hydro + n_ghosts + n_mpi_ghosts;
            mesh->pts_mpi_base    = (int)(n_hydro + n_ghosts);
            voronoi::compute_mesh(mesh, pts, (int)n_total, primvar, primvar_aux, iter);

            // iter == 0 permuted primvar into new-k order, but export_indices and pts_data
            // are still in old-k order. To keep subsequent exchanges and iter > 0 rebuilds
            // consistent with the post-permute primvar:
            //   (1) remap export_indices via inv_gather (old_k → new_k);
            //   (2) re-shuffle pts_data so KNN at iter > 0 indexes cells in new-k order;
            //   (3) set orig_to_k_save = identity so the lookup pass1 gives k = orig.
            if (iter == 0 && have_mpi_neighbors) {
                Profiler::StartTimer("MPI_COMM");
                Profiler::StartTimer("MPI_REMAP");
                static std::vector<unsigned int> inv_gather;
                inv_gather.resize((size_t)n_hydro);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (hsize_t new_k = 0; new_k < n_hydro; new_k++) {
                    inv_gather[mesh->gather_perm[new_k]] = (unsigned int)new_k;
                }
                proteus_mpi::halo_remap_export_indices(inv_gather.data(), (int)n_hydro);

                static std::vector<POINT_TYPE> pts_scratch;
                pts_scratch.resize((size_t)n_hydro);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (hsize_t new_k = 0; new_k < n_hydro; new_k++) {
                    pts_scratch[new_k] = pts_data[mesh->gather_perm[new_k]];
                }
                std::memcpy(pts_data, pts_scratch.data(), (size_t)n_hydro * sizeof(POINT_TYPE));

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (hsize_t k = 0; k < n_hydro; k++)
                    mesh->orig_to_k_save[k] = (unsigned int)k;
                Profiler::EndTimer("MPI_REMAP");
                Profiler::EndTimer("MPI_COMM");
            }

            Profiler::StartTimer("MPI_COMM");
            int local_failed = 0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+:local_failed)
#endif
            for (hsize_t k = 0; k < n_hydro; k++) {
                if (mesh->cell_status[k] != voronoi::success) local_failed++;
            }
            // halo-completeness sentinel: catches the silent-failure case where every cell
            // reports success but some K-nearest sample reached the outermost halo layer,
            // meaning closer cells beyond the halo may have been missed
            Profiler::StartTimer("MPI_COMPLETE");
            const int local_outer = have_mpi_neighbors ? voronoi::halo_completeness_flag(mesh, (int)n_ghosts) : 0;
            Profiler::EndTimer("MPI_COMPLETE");
            Profiler::EndTimer("MPI_COMM");

            int global_signal[2] = {local_failed, local_outer};
#ifdef USE_MPI
            if (have_mpi_neighbors) {
                const int local_signal[2] = {local_failed, local_outer};
                Profiler::StartTimer("MPI_COMM");
                Profiler::StartTimer("MPI_REDUCE");
                MPI_Allreduce(local_signal, global_signal, 2, MPI_INT, MPI_SUM, proteus_mpi::decomp.cart_comm);
                Profiler::EndTimer("MPI_REDUCE");
                Profiler::EndTimer("MPI_COMM");
            }
#endif
            const int global_failed = global_signal[0];
            const int global_outer  = global_signal[1];
            if (global_failed == 0 && global_outer == 0) {
                if (iter > 0) {
                    logging::root() << "VORONOI: halo widening converged in " << (iter + 1) << " iteration(s)."
                                    << std::endl;
                }
                break;
            }
            if (iter == MAX_WIDEN_ITERS - 1) {
                logging::root() << "VORONOI: halo widening hit MAX_ITERS=" << MAX_WIDEN_ITERS
                                << " (failed=" << global_failed << ", outer-reached=" << global_outer
                                << ") — falling through to CPU fallback." << std::endl;
                break;
            }
            W_iter += 2;
        }

        // cross-rank perturbation cascade. cpu_fallback may perturb seeds; if any perturbed cell
        // sits in the boundary layer, the neighbor rank's MPI ghost copy is stale and we must
        // refresh pts_data, regen halos, and rebuild the mesh. Lockstep on global perturbation count.
        int cascade_iters_used    = 0;
        int cascade_perturbed_sum = 0;
        for (int cascade = 0; cascade < MAX_CASCADE_ITERS; cascade++) {
            const int local_perturbed = voronoi::cpu_fallback_failed_cells(mesh);
            cascade_perturbed_sum += local_perturbed;

            int global_perturbed = local_perturbed;
#ifdef USE_MPI
            if (have_mpi_neighbors) {
                Profiler::StartTimer("MPI_COMM");
                Profiler::StartTimer("MPI_REDUCE");
                MPI_Allreduce(&local_perturbed, &global_perturbed, 1, MPI_INT, MPI_SUM, proteus_mpi::decomp.cart_comm);
                Profiler::EndTimer("MPI_REDUCE");
                Profiler::EndTimer("MPI_COMM");
            }
#endif
            if (global_perturbed == 0) {
                cascade_iters_used = cascade;
                if (cascade > 0) {
                    logging::root() << "VORONOI: perturbation cascade converged in " << cascade << " round(s)."
                                    << std::endl;
                }
                break;
            }
            cascade_iters_used = cascade + 1;
            if (cascade == MAX_CASCADE_ITERS - 1) {
                logging::root() << "VORONOI: perturbation cascade hit MAX_ITERS=" << MAX_CASCADE_ITERS << " with "
                                << global_perturbed << " cells perturbed in last round." << std::endl;
                break;
            }
            if (!have_mpi_neighbors) break; // single-rank: no cross-rank cascade

            // push perturbed seed positions back into pts_data and rebuild halos + mesh
            for (hsize_t k = 0; k < n_hydro; k++) {
                pts_data[k].x = mesh->seeds[k].x;
                pts_data[k].y = mesh->seeds[k].y;
#ifdef dim_3D
                pts_data[k].z = mesh->seeds[k].z;
#endif
            }
            n_ghosts = regenerate_periodic_ghosts(n_hydro, pts_data, pts, original_ids, buff);
            proteus_mpi::halo_build_exports(pts_data, (int)n_hydro, buff, W_iter);
            proteus_mpi::halo_exchange_seeds(mesh, pts, (int)(n_hydro + n_ghosts));
            n_mpi_ghosts = (hsize_t)proteus_mpi::halo.n_mpi_ghosts;
            for (int n = 0; n < proteus_mpi::halo.n_neighbors; n++) {
                int ghost_off = proteus_mpi::halo.ghost_offset[n];
                for (int j = 0; j < proteus_mpi::halo.recv_count[n]; j++) {
                    int slot                                        = ghost_off + j;
                    int ext_k                                       = (int)n_hydro + slot;
                    original_ids[(hsize_t)n_ghosts + (hsize_t)slot] = (hsize_t)ext_k;
                }
            }
            const hsize_t n_total = n_hydro + n_ghosts + n_mpi_ghosts;
            mesh->pts_mpi_base    = (int)(n_hydro + n_ghosts);
            voronoi::compute_mesh(mesh,
                                  pts,
                                  (int)n_total,
                                  primvar,
                                  primvar_aux,
                                  /*iter=*/last_iter + 1 + cascade);
        }

        // Mesh is final. Compute the used-MPI-ghost subset (which slots actually
        // appear as Voronoi-face neighbors of any local cell) and refresh primvars
        // and v_mesh on that subset. Unused ghosts are skipped — their primvars
        // stay stale but are never read in the gradient/flux paths.
        if (have_mpi_neighbors) {
            proteus_mpi::halo_build_used_subset(mesh);
            proteus_mpi::halo_exchange_primvars(mesh, primvar);
#ifdef MOVING_MESH
            proteus_mpi::halo_exchange_v_mesh(mesh);
#endif
        }

        // sticky W with slow decay. Lock in the wider W if widening fired this
        // step; otherwise let it shrink by 1 every few steady steps so a one-off
        // density spike at startup doesn't permanently inflate the halo.
        if (have_mpi_neighbors) {
            static int s_steady_count = 0;
            if (last_iter > 0) {
                s_last_W       = std::max(s_last_W, W_iter);
                s_steady_count = 0;
            } else {
                s_steady_count++;
                if (s_steady_count >= 5 && s_last_W > W_base) {
                    s_last_W       = std::max(W_base, s_last_W - 1);
                    s_steady_count = 0;
                }
            }
        }

        // local halo send counts
        int send_total_local = 0;
        for (int n = 0; n < proteus_mpi::halo.n_neighbors; n++) send_total_local += proteus_mpi::halo.send_count[n];
        const int send_used_local = proteus_mpi::halo.n_used_send;

        // global reductions for the per-step summary
        const int widen_global    = logging::max_global(last_iter);          // 0 = no widening fired
        const int cascade_global  = logging::max_global(cascade_iters_used); // 0 = no perturb cascade
        const int perturbed_total = logging::sum_global(cascade_perturbed_sum);
        const int send_total_g    = logging::sum_global(send_total_local);
        const int send_used_g     = logging::sum_global(send_used_local);
        const int migrated_g      = logging::sum_global(proteus_mpi::last_n_migrated());

        // VORONOI: retries — printed only when anomalies occurred
        if (widen_global > 0 || cascade_global > 0 || perturbed_total > 0) {
            logging::root() << "VORONOI: retries widen=" << widen_global
                            << " cascade=" << cascade_global
                            << " perturbed=" << perturbed_total << std::endl;
        }

        // MPI: per-step halo+migrate summary
        if (send_total_g > 0 || migrated_g > 0) {
            const double pct_used = (send_total_g > 0) ? 100.0 * send_used_g / (double)send_total_g : 0.0;
            logging::root() << "MPI: send_used=" << send_used_g << "/" << send_total_g
                            << " (" << pct_used << "% used)  migrated=" << migrated_g << std::endl;
        }

        Profiler::EndTimer("MESH_TOTAL");
    }

#ifdef MOVING_MESH
    // gas-velocity + Lloyd-style regularization, per hydro cell
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads) {

#ifndef CPU_DEBUG
        int tpb    = _MESH_BLOCK_SIZE_;
        int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
        Profiler::StartGPU("kernel_mesh_velocities");
        kernel_mesh_velocities<<<blocks, tpb>>>(mesh->n_hydro, mesh, primvar, grads);
        Profiler::EndGPU("kernel_mesh_velocities");
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            compute_mesh_velocity_for_cell(i, mesh, primvar, grads);
        }
#endif
    }

    // advance seeds by v_mesh*dt (with periodic wrap), then rebuild the mesh
    void move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::primvars* primvar_aux) {

        POINT_TYPE* pts     = mesh->scratch_move;
        hsize_t     n_hydro = mesh->n_hydro;

#ifndef CPU_DEBUG
        int tpb    = _MESH_BLOCK_SIZE_;
        int blocks = ((int)n_hydro + tpb - 1) / tpb;
        kernel_move_mesh<<<blocks, tpb>>>(n_hydro, mesh, dt, pts);
        GPU_LAUNCH_CHECK();
        GPU_SYNC(); // migrate_seeds reads pts on the host below
#else
        for (hsize_t i = 0; i < n_hydro; i++) {
            move_mesh_for_cell(i, mesh, dt, pts);
        }
#endif

        // ship cells whose new bucket is owned by another rank; updates mesh->n_hydro
        // and compacts/extends pts (== mesh->scratch_move) to match
        proteus_mpi::migrate_seeds(mesh, primvar, primvar_aux);
        n_hydro = mesh->n_hydro;

        compute_periodic_mesh(mesh, pts, n_hydro, primvar, primvar_aux);
    }
#endif // MOVING_MESH

    // ============================================================
    // CUDA kernel wrappers
    // ============================================================
#ifndef CPU_DEBUG

#ifdef MOVING_MESH
    GLOBAL void kernel_mesh_velocities(hsize_t                         n_hydro,
                                       VMesh*                          mesh,
                                       const hydro::primvars*          primvar,
                                       const gradients::PrimGradients* grads) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        compute_mesh_velocity_for_cell(i, mesh, primvar, grads);
    }

    GLOBAL void kernel_move_mesh(hsize_t n_hydro, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        move_mesh_for_cell(i, mesh, dt, pts);
    }
#endif // MOVING_MESH

    // warp-aggregated ghost generation: each warp computes its members' ghost counts, prefix-sums
    // them across lanes, then a single atomicAdd by lane 0 claims a contiguous slot range. Each
    // lane writes its ghosts at known offsets within that range. Replaces up to 7 atomics per
    // thread (3D corner cell) with 1 atomic per warp — critical when threads in a warp are
    // spatially clustered (post-spatial-sort) and many cells produce ghosts simultaneously.
    GLOBAL void kernel_generate_ghosts(hsize_t n_hydro,
                                       const POINT_TYPE* __restrict__ pts_data,
                                       POINT_TYPE* __restrict__ pts,
                                       hsize_t* __restrict__ original_ids,
                                       int* __restrict__ d_ghost_count,
                                       double buff_val) {
        hsize_t i      = blockIdx.x * blockDim.x + threadIdx.x;
        bool    active = (i < n_hydro);

        // copy real cell into scratch_pts (every thread does this so the [0, n_hydro) range is
        // populated; out-of-range threads zero pi to keep the geometry test well-defined).
        POINT_TYPE pi;
        if (active) {
            pts[i] = pts_data[i];
            pi     = pts[i];
        } else {
            pi.x = 0.0;
            pi.y = 0.0;
#ifdef dim_3D
            pi.z = 0.0;
#endif
        }

        // pass 1: count ghosts this thread will produce (0..7 in 3D, 0..3 in 2D)
        int my_count = 0;
        if (active) {
            for (int sx = -1; sx <= 1; sx++) {
                for (int sy = -1; sy <= 1; sy++) {
#ifdef dim_3D
                    for (int sz = -1; sz <= 1; sz++) {
#else
                    {
                        int sz = 0;
#endif
                        if (sx == 0 && sy == 0 && sz == 0) continue;

                        double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff_val : 0.0;
                        double xb = (sx == 1) ? buff_val : 1.0;
                        double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff_val : 0.0;
                        double yb = (sy == 1) ? buff_val : 1.0;
                        double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff_val : 0.0;
                        double zb = (sz == 1) ? buff_val : 1.0;

                        if (is_in(pi, xa, xb, ya, yb, za, zb)) my_count++;
                    }
                }
            }
        }

        // warp inclusive prefix sum over my_count (Kogge-Stone via __shfl_up_sync).
        // ALL 32 lanes must participate; inactive threads contribute my_count=0.
        const unsigned full_mask = 0xffffffffu;
        int            s         = my_count;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            int t = __shfl_up_sync(full_mask, s, d);
            if ((int)(threadIdx.x & 31) >= d) s += t;
        }
        int warp_total = __shfl_sync(full_mask, s, 31);
        int my_excl    = s - my_count;

        // one atomicAdd per warp (skip if warp produced no ghosts)
        int warp_base = 0;
        if ((threadIdx.x & 31) == 0 && warp_total > 0) { warp_base = portable_atomicAdd(d_ghost_count, warp_total); }
        warp_base = __shfl_sync(full_mask, warp_base, 0);

        if (!active || my_count == 0) return;

        // pass 2: re-run the geometry tests and write ghosts at known slots.
        // Re-running the tests is cheaper than spilling 7 directions into registers/local memory.
        int my_base   = warp_base + my_excl;
        int n_written = 0;
        for (int sx = -1; sx <= 1; sx++) {
            for (int sy = -1; sy <= 1; sy++) {
#ifdef dim_3D
                for (int sz = -1; sz <= 1; sz++) {
#else
                {
                    int sz = 0;
#endif
                    if (sx == 0 && sy == 0 && sz == 0) continue;

                    double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff_val : 0.0;
                    double xb = (sx == 1) ? buff_val : 1.0;
                    double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff_val : 0.0;
                    double yb = (sy == 1) ? buff_val : 1.0;
                    double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff_val : 0.0;
                    double zb = (sz == 1) ? buff_val : 1.0;

                    if (is_in(pi, xa, xb, ya, yb, za, zb)) {
                        int        slot = my_base + n_written;
                        POINT_TYPE gpt;
                        gpt.x = pi.x + (double)sx;
                        gpt.y = pi.y + (double)sy;
#ifdef dim_3D
                        gpt.z = pi.z + (double)sz;
#endif
                        pts[n_hydro + slot] = gpt;
                        original_ids[slot]  = i;
                        n_written++;
                    }
                }
            }
        }
    }

#endif // !CPU_DEBUG

    // ============================================================
    // Per-cell work (called by kernels and CPU loops)
    // ============================================================
#ifdef MOVING_MESH

    HD void compute_mesh_velocity_for_cell(hsize_t                         i,
                                           VMesh*                          mesh,
                                           const hydro::primvars*          primvar,
                                           const gradients::PrimGradients* grads) {
        // mesh velocity starts as the gas velocity
        double vx_mesh = primvar->v[i].x;
        double vy_mesh = primvar->v[i].y;
#ifdef dim_3D
        double vz_mesh = primvar->v[i].z;
#endif

        // effective cell radius (sphere/disk equivalent)
#ifdef dim_2D
        const double Ri = sqrt(fmax(mesh->volumes[i], 0.0) / PI);
#else
        const double Ri = cbrt(3.0 * fmax(mesh->volumes[i], 0.0) / (4.0 * PI));
#endif

        // displacement of seed from cell centroid (Lloyd target)
        double dx = wrap_periodic_delta(mesh->com[i].x - mesh->seeds[i].x);
        double dy = wrap_periodic_delta(mesh->com[i].y - mesh->seeds[i].y);
#ifdef dim_3D
        double dz = wrap_periodic_delta(mesh->com[i].z - mesh->seeds[i].z);
#endif

        // density-gradient correction: bias the target toward the steeper side, capped at Ri/4.
        // The cap is a smooth clamp (not a binary skip) so the offset stays continuous across
        // steps — otherwise small fluctuations near the cap flip the regularization on/off.
        if (grads != nullptr && Ri > 0.0) {
#ifdef dim_3D
            const double dgrad = sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y +
                                      grads->rho[i].z * grads->rho[i].z);
#else
            const double dgrad = sqrt(grads->rho[i].x * grads->rho[i].x + grads->rho[i].y * grads->rho[i].y);
#endif
            if (dgrad > 0.0) {
                const double scale = primvar->rho[i] / dgrad;
                const double tmp   = 3.0 * Ri + scale;
                const double disc  = tmp * tmp - 8.0 * Ri * Ri;
                if (disc > 0.0) {
                    const double x_off  = (tmp - sqrt(disc)) / 4.0;
                    const double offset = fmin(x_off, 0.25 * Ri);
                    dx += offset * grads->rho[i].x / dgrad;
                    dy += offset * grads->rho[i].y / dgrad;
#ifdef dim_3D
                    dz += offset * grads->rho[i].z / dgrad;
#endif
                }
            }
        }

#ifdef dim_3D
        const double di = sqrt(dx * dx + dy * dy + dz * dz);
#else
        const double di = sqrt(dx * dx + dy * dy);
#endif

        // ramp regularization velocity from 0 (well-shaped) to CellShapingSpeed*c_s (very distorted)
        if (di > 0.0 && Ri > 0.0) {
            const double threshold = CellShapingFactor * Ri;
            double       fraction  = 0.0;
            if (di > 0.75 * threshold) {
                if (di > threshold)
                    fraction = CellShapingSpeed;
                else
                    fraction = CellShapingSpeed * (di - 0.75 * threshold) / (0.25 * threshold);
            }

            if (fraction > 0.0) {
                // scale by sound speed so regularization respects local timescales
                const double rho     = primvar->rho[i];
                hydro::prim  state_i = get_state(i, primvar);
                const double p       = fmax(0.0, hydro::get_P_ideal_gas(&state_i));
                if (rho > 0.0 && p > 0.0) {
                    const double ci = sqrt(gamma_eos * p / rho);
                    vx_mesh += fraction * ci * dx / di;
                    vy_mesh += fraction * ci * dy / di;
#ifdef dim_3D
                    vz_mesh += fraction * ci * dz / di;
#endif
                }
            }
        }

        mesh->v_mesh[i].x = vx_mesh;
        mesh->v_mesh[i].y = vy_mesh;
#ifdef dim_3D
        mesh->v_mesh[i].z = vz_mesh;
#endif
    }

    // advance one seed by v_mesh*dt with periodic wrap into [0,1)
    HD void move_mesh_for_cell(hsize_t i, const VMesh* mesh, double dt, POINT_TYPE* pts) {
        pts[i].x = fmod((mesh->seeds[i].x + dt * mesh->v_mesh[i].x) + 1.0, 1.0);
        pts[i].y = fmod((mesh->seeds[i].y + dt * mesh->v_mesh[i].y) + 1.0, 1.0);
#ifdef dim_3D
        pts[i].z = fmod((mesh->seeds[i].z + dt * mesh->v_mesh[i].z) + 1.0, 1.0);
#endif
    }

#endif // MOVING_MESH

} // namespace voronoi
