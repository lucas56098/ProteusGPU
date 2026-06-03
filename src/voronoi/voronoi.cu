#include "../global/allvars.h"
#include "../global/structs.h"
#include "../knn/knn.h"
#include "../mpi/decomp.h"
#include "../mpi/halo.h"
#include "../mpi/migrate.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "cell.h"
#include "internal.h"
#include "voronoi.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "alloc.cu"
#include "build.cu"
#include "cell.cu"
#include "fallback.cu"
#include "geometry.cu"
#include "ghosts.cu"

namespace voronoi {

    // ---- per-step counters returned by build_mesh_growing_halo ----
    namespace {
        struct BuildStats {
            int  local_failed_cells      = 0; // cells still failing at end of widening (this rank)
            int  widen_iters_used        = 0; // iterations through the widening loop
            int  perturb_loop_iters_used = 0; // iterations through the perturb cascade
            int  cells_perturbed_total   = 0; // cells the perturbation moved across all rounds
            int  final_halo_width        = 0; // halo width W the step ended at
            bool have_mpi_neighbors      = false;
        };
    } // namespace

    // ---- forward declarations ----
    static BuildStats build_mesh_growing_halo(
        VMesh* mesh, POINT_TYPE* pts_data, hsize_t n_hydro, hydro::primvars* primvar, hydro::primvars* primvar_aux);
    static void cpu_perturb_and_rebuild(
        VMesh* mesh, POINT_TYPE* pts_data, hydro::primvars* primvar, hydro::primvars* primvar_aux, BuildStats& stats);
    static void exchange_used_ghost_primvars(VMesh* mesh, hydro::primvars* primvar);
    static void adapt_halo_width(const BuildStats& stats);
    static void print_step_summary(const BuildStats& stats);

    static hsize_t exchange_seeds_across_ranks(VMesh*      mesh,
                                               POINT_TYPE* pts_data,
                                               POINT_TYPE* pts,
                                               hsize_t*    original_ids,
                                               hsize_t     n_hydro,
                                               hsize_t     n_ghosts,
                                               int         W);
    static void    record_mpi_ghost_indices(hsize_t* original_ids, hsize_t n_hydro, hsize_t n_ghosts);
    static void    remap_exports_and_pts(VMesh* mesh, POINT_TYPE* pts_data, hsize_t n_hydro);
    static bool    widen_converged_across_ranks(VMesh* mesh, hsize_t n_ghosts, bool have_mpi, int* local_failed_out);
    static int     count_local_failed_cells(const VMesh* mesh);
    static int     has_cells_hitting_outer_halo(VMesh* mesh, int n_pgh);
    static void    sum_ints_across_ranks(const int* local, int* global, int n);
    static void    check_ghost_count(hsize_t n_ghosts, hsize_t max_ghosts);
    static void    copy_perturbed_seeds_back(const VMesh* mesh, POINT_TYPE* pts_data);
    static int     default_starting_halo_width();
#ifndef CPU_DEBUG
    GLOBAL static void kernel_outer_halo_check(hsize_t              n_hydro,
                                               const knn_problem*   knn,
                                               const unsigned int*  real_sorted_ids,
                                               int                  pts_mpi_base,
                                               int                  n_mpi_ghosts,
                                               const unsigned char* is_outer_layer,
                                               int*                 d_flag);
#endif

    // halo width remembered across calls: ratchets up when widening fires,
    // decays back to base after long streaks of steady steps
    static int s_last_W       = 0;
    static int s_steady_count = 0;

    // ============================================================
    // Main routines
    // ============================================================

    // build the Voronoi mesh on the [-buff, 1+buff]^d domain
    void compute_periodic_mesh(
        VMesh* mesh, POINT_TYPE* pts_data, hsize_t num_points, hydro::primvars* primvar, hydro::primvars* primvar_aux) {
        Profiler::StartTimer("MESH_TOTAL");

        // build the mesh, widening the halo until all cells succeed
        BuildStats stats = build_mesh_growing_halo(mesh, pts_data, num_points, primvar, primvar_aux);

        // perturb-and-rebuild fallback for cells that still failed
        if (stats.local_failed_cells > 0) cpu_perturb_and_rebuild(mesh, pts_data, primvar, primvar_aux, stats);

        // refresh used-ghost primvars, remember halo width, print summary
        exchange_used_ghost_primvars(mesh, primvar);
        adapt_halo_width(stats);
        print_step_summary(stats);

        Profiler::EndTimer("MESH_TOTAL");
    }

    // ============================================================
    // Helpers
    // ============================================================

    // rebuild the mesh, growing the halo each iter until no cells fail (across ranks)
    static BuildStats build_mesh_growing_halo(
        VMesh* mesh, POINT_TYPE* pts_data, hsize_t n_hydro, hydro::primvars* primvar, hydro::primvars* primvar_aux) {
        constexpr int MAX_WIDEN_ITERS = 4;

        // setup: ghost cap, mpi flag, starting halo width
        const double  ghost_frac = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        const hsize_t max_ghosts = (hsize_t)(2.0 * ghost_frac * n_hydro) + 1;
        const bool    have_mpi   = proteus_mpi::halo.n_neighbors > 0;

        BuildStats stats{};
        stats.have_mpi_neighbors = have_mpi;
        stats.final_halo_width   = have_mpi ? std::max(default_starting_halo_width(), s_last_W) : 0;

        POINT_TYPE* pts          = mesh->scratch_pts;
        hsize_t*    original_ids = mesh->ghost_ids;

        for (int iter = 0; iter < MAX_WIDEN_ITERS; iter++) {
            stats.widen_iters_used = iter;

            // generate periodic ghosts + (if MPI) exchange seeds with neighbour ranks
            Profiler::StartTimer("GHOST_GEN");
            const hsize_t n_ghosts = regenerate_periodic_ghosts(n_hydro, pts_data, pts, original_ids, buff);
            Profiler::EndTimer("GHOST_GEN");
            check_ghost_count(n_ghosts, max_ghosts);
            const hsize_t n_mpi =
                have_mpi ? exchange_seeds_across_ranks(
                               mesh, pts_data, pts, original_ids, n_hydro, n_ghosts, stats.final_halo_width)
                         : 0;

            // build the Voronoi cells from the augmented seed buffer
            mesh->pts_mpi_base = (int)(n_hydro + n_ghosts);
            compute_mesh(mesh, pts, (int)(n_hydro + n_ghosts + n_mpi), primvar, primvar_aux, iter);
            if (iter == 0 && have_mpi) remap_exports_and_pts(mesh, pts_data, n_hydro);

            // converged across all ranks: done
            int local_failed = 0;
            if (widen_converged_across_ranks(mesh, n_ghosts, have_mpi, &local_failed)) {
                if (iter > 0)
                    logging::root() << "VORONOI: halo widening converged in " << (iter + 1) << " iteration(s)."
                                    << std::endl;
                return stats;
            }

            // last iter: hand off remaining failures to CPU fallback
            if (iter == MAX_WIDEN_ITERS - 1) {
                stats.local_failed_cells = local_failed;
                logging::root() << "VORONOI: halo widening hit MAX_ITERS=" << MAX_WIDEN_ITERS
                                << " — falling through to CPU fallback." << std::endl;
                return stats;
            }

            // widen the halo and retry
            stats.final_halo_width += 2;
        }
        return stats; // unreachable
    }

    // CPU perturbation cascade: rebuild failed cells with seed perturbation; if any rank
    // perturbed, rebuild the mesh (perturbation invalidates neighbour-rank MPI ghosts)
    static void cpu_perturb_and_rebuild(
        VMesh* mesh, POINT_TYPE* pts_data, hydro::primvars* primvar, hydro::primvars* primvar_aux, BuildStats& stats) {
        constexpr int MAX_CASCADE_ITERS = 4;
        const bool    have_mpi          = stats.have_mpi_neighbors;
        POINT_TYPE*   pts               = mesh->scratch_pts;
        hsize_t*      original_ids      = mesh->ghost_ids;

        for (int iter = 0; iter < MAX_CASCADE_ITERS; iter++) {
            // attempt local CPU rebuild on this rank's failed cells
            const int local_perturbed = cpu_fallback_failed_cells(mesh);
            stats.cells_perturbed_total += local_perturbed;

            // converged across all ranks: no perturbation anywhere
            int global_perturbed = local_perturbed;
            if (have_mpi) sum_ints_across_ranks(&local_perturbed, &global_perturbed, 1);
            if (global_perturbed == 0) {
                stats.perturb_loop_iters_used = iter;
                if (iter > 0)
                    logging::root() << "VORONOI: perturbation cascade converged in " << iter << " round(s)."
                                    << std::endl;
                return;
            }
            stats.perturb_loop_iters_used = iter + 1;

            // last iter or single rank: log and return
            if (iter == MAX_CASCADE_ITERS - 1) {
                logging::root() << "VORONOI: perturbation cascade hit MAX_ITERS=" << MAX_CASCADE_ITERS << " with "
                                << global_perturbed << " cells perturbed in last round." << std::endl;
                return;
            }
            if (!have_mpi) return;

            // push perturbed seeds back into pts_data, regen ghosts, exchange, rebuild
            copy_perturbed_seeds_back(mesh, pts_data);
            const hsize_t n_ghosts = regenerate_periodic_ghosts(mesh->n_hydro, pts_data, pts, original_ids, buff);
            const hsize_t n_mpi    = exchange_seeds_across_ranks(
                mesh, pts_data, pts, original_ids, mesh->n_hydro, n_ghosts, stats.final_halo_width);
            mesh->pts_mpi_base = (int)(mesh->n_hydro + n_ghosts);
            compute_mesh(mesh,
                         pts,
                         (int)(mesh->n_hydro + n_ghosts + n_mpi),
                         primvar,
                         primvar_aux,
                         stats.widen_iters_used + 1 + iter);
        }
    }

    // refresh primvars (+ v_mesh if moving) on the subset of MPI ghosts that appear as
    // Voronoi-face neighbours; unused ghosts stay stale but are never read downstream
    static void exchange_used_ghost_primvars(VMesh* mesh, hydro::primvars* primvar) {
        if (proteus_mpi::halo.n_neighbors == 0) return;
        proteus_mpi::halo_build_used_subset(mesh);
        proteus_mpi::halo_exchange_primvars(mesh, primvar);
#ifdef MOVING_MESH
        proteus_mpi::halo_exchange_v_mesh(mesh);
#endif
    }

    // update the remembered halo width across calls: ratchet up after widening, decay
    // back to base after several steady steps so a one-off spike does not stay forever
    static void adapt_halo_width(const BuildStats& stats) {
        if (!stats.have_mpi_neighbors) return;
        const int W_base = default_starting_halo_width();

        if (stats.widen_iters_used > 0) {
            // widening fired this step: latch the new width, reset the steady counter
            s_last_W       = std::max(s_last_W, stats.final_halo_width);
            s_steady_count = 0;
        } else {
            // no widening this step: count toward a slow decay back to W_base
            s_steady_count++;
            if (s_steady_count >= 5 && s_last_W > W_base) {
                s_last_W       = std::max(W_base, s_last_W - 1);
                s_steady_count = 0;
            }
        }
    }

    // per-step printout (printed only when something interesting happened)
    static void print_step_summary(const BuildStats& stats) {
        const int widen_global    = logging::max_global(stats.widen_iters_used);
        const int cascade_global  = logging::max_global(stats.perturb_loop_iters_used);
        const int perturbed_total = logging::sum_global(stats.cells_perturbed_total);

        // retry counters: widening + perturb cascade + cells perturbed
        if (widen_global > 0 || cascade_global > 0 || perturbed_total > 0) {
            logging::root() << "VORONOI: retries widen=" << widen_global << " cascade=" << cascade_global
                            << " perturbed=" << perturbed_total << std::endl;
        }

        // halo send / migration usage
        int send_total_local = 0;
        for (int n = 0; n < proteus_mpi::halo.n_neighbors; n++)
            send_total_local += proteus_mpi::halo.send_count[n];
        const int send_used_local = proteus_mpi::halo.n_used_send;
        const int send_total_g    = logging::sum_global(send_total_local);
        const int send_used_g     = logging::sum_global(send_used_local);
        const int migrated_g      = logging::sum_global(proteus_mpi::last_n_migrated());

        if (send_total_g > 0 || migrated_g > 0) {
            const double pct_used = (send_total_g > 0) ? 100.0 * send_used_g / (double)send_total_g : 0.0;
            logging::root() << "MPI: send_used=" << send_used_g << "/" << send_total_g << " (" << pct_used
                            << "% used)  migrated=" << migrated_g << std::endl;
        }
    }

    // build the MPI export list, exchange seeds, record indices of freshly-arrived ghost slots
    static hsize_t exchange_seeds_across_ranks(VMesh*      mesh,
                                               POINT_TYPE* pts_data,
                                               POINT_TYPE* pts,
                                               hsize_t*    original_ids,
                                               hsize_t     n_hydro,
                                               hsize_t     n_ghosts,
                                               int         W) {
        proteus_mpi::halo_build_exports(pts_data, (int)n_hydro, buff, W);
        proteus_mpi::halo_exchange_seeds(mesh, pts, (int)(n_hydro + n_ghosts));
        record_mpi_ghost_indices(original_ids, n_hydro, n_ghosts);
        return (hsize_t)proteus_mpi::halo.n_mpi_ghosts;
    }

    // stamp the extended-array indices into original_ids[] for the MPI ghost slots
    static void record_mpi_ghost_indices(hsize_t* original_ids, hsize_t n_hydro, hsize_t n_ghosts) {
        for (int n = 0; n < proteus_mpi::halo.n_neighbors; n++) {
            const int ghost_off = proteus_mpi::halo.ghost_offset[n];
            for (int j = 0; j < proteus_mpi::halo.recv_count[n]; j++) {
                const int slot                         = ghost_off + j;
                const int ext_k                        = (int)n_hydro + slot;
                original_ids[n_ghosts + (hsize_t)slot] = (hsize_t)ext_k;
            }
        }
    }

    // iter 0 permuted primvar into new-k order, but export_indices and pts_data are still in
    // old-k order. Remap both, and reset orig_to_k_save to identity so subsequent lookup-mode
    // pass1s give k = orig.
    static void remap_exports_and_pts(VMesh* mesh, POINT_TYPE* pts_data, hsize_t n_hydro) {
        Profiler::StartTimer("MPI_REMAP");

        // build inverse permutation new_k -> old_k -> inv[old_k] = new_k
        static std::vector<unsigned int> inv_gather;
        inv_gather.resize((size_t)n_hydro);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t new_k = 0; new_k < n_hydro; new_k++) {
            inv_gather[mesh->gather_perm[new_k]] = (unsigned int)new_k;
        }
        proteus_mpi::halo_remap_export_indices(inv_gather.data(), (int)n_hydro);

        // gather pts_data into new-k order using gather_perm
        static std::vector<POINT_TYPE> pts_scratch;
        pts_scratch.resize((size_t)n_hydro);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t new_k = 0; new_k < n_hydro; new_k++) {
            pts_scratch[new_k] = pts_data[mesh->gather_perm[new_k]];
        }
        std::memcpy(pts_data, pts_scratch.data(), (size_t)n_hydro * sizeof(POINT_TYPE));

        // reset orig_to_k_save to identity for subsequent lookup-mode pass1s
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t k = 0; k < n_hydro; k++)
            mesh->orig_to_k_save[k] = (unsigned int)k;

        Profiler::EndTimer("MPI_REMAP");
    }

    // run the failure + outer-halo sentinels, sum across ranks, write local failed count
    static bool widen_converged_across_ranks(VMesh* mesh, hsize_t n_ghosts, bool have_mpi, int* local_failed_out) {
        // count this rank's failed cells
        const int local_failed = count_local_failed_cells(mesh);
        *local_failed_out      = local_failed;

        // sentinel: 1 if any cell's K-th nearest is in the outermost halo layer
        Profiler::StartTimer("MPI_COMPLETE");
        const int local_outer = have_mpi ? has_cells_hitting_outer_halo(mesh, (int)n_ghosts) : 0;
        Profiler::EndTimer("MPI_COMPLETE");

        // single-rank: convergence is purely local
        if (!have_mpi) return (local_failed == 0 && local_outer == 0);

        // MPI: sum both signals across ranks, converged iff both are zero everywhere
        const int local[2]  = {local_failed, local_outer};
        int       global[2] = {local_failed, local_outer};
        sum_ints_across_ranks(local, global, 2);
        return (global[0] == 0 && global[1] == 0);
    }

    // count cells whose status is not success (host-side scan)
    static int count_local_failed_cells(const VMesh* mesh) {
        int n_failed = 0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+ : n_failed)
#endif
        for (hsize_t k = 0; k < mesh->n_hydro; k++) {
            if (mesh->cell_status[k] != success) n_failed++;
        }
        return n_failed;
    }

    // returns 1 if any local cell's K-th nearest is an MPI ghost in the outermost halo
    // layer. The security_radius check can falsely pass when closer cells live beyond
    // the halo; a K-th sample in the outer layer is the signal that happened.
    static int has_cells_hitting_outer_halo(VMesh* mesh, int n_pgh) {
        // no MPI ghosts in play: nothing to check
        if (proteus_mpi::halo.n_neighbors == 0 || proteus_mpi::halo.n_mpi_ghosts == 0) return 0;

        const int n_hydro      = (int)mesh->n_hydro;
        const int pts_mpi_base = n_hydro + n_pgh;
        const int n_mpi        = proteus_mpi::halo.n_mpi_ghosts;

#ifndef CPU_DEBUG
        // GPU: launch the sentinel kernel that atomically sets *d_flag on the first hit
        static int* d_flag = nullptr;
        if (!d_flag) d_flag = gpu_alloc<int>(1);
        *d_flag = 0;

        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (n_hydro + tpb - 1) / tpb;
        kernel_outer_halo_check<<<blocks, tpb>>>(
            n_hydro, mesh->knn, mesh->real_sorted_ids, pts_mpi_base, n_mpi, proteus_mpi::halo.is_outer_layer, d_flag);
        GPU_LAUNCH_CHECK();
        GPU_SYNC();
        return *d_flag;
#else
        // CPU: reduce cell_hit_outer[] (set by compute_single_voronoi_cell on the CPU path)
        (void)pts_mpi_base;
        (void)n_mpi;
        int flag = 0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(| : flag)
#endif
        for (int k = 0; k < n_hydro; k++) {
            if (mesh->cell_hit_outer[k]) flag = 1;
        }
        return flag;
#endif
    }

    // MPI_Allreduce SUM wrapper; no-op when MPI is off
    static void sum_ints_across_ranks(const int* local, int* global, int n) {
#ifdef USE_MPI
        Profiler::StartTimer("MPI_REDUCE");
        MPI_Allreduce(local, global, n, MPI_INT, MPI_SUM, proteus_mpi::decomp.cart_comm);
        Profiler::EndTimer("MPI_REDUCE");
#else
        for (int i = 0; i < n; i++)
            global[i] = local[i];
#endif
    }

    // abort if the periodic-ghost count overflows the pre-allocated cap
    static void check_ghost_count(hsize_t n_ghosts, hsize_t max_ghosts) {
        if (n_ghosts > max_ghosts) {
            std::cerr << "VORONOI: Error! ghost count " << n_ghosts << " exceeds estimated max " << max_ghosts
                      << ". Distribution is highly non-uniform." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // push the perturbed seed positions from mesh->seeds back into pts_data
    static void copy_perturbed_seeds_back(const VMesh* mesh, POINT_TYPE* pts_data) {
        for (hsize_t k = 0; k < mesh->n_hydro; k++) {
            pts_data[k].x = mesh->seeds[k].x;
            pts_data[k].y = mesh->seeds[k].y;
#ifdef dim_3D
            pts_data[k].z = mesh->seeds[k].z;
#endif
        }
    }

    // empirical startup margin on top of the buff-derived halo width, so the first iter is
    // wide enough that the completeness check usually passes immediately
    static int default_starting_halo_width() {
        constexpr int W_STARTUP_MARGIN = 2;
        return proteus_mpi::halo_default_width(buff) + W_STARTUP_MARGIN;
    }

    // ============================================================
    // CUDA kernels
    // ============================================================
#ifndef CPU_DEBUG

    // each thread walks its K-nearest; sets *d_flag if any neighbour lands in the outermost halo layer
    GLOBAL static void kernel_outer_halo_check(hsize_t              n_hydro,
                                               const knn_problem*   knn,
                                               const unsigned int*  real_sorted_ids,
                                               int                  pts_mpi_base,
                                               int                  n_mpi_ghosts,
                                               const unsigned char* is_outer_layer,
                                               int*                 d_flag) {
        const hsize_t k = (hsize_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_hydro) return;
        if (*d_flag) return;

        const int    sid_self = (int)real_sorted_ids[k];
        unsigned int knearest[_K_];
        knn::knn_for_point<_K_>(sid_self, knn, knearest);

        for (int i = 0; i < _K_; i++) {
            const unsigned int sid  = knearest[i];
            const unsigned int orig = knn->d_permutation[sid];
            if ((int)orig < pts_mpi_base) continue;
            const int slot = (int)orig - pts_mpi_base;
            if (slot < 0 || slot >= n_mpi_ghosts) continue;
            if (is_outer_layer[slot]) {
                atomicOr(d_flag, 1);
                return;
            }
        }
    }

#endif // !CPU_DEBUG

} // namespace voronoi
