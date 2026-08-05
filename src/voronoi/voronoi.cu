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
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
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
            int  global_failed_cells     = 0; // same, summed across ranks (set in compute_periodic_mesh)
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
    static void cpu_perturb_and_repair(VMesh* mesh, BuildStats& stats, double dt);
    static void exchange_used_ghost_primvars(VMesh* mesh, hydro::primvars* primvar);
    static void adapt_halo_width(const BuildStats& stats);
    static void print_step_summary(const BuildStats& stats);

    static hsize_t exchange_seeds_across_ranks(VMesh*       mesh,
                                               POINT_TYPE*  pts_data,
                                               POINT_TYPE*& pts,
                                               hsize_t*&    original_ids,
                                               hsize_t      n_hydro,
                                               hsize_t      n_ghosts,
                                               int          W);
    static void    record_mpi_ghost_indices(hsize_t* original_ids, hsize_t n_hydro, hsize_t n_ghosts);
    static void    remap_exports_and_pts(VMesh* mesh, POINT_TYPE* pts_data, hsize_t n_hydro);
    static bool
    widen_converged_across_ranks(VMesh* mesh, bool have_mpi, int* local_failed_out, int* global_beyond_out);
    static int     count_local_failed_cells(const VMesh* mesh);
    static int     count_local_beyond_data_cells(const VMesh* mesh);
    static void    sum_ints_across_ranks(const int* local, int* global, int n);
    static void    check_ghost_count(hsize_t n_ghosts, hsize_t max_ghosts);
    static int     default_starting_halo_width();
    static void    set_data_extent_for_build(VMesh* mesh, int W, bool have_mpi);

    // halo width remembered across calls: ratchets up when widening fires,
    // decays back to base after long streaks of steady steps
    static int s_last_W       = 0;
    static int s_steady_count = 0;

    // ============================================================
    // Main routines
    // ============================================================

    // build the Voronoi mesh on the [-buff, 1+buff]^d domain
    void compute_periodic_mesh(VMesh*           mesh,
                               POINT_TYPE*      pts_data,
                               hsize_t          num_points,
                               hydro::primvars* primvar,
                               hydro::primvars* primvar_aux,
                               double           dt) {
        PROFILE("MESH");

        // build the mesh, widening the halo until all cells succeed
        BuildStats stats = build_mesh_growing_halo(mesh, pts_data, num_points, primvar, primvar_aux);

        // single global reduction so every rank agrees on whether to enter the fallback cascade.
        // Without this the gate would be per-rank and the inner sum_ints_across_ranks calls
        // (line ~195) would deadlock as soon as some ranks have failures and others don't.
        stats.global_failed_cells = logging::sum_global(stats.local_failed_cells);

        // perturb-and-rebuild fallback for cells that still failed (on any rank)
        if (stats.global_failed_cells > 0) {
            PROFILE("PERTURB");
            cpu_perturb_and_repair(mesh, stats, dt);
        }

        // refresh used-ghost primvars, remember halo width, print summary
        exchange_used_ghost_primvars(mesh, primvar);
        adapt_halo_width(stats);
        print_step_summary(stats);
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

        int prev_beyond = INT_MAX; // uncertified count from the previous iteration (no-progress backstop)
        for (int iter = 0; iter < MAX_WIDEN_ITERS; iter++) {
            stats.widen_iters_used = iter;

            // re-read scratch_pts / ghost_ids each iter: halo_build_exports inside
            // exchange_seeds_across_ranks may grow the halo capacity and reallocate them.
            POINT_TYPE* pts          = mesh->scratch_pts;
            hsize_t*    original_ids = mesh->ghost_ids;

            // generate periodic ghosts + (if MPI) exchange seeds with neighbour ranks
            hsize_t n_ghosts;
            {
                PROFILE("GHOSTS");
                n_ghosts = regenerate_periodic_ghosts(n_hydro, pts_data, pts, original_ids, buff);
            }
            check_ghost_count(n_ghosts, max_ghosts);
            const hsize_t n_mpi =
                have_mpi ? exchange_seeds_across_ranks(
                               mesh, pts_data, pts, original_ids, n_hydro, n_ghosts, stats.final_halo_width)
                         : 0;

            // build the Voronoi cells from the augmented seed buffer
            mesh->n_mpi_ghosts   = proteus_mpi::halo.n_mpi_ghosts;
            set_data_extent_for_build(mesh, stats.final_halo_width, have_mpi);
            compute_mesh(mesh, pts, (int)(n_hydro + n_ghosts + n_mpi), primvar, primvar_aux, iter);
            if (iter == 0 && have_mpi) remap_exports_and_pts(mesh, pts_data, n_hydro);

            // "converged" only means no cell is halo-limited; it says nothing about overflows,
            // degeneracies or K-limited cells. Those still have to reach cpu_perturb_and_repair,
            // which fires off stats.local_failed_cells — so record it on every exit path.
            int        local_failed = 0, global_beyond = 0;
            const bool converged = widen_converged_across_ranks(mesh, have_mpi, &local_failed, &global_beyond);
            stats.local_failed_cells = local_failed;

            // converged across all ranks: done
            if (converged) {
                if (iter > 0)
                    logging::root() << "VORONOI: halo widening converged in " << (iter + 1) << " iteration(s)."
                                    << std::endl;
                return stats;
            }

            // no MPI: widening only grows the MPI halo width, which is unused here.
            // Re-running the build with identical inputs would not change anything, so
            // hand the remaining failures straight to the CPU fallback.
            if (!have_mpi) return stats;

            // Backstop: if the previous widening did not reduce the uncertified count, more halo
            // will not either — the remaining cells are limited by something else. Bail out now
            // instead of spending the rest of MAX_WIDEN_ITERS on full-mesh rebuilds that cannot
            // change the outcome.
            if (iter > 0 && global_beyond >= prev_beyond) {
                logging::root() << "VORONOI: widening to W=" << stats.final_halo_width << " still leaves "
                                << global_beyond << " cell(s) uncertified (was " << prev_beyond
                                << ") — halo growth is not helping, handing them to the CPU fallback."
                                << std::endl;
                return stats;
            }
            prev_beyond = global_beyond;

            // last iter: hand off remaining failures to CPU fallback
            if (iter == MAX_WIDEN_ITERS - 1) {
                logging::root() << "VORONOI: halo widening hit MAX_ITERS=" << MAX_WIDEN_ITERS << " with "
                                << global_beyond
                                << " cell(s) still reaching beyond the rank data extent — falling through to "
                                   "CPU fallback."
                                << std::endl;
                return stats;
            }

            // widen the halo and retry. Logged because this costs a full mesh rebuild, and the
            // count says exactly why it is being paid.
            logging::root() << "VORONOI: " << global_beyond << " cell(s) reach beyond the rank data extent; "
                            << "widening halo W " << stats.final_halo_width << " -> "
                            << (stats.final_halo_width + 2) << std::endl;
            stats.final_halo_width += 2;
        }
        return stats; // unreachable
    }

    // CPU perturbation cascade with targeted cross-rank repair.
    //
    // Each round: (1) this rank repairs its own failed cells (perturb ladder + local symmetry
    // cascade); (2) one fused Allreduce establishes whether any moved seed anywhere is
    // exported, i.e. some other rank holds a ghost copy of it; (3) if so, exactly those
    // positions travel to exactly the ranks holding a copy (frozen full-halo layout — same
    // slots, same counts, no re-export), and each receiver rebuilds exactly the cells the
    // moved ghosts can influence, certified per cell by the stored security radius. A repair
    // can itself be forced to perturb further cells; those become the next round's work.
    //
    // This replaces a full cross-rank mesh rebuild per round (ghost regen + seed exchange +
    // KNN sort + all n_hydro cells — measured at 0.6 s/step/rank, 6.6% of runtime, on the
    // 2-node 360^3 KH benchmark) with work proportional to the handful of cells that
    // actually saw a moved seed. Seeds move at most by the perturbation delta, so pts_data
    // staleness is not an issue: the next step re-derives everything from mesh->seeds, which
    // every rebuild path keeps current at emit time.
    static void cpu_perturb_and_repair(VMesh* mesh, BuildStats& stats, double dt) {
        // Rounds are cheap now (no full rebuild), so the cap is generous; a chain that deep
        // means pathological geometry and gets reported loudly below.
        constexpr int MAX_CASCADE_ITERS = 8;
        const bool    have_mpi          = stats.have_mpi_neighbors;

        // seeds moved on this rank whose cross-rank effects are not yet resolved
        std::vector<int> pending;

        for (int iter = 0; iter < MAX_CASCADE_ITERS; iter++) {
            // local repair of this rank's failed cells; appends every permanently perturbed
            // cell (ladder + symmetry cascade) to `pending`. From iter 1 on this is a cheap
            // status sweep: the targeted repair below leaves no failed cells behind, so it
            // only fires if something regressed — defence in depth at one kernel's cost.
            int       local_num_failed = 0;
            const int local_perturbed  = cpu_fallback_failed_cells(mesh, &local_num_failed, dt, &pending);
            stats.cells_perturbed_total += local_perturbed;

            // dedupe + deterministic order (a cell can be re-perturbed across rounds)
            std::sort(pending.begin(), pending.end());
            pending.erase(std::unique(pending.begin(), pending.end()), pending.end());

            // Which pending seeds does some other rank hold a copy of? Scanning the frozen
            // export layout is ground truth — unlike a position-band test it cannot disagree
            // with the layout the ghosts were actually built from (a max-rung perturbation
            // can cross a decomposition bucket, which would fool the band test both ways).
            // A seed with no ghost copy anywhere needs nothing: the local symmetry cascade
            // already restored consistency around it.
            proteus_mpi::MovedExportLists lists;
            const int                     local_exported =
                have_mpi ? proteus_mpi::halo_collect_moved_exports(mesh, pending, &lists) : 0;

            // one fused Allreduce for convergence, diagnostics and the exported gate
            int local[3]  = {(int)pending.size(), local_num_failed, local_exported};
            int global[3] = {local[0], local[1], local[2]};
            if (have_mpi) sum_ints_across_ranks(local, global, 3);
            const int global_pending    = global[0];
            const int global_num_failed = global[1];
            const int global_exported   = global[2];

            if (global_num_failed > 0) {
                logging::root() << "VORONOI: fallback recovered " << global_num_failed << " cells globally (iter "
                                << iter << ")." << std::endl;
            }

            // converged: no seed anywhere moved without its ghost copies being repaired
            if (global_pending == 0) {
                stats.perturb_loop_iters_used = iter;
                if (iter > 0)
                    logging::root() << "VORONOI: perturbation cascade converged in " << iter << " round(s)."
                                    << std::endl;
                return;
            }
            stats.perturb_loop_iters_used = iter + 1;

            // every moved seed is interior to its own rank (single rank always lands here):
            // the local symmetry cascade already restored consistency, nothing to ship
            if (global_exported == 0) return;

            // ship exactly the moved positions to exactly the ranks holding a copy, and
            // repair around whatever arrives. Collective: every rank participates, with or
            // without moved seeds of its own. The repair's own perturbations (if any) become
            // the next round's pending set.
            std::vector<proteus_mpi::MovedSeed> received;
            {
                PROFILE("EXCHANGE");
                proteus_mpi::halo_exchange_moved_seeds(lists, &received);
            }
            pending.clear();
            {
                PROFILE("REPAIR");
                repair_cells_for_moved_ghosts(mesh, received, dt, &pending);
            }
        }

        // MAX_CASCADE_ITERS exhausted. The final repair round ran, but anything IT perturbed
        // was never tested against the export layout — if such a seed is exported, neighbour
        // ranks keep a stale ghost position for one step. Local scan only (no collective:
        // every rank exits this loop together, and the warning is per-rank information).
        if (!pending.empty()) {
            proteus_mpi::MovedExportLists tail;
            const int                     tail_exported = proteus_mpi::halo_collect_moved_exports(mesh, pending, &tail);
            if (tail_exported > 0) {
                std::cerr << "VORONOI: WARNING perturbation cascade hit MAX_ITERS=" << MAX_CASCADE_ITERS << " with "
                          << tail_exported << " exported seed(s) moved in the final repair round; neighbour "
                          << "ranks keep a stale ghost position for one step." << std::endl;
            }
        }
        logging::root() << "VORONOI: perturbation cascade hit MAX_ITERS=" << MAX_CASCADE_ITERS << "." << std::endl;
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
#ifdef VOL_REGULARIZE
        proteus_mpi::halo_exchange_volumes(mesh);
#endif
    }

    // update the remembered halo width across calls: ratchet up after widening, decay
    // back to base after several steady steps so a one-off spike does not stay forever
    static void adapt_halo_width(const BuildStats& stats) {
        if (!stats.have_mpi_neighbors) return;
        const int W_base = default_starting_halo_width();

        constexpr int STEADY_DECAY_THRESHOLD = 50;

        if (stats.widen_iters_used > 0) {
            // widening fired this step: latch the new width, reset the steady counter
            s_last_W       = std::max(s_last_W, stats.final_halo_width);
            s_steady_count = 0;
        } else {
            // no widening this step: count toward a slow decay back to W_base
            s_steady_count++;
            if (s_steady_count >= STEADY_DECAY_THRESHOLD && s_last_W > W_base) {
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
    static hsize_t exchange_seeds_across_ranks(VMesh*       mesh,
                                               POINT_TYPE*  pts_data,
                                               POINT_TYPE*& pts,
                                               hsize_t*&    original_ids,
                                               hsize_t      n_hydro,
                                               hsize_t      n_ghosts,
                                               int          W) {
        proteus_mpi::halo_build_exports(pts_data, (int)n_hydro, buff, W);
        // halo_build_exports may have grown n_mpi_capacity, which reallocates mesh->scratch_pts
        // and mesh->ghost_ids inside halo_grow_capacity, freeing the old buffers. Re-read the
        // live pointers so subsequent writes go to the new ones. pts/original_ids are references
        // so this also repairs the caller's copies, which it otherwise hands to compute_mesh.
        pts          = mesh->scratch_pts;
        original_ids = mesh->ghost_ids;
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
        PROFILE("REMAP");

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
    }

    // Has the widen loop converged? Also writes this rank's total failure count for the
    // post-loop perturbation cascade.
    //
    // The criterion is exactly the count of security_radius_beyond_data cells: those are the
    // cells whose bounding sphere reaches outside [data_lo, data_hi], which is the one and only
    // condition a wider halo repairs. Every other status is deliberately excluded --
    // security_radius_not_reached, the template overflows, and inconsistent_boundary /
    // needs_exact_predicates are all fixed downstream by the slow tier, the wide tier or the
    // perturb ladder, and none of them changes when W grows.
    //
    // This distinction is why the status split exists. When the criterion was the generic
    // security_radius_not_reached count, a single degenerate cell -- the kind the CPU fallback
    // repairs by perturbation in microseconds -- made the loop run to MAX_WIDEN_ITERS every
    // single step, rebuilding all n_hydro cells four times over for an identical result, while
    // adapt_halo_width ratcheted W up and the halo grew without bound.
    static bool widen_converged_across_ranks(VMesh* mesh, bool have_mpi, int* local_failed_out,
                                             int* global_beyond_out) {
        // all statuses — drives the post-loop perturb cascade
        *local_failed_out = count_local_failed_cells(mesh);

        // single rank: no halo to widen, and the extent check is disabled, so nothing to do
        if (!have_mpi) {
            *global_beyond_out = 0;
            return true;
        }

        const int local_beyond  = count_local_beyond_data_cells(mesh);
        int       global_beyond = local_beyond;
        sum_ints_across_ranks(&local_beyond, &global_beyond, 1);
        *global_beyond_out = global_beyond;
        return (global_beyond == 0);
    }

    // count cells whose status is not success (host-side scan). All failure modes counted.
    // Used to decide whether to fall through to the perturbation cascade after the widen loop.
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

    // count cells the build could not certify against this rank's data extent — the only
    // failure a wider halo can repair, and hence the widen-loop convergence criterion.
    static int count_local_beyond_data_cells(const VMesh* mesh) {
        int n_failed = 0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+ : n_failed)
#endif
        for (hsize_t k = 0; k < mesh->n_hydro; k++) {
            if (mesh->cell_status[k] == security_radius_beyond_data) n_failed++;
        }
        return n_failed;
    }

    // MPI_Allreduce SUM wrapper; no-op when MPI is off
    static void sum_ints_across_ranks(const int* local, int* global, int n) {
#ifdef USE_MPI
        PROFILE_MPI("ALLREDUCE");
        MPI_Allreduce(local, global, n, MPI_INT, MPI_SUM, proteus_mpi::decomp.cart_comm);
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

    // empirical startup margin on top of the buff-derived halo width, so the first iter is
    // wide enough that the completeness check usually passes immediately
    static int default_starting_halo_width() {
        constexpr int W_STARTUP_MARGIN = 2;
        return proteus_mpi::halo_default_width(buff) + W_STARTUP_MARGIN;
    }

    // physical extent of valid neighbour data this rank can see during the cell build:
    // own brick plus W buckets of halo, clamped to the extended [-buff, 1+buff]^d domain.
    // The fast-tier cell build uses this per-seed: a cell whose security sphere reaches
    // past these faces gets forced to security_radius_not_reached so the widen-W loop
    // iterates. A middle-of-brick cell with a large security radius stays "success"
    // because its sphere is fully contained in local data.
    // Single-rank: data_lo == data_hi (check disabled; periodic ghosts cover everything).
    // 2D: z extents stay at 0; the kernel skips the z face.
    static void set_data_extent_for_build(VMesh* mesh, int W, bool have_mpi) {
        if (!have_mpi || W <= 0) {
            for (int a = 0; a < 3; a++) {
                mesh->data_lo[a] = 0.0;
                mesh->data_hi[a] = 0.0;
            }
            return;
        }
        const double bs     = (1.0 + 2.0 * buff) / (double)proteus_mpi::decomp.N_grid_global;
        const double halo   = (double)W * bs;
        const double dom_lo = -buff;
        const double dom_hi = 1.0 + buff;
#ifdef dim_3D
        constexpr int n_ax = 3;
#else
        constexpr int n_ax = 2;
#endif
        for (int a = 0; a < n_ax; a++) {
            const double lo  = (double)proteus_mpi::decomp.b0[a] * bs - buff - halo;
            const double hi  = (double)proteus_mpi::decomp.b1[a] * bs - buff + halo;
            mesh->data_lo[a] = fmax(lo, dom_lo);
            mesh->data_hi[a] = fmin(hi, dom_hi);
        }
#ifndef dim_3D
        mesh->data_lo[2] = 0.0;
        mesh->data_hi[2] = 0.0;
#endif
    }

} // namespace voronoi
