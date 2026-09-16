
// mesh build for one step (voronoi.h)

#include "../global/allvars.h"
#include "../global/structs.h"
#include "../io/input.h"
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

    namespace {
        // what the build needed, for the summary line
        struct BuildStats {
            int  local_failed_cells      = 0; // cells no GPU tier could build
            int  global_failed_cells     = 0;
            int  widen_iters_used        = 0;
            int  perturb_loop_iters_used = 0;
            int  cells_perturbed_total   = 0; // cells with a moved seed
            int  final_halo_width        = 0;
            bool have_mpi_neighbors      = false;
        };
    } // namespace

    static BuildStats build_mesh_growing_halo(
        VMesh* mesh, POINT_TYPE* pts_data, uint64_t n_hydro, hydro::primvars* primvar, hydro::primvars* primvar_aux);
    static void cpu_perturb_and_repair(VMesh* mesh, BuildStats& stats, double dt);
    static void exchange_used_ghost_primvars(VMesh* mesh, hydro::primvars* primvar);
    static void adapt_halo_width(const BuildStats& stats);
    static void print_step_summary(const BuildStats& stats);

    static uint64_t exchange_seeds_across_ranks(VMesh*       mesh,
                                                POINT_TYPE*  pts_data,
                                                POINT_TYPE*& pts,
                                                uint64_t*&   original_ids,
                                                uint64_t     n_hydro,
                                                uint64_t     n_ghosts,
                                                int          W);
    static void     record_mpi_ghost_indices(uint64_t* original_ids, uint64_t n_hydro, uint64_t n_ghosts);
    static void     remap_exports_and_pts(VMesh* mesh, POINT_TYPE* pts_data, uint64_t n_hydro);
    static bool widen_converged_across_ranks(VMesh* mesh, bool have_mpi, int* local_failed_out, int* global_beyond_out);
    static int  count_local_failed_cells(const VMesh* mesh);
    static int  count_local_beyond_data_cells(const VMesh* mesh);
    static void sum_ints_across_ranks(const int* local, int* global, int n);
    static void check_ghost_count(uint64_t n_ghosts, uint64_t max_ghosts);
    static int  default_starting_halo_width();
    static void set_data_extent_for_build(VMesh* mesh, int W, bool have_mpi);

    static int s_last_W       = 0; // halo width of the last builds
    static int s_steady_count = 0; // steps in a row without widening

    // builds the mesh, repairs what failed, sends the ghost state
    void compute_periodic_mesh(VMesh*           mesh,
                               POINT_TYPE*      pts_data,
                               uint64_t         num_points,
                               hydro::primvars* primvar,
                               hydro::primvars* primvar_aux,
                               double           dt) {
        PROFILE("MESH");

        BuildStats stats = build_mesh_growing_halo(mesh, pts_data, num_points, primvar, primvar_aux);

        stats.global_failed_cells = logging::sum_global(stats.local_failed_cells);

        // cells no tier could build go to the CPU
        if (stats.global_failed_cells > 0) {
            PROFILE("PERTURB");
            cpu_perturb_and_repair(mesh, stats, dt);
        }

        // ghosts have their final position now
        exchange_used_ghost_primvars(mesh, primvar);
        adapt_halo_width(stats);
        print_step_summary(stats);
    }

    // builds all cells, with a wider halo each round while cells reach past the rank data
    static BuildStats build_mesh_growing_halo(
        VMesh* mesh, POINT_TYPE* pts_data, uint64_t n_hydro, hydro::primvars* primvar, hydro::primvars* primvar_aux) {
        constexpr int MAX_WIDEN_ITERS = 4;

        // upper bound for the periodic copies
        const double   ghost_frac = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        const uint64_t max_ghosts = (uint64_t)(2.0 * ghost_frac * n_hydro) + 1;
        const bool     have_mpi   = proteus_mpi::halo.n_neighbors > 0;

        BuildStats stats{};
        stats.have_mpi_neighbors = have_mpi;
        stats.final_halo_width   = have_mpi ? std::max(default_starting_halo_width(), s_last_W) : 0;

        // uncertified cells of the round before
        int prev_beyond = INT_MAX;
        for (int iter = 0; iter < MAX_WIDEN_ITERS; iter++) {
            stats.widen_iters_used = iter;

            // point list: cells, periodic ghosts, MPI ghosts
            POINT_TYPE* pts          = mesh->scratch_pts;
            uint64_t*   original_ids = mesh->ghost_ids;

            uint64_t n_ghosts;
            {
                PROFILE("GHOSTS");
                n_ghosts = regenerate_periodic_ghosts(n_hydro, pts_data, pts, original_ids, buff);
            }
            check_ghost_count(n_ghosts, max_ghosts);
            const uint64_t n_mpi =
                have_mpi ? exchange_seeds_across_ranks(
                               mesh, pts_data, pts, original_ids, n_hydro, n_ghosts, stats.final_halo_width)
                         : 0;

            mesh->n_mpi_ghosts = proteus_mpi::halo.n_mpi_ghosts;
            set_data_extent_for_build(mesh, stats.final_halo_width, have_mpi);
            compute_mesh(mesh, pts, (int)(n_hydro + n_ghosts + n_mpi), primvar, primvar_aux, iter);
            // the first round sorted the cells, the exports follow
            if (iter == 0 && have_mpi) remap_exports_and_pts(mesh, pts_data, n_hydro);

            int        local_failed = 0, global_beyond = 0;
            const bool converged     = widen_converged_across_ranks(mesh, have_mpi, &local_failed, &global_beyond);
            stats.local_failed_cells = local_failed;

            if (converged) {
                if (iter > 0)
                    logging::root() << "VORONOI: halo widening converged in " << (iter + 1) << " iteration(s)."
                                    << std::endl;
                return stats;
            }

            if (!have_mpi) return stats;

            // widening does not help any more
            if (iter > 0 && global_beyond >= prev_beyond) {
                logging::root() << "VORONOI: widening to W=" << stats.final_halo_width << " still leaves "
                                << global_beyond << " cell(s) uncertified (was " << prev_beyond
                                << ") — halo growth is not helping, handing them to the CPU fallback." << std::endl;
                return stats;
            }
            prev_beyond = global_beyond;

            if (iter == MAX_WIDEN_ITERS - 1) {
                logging::root() << "VORONOI: halo widening hit MAX_ITERS=" << MAX_WIDEN_ITERS << " with "
                                << global_beyond
                                << " cell(s) still reaching beyond the rank data extent — falling through to "
                                   "CPU fallback."
                                << std::endl;
                return stats;
            }

            logging::root() << "VORONOI: " << global_beyond << " cell(s) reach beyond the rank data extent; "
                            << "widening halo W " << stats.final_halo_width << " -> " << (stats.final_halo_width + 2)
                            << std::endl;
            stats.final_halo_width += 2;
        }
        return stats;
    }

    // CPU fallback, and the rebuilds a moved seed causes here and on the neighbour ranks
    static void cpu_perturb_and_repair(VMesh* mesh, BuildStats& stats, double dt) {
        constexpr int MAX_CASCADE_ITERS = 8;
        const bool    have_mpi          = stats.have_mpi_neighbors;

        std::vector<int> pending;

        for (int iter = 0; iter < MAX_CASCADE_ITERS; iter++) {
            int       local_num_failed = 0;
            const int local_perturbed  = cpu_fallback_failed_cells(mesh, &local_num_failed, dt, &pending);
            stats.cells_perturbed_total += local_perturbed;

            // cells with a moved seed, each one once
            std::sort(pending.begin(), pending.end());
            pending.erase(std::unique(pending.begin(), pending.end()), pending.end());

            proteus_mpi::MovedExportLists lists;
            // a moved seed that a neighbour holds as a ghost has to be sent there
            const int local_exported = have_mpi ? proteus_mpi::halo_collect_moved_exports(mesh, pending, &lists) : 0;

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

            if (global_pending == 0) {
                stats.perturb_loop_iters_used = iter;
                if (iter > 0)
                    logging::root() << "VORONOI: perturbation cascade converged in " << iter << " round(s)."
                                    << std::endl;
                return;
            }
            stats.perturb_loop_iters_used = iter + 1;

            if (global_exported == 0) return;

            std::vector<proteus_mpi::MovedSeed> received;
            {
                PROFILE("EXCHANGE");
                // and their moved seeds come back
                proteus_mpi::halo_exchange_moved_seeds(lists, &received);
            }
            pending.clear();
            {
                PROFILE("REPAIR");
                repair_cells_for_moved_ghosts(mesh, received, dt, &pending);
            }
        }

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

    // sends the state of the ghosts that a cell really uses
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

    // keeps the width of the last builds, drops a layer after 50 steps without widening
    static void adapt_halo_width(const BuildStats& stats) {
        if (!stats.have_mpi_neighbors) return;
        const int W_base = default_starting_halo_width();

        constexpr int STEADY_DECAY_THRESHOLD = 50;

        if (stats.widen_iters_used > 0) {
            s_last_W       = std::max(s_last_W, stats.final_halo_width);
            s_steady_count = 0;
        } else {
            s_steady_count++;
            if (s_steady_count >= STEADY_DECAY_THRESHOLD && s_last_W > W_base) {
                s_last_W       = std::max(W_base, s_last_W - 1);
                s_steady_count = 0;
            }
        }
    }

    // one line with retries, halo traffic and migration
    static void print_step_summary(const BuildStats& stats) {
        const int widen_global    = logging::max_global(stats.widen_iters_used);
        const int cascade_global  = logging::max_global(stats.perturb_loop_iters_used);
        const int perturbed_total = logging::sum_global(stats.cells_perturbed_total);

        if (widen_global > 0 || cascade_global > 0 || perturbed_total > 0) {
            logging::root() << "VORONOI: retries widen=" << widen_global << " cascade=" << cascade_global
                            << " perturbed=" << perturbed_total << std::endl;
        }

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

    // exports the seeds at the brick edge, takes the neighbours' ones as ghosts
    static uint64_t exchange_seeds_across_ranks(VMesh*       mesh,
                                                POINT_TYPE*  pts_data,
                                                POINT_TYPE*& pts,
                                                uint64_t*&   original_ids,
                                                uint64_t     n_hydro,
                                                uint64_t     n_ghosts,
                                                int          W) {
        proteus_mpi::halo_build_exports(pts_data, (int)n_hydro, buff, W);
        pts          = mesh->scratch_pts;
        original_ids = mesh->ghost_ids;
        proteus_mpi::halo_exchange_seeds(mesh, pts, (int)(n_hydro + n_ghosts));
        record_mpi_ghost_indices(original_ids, n_hydro, n_ghosts);
        return (uint64_t)proteus_mpi::halo.n_mpi_ghosts;
    }

    // an MPI ghost has no cell here, its entry is the neighbour index n_hydro + slot
    static void record_mpi_ghost_indices(uint64_t* original_ids, uint64_t n_hydro, uint64_t n_ghosts) {
        for (int n = 0; n < proteus_mpi::halo.n_neighbors; n++) {
            const int ghost_off = proteus_mpi::halo.ghost_offset[n];
            for (int j = 0; j < proteus_mpi::halo.recv_count[n]; j++) {
                const int slot                          = ghost_off + j;
                const int ext_k                         = (int)n_hydro + slot;
                original_ids[n_ghosts + (uint64_t)slot] = (uint64_t)ext_k;
            }
        }
    }

    // the build sorted the cells, so export lists and input positions follow
    static void remap_exports_and_pts(VMesh* mesh, POINT_TYPE* pts_data, uint64_t n_hydro) {
        PROFILE("REMAP");

        // cell index of every input point
        static std::vector<unsigned int> inv_gather;
        inv_gather.resize((size_t)n_hydro);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (uint64_t new_k = 0; new_k < n_hydro; new_k++) {
            inv_gather[mesh->gather_perm[new_k]] = (unsigned int)new_k;
        }
        proteus_mpi::halo_remap_export_indices(inv_gather.data(), (int)n_hydro);

        static std::vector<POINT_TYPE> pts_scratch;
        pts_scratch.resize((size_t)n_hydro);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (uint64_t new_k = 0; new_k < n_hydro; new_k++) {
            pts_scratch[new_k] = pts_data[mesh->gather_perm[new_k]];
        }
        std::memcpy(pts_data, pts_scratch.data(), (size_t)n_hydro * sizeof(POINT_TYPE));

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (uint64_t k = 0; k < n_hydro; k++)
            mesh->orig_to_k_save[k] = (unsigned int)k;
    }

    // done when no rank has a cell left that reaches past its data
    static bool
    widen_converged_across_ranks(VMesh* mesh, bool have_mpi, int* local_failed_out, int* global_beyond_out) {
        *local_failed_out = count_local_failed_cells(mesh);

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

    static int count_local_failed_cells(const VMesh* mesh) {
        const Status* stat = mesh->cell_status;
        return parallel_reduce_sum<_MESH_BLOCK_SIZE_, int>(
            "COUNT_FAILED", mesh->n_hydro, [=] HD(size_t k) { return (stat[k] != success) ? 1 : 0; });
    }

    static int count_local_beyond_data_cells(const VMesh* mesh) {
        const Status* stat = mesh->cell_status;
        return parallel_reduce_sum<_MESH_BLOCK_SIZE_, int>("COUNT_BEYOND", mesh->n_hydro, [=] HD(size_t k) {
            return (stat[k] == security_radius_beyond_data) ? 1 : 0;
        });
    }

    static void sum_ints_across_ranks(const int* local, int* global, int n) {
#ifdef USE_MPI
        PROFILE_MPI("ALLREDUCE");
        MPI_Allreduce(local, global, n, MPI_INT, MPI_SUM, proteus_mpi::decomp.cart_comm);
#else
        for (int i = 0; i < n; i++)
            global[i] = local[i];
#endif
    }

    // the ghost array is sized from an estimate
    static void check_ghost_count(uint64_t n_ghosts, uint64_t max_ghosts) {
        if (n_ghosts > max_ghosts) {
            proteus_mpi::exit_failure("VORONOI: Error! ghost count %llu exceeds estimated max %llu. Distribution "
                                      "is highly non-uniform.\n",
                                      (unsigned long long)n_ghosts,
                                      (unsigned long long)max_ghosts);
        }
    }

    // start width: what the band needs plus a margin
    static int default_starting_halo_width() {
        constexpr int W_STARTUP_MARGIN = 2;
        return proteus_mpi::halo_default_width(buff) + W_STARTUP_MARGIN;
    }

    // brick plus W bucket layers, cut at the box edge
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
