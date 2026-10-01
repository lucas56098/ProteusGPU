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
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "alloc.cu"
#include "build.cu"
#include "cell.cu"
#include "fallback.cu"
#include "geometry.cu"
#include "reach.cu"

namespace voronoi {

    namespace {
        // what the build needed, for the summary line
        struct BuildStats {
            int local_failed_cells      = 0; // cells no GPU tier could build
            int global_failed_cells     = 0;
            int rounds                  = 0; // builds after the first one, for cells that asked for more
            int perturb_loop_iters_used = 0;
            int cells_perturbed_total   = 0; // cells with a moved seed
        };
    } // namespace

    static BuildStats build_mesh_by_requests(VMesh*                    mesh,
                                             POINT_TYPE*               cell_pos,
                                             hydro::primvars*          primvar,
                                             hydro::ConsVars*          cons,
                                             gradients::PrimGradients* grads);
    static void       cpu_perturb_and_repair(VMesh* mesh, BuildStats& stats, double dt);
    static void       exchange_ghost_geometry(VMesh* mesh);
    static void       print_step_summary(const BuildStats& stats);
    static int        sum_int_across_ranks(int local);
    static void       sum_ints_across_ranks(const int* local, int* global, int n);

    // the most rounds a build may take; every round doubles the balls that are still open
    constexpr int MAX_REQUEST_ROUNDS = 24;

    // builds the mesh, repairs what failed, sends the ghost state
    void compute_periodic_mesh(VMesh*                    mesh,
                               POINT_TYPE*               pts_data,
                               uint64_t                  num_points,
                               hydro::primvars*          primvar,
                               hydro::ConsVars*          cons,
                               gradients::PrimGradients* grads,
                               double                    dt) {
        PROFILE("MESH");

        // the cell positions of this build live in managed memory, in k-order after the first sort
        POINT_TYPE* cell_pos = mesh->scratch_move;
        mesh->n_hydro        = num_points;
        if (pts_data != cell_pos) gpu_memcpy(cell_pos, pts_data, num_points * sizeof(POINT_TYPE));

        BuildStats stats = build_mesh_by_requests(mesh, cell_pos, primvar, cons, grads);

        stats.global_failed_cells = logging::sum_global(stats.local_failed_cells);

        // cells no tier could build go to the CPU
        if (stats.global_failed_cells > 0) {
            PROFILE("PERTURB");
            cpu_perturb_and_repair(mesh, stats, dt);
        }

        // ghosts have their final position now
        exchange_ghost_geometry(mesh);
        print_step_summary(stats);
    }

    // the cell order first, then rounds of building and asking until every cell has what it needs
    static BuildStats build_mesh_by_requests(VMesh*                    mesh,
                                             POINT_TYPE*               cell_pos,
                                             hydro::primvars*          primvar,
                                             hydro::ConsVars*          cons,
                                             gradients::PrimGradients* grads) {
        const int  n_hydro = (int)mesh->n_hydro;
        BuildStats stats{};

        proteus_mpi::halo_begin_build();
        {
            PROFILE("ORDER");
            fix_cell_order(mesh, cell_pos, primvar, cons, grads);
        }

        // what every cell is guessed to reach, asked for where it leaves the rank or the box
        first_guess_balls(mesh, cell_pos);

        for (int round = 0;; round++) {
            stats.rounds = round;

            // point list: cells, then ghosts
            POINT_TYPE* pts      = mesh->scratch_pts;
            const int   n_ghosts = (int)proteus_mpi::halo.g_owner.size();
            mesh_ensure_ghost_capacity(mesh, (uint64_t)n_ghosts);
            pts = mesh->scratch_pts;
            gpu_memcpy(pts, cell_pos, (size_t)n_hydro * sizeof(POINT_TYPE));
            proteus_mpi::halo_write_ghosts(mesh, pts, mesh->ghost_ids, n_hydro);

            compute_mesh(mesh, pts, n_hydro + n_ghosts, round > 0);
            certify_cells(mesh, cell_pos);

            int       local_stuck = 0;
            const int nb          = request_open_balls(mesh, &local_stuck);
            const int open        = sum_int_across_ranks(nb);
            if (sum_int_across_ranks(local_stuck) > 0) {
                proteus_mpi::exit_failure("[rank %d] VORONOI: %d cell(s) reach more than half the box. Too few "
                                          "cells for this box.\n",
                                          proteus_mpi::rank(),
                                          local_stuck);
            }
            if (open == 0) break;
            if (round + 1 >= MAX_REQUEST_ROUNDS) {
                proteus_mpi::exit_failure(
                    "VORONOI: %d cell(s) still open after %d request rounds.\n", open, MAX_REQUEST_ROUNDS);
            }
            logging::root() << "VORONOI: " << open << " cell(s) ask for a larger ball, round " << (round + 1)
                            << std::endl;
            send_open_balls(mesh, cell_pos, nb);
        }

        stats.local_failed_cells = count_failed_cells(mesh);
        return stats;
    }

    // CPU fallback, and the rebuilds a moved seed causes here and on the ranks that hold it as a ghost
    static void cpu_perturb_and_repair(VMesh* mesh, BuildStats& stats, double dt) {
        constexpr int MAX_CASCADE_ITERS = 8;

        std::vector<int> pending;

        for (int iter = 0; iter < MAX_CASCADE_ITERS; iter++) {
            int       local_num_failed = 0;
            const int local_perturbed  = cpu_fallback_failed_cells(mesh, &local_num_failed, dt, &pending);
            stats.cells_perturbed_total += local_perturbed;

            // cells with a moved seed, each one once
            std::sort(pending.begin(), pending.end());
            pending.erase(std::unique(pending.begin(), pending.end()), pending.end());

            // a moved seed that another rank holds as a ghost has to be sent there
            const int local_exported = proteus_mpi::halo_count_moved_exports(pending);

            int local[3]  = {(int)pending.size(), local_num_failed, local_exported};
            int global[3] = {local[0], local[1], local[2]};
            sum_ints_across_ranks(local, global, 3);
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
                proteus_mpi::halo_exchange_moved_seeds(mesh, pending, &received);
            }
            pending.clear();
            {
                PROFILE("REPAIR");
                repair_cells_for_moved_ghosts(mesh, received, dt, &pending);
            }
        }

        if (proteus_mpi::halo_count_moved_exports(pending) > 0) {
            proteus_mpi::exit_failure("[rank %d] VORONOI: perturbation cascade did not converge in %d rounds: "
                                      "exported seed(s) moved in the last round, other ranks still hold the old "
                                      "position. Aborting.\n",
                                      proteus_mpi::rank(),
                                      MAX_CASCADE_ITERS);
        }
        logging::root() << "VORONOI: perturbation cascade hit MAX_ITERS=" << MAX_CASCADE_ITERS << "." << std::endl;
    }

    // finds the ghosts a cell really uses and sends their geometry; their state is up to the caller
    static void exchange_ghost_geometry(VMesh* mesh) {
        proteus_mpi::halo_build_used_subset(mesh);
        proteus_mpi::halo_exchange_centroids(mesh);
#ifdef VOL_REGULARIZE
        proteus_mpi::halo_exchange_volumes(mesh);
#endif
    }

    // one line with rounds, retries, ghosts and migration
    static void print_step_summary(const BuildStats& stats) {
        const int rounds_global   = logging::max_global(stats.rounds);
        const int cascade_global  = logging::max_global(stats.perturb_loop_iters_used);
        const int perturbed_total = logging::sum_global(stats.cells_perturbed_total);

        if (rounds_global > 0 || cascade_global > 0 || perturbed_total > 0) {
            logging::root() << "VORONOI: retries rounds=" << rounds_global << " cascade=" << cascade_global
                            << " perturbed=" << perturbed_total << std::endl;
        }

        const int ghosts_g   = logging::sum_global((int)proteus_mpi::halo.g_owner.size());
        const int mpi_g      = logging::sum_global(proteus_mpi::halo.n_mpi_ghosts);
        const int used_g     = logging::sum_global(proteus_mpi::halo.n_used_recv);
        const int migrated_g = logging::sum_global(proteus_mpi::last_n_migrated());
        if (mpi_g > 0 || migrated_g > 0) {
            logging::root() << "MPI: ghosts=" << ghosts_g << " from other ranks=" << mpi_g << " used=" << used_g
                            << "  migrated=" << migrated_g << std::endl;
        }
    }

    static int sum_int_across_ranks(int local) {
        int global = local;
        sum_ints_across_ranks(&local, &global, 1);
        return global;
    }

    static void sum_ints_across_ranks(const int* local, int* global, int n) {
#ifdef USE_MPI
        PROFILE_MPI("ALLREDUCE");
        MPI_Allreduce(local, global, n, MPI_INT, MPI_SUM, proteus_mpi::decomp.comm);
#else
        for (int i = 0; i < n; i++)
            global[i] = local[i];
#endif
    }

} // namespace voronoi
