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
            int cpu_cells = 0; // cells no GPU tier could build, built on the CPU
            int rounds    = 0; // builds after the first one, for cells that asked for more
        };
    } // namespace

    static BuildStats build_mesh_by_requests(VMesh*                    mesh,
                                             POINT_TYPE*               cell_pos,
                                             hydro::primvars*          primvar,
                                             hydro::ConsVars*          cons,
                                             gradients::PrimGradients* grads);
    static void       exchange_ghost_geometry(VMesh* mesh);
    static void       print_step_summary(const BuildStats& stats);
    static void       sum_ints_across_ranks(const int* local, int* global, int n);

    // the most rounds a build may take; every round doubles the balls that are still open
    constexpr int MAX_REQUEST_ROUNDS = 24;

    // builds the mesh, the CPU builds what the GPU could not, sends the ghost geometry
    void compute_periodic_mesh(VMesh*                    mesh,
                               POINT_TYPE*               pts_data,
                               uint64_t                  num_points,
                               hydro::primvars*          primvar,
                               hydro::ConsVars*          cons,
                               gradients::PrimGradients* grads) {
        PROFILE("MESH");

        // the cell positions of this build live in managed memory, in k-order after the first sort
        POINT_TYPE* cell_pos = mesh->scratch_move;
        mesh->n_hydro        = num_points;
        if (pts_data != cell_pos) gpu_memcpy(cell_pos, pts_data, num_points * sizeof(POINT_TYPE));

        const BuildStats stats = build_mesh_by_requests(mesh, cell_pos, primvar, cons, grads);

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

        mpi::halo_begin_build();
        take_cpu_built();
        {
            PROFILE("ORDER");
            fix_cell_order(mesh, cell_pos, primvar, cons, grads);
        }

        // what every cell is guessed to reach, asked for where it leaves the rank or the box
        first_guess_balls(mesh, cell_pos);

        // point list: the cells, then the ghosts of every round behind them
        gpu_memcpy(mesh->scratch_pts, cell_pos, (size_t)n_hydro * sizeof(POINT_TYPE));

        int n_ghosts_built = -1;
        for (int round = 0;; round++) {
            stats.rounds = round;

            // a round that brought this rank no new ghost would build the same cells again
            const int n_ghosts = mpi::halo.n_ghosts;
            if (n_ghosts != n_ghosts_built) {
                mesh_ensure_ghost_capacity(mesh, (uint64_t)n_ghosts);
                POINT_TYPE* pts = mesh->scratch_pts;
                mpi::halo_write_ghosts(mesh, pts);

                compute_mesh(mesh, pts, n_hydro + n_ghosts, round > 0);
                n_ghosts_built = n_ghosts;
            } else {
                reopen_uncertified_cells(mesh);
            }
            certify_cells(mesh, cell_pos);

            // cells that ask, and cells that cannot grow any more
            int local[2]  = {0, 0};
            local[0]      = request_open_balls(mesh, &local[1]);
            int global[2] = {local[0], local[1]};
            sum_ints_across_ranks(local, global, 2);
            const int nb   = local[0];
            const int open = global[0];
            if (global[1] > 0) {
                mpi::exit_failure("[rank %d] VORONOI: %d cell(s) reach more than half the box. Too few "
                                  "cells for this box.\n",
                                  mpi::rank(),
                                  local[1]);
            }
            if (open == 0) break;
            if (round + 1 >= MAX_REQUEST_ROUNDS) {
                mpi::exit_failure(
                    "VORONOI: %d cell(s) still open after %d request rounds.\n", open, MAX_REQUEST_ROUNDS);
            }
            logging::root() << "VORONOI: " << open << " cell(s) ask for a larger ball, round " << (round + 1)
                            << std::endl;
            send_open_balls(mesh, cell_pos, nb);
        }

        // the CPU built and wrote what the GPU could not while the rounds went on
        stats.cpu_cells     = take_cpu_built();
        const int n_unbuilt = count_failed_cells(mesh);
        if (n_unbuilt > 0) {
            mpi::exit_failure(
                "[rank %d] VORONOI: %d cell(s) left unbuilt after the request rounds.\n", mpi::rank(), n_unbuilt);
        }
        return stats;
    }

    // finds the ghosts a cell really uses and sends their geometry; their state is up to the caller
    static void exchange_ghost_geometry(VMesh* mesh) {
        mpi::halo_build_used_subset(mesh);
        mpi::exchange_centroids(mesh);
#ifdef VOL_REGULARIZE
        mpi::exchange_volumes(mesh);
#endif
    }

    // one line with rounds, ghosts and migration
    static void print_step_summary(const BuildStats& stats) {
        const int       rounds_global = logging::max_global(stats.rounds);
        const long long cpu_global    = logging::sum_global((long long)stats.cpu_cells);
        if (rounds_global > 0 || cpu_global > 0) {
            logging::root() << "VORONOI: request rounds=" << rounds_global << "  built on the CPU=" << cpu_global
                            << std::endl;
        }

        const long long ghosts_g   = logging::sum_global((long long)mpi::halo.n_ghosts);
        const long long mpi_g      = logging::sum_global((long long)mpi::halo.n_mpi_ghosts);
        const long long used_g     = logging::sum_global((long long)mpi::halo.state_recv.total);
        const long long migrated_g = logging::sum_global((long long)mpi::last_n_migrated());
        if (mpi_g > 0 || migrated_g > 0) {
            logging::root() << "MPI: ghosts=" << ghosts_g << " from other ranks=" << mpi_g << " used=" << used_g
                            << "  migrated=" << migrated_g << std::endl;
        }
    }

    static void sum_ints_across_ranks(const int* local, int* global, int n) {
#ifdef USE_MPI
        PROFILE_MPI("ALLREDUCE");
        MPI_Allreduce(local, global, n, MPI_INT, MPI_SUM, mpi::decomp.comm);
#else
        for (int i = 0; i < n; i++)
            global[i] = local[i];
#endif
    }

} // namespace voronoi
