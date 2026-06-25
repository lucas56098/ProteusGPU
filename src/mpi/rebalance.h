#ifndef MPI_REBALANCE_H
#define MPI_REBALANCE_H
#pragma once

#include "../global/gpu_compat.h"
#include "mpi_compat.h"

struct VMesh;

// Cost-weighted per-axis brick rebalance, folded into move_mesh.
// Every step the imbalance probe logs max(n_hydro) / mean(n_hydro) across ranks
// (one fused Allreduce — cheap). Every REBALANCE_INTERVAL steps move_mesh calls
// rebalance_decide between advance_seeds_by_dt and the mesh rebuild:
//   1. gather global marginal histograms of cells per bucket along each axis
//      (from the post-advance positions in scratch_move)
//   2. place new split lines so each axis-slab carries equal total cells
//   3. install the new splits via decomp_apply_splits
//   4. signal the caller to invoke migrate_for_rebalance (full Alltoallv over cart_comm)
// The single compute_periodic_mesh later in move_mesh then builds the new mesh
// on both rebalance-migrated and regularly-drifting cells in one pass.

namespace proteus_mpi {

    // one fused 2-int Allreduce + a logging printf. No mesh mutation.
    void rebalance_imbalance_log(int step, VMesh* mesh);

    // Called inside voronoi::move_mesh after advance_seeds_by_dt with the
    // post-advance positions in pts. Computes new splits and installs them
    // via decomp_apply_splits. Returns true iff splits changed — caller
    // must then run migrate_for_rebalance instead of the per-step migrate_seeds.
    // No-op (returns false) when rebalance is disabled, when step is not a
    // multiple of rebalance_interval, or when nranks == 1.
    bool rebalance_decide(int step, VMesh* mesh, POINT_TYPE* pts);

    // Called from voronoi::move_mesh immediately after migrate_for_rebalance,
    // on rebalance steps where rebalance_decide returned true. Emits the
    // post-rebalance summary line (pre→post imbalance and migrated count).
    void rebalance_log_after_migration(VMesh* mesh);

} // namespace proteus_mpi

#endif // MPI_REBALANCE_H
