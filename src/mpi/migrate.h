#ifndef MPI_MIGRATE_H
#define MPI_MIGRATE_H
#pragma once

// Moves cells that left the brick of this rank to the rank that owns them now.

#include "global/gpu_compat.h"
#include "mpi_compat.h"

struct VMesh;
namespace hydro {
    struct primvars;
}

namespace proteus_mpi {

    void migrate_init(int n_local_initial);

    // per step: cells can only move to a Cartesian neighbour
    void migrate_seeds(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new);

    // after new splits: a cell can land on any rank
    void migrate_for_rebalance(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new);

    int last_n_migrated();

} // namespace proteus_mpi

#endif
