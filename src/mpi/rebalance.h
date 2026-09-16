#ifndef MPI_REBALANCE_H
#define MPI_REBALANCE_H
#pragma once

// Moves the brick borders when the cells are spread unevenly over the ranks.

#include "../global/gpu_compat.h"
#include "mpi_compat.h"

struct VMesh;

namespace proteus_mpi {

    // prints the imbalance every imbalance_log_interval steps
    void rebalance_imbalance_log(int step, VMesh* mesh);

    // true when new splits were applied, then the cells have to migrate
    bool rebalance_decide(int step, VMesh* mesh, POINT_TYPE* pts);

    void rebalance_log_after_migration(VMesh* mesh);

} // namespace proteus_mpi

#endif
