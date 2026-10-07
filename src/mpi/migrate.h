#ifndef MPI_MIGRATE_H
#define MPI_MIGRATE_H
#pragma once

// Moves cells that left the part of the curve this rank owns to the rank that owns them now.

#include "global/gpu_compat.h"
#include "mpi_compat.h"

struct VMesh;
namespace hydro {
    struct primvars;
    struct ConsVars;
} // namespace hydro
namespace gradients {
    struct PrimGradients;
}

namespace mpi {

    void migrate_free();

    // every cell goes to the owner of its new position, which can be any rank
    void migrate_cells(VMesh* mesh, hydro::primvars* primvar, hydro::ConsVars* cons, gradients::PrimGradients* grads);

    int last_n_migrated();

} // namespace mpi

#endif
