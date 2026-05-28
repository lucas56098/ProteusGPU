#ifndef MPI_MIGRATE_H
#define MPI_MIGRATE_H
#pragma once

#include "global/gpu_compat.h"
#include "mpi_compat.h"

// Mid-step seed migration. Called from voronoi::move_mesh between the position
// update (writes new seeds into mesh->scratch_move) and compute_periodic_mesh.
// Cells whose new bucket is owned by a different rank are shipped and appended
// to the receiver's SoA arrays; outgoing cells are removed via swap-with-last.
// Migrated payload per cell: new position, primvar(rho,v,E), prim_new(rho,v,E),
// v_mesh, old_volume. Capacity bound: n_hydro_new <= max_n_local(n_initial).

struct VMesh;
namespace hydro {
    struct primvars;
}

namespace proteus_mpi {

// captures n_initial for the migration capacity check; single-rank: no-op
void migrate_init(int n_local_initial);

// main entry, called from voronoi::move_mesh; updates mesh->n_hydro in place
void migrate_seeds(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new);

// number of cells this rank sent in the most recent migrate_seeds call (0 before any call)
int last_n_migrated();

}  // namespace proteus_mpi

#endif  // MPI_MIGRATE_H
