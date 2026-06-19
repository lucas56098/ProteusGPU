#ifndef VORONOI_INTERNAL_H
#define VORONOI_INTERNAL_H

#include "voronoi.h"

namespace voronoi {

    // build the Voronoi cells from the current seeds + ghosts buffer. iter == 0 runs the
    // full pipeline (atomic-counter pass1 + save orig_to_k + permute primvar); iter > 0
    // reuses the saved orig_to_k mapping and skips the primvar permute.
    void compute_mesh(VMesh*           mesh,
                      POINT_TYPE*      pts_data,
                      int              n_total,
                      hydro::primvars* primvar,
                      hydro::primvars* primvar_aux,
                      int              iter = 0);

    // retry geometrically failed cells on the CPU with hash-based seed perturbation and a
    // symmetry cascade across affected neighbors. Returns the number of cells perturbed;
    // caller uses this to decide whether a cross-rank halo re-export + rebuild is needed.
    // *num_failed_out (may be null) receives this rank's failed-cell count so the caller
    // can sum it across ranks for a global diagnostic.
    // `dt` is the step length used to advance the seeds; perturbed cells need their
    // v_mesh shifted by delta/dt so face velocities match the perturbed geometry.
    // Pass 0.0 when there is no associated step (initial mesh build).
    int cpu_fallback_failed_cells(VMesh* mesh, int* num_failed_out, double dt);

    // emit periodic ghost copies for cells whose seeds lie within `buff_val` of the box
    // boundary in any dimension. Writes to `pts` (reals first, ghosts after) and
    // `original_ids` (ghost slot -> source real index). Returns the ghost count.
    hsize_t regenerate_periodic_ghosts(
        hsize_t n_hydro, const POINT_TYPE* pts_data, POINT_TYPE* pts, hsize_t* original_ids, double buff_val);

} // namespace voronoi

#endif // VORONOI_INTERNAL_H
