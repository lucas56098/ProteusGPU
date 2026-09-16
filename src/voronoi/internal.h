#ifndef VORONOI_INTERNAL_H
#define VORONOI_INTERNAL_H

// What the parts of voronoi.cu call in each other.

#include <vector>

#include "../mpi/halo.h"
#include "voronoi.h"

namespace voronoi {

    // one build round
    void compute_mesh(VMesh*           mesh,
                      POINT_TYPE*      pts_data,
                      int              n_total,
                      hydro::primvars* primvar,
                      hydro::primvars* primvar_aux,
                      int              iter = 0);

    // rebuild failed cells on the CPU, returns the number of moved seeds
    int cpu_fallback_failed_cells(VMesh*            mesh,
                                  int*              num_failed_out,
                                  double            dt,
                                  std::vector<int>* perturbed_ks_out = nullptr);

    // rebuild after a neighbour rank moved a seed
    int repair_cells_for_moved_ghosts(VMesh*                                     mesh,
                                      const std::vector<proteus_mpi::MovedSeed>& moved,
                                      double                                     dt,
                                      std::vector<int>*                          newly_perturbed_out);

    // copy the seeds into pts and add the periodic ghosts
    uint64_t regenerate_periodic_ghosts(
        uint64_t n_hydro, const POINT_TYPE* pts_data, POINT_TYPE* pts, uint64_t* original_ids, double buff_val);

} // namespace voronoi

#endif
