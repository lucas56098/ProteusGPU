#ifndef VORONOI_INTERNAL_H
#define VORONOI_INTERNAL_H

// What the parts of voronoi.cu call in each other.

#include <vector>

#include "../mpi/halo.h"
#include "voronoi.h"

namespace voronoi {

    // sorts the cells alone into k-order; the persistent state and cell_pos follow
    void fix_cell_order(VMesh*                    mesh,
                        POINT_TYPE*               cell_pos,
                        hydro::primvars*          primvar,
                        hydro::ConsVars*          cons,
                        gradients::PrimGradients* grads);

    // one build round over the cells and the ghosts behind them in pts; only_open keeps the finished cells
    void compute_mesh(VMesh* mesh, POINT_TYPE* pts, int n_total, bool only_open);

    // first guess of every cell's reach; the cells whose guess leaves the rank or the box ask for it. Collective
    void first_guess_balls(VMesh* mesh, const POINT_TYPE* cell_pos);

    // whether the sphere of a cell lies in a ball it asked for (centred where the seed was then) or in what this
    // rank holds anyway
    HD inline bool sphere_is_covered(const POINT_TYPE& seed,
                                     const POINT_TYPE& ball_centre,
                                     double            sec_d2,
                                     double            req_r2,
                                     const uint64_t*   cuts,
                                     int               nranks,
                                     int               me);

    // a finished cell whose sphere is not covered becomes security_radius_beyond_data
    void certify_cells(VMesh* mesh, const POINT_TYPE* cell_pos);

    // the next ball of every cell left open; returns how many ask, stuck how many cannot grow any more
    int request_open_balls(VMesh* mesh, int* stuck);

    // sends the balls request_open_balls collected; collective
    void send_open_balls(VMesh* mesh, const POINT_TYPE* cell_pos, int nb);

    // failed cells whose CPU build reaches past what they asked for, with the squared radius they need
    void fallback_needs(VMesh* mesh, std::vector<int>* cells, std::vector<double>* need_d2);

    // rebuild failed cells on the CPU, returns the number of moved seeds
    int cpu_fallback_failed_cells(VMesh*            mesh,
                                  int*              num_failed_out,
                                  double            dt,
                                  std::vector<int>* perturbed_ks_out = nullptr);

    // rebuild after another rank moved a seed
    int repair_cells_for_moved_ghosts(VMesh*                                     mesh,
                                      const std::vector<proteus_mpi::MovedSeed>& moved,
                                      double                                     dt,
                                      std::vector<int>*                          newly_perturbed_out);

} // namespace voronoi

#endif
