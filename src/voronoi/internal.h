#ifndef VORONOI_INTERNAL_H
#define VORONOI_INTERNAL_H

// What the parts of voronoi.cu call in each other.

#include <vector>

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

    // whether the sphere of a cell lies in the ball it asked for or in what this rank holds anyway
    HD inline bool
    sphere_is_covered(const POINT_TYPE& seed, double sec_d2, double req_r2, const uint64_t* cuts, int nranks, int me);

    // a finished cell whose sphere is not covered becomes security_radius_beyond_data
    void certify_cells(VMesh* mesh, const POINT_TYPE* cell_pos);

    // cells left open only by certify_cells count as finished again, for a round without new points
    void reopen_uncertified_cells(VMesh* mesh);

    // the next ball of every cell left open; returns how many ask, stuck how many cannot grow any more
    int request_open_balls(VMesh* mesh, int* stuck);

    // sends the balls request_open_balls collected; collective
    void send_open_balls(VMesh* mesh, const POINT_TYPE* cell_pos, int nb);

    // builds the cells no GPU tier could build on the CPU with exact tests and writes those that are final;
    // the others, which reach past what they asked for, come back with the squared radius they need
    void fallback_needs(VMesh* mesh, std::vector<int>* cells, std::vector<double>* need_d2);

    // how many cells the CPU wrote since the last call
    int take_cpu_built();

} // namespace voronoi

#endif
