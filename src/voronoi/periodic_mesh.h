#ifndef PERIODIC_MESH_H
#define PERIODIC_MESH_H

#include "../global/allvars.h"
#include "../voronoi/voronoi.h"

namespace voronoi {

    inline bool is_in(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za = 0., double zb = 1.);

    inline void add_ghost(POINT_TYPE*    pts,
                          hsize_t        index,
                          hsize_t*       n_ghosts,
                          const hsize_t* n_hydro,
                          hsize_t*       original_ids,
                          double         shift_x,
                          double         shift_y,
                          double         shift_z = 0.);

    VMesh* compute_periodic_mesh(POINT_TYPE* pts_data, hsize_t num_points, VMesh* reuse = nullptr);

    void
    compute_mesh_velocities(const VMesh* mesh, const primvars* primvar, const PrimGradients* grads, POINT_TYPE* v_mesh);
    VMesh* move_mesh(VMesh* mesh, const POINT_TYPE* v_mesh, double dt);

} // namespace voronoi
#endif // PERIODIC_MESH_H