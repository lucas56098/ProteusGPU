#ifndef PERIODIC_MESH_H
#define PERIODIC_MESH_H

#include "../global/allvars.h"
#include "../voronoi/voronoi.h"

namespace voronoi {

    // periodic ghost generation and mesh movement
    void compute_periodic_mesh(VMesh* mesh, POINT_TYPE* pts_data, hsize_t num_points);
    void move_mesh(VMesh* mesh, double dt);

    // mesh-point velocity computation (gas velocity + regularization)
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads);

} // namespace voronoi
#endif // PERIODIC_MESH_H