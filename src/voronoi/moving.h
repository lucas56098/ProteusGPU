#ifndef VORONOI_MOVING_H
#define VORONOI_MOVING_H

#include "../global/allvars.h"
#include "../voronoi/voronoi.h"

namespace voronoi {

    // mesh-point velocity computation (gas velocity + regularization)
    void compute_mesh_velocities(VMesh* mesh, const hydro::primvars* primvar, const gradients::PrimGradients* grads);

    // advance seeds by v_mesh*dt (with periodic wrap), then rebuild the mesh
    void move_mesh(VMesh* mesh, double dt, hydro::primvars* primvar, hydro::primvars* primvar_aux);

} // namespace voronoi
#endif // VORONOI_MOVING_H
