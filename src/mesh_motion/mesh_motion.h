#pragma once

// The mesh velocity of a cell: the gas velocity, and a regularization that keeps the cell round.

#include "gas/gas.h"
#include "mesh/mesh.h"

namespace mesh_motion {

    // a seed off the centroid of its cell by more than this share of the radius of its ball is moved back
    constexpr double OFF_CENTRE_LIMIT = 0.2;

    // a cell whose largest face angle is above this has its seed moved to the centroid
    constexpr double LARGEST_FACE_ANGLE = 2.25;

    // a cell with the reference radius over its own radius above this moves towards its larger neighbours
    constexpr double LARGEST_RADIUS_RATIO = 1.0;

    // mesh velocity of every cell of the list: the gas velocity, the stronger of the first two regularizations, and
    // the one by volume
    void mesh_velocities(const mesh::MeshPointers&     mesh,
                         const gas::PrimitivePointers& state,
                         const Vec*                    density_gradient,
                         const mesh::CellList&         cells,
                         Vec*                          mesh_velocity);

} // namespace mesh_motion
