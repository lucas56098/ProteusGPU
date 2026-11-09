#pragma once

// Limited gradients of the primitive variables, and the state of a cell predicted to a point and a time.

#include "gas/gas.h"
#include "mesh/mesh.h"
#include "timesteps/timesteps.h"

namespace gradients {

    // the gradients of one cell; anchor, the centroid relative to the seed, is where they were taken
    struct Gradient {
        Vec density;
        Vec velocity[config::dim]; // velocity[d]: gradient of component d
        Vec pressure;
        Vec anchor;
    };

    // one array per variable, for kernels
    struct GradientPointers {
        Vec* density;
        Vec* velocity[config::dim];
        Vec* pressure;
        Vec* anchor;
    };

    HD inline Gradient load(const GradientPointers& gradients, int cell);
    HD inline void store(const GradientPointers& gradients, int cell, const Gradient& value);

    // limited least-squares gradient of every cell of the list; state holds the cells first, then the ghost cells
    void compute_gradients(const mesh::MeshPointers&      mesh,
                           const gas::PrimitivePointers&  state,
                           const Vec*                     mesh_velocity,
                           const timesteps::StepPointers& steps,
                           const mesh::CellList&          cells,
                           const GradientPointers&        gradients);

    // state at r from the anchor, dt later, to first order; w itself if density or pressure would not stay positive;
    // on a moving mesh r also carries the motion of the gas over dt
    HD inline gas::Primitive predicted(const gas::Primitive& w, const Gradient& gradient, const Vec& r, double dt);

} // namespace gradients

// how load, store and predicted work
#include "prediction.h"
