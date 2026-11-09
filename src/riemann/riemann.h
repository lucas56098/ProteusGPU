#pragma once

// The Riemann flux between two gas states, from an HLLC solver.

#include "gas/gas.h"

namespace riemann {

    // flux of a left and a right state, both in the frame of the face
    HD inline gas::Flux hllc(const gas::Primitive& left, const gas::Primitive& right);

} // namespace riemann

// how the solver works
#include "hllc.h"
