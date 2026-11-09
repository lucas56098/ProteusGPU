#pragma once

// The gas of one cell as the sources change it, with its total energy per volume.

#include "gas/gas.h"

namespace astro {

    struct CellGas {
        double density;
        Vec    velocity;
        double energy; // total, per volume
    };

    HD inline CellGas with_energy(const gas::Primitive& w) {
        return CellGas{w.density, w.velocity, gas::energy_density(w)};
    }

    HD inline gas::Primitive with_pressure(const CellGas& g) {
        return gas::Primitive{g.density, g.velocity, gas::pressure_of(g.density, g.velocity, g.energy)};
    }

    // the internal energy per volume
    HD inline double internal_energy(const CellGas& g) {
        return g.energy - 0.5 * g.density * norm2(g.velocity);
    }

    // mass dm comes at the velocity of the cell: the energies per mass stay
    HD inline void add_mass_at_rest_with_the_cell(double dm, CellGas& g) {
        g.energy *= (1.0 + dm / g.density);
        g.density += dm;
    }

    // offset from the middle of the box, where every profile has its centre
    HD inline Vec from_centre(const Vec& position) {
        Vec offset = position;
        for (int d = 0; d < config::dim; d++)
            offset[d] -= 0.5;
        return offset;
    }

} // namespace astro
