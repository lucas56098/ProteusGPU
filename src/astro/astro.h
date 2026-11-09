#pragma once

// The astrophysical source terms, in two halves around the hydro.

#include "astro/agn.h"
#include "astro/cooling.h"
#include "astro/limiters.h"
#include "astro/parameters.h"
#include "astro/potential.h"
#include "astro/stars.h"
#include "gas/gas.h"
#include "hydro/hydro.h"
#include "timesteps/timesteps.h"

namespace astro {

    struct SourcePointers;

    class Sources {
      public:
        // the sources of the switches turned on, in code units
        void set_up(const Parameters& parameters);

        // START, before the CFL steps: the cold gas near the centre, whether the AGN fires or switched on; collective
        void prepare(const timesteps::Sync&        sync,
                     const long*                   id,
                     const Vec*                    position,
                     const gas::PrimitivePointers& state,
                     const mesh::MeshPointers&     mesh);

        // while the AGN fires: shorter CFL steps for the cells of the list in its heated sphere and jets
        void shorten_cfl_steps_near_the_agn(const mesh::CellList&         cells,
                                            const Vec*                    position,
                                            const gas::PrimitivePointers& state,
                                            const Vec*                    mesh_velocity,
                                            const mesh::MeshPointers&     mesh,
                                            double                        cfl_fraction,
                                            double*                       cfl_step) const;

        // whether the AGN switched on at the last START, inside a top step
        bool agn_switched_on() const;

        // first half of the step, before the hydro: the static potential, then cooling, the stars and the AGN
        void apply_first_half(const timesteps::SourceTimes& time,
                              const mesh::CellList&         active,
                              const Vec*                    position,
                              const gas::PrimitivePointers& state) const;

        // second half, after the hydro: the same in the reverse order, then the limiters
        void apply_second_half(const timesteps::SourceTimes& time,
                               const mesh::CellList&         active,
                               const Vec*                    position,
                               const gas::PrimitivePointers& state) const;

        SourcePointers pointers() const;

        // specific internal energy no cell goes below: that of the temperature floor with COOLING, else 0
        double lowest_specific_energy() const;

      private:
        CodeUnits    units;
        Potential    potential;
        CoolingTable cooling_table;
        Cooling      cooling;
        Stars        stars;
        Agn          agn;
        Limiters     limiters;

        device::Array<double> cold_mass{"astro: cold mass of every cell", 0};
        double                cold_mass_near_the_centre = 0.0;
        bool                  agn_fired                 = false;
        bool                  agn_switched              = false;
    };

    // the sources in code units, as kernels read them
    struct SourcePointers {
        Potential potential;
        Cooling   cooling;
        Stars     stars;
        Agn       agn;
        Limiters  limiters;
        double    accretion_rate; // of the black hole, while it fires
        double    power;
    };

} // namespace astro
