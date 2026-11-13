#pragma once

// All per-cell data of this rank, one array per variable, and the one list of these fields.

#include "device/device.h"
#include "domain/domain.h"
#include "gas/gas.h"
#include "ghosts/ghosts.h"
#include "gradients/gradients.h"
#include "timesteps/timesteps.h"

namespace cells {

    // with a moving mesh and individual timesteps a ghost cell follows the path of its cell, so it holds a position
    constexpr bool GHOST_CELLS_HOLD_POSITIONS = config::moving_mesh && config::individual_timesteps;

    struct Cells {
        int                   n_cells = 0;
        device::Array<long>   id{"cells: ids", 0}; // row of the cell in the initial conditions
        device::Array<Vec>    position{"cells: seed positions", 0};
        device::Array<double> density{"cells: densities", 0};
        device::Array<Vec>    velocity{"cells: velocities", 0};
        device::Array<double> pressure{"cells: pressures", 0};
        device::Array<double> mass{"cells: masses", 0};
        device::Array<Vec>    momentum{"cells: momenta", 0};
        device::Array<double> energy{"cells: energies", 0};
        device::Array<Vec>    density_gradient{"cells: density gradients", 0};
        device::Array<Vec>    velocity_gradient[config::dim]; // [d]: gradient of velocity component d
        device::Array<Vec>    pressure_gradient{"cells: pressure gradients", 0};
        device::Array<Vec>    gradient_anchor{"cells: gradient anchors", 0};
        device::Array<double> step_mass{"cells: masses at the start of their steps", 0}; // for the flux limiter
        // the mass a cell would lose over its step at the fluxes of one half; also of the ghost cells, for the limiter
        device::Array<double> outflow_exposure{"cells: mass a cell would lose over its step", 0};
        // the conserved variables of a list before its fluxes, for a second try with the limiter
        device::Array<gas::Conserved> conserved_backup{"cells: conserved variables before the fluxes", 0};
        device::Array<Vec> mesh_velocity{"cells: mesh velocities", 0}; // from the start of the step; moving mesh only
        device::Array<double> cfl_step{"cells: CFL steps", 0}; // of the cells that begin a step, at its start
        // with ASTRO_PHYSICS and individual timesteps: the tick up to which the sources of the cell ran
        device::Array<double> source_clock{"cells: source clocks", 0};
        timesteps::CellSteps  steps; // with individual timesteps

        // the arrays of velocity_gradient get their names here
        Cells();
    };

    gas::PrimitivePointers primitive_of(Cells& cells);
    gas::ConservedPointers conserved_of(Cells& cells);
    gradients::GradientPointers gradients_of(Cells& cells);

    // the one list of the fields, in a fixed order; the primitive, gradient, seed and step groups also hold ghost cells
    template <typename Function> void for_each_field(Cells& cells, Function function);
    template <typename Function> void for_each_primitive_field(Cells& cells, Function function);
    template <typename Function> void for_each_gradient_field(Cells& cells, Function function);
    template <typename Function> void for_each_seed_field(Cells& cells, Function function);
    template <typename Function> void for_each_step_field(Cells& cells, Function function);

    // room for n_cells cells in every field, and for n_ghost_cells ghost cells behind them in the four groups
    void make_room_for(Cells& cells, int n_cells, int n_ghost_cells);

    // the cells in Morton order of their positions, every field along; equal keys keep their order
    void sort_along_curve(Cells& cells);

    // every cell to the rank that owns its position, with all its fields; collective
    void move_to_owners(Cells& cells, const domain::Domain& domain);

    // on a moving mesh: the seeds of the ending cells to where their next step begins; after a top step into the box
    void move_seeds(Cells& cells, const mesh::CellList& ending, const timesteps::StepPointers& steps);

    // moving mesh with individual timesteps: seeds[c] and seeds[n_cells + ghost_cell] at the sync, without shifts
    void seeds_at_sync(Cells& cells, int n_ghost_cells, const timesteps::StepPointers& steps, Vec* seeds);

    // the active cells into their ghost cells, the gradients together with the seeds;
    // all ranks pass every cell, or all pass a list after ghosts.find_active_cells; collective
    void copy_primitive_variables_to_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts, const mesh::CellList& active);
    void copy_gradients_to_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts, const mesh::CellList& active);

    // the four groups of every cell into every ghost cell, in one exchange; collective
    void copy_everything_to_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts);

    // the same, but only into the ghost cells of the last ghosts.give_ghost_cells_to; collective
    void copy_everything_to_new_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts);

    // the outflow exposure of every cell into its ghost cells, for the flux limiter; collective
    void copy_outflow_exposure_to_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts);

} // namespace cells

#include "fields.h"
