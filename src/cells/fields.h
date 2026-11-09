#pragma once

// the one list of the fields of cells.h (included by cells.h)

namespace cells {

    // every field of Cells once; a field missing here would not move with its cell
    template <typename Function> void for_each_field(Cells& cells, Function function) {
        function(cells.id);
        if constexpr (!GHOST_CELLS_HOLD_POSITIONS) function(cells.position);
        for_each_seed_field(cells, function);
        for_each_primitive_field(cells, function);
        function(cells.mass);
        function(cells.momentum);
        function(cells.energy);
        for_each_gradient_field(cells, function);
        function(cells.cfl_step);
        if constexpr (config::astro_physics && config::individual_timesteps) function(cells.source_clock);
        for_each_step_field(cells, function);
    }

    // the fields a neighbour on another rank reads before the gradients
    template <typename Function> void for_each_primitive_field(Cells& cells, Function function) {
        function(cells.density);
        function(cells.velocity);
        function(cells.pressure);
    }

    // the fields of the seed that a neighbour on another rank follows; none on a mesh at rest
    template <typename Function> void for_each_seed_field(Cells& cells, Function function) {
        if constexpr (GHOST_CELLS_HOLD_POSITIONS) function(cells.position);
        if constexpr (config::moving_mesh) function(cells.mesh_velocity);
    }

    // the fields a neighbour on another rank reads before the fluxes
    template <typename Function> void for_each_gradient_field(Cells& cells, Function function) {
        function(cells.density_gradient);
        for (int d = 0; d < config::dim; d++)
            function(cells.velocity_gradient[d]);
        function(cells.pressure_gradient);
        function(cells.gradient_anchor);
    }

    // the fields a neighbour on another rank reads for the shares of the faces and for the bins; none with one bin
    template <typename Function> void for_each_step_field(Cells& cells, Function function) {
        timesteps::for_each_field(cells.steps, function);
    }

} // namespace cells
