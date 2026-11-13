// implements the fields of every cell (cells.h)

#include "cells/cells.h"

#include "keys/keys.h"
#include "migration/migration.h"
#include "seed/seed.h"

#include <algorithm>

namespace cells {

    namespace {

        template <typename ElementType>
        void gathered_into_order(device::Array<ElementType>& field,
                                 const device::Array<int>&   order,
                                 int                         n_cells,
                                 device::Array<char>&        in_order);

    } // namespace

    Cells::Cells() {
        for (int d = 0; d < config::dim; d++)
            velocity_gradient[d] = device::Array<Vec>("cells: velocity gradients", 0);
    }

    gas::PrimitivePointers primitive_of(Cells& cells) {
        return gas::PrimitivePointers{cells.density.data(), cells.velocity.data(), cells.pressure.data()};
    }

    gas::ConservedPointers conserved_of(Cells& cells) {
        return gas::ConservedPointers{cells.mass.data(), cells.momentum.data(), cells.energy.data()};
    }

    gradients::GradientPointers gradients_of(Cells& cells) {
        gradients::GradientPointers pointers;
        pointers.density = cells.density_gradient.data();
        for (int d = 0; d < config::dim; d++)
            pointers.velocity[d] = cells.velocity_gradient[d].data();
        pointers.pressure = cells.pressure_gradient.data();
        pointers.anchor   = cells.gradient_anchor.data();
        return pointers;
    }

    void make_room_for(Cells& cells, int n_cells, int n_ghost_cells) {
        for_each_field(cells, [&](auto& field) { field.grow_to_at_least(n_cells); });
        for_each_primitive_field(cells, [&](auto& field) { field.grow_to_at_least(n_cells + n_ghost_cells); });
        for_each_gradient_field(cells, [&](auto& field) { field.grow_to_at_least(n_cells + n_ghost_cells); });
        for_each_seed_field(cells, [&](auto& field) { field.grow_to_at_least(n_cells + n_ghost_cells); });
        for_each_step_field(cells, [&](auto& field) { field.grow_to_at_least(n_cells + n_ghost_cells); });
        cells.outflow_exposure.grow_to_at_least(n_cells + n_ghost_cells);
    }

    // order[new place] = old place; every field then follows, one after the other through the same array
    void sort_along_curve(Cells& cells) {
        const int               n_cells = cells.n_cells;
        device::Array<uint64_t> key("cells: Morton keys", n_cells);
        device::Array<int>      order("cells: order along the curve", n_cells);

        const Vec* position = cells.position.data();
        uint64_t*  to_key   = key.data();
        int*       to_order = order.data();
        device::parallel_for(n_cells, [=] HD(long c) {
            to_key[c]   = keys::morton_key(position[c]);
            to_order[c] = (int)c;
        });
        device::sort_by_key(n_cells, key.data(), order.data());

        long largest_element = 0;
        for_each_field(cells,
                       [&](auto& field) { largest_element = std::max(largest_element, (long)sizeof(field[0])); });
        device::Array<char> in_order("cells: a field in its new order", n_cells * largest_element);
        for_each_field(cells, [&](auto& field) { gathered_into_order(field, order, n_cells, in_order); });
    }

    void move_to_owners(Cells& cells, const domain::Domain& domain) {
        device::Array<int>           owner("cells: owners", cells.n_cells);
        const domain::DomainPointers ranks    = domain.pointers();
        const Vec*                   position = cells.position.data();
        int*                         to       = owner.data();
        device::parallel_for(cells.n_cells, [=] HD(long c) { to[c] = domain::owner_of(ranks, position[c]); });

        migration::Plan plan;
        plan.make(owner.data(), cells.n_cells);
        plan.move_cells([&](auto function) { for_each_field(cells, function); });
        cells.n_cells = plan.n_cells_after();
    }

    // moved in place: the mesh keeps the seeds of its build
    void move_seeds(Cells& cells, const mesh::CellList& ending, const timesteps::StepPointers& steps) {
        Vec*       position      = cells.position.data();
        const Vec* mesh_velocity = cells.mesh_velocity.data();
        const bool into_box      = (steps.tick == timesteps::TICKS_PER_TOP_STEP);
        device::parallel_for(ending.n, [=] HD(long i) {
            const int        cell = ending[(int)i];
            const seed::Seed from{position[cell], mesh_velocity[cell], timesteps::start_of_step(steps, cell)};
            const Vec        moved = seed::position_at_tick(from, steps.tick, steps.tick_length);
            position[cell]         = into_box ? wrapped_into_box(moved) : moved;
        });
    }

    void seeds_at_sync(Cells& cells, int n_ghost_cells, const timesteps::StepPointers& steps, Vec* seeds) {
        const Vec* position      = cells.position.data();
        const Vec* mesh_velocity = cells.mesh_velocity.data();
        device::parallel_for(cells.n_cells + n_ghost_cells, [=] HD(long c) {
            const seed::Seed from{position[c], mesh_velocity[c], timesteps::start_of_step(steps, (int)c)};
            seeds[c] = seed::position_at_tick(from, steps.tick, steps.tick_length);
        });
    }

    void copy_primitive_variables_to_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts, const mesh::CellList& active) {
        ghosts.copy_active_to_ghost_cells(
            active.is_every_cell(), cells.n_cells, [&](auto function) { for_each_primitive_field(cells, function); });
    }

    void copy_gradients_to_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts, const mesh::CellList& active) {
        ghosts.copy_active_to_ghost_cells(active.is_every_cell(), cells.n_cells, [&](auto function) {
            for_each_gradient_field(cells, function);
            for_each_seed_field(cells, function);
        });
    }

    void copy_everything_to_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts) {
        ghosts.copy_to_ghost_cells(cells.n_cells, [&](auto function) {
            for_each_primitive_field(cells, function);
            for_each_gradient_field(cells, function);
            for_each_seed_field(cells, function);
            for_each_step_field(cells, function);
        });
    }

    void copy_outflow_exposure_to_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts) {
        ghosts.copy_to_ghost_cells(cells.n_cells, [&](auto function) { function(cells.outflow_exposure); });
    }

    void copy_everything_to_new_ghost_cells(Cells& cells, ghosts::Ghosts& ghosts) {
        ghosts.copy_to_new_ghost_cells([&](auto function) {
            for_each_primitive_field(cells, function);
            for_each_gradient_field(cells, function);
            for_each_seed_field(cells, function);
            for_each_step_field(cells, function);
        });
    }

    namespace {

        // through an array of bytes, copied back so that the field keeps its own array and name
        template <typename ElementType>
        void gathered_into_order(device::Array<ElementType>& field,
                                 const device::Array<int>&   order,
                                 int                         n_cells,
                                 device::Array<char>&        in_order) {
            ElementType* values   = field.data();
            const int*   old_cell = order.data();
            ElementType* moved    = reinterpret_cast<ElementType*>(in_order.data());
            device::parallel_for(n_cells, [=] HD(long c) { moved[c] = values[old_cell[c]]; });
            device::parallel_for(n_cells, [=] HD(long c) { values[c] = moved[c]; });
        }

    } // namespace

} // namespace cells
