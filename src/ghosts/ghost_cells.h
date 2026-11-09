#pragma once

// how the copies to the ghost cells pack, send and unpack the values (included by ghosts.h)

namespace ghosts {

    // one item per cell sent, its fields at fixed offsets; item i goes to ghost_cell_place_by_owner[i]
    template <typename FieldList> void Ghosts::copy_to_ghost_cells(int, FieldList for_each_field) {
        int item_bytes = 0;
        for_each_field([&](auto& field) { item_bytes += (int)sizeof(field[0]); });
        values_sent.grow_to_at_least((long)n_cells_sent * item_bytes);
        values_received.grow_to_at_least((long)n_ghost_cells * item_bytes);

        int offset = 0;
        for_each_field([&](auto& field) {
            device::pack_into_items(field, cell_sent.data(), n_cells_sent, item_bytes, offset, values_sent);
            offset += (int)sizeof(field[0]);
        });

        device::exchange_items(
            values_sent.data(), cells_to_askers, values_received.data(), ghost_cells_from_owners, item_bytes);

        offset = 0;
        for_each_field([&](auto& field) {
            device::unpack_from_items_at(
                values_received, n_ghost_cells, item_bytes, offset, ghost_cell_place_by_owner.data(), field);
            offset += (int)sizeof(field[0]);
        });
    }

    // the answer to what the new ghost cells asked for: their owners send in the order they were asked
    template <typename FieldList> void Ghosts::copy_to_new_ghost_cells(FieldList for_each_field) {
        int item_bytes = 0;
        for_each_field([&](auto& field) { item_bytes += (int)sizeof(field[0]); });
        values_sent.grow_to_at_least((long)new_to_askers.total * item_bytes);
        values_received.grow_to_at_least((long)new_from_owners.total * item_bytes);

        int offset = 0;
        for_each_field([&](auto& field) {
            device::pack_into_items(
                field, cell_of_new_asked.data(), new_to_askers.total, item_bytes, offset, values_sent);
            offset += (int)sizeof(field[0]);
        });

        device::exchange_items(values_sent.data(), new_to_askers, values_received.data(), new_from_owners, item_bytes);

        offset = 0;
        for_each_field([&](auto& field) {
            device::unpack_from_items_at(
                values_received, new_from_owners.total, item_bytes, offset, place_of_new_ghost_cell.data(), field);
            offset += (int)sizeof(field[0]);
        });
    }

    // the counts are known on both sides: an active cell and its ghost cells end their step at the same tick
    template <typename FieldList>
    void Ghosts::copy_active_to_ghost_cells(bool every_cell_is_active, int n_cells, FieldList for_each_field) {
        if (every_cell_is_active) {
            copy_to_ghost_cells(n_cells, for_each_field);
            return;
        }
        int item_bytes = 0;
        for_each_field([&](auto& field) { item_bytes += (int)sizeof(field[0]); });
        values_sent.grow_to_at_least((long)n_active_cells_sent * item_bytes);
        values_received.grow_to_at_least((long)n_active_ghost_cells * item_bytes);

        int offset = 0;
        for_each_field([&](auto& field) {
            device::pack_into_items(
                field, active_cell_sent.data(), n_active_cells_sent, item_bytes, offset, values_sent);
            offset += (int)sizeof(field[0]);
        });

        device::exchange_items(
            values_sent.data(), active_to_askers, values_received.data(), active_from_owners, item_bytes);

        offset = 0;
        for_each_field([&](auto& field) {
            device::unpack_from_items_at(
                values_received, n_active_ghost_cells, item_bytes, offset, active_ghost_cell.data(), field);
            offset += (int)sizeof(field[0]);
        });
    }

} // namespace ghosts
