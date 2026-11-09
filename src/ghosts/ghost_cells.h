#pragma once

// how the copies to the ghost cells pack, send and unpack the values (included by ghosts.h)

#include "mpi/mpi.h"

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
            values_sent.data(), cells_for_rank.data(), values_received.data(), ghost_cells_of_rank.data(), item_bytes);

        offset = 0;
        for_each_field([&](auto& field) {
            device::unpack_from_items_at(
                values_received, n_ghost_cells, item_bytes, offset, ghost_cell_place_by_owner.data(), field);
            offset += (int)sizeof(field[0]);
        });
    }

    // one item per changed cell sent: its place, then its fields at fixed offsets
    template <typename FieldList>
    void Ghosts::copy_changed_to_ghost_cells(int n_cells, const int* changed, int n_changed, FieldList for_each_field) {
        if (changed == nullptr) {
            copy_to_ghost_cells(n_cells, for_each_field);
            return;
        }
        int item_bytes = (int)sizeof(int);
        for_each_field([&](auto& field) { item_bytes += (int)sizeof(field[0]); });
        const int n_items = start_items_of_changed_cells(changed, n_changed, item_bytes);

        int offset = (int)sizeof(int);
        for_each_field([&](auto& field) {
            device::pack_into_items(field, changed_cell_sent.data(), n_items, item_bytes, offset, values_sent);
            offset += (int)sizeof(field[0]);
        });

        device::exchange_items(
            values_sent.data(), changed_for_rank.data(), values_received.data(), changed_from_rank.data(), item_bytes);

        find_ghost_cells_of_items_that_came(item_bytes);
        offset = (int)sizeof(int);
        for_each_field([&](auto& field) {
            device::unpack_from_items_at(
                values_received, n_changed_received, item_bytes, offset, ghost_cell_of_changed.data(), field);
            offset += (int)sizeof(field[0]);
        });
    }

} // namespace ghosts
