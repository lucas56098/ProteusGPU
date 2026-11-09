#pragma once

// how move_cells packs, sends and unpacks the fields (included by migration.h)

#include "mpi/mpi.h"

namespace migration {

    // the values of the staying cells filler[i] into the places place[i]
    template <typename ElementType>
    void fill_places(device::Array<ElementType>& field, const int* place, const int* filler, int n) {
        ElementType* values = field.data();
        device::parallel_for(n, [=] HD(long i) { values[place[i]] = values[filler[i]]; });
    }

    // one item per leaving cell, its fields at fixed offsets; every field fills its places, then takes the arrivals
    template <typename FieldList> void Plan::move_cells(FieldList for_each_field) const {
        int item_bytes = 0;
        for_each_field([&](auto& field) { item_bytes += (int)sizeof(field[0]); });
        const int n_staying = n_cells - n_leaving;

        device::Array<char> leaving("migration: the cells that leave", (long)n_leaving * item_bytes);
        int                 offset = 0;
        for_each_field([&](auto& field) {
            device::pack_into_items(field, leaving_order.data(), n_leaving, item_bytes, offset, leaving);
            offset += (int)sizeof(field[0]);
        });

        device::Array<char> arriving("migration: the cells that come", (long)n_arriving * item_bytes);
        device::exchange_items(leaving.data(), send_count.data(), arriving.data(), receive_count.data(), item_bytes);

        offset = 0;
        for_each_field([&](auto& field) {
            fill_places(field, place.data(), filler.data(), n_places);
            field.grow_to_at_least(n_staying + n_arriving);
            device::unpack_from_items(arriving, n_arriving, item_bytes, offset, n_staying, field);
            offset += (int)sizeof(field[0]);
        });
    }

} // namespace migration
