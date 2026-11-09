// implements the plan of a migration (migration.h)

#include "migration/migration.h"

#include <algorithm>

namespace migration {

    void Plan::make(const int* destination, int n_cells) {
        const int n_ranks = mpi::nranks();
        const int here    = mpi::rank();
        this->n_cells     = n_cells;
        leaving_order.grow_to_at_least(n_cells);
        send_count.grow_to_at_least(n_ranks);
        receive_count.grow_to_at_least(n_ranks);

        // the leaving cells by destination, equal keys keeping their order
        int* leaving = leaving_order.data();
        n_leaving    = device::indices_where(n_cells, [=] HD(int c) { return destination[c] != here; }, leaving);

        device::Array<uint64_t> key("migration: destinations as sort keys", n_leaving);
        uint64_t*               to_key = key.data();
        device::parallel_for(n_leaving, [=] HD(long i) { to_key[i] = (uint64_t)destination[leaving[i]]; });
        device::sort_by_key(n_leaving, key.data(), leaving);

        const uint64_t* first = key.data();
        const uint64_t* end   = key.data() + n_leaving;
        for (int r = 0; r < n_ranks; r++) {
            const uint64_t* start_of_rank = std::lower_bound(first, end, (uint64_t)r);
            const uint64_t* end_of_rank   = std::lower_bound(first, end, (uint64_t)r + 1);
            send_count[r]                 = (int)(end_of_rank - start_of_rank);
        }

        // the staying cells behind the first n_staying places move into the places of the leaving cells before them
        const int n_staying = n_cells - n_leaving;
        place.grow_to_at_least(n_leaving);
        filler.grow_to_at_least(n_leaving);
        int* to_filler = filler.data();
        n_places = device::indices_where(n_staying, [=] HD(int c) { return destination[c] != here; }, place.data());
        device::indices_where(n_leaving, [=] HD(int i) { return destination[n_staying + i] == here; }, to_filler);
        device::parallel_for(n_places, [=] HD(long i) { to_filler[i] += n_staying; });

        mpi::exchange_counts(send_count.data(), receive_count.data());
        n_arriving = 0;
        for (int r = 0; r < n_ranks; r++)
            n_arriving += receive_count[r];
    }

    int Plan::n_cells_after() const {
        return n_cells - n_leaving + n_arriving;
    }

} // namespace migration
