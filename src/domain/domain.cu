// implements the cuts of the box among the ranks (domain.h)

#include "domain/domain.h"

#include "mpi/mpi.h"

#include <algorithm>
#include <vector>

namespace domain {

    namespace {

        // the Hilbert keys run from 0 to END_OF_CURVE - 1
        constexpr uint64_t END_OF_CURVE = 1ull << (config::dim * keys::HILBERT_LEVELS);

        device::Array<uint64_t> sorted_hilbert_keys(const Vec* positions, int n_cells, long* work);
        bool a_cut_is_open(const std::vector<uint64_t>& lowest, const std::vector<uint64_t>& highest);

    } // namespace

    void Domain::cut_by_number_of_cells(const Vec* positions, int n_cells) {
        cut_by_work(positions, nullptr, n_cells);
    }

    // bisects every cut on the work of all ranks below it; only integers are summed, so all ranks find the same cuts
    void Domain::cut_by_work(const Vec* positions, const long* work, int n_cells) {
        n_ranks = mpi::nranks();
        cuts.grow_to_at_least(n_ranks + 1);

        // the keys along the curve, and the work in front of every key
        device::Array<long> sorted_work("domain: work of the cells along the curve", work ? n_cells : 0);
        device::Array<long> work_below("domain: work of the cells before every key", work ? n_cells + 1 : 0);
        if (work) {
            long* to_work = sorted_work.data();
            device::parallel_for(n_cells, [=] HD(long c) { to_work[c] = work[c]; });
        }
        const device::Array<uint64_t> key =
            sorted_hilbert_keys(positions, n_cells, work ? sorted_work.data() : nullptr);
        if (work) work_below[n_cells] = device::offsets_from_counts(n_cells, sorted_work.data(), work_below.data());

        const long*     below_key   = work ? work_below.data() : nullptr;
        const uint64_t* first       = key.data();
        const uint64_t* end         = key.data() + n_cells;
        const auto      work_before = [=](long i) { return below_key ? (long long)below_key[i] : (long long)i; };
        const long long all_work    = mpi::sum_over_ranks(work_before(n_cells));

        // cut 0 and cut n_ranks are the ends of the curve
        std::vector<long long> target(n_ranks + 1);
        std::vector<uint64_t>  lowest(n_ranks + 1, 0), highest(n_ranks + 1, END_OF_CURVE);
        for (int r = 0; r <= n_ranks; r++)
            target[r] = (long long)r * all_work / n_ranks;
        highest[0]      = 0;
        lowest[n_ranks] = END_OF_CURVE;

        std::vector<uint64_t>  middle(n_ranks + 1);
        std::vector<long long> below(n_ranks + 1);
        while (a_cut_is_open(lowest, highest)) {
            for (int r = 0; r <= n_ranks; r++) {
                middle[r] = lowest[r] + (highest[r] - lowest[r]) / 2;
                below[r]  = work_before(std::lower_bound(first, end, middle[r]) - first);
            }
            mpi::sum_over_ranks(below.data(), n_ranks + 1);
            for (int r = 0; r <= n_ranks; r++) {
                if (lowest[r] == highest[r]) continue;
                if (below[r] >= target[r])
                    highest[r] = middle[r];
                else
                    lowest[r] = middle[r] + 1;
            }
        }
        for (int r = 0; r <= n_ranks; r++)
            cuts[r] = lowest[r];
    }

    // from integers that are the same on every rank, so that every rank decides the same
    bool is_unbalanced(int n_cells) {
        const long long most = mpi::max_over_ranks((long long)n_cells);
        const long long all  = mpi::sum_over_ranks((long long)n_cells);
        return (double)most * (double)mpi::nranks() > MOST_CELLS_OVER_MEAN * (double)all;
    }

    bool is_unbalanced_by_work(const long* work, int n_cells) {
        const long long here = device::sum(n_cells, [=] HD(long c) { return (long long)work[c]; });
        const long long most = mpi::max_over_ranks(here);
        const long long all  = mpi::sum_over_ranks(here);
        return (double)most * (double)mpi::nranks() > MOST_CELLS_OVER_MEAN * (double)all;
    }

    double imbalance(int n_cells) {
        const long long most = mpi::max_over_ranks((long long)n_cells);
        const long long all  = mpi::sum_over_ranks((long long)n_cells);
        return (double)most * (double)mpi::nranks() / (double)all;
    }

    double imbalance_by_work(const long* work, int n_cells) {
        const long long here = device::sum(n_cells, [=] HD(long c) { return (long long)work[c]; });
        const long long most = mpi::max_over_ranks(here);
        const long long all  = mpi::sum_over_ranks(here);
        return (double)most * (double)mpi::nranks() / (double)all;
    }

    DomainPointers Domain::pointers() const {
        return DomainPointers{n_ranks, mpi::rank(), cuts.data()};
    }

    namespace {

        // on the GPU, with the work of the cells along when it is there; the order of the cells is not needed
        device::Array<uint64_t> sorted_hilbert_keys(const Vec* positions, int n_cells, long* work) {
            device::Array<uint64_t> key("domain: Hilbert keys of the cells", n_cells);
            device::Array<int>      order("domain: order of the Hilbert keys", n_cells);
            uint64_t*               to_key   = key.data();
            int*                    to_order = order.data();
            device::parallel_for(n_cells, [=] HD(long c) {
                to_key[c]   = keys::hilbert_key(positions[c]);
                to_order[c] = (int)c;
            });
            if (work) {
                device::sort_by_key(n_cells, key.data(), work);
            } else {
                device::sort_by_key(n_cells, key.data(), order.data());
            }
            return key;
        }

        // the same answer on every rank, as every rank has the same ranges
        bool a_cut_is_open(const std::vector<uint64_t>& lowest, const std::vector<uint64_t>& highest) {
            for (size_t r = 0; r < lowest.size(); r++) {
                if (lowest[r] != highest[r]) return true;
            }
            return false;
        }

    } // namespace

} // namespace domain
