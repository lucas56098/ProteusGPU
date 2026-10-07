#ifndef MPI_DECOMP_H
#define MPI_DECOMP_H
#pragma once

// Splits the unit box along a Hilbert curve, one stretch of it per rank, says who owns what, and moves the
// cuts when the cells are spread unevenly.

#include "global/gpu_compat.h"
#include "global/parallel.h"
#include "knn/keys.h"
#include "mpi_compat.h"

#include <cstdint>
#include <vector>

struct ICData;
struct VMesh;

namespace proteus_mpi {

    // rank r owns the cells whose Hilbert key is in [cuts[r], cuts[r + 1]); the table is the same on all ranks
    struct MpiDecomp {
        int       rank;
        int       nranks;
        uint64_t* cuts; // nranks + 1 keys

#ifdef USE_MPI
        MPI_Comm comm;
#endif

        // the open cuts of a cut search and the local cell count below each
        GpuArray<uint64_t>  probe;
        GpuArray<long long> below;
    };

    extern MpiDecomp decomp;

    // the communicator and an even split of the keys to start from
    void decomp_init();

    // takes a new cut table, from a rebalance or from a snapshot; all ranks pass the same one
    void decomp_set_cuts(const uint64_t* cuts);

    void decomp_free();

    // cuts that give every rank about the same number of cells out of all ranks' points; collective.
    // pts are the n local points in gpu memory, sort has room for n
    void decomp_balanced_cuts(const POINT_TYPE* pts, int n, PairSort sort, std::vector<uint64_t>* cuts_out);

    // prints the imbalance every imbalance_log_interval steps
    void rebalance_imbalance_log(int step, VMesh* mesh);

    // true when new cuts were applied, then the cells have to migrate
    bool rebalance_decide(int step, VMesh* mesh, POINT_TYPE* pts);

    void rebalance_log_after_migration(VMesh* mesh);

    // bits a rank number takes as a sort key
    inline int rank_key_bits() {
        int b = 1;
        while ((1 << b) < decomp.nranks)
            b++;
        return b;
    }

    // rank that owns a key: the last cut at or below it
    HD inline int owner_of_key(uint64_t key, const uint64_t* cuts, int nranks) {
        return (int)upper_bound_of(cuts, (size_t)nranks, key) - 1;
    }

    HD inline int owner_of_point(const POINT_TYPE& p, const uint64_t* cuts, int nranks) {
        return owner_of_key(knn::hilbert_key(p), cuts, nranks);
    }

    // a cube of the unit box: side 2^-level, integer corner c on that level
    struct KeyCube {
        int          level;
        unsigned int c[3];
    };

    // its keys are [first, last]
    HD inline void key_range_of_cube(const KeyCube& q, uint64_t* first, uint64_t* last) {
        unsigned int X[DIMENSION];
        for (int i = 0; i < DIMENSION; i++)
            X[i] = q.c[i];
        const int shift = DIMENSION * (knn::DOMAIN_BITS - q.level);
        *first          = knn::hilbert_index(X, q.level) << shift;
        *last           = *first + ((1ull << shift) - 1);
    }

    // squared distance from p to the cube
    HD inline double dist2_to_cube(const KeyCube& q, const POINT_TYPE& p) {
        const double a    = 1.0 / (double)(1ull << q.level);
        double       d2   = 0.0;
        const double x[3] = {p.x,
                             p.y,
#ifdef dim_3D
                             p.z
#else
                             0.0
#endif
        };
        for (int i = 0; i < DIMENSION; i++) {
            const double lo = (double)q.c[i] * a;
            const double hi = lo + a;
            const double d  = (x[i] < lo) ? lo - x[i] : ((x[i] > hi) ? x[i] - hi : 0.0);
            d2 += d * d;
        }
        return d2;
    }

    // the pending cubes of one ball walk; a cube that leaves gives its children
    constexpr int BALL_WALK_STACK = (DIMENSION == 3) ? 7 * 20 + 8 : 3 * 30 + 4;

    // calls visit(rank) for every rank whose keys the ball may touch inside the box, a rank possibly
    // more than once; visit returns false to stop the walk early
    template <typename F>
    HD inline void for_each_rank_in_ball(const POINT_TYPE& c, double r, const uint64_t* cuts, int nranks, F visit) {
        const double r2 = r * r;
        KeyCube      stack[BALL_WALK_STACK];
        int          sp = 0;
        stack[sp++]     = KeyCube{0, {0u, 0u, 0u}};
        while (sp > 0) {
            const KeyCube q = stack[--sp];
            if (dist2_to_cube(q, c) > r2) continue;
            uint64_t first, last;
            key_range_of_cube(q, &first, &last);
            const int r_lo = owner_of_key(first, cuts, nranks);
            const int r_hi = owner_of_key(last, cuts, nranks);
            if (r_lo == r_hi) {
                if (!visit(r_lo)) return;
                continue;
            }
            // a cube on the finest level is one key, so the walk always ends
            for (int ch = 0; ch < (1 << DIMENSION); ch++) {
                KeyCube k;
                k.level = q.level + 1;
                for (int i = 0; i < 3; i++)
                    k.c[i] = (i < DIMENSION) ? 2 * q.c[i] + ((ch >> i) & 1) : 0u;
                stack[sp++] = k;
            }
        }
    }

    // the cubes of the coarsest level that still fits the ball's bounding box twice: at most 2 per axis.
    // True if all of them lie in [lo, hi); a cheap test that says no near a cut although the ball may not cross it
    HD inline bool ball_cubes_in_keys(const POINT_TYPE& c, double r, uint64_t lo, uint64_t hi) {
        int level = 0;
        while (level < knn::DOMAIN_BITS && 2.0 * r <= 1.0 / (double)(2ull << level))
            level++;
        const double n    = (double)(1ull << level);
        const double x[3] = {c.x,
                             c.y,
#ifdef dim_3D
                             c.z
#else
                             0.0
#endif
        };
        unsigned int first[3] = {0u, 0u, 0u}, last[3] = {0u, 0u, 0u};
        for (int i = 0; i < DIMENSION; i++) {
            first[i] = (unsigned int)((x[i] - r) * n);
            last[i]  = (unsigned int)((x[i] + r) * n);
        }
        for (unsigned int a = first[0]; a <= last[0]; a++) {
            for (unsigned int b = first[1]; b <= last[1]; b++) {
                for (unsigned int d = first[2]; d <= last[2]; d++) {
                    uint64_t k0, k1;
                    key_range_of_cube(KeyCube{level, {a, b, d}}, &k0, &k1);
                    if (k0 < lo || k1 >= hi) return false;
                }
            }
        }
        return true;
    }

    // true if the ball lies inside the box and only touches keys of rank me
    HD inline bool ball_is_local(const POINT_TYPE& c, double r, const uint64_t* cuts, int nranks, int me) {
        if (c.x - r < 0.0 || c.x + r >= 1.0 || c.y - r < 0.0 || c.y + r >= 1.0) return false;
#ifdef dim_3D
        if (c.z - r < 0.0 || c.z + r >= 1.0) return false;
#endif
        if (nranks == 1) return true;
        if (ball_cubes_in_keys(c, r, cuts[me], cuts[me + 1])) return true;
        bool local = true;
        for_each_rank_in_ball(c, r, cuts, nranks, [&](int rk) {
            if (rk != me) local = false;
            return local;
        });
        return local;
    }

    // sends every IC cell to the rank that owns it, after cutting the curve so they come out even
    void distribute_ic_parallel(::ICData& ic);

    // rows [lo, hi) of N that part i reads, used for the parallel IC read
    void decomp_even_split(int64_t N, int P, int i, int64_t* lo, int64_t* hi);

} // namespace proteus_mpi

#endif
