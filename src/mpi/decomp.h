#ifndef MPI_DECOMP_H
#define MPI_DECOMP_H
#pragma once

// Splits the box into bricks, one per rank, and says which rank owns a point.

#include "global/gpu_compat.h"
#include "mpi_compat.h"

#include <cstdint>

struct ICData;

namespace proteus_mpi {

    // the decomposition of this run, set up once in decomp_init
    struct MpiDecomp {
        int rank;
        int nranks;

        int dims[3];   // ranks per axis
        int coords[3]; // place of this rank in that grid

        int N_grid_global; // buckets per axis over the box plus both ghost bands

        // bucket range of this rank, [b0, b1) per axis
        int b0[3];
        int b1[3];

        int* splits[3]; // dims[a] + 1 bucket borders per axis, one brick between two of them

        int* coord_to_rank; // rank at coords (cx, cy, cz)

#ifdef USE_MPI
        MPI_Comm cart_comm;
#endif
    };

    extern MpiDecomp decomp;

    // bucket grid, rank grid and an even split to start from
    void decomp_init(int64_t n_total, double buff);

    // takes new split tables, from a rebalance or from a snapshot
    void decomp_apply_splits(const int* sx, const int* sy, const int* sz);

    // which slab of one axis holds bucket b
    HD inline int decomp_coord_of_bucket(const int* splits, int n_slabs, int b) {
        int lo = 0;
        int hi = n_slabs;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (splits[mid] <= b)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }

    // rank that owns a bucket, -1 if it is outside the grid
    int decomp_owner_of_bucket(int bx, int by, int bz);

    HD inline int decomp_owner_of_bucket_dev(int        bx,
                                             int        by,
                                             int        bz,
                                             int        N_grid_global,
                                             int        dims_x,
                                             int        dims_y,
                                             int        dims_z,
                                             const int* splits_x,
                                             const int* splits_y,
                                             const int* splits_z,
                                             const int* coord_to_rank) {
        if (bx < 0 || bx >= N_grid_global || by < 0 || by >= N_grid_global) return -1;
#ifdef dim_3D
        if (bz < 0 || bz >= N_grid_global) return -1;
#else
        (void)bz;
        (void)dims_z;
        (void)splits_z;
#endif
        const int cx = decomp_coord_of_bucket(splits_x, dims_x, bx);
        const int cy = decomp_coord_of_bucket(splits_y, dims_y, by);
#ifdef dim_3D
        const int cz = decomp_coord_of_bucket(splits_z, dims_z, bz);
#else
        const int cz = 0;
#endif
        const int idx = (cx * dims_y + cy) * dims_z + cz;
        return coord_to_rank[idx];
    }

    // bucket a position falls into, clamped to the grid
    HD inline void
    decomp_bucket_of_point(double px, double py, double pz, int N_grid, double buff, int* bx, int* by, int* bz) {
        const double inv = 1.0 / (1.0 + 2.0 * buff);
        int          ix  = (int)((px + buff) * inv * (double)N_grid);
        int          iy  = (int)((py + buff) * inv * (double)N_grid);
        if (ix < 0)
            ix = 0;
        else if (ix >= N_grid)
            ix = N_grid - 1;
        if (iy < 0)
            iy = 0;
        else if (iy >= N_grid)
            iy = N_grid - 1;
        *bx = ix;
        *by = iy;
#ifdef dim_3D
        int iz = (int)((pz + buff) * inv * (double)N_grid);
        if (iz < 0)
            iz = 0;
        else if (iz >= N_grid)
            iz = N_grid - 1;
        *bz = iz;
#else
        (void)pz;
        *bz = 0;
#endif
    }

    // sends every IC cell to the rank that owns its bucket
    void distribute_ic_parallel(::ICData& ic, double buff);

    // rows [lo, hi) of N that part i reads, used for the parallel IC read
    void decomp_even_split(int64_t N, int P, int i, int64_t* lo, int64_t* hi);

} // namespace proteus_mpi

#endif
