#ifndef MPI_DECOMP_H
#define MPI_DECOMP_H
#pragma once

#include "global/gpu_compat.h"
#include "mpi_compat.h"

#include <cstdint>

struct ICData;

// static brick decomposition over the global KNN bucket grid. each rank owns
// a contiguous brick of buckets; cell ownership follows the bucket containing
// the seed position.

namespace proteus_mpi {

    struct MpiDecomp {
        int rank;
        int nranks;

        // Cartesian topology (Pz = 1 in 2D)
        int dims[3];
        int coords[3];

        // global bucket grid, shared across ranks
        int N_grid_global;

        // this rank's brick of buckets: [b0[i], b1[i]) per axis. b0[2]=0, b1[2]=1 in 2D.
        int b0[3];
        int b1[3];

        // per-axis split tables, identical on every rank. size dims[a]+1; coord cx along axis a
        // owns buckets [splits[a][cx], splits[a][cx+1]). Initial values come from even_split;
        // rebalance overwrites them via decomp_apply_splits. 2D forces splits[2] = {0, 1}.
        // Allocated via gpu_malloc (managed) so device kernels can read them.
        int* splits[3];

        // coord-tuple -> rank lookup, size dims[0]*dims[1]*dims[2], managed memory.
        // Index = (cx * dims[1] + cy) * dims[2] + cz. Built from MPI_Cart_rank at decomp_init
        // and refreshed by decomp_apply_splits. Lets device kernels resolve an owner rank
        // without calling MPI_Cart_rank (which is host-only).
        int* coord_to_rank;

#ifdef USE_MPI
        MPI_Comm cart_comm;
#endif
    };

    extern MpiDecomp decomp;

    // initialize decomposition; computes this rank's brick and prints the partition.
    // single-node mode is a no-op (full domain owned by rank 0).
    // n_total is the global cell count — int64 because >= 1290^3 overflows int32.
    void decomp_init(int64_t n_total, double buff);

    // install new per-axis split tables (replaces the values for splits[0..2]) and recomputes
    // this rank's b0/b1. Inputs must satisfy: monotone non-decreasing, splits[a][0]=0,
    // splits[a][dims[a]]=N_grid_global (z: dims[2]=1, splits[2]={0,1}). All ranks must call
    // with identical arrays — they hold the global decomposition.
    void decomp_apply_splits(const int* sx, const int* sy, const int* sz);

    // binary search for the slab containing bucket index b. Returns coord c such that
    // splits[c] <= b < splits[c+1]. Assumes the caller already checked b ∈ [0, N_grid_global).
    HD inline int decomp_coord_of_bucket(const int* splits, int n_slabs, int b) {
        int lo = 0;
        int hi = n_slabs; // upper bound on coord (exclusive)
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (splits[mid] <= b)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }

    inline bool decomp_owns_bucket(int bx, int by, int bz) {
        const MpiDecomp& d = decomp;
        return bx >= d.b0[0] && bx < d.b1[0] && by >= d.b0[1] && by < d.b1[1] && bz >= d.b0[2] && bz < d.b1[2];
    }

    // owner rank of the given bucket; returns -1 if out of global range
    int decomp_owner_of_bucket(int bx, int by, int bz);

    // device-callable owner-of-bucket: same semantics as decomp_owner_of_bucket, but
    // uses the precomputed coord_to_rank lookup (no MPI_Cart_rank, no globals). The
    // caller passes per-axis split arrays + dims + N_grid so this works from a kernel.
    // 2D: callers pass dims[2]=1, splits[2]={0,1}.
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
        bz = 0;
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

    // bucket coords for a position. mirrors knn::cellFromPoint's index math but uses
    // the global N_grid so all ranks agree. 2D: z bucket is always 0.
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

    // parallel-read variant: each rank arrives with its own chunk of the global IC
    // (read via parallel HDF5 hyperslab), holding cells with global IDs [row_lo, row_lo+n_local).
    // Routes each cell to its owner via decomp_owner_of_bucket + MPI_Alltoallv over the Cart
    // comm. After the call, ic holds only this rank's owned cells. No-op for nranks <= 1.
    void distribute_ic_parallel(::ICData& ic, double buff);

    // even split of N items across P slots; slot i gets the i'th contiguous chunk.
    // Exposed so begrun can compute this rank's IC row range before the parallel read.
    // N and the row offsets are int64 because rank R's lo grows as R*(N/P), which crosses
    // int32 well before the global cell count itself does.
    void decomp_even_split(int64_t N, int P, int i, int64_t* lo, int64_t* hi);

} // namespace proteus_mpi

#endif // MPI_DECOMP_H
