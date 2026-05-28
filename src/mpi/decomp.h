#ifndef MPI_DECOMP_H
#define MPI_DECOMP_H
#pragma once

#include "global/gpu_compat.h"
#include "mpi_compat.h"

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

#ifdef USE_MPI
    MPI_Comm cart_comm;
#endif
};

extern MpiDecomp decomp;

// initialize decomposition; computes this rank's brick and prints the partition.
// single-node mode is a no-op (full domain owned by rank 0)
void decomp_init(int n_total, double buff);

inline bool decomp_owns_bucket(int bx, int by, int bz) {
    const MpiDecomp& d = decomp;
    return bx >= d.b0[0] && bx < d.b1[0]
        && by >= d.b0[1] && by < d.b1[1]
        && bz >= d.b0[2] && bz < d.b1[2];
}

// owner rank of the given bucket; returns -1 if out of global range
int decomp_owner_of_bucket(int bx, int by, int bz);

// bucket coords for a position. mirrors knn::cellFromPoint's index math but uses
// the global N_grid so all ranks agree. 2D: z bucket is always 0.
HD inline void decomp_bucket_of_point(double px, double py, double pz, int N_grid, double buff,
                                      int* bx, int* by, int* bz) {
    const double inv = 1.0 / (1.0 + 2.0 * buff);
    int          ix  = (int)((px + buff) * inv * (double)N_grid);
    int          iy  = (int)((py + buff) * inv * (double)N_grid);
    if (ix < 0) ix = 0; else if (ix >= N_grid) ix = N_grid - 1;
    if (iy < 0) iy = 0; else if (iy >= N_grid) iy = N_grid - 1;
    *bx = ix;
    *by = iy;
#ifdef dim_3D
    int iz = (int)((pz + buff) * inv * (double)N_grid);
    if (iz < 0) iz = 0; else if (iz >= N_grid) iz = N_grid - 1;
    *bz = iz;
#else
    (void)pz;
    *bz = 0;
#endif
}

// distribute already-loaded ICData: each rank keeps only cells whose bucket lies
// in its brick. assigns sequential global IDs in input order.
void distribute_ic_local(::ICData& ic, double buff);

}  // namespace proteus_mpi

#endif  // MPI_DECOMP_H
