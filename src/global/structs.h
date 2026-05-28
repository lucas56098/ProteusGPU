#ifndef STRUCTS_H
#define STRUCTS_H
#pragma once

#include "gpu_compat.h"

namespace voronoi {

    // status codes for voronoi mesh generation
    enum Status {
        triangle_overflow           = 0, // to fix increase _MAX_T_
        vertex_overflow             = 1, // to fix increase _MAX_P_
        inconsistent_boundary       = 2,
        security_radius_not_reached = 3, // to fix increase _K_
        success                     = 4,
        needs_exact_predicates      = 5 // to fix perturb seedpoints
    };

} // namespace voronoi

namespace hydro {

    // hydro primitive variable arrays
    struct primvars {
        double*     rho;
        POINT_TYPE* v;
        double*     E; // per unit volume
    };

    // single-cell primitive state
    struct prim {
        double rho = 0;
#ifdef dim_2D
        POINT_TYPE v = {0., 0.};
#else
        POINT_TYPE v = {0., 0., 0.};
#endif
        double E = 0; // per unit volume
    };

    // flux_t: alias for prim when used for fluxes
    using flux_t = prim;

    // unphysical primvar-state check
    enum UnphysCount {
        UNPHYS_RHO = 0, // rho <= 0
        UNPHYS_E   = 1, // E <= 0
        UNPHYS_NAN = 2, // any field NaN
        UNPHYS_N   = 3
    };

} // namespace hydro

// per cell array sizing for MPI ghosts and migration
namespace proteus_mpi {

    // total MPI-ghost capacity
    extern int n_mpi_capacity;

    // migration headroom on n_local
    constexpr double ALLOC_GROWTH = 1.5;

    inline int max_n_local(int n_initial) {
        return (int)((double)n_initial * ALLOC_GROWTH);
    }

    // runtime live size
    inline int extended_size(int n_local) {
        return n_local + n_mpi_capacity;
    }

    // allocated size
    inline int alloc_per_cell_size(int n_initial) {
        return max_n_local(n_initial) + n_mpi_capacity;
    }

} // namespace proteus_mpi

namespace gradients {

    // gradient data for a single cell
    struct PrimGradient {
        POINT_TYPE rho;
        POINT_TYPE vx;
        POINT_TYPE vy;
#ifdef dim_3D
        POINT_TYPE vz;
#endif
        POINT_TYPE E;
    };

    // gradient arrays for all cells
    struct PrimGradients {
        POINT_TYPE* rho;
        POINT_TYPE* vx;
        POINT_TYPE* vy;
#ifdef dim_3D
        POINT_TYPE* vz;
#endif
        POINT_TYPE* E;
        size_t      n; // number of cells

        // load single-cell gradients from SoA arrays
        HD inline PrimGradient load(size_t i) const {
            PrimGradient g;
            g.rho = rho[i];
            g.vx  = vx[i];
            g.vy  = vy[i];
#ifdef dim_3D
            g.vz = vz[i];
#endif
            g.E = E[i];
            return g;
        }
    };

} // namespace gradients

// geometry (normalized basis for face orientation)
struct geom {
    double3 n; // normal
    double3 m; // 1. tangential
    double3 p; // 2. tangential
};

#endif // STRUCTS_H
