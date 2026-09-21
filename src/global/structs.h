#ifndef STRUCTS_H
#define STRUCTS_H
#pragma once

// Small shared types: cell status, primitive variables, gradients and the face frame.

#include "gpu_compat.h"

namespace voronoi {

    // how a cell build ended; anything but success escalates to the next tier
    enum Status {
        triangle_overflow           = 0, // more cell vertices than the tier has slots
        vertex_overflow             = 1, // more clip planes than the tier has slots
        inconsistent_boundary       = 2, // clipping left a boundary that does not close
        security_radius_not_reached = 3, // neighbour list ran out before the cell was final
        success                     = 4,
        needs_exact_predicates      = 5, // a determinant was too small to decide
        security_radius_beyond_data = 6  // cell may reach past this rank's data, a wider halo can fix it
    };

} // namespace voronoi

namespace hydro {

    // primitive variables as SoA; local cells in rho/v/E, MPI ghosts in the _g arrays
    struct primvars {
        double*     rho;
        POINT_TYPE* v;
        double*     E;

        double*     rho_g;
        POINT_TYPE* v_g;
        double*     E_g;
    };

    // one cell's primitive state; E is total energy per volume
    struct prim {
        double rho = 0;
#ifdef dim_2D
        POINT_TYPE v = {0., 0.};
#else
        POINT_TYPE v = {0., 0., 0.};
#endif
        double E = 0;
    };

    using flux_t = prim; // a flux has the same components

} // namespace hydro

namespace proteus_mpi {

    extern int n_mpi_capacity; // ghost slots allocated on this rank

    extern int n_local_initial_max; // largest n_local at startup over all ranks

    extern double alloc_growth; // headroom of the per-cell arrays over that count, from the param file

    // size of the per-cell arrays
    inline int max_n_local(int n_initial) {
        const int base = (n_local_initial_max > 0) ? n_local_initial_max : n_initial;
        return (int)((double)base * alloc_growth);
    }

} // namespace proteus_mpi

namespace gradients {

    // gradients of one cell, one POINT_TYPE per primitive
    struct PrimGradient {
        POINT_TYPE rho;
        POINT_TYPE vx;
        POINT_TYPE vy;
#ifdef dim_3D
        POINT_TYPE vz;
#endif
        POINT_TYPE E;
    };

    // SoA gradients with the same ghost split as primvars
    struct PrimGradients {
        POINT_TYPE* rho;
        POINT_TYPE* vx;
        POINT_TYPE* vy;
#ifdef dim_3D
        POINT_TYPE* vz;
#endif
        POINT_TYPE* E;
        size_t      n;

        POINT_TYPE* rho_g;
        POINT_TYPE* vx_g;
        POINT_TYPE* vy_g;
#ifdef dim_3D
        POINT_TYPE* vz_g;
#endif
        POINT_TYPE* E_g;

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

        // k < n_hydro: local cell, above that MPI ghost k - n_hydro
        HD inline PrimGradient load_at(int k, int n_hydro) const {
            PrimGradient g;
            if (k < n_hydro) {
                g.rho = rho[k];
                g.vx  = vx[k];
                g.vy  = vy[k];
#ifdef dim_3D
                g.vz = vz[k];
#endif
                g.E = E[k];
            } else {
                const int s = k - n_hydro;
                g.rho       = rho_g[s];
                g.vx        = vx_g[s];
                g.vy        = vy_g[s];
#ifdef dim_3D
                g.vz = vz_g[s];
#endif
                g.E = E_g[s];
            }
            return g;
        }
    };

} // namespace gradients

// face frame: n along the line between the two seeds, m and p tangential
struct geom {
    double3 n;
    double3 m;
    double3 p;
};

#endif
