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
        needs_exact_predicates      = 5, // to fix perturb seedpoints
        // The cell IS closed against every seed this rank holds, but its bounding sphere
        // reaches far enough that a seed outside the rank's data extent could still clip it.
        // Distinct from security_radius_not_reached on purpose: this is the ONE failure a
        // wider MPI halo can repair, and it is the only status the widen-W loop iterates on.
        // Meaningless on a single rank, where the periodic ghost band covers everything.
        security_radius_beyond_data = 6
    };

} // namespace voronoi

namespace hydro {

    // hydro primitive variable arrays
    struct primvars {
        // real-cell SoA storage, size max_n_local. indexed [0, n_hydro).
        double*     rho;
        POINT_TYPE* v;
        double*     E; // per unit volume

        // MPI ghost SoA storage, size proteus_mpi::n_mpi_capacity. indexed [0, n_mpi_ghosts).
        // populated by halo_exchange_primvars; nullptr on single-rank builds.
        // grows independently of the real arrays via proteus_mpi::grow_ghost_arrays.
        double*     rho_g;
        POINT_TYPE* v_g;
        double*     E_g;
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

    // global ceiling on n_local across ranks. Set once in begrun via Allreduce(MAX) on the
    // per-rank initial count; 0 means "not yet set, fall back to the n_initial arg".
    // Sizing all per-cell buffers from this rather than per-rank n_initial lets sparse ranks
    // receive migrants from denser ones during rebalance without overflowing.
    extern int n_local_initial_max;

    // migration headroom on n_local. Sized for "homogeneous + occasional rebalance":
    // a 1.10 imbalance threshold (sim.imbalance_threshold default) lets one rank grow to
    // ~10% past its initial n_local before rebalance fires, and the split chooser can
    // overshoot at bucket granularity. 2.0 gives generous headroom (100%) for both
    // the steady-state imbalance and any transient spikes during a rebalance step.
    // If a run still hits the "n_hydro_new > n_local_max" exit from migrate.cu,
    // either the IC is far from homogeneous or this needs bumping further.
    constexpr double ALLOC_GROWTH = 2.0;

    inline int max_n_local(int n_initial) {
        const int base = (n_local_initial_max > 0) ? n_local_initial_max : n_initial;
        return (int)((double)base * ALLOC_GROWTH);
    }

    // runtime live size — local cells only; ghosts live in separate _g arrays now.
    inline int extended_size(int n_local) {
        return n_local;
    }

    // allocated per-cell size — local-only; MPI ghosts live in separate _g buffers
    // sized to n_mpi_capacity, which is grown independently by halo_grow_capacity.
    inline int alloc_per_cell_size(int n_initial) {
        return max_n_local(n_initial);
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

        // MPI ghost SoA storage, size proteus_mpi::n_mpi_capacity. populated by
        // halo_exchange_gradients; nullptr on single-rank.
        POINT_TYPE* rho_g;
        POINT_TYPE* vx_g;
        POINT_TYPE* vy_g;
#ifdef dim_3D
        POINT_TYPE* vz_g;
#endif
        POINT_TYPE* E_g;

        // load single-cell gradients from real SoA arrays (own cell, never a ghost)
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

        // load with possible ghost-region read; k may be a neighbor (>= n_hydro)
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

// geometry (normalized basis for face orientation)
struct geom {
    double3 n; // normal
    double3 m; // 1. tangential
    double3 p; // 2. tangential
};

#endif // STRUCTS_H
