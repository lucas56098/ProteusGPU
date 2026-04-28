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

} // namespace hydro

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
        size_t     n; // number of cells

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

    inline void allocate_grad(size_t n, PrimGradients* g) {
        g->rho = gpu_alloc<POINT_TYPE>(n);
        g->vx  = gpu_alloc<POINT_TYPE>(n);
        g->vy  = gpu_alloc<POINT_TYPE>(n);
#ifdef dim_3D
        g->vz = gpu_alloc<POINT_TYPE>(n);
#endif
        g->E = gpu_alloc<POINT_TYPE>(n);
        g->n = n;

        gpu_advise_gpu_preferred(g->rho, n * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(g->vx, n * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(g->vy, n * sizeof(POINT_TYPE));
#ifdef dim_3D
        gpu_advise_gpu_preferred(g->vz, n * sizeof(POINT_TYPE));
#endif
        gpu_advise_gpu_preferred(g->E, n * sizeof(POINT_TYPE));
    }

    inline void free_grad(PrimGradients* g) {
        gpu_free(g->rho);
        gpu_free(g->vx);
        gpu_free(g->vy);
#ifdef dim_3D
        gpu_free(g->vz);
#endif
        gpu_free(g->E);
        g->rho = nullptr;
        g->vx  = nullptr;
        g->vy  = nullptr;
#ifdef dim_3D
        g->vz = nullptr;
#endif
        g->E = nullptr;
        g->n = 0;
    }

    inline void zero_grad(PrimGradients* g) {
        gpu_memset(g->rho, 0, g->n * sizeof(POINT_TYPE));
        gpu_memset(g->vx, 0, g->n * sizeof(POINT_TYPE));
        gpu_memset(g->vy, 0, g->n * sizeof(POINT_TYPE));
#ifdef dim_3D
        gpu_memset(g->vz, 0, g->n * sizeof(POINT_TYPE));
#endif
        gpu_memset(g->E, 0, g->n * sizeof(POINT_TYPE));
    }

} // namespace gradients

// geometry (normalized basis for face orientation)
struct geom {
    double3 n; // normal
    double3 m; // 1. tangential
    double3 p; // 2. tangential
};

#endif // STRUCTS_H
