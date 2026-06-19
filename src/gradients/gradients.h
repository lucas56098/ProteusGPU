#ifndef HYDRO_GRADIENTS_H
#define HYDRO_GRADIENTS_H

#include "../global/allvars.h"
#include "../voronoi/voronoi.h"
#include <cmath>
#include <cstdlib>

namespace gradients {

    // alloc real arrays (sized to max_n_local) + ghost arrays (sized to n_mpi_capacity)
    inline void allocate_grad(size_t n, PrimGradients* g) {
        const size_t ext = (size_t)proteus_mpi::alloc_per_cell_size((int)n);
        g->rho           = gpu_alloc<POINT_TYPE>(ext);
        g->vx            = gpu_alloc<POINT_TYPE>(ext);
        g->vy            = gpu_alloc<POINT_TYPE>(ext);
#ifdef dim_3D
        g->vz = gpu_alloc<POINT_TYPE>(ext);
#endif
        g->E = gpu_alloc<POINT_TYPE>(ext);
        g->n = n;

        gpu_advise_gpu_preferred(g->rho, ext * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(g->vx, ext * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(g->vy, ext * sizeof(POINT_TYPE));
#ifdef dim_3D
        gpu_advise_gpu_preferred(g->vz, ext * sizeof(POINT_TYPE));
#endif
        gpu_advise_gpu_preferred(g->E, ext * sizeof(POINT_TYPE));

        const int gc = proteus_mpi::n_mpi_capacity;
        if (gc > 0) {
            g->rho_g = gpu_alloc<POINT_TYPE>(gc);
            g->vx_g  = gpu_alloc<POINT_TYPE>(gc);
            g->vy_g  = gpu_alloc<POINT_TYPE>(gc);
#ifdef dim_3D
            g->vz_g = gpu_alloc<POINT_TYPE>(gc);
#endif
            g->E_g = gpu_alloc<POINT_TYPE>(gc);
            gpu_advise_gpu_preferred(g->rho_g, gc * sizeof(POINT_TYPE));
            gpu_advise_gpu_preferred(g->vx_g, gc * sizeof(POINT_TYPE));
            gpu_advise_gpu_preferred(g->vy_g, gc * sizeof(POINT_TYPE));
#ifdef dim_3D
            gpu_advise_gpu_preferred(g->vz_g, gc * sizeof(POINT_TYPE));
#endif
            gpu_advise_gpu_preferred(g->E_g, gc * sizeof(POINT_TYPE));
        } else {
            g->rho_g = nullptr;
            g->vx_g  = nullptr;
            g->vy_g  = nullptr;
#ifdef dim_3D
            g->vz_g = nullptr;
#endif
            g->E_g = nullptr;
        }
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

        if (g->rho_g) gpu_free(g->rho_g);
        if (g->vx_g) gpu_free(g->vx_g);
        if (g->vy_g) gpu_free(g->vy_g);
#ifdef dim_3D
        if (g->vz_g) gpu_free(g->vz_g);
#endif
        if (g->E_g) gpu_free(g->E_g);
        g->rho_g = nullptr;
        g->vx_g  = nullptr;
        g->vy_g  = nullptr;
#ifdef dim_3D
        g->vz_g = nullptr;
#endif
        g->E_g = nullptr;
    }

    // resize the ghost arrays to new_cap. Contents discarded; halo_exchange_gradients
    // repopulates them. Called by proteus_mpi::halo_grow_capacity.
    inline void grad_grow_ghosts(PrimGradients* g, int new_cap) {
        if (g->rho_g) gpu_free(g->rho_g);
        if (g->vx_g) gpu_free(g->vx_g);
        if (g->vy_g) gpu_free(g->vy_g);
#ifdef dim_3D
        if (g->vz_g) gpu_free(g->vz_g);
#endif
        if (g->E_g) gpu_free(g->E_g);
        g->rho_g = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
        g->vx_g  = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
        g->vy_g  = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
#ifdef dim_3D
        g->vz_g = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
#endif
        g->E_g = (new_cap > 0) ? gpu_alloc<POINT_TYPE>(new_cap) : nullptr;
    }

    // calc spatial gradients
    void compute_prim_gradients(const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads);

    // calc "time gradients" (dW/dt)
    HD void time_gradient(hydro::prim state_i, PrimGradient grad_i, hydro::prim* dWdt);

} // namespace gradients

#endif // HYDRO_GRADIENTS_H
