#ifndef HYDRO_GRADIENTS_H
#define HYDRO_GRADIENTS_H

#include "../global/allvars.h"
#include "../voronoi/voronoi.h"
#include <cmath>
#include <cstdlib>

namespace gradients {

    // alloc / free gradient arrays
    inline void allocate_grad(size_t n, PrimGradients* g) {
        const size_t ext = (size_t)proteus_mpi::alloc_per_cell_size((int)n);
        g->rho           = gpu_alloc<POINT_TYPE>(ext);
        g->vx            = gpu_alloc<POINT_TYPE>(ext);
        g->vy            = gpu_alloc<POINT_TYPE>(ext);
#ifdef dim_3D
        g->vz = gpu_alloc<POINT_TYPE>(ext);
#endif
        g->E = gpu_alloc<POINT_TYPE>(ext);
        g->n = n; // local count; ghost slots [n, ext) filled by halo exchange

        gpu_advise_gpu_preferred(g->rho, ext * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(g->vx, ext * sizeof(POINT_TYPE));
        gpu_advise_gpu_preferred(g->vy, ext * sizeof(POINT_TYPE));
#ifdef dim_3D
        gpu_advise_gpu_preferred(g->vz, ext * sizeof(POINT_TYPE));
#endif
        gpu_advise_gpu_preferred(g->E, ext * sizeof(POINT_TYPE));
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

    // calc spatial gradients
    void compute_prim_gradients(const VMesh* mesh, const hydro::primvars* primvar, PrimGradients* grads);

    // calc "time gradients" (dW/dt)
    HD void time_gradient(hydro::prim state_i, PrimGradient grad_i, hydro::prim* dWdt);

} // namespace gradients

#endif // HYDRO_GRADIENTS_H
