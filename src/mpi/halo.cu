// the MPI halo (halo.h)

#include "halo.h"

#include "decomp.h"
#include "global/allvars.h"
#include "global/structs.h"
#include "gradients/gradients.h"
#include "hydro/finite_volume_solver.h"
#include "knn/knn.h"
#include "profiler/profiler.h"
#include "rebalance.h"
#include "voronoi/voronoi.h"

#include "halo_packing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace proteus_mpi {

    MpiHalo halo                = {};
    int     n_mpi_capacity      = 0;
    int     n_local_initial_max = 0;
    double  alloc_growth        = 0.0;

    // clang-format off
    // one translation unit, so the include order matters
    #include "halo_internal.cu"
    #include "halo_build.cu"
    #include "halo_exchange.cu"
    #include "halo_init.cu"
    // clang-format on

#ifdef USE_MPI
    // the estimate was too small: at least double, and take everything that is sized by it along
    void halo_grow_capacity(int new_capacity) {
        const int old_cap = halo.n_mpi_capacity;
        const int target  = std::max(new_capacity, std::max(1, 2 * old_cap));

        free_halo_buffers();
        allocate_halo_buffers(target);

        halo.n_mpi_capacity = target;
        n_mpi_capacity      = target;

        if (sim.mesh) voronoi::mesh_grow_ghosts(sim.mesh, target);
        if (sim.primvar) hydro::primvar_grow_ghosts(sim.primvar, target);
        if (sim.grads) gradients::grad_grow_ghosts(sim.grads, target);

        if (sim.mesh) {
            voronoi::mesh_grow_build_buffers(sim.mesh, target);
            if (sim.mesh->knn) knn::knn_grow(sim.mesh->knn, (int)sim.mesh->total_capacity);
        }

        if (decomp.rank == 0) {
            printf("HALO: grew n_mpi_capacity %d -> %d.\n", old_cap, target);
            fflush(stdout);
        }
    }
#else
    void halo_grow_capacity(int new_capacity) {
        (void)new_capacity;
    }
#endif

} // namespace proteus_mpi
