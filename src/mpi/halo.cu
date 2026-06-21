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

// per-element pack/unpack bodies in namespace proteus_mpi::pack — must be at global
// scope so the namespace nests cleanly inside proteus_mpi (else it becomes
// proteus_mpi::proteus_mpi::pack).
#include "halo_packing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace proteus_mpi {

    MpiHalo halo                = {};
    int     n_mpi_capacity      = 0;
    int     n_local_initial_max = 0;

// composed from sub-files in dependency order
#include "halo_internal.cu" // shared low-level helpers (must come first)
#include "halo_build.cu"    // halo_build_exports / halo_build_used_subset / halo_remap_export_indices
#include "halo_exchange.cu" // halo_exchange_seeds / _primvars / _gradients / _v_mesh / halo_dt_allreduce
#include "halo_init.cu"     // halo_init / halo_free / halo_default_width

#ifdef USE_MPI
    // runtime growth driven by halo_build_exports overflow. Grows the halo struct's send/recv
    // buffers, the per-cell MPI-ghost arrays (mesh.seeds_g, primvar.*_g, grads.*_g), the mesh-
    // build buffers (scratch_pts / ghost_ids / sid_to_neighbor — periodic-ghost data preserved
    // across the realloc), and the KNN point-arrays. Doubles the capacity each time to amortize
    // repeat growth. printf is per-rank-0; other ranks grow silently.
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

        // mesh-build buffers + KNN follow the halo capacity so the next build pass has room
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
