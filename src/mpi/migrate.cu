#include "migrate.h"

#include "decomp.h"
#include "global/structs.h"
#include "halo.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

#include <algorithm>
#include <vector>

namespace proteus_mpi {

    // per-migrant payload — all fields shipped in one MPI message
    struct MigrantCell {
        POINT_TYPE pos;
        double     rho_old;
        POINT_TYPE v_old;
        double     E_old;
        double     rho_new;
        POINT_TYPE v_new;
        double     E_new;
#ifdef MOVING_MESH
        POINT_TYPE v_mesh;
        double     old_volume;
#endif
    };

} // namespace proteus_mpi

// per-element pack/unpack bodies in namespace proteus_mpi::pack (at global scope
// so the namespace nests cleanly inside proteus_mpi).
#include "migrate_packing.h"

namespace proteus_mpi {

    static int s_n_local_max     = 0;
    static int s_last_n_migrated = 0;
#ifdef USE_MPI
    static MPI_Datatype s_mpi_migrant_t = MPI_DATATYPE_NULL;
#endif

#ifdef USE_MPI

    // persistent staging buffers — managed memory so the pack/unpack/assign kernels
    // introduced in Phase 3 can read/write without page-faulting through the host.
    // Each pair (ptr + cap) tracks an own-grown allocation; logical sizes are
    // tracked separately at each call site.
    static int*         s_send_counts         = nullptr;
    static int          s_send_counts_cap     = 0;
    static int*         s_recv_counts         = nullptr;
    static int          s_recv_counts_cap     = 0;
    static int*         s_send_displs         = nullptr;
    static int          s_send_displs_cap     = 0;
    static int*         s_recv_displs         = nullptr;
    static int          s_recv_displs_cap     = 0;
    static int*         s_per_cell_slot       = nullptr;
    static int          s_per_cell_slot_cap   = 0;
    static MigrantCell* s_sendbuf             = nullptr;
    static int          s_sendbuf_cap         = 0;
    static MigrantCell* s_recvbuf             = nullptr;
    static int          s_recvbuf_cap         = 0;
    static int*         s_migrant_local_k     = nullptr;
    static int          s_migrant_local_k_cap = 0;
    // logical count of local cells currently marked for removal (= total_send post-pack)
    static int s_n_migrant_local = 0;
    // per-slot output cursor for the pack loop (= s_send_displs initially; advanced once per migrant)
    static int* s_cursor     = nullptr;
    static int  s_cursor_cap = 0;
    // neighbor_rank -> Cart-neighbor slot index, or -1 if not a neighbor. Sized to nranks,
    // refilled on each migrate_seeds call (cheap; size <= nranks).
    static int* s_nbr_rank_to_slot     = nullptr;
    static int  s_nbr_rank_to_slot_cap = 0;
    // 1-int error signal for assign_destinations kernels (kernel can't exit_failure
    // cleanly; host checks and exits after the kernel sync).
    static int* s_assign_err = nullptr;
    // running count for s_migrant_local_k (kernel atomicAdds into it)
    static int* s_n_migrant_local_dev = nullptr;
    // 1-int counter for migrants per kernel call
#endif

    // grow a managed buffer to >= need elements (doubling, floor 64). nullptr-safe.
    template <typename T> static void ensure_managed(T*& ptr, int& cap, int need) {
        if (need <= cap) return;
        const int new_cap = std::max(64, std::max(need, 2 * cap));
        if (ptr) gpu_free(ptr);
        ptr = (T*)gpu_malloc(sizeof(T) * (size_t)new_cap);
        cap = new_cap;
    }

#ifdef USE_MPI
    // lazy-alloc single-int managed counters (called on first use).
    static void ensure_scratch_singletons() {
        if (!s_assign_err) s_assign_err = (int*)gpu_malloc(sizeof(int));
        if (!s_n_migrant_local_dev) s_n_migrant_local_dev = (int*)gpu_malloc(sizeof(int));
    }
#endif

#ifdef USE_MPI
    // forward declarations
    static int  migrate_count_tag(int dx, int dy, int dz);
    static int  migrate_payload_tag(int dx, int dy, int dz);
    static void assign_destinations(VMesh* mesh, int n_hydro, int my_rank);
    static void assign_destinations_rebal(VMesh* mesh, int n_hydro, int my_rank);
    static void exchange_counts();
    static void build_displacements(int nn, int* total_send, int* total_recv);
    static void pack_outgoing_migrants(
        VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new, POINT_TYPE* pts, int n_hydro, int nslots);
    static void exchange_payload(int total_send, int total_recv);
    static int  remove_migrated_local(
        VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new, POINT_TYPE* pts, int n_hydro);
    static void append_incoming_migrants(VMesh*           mesh,
                                         hydro::primvars* primvar,
                                         hydro::primvars* prim_new,
                                         POINT_TYPE*      pts,
                                         int              n_after_remove,
                                         int              total_recv,
                                         int              my_rank);
    static void check_conservation(int n_new);

#ifndef CPU_DEBUG
    // CUDA kernel wrappers, one per per-element body.
    GLOBAL static void kernel_assign_destinations(int               n_hydro,
                                                  const POINT_TYPE* pts,
                                                  int               my_rank,
                                                  int               N_grid_global,
                                                  double            buff,
                                                  int               dims_x,
                                                  int               dims_y,
                                                  int               dims_z,
                                                  const int*        splits_x,
                                                  const int*        splits_y,
                                                  const int*        splits_z,
                                                  const int*        coord_to_rank,
                                                  const int*        neighbor_rank_to_slot,
                                                  int               variant,
                                                  int*              per_cell_slot,
                                                  int*              send_counts,
                                                  int*              error_flag) {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_hydro) return;
        pack::assign_destination_body(k,
                                      pts,
                                      my_rank,
                                      N_grid_global,
                                      buff,
                                      dims_x,
                                      dims_y,
                                      dims_z,
                                      splits_x,
                                      splits_y,
                                      splits_z,
                                      coord_to_rank,
                                      neighbor_rank_to_slot,
                                      variant,
                                      per_cell_slot,
                                      send_counts,
                                      error_flag);
    }

    GLOBAL static void kernel_pack_migrants(int               n_hydro,
                                            const int*        per_cell_slot,
                                            const POINT_TYPE* pts,
                                            const double*     primvar_rho,
                                            const POINT_TYPE* primvar_v,
                                            const double*     primvar_E,
                                            const double*     prim_new_rho,
                                            const POINT_TYPE* prim_new_v,
                                            const double*     prim_new_E,
#ifdef MOVING_MESH
                                            const POINT_TYPE* v_mesh,
                                            const double*     old_volumes,
#endif
                                            int*         cursor,
                                            MigrantCell* sendbuf,
                                            int*         n_migrant_local_counter,
                                            int*         migrant_local_k) {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_hydro) return;
        pack::pack_migrant_body(k,
                                per_cell_slot,
                                pts,
                                primvar_rho,
                                primvar_v,
                                primvar_E,
                                prim_new_rho,
                                prim_new_v,
                                prim_new_E,
#ifdef MOVING_MESH
                                v_mesh,
                                old_volumes,
#endif
                                cursor,
                                sendbuf,
                                n_migrant_local_counter,
                                migrant_local_k);
    }

    GLOBAL static void kernel_append_migrants(int                total_recv,
                                              int                n_after_remove,
                                              const MigrantCell* recvbuf,
                                              POINT_TYPE*        pts,
                                              double3*           seeds,
                                              double*            primvar_rho,
                                              POINT_TYPE*        primvar_v,
                                              double*            primvar_E,
                                              double*            prim_new_rho,
                                              POINT_TYPE*        prim_new_v,
                                              double*            prim_new_E,
#ifdef MOVING_MESH
                                              POINT_TYPE* v_mesh,
                                              double*     old_volumes,
#endif
                                              unsigned int* cell_to_original) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= total_recv) return;
        pack::unpack_migrant_body(j,
                                  n_after_remove,
                                  recvbuf,
                                  pts,
                                  seeds,
                                  primvar_rho,
                                  primvar_v,
                                  primvar_E,
                                  prim_new_rho,
                                  prim_new_v,
                                  prim_new_E,
#ifdef MOVING_MESH
                                  v_mesh,
                                  old_volumes,
#endif
                                  cell_to_original);
    }
#endif // !CPU_DEBUG
#endif // USE_MPI

    // ============================================================
    // Public entry points
    // ============================================================

    int last_n_migrated() {
        return s_last_n_migrated;
    }

    void migrate_init(int n_local_initial) {
        s_n_local_max = max_n_local(n_local_initial);
#ifdef USE_MPI
        MPI_Type_contiguous(sizeof(MigrantCell), MPI_BYTE, &s_mpi_migrant_t);
        MPI_Type_commit(&s_mpi_migrant_t);
#endif
    }

    // rebalance variant: any-rank destination via decomp_owner_of_bucket (using updated splits);
    // counts/payload over the full Cart comm. Reuses the existing pack/remove/append/check helpers.
    // Called from voronoi::move_mesh after advance_seeds_by_dt has populated scratch_move with
    // post-advance positions — those are the positions we redistribute against.
    void migrate_for_rebalance(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new) {
#ifndef USE_MPI
        (void)mesh;
        (void)primvar;
        (void)prim_new;
        return;
#else
        if (decomp.nranks <= 1) return;

        PROFILE("MIGRATE_REBAL");

        const int    my_rank = decomp.rank;
        const int    nr      = decomp.nranks;
        const int    n_hydro = (int)mesh->n_hydro;
        const int    N_grid  = decomp.N_grid_global;
        const double bf      = mesh->buff;

        POINT_TYPE* pts = mesh->scratch_move;

        // assign destination rank per cell — any rank, no Cart-neighbor restriction.
        (void)N_grid;
        (void)bf;
        assign_destinations_rebal(mesh, n_hydro, my_rank);

        // exchange counts over the full Cart comm (every rank can talk to every other rank).
        ensure_managed(s_recv_counts, s_recv_counts_cap, nr);
        for (int r = 0; r < nr; r++)
            s_recv_counts[r] = 0;
        mpi_sync_before_send(s_send_counts, sizeof(int) * (size_t)nr);
        {
            PROFILE_MPI("COUNTS_WAIT");
            MPI_Alltoall(s_send_counts, 1, MPI_INT, s_recv_counts, 1, MPI_INT, decomp.cart_comm);
        }
        mpi_sync_after_recv(s_recv_counts, sizeof(int) * (size_t)nr);

        int total_send = 0, total_recv = 0;
        build_displacements(nr, &total_send, &total_recv);
        s_last_n_migrated = total_send;

        pack_outgoing_migrants(mesh, primvar, prim_new, pts, n_hydro, nr);

        // payload via single Alltoallv over the full Cart comm.
        mpi_sync_before_send(s_sendbuf, sizeof(MigrantCell) * (size_t)total_send);
        {
            PROFILE_MPI("PAYLOAD_WAIT");
            MPI_Alltoallv(s_sendbuf,
                          s_send_counts,
                          s_send_displs,
                          s_mpi_migrant_t,
                          s_recvbuf,
                          s_recv_counts,
                          s_recv_displs,
                          s_mpi_migrant_t,
                          decomp.cart_comm);
        }
        mpi_sync_after_recv(s_recvbuf, sizeof(MigrantCell) * (size_t)total_recv);

        const int n_after_remove = remove_migrated_local(mesh, primvar, prim_new, pts, n_hydro);

        const int n_new = n_after_remove + total_recv;
        if (n_new > s_n_local_max) {
            // No adaptive realloc yet — every per-cell buffer (mesh, primvars, gradients)
            // would need to grow in lockstep, which is a larger refactor. For homogeneous
            // runs ALLOC_GROWTH=2.0 is generous; if this still trips, either the IC has
            // a strong inhomogeneity we didn't expect or the imbalance threshold is too
            // loose. Restart from the last snapshot after bumping ALLOC_GROWTH.
            exit_failure("[rank %d] REBALANCE: n_hydro_new=%d > n_local_max=%d "
                         "(post-rebalance migration overflows per-cell capacity). "
                         "Bump ALLOC_GROWTH in src/global/structs.h or tighten "
                         "imbalance_threshold in param.txt, then restart from the last snapshot.\n",
                         my_rank,
                         n_new,
                         s_n_local_max);
        }

        append_incoming_migrants(mesh, primvar, prim_new, pts, n_after_remove, total_recv, my_rank);
        mesh->n_hydro = (hsize_t)n_new;

        check_conservation(n_new);
#endif
    }

    void migrate_seeds(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new) {
#ifndef USE_MPI
        (void)mesh;
        (void)primvar;
        (void)prim_new;
        return;
#else
        if (decomp.nranks <= 1) return;

        PROFILE("MIGRATE");

        const int   my_rank = decomp.rank;
        const int   nn      = halo.n_neighbors;
        const int   n_hydro = (int)mesh->n_hydro;
        POINT_TYPE* pts     = mesh->scratch_move;

        assign_destinations(mesh, n_hydro, my_rank);

        {
            PROFILE_MPI("COUNTS_WAIT");
            exchange_counts();
        }

        int total_send = 0, total_recv = 0;
        build_displacements(nn, &total_send, &total_recv);
        s_last_n_migrated = total_send;

        pack_outgoing_migrants(mesh, primvar, prim_new, pts, n_hydro, nn);

        {
            PROFILE_MPI("PAYLOAD_WAIT");
            exchange_payload(total_send, total_recv);
        }

        const int n_after_remove = remove_migrated_local(mesh, primvar, prim_new, pts, n_hydro);

        const int n_new = n_after_remove + total_recv;
        if (n_new > s_n_local_max) {
            // See the REBALANCE overflow note above — same constraint, same fix.
            exit_failure("[rank %d] MIGRATE: n_hydro_new=%d > n_local_max=%d "
                         "(per-step Cart-neighbor migration overflows per-cell capacity). "
                         "Bump ALLOC_GROWTH in src/global/structs.h or enable rebalance "
                         "with a tighter imbalance_threshold, then restart from the last snapshot.\n",
                         my_rank,
                         n_new,
                         s_n_local_max);
        }

        append_incoming_migrants(mesh, primvar, prim_new, pts, n_after_remove, total_recv, my_rank);
        mesh->n_hydro = (hsize_t)n_new;

        check_conservation(n_new);
#endif
    }

#ifdef USE_MPI

    // ============================================================
    // Static helpers
    // ============================================================

    // dir-encoded message tags, offset so they don't collide with halo's
    static int migrate_count_tag(int dx, int dy, int dz) {
        return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1) + 1 + 500;
    }
    static int migrate_payload_tag(int dx, int dy, int dz) {
        return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1) + 1 + 600;
    }

    // Shared assign-destinations dispatcher. variant=0 (per-step): owner must be a Cart
    // neighbor (slot = neighbor index). variant=1 (rebalance): owner IS the slot.
    static void assign_destinations_dispatch(VMesh* mesh, int n_hydro, int my_rank, int variant, int nslots) {
        const int    N_grid = decomp.N_grid_global;
        const double bf     = mesh->buff;
        POINT_TYPE*  pts    = mesh->scratch_move;

        ensure_managed(s_send_counts, s_send_counts_cap, nslots);
        ensure_managed(s_per_cell_slot, s_per_cell_slot_cap, n_hydro);
        for (int n = 0; n < nslots; n++)
            s_send_counts[n] = 0;

        // build neighbor_rank -> slot lookup for variant 0 (per-step)
        const int* nbr_lookup = nullptr;
        if (variant == 0) {
            const int nr = decomp.nranks;
            ensure_managed(s_nbr_rank_to_slot, s_nbr_rank_to_slot_cap, nr);
            for (int r = 0; r < nr; r++)
                s_nbr_rank_to_slot[r] = -1;
            for (int n = 0; n < halo.n_neighbors; n++) {
                s_nbr_rank_to_slot[halo.neighbor_ranks[n]] = n;
            }
            nbr_lookup = s_nbr_rank_to_slot;
        }

        ensure_scratch_singletons();
        *s_assign_err = 0;

#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (n_hydro + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("ASSIGN");
            kernel_assign_destinations<<<blocks, tpb>>>(n_hydro,
                                                        pts,
                                                        my_rank,
                                                        N_grid,
                                                        bf,
                                                        decomp.dims[0],
                                                        decomp.dims[1],
                                                        decomp.dims[2],
                                                        decomp.splits[0],
                                                        decomp.splits[1],
                                                        decomp.splits[2],
                                                        decomp.coord_to_rank,
                                                        nbr_lookup,
                                                        variant,
                                                        s_per_cell_slot,
                                                        s_send_counts,
                                                        s_assign_err);
        }
        GPU_SYNC();
        GPU_SYNC(); // need *s_assign_err host-visible for the error check below
#else
        {
            PROFILE("ASSIGN");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < n_hydro; k++) {
                pack::assign_destination_body(k,
                                              pts,
                                              my_rank,
                                              N_grid,
                                              bf,
                                              decomp.dims[0],
                                              decomp.dims[1],
                                              decomp.dims[2],
                                              decomp.splits[0],
                                              decomp.splits[1],
                                              decomp.splits[2],
                                              decomp.coord_to_rank,
                                              nbr_lookup,
                                              variant,
                                              s_per_cell_slot,
                                              s_send_counts,
                                              s_assign_err);
            }
        }
#endif

        if (*s_assign_err == 1) {
            exit_failure("[rank %d] %s: invalid owner for some migrating cell. Bucket coords out of range; "
                         "check decomp/buff configuration.\n",
                         my_rank,
                         (variant == 1) ? "REBALANCE" : "MIGRATE");
        }
        if (*s_assign_err == 2) {
            exit_failure("[rank %d] MIGRATE: some cell would migrate to a non-Cart-neighbor rank. "
                         "Cells must not cross more than one bucket per step (CFL).\n",
                         my_rank);
        }
    }

    static void assign_destinations(VMesh* mesh, int n_hydro, int my_rank) {
        assign_destinations_dispatch(mesh, n_hydro, my_rank, /*variant=*/0, halo.n_neighbors);
    }

    static void assign_destinations_rebal(VMesh* mesh, int n_hydro, int my_rank) {
        assign_destinations_dispatch(mesh, n_hydro, my_rank, /*variant=*/1, decomp.nranks);
    }

    // neighbor-only count exchange (Neighbor_alltoall when peers distinct,
    // Isend/Irecv per direction otherwise — never MPI_Alltoall over the full comm)
    static void exchange_counts() {
        const int nn = halo.n_neighbors;
        ensure_managed(s_recv_counts, s_recv_counts_cap, nn);
        for (int n = 0; n < nn; n++)
            s_recv_counts[n] = 0;
        mpi_sync_before_send(s_send_counts, sizeof(int) * (size_t)nn);
        if (halo.use_neighbor_coll) {
            MPI_Neighbor_alltoall(s_send_counts, 1, MPI_INT, s_recv_counts, 1, MPI_INT, halo.graph_comm);
            mpi_sync_after_recv(s_recv_counts, sizeof(int) * (size_t)nn);
            return;
        }
        MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
        int         n_reqs = 0;
        for (int n = 0; n < nn; n++) {
            const int dx   = halo.neighbor_dirs[n][0];
            const int dy   = halo.neighbor_dirs[n][1];
            const int dz   = halo.neighbor_dirs[n][2];
            const int peer = halo.neighbor_ranks[n];
            MPI_Isend(
                &s_send_counts[n], 1, MPI_INT, peer, migrate_count_tag(dx, dy, dz), decomp.cart_comm, &reqs[n_reqs++]);
            MPI_Irecv(&s_recv_counts[n],
                      1,
                      MPI_INT,
                      peer,
                      migrate_count_tag(-dx, -dy, -dz),
                      decomp.cart_comm,
                      &reqs[n_reqs++]);
        }
        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
        mpi_sync_after_recv(s_recv_counts, sizeof(int) * (size_t)nn);
    }

    static void build_displacements(int nn, int* total_send, int* total_recv) {
        ensure_managed(s_send_displs, s_send_displs_cap, nn);
        ensure_managed(s_recv_displs, s_recv_displs_cap, nn);
        int ts = 0, tr = 0;
        for (int n = 0; n < nn; n++) {
            s_send_displs[n] = ts;
            s_recv_displs[n] = tr;
            ts += s_send_counts[n];
            tr += s_recv_counts[n];
        }
        // grow with a floor so a transient 0-length call doesn't free + reallocate
        // a previously-sized buffer (cap-doubling growth, no shrink).
        ensure_managed(s_sendbuf, s_sendbuf_cap, std::max(ts, 1));
        ensure_managed(s_recvbuf, s_recvbuf_cap, std::max(tr, 1));
        *total_send = ts;
        *total_recv = tr;
    }

    static void pack_outgoing_migrants(
        VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new, POINT_TYPE* pts, int n_hydro, int nslots) {
        ensure_managed(s_migrant_local_k, s_migrant_local_k_cap, std::max(n_hydro, 1));
        ensure_managed(s_cursor, s_cursor_cap, std::max(nslots, 1));
        for (int i = 0; i < nslots; i++)
            s_cursor[i] = s_send_displs[i];
        ensure_scratch_singletons();
        *s_n_migrant_local_dev = 0;

#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (n_hydro + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("PACK");
            kernel_pack_migrants<<<blocks, tpb>>>(n_hydro,
                                                  s_per_cell_slot,
                                                  pts,
                                                  primvar->rho,
                                                  primvar->v,
                                                  primvar->E,
                                                  prim_new->rho,
                                                  prim_new->v,
                                                  prim_new->E,
#ifdef MOVING_MESH
                                                  mesh->v_mesh,
                                                  mesh->old_volumes,
#endif
                                                  s_cursor,
                                                  s_sendbuf,
                                                  s_n_migrant_local_dev,
                                                  s_migrant_local_k);
        }
        GPU_SYNC();
        GPU_SYNC(); // need *s_n_migrant_local_dev host-visible (used by remove_migrated_local)
#else
        {
            PROFILE("PACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < n_hydro; k++) {
                pack::pack_migrant_body(k,
                                        s_per_cell_slot,
                                        pts,
                                        primvar->rho,
                                        primvar->v,
                                        primvar->E,
                                        prim_new->rho,
                                        prim_new->v,
                                        prim_new->E,
#ifdef MOVING_MESH
                                        mesh->v_mesh,
                                        mesh->old_volumes,
#endif
                                        s_cursor,
                                        s_sendbuf,
                                        s_n_migrant_local_dev,
                                        s_migrant_local_k);
            }
        }
#endif
        s_n_migrant_local = *s_n_migrant_local_dev;
#ifndef MOVING_MESH
        (void)mesh;
#endif
    }

    // single neighbor-only collective for all MigrantCell fields
    static void exchange_payload(int total_send, int total_recv) {
        const int nn = halo.n_neighbors;
        mpi_sync_before_send(s_sendbuf, sizeof(MigrantCell) * (size_t)total_send);
        if (halo.use_neighbor_coll) {
            MPI_Neighbor_alltoallv(s_sendbuf,
                                   s_send_counts,
                                   s_send_displs,
                                   s_mpi_migrant_t,
                                   s_recvbuf,
                                   s_recv_counts,
                                   s_recv_displs,
                                   s_mpi_migrant_t,
                                   halo.graph_comm);
            mpi_sync_after_recv(s_recvbuf, sizeof(MigrantCell) * (size_t)total_recv);
            return;
        }
        MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
        int         n_reqs = 0;
        for (int n = 0; n < nn; n++) {
            const int dx   = halo.neighbor_dirs[n][0];
            const int dy   = halo.neighbor_dirs[n][1];
            const int dz   = halo.neighbor_dirs[n][2];
            const int peer = halo.neighbor_ranks[n];
            const int sc   = s_send_counts[n];
            const int rc   = s_recv_counts[n];
            if (sc > 0) {
                MPI_Isend(s_sendbuf + s_send_displs[n],
                          sc,
                          s_mpi_migrant_t,
                          peer,
                          migrate_payload_tag(dx, dy, dz),
                          decomp.cart_comm,
                          &reqs[n_reqs++]);
            }
            if (rc > 0) {
                MPI_Irecv(s_recvbuf + s_recv_displs[n],
                          rc,
                          s_mpi_migrant_t,
                          peer,
                          migrate_payload_tag(-dx, -dy, -dz),
                          decomp.cart_comm,
                          &reqs[n_reqs++]);
            }
        }
        if (n_reqs > 0) MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
        mpi_sync_after_recv(s_recvbuf, sizeof(MigrantCell) * (size_t)total_recv);
    }

    // remove migrated cells via swap-with-last (largest k first)
    static int remove_migrated_local(
        VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new, POINT_TYPE* pts, int n_hydro) {
        // s_migrant_local_k lives in managed memory; under CUDA the pack kernel will
        // have written into it. Sort + the swap-with-last loop both run on host because
        // the index array is small and access is irregular — not worth a kernel.
        std::sort(s_migrant_local_k, s_migrant_local_k + s_n_migrant_local, std::greater<int>());
        int n_after = n_hydro;
        for (int i = 0; i < s_n_migrant_local; i++) {
            const int k_remove = s_migrant_local_k[i];
            const int k_last   = n_after - 1;
            if (k_remove != k_last) {
                pts[k_remove]           = pts[k_last];
                primvar->rho[k_remove]  = primvar->rho[k_last];
                primvar->v[k_remove]    = primvar->v[k_last];
                primvar->E[k_remove]    = primvar->E[k_last];
                prim_new->rho[k_remove] = prim_new->rho[k_last];
                prim_new->v[k_remove]   = prim_new->v[k_last];
                prim_new->E[k_remove]   = prim_new->E[k_last];
#ifdef MOVING_MESH
                mesh->v_mesh[k_remove]      = mesh->v_mesh[k_last];
                mesh->old_volumes[k_remove] = mesh->old_volumes[k_last];
#endif
                mesh->cell_to_original[k_remove] = mesh->cell_to_original[k_last];
            }
            n_after--;
        }
        return n_after;
    }

    static void append_incoming_migrants(VMesh*           mesh,
                                         hydro::primvars* primvar,
                                         hydro::primvars* prim_new,
                                         POINT_TYPE*      pts,
                                         int              n_after_remove,
                                         int              total_recv,
                                         int              my_rank) {
        (void)my_rank;
        if (total_recv <= 0) return;

#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (total_recv + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("APPEND");
            kernel_append_migrants<<<blocks, tpb>>>(total_recv,
                                                    n_after_remove,
                                                    s_recvbuf,
                                                    pts,
                                                    mesh->seeds,
                                                    primvar->rho,
                                                    primvar->v,
                                                    primvar->E,
                                                    prim_new->rho,
                                                    prim_new->v,
                                                    prim_new->E,
#ifdef MOVING_MESH
                                                    mesh->v_mesh,
                                                    mesh->old_volumes,
#endif
                                                    mesh->cell_to_original);
        }
        GPU_SYNC();
#else
        {
            PROFILE("APPEND");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j < total_recv; j++) {
                pack::unpack_migrant_body(j,
                                          n_after_remove,
                                          s_recvbuf,
                                          pts,
                                          mesh->seeds,
                                          primvar->rho,
                                          primvar->v,
                                          primvar->E,
                                          prim_new->rho,
                                          prim_new->v,
                                          prim_new->E,
#ifdef MOVING_MESH
                                          mesh->v_mesh,
                                          mesh->old_volumes,
#endif
                                          mesh->cell_to_original);
            }
        }
#endif
    }

    // global cell count must stay constant. Long long because the global sum is
    // n_global (few_thousand^3), well past int32.
    static void check_conservation(int n_new) {
        const long long n_new_ll = (long long)n_new;
        long long       n_global = 0;
        {
            PROFILE_MPI("CONS_ALLREDUCE");
            MPI_Allreduce(&n_new_ll, &n_global, 1, MPI_LONG_LONG, MPI_SUM, decomp.cart_comm);
        }
        static long long s_n_total_expected = 0;
        if (s_n_total_expected == 0) s_n_total_expected = n_global;
        if (n_global != s_n_total_expected) {
            exit_failure("MIGRATE: FATAL global cell-count drift: %lld != %lld. A cell was duplicated or lost.\n",
                         n_global,
                         s_n_total_expected);
        }
    }

#endif // USE_MPI

} // namespace proteus_mpi
