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

#ifdef USE_MPI
    static int s_n_local_max = 0;
#endif
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
    // neighbor_rank -> Cart-neighbor slot index, or -1 if not a neighbor. Sized to nranks,
    // refilled on each migrate_seeds call (cheap; size <= nranks).
    static int* s_nbr_rank_to_slot     = nullptr;
    static int  s_nbr_rank_to_slot_cap = 0;
    // 1-int error signal for assign_destinations kernels (kernel can't exit_failure
    // cleanly; host checks and exits after the kernel sync).
    static int* s_assign_err       = nullptr;
    static int* s_mig_scan         = nullptr;
    static int  s_mig_scan_cap     = 0;
    static int* s_scan_scratch     = nullptr;
    static int  s_scan_scratch_cap = 0;
    static int* s_dest_pos         = nullptr;
    static int  s_dest_pos_cap     = 0;
    static int* s_chunk_tab        = nullptr;
    static int  s_chunk_tab_cap    = 0;
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

#endif // USE_MPI

    // ============================================================
    // Public entry points
    // ============================================================

    int last_n_migrated() {
        return s_last_n_migrated;
    }

    void migrate_init(int n_local_initial) {
#ifdef USE_MPI
        s_n_local_max = max_n_local(n_local_initial);
        MPI_Type_contiguous(sizeof(MigrantCell), MPI_BYTE, &s_mpi_migrant_t);
        MPI_Type_commit(&s_mpi_migrant_t);
#else
        (void)n_local_initial;
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
        mesh->n_hydro = (uint64_t)n_new;

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
        mesh->n_hydro = (uint64_t)n_new;

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

        const int  dims_x        = decomp.dims[0];
        const int  dims_y        = decomp.dims[1];
        const int  dims_z        = decomp.dims[2];
        const int* splits_x      = decomp.splits[0];
        const int* splits_y      = decomp.splits[1];
        const int* splits_z      = decomp.splits[2];
        const int* coord_to_rank = decomp.coord_to_rank;
        auto*      per_cell_slot = s_per_cell_slot;
        auto*      send_counts   = s_send_counts;
        auto*      assign_err    = s_assign_err;

        parallel_for<_MPI_PACK_BLOCK_SIZE_>("ASSIGN", n_hydro, [=] HD(int k) {
            pack::assign_destination_body(k,
                                          pts,
                                          my_rank,
                                          N_grid,
                                          bf,
                                          dims_x,
                                          dims_y,
                                          dims_z,
                                          splits_x,
                                          splits_y,
                                          splits_z,
                                          coord_to_rank,
                                          nbr_lookup,
                                          variant,
                                          per_cell_slot,
                                          send_counts,
                                          assign_err);
        });

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

    static constexpr int PACK_TABLE_BUDGET = 1 << 20; // table entries; 4 MiB at 4 B each
    static constexpr int PACK_MAX_CHUNKS   = 1024;

    static int pack_chunks_for(int nslots) {
        int c = (nslots > 0) ? (PACK_TABLE_BUDGET / nslots) : PACK_MAX_CHUNKS;
        if (c > PACK_MAX_CHUNKS) c = PACK_MAX_CHUNKS;
        if (c < 1) c = 1;
        return c;
    }

    // half-open range of the compacted list owned by chunk c
    HD inline void pack_chunk_range(int c, int m, int chunks, int* lo, int* hi) {
        const int span = (m + chunks - 1) / chunks;
        int       a    = c * span;
        int       b    = a + span;
        if (a > m) a = m;
        if (b > m) b = m;
        *lo = a;
        *hi = b;
    }

    static void build_pack_layout(int n_hydro, int nslots) {
        ensure_managed(s_mig_scan, s_mig_scan_cap, std::max(n_hydro, 1));
        ensure_managed(s_dest_pos, s_dest_pos_cap, std::max(n_hydro, 1));
        ensure_managed(s_migrant_local_k, s_migrant_local_k_cap, std::max(n_hydro, 1));

        const int* per_cell_slot = s_per_cell_slot;
        int*       off           = s_mig_scan;

        s_n_migrant_local = 0;
        if (n_hydro <= 0) return;

        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "MIG_FLAG", n_hydro, [=] HD(size_t k) { off[k] = (per_cell_slot[k] >= 0) ? 1 : 0; });

        const size_t need = scan_scratch_size((size_t)n_hydro, _MPI_PACK_BLOCK_SIZE_);
        ensure_managed(s_scan_scratch, s_scan_scratch_cap, (int)need);
        parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_, int>("MIG_SCAN", (size_t)n_hydro, off, off, s_scan_scratch);

        // exclusive scan, so the total is the last offset plus whether the last cell migrates
        const int m       = off[n_hydro - 1] + ((per_cell_slot[n_hydro - 1] >= 0) ? 1 : 0);
        s_n_migrant_local = m;
        if (m == 0) return;

        int* mig_k = s_migrant_local_k;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("MIG_COMPACT", n_hydro, [=] HD(size_t k) {
            if (per_cell_slot[k] >= 0) mig_k[off[k]] = (int)k;
        });

        const int chunks = pack_chunks_for(nslots);
        ensure_managed(s_chunk_tab, s_chunk_tab_cap, chunks * nslots);
        int*       tab         = s_chunk_tab;
        const int* send_displs = s_send_displs;
        int*       dest_pos    = s_dest_pos;

        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "MIG_TAB_ZERO", (size_t)chunks * (size_t)nslots, [=] HD(size_t i) { tab[i] = 0; });

        // chunk c owns row c outright, so counting into it needs no atomic
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("MIG_TAB_COUNT", chunks, [=] HD(size_t c) {
            int lo, hi;
            pack_chunk_range((int)c, m, chunks, &lo, &hi);
            for (int j = lo; j < hi; j++)
                tab[(size_t)c * nslots + per_cell_slot[mig_k[j]]]++;
        });

        parallel_for<_MPI_PACK_BLOCK_SIZE_>("MIG_TAB_SCAN", nslots, [=] HD(size_t sl) {
            int run = send_displs[sl];
            for (int c = 0; c < chunks; c++) {
                const size_t idx = (size_t)c * nslots + sl;
                const int    t   = tab[idx];
                tab[idx]         = run;
                run += t;
            }
        });

        parallel_for<_MPI_PACK_BLOCK_SIZE_>("MIG_TAB_POS", chunks, [=] HD(size_t c) {
            int lo, hi;
            pack_chunk_range((int)c, m, chunks, &lo, &hi);
            for (int j = lo; j < hi; j++) {
                const int k = mig_k[j];
                dest_pos[k] = tab[(size_t)c * nslots + per_cell_slot[k]]++;
            }
        });
    }

    static void pack_outgoing_migrants(
        VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new, POINT_TYPE* pts, int n_hydro, int nslots) {
        build_pack_layout(n_hydro, nslots);

        auto*   per_cell_slot = s_per_cell_slot;
        auto*   dest_pos      = s_dest_pos;
        auto*   sendbuf       = s_sendbuf;
        double* rho           = primvar->rho;
        auto*   v             = primvar->v;
        double* E             = primvar->E;
        double* rho_new       = prim_new->rho;
        auto*   v_new         = prim_new->v;
        double* E_new         = prim_new->E;
#ifdef MOVING_MESH
        auto*   v_mesh      = mesh->v_mesh;
        double* old_volumes = mesh->old_volumes;
#endif

        parallel_for<_MPI_PACK_BLOCK_SIZE_>("PACK", n_hydro, [=] HD(int k) {
            pack::pack_migrant_body(k,
                                    per_cell_slot,
                                    dest_pos,
                                    pts,
                                    rho,
                                    v,
                                    E,
                                    rho_new,
                                    v_new,
                                    E_new,
#ifdef MOVING_MESH
                                    v_mesh,
                                    old_volumes,
#endif
                                    sendbuf);
        });
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

        auto*   recvbuf          = s_recvbuf;
        auto*   seeds            = mesh->seeds;
        double* rho              = primvar->rho;
        auto*   v                = primvar->v;
        double* E                = primvar->E;
        double* rho_new          = prim_new->rho;
        auto*   v_new            = prim_new->v;
        double* E_new            = prim_new->E;
        auto*   cell_to_original = mesh->cell_to_original;
#ifdef MOVING_MESH
        auto*   v_mesh      = mesh->v_mesh;
        double* old_volumes = mesh->old_volumes;
#endif

        parallel_for<_MPI_PACK_BLOCK_SIZE_>("APPEND", total_recv, [=] HD(int j) {
            pack::unpack_migrant_body(j,
                                      n_after_remove,
                                      recvbuf,
                                      pts,
                                      seeds,
                                      rho,
                                      v,
                                      E,
                                      rho_new,
                                      v_new,
                                      E_new,
#ifdef MOVING_MESH
                                      v_mesh,
                                      old_volumes,
#endif
                                      cell_to_original);
        });
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
