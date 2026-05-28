#include "migrate.h"

#include "decomp.h"
#include "halo.h"
#include "global/structs.h"
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

    static int s_n_local_max     = 0;
    static int s_last_n_migrated = 0;
#ifdef USE_MPI
    static MPI_Datatype s_mpi_migrant_t = MPI_DATATYPE_NULL;
#endif

    // persistent staging buffers — sized once, reused every call
    static std::vector<int>         s_send_counts;
    static std::vector<int>         s_recv_counts;
    static std::vector<int>         s_send_displs;
    static std::vector<int>         s_recv_displs;
    static std::vector<int>         s_per_cell_slot;
    static std::vector<MigrantCell> s_sendbuf;
    static std::vector<MigrantCell> s_recvbuf;
    static std::vector<int>         s_migrant_local_k;

#ifdef USE_MPI
    // forward declarations
    static int  neighbor_index_of(int dest_rank);
    static int  migrate_count_tag(int dx, int dy, int dz);
    static int  migrate_payload_tag(int dx, int dy, int dz);
    static void assign_destinations(VMesh* mesh, int n_hydro, int my_rank);
    static void exchange_counts();
    static void build_displacements(int nn, int* total_send, int* total_recv);
    static void pack_outgoing_migrants(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new,
                                       POINT_TYPE* pts, int n_hydro);
    static void exchange_payload();
    static int  remove_migrated_local(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new,
                                      POINT_TYPE* pts, int n_hydro);
    static void append_incoming_migrants(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new,
                                         POINT_TYPE* pts, int n_after_remove, int total_recv, int my_rank);
    static void check_conservation(int n_new);
#endif

    // ============================================================
    // Public entry points
    // ============================================================

    int last_n_migrated() { return s_last_n_migrated; }

    void migrate_init(int n_local_initial) {
        s_n_local_max = max_n_local(n_local_initial);
#ifdef USE_MPI
        MPI_Type_contiguous(sizeof(MigrantCell), MPI_BYTE, &s_mpi_migrant_t);
        MPI_Type_commit(&s_mpi_migrant_t);
#endif
    }

    void migrate_seeds(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new) {
#ifndef USE_MPI
        (void)mesh; (void)primvar; (void)prim_new;
        return;
#else
        if (decomp.nranks <= 1) return;

        const int   my_rank = decomp.rank;
        const int   nn      = halo.n_neighbors;
        const int   n_hydro = (int)mesh->n_hydro;
        POINT_TYPE* pts     = mesh->scratch_move;

        assign_destinations(mesh, n_hydro, my_rank);

        Profiler::StartTimer("MPI_COMM");
        Profiler::StartTimer("MPI_WAIT");
        exchange_counts();
        Profiler::EndTimer("MPI_WAIT");
        Profiler::EndTimer("MPI_COMM");

        int total_send = 0, total_recv = 0;
        build_displacements(nn, &total_send, &total_recv);
        s_last_n_migrated = total_send;

        pack_outgoing_migrants(mesh, primvar, prim_new, pts, n_hydro);

        Profiler::StartTimer("MPI_COMM");
        Profiler::StartTimer("MPI_WAIT");
        exchange_payload();
        Profiler::EndTimer("MPI_WAIT");
        Profiler::EndTimer("MPI_COMM");

        const int n_after_remove = remove_migrated_local(mesh, primvar, prim_new, pts, n_hydro);

        const int n_new = n_after_remove + total_recv;
        if (n_new > s_n_local_max) {
            exit_failure("[rank %d] MIGRATE: n_hydro_new=%d > n_local_max=%d. "
                "Increase ALLOC_GROWTH in src/global/structs.h or rebalance the IC.\n",
                my_rank, n_new, s_n_local_max);
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

    // cells can only migrate to direct neighbors (movement < 1 bucket per step under CFL)
    static int neighbor_index_of(int dest_rank) {
        const int nn = halo.n_neighbors;
        for (int n = 0; n < nn; n++) {
            if (halo.neighbor_ranks[n] == dest_rank) return n;
        }
        return -1;
    }

    // dir-encoded message tags, offset so they don't collide with halo's
    static int migrate_count_tag(int dx, int dy, int dz) {
        return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1) + 1 + 500;
    }
    static int migrate_payload_tag(int dx, int dy, int dz) {
        return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1) + 1 + 600;
    }

    // per-cell destination as neighbor slot (no global-rank fallback)
    static void assign_destinations(VMesh* mesh, int n_hydro, int my_rank) {
        const int    nn     = halo.n_neighbors;
        const int    N_grid = decomp.N_grid_global;
        const double bf     = mesh->buff;
        POINT_TYPE*  pts    = mesh->scratch_move;

        s_send_counts.assign(nn, 0);
        s_per_cell_slot.assign(n_hydro, -1);

        for (int k = 0; k < n_hydro; k++) {
            const double px = pts[k].x;
            const double py = pts[k].y;
#ifdef dim_3D
            const double pz = pts[k].z;
#else
            const double pz = 0.0;
#endif
            int bx, by, bz;
            decomp_bucket_of_point(px, py, pz, N_grid, bf, &bx, &by, &bz);
            const int owner = decomp_owner_of_bucket(bx, by, bz);
            if (owner == my_rank) continue;
            if (owner < 0) {
                exit_failure("[rank %d] MIGRATE: invalid owner for cell %d at (%g,%g,%g) → bucket (%d,%d,%d).\n",
                    my_rank, k, px, py, pz, bx, by, bz);
            }
            const int slot = neighbor_index_of(owner);
            if (slot < 0) {
                exit_failure("[rank %d] MIGRATE: cell %d would migrate to rank %d which is not a Cart neighbor. "
                    "Cells must not cross more than one bucket per step (CFL).\n",
                    my_rank, k, owner);
            }
            s_per_cell_slot[k] = slot;
            s_send_counts[slot]++;
        }
    }

    // neighbor-only count exchange (Neighbor_alltoall when peers distinct,
    // Isend/Irecv per direction otherwise — never MPI_Alltoall over the full comm)
    static void exchange_counts() {
        const int nn = halo.n_neighbors;
        s_recv_counts.assign(nn, 0);
        if (halo.use_neighbor_coll) {
            MPI_Neighbor_alltoall(s_send_counts.data(), 1, MPI_INT,
                                  s_recv_counts.data(), 1, MPI_INT, halo.graph_comm);
            return;
        }
        MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
        int         n_reqs = 0;
        for (int n = 0; n < nn; n++) {
            const int dx   = halo.neighbor_dirs[n][0];
            const int dy   = halo.neighbor_dirs[n][1];
            const int dz   = halo.neighbor_dirs[n][2];
            const int peer = halo.neighbor_ranks[n];
            MPI_Isend(&s_send_counts[n], 1, MPI_INT, peer, migrate_count_tag(dx, dy, dz),
                      decomp.cart_comm, &reqs[n_reqs++]);
            MPI_Irecv(&s_recv_counts[n], 1, MPI_INT, peer, migrate_count_tag(-dx, -dy, -dz),
                      decomp.cart_comm, &reqs[n_reqs++]);
        }
        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    }

    static void build_displacements(int nn, int* total_send, int* total_recv) {
        s_send_displs.assign(nn, 0);
        s_recv_displs.assign(nn, 0);
        int ts = 0, tr = 0;
        for (int n = 0; n < nn; n++) {
            s_send_displs[n] = ts;
            s_recv_displs[n] = tr;
            ts += s_send_counts[n];
            tr += s_recv_counts[n];
        }
        s_sendbuf.resize(ts);
        s_recvbuf.resize(tr);
        *total_send = ts;
        *total_recv = tr;
    }

    static void pack_outgoing_migrants(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new,
                                       POINT_TYPE* pts, int n_hydro) {
        s_migrant_local_k.clear();
        s_migrant_local_k.reserve(s_sendbuf.size());

        std::vector<int> cursor = s_send_displs;
        for (int k = 0; k < n_hydro; k++) {
            const int slot_id = s_per_cell_slot[k];
            if (slot_id < 0) continue;
            const int    slot = cursor[slot_id]++;
            MigrantCell& mc   = s_sendbuf[slot];
            mc.pos     = pts[k];
            mc.rho_old = primvar->rho[k];
            mc.v_old   = primvar->v[k];
            mc.E_old   = primvar->E[k];
            mc.rho_new = prim_new->rho[k];
            mc.v_new   = prim_new->v[k];
            mc.E_new   = prim_new->E[k];
#ifdef MOVING_MESH
            mc.v_mesh     = mesh->v_mesh[k];
            mc.old_volume = mesh->old_volumes[k];
#else
            (void)mesh;
#endif
            s_migrant_local_k.push_back(k);
        }
    }

    // single neighbor-only collective for all MigrantCell fields
    static void exchange_payload() {
        const int nn = halo.n_neighbors;
        if (halo.use_neighbor_coll) {
            MPI_Neighbor_alltoallv(s_sendbuf.data(), s_send_counts.data(), s_send_displs.data(), s_mpi_migrant_t,
                                   s_recvbuf.data(), s_recv_counts.data(), s_recv_displs.data(), s_mpi_migrant_t,
                                   halo.graph_comm);
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
                MPI_Isend(s_sendbuf.data() + s_send_displs[n], sc, s_mpi_migrant_t, peer,
                          migrate_payload_tag(dx, dy, dz), decomp.cart_comm, &reqs[n_reqs++]);
            }
            if (rc > 0) {
                MPI_Irecv(s_recvbuf.data() + s_recv_displs[n], rc, s_mpi_migrant_t, peer,
                          migrate_payload_tag(-dx, -dy, -dz), decomp.cart_comm, &reqs[n_reqs++]);
            }
        }
        if (n_reqs > 0) MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    }

    // remove migrated cells via swap-with-last (largest k first)
    static int remove_migrated_local(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new,
                                     POINT_TYPE* pts, int n_hydro) {
        std::sort(s_migrant_local_k.begin(), s_migrant_local_k.end(), std::greater<int>());
        int n_after = n_hydro;
        for (int k_remove : s_migrant_local_k) {
            const int k_last = n_after - 1;
            if (k_remove != k_last) {
                pts[k_remove]                    = pts[k_last];
                primvar->rho[k_remove]           = primvar->rho[k_last];
                primvar->v[k_remove]             = primvar->v[k_last];
                primvar->E[k_remove]             = primvar->E[k_last];
                prim_new->rho[k_remove]          = prim_new->rho[k_last];
                prim_new->v[k_remove]            = prim_new->v[k_last];
                prim_new->E[k_remove]            = prim_new->E[k_last];
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

    static void append_incoming_migrants(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new,
                                         POINT_TYPE* pts, int n_after_remove, int total_recv, int my_rank) {
        (void)my_rank;
        for (int j = 0; j < total_recv; j++) {
            const int          k  = n_after_remove + j;
            const MigrantCell& mc = s_recvbuf[j];
            pts[k] = mc.pos;
#ifdef dim_3D
            mesh->seeds[k] = double3{mc.pos.x, mc.pos.y, mc.pos.z};
#else
            mesh->seeds[k] = double3{mc.pos.x, mc.pos.y, 0.0};
#endif
            primvar->rho[k]  = mc.rho_old;
            primvar->v[k]    = mc.v_old;
            primvar->E[k]    = mc.E_old;
            prim_new->rho[k] = mc.rho_new;
            prim_new->v[k]   = mc.v_new;
            prim_new->E[k]   = mc.E_new;
#ifdef MOVING_MESH
            mesh->v_mesh[k]      = mc.v_mesh;
            mesh->old_volumes[k] = mc.old_volume;
#endif
            mesh->cell_to_original[k] = (unsigned int)k;
        }
    }

    // global cell count must stay constant
    static void check_conservation(int n_new) {
        int n_global = 0;
        Profiler::StartTimer("MPI_COMM");
        Profiler::StartTimer("MPI_REDUCE");
        MPI_Allreduce(&n_new, &n_global, 1, MPI_INT, MPI_SUM, decomp.cart_comm);
        Profiler::EndTimer("MPI_REDUCE");
        Profiler::EndTimer("MPI_COMM");
        static int s_n_total_expected = 0;
        if (s_n_total_expected == 0) s_n_total_expected = n_global;
        if (n_global != s_n_total_expected) {
            exit_failure("MIGRATE: FATAL global cell-count drift: %d != %d. A cell was duplicated or lost.\n",
                n_global, s_n_total_expected);
        }
    }

#endif // USE_MPI

}  // namespace proteus_mpi
