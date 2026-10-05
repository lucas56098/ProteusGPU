// implements the cell migration (migrate.h)

#include "migrate.h"

#include "decomp.h"
#include "exchange.h"
#include "global/structs.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace proteus_mpi {

    // everything a cell needs to carry to its new rank
    struct MigrantCell {
        POINT_TYPE pos;
        double     rho_old;
        POINT_TYPE v_old;
        double     E_old;
        double     mass;
        POINT_TYPE momentum;
        double     energy;
#ifdef MOVING_MESH
        POINT_TYPE              v_mesh;
        gradients::PrimGradient grad;
#endif
    };

} // namespace proteus_mpi

#include "migrate_packing.h"

namespace proteus_mpi {

#ifdef USE_MPI
    static int s_n_local_max = 0;
#endif
    static int s_last_n_migrated = 0;

#ifdef USE_MPI

    // scratch of the migration, grown when it gets too small
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
    static int          s_n_migrant_local     = 0;
    static int*         s_mig_scan            = nullptr;
    static int          s_mig_scan_cap        = 0;
    static int*         s_scan_scratch        = nullptr;
    static int          s_scan_scratch_cap    = 0;
    static int*         s_dest_pos            = nullptr;
    static int          s_dest_pos_cap        = 0;
    static int*         s_chunk_tab           = nullptr;
    static int          s_chunk_tab_cap       = 0;
#endif

    // scratch that grows and then stays
    template <typename T> static void ensure_managed(T*& ptr, int& cap, int need) {
        if (need <= cap) return;
        const int new_cap = std::max(64, std::max(need, 2 * cap));
        if (ptr) gpu_free(ptr);
        ptr = (T*)gpu_malloc(sizeof(T) * (size_t)new_cap);
        cap = new_cap;
    }

#ifdef USE_MPI
    static int  assign_destinations(VMesh* mesh, int n_hydro, std::vector<int>* dests);
    static void build_displacements(int nn, int* total_send, int* total_recv);
    static void pack_outgoing_migrants(VMesh*                    mesh,
                                       hydro::primvars*          primvar,
                                       hydro::ConsVars*          cons,
                                       gradients::PrimGradients* grads,
                                       POINT_TYPE*               pts,
                                       int                       n_hydro,
                                       int                       nslots);
    static int  exchange_payload(const std::vector<int>& dests, int total_send);
    static int  remove_migrated_local(VMesh*                    mesh,
                                      hydro::primvars*          primvar,
                                      hydro::ConsVars*          cons,
                                      gradients::PrimGradients* grads,
                                      POINT_TYPE*               pts,
                                      int                       n_hydro);
    static void append_incoming_migrants(VMesh*                    mesh,
                                         hydro::primvars*          primvar,
                                         hydro::ConsVars*          cons,
                                         gradients::PrimGradients* grads,
                                         POINT_TYPE*               pts,
                                         int                       n_after_remove,
                                         int                       total_recv,
                                         int                       my_rank);
    static void check_conservation(int n_new);

#endif

    int last_n_migrated() {
        return s_last_n_migrated;
    }

    // the cell arrays never grow, so this is the ceiling
    void migrate_init(int n_local_initial) {
#ifdef USE_MPI
        s_n_local_max = max_n_local(n_local_initial);
#else
        (void)n_local_initial;
#endif
    }

    // every cell whose new position belongs to another rank goes there; any rank, since the cuts can move
    void migrate_cells(VMesh* mesh, hydro::primvars* primvar, hydro::ConsVars* cons, gradients::PrimGradients* grads) {
#ifndef USE_MPI
        (void)mesh;
        (void)primvar;
        (void)cons;
        (void)grads;
        return;
#else
        if (decomp.nranks <= 1) return;

        PROFILE("MIGRATE");

        const int   my_rank = decomp.rank;
        const int   n_hydro = (int)mesh->n_hydro;
        POINT_TYPE* pts     = mesh->scratch_move;

        // target rank of every cell, as an index into the sorted list of targets
        std::vector<int> dests;
        const int        nslots = assign_destinations(mesh, n_hydro, &dests);

        int total_send = 0, total_recv = 0;
        for (int i = 0; i < nslots; i++)
            s_recv_counts[i] = 0;
        build_displacements(nslots, &total_send, &total_recv);
        s_last_n_migrated = total_send;

        pack_outgoing_migrants(mesh, primvar, cons, grads, pts, n_hydro, nslots);

        {
            PROFILE_MPI("PAYLOAD_WAIT");
            total_recv = exchange_payload(dests, total_send);
        }

        // the leavers go out of the arrays, the arrivals come behind the rest
        const int n_after_remove = remove_migrated_local(mesh, primvar, cons, grads, pts, n_hydro);

        const int n_new = n_after_remove + total_recv;
        if (n_new > s_n_local_max) {
            exit_failure("[rank %d] MIGRATE: n_hydro_new=%d > n_local_max=%d (migration overflows the per-cell "
                         "arrays). Raise alloc_growth in the param file or enable rebalance with a tighter "
                         "imbalance_threshold, then restart from the last snapshot.\n",
                         my_rank,
                         n_new,
                         s_n_local_max);
        }

        append_incoming_migrants(mesh, primvar, cons, grads, pts, n_after_remove, total_recv, my_rank);
        mesh->n_hydro = (uint64_t)n_new;

        check_conservation(n_new);
#endif
    }

#ifdef USE_MPI

    // owner of every cell at its new position; returns how many other ranks get cells, dests lists them
    static int assign_destinations(VMesh* mesh, int n_hydro, std::vector<int>* dests) {
        ensure_managed(s_per_cell_slot, s_per_cell_slot_cap, std::max(n_hydro, 1));
        const POINT_TYPE* pts    = mesh->scratch_move;
        const uint64_t*   cuts   = decomp.cuts;
        const int         nranks = decomp.nranks;
        const int         me     = decomp.rank;
        int*              owner  = s_per_cell_slot;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("ASSIGN", n_hydro, [=] HD(int k) {
            const int o = owner_of_point(pts[k], cuts, nranks);
            owner[k]    = (o == me) ? -1 : o;
        });

        // targets in rank order, then every cell gets the index of its target
        dests->clear();
        for (int k = 0; k < n_hydro; k++)
            if (owner[k] >= 0) dests->push_back(owner[k]);
        std::sort(dests->begin(), dests->end());
        dests->erase(std::unique(dests->begin(), dests->end()), dests->end());

        const int nslots = (int)dests->size();
        ensure_managed(s_send_counts, s_send_counts_cap, std::max(nslots, 1));
        ensure_managed(s_recv_counts, s_recv_counts_cap, std::max(nslots, 1));
        for (int i = 0; i < nslots; i++)
            s_send_counts[i] = 0;
        for (int k = 0; k < n_hydro; k++) {
            if (owner[k] < 0) continue;
            const int slot = (int)(std::lower_bound(dests->begin(), dests->end(), owner[k]) - dests->begin());
            owner[k]       = slot;
            s_send_counts[slot]++;
        }
        return nslots;
    }

    // one block per target in both buffers
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
        ensure_managed(s_sendbuf, s_sendbuf_cap, std::max(ts, 1));
        ensure_managed(s_recvbuf, s_recvbuf_cap, std::max(tr, 1));
        *total_send = ts;
        *total_recv = tr;
    }

    static constexpr int PACK_TABLE_BUDGET = 1 << 20;
    static constexpr int PACK_MAX_CHUNKS   = 1024;

    static int pack_chunks_for(int nslots) {
        int c = (nslots > 0) ? (PACK_TABLE_BUDGET / nslots) : PACK_MAX_CHUNKS;
        if (c > PACK_MAX_CHUNKS) c = PACK_MAX_CHUNKS;
        if (c < 1) c = 1;
        return c;
    }

    HD inline void pack_chunk_range(int c, int m, int chunks, int* lo, int* hi) {
        const int span = (m + chunks - 1) / chunks;
        int       a    = c * span;
        int       b    = a + span;
        if (a > m) a = m;
        if (b > m) b = m;
        *lo = a;
        *hi = b;
    }

    // place of every migrating cell in the send buffer, found without atomics so it is the same in every run
    static void build_pack_layout(int n_hydro, int nslots) {
        ensure_managed(s_mig_scan, s_mig_scan_cap, std::max(n_hydro, 1));
        ensure_managed(s_dest_pos, s_dest_pos_cap, std::max(n_hydro, 1));
        ensure_managed(s_migrant_local_k, s_migrant_local_k_cap, std::max(n_hydro, 1));

        const int* per_cell_slot = s_per_cell_slot;
        int*       off           = s_mig_scan;

        s_n_migrant_local = 0;
        if (n_hydro <= 0) return;

        // flag and scan gives every migrating cell a number
        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "MIG_FLAG", n_hydro, [=] HD(size_t k) { off[k] = (per_cell_slot[k] >= 0) ? 1 : 0; });

        const size_t need = scan_scratch_size((size_t)n_hydro, _MPI_PACK_BLOCK_SIZE_);
        ensure_managed(s_scan_scratch, s_scan_scratch_cap, (int)need);
        parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_, int>("MIG_SCAN", (size_t)n_hydro, off, off, s_scan_scratch);

        const int m       = off[n_hydro - 1] + ((per_cell_slot[n_hydro - 1] >= 0) ? 1 : 0);
        s_n_migrant_local = m;
        if (m == 0) return;

        int* mig_k = s_migrant_local_k;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("MIG_COMPACT", n_hydro, [=] HD(size_t k) {
            if (per_cell_slot[k] >= 0) mig_k[off[k]] = (int)k;
        });

        // count per chunk and target, scan the small table, then every chunk fills from its own base
        const int chunks = pack_chunks_for(nslots);
        ensure_managed(s_chunk_tab, s_chunk_tab_cap, chunks * nslots);
        int*       tab         = s_chunk_tab;
        const int* send_displs = s_send_displs;
        int*       dest_pos    = s_dest_pos;

        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "MIG_TAB_ZERO", (size_t)chunks * (size_t)nslots, [=] HD(size_t i) { tab[i] = 0; });

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

    // copy the cells into the send buffer
    static void pack_outgoing_migrants(VMesh*                    mesh,
                                       hydro::primvars*          primvar,
                                       hydro::ConsVars*          cons,
                                       gradients::PrimGradients* grads,
                                       POINT_TYPE*               pts,
                                       int                       n_hydro,
                                       int                       nslots) {
        build_pack_layout(n_hydro, nslots);

        auto*   per_cell_slot = s_per_cell_slot;
        auto*   dest_pos      = s_dest_pos;
        auto*   sendbuf       = s_sendbuf;
        double* rho           = primvar->rho;
        auto*   v             = primvar->v;
        double* E             = primvar->E;
        double* mass          = cons->mass;
        auto*   momentum      = cons->momentum;
        double* energy        = cons->energy;
#ifdef MOVING_MESH
        auto* v_mesh = mesh->v_mesh;
#endif

        parallel_for<_MPI_PACK_BLOCK_SIZE_>("PACK", n_hydro, [=] HD(int k) {
            pack::pack_migrant_body(k,
                                    per_cell_slot,
                                    dest_pos,
                                    pts,
                                    rho,
                                    v,
                                    E,
                                    mass,
                                    momentum,
                                    energy,
#ifdef MOVING_MESH
                                    v_mesh,
                                    grads,
#endif
                                    sendbuf);
        });
#ifndef MOVING_MESH
        (void)mesh;
        (void)grads;
#endif
    }

    // the cells themselves, to ranks that do not know they get some; what comes in is in source rank order
    static int exchange_payload(const std::vector<int>& dests, int total_send) {
        mpi_sync_before_send(s_sendbuf, sizeof(MigrantCell) * (size_t)total_send);
        Messages out, in;
        for (size_t i = 0; i < dests.size(); i++) {
            const char* first = (const char*)(s_sendbuf + s_send_displs[i]);
            out.to(dests[i]).assign(first, first + sizeof(MigrantCell) * (size_t)s_send_counts[i]);
        }
        sparse_exchange(out, &in);

        int total_recv = 0;
        for (size_t m = 0; m < in.ranks.size(); m++)
            total_recv += (int)count_of<MigrantCell>(in.data[m]);
        ensure_managed(s_recvbuf, s_recvbuf_cap, std::max(total_recv, 1));
        size_t at = 0;
        for (size_t m = 0; m < in.ranks.size(); m++) {
            std::memcpy((char*)(s_recvbuf + at), in.data[m].data(), in.data[m].size());
            at += count_of<MigrantCell>(in.data[m]);
        }
        mpi_sync_after_recv(s_recvbuf, sizeof(MigrantCell) * (size_t)total_recv);
        return total_recv;
    }

    // close the holes the leaving cells left: the last cell moves into the hole
    static int remove_migrated_local(VMesh*                    mesh,
                                     hydro::primvars*          primvar,
                                     hydro::ConsVars*          cons,
                                     gradients::PrimGradients* grads,
                                     POINT_TYPE*               pts,
                                     int                       n_hydro) {
        std::sort(s_migrant_local_k, s_migrant_local_k + s_n_migrant_local, std::greater<int>());
        int n_after = n_hydro;
        for (int i = 0; i < s_n_migrant_local; i++) {
            const int k_remove = s_migrant_local_k[i];
            const int k_last   = n_after - 1;
            if (k_remove != k_last) {
                pts[k_remove]            = pts[k_last];
                primvar->rho[k_remove]   = primvar->rho[k_last];
                primvar->v[k_remove]     = primvar->v[k_last];
                primvar->E[k_remove]     = primvar->E[k_last];
                cons->mass[k_remove]     = cons->mass[k_last];
                cons->momentum[k_remove] = cons->momentum[k_last];
                cons->energy[k_remove]   = cons->energy[k_last];
#ifdef MOVING_MESH
                mesh->v_mesh[k_remove] = mesh->v_mesh[k_last];
                grads->rho[k_remove]   = grads->rho[k_last];
                grads->vx[k_remove]    = grads->vx[k_last];
                grads->vy[k_remove]    = grads->vy[k_last];
#ifdef dim_3D
                grads->vz[k_remove] = grads->vz[k_last];
#endif
                grads->P[k_remove]      = grads->P[k_last];
                grads->anchor[k_remove] = grads->anchor[k_last];
#endif
            }
            n_after--;
        }
#ifndef MOVING_MESH
        (void)mesh;
        (void)grads;
#endif
        return n_after;
    }

    // the arrivals go behind the cells that stayed
    static void append_incoming_migrants(VMesh*                    mesh,
                                         hydro::primvars*          primvar,
                                         hydro::ConsVars*          cons,
                                         gradients::PrimGradients* grads,
                                         POINT_TYPE*               pts,
                                         int                       n_after_remove,
                                         int                       total_recv,
                                         int                       my_rank) {
        (void)my_rank;
#ifndef MOVING_MESH
        (void)grads;
#endif
        if (total_recv <= 0) return;

        auto*   recvbuf  = s_recvbuf;
        auto*   seeds    = mesh->seeds;
        double* rho      = primvar->rho;
        auto*   v        = primvar->v;
        double* E        = primvar->E;
        double* mass     = cons->mass;
        auto*   momentum = cons->momentum;
        double* energy   = cons->energy;
#ifdef MOVING_MESH
        auto* v_mesh = mesh->v_mesh;
#endif

        parallel_for<_MPI_PACK_BLOCK_SIZE_>("APPEND", total_recv, [=] HD(int j) {
            pack::unpack_migrant_body(j,
                                      n_after_remove,
                                      recvbuf,
#ifdef MOVING_MESH
                                      v_mesh,
                                      grads,
#endif
                                      pts,
                                      seeds,
                                      rho,
                                      v,
                                      E,
                                      mass,
                                      momentum,
                                      energy);
        });
    }

    // the total cell count of the run may never change
    static void check_conservation(int n_new) {
        const long long n_new_ll = (long long)n_new;
        long long       n_global = 0;
        {
            PROFILE_MPI("CONS_ALLREDUCE");
            MPI_Allreduce(&n_new_ll, &n_global, 1, MPI_LONG_LONG, MPI_SUM, decomp.comm);
        }
        static long long s_n_total_expected = 0;
        if (s_n_total_expected == 0) s_n_total_expected = n_global;
        if (n_global != s_n_total_expected) {
            exit_failure("MIGRATE: FATAL global cell-count drift: %lld != %lld. A cell was duplicated or lost.\n",
                         n_global,
                         s_n_total_expected);
        }
    }

#endif

} // namespace proteus_mpi
