// implements the cell migration (migrate.h)

#include "migrate.h"

#include "decomp.h"
#include "exchange.h"
#include "global/structs.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

#include <vector>

namespace proteus_mpi {

    static int s_last_n_migrated = 0;

    int last_n_migrated() {
        return s_last_n_migrated;
    }

#ifdef USE_MPI

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

    // the per-cell arrays a migrant is taken from and put into
    struct CellArrays {
        POINT_TYPE*      pts;
        hydro::primvars* prim;
        hydro::ConsVars* cons;
#ifdef MOVING_MESH
        POINT_TYPE*               v_mesh;
        gradients::PrimGradients* grads;
#endif

        HD MigrantCell load(int k) const {
            MigrantCell mc;
            mc.pos      = pts[k];
            mc.rho_old  = prim->rho[k];
            mc.v_old    = prim->v[k];
            mc.E_old    = prim->E[k];
            mc.mass     = cons->mass[k];
            mc.momentum = cons->momentum[k];
            mc.energy   = cons->energy[k];
#ifdef MOVING_MESH
            mc.v_mesh = v_mesh[k];
            mc.grad   = grads->load(k);
#endif
            return mc;
        }

        HD void store(int k, const MigrantCell& mc) const {
            pts[k]            = mc.pos;
            prim->rho[k]      = mc.rho_old;
            prim->v[k]        = mc.v_old;
            prim->E[k]        = mc.E_old;
            cons->mass[k]     = mc.mass;
            cons->momentum[k] = mc.momentum;
            cons->energy[k]   = mc.energy;
#ifdef MOVING_MESH
            v_mesh[k]     = mc.v_mesh;
            grads->rho[k] = mc.grad.rho;
            grads->vx[k]  = mc.grad.vx;
            grads->vy[k]  = mc.grad.vy;
#ifdef dim_3D
            grads->vz[k] = mc.grad.vz;
#endif
            grads->P[k]      = mc.grad.P;
            grads->anchor[k] = mc.grad.anchor;
#endif
        }
    };

    // the arrays of the migration, kept between steps
    struct MigrateBuffers {
        GpuArray<int>          owner; // target rank of every cell, -1 if it stays
        GpuArray<int>          flag, pos, scan_scratch;
        GpuArray<int>          leaving; // the leaving cells in cell order
        GpuArray<int>          filler;  // the staying cells behind the new end
        PairSortArrays         sort;
        GpuArray<unsigned int> run_scratch;
        GpuArray<MigrantCell>  sendbuf, recvbuf;

        void free() {
            owner.free();
            flag.free();
            pos.free();
            scan_scratch.free();
            leaving.free();
            filler.free();
            sort.free();
            run_scratch.free();
            sendbuf.free();
            recvbuf.free();
        }
    };

    static MigrateBuffers s_buffers;

    // exclusive scan of the n flags into pos; returns how many are set
    static int scan_flags(int n) {
        if (n <= 0) return 0;
        int*      flag = s_buffers.flag.data;
        int*      pos  = s_buffers.pos.data;
        const int last = flag[n - 1];
        parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_, int>(
            "MIG_SCAN", (size_t)n, flag, pos, s_buffers.scan_scratch.fit(scan_scratch_size(n, _MPI_PACK_BLOCK_SIZE_)));
        return pos[n - 1] + last;
    }

    static int  find_leaving_cells(const VMesh* mesh, int n_hydro);
    static void pack_outgoing(const CellArrays& cells, int m, Blocks* out);
    static void fill_holes(const CellArrays& cells, int n_hydro, int m);
    static void append_incoming(const CellArrays& cells, double3* seeds, int n_after_remove, int total_recv);
    static void check_conservation(int n_new);

#endif

    void migrate_free() {
#ifdef USE_MPI
        s_buffers.free();
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

        const int  n_hydro = (int)mesh->n_hydro;
        CellArrays cells;
        cells.pts  = mesh->scratch_move;
        cells.prim = primvar;
        cells.cons = cons;
#ifdef MOVING_MESH
        cells.v_mesh = mesh->v_mesh;
        cells.grads  = grads;
#else
        (void)grads;
#endif

        const int m       = find_leaving_cells(mesh, n_hydro);
        s_last_n_migrated = m;

        Blocks out, in;
        pack_outgoing(cells, m, &out);
        {
            PROFILE_MPI("PAYLOAD_WAIT");
            sparse_counts(out, &in);
            exchange_items(s_buffers.sendbuf.data, out, s_buffers.recvbuf.fit(in.total), in, sizeof(MigrantCell));
        }

        // the leavers out of the arrays, the arrivals behind the rest; the arrays never grow
        const int n_after_remove = n_hydro - m;
        const int n_new          = n_after_remove + (int)in.total;
        const int n_max          = max_n_local(n_hydro);
        if (n_new > n_max) {
            exit_failure("[rank %d] MIGRATE: n_hydro_new=%d > n_local_max=%d (migration overflows the per-cell "
                         "arrays). Raise alloc_growth in the param file or enable rebalance with a tighter "
                         "imbalance_threshold, then restart from the last snapshot.\n",
                         decomp.rank,
                         n_new,
                         n_max);
        }
        fill_holes(cells, n_hydro, m);
        append_incoming(cells, mesh->seeds, n_after_remove, (int)in.total);
        mesh->n_hydro = (uint64_t)n_new;

        check_conservation(n_new);
#endif
    }

#ifdef USE_MPI

    // owner of every cell at its new position, and the cells that leave in cell order; returns how many leave
    static int find_leaving_cells(const VMesh* mesh, int n_hydro) {
        const POINT_TYPE* pts    = mesh->scratch_move;
        const uint64_t*   cuts   = decomp.cuts;
        const int         nranks = decomp.nranks;
        const int         me     = decomp.rank;
        int*              owner  = s_buffers.owner.fit(n_hydro);
        int*              flag   = s_buffers.flag.fit(n_hydro);
        s_buffers.pos.fit(n_hydro);
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("ASSIGN", n_hydro, [=] HD(int k) {
            const int o = owner_of_point(pts[k], cuts, nranks);
            owner[k]    = (o == me) ? -1 : o;
            flag[k]     = (o == me) ? 0 : 1;
        });
        const int m = scan_flags(n_hydro);

        const int* pos     = s_buffers.pos.data;
        int*       leaving = s_buffers.leaving.fit(m);
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("MIG_COMPACT", n_hydro, [=] HD(int k) {
            if (flag[k]) leaving[pos[k]] = k;
        });
        return m;
    }

    // the leaving cells grouped by target rank, cell order kept within a rank, into the send buffer
    static void pack_outgoing(const CellArrays& cells, int m, Blocks* out) {
        out->clear();
        if (m == 0) return;

        const int* owner   = s_buffers.owner.data;
        const int* leaving = s_buffers.leaving.data;
        PairSort   sort    = s_buffers.sort.fit(m);
        {
            uint64_t*     keys = sort.keys;
            unsigned int* vals = sort.vals;
            parallel_for<_MPI_PACK_BLOCK_SIZE_>("MIG_DEST", m, [=] HD(int j) {
                keys[j] = (uint64_t)owner[leaving[j]];
                vals[j] = (unsigned int)j;
            });
        }
        sort.sort("MIG_SORT", (size_t)m, rank_key_bits());
        *out = blocks_of_sorted_ranks(sort.keys, (size_t)m, &s_buffers.run_scratch);

        const unsigned int* order   = sort.vals;
        MigrantCell*        sendbuf = s_buffers.sendbuf.fit(m);
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("PACK", m, [=] HD(int j) { sendbuf[j] = cells.load(leaving[order[j]]); });
    }

    // the i-th leaving cell below the new end takes the i-th staying cell above it; there are as many of one
    // as of the other
    static void fill_holes(const CellArrays& cells, int n_hydro, int m) {
        if (m == 0) return;
        const int  n_after = n_hydro - m;
        const int* owner   = s_buffers.owner.data;
        const int* leaving = s_buffers.leaving.data;

        // holes: the leaving cells are in cell order, so the ones below the new end come first
        const int holes = parallel_reduce_sum<_MPI_PACK_BLOCK_SIZE_, int>(
            "MIG_HOLES", (size_t)m, [=] HD(size_t j) { return (leaving[j] < n_after) ? 1 : 0; });
        if (holes == 0) return;

        // fillers: the staying cells in [n_after, n_hydro), in cell order
        int* flag = s_buffers.flag.data;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "MIG_STAY", m, [=] HD(int t) { flag[t] = (owner[n_after + t] < 0) ? 1 : 0; });
        const int fillers = scan_flags(m);
        if (fillers != holes) {
            exit_failure("[rank %d] MIGRATE: %d holes but %d cells to fill them\n", decomp.rank, holes, fillers);
        }
        const int* pos    = s_buffers.pos.data;
        int*       filler = s_buffers.filler.fit(m);
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("MIG_FILLERS", m, [=] HD(int t) {
            if (flag[t]) filler[pos[t]] = n_after + t;
        });

        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "MIG_MOVE", holes, [=] HD(int i) { cells.store(leaving[i], cells.load(filler[i])); });
    }

    // the arrivals go behind the cells that stayed
    static void append_incoming(const CellArrays& cells, double3* seeds, int n_after_remove, int total_recv) {
        const MigrantCell* recvbuf = s_buffers.recvbuf.data;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("APPEND", total_recv, [=] HD(int j) {
            const int         k  = n_after_remove + j;
            const MigrantCell mc = recvbuf[j];
            cells.store(k, mc);
#ifdef dim_3D
            seeds[k] = double3{mc.pos.x, mc.pos.y, mc.pos.z};
#else
            seeds[k] = double3{mc.pos.x, mc.pos.y, 0.0};
#endif
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
