// Per-step halo data exchanges and dt allreduce.
// Included into halo.cu inside namespace proteus_mpi.

// ============================================================
// CUDA kernels (CUDA mode only)
// ============================================================

#if !defined(CPU_DEBUG) && defined(USE_MPI)
GLOBAL static void kernel_pack_seed(int                  total_send,
                                    const POINT_TYPE*    pts,
                                    const int*           export_indices,
                                    const unsigned char* dir_of_slot,
                                    const double*        neighbor_shift_flat,
                                    POINT_TYPE*          sendbuf) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= total_send) return;
    pack::pack_seed_body(s, pts, export_indices, dir_of_slot, neighbor_shift_flat, sendbuf);
}

GLOBAL static void
kernel_unpack_seed(int n_mpi, int pts_mpi_base, const POINT_TYPE* recvbuf, POINT_TYPE* pts, double3* seeds_g) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_mpi) return;
    pack::unpack_seed_body(slot, pts_mpi_base, recvbuf, pts, seeds_g);
}

GLOBAL static void kernel_fill_is_outer_layer(
    int nn, const int* recv_n_outer, const int* ghost_offset, const int* recv_count, unsigned char* is_outer_layer) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nn) return;
    pack::fill_is_outer_layer_body(n, recv_n_outer, ghost_offset, recv_count, is_outer_layer);
}

GLOBAL static void kernel_pack_prim(int                    total_send,
                                    const int*             used_export_indices,
                                    const hydro::primvars* primvar,
                                    HaloPrimCell*          sendbuf) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= total_send) return;
    pack::pack_prim_body(s, used_export_indices, primvar, sendbuf);
}

GLOBAL static void
kernel_unpack_prim(int n_recv, const int* used_to_full_slot, const HaloPrimCell* recvbuf, hydro::primvars* primvar) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_recv) return;
    pack::unpack_prim_body(s, used_to_full_slot, recvbuf, primvar);
}

GLOBAL static void kernel_pack_grad(int                             total_send,
                                    const int*                      used_export_indices,
                                    const gradients::PrimGradients* grads,
                                    POINT_TYPE*                     sendbuf) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_send) return;
    pack::pack_grad_body(slot, used_export_indices, grads, sendbuf);
}

GLOBAL static void kernel_unpack_grad(int                       n_recv,
                                      const int*                used_to_full_slot,
                                      const POINT_TYPE*         recvbuf,
                                      gradients::PrimGradients* grads) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_recv) return;
    pack::unpack_grad_body(slot, used_to_full_slot, recvbuf, grads);
}

#ifdef MOVING_MESH
GLOBAL static void
kernel_pack_v_mesh(int total_send, const int* used_export_indices, const POINT_TYPE* v_mesh, POINT_TYPE* sendbuf) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= total_send) return;
    pack::pack_v_mesh_body(s, used_export_indices, v_mesh, sendbuf);
}

GLOBAL static void
kernel_unpack_v_mesh(int n_recv, const int* used_to_full_slot, const POINT_TYPE* recvbuf, POINT_TYPE* v_mesh_g) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_recv) return;
    pack::unpack_v_mesh_body(slot, used_to_full_slot, recvbuf, v_mesh_g);
}
#endif // MOVING_MESH

#ifdef VOL_REGULARIZE
GLOBAL static void
kernel_pack_vol(int total_send, const int* used_export_indices, const double* volumes, double* sendbuf) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= total_send) return;
    pack::pack_vol_body(s, used_export_indices, volumes, sendbuf);
}

GLOBAL static void
kernel_unpack_vol(int n_recv, const int* used_to_full_slot, const double* recvbuf, double* volumes_g) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_recv) return;
    pack::unpack_vol_body(slot, used_to_full_slot, recvbuf, volumes_g);
}
#endif // VOL_REGULARIZE

// Small managed staging buffer holding per-direction counts the is_outer_layer
// kernel needs (recv_n_outer, ghost_offset (n+1 entries), recv_count). Lazily
// allocated on first use; freed in halo_free.
//   layout: [recv_n_outer | ghost_offset (n+1) | recv_count]
static int* s_is_outer_meta_dev = nullptr;

#endif // !CPU_DEBUG && USE_MPI

// ============================================================
// Public entry points
// ============================================================

void halo_exchange_seeds(VMesh* mesh, POINT_TYPE* pts, int pts_mpi_base) {
#ifndef USE_MPI
    (void)mesh;
    (void)pts;
    (void)pts_mpi_base;
    return;
#else
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;

    PROFILE("HALO_SEED");
    const int total_send = halo.send_offset[halo.n_neighbors];
    const int n_mpi      = halo.n_mpi_ghosts;
    const int nn         = halo.n_neighbors;

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (total_send + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("PACK");
            kernel_pack_seed<<<blocks, tpb>>>(
                total_send, pts, halo.export_indices, halo.dir_of_slot, halo.neighbor_shift_flat, halo.sendbuf_seed);
        }
        GPU_SYNC();
#else
        PROFILE("PACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int s = 0; s < total_send; s++) {
            // CPU_DEBUG: neighbor_shift inline array is flat-equivalent (3 doubles per direction,
            // row-major); cast to flat double* matches what the kernel sees.
            pack::pack_seed_body(s,
                                 pts,
                                 halo.export_indices,
                                 halo.dir_of_slot,
                                 (const double*)&halo.neighbor_shift[0][0],
                                 halo.sendbuf_seed);
        }
#endif
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_seed, sizeof(POINT_TYPE) * (size_t)total_send);
        exchange_full_halo(halo.sendbuf_seed, halo.recvbuf_seed, halo.mpi_point_t, MSG_SEED);
        mpi_sync_after_recv(halo.recvbuf_seed, sizeof(POINT_TYPE) * (size_t)n_mpi);
    }

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (n_mpi + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("UNPACK");
            kernel_unpack_seed<<<blocks, tpb>>>(n_mpi, pts_mpi_base, halo.recvbuf_seed, pts, mesh->seeds_g);
        }
        GPU_SYNC();

        // is_outer_layer: small managed staging buffer with the 3 host inline arrays,
        // then one thread per direction. Stays inside the unpack scope so PROFILE_KERNEL
        // covers the seed unpack itself; this trailing fill is fast (n_neighbors threads).
        if (s_is_outer_meta_dev == nullptr) {
            s_is_outer_meta_dev = (int*)gpu_malloc(sizeof(int) * (3 * HALO_MAX_NEIGHBORS + 1));
        }
        int* recv_n_outer_dev = s_is_outer_meta_dev;
        int* ghost_offset_dev = s_is_outer_meta_dev + HALO_MAX_NEIGHBORS;
        int* recv_count_dev   = s_is_outer_meta_dev + 2 * HALO_MAX_NEIGHBORS + 1; // ghost_offset has n+1
        for (int n = 0; n < nn; n++) {
            recv_n_outer_dev[n] = halo.recv_n_outer[n];
            ghost_offset_dev[n] = halo.ghost_offset[n];
            recv_count_dev[n]   = halo.recv_count[n];
        }
        ghost_offset_dev[nn] = halo.ghost_offset[nn];

        {
            const int tpb_n    = (nn < _MPI_PACK_BLOCK_SIZE_) ? std::max(nn, 1) : _MPI_PACK_BLOCK_SIZE_;
            const int blocks_n = (nn + tpb_n - 1) / tpb_n;
            kernel_fill_is_outer_layer<<<blocks_n, tpb_n>>>(
                nn, recv_n_outer_dev, ghost_offset_dev, recv_count_dev, halo.is_outer_layer);
        }
        GPU_SYNC();
#else
        PROFILE("UNPACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int slot = 0; slot < n_mpi; slot++) {
            pack::unpack_seed_body(slot, pts_mpi_base, halo.recvbuf_seed, pts, mesh->seeds_g);
        }
        for (int n = 0; n < nn; n++) {
            pack::fill_is_outer_layer_body(
                n, halo.recv_n_outer, halo.ghost_offset, halo.recv_count, halo.is_outer_layer);
        }
#endif
    }
#endif
}

void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar) {
#ifndef USE_MPI
    (void)mesh;
    (void)primvar;
    return;
#else
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;
    if (!halo.used_subset_ready) return; // nothing to do until mesh exists
    (void)mesh;

    PROFILE("HALO_PRIM");
    const int total_send = halo.n_used_send;
    const int n_recv     = halo.n_used_recv;

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (total_send + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("PACK");
            kernel_pack_prim<<<blocks, tpb>>>(total_send, halo.used_export_indices, primvar, halo.sendbuf_prim);
        }
        GPU_SYNC();
#else
        PROFILE("PACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int s = 0; s < total_send; s++) {
            pack::pack_prim_body(s, halo.used_export_indices, primvar, halo.sendbuf_prim);
        }
#endif
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_prim, sizeof(HaloPrimCell) * (size_t)total_send);
        exchange_used_subset(halo.sendbuf_prim, halo.recvbuf_prim, halo.mpi_prim_t, MSG_PRIM);
        mpi_sync_after_recv(halo.recvbuf_prim, sizeof(HaloPrimCell) * (size_t)n_recv);
    }

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (n_recv + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("UNPACK");
            kernel_unpack_prim<<<blocks, tpb>>>(n_recv, halo.used_to_full_slot, halo.recvbuf_prim, primvar);
        }
        GPU_SYNC();
#else
        PROFILE("UNPACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int s = 0; s < n_recv; s++) {
            pack::unpack_prim_body(s, halo.used_to_full_slot, halo.recvbuf_prim, primvar);
        }
#endif
    }
#endif
}

void halo_exchange_gradients(VMesh* mesh, gradients::PrimGradients* grads) {
#ifndef USE_MPI
    (void)mesh;
    (void)grads;
    return;
#else
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;
    if (!halo.used_subset_ready) return;
    (void)mesh;

    PROFILE("HALO_GRAD");
    const int N_COMP     = 3 + DIMENSION;
    const int total_send = halo.n_used_send;
    const int n_recv     = halo.n_used_recv;

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (total_send + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("PACK");
            kernel_pack_grad<<<blocks, tpb>>>(total_send, halo.used_export_indices, grads, halo.sendbuf_grad);
        }
        GPU_SYNC();
#else
        PROFILE("PACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int slot = 0; slot < total_send; slot++) {
            pack::pack_grad_body(slot, halo.used_export_indices, grads, halo.sendbuf_grad);
        }
#endif
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_grad, sizeof(POINT_TYPE) * (size_t)total_send * N_COMP);
        exchange_used_subset(halo.sendbuf_grad, halo.recvbuf_grad, halo.mpi_grad_cell_t, MSG_GRAD);
        mpi_sync_after_recv(halo.recvbuf_grad, sizeof(POINT_TYPE) * (size_t)n_recv * N_COMP);
    }

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (n_recv + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("UNPACK");
            kernel_unpack_grad<<<blocks, tpb>>>(n_recv, halo.used_to_full_slot, halo.recvbuf_grad, grads);
        }
        GPU_SYNC();
#else
        PROFILE("UNPACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int slot = 0; slot < n_recv; slot++) {
            pack::unpack_grad_body(slot, halo.used_to_full_slot, halo.recvbuf_grad, grads);
        }
#endif
    }
#endif
}

void halo_exchange_v_mesh(VMesh* mesh) {
#ifndef USE_MPI
    (void)mesh;
    return;
#else
#ifdef MOVING_MESH
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;
    if (!halo.used_subset_ready) return;

    PROFILE("HALO_VMESH");
    const int total_send = halo.n_used_send;
    const int n_recv     = halo.n_used_recv;

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (total_send + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("PACK");
            kernel_pack_v_mesh<<<blocks, tpb>>>(
                total_send, halo.used_export_indices, mesh->v_mesh, halo.sendbuf_v_mesh);
        }
        GPU_SYNC();
#else
        PROFILE("PACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int s = 0; s < total_send; s++) {
            pack::pack_v_mesh_body(s, halo.used_export_indices, mesh->v_mesh, halo.sendbuf_v_mesh);
        }
#endif
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_v_mesh, sizeof(POINT_TYPE) * (size_t)total_send);
        exchange_used_subset(halo.sendbuf_v_mesh, halo.recvbuf_v_mesh, halo.mpi_point_t, MSG_V_MESH);
        mpi_sync_after_recv(halo.recvbuf_v_mesh, sizeof(POINT_TYPE) * (size_t)n_recv);
    }

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (n_recv + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("UNPACK");
            kernel_unpack_v_mesh<<<blocks, tpb>>>(n_recv, halo.used_to_full_slot, halo.recvbuf_v_mesh, mesh->v_mesh_g);
        }
        GPU_SYNC();
#else
        PROFILE("UNPACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int slot = 0; slot < n_recv; slot++) {
            pack::unpack_v_mesh_body(slot, halo.used_to_full_slot, halo.recvbuf_v_mesh, mesh->v_mesh_g);
        }
#endif
    }
#else
    (void)mesh;
#endif
#endif
}

void halo_dt_allreduce(double* dt) {
#ifdef USE_MPI
    // 1 double — launch-overhead-bound, no GPU-aware benefit. Stays host even when
    // GPU_AWARE_MPI is on; no sync_before/after_recv calls.
    PROFILE_MPI("DT_ALLREDUCE");
    double local = *dt;
    MPI_Allreduce(&local, dt, 1, MPI_DOUBLE, MPI_MIN, decomp.cart_comm);
#else
    (void)dt;
#endif
}

// Scan the frozen export layout for slots shipping one of `moved_ks`. Positions are read
// from mesh->seeds (the fallback rewrites them at emit, so they hold the post-perturbation
// values) and pre-shifted per direction exactly like pack_seed_body. Cost is one pass over
// the export list with a hash lookup per slot — a few ms at worst, paid only on the rare
// steps where a perturbation happened at all.
int halo_collect_moved_exports(const VMesh* mesh, const std::vector<int>& moved_ks, MovedExportLists* lists) {
    for (int n = 0; n < HALO_MAX_NEIGHBORS; n++) {
        lists->js[n].clear();
        lists->pos[n].clear();
    }
    if (halo.n_neighbors == 0 || moved_ks.empty()) return 0;

#ifdef USE_MPI
    const std::unordered_set<int> moved(moved_ks.begin(), moved_ks.end());
    std::unordered_set<int>       exported;

    const int total_send = halo.send_offset[halo.n_neighbors];
    for (int s = 0; s < total_send; s++) {
        const int k = halo.export_indices[s];
        if (!moved.count(k)) continue;
        const int n = (int)halo.dir_of_slot[s];

        POINT_TYPE p;
        p.x = mesh->seeds[k].x + halo.neighbor_shift[n][0];
        p.y = mesh->seeds[k].y + halo.neighbor_shift[n][1];
#ifdef dim_3D
        p.z = mesh->seeds[k].z + halo.neighbor_shift[n][2];
#endif
        lists->js[n].push_back(s - halo.send_offset[n]);
        lists->pos[n].push_back(p);
        exported.insert(k);
    }
    return (int)exported.size();
#else
    (void)mesh;
    return 0;
#endif
}

// Counts handshake, then slot-offset + position payloads per neighbour, all requests in one
// Waitall. Buffers are plain host vectors: the payload is a handful of entries on the rare
// repair rounds, so there is nothing for GPU-aware MPI to win here and no sync_before/after
// wrappers are needed.
void halo_exchange_moved_seeds(const MovedExportLists& lists, std::vector<MovedSeed>* received) {
    received->clear();
    if (halo.n_neighbors == 0) return;

#ifdef USE_MPI
    const int nn = halo.n_neighbors;

    // phase 1: per-neighbour counts (same pattern as exchange_send_recv_counts)
    int sendcnt[HALO_MAX_NEIGHBORS] = {0};
    int recvcnt[HALO_MAX_NEIGHBORS] = {0};
    for (int n = 0; n < nn; n++)
        sendcnt[n] = (int)lists.js[n].size();

    if (halo.use_neighbor_coll) {
        MPI_Neighbor_alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, halo.graph_comm);
    } else {
        MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
        int         n_reqs = 0;
        for (int n = 0; n < nn; n++) {
            const int dx   = halo.neighbor_dirs[n][0];
            const int dy   = halo.neighbor_dirs[n][1];
            const int dz   = halo.neighbor_dirs[n][2];
            const int peer = halo.neighbor_ranks[n];
            MPI_Isend(&sendcnt[n], 1, MPI_INT, peer, msg_tag(dx, dy, dz, MSG_MOVED_COUNT), decomp.cart_comm,
                      &reqs[n_reqs++]);
            MPI_Irecv(&recvcnt[n], 1, MPI_INT, peer, msg_tag(-dx, -dy, -dz, MSG_MOVED_COUNT), decomp.cart_comm,
                      &reqs[n_reqs++]);
        }
        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    }

    // phase 2: payloads. Point-to-point unconditionally — the neighbour-collective path
    // would need flattened displacement arrays for a message of a few dozen bytes.
    std::vector<int>        recv_js[HALO_MAX_NEIGHBORS];
    std::vector<POINT_TYPE> recv_pos[HALO_MAX_NEIGHBORS];
    {
        PROFILE_MPI("WAIT");
        MPI_Request reqs[4 * HALO_MAX_NEIGHBORS];
        int         n_reqs = 0;
        for (int n = 0; n < nn; n++) {
            const int dx   = halo.neighbor_dirs[n][0];
            const int dy   = halo.neighbor_dirs[n][1];
            const int dz   = halo.neighbor_dirs[n][2];
            const int peer = halo.neighbor_ranks[n];
            if (sendcnt[n] > 0) {
                MPI_Isend(lists.js[n].data(), sendcnt[n], MPI_INT, peer, msg_tag(dx, dy, dz, MSG_MOVED_SLOT),
                          decomp.cart_comm, &reqs[n_reqs++]);
                MPI_Isend(lists.pos[n].data(), sendcnt[n], halo.mpi_point_t, peer,
                          msg_tag(dx, dy, dz, MSG_MOVED_POS), decomp.cart_comm, &reqs[n_reqs++]);
            }
            if (recvcnt[n] > 0) {
                recv_js[n].resize(recvcnt[n]);
                recv_pos[n].resize(recvcnt[n]);
                MPI_Irecv(recv_js[n].data(), recvcnt[n], MPI_INT, peer, msg_tag(-dx, -dy, -dz, MSG_MOVED_SLOT),
                          decomp.cart_comm, &reqs[n_reqs++]);
                MPI_Irecv(recv_pos[n].data(), recvcnt[n], halo.mpi_point_t, peer,
                          msg_tag(-dx, -dy, -dz, MSG_MOVED_POS), decomp.cart_comm, &reqs[n_reqs++]);
            }
        }
        if (n_reqs > 0) MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    }

    // unpack: slot offset j within neighbour n's receive range -> full ghost slot
    for (int n = 0; n < nn; n++) {
        for (int i = 0; i < recvcnt[n]; i++) {
            const int j = recv_js[n][i];
            if (j < 0 || j >= halo.recv_count[n]) {
                exit_failure("HALO: moved-seed slot offset %d out of range [0, %d) for neighbour %d\n", j,
                             halo.recv_count[n], n);
            }
            MovedSeed ms;
            ms.pos        = recv_pos[n][i];
            ms.ghost_slot = halo.ghost_offset[n] + j;
            received->push_back(ms);
        }
    }
#else
    (void)lists;
#endif
}

#ifdef VOL_REGULARIZE
void halo_exchange_volumes(VMesh* mesh) {
#ifndef USE_MPI
    (void)mesh;
    return;
#else
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;
    if (!halo.used_subset_ready) return;

    PROFILE("HALO_VOL");
    const int total_send = halo.n_used_send;
    const int n_recv     = halo.n_used_recv;

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (total_send + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("PACK");
            kernel_pack_vol<<<blocks, tpb>>>(total_send, halo.used_export_indices, mesh->volumes, halo.sendbuf_vol);
        }
        GPU_SYNC();
#else
        PROFILE("PACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int s = 0; s < total_send; s++) {
            pack::pack_vol_body(s, halo.used_export_indices, mesh->volumes, halo.sendbuf_vol);
        }
#endif
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_vol, sizeof(double) * (size_t)total_send);
        exchange_used_subset(halo.sendbuf_vol, halo.recvbuf_vol, MPI_DOUBLE, MSG_VOL);
        mpi_sync_after_recv(halo.recvbuf_vol, sizeof(double) * (size_t)n_recv);
    }

    {
#ifndef CPU_DEBUG
        const int tpb    = _MPI_PACK_BLOCK_SIZE_;
        const int blocks = (n_recv + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("UNPACK");
            kernel_unpack_vol<<<blocks, tpb>>>(n_recv, halo.used_to_full_slot, halo.recvbuf_vol, mesh->volumes_g);
        }
        GPU_SYNC();
#else
        PROFILE("UNPACK");
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int slot = 0; slot < n_recv; slot++) {
            pack::unpack_vol_body(slot, halo.used_to_full_slot, halo.recvbuf_vol, mesh->volumes_g);
        }
#endif
    }
#endif
}
#endif // VOL_REGULARIZE

// global SUM of one double across ranks (e.g. AGN cold-accretion mass). Same lightweight
// pattern as halo_dt_allreduce — no-op on single-rank builds.
void halo_sum_allreduce(double* v) {
#ifdef USE_MPI
    PROFILE_MPI("SUM_ALLREDUCE");
    double local = *v;
    MPI_Allreduce(&local, v, 1, MPI_DOUBLE, MPI_SUM, decomp.cart_comm);
#else
    (void)v;
#endif
}
