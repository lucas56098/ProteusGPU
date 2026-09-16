// the exchanges themselves (included by halo.cu)

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

#endif

#ifdef USE_MPI
static int* s_is_outer_meta_dev = nullptr;
#endif

// sends the export seeds and writes the ones that come back into the point list
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
        auto* recvbuf = halo.recvbuf_seed;
        auto* seeds_g = mesh->seeds_g;

        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "UNPACK", n_mpi, [=] HD(int slot) { pack::unpack_seed_body(slot, pts_mpi_base, recvbuf, pts, seeds_g); });

        // the kernel cannot read the per neighbour arrays of the struct
        if (s_is_outer_meta_dev == nullptr) {
            s_is_outer_meta_dev = (int*)gpu_malloc(sizeof(int) * (3 * HALO_MAX_NEIGHBORS + 1));
        }
        int* recv_n_outer = s_is_outer_meta_dev;
        int* ghost_offset = s_is_outer_meta_dev + HALO_MAX_NEIGHBORS;
        int* recv_count   = s_is_outer_meta_dev + 2 * HALO_MAX_NEIGHBORS + 1;
        for (int n = 0; n < nn; n++) {
            recv_n_outer[n] = halo.recv_n_outer[n];
            ghost_offset[n] = halo.ghost_offset[n];
            recv_count[n]   = halo.recv_count[n];
        }
        ghost_offset[nn] = halo.ghost_offset[nn];

        auto* is_outer_layer = halo.is_outer_layer;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("IS_OUTER", nn, [=] HD(int n) {
            pack::fill_is_outer_layer_body(n, recv_n_outer, ghost_offset, recv_count, is_outer_layer);
        });
    }
#endif
}

// state of the used ghosts
void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar) {
#ifndef USE_MPI
    (void)mesh;
    (void)primvar;
    return;
#else
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;
    // the mesh build tells us which ghosts matter; before that there is nothing to send
    if (!halo.used_subset_ready) return;
    (void)mesh;

    PROFILE("HALO_PRIM");
    const int total_send = halo.n_used_send;
    const int n_recv     = halo.n_used_recv;

    {
        auto* sendbuf_prim        = halo.sendbuf_prim;
        auto* used_export_indices = halo.used_export_indices;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "PACK", total_send, [=] HD(int s) { pack::pack_prim_body(s, used_export_indices, primvar, sendbuf_prim); });
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_prim, sizeof(HaloPrimCell) * (size_t)total_send);
        exchange_used_subset(halo.sendbuf_prim, halo.recvbuf_prim, halo.mpi_prim_t, MSG_PRIM);
        mpi_sync_after_recv(halo.recvbuf_prim, sizeof(HaloPrimCell) * (size_t)n_recv);
    }

    {
        auto* recvbuf_prim      = halo.recvbuf_prim;
        auto* used_to_full_slot = halo.used_to_full_slot;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>(
            "UNPACK", n_recv, [=] HD(int s) { pack::unpack_prim_body(s, used_to_full_slot, recvbuf_prim, primvar); });
    }
#endif
}

// their gradients, all components in one message
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
        auto* sendbuf_grad        = halo.sendbuf_grad;
        auto* used_export_indices = halo.used_export_indices;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("PACK", total_send, [=] HD(int slot) {
            pack::pack_grad_body(slot, used_export_indices, grads, sendbuf_grad);
        });
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_grad, sizeof(POINT_TYPE) * (size_t)total_send * N_COMP);
        exchange_used_subset(halo.sendbuf_grad, halo.recvbuf_grad, halo.mpi_grad_cell_t, MSG_GRAD);
        mpi_sync_after_recv(halo.recvbuf_grad, sizeof(POINT_TYPE) * (size_t)n_recv * N_COMP);
    }

    {
        auto* recvbuf_grad      = halo.recvbuf_grad;
        auto* used_to_full_slot = halo.used_to_full_slot;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("UNPACK", n_recv, [=] HD(int slot) {
            pack::unpack_grad_body(slot, used_to_full_slot, recvbuf_grad, grads);
        });
    }
#endif
}

// and their mesh velocity
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
        auto* sendbuf_v_mesh      = halo.sendbuf_v_mesh;
        auto* used_export_indices = halo.used_export_indices;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("PACK", total_send, [=] HD(int s) {
            pack::pack_v_mesh_body(s, used_export_indices, mesh->v_mesh, sendbuf_v_mesh);
        });
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_v_mesh, sizeof(POINT_TYPE) * (size_t)total_send);
        exchange_used_subset(halo.sendbuf_v_mesh, halo.recvbuf_v_mesh, halo.mpi_point_t, MSG_V_MESH);
        mpi_sync_after_recv(halo.recvbuf_v_mesh, sizeof(POINT_TYPE) * (size_t)n_recv);
    }

    {
        auto* recvbuf_v_mesh    = halo.recvbuf_v_mesh;
        auto* used_to_full_slot = halo.used_to_full_slot;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("UNPACK", n_recv, [=] HD(int slot) {
            pack::unpack_v_mesh_body(slot, used_to_full_slot, recvbuf_v_mesh, mesh->v_mesh_g);
        });
    }
#else
    (void)mesh;
#endif
#endif
}

// the smallest timestep of all ranks
void halo_dt_allreduce(double* dt) {
#ifdef USE_MPI
    PROFILE_MPI("DT_ALLREDUCE");
    double local = *dt;
    MPI_Allreduce(&local, dt, 1, MPI_DOUBLE, MPI_MIN, decomp.cart_comm);
#else
    (void)dt;
#endif
}

// exported cells whose seed the fallback moved, sorted by neighbour
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

// tells every neighbour which of its ghosts moved, and where to
void halo_exchange_moved_seeds(const MovedExportLists& lists, std::vector<MovedSeed>* received) {
    received->clear();
    if (halo.n_neighbors == 0) return;

#ifdef USE_MPI
    const int nn = halo.n_neighbors;

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
            MPI_Isend(
                &sendcnt[n], 1, MPI_INT, peer, msg_tag(dx, dy, dz, MSG_MOVED_COUNT), decomp.cart_comm, &reqs[n_reqs++]);
            MPI_Irecv(&recvcnt[n],
                      1,
                      MPI_INT,
                      peer,
                      msg_tag(-dx, -dy, -dz, MSG_MOVED_COUNT),
                      decomp.cart_comm,
                      &reqs[n_reqs++]);
        }
        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    }

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
                MPI_Isend(lists.js[n].data(),
                          sendcnt[n],
                          MPI_INT,
                          peer,
                          msg_tag(dx, dy, dz, MSG_MOVED_SLOT),
                          decomp.cart_comm,
                          &reqs[n_reqs++]);
                MPI_Isend(lists.pos[n].data(),
                          sendcnt[n],
                          halo.mpi_point_t,
                          peer,
                          msg_tag(dx, dy, dz, MSG_MOVED_POS),
                          decomp.cart_comm,
                          &reqs[n_reqs++]);
            }
            if (recvcnt[n] > 0) {
                recv_js[n].resize(recvcnt[n]);
                recv_pos[n].resize(recvcnt[n]);
                MPI_Irecv(recv_js[n].data(),
                          recvcnt[n],
                          MPI_INT,
                          peer,
                          msg_tag(-dx, -dy, -dz, MSG_MOVED_SLOT),
                          decomp.cart_comm,
                          &reqs[n_reqs++]);
                MPI_Irecv(recv_pos[n].data(),
                          recvcnt[n],
                          halo.mpi_point_t,
                          peer,
                          msg_tag(-dx, -dy, -dz, MSG_MOVED_POS),
                          decomp.cart_comm,
                          &reqs[n_reqs++]);
            }
        }
        if (n_reqs > 0) MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    }

    for (int n = 0; n < nn; n++) {
        for (int i = 0; i < recvcnt[n]; i++) {
            const int j = recv_js[n][i];
            if (j < 0 || j >= halo.recv_count[n]) {
                exit_failure("HALO: moved-seed slot offset %d out of range [0, %d) for neighbour %d\n",
                             j,
                             halo.recv_count[n],
                             n);
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
// cell volumes of the used ghosts
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
        auto* sendbuf_vol         = halo.sendbuf_vol;
        auto* used_export_indices = halo.used_export_indices;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("PACK", total_send, [=] HD(int s) {
            pack::pack_vol_body(s, used_export_indices, mesh->volumes, sendbuf_vol);
        });
    }

    {
        PROFILE_MPI("WAIT");
        mpi_sync_before_send(halo.sendbuf_vol, sizeof(double) * (size_t)total_send);
        exchange_used_subset(halo.sendbuf_vol, halo.recvbuf_vol, MPI_DOUBLE, MSG_VOL);
        mpi_sync_after_recv(halo.recvbuf_vol, sizeof(double) * (size_t)n_recv);
    }

    {
        auto* recvbuf_vol       = halo.recvbuf_vol;
        auto* used_to_full_slot = halo.used_to_full_slot;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("UNPACK", n_recv, [=] HD(int slot) {
            pack::unpack_vol_body(slot, used_to_full_slot, recvbuf_vol, mesh->volumes_g);
        });
    }
#endif
}
#endif

// one number summed over all ranks
void halo_sum_allreduce(double* v) {
#ifdef USE_MPI
    PROFILE_MPI("SUM_ALLREDUCE");
    double local = *v;
    MPI_Allreduce(&local, v, 1, MPI_DOUBLE, MPI_SUM, decomp.cart_comm);
#else
    (void)v;
#endif
}
