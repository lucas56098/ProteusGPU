// Build the export list (full halo + used subset) and remap indices after a
// local reorder. Included into halo.cu inside namespace proteus_mpi.

#ifdef USE_MPI
// forward declarations
static void count_send_per_neighbor(const POINT_TYPE* local_seeds, int n_local, double buff, int W);
static void fill_export_slots(const POINT_TYPE* local_seeds, int n_local, double buff, int W);
static void exchange_send_recv_counts();
static void mark_used_recv_bitmap(VMesh* mesh, int n_hydro, int n_mpi);
static int  build_used_recv_layout();
static int  pack_used_export_indices();
#endif

// ============================================================
// Public entry points
// ============================================================

void halo_build_exports(const POINT_TYPE* local_seeds, int n_local, double buff, int W_in) {
    halo.n_mpi_ghosts      = 0;
    halo.used_subset_ready = 0;
    for (int n = 0; n < HALO_MAX_NEIGHBORS; n++) {
        halo.send_count[n]   = 0;
        halo.recv_count[n]   = 0;
        halo.send_n_outer[n] = 0;
        halo.recv_n_outer[n] = 0;
        halo.ghost_offset[n] = 0;
    }
    halo.ghost_offset[HALO_MAX_NEIGHBORS] = 0;
    if (halo.n_neighbors == 0) return;

#ifdef USE_MPI
    Profiler::StartTimer("MPI_BUILD");

    const int nn = halo.n_neighbors;
    const int W  = (W_in > 0) ? W_in : brick_halo_width(buff, decomp.N_grid_global);

    count_send_per_neighbor(local_seeds, n_local, buff, W);

    // send_offset prefix scan + capacity check
    int total_send = 0;
    for (int n = 0; n < nn; n++) {
        halo.send_offset[n] = total_send;
        total_send += halo.send_count[n];
    }
    halo.send_offset[nn] = total_send;
    if (total_send > halo.n_mpi_capacity) {
        exit_failure("[rank %d] HALO: total send=%d > n_mpi_capacity=%d (n_local=%d).\n",
            decomp.rank, total_send, halo.n_mpi_capacity, n_local);
    }

    fill_export_slots(local_seeds, n_local, buff, W);
    Profiler::EndTimer("MPI_BUILD");

    Profiler::StartTimer("MPI_WAIT");
    exchange_send_recv_counts();
    Profiler::EndTimer("MPI_WAIT");

    // ghost_offset prefix scan + capacity check
    int sum = 0;
    for (int n = 0; n < nn; n++) {
        halo.ghost_offset[n] = sum;
        sum += halo.recv_count[n];
    }
    halo.ghost_offset[nn] = sum;
    halo.n_mpi_ghosts     = sum;

    if (sum > halo.n_mpi_capacity) {
        exit_failure("[rank %d] HALO: total recv (%d) > n_mpi_capacity (%d).\n",
            decomp.rank, sum, halo.n_mpi_capacity);
    }
#else
    (void)local_seeds; (void)n_local; (void)buff; (void)W_in;
#endif
}

void halo_remap_export_indices(const unsigned int* inv_gather, int n_local) {
#ifndef USE_MPI
    (void)inv_gather; (void)n_local;
    return;
#else
    if (halo.n_neighbors == 0) return;
    const int total = halo.send_offset[halo.n_neighbors];
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < total; s++) {
        const int old_k = halo.export_indices[s];
        if (old_k < 0 || old_k >= n_local) continue;
        halo.export_indices[s] = (int)inv_gather[old_k];
    }
    // used-subset will be rebuilt after the mesh converges on the post-remap layout
    halo.used_subset_ready = 0;
#endif
}

void halo_build_used_subset(VMesh* mesh) {
    // default: empty subset — lets mesh-only paths (no MPI neighbors) skip the rest
    for (int n = 0; n < HALO_MAX_NEIGHBORS; n++) {
        halo.used_send_count[n]  = 0;
        halo.used_recv_count[n]  = 0;
        halo.used_send_offset[n] = 0;
        halo.used_recv_offset[n] = 0;
    }
    halo.used_send_offset[HALO_MAX_NEIGHBORS] = 0;
    halo.used_recv_offset[HALO_MAX_NEIGHBORS] = 0;
    halo.n_used_send       = 0;
    halo.n_used_recv       = 0;
    halo.used_subset_ready = 1;

    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;

#ifdef USE_MPI
    Profiler::StartTimer("MPI_BUILD");
    const int n_hydro = (int)mesh->n_hydro;
    const int n_mpi   = halo.n_mpi_ghosts;

    mark_used_recv_bitmap(mesh, n_hydro, n_mpi);
    halo.n_used_recv = build_used_recv_layout();
    Profiler::EndTimer("MPI_BUILD");

    // exchange bitmap so each sender learns which of its cells are used.
    // the bitmap lives in receive-side layout on us; remotely it lands in
    // send-side layout — i.e. roles are flipped vs the data path.
    Profiler::StartTimer("MPI_WAIT");
    neighbor_exchange(halo.recv_used_bitmap, halo.send_used_bitmap, MPI_BYTE, MSG_USED_BITMAP,
                      halo.recv_count, halo.ghost_offset,
                      halo.send_count, halo.send_offset);
    Profiler::EndTimer("MPI_WAIT");

    Profiler::StartTimer("MPI_BUILD");
    halo.n_used_send = pack_used_export_indices();
    Profiler::EndTimer("MPI_BUILD");
#else
    (void)mesh;
#endif
}

// ============================================================
// Static helpers
// ============================================================

#ifdef USE_MPI

// pass 1: for each local cell, count how many neighbor directions it ships to
// (and how many of those are in the outermost layer)
static void count_send_per_neighbor(const POINT_TYPE* local_seeds, int n_local, double buff, int W) {
    const int nn     = halo.n_neighbors;
    const int N_grid = decomp.N_grid_global;
    const int b0x = decomp.b0[0], b0y = decomp.b0[1], b0z = decomp.b0[2];
    const int b1x = decomp.b1[0], b1y = decomp.b1[1], b1z = decomp.b1[2];

    int ndx[HALO_MAX_NEIGHBORS], ndy[HALO_MAX_NEIGHBORS], ndz[HALO_MAX_NEIGHBORS];
    for (int n = 0; n < nn; n++) {
        ndx[n] = halo.neighbor_dirs[n][0];
        ndy[n] = halo.neighbor_dirs[n][1];
        ndz[n] = halo.neighbor_dirs[n][2];
    }

    int send_count[HALO_MAX_NEIGHBORS]   = {0};
    int send_n_outer[HALO_MAX_NEIGHBORS] = {0};

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+:send_count[:HALO_MAX_NEIGHBORS]) reduction(+:send_n_outer[:HALO_MAX_NEIGHBORS])
#endif
    for (int k = 0; k < n_local; k++) {
        const double px = local_seeds[k].x;
        const double py = local_seeds[k].y;
#ifdef dim_3D
        const double pz = local_seeds[k].z;
#else
        const double pz = 0.0;
#endif
        int bx, by, bz;
        decomp_bucket_of_point(px, py, pz, N_grid, buff, &bx, &by, &bz);

        const BoundaryFlags f = classify_brick_boundary(bx, by, bz, b0x, b1x, b0y, b1y, b0z, b1z, W);
        if (!touches_brick_boundary(f)) continue;

        for (int n = 0; n < nn; n++) {
            const int dx = ndx[n], dy = ndy[n], dz = ndz[n];
            if (!ships_to_neighbor(f, dx, dy, dz)) continue;
            send_count[n]++;
            if (ships_to_outer_layer(f, dx, dy, dz)) send_n_outer[n]++;
        }
    }

    for (int n = 0; n < nn; n++) {
        halo.send_count[n]   = send_count[n];
        halo.send_n_outer[n] = send_n_outer[n];
    }
}

// pass 2: pack export_indices and dir_of_slot. outermost-layer cells go to the
// front of each neighbor's range, inner-layer cells after them.
static void fill_export_slots(const POINT_TYPE* local_seeds, int n_local, double buff, int W) {
    const int nn     = halo.n_neighbors;
    const int N_grid = decomp.N_grid_global;
    const int b0x = decomp.b0[0], b0y = decomp.b0[1], b0z = decomp.b0[2];
    const int b1x = decomp.b1[0], b1y = decomp.b1[1], b1z = decomp.b1[2];

    int ndx[HALO_MAX_NEIGHBORS], ndy[HALO_MAX_NEIGHBORS], ndz[HALO_MAX_NEIGHBORS];
    for (int n = 0; n < nn; n++) {
        ndx[n] = halo.neighbor_dirs[n][0];
        ndy[n] = halo.neighbor_dirs[n][1];
        ndz[n] = halo.neighbor_dirs[n][2];
    }

    int outer_cur[HALO_MAX_NEIGHBORS] = {0};
    int inner_cur[HALO_MAX_NEIGHBORS];
    for (int n = 0; n < nn; n++) inner_cur[n] = halo.send_n_outer[n];

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < n_local; k++) {
        const double px = local_seeds[k].x;
        const double py = local_seeds[k].y;
#ifdef dim_3D
        const double pz = local_seeds[k].z;
#else
        const double pz = 0.0;
#endif
        int bx, by, bz;
        decomp_bucket_of_point(px, py, pz, N_grid, buff, &bx, &by, &bz);

        const BoundaryFlags f = classify_brick_boundary(bx, by, bz, b0x, b1x, b0y, b1y, b0z, b1z, W);
        if (!touches_brick_boundary(f)) continue;

        for (int n = 0; n < nn; n++) {
            const int dx = ndx[n], dy = ndy[n], dz = ndz[n];
            if (!ships_to_neighbor(f, dx, dy, dz)) continue;

            int j;
            if (ships_to_outer_layer(f, dx, dy, dz)) {
#ifdef USE_OPENMP
#pragma omp atomic capture
#endif
                j = outer_cur[n]++;
            } else {
#ifdef USE_OPENMP
#pragma omp atomic capture
#endif
                j = inner_cur[n]++;
            }
            const int slot            = halo.send_offset[n] + j;
            halo.export_indices[slot] = k;
            halo.dir_of_slot[slot]    = (unsigned char)n;
        }
    }
}

// ship (send_count, send_n_outer) as a paired-int handshake per neighbor
static void exchange_send_recv_counts() {
    const int nn = halo.n_neighbors;
    int sendpair[2 * HALO_MAX_NEIGHBORS];
    int recvpair[2 * HALO_MAX_NEIGHBORS] = {0};
    for (int n = 0; n < nn; n++) {
        sendpair[2 * n + 0] = halo.send_count[n];
        sendpair[2 * n + 1] = halo.send_n_outer[n];
    }
    if (halo.use_neighbor_coll) {
        MPI_Neighbor_alltoall(sendpair, 2, MPI_INT, recvpair, 2, MPI_INT, halo.graph_comm);
    } else {
        MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
        int         n_reqs = 0;
        for (int n = 0; n < nn; n++) {
            const int dx   = halo.neighbor_dirs[n][0];
            const int dy   = halo.neighbor_dirs[n][1];
            const int dz   = halo.neighbor_dirs[n][2];
            const int peer = halo.neighbor_ranks[n];
            MPI_Isend(&sendpair[2 * n], 2, MPI_INT, peer,
                      msg_tag(dx, dy, dz, MSG_COUNTS), decomp.cart_comm, &reqs[n_reqs++]);
            MPI_Irecv(&recvpair[2 * n], 2, MPI_INT, peer,
                      msg_tag(-dx, -dy, -dz, MSG_COUNTS), decomp.cart_comm, &reqs[n_reqs++]);
        }
        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    }
    for (int n = 0; n < nn; n++) {
        halo.recv_count[n]   = recvpair[2 * n + 0];
        halo.recv_n_outer[n] = recvpair[2 * n + 1];
    }
}

// mark recv_used_bitmap[i]=1 for every MPI ghost referenced by a local face.
// periodic ghosts get remapped to their source-real local k (< n_hydro) by
// sid_to_neighbor, so they never appear here as ghosts.
static void mark_used_recv_bitmap(VMesh* mesh, int n_hydro, int n_mpi) {
    const hsize_t num_faces = mesh->num_faces;
    const int*    nc        = mesh->neighbor_cell;
    const int     mpi_base  = n_hydro;
    const int     mpi_top   = n_hydro + n_mpi;

    std::memset(halo.recv_used_bitmap, 0, (size_t)n_mpi);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (hsize_t f = 0; f < num_faces; f++) {
        const int kn = nc[f];
        if (kn < mpi_base || kn >= mpi_top) continue;
        halo.recv_used_bitmap[kn - mpi_base] = 1;
    }
}

// from the bitmap: per-direction counts/offsets + used_to_full_slot
static int build_used_recv_layout() {
    const int nn = halo.n_neighbors;

    int total_used_recv = 0;
    for (int n = 0; n < nn; n++) {
        int count = 0;
        const int ghost_off = halo.ghost_offset[n];
        for (int j = 0; j < halo.recv_count[n]; j++) {
            if (halo.recv_used_bitmap[ghost_off + j]) count++;
        }
        halo.used_recv_count[n]  = count;
        halo.used_recv_offset[n] = total_used_recv;
        total_used_recv += count;
    }
    halo.used_recv_offset[nn] = total_used_recv;

    int cursor = 0;
    for (int n = 0; n < nn; n++) {
        const int ghost_off = halo.ghost_offset[n];
        for (int j = 0; j < halo.recv_count[n]; j++) {
            if (halo.recv_used_bitmap[ghost_off + j]) {
                halo.used_to_full_slot[cursor++] = ghost_off + j;
            }
        }
    }
    return total_used_recv;
}

// compact used_export_indices using send_used_bitmap (received from peers)
static int pack_used_export_indices() {
    const int nn = halo.n_neighbors;
    int total_used_send = 0;
    for (int n = 0; n < nn; n++) {
        halo.used_send_offset[n] = total_used_send;
        const int s_off = halo.send_offset[n];
        const int sc    = halo.send_count[n];
        int       count = 0;
        for (int j = 0; j < sc; j++) {
            if (halo.send_used_bitmap[s_off + j]) {
                halo.used_export_indices[total_used_send + count] = halo.export_indices[s_off + j];
                count++;
            }
        }
        halo.used_send_count[n] = count;
        total_used_send += count;
    }
    halo.used_send_offset[nn] = total_used_send;
    return total_used_send;
}

#endif // USE_MPI
