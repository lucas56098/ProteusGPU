// picks the cells to export and finds the ghosts they become (included by halo.cu)

#ifdef USE_MPI
static void count_send_per_neighbor(const POINT_TYPE* local_seeds, int n_local, double buff, int W);
static void fill_export_slots(const POINT_TYPE* local_seeds, int n_local, double buff, int W);
static void exchange_send_recv_counts();
static void mark_used_recv_bitmap(VMesh* mesh, int n_hydro, int n_mpi);
static int  build_used_recv_layout();
static int  pack_used_export_indices();
#endif

// export lists for every neighbour, from the current seed positions
void halo_build_exports(const POINT_TYPE* local_seeds, int n_local, double buff, int W_in) {
    halo.n_mpi_ghosts      = 0;
    halo.used_subset_ready = 0;
    for (int n = 0; n < HALO_MAX_NEIGHBORS; n++) {
        halo.send_count[n]   = 0;
        halo.recv_count[n]   = 0;
        halo.ghost_offset[n] = 0;
    }
    halo.ghost_offset[HALO_MAX_NEIGHBORS] = 0;
    if (halo.n_neighbors == 0) return;

#ifdef USE_MPI
    PROFILE("HALO_BUILD");

    const int nn = halo.n_neighbors;
    const int W  = (W_in > 0) ? W_in : brick_halo_width(buff, decomp.N_grid_global); // 0 asks for the default

    count_send_per_neighbor(local_seeds, n_local, buff, W);

    // one block of slots per neighbour
    int total_send = 0;
    for (int n = 0; n < nn; n++) {
        halo.send_offset[n] = total_send;
        total_send += halo.send_count[n];
    }
    halo.send_offset[nn] = total_send;
    // more cells to send than the buffers hold
    if (total_send > halo.n_mpi_capacity) { halo_grow_capacity(total_send); }

    fill_export_slots(local_seeds, n_local, buff, W);

    {
        PROFILE_MPI("COUNT_WAIT");
        exchange_send_recv_counts();
    }

    // the ghosts arrive in the same order, one block per neighbour
    int sum = 0;
    for (int n = 0; n < nn; n++) {
        halo.ghost_offset[n] = sum;
        sum += halo.recv_count[n];
    }
    halo.ghost_offset[nn] = sum;
    halo.n_mpi_ghosts     = sum;

    // the export lists are gone after the grow, so fill them again
    if (sum > halo.n_mpi_capacity) {
        halo_grow_capacity(sum);
        fill_export_slots(local_seeds, n_local, buff, W);
    }
#else
    (void)local_seeds;
    (void)n_local;
    (void)buff;
    (void)W_in;
#endif
}

// the build sorted the cells into k-order, the stored cell indices follow
void halo_remap_export_indices(const unsigned int* inv_gather, int n_local) {
#ifndef USE_MPI
    (void)inv_gather;
    (void)n_local;
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
    halo.used_subset_ready = 0;
#endif
}

// finds the ghosts a local cell really touches, and tells their owner
void halo_build_used_subset(VMesh* mesh) {
    for (int n = 0; n < HALO_MAX_NEIGHBORS; n++) {
        halo.used_send_count[n]  = 0;
        halo.used_recv_count[n]  = 0;
        halo.used_send_offset[n] = 0;
        halo.used_recv_offset[n] = 0;
    }
    halo.used_send_offset[HALO_MAX_NEIGHBORS] = 0;
    halo.used_recv_offset[HALO_MAX_NEIGHBORS] = 0;
    halo.n_used_send                          = 0;
    halo.n_used_recv                          = 0;
    halo.used_subset_ready                    = 1;

    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;

#ifdef USE_MPI
    PROFILE("HALO_USED_BUILD");
    const int n_hydro = (int)mesh->n_hydro;
    const int n_mpi   = halo.n_mpi_ghosts;

    // a ghost is used when it is a face neighbour of a local cell
    mark_used_recv_bitmap(mesh, n_hydro, n_mpi);
    halo.n_used_recv = build_used_recv_layout();

    {
        PROFILE_MPI("BITMAP_WAIT");
        // the owner of the cell needs to know that, so the bitmap goes back
        neighbor_exchange(halo.recv_used_bitmap,
                          halo.send_used_bitmap,
                          MPI_BYTE,
                          MSG_USED_BITMAP,
                          halo.recv_count,
                          halo.ghost_offset,
                          halo.send_count,
                          halo.send_offset);
    }

    halo.n_used_send = pack_used_export_indices();
#else
    (void)mesh;
#endif
}

#ifdef USE_MPI

// the cells are counted in fixed chunks, so every slot has the same place in every run
static constexpr int EXPORT_CHUNKS = 1024;

static int s_chunk_outer[EXPORT_CHUNKS][HALO_MAX_NEIGHBORS];
static int s_chunk_inner[EXPORT_CHUNKS][HALO_MAX_NEIGHBORS];

static inline void export_chunk_range(int c, int n_local, int* lo, int* hi) {
    const long long chunk = ((long long)n_local + EXPORT_CHUNKS - 1) / EXPORT_CHUNKS;
    long long       a     = (long long)c * chunk;
    long long       b     = a + chunk;
    if (a > n_local) a = n_local;
    if (b > n_local) b = n_local;
    *lo = (int)a;
    *hi = (int)b;
}

// how many cells each chunk sends to each neighbour
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

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < EXPORT_CHUNKS; c++) {
        int cnt_outer[HALO_MAX_NEIGHBORS] = {0};
        int cnt_inner[HALO_MAX_NEIGHBORS] = {0};

        int lo, hi;
        export_chunk_range(c, n_local, &lo, &hi);
        for (int k = lo; k < hi; k++) {
            const double px = local_seeds[k].x;
            const double py = local_seeds[k].y;
#ifdef dim_3D
            const double pz = local_seeds[k].z;
#else
            const double pz = 0.0;
#endif
            int bx, by, bz;
            decomp_bucket_of_point(px, py, pz, N_grid, buff, &bx, &by, &bz);

            // a cell deep inside the brick goes nowhere
            const BoundaryFlags f = classify_brick_boundary(bx, by, bz, b0x, b1x, b0y, b1y, b0z, b1z, W);
            if (!touches_brick_boundary(f)) continue;

            for (int n = 0; n < nn; n++) {
                const int dx = ndx[n], dy = ndy[n], dz = ndz[n];
                if (!ships_to_neighbor(f, dx, dy, dz)) continue;
                if (ships_to_outer_layer(f, dx, dy, dz))
                    cnt_outer[n]++;
                else
                    cnt_inner[n]++;
            }
        }

        for (int n = 0; n < nn; n++) {
            s_chunk_outer[c][n] = cnt_outer[n];
            s_chunk_inner[c][n] = cnt_inner[n];
        }
    }

    // turn the counts into the first slot of every chunk, outer layer cells before the inner ones
    for (int n = 0; n < nn; n++) {
        int run = 0;
        for (int c = 0; c < EXPORT_CHUNKS; c++) {
            const int t         = s_chunk_outer[c][n];
            s_chunk_outer[c][n] = run;
            run += t;
        }
        for (int c = 0; c < EXPORT_CHUNKS; c++) {
            const int t         = s_chunk_inner[c][n];
            s_chunk_inner[c][n] = run;
            run += t;
        }
        halo.send_count[n] = run;
    }
}

// second pass: every chunk writes its cells into the slots it got
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

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < EXPORT_CHUNKS; c++) {

        int outer_cur[HALO_MAX_NEIGHBORS], inner_cur[HALO_MAX_NEIGHBORS];
        for (int n = 0; n < nn; n++) {
            outer_cur[n] = s_chunk_outer[c][n];
            inner_cur[n] = s_chunk_inner[c][n];
        }

        int lo, hi;
        export_chunk_range(c, n_local, &lo, &hi);
        for (int k = lo; k < hi; k++) {
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

                const int j               = ships_to_outer_layer(f, dx, dy, dz) ? outer_cur[n]++ : inner_cur[n]++;
                const int slot            = halo.send_offset[n] + j;
                halo.export_indices[slot] = k;
                halo.dir_of_slot[slot]    = (unsigned char)n;
            }
        }
    }
}

// trade the counts, so every rank knows how many ghosts it gets
static void exchange_send_recv_counts() {
    const int nn = halo.n_neighbors;
    if (halo.use_neighbor_coll) {
        MPI_Neighbor_alltoall(halo.send_count, 1, MPI_INT, halo.recv_count, 1, MPI_INT, halo.graph_comm);
    } else {
        MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
        int         n_reqs = 0;
        for (int n = 0; n < nn; n++) {
            const int dx   = halo.neighbor_dirs[n][0];
            const int dy   = halo.neighbor_dirs[n][1];
            const int dz   = halo.neighbor_dirs[n][2];
            const int peer = halo.neighbor_ranks[n];
            MPI_Isend(&halo.send_count[n],
                      1,
                      MPI_INT,
                      peer,
                      msg_tag(dx, dy, dz, MSG_COUNTS),
                      decomp.cart_comm,
                      &reqs[n_reqs++]);
            MPI_Irecv(&halo.recv_count[n],
                      1,
                      MPI_INT,
                      peer,
                      msg_tag(-dx, -dy, -dz, MSG_COUNTS),
                      decomp.cart_comm,
                      &reqs[n_reqs++]);
        }
        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    }
}

// walk the faces: every ghost behind one of them is used
static void mark_used_recv_bitmap(VMesh* mesh, int n_hydro, int n_mpi) {
    const int  num_faces = (int)mesh->num_faces;
    const int* nc        = mesh->neighbor_cell;
    const int  mpi_base  = n_hydro;
    const int  mpi_top   = n_hydro + n_mpi;

    gpu_memset(halo.recv_used_bitmap, 0, (size_t)n_mpi);

    auto* recv_used_bitmap = halo.recv_used_bitmap;

    parallel_for<_MPI_PACK_BLOCK_SIZE_>("BITMAP_MARK", num_faces, [=] HD(int f) {
        pack::mark_used_bitmap_body(f, nc, mpi_base, mpi_top, recv_used_bitmap);
    });
}

// pack the used ghosts of every neighbour into one block
static int build_used_recv_layout() {
    const int nn = halo.n_neighbors;

    int total_used_recv = 0;
    for (int n = 0; n < nn; n++) {
        int       count     = 0;
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
            if (halo.recv_used_bitmap[ghost_off + j]) { halo.used_to_full_slot[cursor++] = ghost_off + j; }
        }
    }
    return total_used_recv;
}

// same on the send side, from the bitmap the neighbour sent back
static int pack_used_export_indices() {
    const int nn              = halo.n_neighbors;
    int       total_used_send = 0;
    for (int n = 0; n < nn; n++) {
        halo.used_send_offset[n] = total_used_send;
        const int s_off          = halo.send_offset[n];
        const int sc             = halo.send_count[n];
        int       count          = 0;
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

#endif
