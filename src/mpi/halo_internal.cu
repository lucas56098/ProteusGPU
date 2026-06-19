// Shared low-level halo helpers used by halo_init / halo_build / halo_exchange.
// Included into halo.cu inside namespace proteus_mpi.

// ============================================================
// Neighbor topology
// ============================================================

#ifdef USE_MPI
static void build_neighbor_table() {
    halo.n_neighbors = 0;

    int my_coords[3];
    MPI_Cart_coords(decomp.cart_comm, decomp.rank, 3, my_coords);

#ifdef dim_3D
    const int dz_lo = -1, dz_hi = 1;
#else
    const int dz_lo = 0, dz_hi = 0;
#endif

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = dz_lo; dz <= dz_hi; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                int coords[3]     = {my_coords[0] + dx, my_coords[1] + dy, my_coords[2] + dz};
                int neighbor_rank = 0;
                MPI_Cart_rank(decomp.cart_comm, coords, &neighbor_rank);

                if (neighbor_rank == decomp.rank) continue;

                double shift[3] = {0.0, 0.0, 0.0};
                for (int a = 0; a < 3; a++) {
                    if (coords[a] < 0) shift[a] = +1.0;
                    if (coords[a] >= decomp.dims[a]) shift[a] = -1.0;
                }

                const int n               = halo.n_neighbors;
                halo.neighbor_ranks[n]    = neighbor_rank;
                halo.neighbor_dirs[n][0]  = dx;
                halo.neighbor_dirs[n][1]  = dy;
                halo.neighbor_dirs[n][2]  = dz;
                halo.neighbor_shift[n][0] = shift[0];
                halo.neighbor_shift[n][1] = shift[1];
                halo.neighbor_shift[n][2] = shift[2];
                halo.n_neighbors++;
            }
        }
    }
}
#endif

// ============================================================
// Halo width + geometric cell count
// ============================================================

#ifdef USE_MPI
static inline int brick_halo_width(double buff, int N_grid) {
    const double w = buff * (double)N_grid / (1.0 + 2.0 * buff);
    const int    W = (int)std::ceil(w);
    return W < 1 ? 1 : W;
}

// total ghost cell count for a brick of size (Lx,Ly,Lz) and halo width W
static long long geom_total_cells(int Lx, int Ly, int Lz, int W) {
    long long total = 0;
#ifdef dim_3D
    const int dz_lo = -1, dz_hi = 1;
#else
    const int dz_lo = 0, dz_hi = 0;
    Lz = 1;
#endif
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = dz_lo; dz <= dz_hi; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                long long c = 1;
                c *= (dx == 0) ? Lx : W;
                c *= (dy == 0) ? Ly : W;
                c *= (dz == 0) ? Lz : W;
                total += c;
            }
        }
    }
    return total;
}
#endif

// ============================================================
// Brick-boundary classification
// ============================================================

// per-cell boundary-layer flags: which faces of the brick this cell sits near,
// and whether it sits on the *outermost* bucket of each face.
struct BoundaryFlags {
    int x_lo, x_hi, y_lo, y_hi, z_lo, z_hi;
    int x_out_lo, x_out_hi, y_out_lo, y_out_hi, z_out_lo, z_out_hi;
};

static inline BoundaryFlags
classify_brick_boundary(int bx, int by, int bz, int b0x, int b1x, int b0y, int b1y, int b0z, int b1z, int W) {
    BoundaryFlags f;
    f.x_lo = (bx < b0x + W);
    f.x_hi = (bx >= b1x - W);
    f.y_lo = (by < b0y + W);
    f.y_hi = (by >= b1y - W);
#ifdef dim_3D
    f.z_lo = (bz < b0z + W);
    f.z_hi = (bz >= b1z - W);
#else
    (void)bz;
    (void)b0z;
    (void)b1z;
    f.z_lo = 0;
    f.z_hi = 0;
#endif
    f.x_out_lo = (bx == b0x + W - 1);
    f.x_out_hi = (bx == b1x - W);
    f.y_out_lo = (by == b0y + W - 1);
    f.y_out_hi = (by == b1y - W);
#ifdef dim_3D
    f.z_out_lo = (bz == b0z + W - 1);
    f.z_out_hi = (bz == b1z - W);
#else
    f.z_out_lo = 0;
    f.z_out_hi = 0;
#endif
    return f;
}

static inline bool touches_brick_boundary(const BoundaryFlags& f) {
    return (f.x_lo | f.x_hi | f.y_lo | f.y_hi | f.z_lo | f.z_hi);
}

static inline bool ships_to_neighbor(const BoundaryFlags& f, int dx, int dy, int dz) {
    const int x_ok = (dx == 0) ? 1 : (dx < 0 ? f.x_lo : f.x_hi);
    const int y_ok = (dy == 0) ? 1 : (dy < 0 ? f.y_lo : f.y_hi);
    const int z_ok = (dz == 0) ? 1 : (dz < 0 ? f.z_lo : f.z_hi);
    return (x_ok & y_ok & z_ok);
}

static inline bool ships_to_outer_layer(const BoundaryFlags& f, int dx, int dy, int dz) {
    const int x_out = (dx == 0) ? 1 : (dx < 0 ? f.x_out_lo : f.x_out_hi);
    const int y_out = (dy == 0) ? 1 : (dy < 0 ? f.y_out_lo : f.y_out_hi);
    const int z_out = (dz == 0) ? 1 : (dz < 0 ? f.z_out_lo : f.z_out_hi);
    return (x_out & y_out & z_out);
}

// ============================================================
// Direction tags + neighbor exchange primitive
// ============================================================

#ifdef USE_MPI
enum HaloMsgKind {
    MSG_COUNTS      = 0,
    MSG_SEED        = 1,
    MSG_PRIM        = 2,
    MSG_GRAD        = 3,
    MSG_V_MESH      = 4,
    MSG_USED_BITMAP = 5,
};

// (dx,dy,dz) -> [1, 27]
static inline int dir_tag(int dx, int dy, int dz) {
    return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1) + 1;
}
static inline int msg_tag(int dx, int dy, int dz, HaloMsgKind kind) {
    return dir_tag(dx, dy, dz) + (int)kind * 100;
}

// per-neighbor send/recv. uses MPI_Neighbor_alltoallv when peers are all
// distinct; otherwise issues one direction-tagged Isend/Irecv pair per neighbor.
static void neighbor_exchange(const void*  sendbuf,
                              void*        recvbuf,
                              MPI_Datatype dtype,
                              HaloMsgKind  kind,
                              const int*   sendcounts,
                              const int*   sdispls,
                              const int*   recvcounts,
                              const int*   rdispls) {
    if (halo.use_neighbor_coll) {
        MPI_Neighbor_alltoallv(
            sendbuf, sendcounts, sdispls, dtype, recvbuf, recvcounts, rdispls, dtype, halo.graph_comm);
        return;
    }
    int elem_bytes = 0;
    MPI_Type_size(dtype, &elem_bytes);
    const char* sbuf = (const char*)sendbuf;
    char*       rbuf = (char*)recvbuf;
    MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
    int         n_reqs = 0;
    for (int n = 0; n < halo.n_neighbors; n++) {
        const int dx   = halo.neighbor_dirs[n][0];
        const int dy   = halo.neighbor_dirs[n][1];
        const int dz   = halo.neighbor_dirs[n][2];
        const int peer = halo.neighbor_ranks[n];
        const int sc   = sendcounts[n];
        const int rc   = recvcounts[n];
        if (sc > 0) {
            MPI_Isend(sbuf + (size_t)sdispls[n] * elem_bytes,
                      sc,
                      dtype,
                      peer,
                      msg_tag(dx, dy, dz, kind),
                      decomp.cart_comm,
                      &reqs[n_reqs++]);
        }
        if (rc > 0) {
            MPI_Irecv(rbuf + (size_t)rdispls[n] * elem_bytes,
                      rc,
                      dtype,
                      peer,
                      msg_tag(-dx, -dy, -dz, kind),
                      decomp.cart_comm,
                      &reqs[n_reqs++]);
        }
    }
    if (n_reqs > 0) MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
}

static inline void exchange_full_halo(const void* sendbuf, void* recvbuf, MPI_Datatype dtype, HaloMsgKind kind) {
    neighbor_exchange(
        sendbuf, recvbuf, dtype, kind, halo.send_count, halo.send_offset, halo.recv_count, halo.ghost_offset);
}

static inline void exchange_used_subset(const void* sendbuf, void* recvbuf, MPI_Datatype dtype, HaloMsgKind kind) {
    neighbor_exchange(sendbuf,
                      recvbuf,
                      dtype,
                      kind,
                      halo.used_send_count,
                      halo.used_send_offset,
                      halo.used_recv_count,
                      halo.used_recv_offset);
}
#endif
