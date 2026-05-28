// Halo allocation, deallocation, and capacity estimation.
// Included into halo.cu inside namespace proteus_mpi.

#ifdef USE_MPI
// forward declarations
struct HaloCapacityInfo {
    int       n_capacity;
    int       W_alloc;
    long long geom_cells;
    double    rho_cells;
    int       Lx, Ly, Lz;
};
static void             pick_neighbor_collective_mode();
static HaloCapacityInfo estimate_halo_capacity(int n_local, double buff);
static void             allocate_halo_buffers(int n_capacity);
static void             register_mpi_datatypes();
#endif

// ============================================================
// Public entry points
// ============================================================

void halo_init(int n_local, double buff) {
#ifndef USE_MPI
    (void)n_local; (void)buff;
    halo.n_neighbors    = 0;
    halo.n_mpi_capacity = 0;
    return;
#else
    build_neighbor_table();

    if (halo.n_neighbors == 0) {
        halo.n_mpi_capacity    = 0;
        n_mpi_capacity         = 0;
        halo.use_neighbor_coll = 0;
        halo.graph_comm        = MPI_COMM_NULL;
        if (decomp.rank == 0) {
            printf("HALO: 0 Cart neighbors (single-rank topology) — halo disabled.\n");
            fflush(stdout);
        }
        return;
    }

    pick_neighbor_collective_mode();

    const HaloCapacityInfo info = estimate_halo_capacity(n_local, buff);
    halo.n_mpi_capacity = info.n_capacity;
    n_mpi_capacity      = info.n_capacity;

    allocate_halo_buffers(info.n_capacity);
    register_mpi_datatypes();
    halo.used_subset_ready = 0;

    if (decomp.rank == 0) {
        printf("HALO: %d Cart neighbors (%s), W_alloc=%d, brick=[%d,%d,%d], geom_cells=%lld, "
               "rho_cells=%.3f, total_cap=%d\n",
               halo.n_neighbors,
               halo.use_neighbor_coll ? "dist-graph + neighbor_alltoallv"
                                      : "Isend/Irecv per direction",
               info.W_alloc, info.Lx, info.Ly, info.Lz, info.geom_cells, info.rho_cells, info.n_capacity);
        fflush(stdout);
    }
#endif
}

void halo_free() {
#ifdef USE_MPI
    if (halo.export_indices)      gpu_free(halo.export_indices);
    if (halo.dir_of_slot)         gpu_free(halo.dir_of_slot);
    if (halo.used_export_indices) gpu_free(halo.used_export_indices);
    if (halo.used_to_full_slot)   gpu_free(halo.used_to_full_slot);
    if (halo.send_used_bitmap)    gpu_free(halo.send_used_bitmap);
    if (halo.recv_used_bitmap)    gpu_free(halo.recv_used_bitmap);
    if (halo.sendbuf_seed)        gpu_free(halo.sendbuf_seed);
    if (halo.recvbuf_seed)        gpu_free(halo.recvbuf_seed);
    if (halo.sendbuf_prim)        gpu_free(halo.sendbuf_prim);
    if (halo.recvbuf_prim)        gpu_free(halo.recvbuf_prim);
    if (halo.sendbuf_v_mesh)       gpu_free(halo.sendbuf_v_mesh);
    if (halo.recvbuf_v_mesh)       gpu_free(halo.recvbuf_v_mesh);
    if (halo.sendbuf_grad)        gpu_free(halo.sendbuf_grad);
    if (halo.recvbuf_grad)        gpu_free(halo.recvbuf_grad);
    if (halo.is_outer_layer)      gpu_free(halo.is_outer_layer);

    if (halo.n_neighbors > 0) {
        MPI_Type_free(&halo.mpi_prim_t);
        MPI_Type_free(&halo.mpi_point_t);
        MPI_Type_free(&halo.mpi_grad_cell_t);
        if (halo.graph_comm != MPI_COMM_NULL) MPI_Comm_free(&halo.graph_comm);
    }

    halo           = {};
    n_mpi_capacity = 0;
#endif
}

int halo_default_width(double buff) {
#ifdef USE_MPI
    return brick_halo_width(buff, decomp.N_grid_global);
#else
    (void)buff;
    return 0;
#endif
}

// ============================================================
// Static helpers
// ============================================================

#ifdef USE_MPI

// use MPI_Neighbor collectives via a dist-graph comm only when peers are
// all distinct (one message per neighbor); otherwise fall back to Isend/Irecv
static void pick_neighbor_collective_mode() {
    int distinct = 1;
    for (int i = 0; i < halo.n_neighbors && distinct; i++)
        for (int j = i + 1; j < halo.n_neighbors && distinct; j++)
            if (halo.neighbor_ranks[i] == halo.neighbor_ranks[j]) distinct = 0;
    halo.use_neighbor_coll = distinct;

    if (halo.use_neighbor_coll) {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Dist_graph_create_adjacent(decomp.cart_comm,
                                       halo.n_neighbors, halo.neighbor_ranks, MPI_UNWEIGHTED,
                                       halo.n_neighbors, halo.neighbor_ranks, MPI_UNWEIGHTED,
                                       info, /*reorder=*/0, &halo.graph_comm);
        MPI_Info_free(&info);
    } else {
        halo.graph_comm = MPI_COMM_NULL;
    }
}

// sized to fit the worst-case widening budget the periodic mesh might ask for
static HaloCapacityInfo estimate_halo_capacity(int n_local, double buff) {
    constexpr int    MAX_WIDEN_ITERS  = 4;
    constexpr int    W_STARTUP_MARGIN = 2;
    constexpr double SAFETY           = 1.5;

    HaloCapacityInfo info;
    info.W_alloc    = halo_default_width(buff) + W_STARTUP_MARGIN + 2 * (MAX_WIDEN_ITERS - 1);
    info.Lx         = decomp.b1[0] - decomp.b0[0];
    info.Ly         = decomp.b1[1] - decomp.b0[1];
    info.Lz         = decomp.b1[2] - decomp.b0[2];
    info.geom_cells = geom_total_cells(info.Lx, info.Ly, info.Lz, info.W_alloc);

    const long long total_buckets = (long long)info.Lx * (long long)info.Ly * (long long)info.Lz;
    info.rho_cells = (total_buckets > 0) ? (double)n_local / (double)total_buckets : 1.0;

    const long long est       = (long long)std::ceil(SAFETY * info.rho_cells * (double)info.geom_cells);
    const long long floor_cap = 1024;
    info.n_capacity = (int)std::max<long long>(floor_cap, est);
    return info;
}

static void allocate_halo_buffers(int n_capacity) {
    halo.export_indices      = (int*)gpu_malloc(sizeof(int) * n_capacity);
    halo.dir_of_slot         = (unsigned char*)gpu_malloc(sizeof(unsigned char) * n_capacity);
    halo.used_export_indices = (int*)gpu_malloc(sizeof(int) * n_capacity);
    halo.used_to_full_slot   = (int*)gpu_malloc(sizeof(int) * n_capacity);
    halo.send_used_bitmap    = (unsigned char*)gpu_malloc(sizeof(unsigned char) * n_capacity);
    halo.recv_used_bitmap    = (unsigned char*)gpu_malloc(sizeof(unsigned char) * n_capacity);

    halo.sendbuf_seed  = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity);
    halo.recvbuf_seed  = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity);
    halo.sendbuf_prim  = (HaloPrimCell*)gpu_malloc(sizeof(HaloPrimCell) * n_capacity);
    halo.recvbuf_prim  = (HaloPrimCell*)gpu_malloc(sizeof(HaloPrimCell) * n_capacity);
    halo.sendbuf_v_mesh = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity);
    halo.recvbuf_v_mesh = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity);

    const int grad_components = 3 + DIMENSION;
    halo.sendbuf_grad = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity * grad_components);
    halo.recvbuf_grad = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity * grad_components);

    halo.is_outer_layer = (unsigned char*)gpu_malloc(sizeof(unsigned char) * n_capacity);
}

static void register_mpi_datatypes() {
    const int grad_components = 3 + DIMENSION;
    MPI_Type_contiguous(sizeof(HaloPrimCell), MPI_BYTE, &halo.mpi_prim_t);
    MPI_Type_commit(&halo.mpi_prim_t);
    MPI_Type_contiguous(sizeof(POINT_TYPE), MPI_BYTE, &halo.mpi_point_t);
    MPI_Type_commit(&halo.mpi_point_t);
    MPI_Type_contiguous(grad_components * (int)sizeof(POINT_TYPE), MPI_BYTE, &halo.mpi_grad_cell_t);
    MPI_Type_commit(&halo.mpi_grad_cell_t);
}

#endif // USE_MPI
