// halo setup and teardown (included by halo.cu)

#ifdef USE_MPI
// what the capacity estimate is made of, for the startup line
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
static void             sync_neighbor_shift_to_flat();
static void             register_mpi_datatypes();
#endif

// neighbour table, transport mode, buffers
void halo_init(int n_local, double buff) {
#ifndef USE_MPI
    (void)n_local;
    (void)buff;
    halo.n_neighbors    = 0;
    halo.n_mpi_capacity = 0;
    return;
#else
    build_neighbor_table();

    // a single rank, or a rank that only ever sees itself
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
    halo.n_mpi_capacity         = info.n_capacity;
    n_mpi_capacity              = info.n_capacity;

    allocate_halo_buffers(info.n_capacity);

    halo.neighbor_shift_flat = (double*)gpu_malloc(sizeof(double) * HALO_MAX_NEIGHBORS * 3);
    sync_neighbor_shift_to_flat();

    register_mpi_datatypes();
    halo.used_subset_ready = 0;

    if (decomp.rank == 0) {
        printf("HALO: %d Cart neighbors (%s), W_alloc=%d, brick=[%d,%d,%d], geom_cells=%lld, "
               "rho_cells=%.3f, total_cap=%d\n",
               halo.n_neighbors,
               halo.use_neighbor_coll ? "dist-graph + neighbor_alltoallv" : "Isend/Irecv per direction",
               info.W_alloc,
               info.Lx,
               info.Ly,
               info.Lz,
               info.geom_cells,
               info.rho_cells,
               info.n_capacity);
        fflush(stdout);
    }
#endif
}

// buffers, MPI types and the graph communicator
void halo_free() {
#ifdef USE_MPI
    if (halo.export_indices) gpu_free(halo.export_indices);
    if (halo.dir_of_slot) gpu_free(halo.dir_of_slot);
    if (halo.used_export_indices) gpu_free(halo.used_export_indices);
    if (halo.used_to_full_slot) gpu_free(halo.used_to_full_slot);
    if (halo.send_used_bitmap) gpu_free(halo.send_used_bitmap);
    if (halo.recv_used_bitmap) gpu_free(halo.recv_used_bitmap);
    if (halo.sendbuf_seed) gpu_free(halo.sendbuf_seed);
    if (halo.recvbuf_seed) gpu_free(halo.recvbuf_seed);
    if (halo.sendbuf_prim) gpu_free(halo.sendbuf_prim);
    if (halo.recvbuf_prim) gpu_free(halo.recvbuf_prim);
    if (halo.sendbuf_v_mesh) gpu_free(halo.sendbuf_v_mesh);
    if (halo.recvbuf_v_mesh) gpu_free(halo.recvbuf_v_mesh);
    if (halo.sendbuf_grad) gpu_free(halo.sendbuf_grad);
    if (halo.recvbuf_grad) gpu_free(halo.recvbuf_grad);
#ifdef VOL_REGULARIZE
    if (halo.sendbuf_vol) gpu_free(halo.sendbuf_vol);
    if (halo.recvbuf_vol) gpu_free(halo.recvbuf_vol);
#endif
    if (halo.is_outer_layer) gpu_free(halo.is_outer_layer);
    if (halo.neighbor_shift_flat) gpu_free(halo.neighbor_shift_flat);
    if (s_is_outer_meta_dev) {
        gpu_free(s_is_outer_meta_dev);
        s_is_outer_meta_dev = nullptr;
    }

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

// bucket layers that cover the ghost band
int halo_default_width(double buff) {
#ifdef USE_MPI
    return brick_halo_width(buff, decomp.N_grid_global);
#else
    (void)buff;
    return 0;
#endif
}

#ifdef USE_MPI

// the neighbour collective needs every neighbour rank to appear once, else send one message per direction
static void pick_neighbor_collective_mode() {
    int distinct = 1;
    for (int i = 0; i < halo.n_neighbors && distinct; i++)
        for (int j = i + 1; j < halo.n_neighbors && distinct; j++)
            if (halo.neighbor_ranks[i] == halo.neighbor_ranks[j]) distinct = 0;
    halo.use_neighbor_coll = distinct;

    if (halo.use_neighbor_coll) {
        MPI_Info info;
        MPI_Info_create(&info);
        // built from MPI_COMM_WORLD: on cart_comm the neighbour collectives fail in some MPI versions
        MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                       halo.n_neighbors,
                                       halo.neighbor_ranks,
                                       MPI_UNWEIGHTED,
                                       halo.n_neighbors,
                                       halo.neighbor_ranks,
                                       MPI_UNWEIGHTED,
                                       info,
                                       0,
                                       &halo.graph_comm);
        MPI_Info_free(&info);
    } else {
        halo.graph_comm = MPI_COMM_NULL;
    }
}

// buffer size: cells in the widest halo we may ever need, at the local cell density
static HaloCapacityInfo estimate_halo_capacity(int n_local, double buff) {
    constexpr int    MAX_WIDEN_ITERS  = 4;
    constexpr int    W_STARTUP_MARGIN = 2;
    constexpr double SAFETY           = 1.5;

    HaloCapacityInfo info;
    // the widest the build can ask for: start width plus every widening round
    info.W_alloc    = halo_default_width(buff) + W_STARTUP_MARGIN + 2 * (MAX_WIDEN_ITERS - 1);
    info.Lx         = decomp.b1[0] - decomp.b0[0];
    info.Ly         = decomp.b1[1] - decomp.b0[1];
    info.Lz         = decomp.b1[2] - decomp.b0[2];
    info.geom_cells = geom_total_cells(info.Lx, info.Ly, info.Lz, info.W_alloc);

    // a thin rank still uses the global mean density
    const long long n_local_ll = (long long)n_local;
    long long       n_global   = 0;
    MPI_Allreduce(&n_local_ll, &n_global, 1, MPI_LONG_LONG, MPI_SUM, decomp.cart_comm);
    const long long N_grid_dim    = (long long)decomp.N_grid_global;
    long long       global_volume = N_grid_dim * N_grid_dim;
#ifdef dim_3D
    global_volume *= N_grid_dim;
#endif
    const double rho_global_mean = (global_volume > 0) ? (double)n_global / (double)global_volume : 1.0;

    const long long total_buckets = (long long)info.Lx * (long long)info.Ly * (long long)info.Lz;
    const double    rho_local     = (total_buckets > 0) ? (double)n_local / (double)total_buckets : 1.0;
    info.rho_cells                = std::max(rho_local, rho_global_mean);

    const long long est       = (long long)std::ceil(SAFETY * info.rho_cells * (double)info.geom_cells);
    const long long floor_cap = 1024;
    info.n_capacity           = (int)std::max<long long>(floor_cap, est);
    return info;
}

static void free_halo_buffers() {
    if (halo.export_indices) gpu_free(halo.export_indices);
    if (halo.dir_of_slot) gpu_free(halo.dir_of_slot);
    if (halo.used_export_indices) gpu_free(halo.used_export_indices);
    if (halo.used_to_full_slot) gpu_free(halo.used_to_full_slot);
    if (halo.send_used_bitmap) gpu_free(halo.send_used_bitmap);
    if (halo.recv_used_bitmap) gpu_free(halo.recv_used_bitmap);
    if (halo.sendbuf_seed) gpu_free(halo.sendbuf_seed);
    if (halo.recvbuf_seed) gpu_free(halo.recvbuf_seed);
    if (halo.sendbuf_prim) gpu_free(halo.sendbuf_prim);
    if (halo.recvbuf_prim) gpu_free(halo.recvbuf_prim);
    if (halo.sendbuf_v_mesh) gpu_free(halo.sendbuf_v_mesh);
    if (halo.recvbuf_v_mesh) gpu_free(halo.recvbuf_v_mesh);
    if (halo.sendbuf_grad) gpu_free(halo.sendbuf_grad);
    if (halo.recvbuf_grad) gpu_free(halo.recvbuf_grad);
#ifdef VOL_REGULARIZE
    if (halo.sendbuf_vol) gpu_free(halo.sendbuf_vol);
    if (halo.recvbuf_vol) gpu_free(halo.recvbuf_vol);
#endif
    if (halo.is_outer_layer) gpu_free(halo.is_outer_layer);
}

// every buffer holds one entry per slot
static void allocate_halo_buffers(int n_capacity) {
    halo.export_indices      = (int*)gpu_malloc(sizeof(int) * n_capacity);
    halo.dir_of_slot         = (unsigned char*)gpu_malloc(sizeof(unsigned char) * n_capacity);
    halo.used_export_indices = (int*)gpu_malloc(sizeof(int) * n_capacity);
    halo.used_to_full_slot   = (int*)gpu_malloc(sizeof(int) * n_capacity);
    halo.send_used_bitmap    = (unsigned char*)gpu_malloc(sizeof(unsigned char) * n_capacity);
    halo.recv_used_bitmap    = (unsigned char*)gpu_malloc(sizeof(unsigned char) * n_capacity);

    halo.sendbuf_seed   = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity);
    halo.recvbuf_seed   = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity);
    halo.sendbuf_prim   = (HaloPrimCell*)gpu_malloc(sizeof(HaloPrimCell) * n_capacity);
    halo.recvbuf_prim   = (HaloPrimCell*)gpu_malloc(sizeof(HaloPrimCell) * n_capacity);
    halo.sendbuf_v_mesh = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity);
    halo.recvbuf_v_mesh = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity);

    const int grad_components = 3 + DIMENSION;
    halo.sendbuf_grad         = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity * grad_components);
    halo.recvbuf_grad         = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * n_capacity * grad_components);
#ifdef VOL_REGULARIZE
    halo.sendbuf_vol = (double*)gpu_malloc(sizeof(double) * n_capacity);
    halo.recvbuf_vol = (double*)gpu_malloc(sizeof(double) * n_capacity);
#endif

    halo.is_outer_layer = (unsigned char*)gpu_malloc(sizeof(unsigned char) * n_capacity);
}

// the kernels cannot read the 2D array of the struct
static void sync_neighbor_shift_to_flat() {
    if (halo.neighbor_shift_flat == nullptr) return;
    for (int n = 0; n < HALO_MAX_NEIGHBORS; n++) {
        halo.neighbor_shift_flat[n * 3 + 0] = halo.neighbor_shift[n][0];
        halo.neighbor_shift_flat[n * 3 + 1] = halo.neighbor_shift[n][1];
        halo.neighbor_shift_flat[n * 3 + 2] = halo.neighbor_shift[n][2];
    }
}

// the payloads travel as plain bytes
static void register_mpi_datatypes() {
    const int grad_components = 3 + DIMENSION;
    MPI_Type_contiguous(sizeof(HaloPrimCell), MPI_BYTE, &halo.mpi_prim_t);
    MPI_Type_commit(&halo.mpi_prim_t);
    MPI_Type_contiguous(sizeof(POINT_TYPE), MPI_BYTE, &halo.mpi_point_t);
    MPI_Type_commit(&halo.mpi_point_t);
    MPI_Type_contiguous(grad_components * (int)sizeof(POINT_TYPE), MPI_BYTE, &halo.mpi_grad_cell_t);
    MPI_Type_commit(&halo.mpi_grad_cell_t);
}

#endif
