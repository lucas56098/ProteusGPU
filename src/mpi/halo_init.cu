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
static void             sync_neighbor_shift_to_flat();
static void             register_mpi_datatypes();
#endif

// ============================================================
// Public entry points
// ============================================================

void halo_init(int n_local, double buff) {
#ifndef USE_MPI
    (void)n_local;
    (void)buff;
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
    halo.n_mpi_capacity         = info.n_capacity;
    n_mpi_capacity              = info.n_capacity;

    allocate_halo_buffers(info.n_capacity);

    // Topology metadata mirror: small fixed-size managed buffer that doesn't grow with capacity.
    // Allocated here (once) and freed in halo_free.
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
                                       halo.n_neighbors,
                                       halo.neighbor_ranks,
                                       MPI_UNWEIGHTED,
                                       halo.n_neighbors,
                                       halo.neighbor_ranks,
                                       MPI_UNWEIGHTED,
                                       info,
                                       /*reorder=*/0,
                                       &halo.graph_comm);
        MPI_Info_free(&info);
    } else {
        halo.graph_comm = MPI_COMM_NULL;
    }
}

// halo capacity sizing — uses the global mean density invariant:
//   after any cost-balanced rebalance, every rank's local cell density tends to the global
//   mean = n_global / N_grid^DIM, because each rank gets ~n_global/nranks cells in a brick of
//   volume ~N_grid^DIM/nranks. Sizing from this invariant means each rank's halo can
//   accommodate the densest possible neighbouring rank after any rebalance, with one safety
//   factor for spread around the mean. n_mpi_capacity may still grow at runtime via
//   halo_grow_capacity if reality exceeds the estimate (e.g., highly anisotropic density).
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

    // global mean density: n_global / N_grid^DIM. Independent of which rank or its brick shape.
    // long long because the global sum is n_global (few_thousand^3), past int32 from ~1300^3 up.
    const long long n_local_ll = (long long)n_local;
    long long       n_global   = 0;
    MPI_Allreduce(&n_local_ll, &n_global, 1, MPI_LONG_LONG, MPI_SUM, decomp.cart_comm);
    const long long N_grid_dim    = (long long)decomp.N_grid_global;
    long long       global_volume = N_grid_dim * N_grid_dim;
#ifdef dim_3D
    global_volume *= N_grid_dim;
#endif
    const double rho_global_mean = (global_volume > 0) ? (double)n_global / (double)global_volume : 1.0;

    // local rho is what we'd use if no rebalance were ever to fire; global rho is the
    // post-rebalance ceiling. take whichever is larger so static-decomp runs don't pay
    // the global penalty unless they explicitly enable rebalance.
    const long long total_buckets = (long long)info.Lx * (long long)info.Ly * (long long)info.Lz;
    const double    rho_local     = (total_buckets > 0) ? (double)n_local / (double)total_buckets : 1.0;
    info.rho_cells                = std::max(rho_local, rho_global_mean);

    const long long est       = (long long)std::ceil(SAFETY * info.rho_cells * (double)info.geom_cells);
    const long long floor_cap = 1024;
    info.n_capacity           = (int)std::max<long long>(floor_cap, est);
    return info;
}

// free every halo struct buffer; safe to call on a partially-initialized halo (nullptr-tolerant).
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

// after build_neighbor_table has populated halo.neighbor_shift, mirror it into
// the managed flat array so kernels can read it device-side.
static void sync_neighbor_shift_to_flat() {
    if (halo.neighbor_shift_flat == nullptr) return;
    for (int n = 0; n < HALO_MAX_NEIGHBORS; n++) {
        halo.neighbor_shift_flat[n * 3 + 0] = halo.neighbor_shift[n][0];
        halo.neighbor_shift_flat[n * 3 + 1] = halo.neighbor_shift[n][1];
        halo.neighbor_shift_flat[n * 3 + 2] = halo.neighbor_shift[n][2];
    }
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
