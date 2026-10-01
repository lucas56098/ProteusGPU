// halo buffers: setup, growth, teardown (included by halo.cu)

static void free_halo_buffers();
static void allocate_recv_buffers(int capacity);
static void allocate_send_buffers(int capacity);

// a guess of a few cell layers around the domain; the buffers grow when a build needs more
void halo_init(int n_local) {
    halo         = MpiHalo();
    int capacity = 0;
#ifdef USE_MPI
    if (decomp.nranks > 1) {
        const double n       = (double)std::max(n_local, 1);
        const double surface = (DIMENSION == 3) ? 6.0 * std::pow(n, 2.0 / 3.0) : 4.0 * std::sqrt(n);
        capacity             = std::max(1024, (int)(4.0 * surface));
    }
#else
    (void)n_local;
#endif
    n_mpi_capacity = capacity;
    allocate_recv_buffers(capacity);
    allocate_send_buffers(capacity);
}

void halo_free() {
    free_halo_buffers();
    halo           = MpiHalo();
    n_mpi_capacity = 0;
}

static void free_halo_buffers() {
    if (halo.used_to_full_slot) gpu_free(halo.used_to_full_slot);
    if (halo.recvbuf_prim) gpu_free(halo.recvbuf_prim);
    if (halo.recvbuf_point) gpu_free(halo.recvbuf_point);
    if (halo.recvbuf_grad) gpu_free(halo.recvbuf_grad);
    if (halo.recvbuf_double) gpu_free(halo.recvbuf_double);
    if (halo.used_export_indices) gpu_free(halo.used_export_indices);
    if (halo.sendbuf_prim) gpu_free(halo.sendbuf_prim);
    if (halo.sendbuf_point) gpu_free(halo.sendbuf_point);
    if (halo.sendbuf_grad) gpu_free(halo.sendbuf_grad);
    if (halo.sendbuf_double) gpu_free(halo.sendbuf_double);
    halo.used_to_full_slot   = nullptr;
    halo.recvbuf_prim        = nullptr;
    halo.recvbuf_point       = nullptr;
    halo.recvbuf_grad        = nullptr;
    halo.recvbuf_double      = nullptr;
    halo.used_export_indices = nullptr;
    halo.sendbuf_prim        = nullptr;
    halo.sendbuf_point       = nullptr;
    halo.sendbuf_grad        = nullptr;
    halo.sendbuf_double      = nullptr;
}

// one entry per ghost slot
static void allocate_recv_buffers(int capacity) {
    if (capacity <= 0) return;
    halo.used_to_full_slot = gpu_alloc<int>(capacity);
    halo.recvbuf_prim      = gpu_alloc<HaloPrimCell>(capacity);
    halo.recvbuf_point     = gpu_alloc<POINT_TYPE>(capacity);
    halo.recvbuf_grad      = gpu_alloc<POINT_TYPE>((size_t)capacity * HALO_GRAD_COMPONENTS);
    halo.recvbuf_double    = gpu_alloc<double>(capacity);
}

// one entry per cell sent; a cell can go to several ranks
static void allocate_send_buffers(int capacity) {
    halo.send_capacity = capacity;
    if (capacity <= 0) return;
    halo.used_export_indices = gpu_alloc<int>(capacity);
    halo.sendbuf_prim        = gpu_alloc<HaloPrimCell>(capacity);
    halo.sendbuf_point       = gpu_alloc<POINT_TYPE>(capacity);
    halo.sendbuf_grad        = gpu_alloc<POINT_TYPE>((size_t)capacity * HALO_GRAD_COMPONENTS);
    halo.sendbuf_double      = gpu_alloc<double>(capacity);
}

// the send side grows on its own, nothing else is sized by it
static void ensure_send_capacity(int needed) {
    if (needed <= halo.send_capacity) return;
    const int target = std::max(needed, 2 * halo.send_capacity);
    if (halo.used_export_indices) gpu_free(halo.used_export_indices);
    if (halo.sendbuf_prim) gpu_free(halo.sendbuf_prim);
    if (halo.sendbuf_point) gpu_free(halo.sendbuf_point);
    if (halo.sendbuf_grad) gpu_free(halo.sendbuf_grad);
    if (halo.sendbuf_double) gpu_free(halo.sendbuf_double);
    allocate_send_buffers(target);
}

// more slots: at least double, and take everything that is sized by them along
void halo_grow_capacity(int new_capacity) {
    const int old_cap = n_mpi_capacity;
    const int target  = std::max(new_capacity, std::max(1024, 2 * old_cap));

    if (halo.used_to_full_slot) gpu_free(halo.used_to_full_slot);
    if (halo.recvbuf_prim) gpu_free(halo.recvbuf_prim);
    if (halo.recvbuf_point) gpu_free(halo.recvbuf_point);
    if (halo.recvbuf_grad) gpu_free(halo.recvbuf_grad);
    if (halo.recvbuf_double) gpu_free(halo.recvbuf_double);
    allocate_recv_buffers(target);
    n_mpi_capacity = target;

    if (sim.mesh) voronoi::mesh_grow_ghosts(sim.mesh, target);
    if (sim.primvar) hydro::primvar_grow_ghosts(sim.primvar, target);
    if (sim.grads) gradients::grad_grow_ghosts(sim.grads, target);

    printf("HALO: rank %d grew its ghost slots %d -> %d.\n", decomp.rank, old_cap, target);
    fflush(stdout);
}
