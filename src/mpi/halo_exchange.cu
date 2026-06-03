// Per-step halo data exchanges and dt allreduce.
// Included into halo.cu inside namespace proteus_mpi.

// ============================================================
// Public entry points
// ============================================================

void halo_exchange_seeds(VMesh* mesh, POINT_TYPE* pts, int pts_mpi_base) {
#ifndef USE_MPI
    (void)mesh; (void)pts; (void)pts_mpi_base;
    return;
#else
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;

    Profiler::StartTimer("MPI_PACK");
    const int n_hydro    = (int)mesh->n_hydro;
    const int total_send = halo.send_offset[halo.n_neighbors];
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < total_send; s++) {
        const int    n  = halo.dir_of_slot[s];
        const int    k  = halo.export_indices[s];
        const double sx = halo.neighbor_shift[n][0];
        const double sy = halo.neighbor_shift[n][1];
        POINT_TYPE   p  = pts[k];
        p.x += sx;
        p.y += sy;
#ifdef dim_3D
        p.z += halo.neighbor_shift[n][2];
#endif
        halo.sendbuf_seed[s] = p;
    }
    Profiler::EndTimer("MPI_PACK");

    Profiler::StartTimer("MPI_WAIT");
    exchange_full_halo(halo.sendbuf_seed, halo.recvbuf_seed, halo.mpi_point_t, MSG_SEED);
    Profiler::EndTimer("MPI_WAIT");

    Profiler::StartTimer("MPI_UNPACK");
    const int nn    = halo.n_neighbors;
    const int n_mpi = halo.n_mpi_ghosts;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int slot = 0; slot < n_mpi; slot++) {
        const POINT_TYPE p     = halo.recvbuf_seed[slot];
        const int        ext_k = n_hydro + slot;
        const int        pts_k = pts_mpi_base + slot;
        pts[pts_k] = p;
#ifdef dim_3D
        mesh->seeds[ext_k] = double3{p.x, p.y, p.z};
#else
        mesh->seeds[ext_k] = double3{p.x, p.y, 0.0};
#endif
    }

    // derive is_outer_layer positionally: the first recv_n_outer[n] slots of
    // each direction's receive range are the outermost-layer cells
    for (int n = 0; n < nn; n++) {
        const int base  = halo.ghost_offset[n];
        const int n_out = halo.recv_n_outer[n];
        const int n_tot = halo.recv_count[n];
        for (int j = 0; j < n_out; j++)        halo.is_outer_layer[base + j] = 1;
        for (int j = n_out; j < n_tot; j++)    halo.is_outer_layer[base + j] = 0;
    }
    Profiler::EndTimer("MPI_UNPACK");
#endif
}

void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar) {
#ifndef USE_MPI
    (void)mesh; (void)primvar;
    return;
#else
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;
    if (!halo.used_subset_ready) return;  // nothing to do until mesh exists

    Profiler::StartTimer("MPI_PACK");
    const int total_send = halo.n_used_send;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < total_send; s++) {
        const int    k = halo.used_export_indices[s];
        HaloPrimCell pkt;
        pkt.rho = primvar->rho[k];
        pkt.v   = primvar->v[k];
        pkt.E   = primvar->E[k];
        halo.sendbuf_prim[s] = pkt;
    }
    Profiler::EndTimer("MPI_PACK");

    Profiler::StartTimer("MPI_WAIT");
    exchange_used_subset(halo.sendbuf_prim, halo.recvbuf_prim, halo.mpi_prim_t, MSG_PRIM);
    Profiler::EndTimer("MPI_WAIT");

    Profiler::StartTimer("MPI_UNPACK");
    const int n_hydro = (int)mesh->n_hydro;
    const int n_recv  = halo.n_used_recv;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < n_recv; s++) {
        const HaloPrimCell pkt   = halo.recvbuf_prim[s];
        const int          ext_k = n_hydro + halo.used_to_full_slot[s];
        primvar->rho[ext_k] = pkt.rho;
        primvar->v[ext_k]   = pkt.v;
        primvar->E[ext_k]   = pkt.E;
    }
    Profiler::EndTimer("MPI_UNPACK");
#endif
}

void halo_exchange_gradients(VMesh* mesh, gradients::PrimGradients* grads) {
#ifndef USE_MPI
    (void)mesh; (void)grads;
    return;
#else
    if (halo.n_neighbors == 0 || halo.n_mpi_ghosts == 0) return;
    if (!halo.used_subset_ready) return;
    Profiler::StartTimer("MPI_PACK");
    const int N_COMP     = 3 + DIMENSION;
    const int total_send = halo.n_used_send;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int slot = 0; slot < total_send; slot++) {
        const int k = halo.used_export_indices[slot];
        const int s = slot * N_COMP;
        int       c = 0;
        halo.sendbuf_grad[s + c++] = grads->rho[k];
        halo.sendbuf_grad[s + c++] = grads->vx[k];
        halo.sendbuf_grad[s + c++] = grads->vy[k];
#ifdef dim_3D
        halo.sendbuf_grad[s + c++] = grads->vz[k];
#endif
        halo.sendbuf_grad[s + c++] = grads->E[k];
    }
    Profiler::EndTimer("MPI_PACK");

    Profiler::StartTimer("MPI_WAIT");
    exchange_used_subset(halo.sendbuf_grad, halo.recvbuf_grad, halo.mpi_grad_cell_t, MSG_GRAD);
    Profiler::EndTimer("MPI_WAIT");

    Profiler::StartTimer("MPI_UNPACK");
    const int n_hydro = (int)mesh->n_hydro;
    const int n_recv  = halo.n_used_recv;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int slot = 0; slot < n_recv; slot++) {
        const int ext_k = n_hydro + halo.used_to_full_slot[slot];
        const int s     = slot * N_COMP;
        int       c     = 0;
        grads->rho[ext_k] = halo.recvbuf_grad[s + c++];
        grads->vx[ext_k]  = halo.recvbuf_grad[s + c++];
        grads->vy[ext_k]  = halo.recvbuf_grad[s + c++];
#ifdef dim_3D
        grads->vz[ext_k] = halo.recvbuf_grad[s + c++];
#endif
        grads->E[ext_k] = halo.recvbuf_grad[s + c++];
    }
    Profiler::EndTimer("MPI_UNPACK");
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
    Profiler::StartTimer("MPI_PACK");
    const int total_send = halo.n_used_send;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < total_send; s++) {
        const int k              = halo.used_export_indices[s];
        halo.sendbuf_v_mesh[s] = mesh->v_mesh[k];
    }
    Profiler::EndTimer("MPI_PACK");

    Profiler::StartTimer("MPI_WAIT");
    exchange_used_subset(halo.sendbuf_v_mesh, halo.recvbuf_v_mesh, halo.mpi_point_t, MSG_V_MESH);
    Profiler::EndTimer("MPI_WAIT");

    Profiler::StartTimer("MPI_UNPACK");
    const int n_hydro = (int)mesh->n_hydro;
    const int n_recv  = halo.n_used_recv;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int slot = 0; slot < n_recv; slot++) {
        const int ext_k     = n_hydro + halo.used_to_full_slot[slot];
        mesh->v_mesh[ext_k] = halo.recvbuf_v_mesh[slot];
    }
    Profiler::EndTimer("MPI_UNPACK");
#else
    (void)mesh;
#endif
#endif
}

void halo_dt_allreduce(double* dt) {
#ifdef USE_MPI
    Profiler::StartTimer("MPI_REDUCE");
    double local = *dt;
    MPI_Allreduce(&local, dt, 1, MPI_DOUBLE, MPI_MIN, decomp.cart_comm);
    Profiler::EndTimer("MPI_REDUCE");
#else
    (void)dt;
#endif
}
