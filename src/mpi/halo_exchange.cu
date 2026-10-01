// the used subset and the state exchanges with the partners of the build (included by halo.cu)

static unsigned char* s_used_bitmap     = nullptr; // per ghost slot, whether a local cell has it as a face neighbour
static int            s_used_bitmap_cap = 0;

static size_t index_of_partner(const std::vector<int>& list, int r) {
    return std::lower_bound(list.begin(), list.end(), r) - list.begin();
}

// marks the slots behind a face, then tells every owner which of its cells are used here
void halo_build_used_subset(VMesh* mesh) {
    halo.used_subset_ready = 1;
    halo.n_used_send       = 0;
    halo.n_used_recv       = 0;
    halo.used_send_count.assign(halo.askers.size(), 0);
    halo.used_recv_count.assign(halo.asked.size(), 0);
    if (halo.asked.empty() && halo.askers.empty()) return;

    PROFILE("HALO_USED_BUILD");
    const int n_hydro = (int)mesh->n_hydro;
    const int n_mpi   = halo.n_mpi_ghosts;

    if (n_mpi > s_used_bitmap_cap) {
        if (s_used_bitmap) gpu_free(s_used_bitmap);
        s_used_bitmap_cap = std::max(n_mpi, 2 * s_used_bitmap_cap);
        s_used_bitmap     = gpu_alloc<unsigned char>(s_used_bitmap_cap);
    }
    if (n_mpi > 0) {
        gpu_memset(s_used_bitmap, 0, (size_t)n_mpi);
        unsigned char* used = s_used_bitmap;
        const int*     nc   = mesh->neighbor_cell;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("BITMAP_MARK", mesh->num_faces, [=] HD(int f) {
            pack::mark_used_bitmap_body(f, nc, n_hydro, n_hydro + n_mpi, used);
        });
        GPU_SYNC();
    }

    // per owner the used slots in slot order, and the cells behind them
    std::vector<std::vector<int>> slots(halo.asked.size()), cells(halo.asked.size());
    for (size_t g = 0; g < halo.g_owner.size(); g++) {
        const int slot = halo.g_slot[g];
        if (slot < 0 || !s_used_bitmap[slot]) continue;
        const size_t i = index_of_partner(halo.asked, halo.g_owner[g]);
        slots[i].push_back(slot);
        cells[i].push_back(halo.g_k[g]);
    }
    Messages out;
    for (size_t i = 0; i < halo.asked.size(); i++) {
        std::vector<char>& msg = out.to(halo.asked[i]);
        for (int k : cells[i])
            append(msg, k);
        for (int slot : slots[i])
            halo.used_to_full_slot[halo.n_used_recv++] = slot;
        halo.used_recv_count[i] = (int)slots[i].size();
    }

    Messages in;
    partner_exchange(out, halo.askers, &in);

    int total = 0;
    for (size_t m = 0; m < in.ranks.size(); m++)
        total += (int)count_of<int>(in.data[m]);
    ensure_send_capacity(total);
    for (size_t m = 0; m < in.ranks.size(); m++) {
        const size_t i  = index_of_partner(halo.askers, in.ranks[m]);
        const size_t n  = count_of<int>(in.data[m]);
        const int*   ks = items_of<int>(in.data[m]);
        for (size_t j = 0; j < n; j++)
            halo.used_export_indices[halo.n_used_send++] = ks[j];
        halo.used_send_count[i] = (int)n;
    }
}

#ifdef USE_MPI
// the used cells to the askers, the used ghosts from the asked; both sides know the counts
static void exchange_used(const void* sendbuf, void* recvbuf, size_t item_bytes, int tag) {
    mpi_sync_before_send(sendbuf, item_bytes * (size_t)halo.n_used_send);
    std::vector<MPI_Request> reqs;
    const char*              s   = (const char*)sendbuf;
    char*                    r   = (char*)recvbuf;
    size_t                   off = 0;
    for (size_t i = 0; i < halo.askers.size(); i++) {
        const int n = halo.used_send_count[i];
        if (n > 0) {
            reqs.emplace_back();
            MPI_Isend(
                s + off * item_bytes, (int)(n * item_bytes), MPI_BYTE, halo.askers[i], tag, decomp.comm, &reqs.back());
        }
        off += (size_t)n;
    }
    off = 0;
    for (size_t j = 0; j < halo.asked.size(); j++) {
        const int n = halo.used_recv_count[j];
        if (n > 0) {
            reqs.emplace_back();
            MPI_Irecv(
                r + off * item_bytes, (int)(n * item_bytes), MPI_BYTE, halo.asked[j], tag, decomp.comm, &reqs.back());
        }
        off += (size_t)n;
    }
    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    mpi_sync_after_recv(recvbuf, item_bytes * (size_t)halo.n_used_recv);
}

static bool have_partners() {
    return halo.used_subset_ready && !(halo.asked.empty() && halo.askers.empty());
}

enum HaloTag { TAG_PRIM = 6001, TAG_GRAD, TAG_V_MESH, TAG_COM_OFF, TAG_VOL };
#endif

// state of the used ghosts
void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar) {
    (void)mesh;
#ifndef USE_MPI
    (void)primvar;
#else
    if (!have_partners()) return;
    PROFILE("HALO_PRIM");
    auto* sendbuf = halo.sendbuf_prim;
    auto* exports = halo.used_export_indices;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "PACK", halo.n_used_send, [=] HD(int s) { pack::pack_prim_body(s, exports, primvar, sendbuf); });
    {
        PROFILE_MPI("WAIT");
        exchange_used(halo.sendbuf_prim, halo.recvbuf_prim, sizeof(HaloPrimCell), TAG_PRIM);
    }
    auto* recvbuf = halo.recvbuf_prim;
    auto* slots   = halo.used_to_full_slot;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "UNPACK", halo.n_used_recv, [=] HD(int s) { pack::unpack_prim_body(s, slots, recvbuf, primvar); });
#endif
}

// their gradients, all components in one message
void halo_exchange_gradients(VMesh* mesh, gradients::PrimGradients* grads) {
    (void)mesh;
#ifndef USE_MPI
    (void)grads;
#else
    if (!have_partners()) return;
    PROFILE("HALO_GRAD");
    auto* sendbuf = halo.sendbuf_grad;
    auto* exports = halo.used_export_indices;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "PACK", halo.n_used_send, [=] HD(int s) { pack::pack_grad_body(s, exports, grads, sendbuf); });
    {
        PROFILE_MPI("WAIT");
        exchange_used(halo.sendbuf_grad, halo.recvbuf_grad, sizeof(POINT_TYPE) * HALO_GRAD_COMPONENTS, TAG_GRAD);
    }
    auto* recvbuf = halo.recvbuf_grad;
    auto* slots   = halo.used_to_full_slot;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "UNPACK", halo.n_used_recv, [=] HD(int s) { pack::unpack_grad_body(s, slots, recvbuf, grads); });
#endif
}

// and their mesh velocity
void halo_exchange_v_mesh(VMesh* mesh) {
#if !defined(USE_MPI) || !defined(MOVING_MESH)
    (void)mesh;
#else
    if (!have_partners()) return;
    PROFILE("HALO_VMESH");
    auto* sendbuf = halo.sendbuf_point;
    auto* exports = halo.used_export_indices;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "PACK", halo.n_used_send, [=] HD(int s) { pack::pack_v_mesh_body(s, exports, mesh->v_mesh, sendbuf); });
    {
        PROFILE_MPI("WAIT");
        exchange_used(halo.sendbuf_point, halo.recvbuf_point, sizeof(POINT_TYPE), TAG_V_MESH);
    }
    auto* recvbuf = halo.recvbuf_point;
    auto* slots   = halo.used_to_full_slot;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "UNPACK", halo.n_used_recv, [=] HD(int s) { pack::unpack_v_mesh_body(s, slots, recvbuf, mesh->v_mesh_g); });
#endif
}

// where the centroid of each used ghost sits, relative to its seed
void halo_exchange_centroids(VMesh* mesh) {
#ifndef USE_MPI
    (void)mesh;
#else
    if (!have_partners()) return;
    PROFILE("HALO_COM");
    auto* sendbuf = halo.sendbuf_point;
    auto* exports = halo.used_export_indices;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("PACK", halo.n_used_send, [=] HD(int s) {
        pack::pack_com_off_body(s, exports, mesh->com, mesh->seeds, sendbuf);
    });
    {
        PROFILE_MPI("WAIT");
        exchange_used(halo.sendbuf_point, halo.recvbuf_point, sizeof(POINT_TYPE), TAG_COM_OFF);
    }
    auto* recvbuf = halo.recvbuf_point;
    auto* slots   = halo.used_to_full_slot;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "UNPACK", halo.n_used_recv, [=] HD(int s) { pack::unpack_com_off_body(s, slots, recvbuf, mesh->com_off_g); });
#endif
}

#ifdef VOL_REGULARIZE
// cell volumes of the used ghosts
void halo_exchange_volumes(VMesh* mesh) {
#ifndef USE_MPI
    (void)mesh;
#else
    if (!have_partners()) return;
    PROFILE("HALO_VOL");
    auto* sendbuf = halo.sendbuf_double;
    auto* exports = halo.used_export_indices;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "PACK", halo.n_used_send, [=] HD(int s) { pack::pack_vol_body(s, exports, mesh->volumes, sendbuf); });
    {
        PROFILE_MPI("WAIT");
        exchange_used(halo.sendbuf_double, halo.recvbuf_double, sizeof(double), TAG_VOL);
    }
    auto* recvbuf = halo.recvbuf_double;
    auto* slots   = halo.used_to_full_slot;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "UNPACK", halo.n_used_recv, [=] HD(int s) { pack::unpack_vol_body(s, slots, recvbuf, mesh->volumes_g); });
#endif
}
#endif

// cells another rank holds as a ghost, counted once per ghost
int halo_count_moved_exports(const std::vector<int>& moved_ks) {
    if (moved_ks.empty() || halo.askers.empty()) return 0;
    const std::unordered_set<int> moved(moved_ks.begin(), moved_ks.end());
    int                           n = 0;
    for (const auto& sent : halo.sent) {
        for (uint64_t key : sent)
            if (moved.count((int)(key >> 5))) n++;
    }
    return n;
}

// every asker gets the new position of the cells it holds, every owner sends ours
void halo_exchange_moved_seeds(const VMesh* mesh, const std::vector<int>& moved_ks, std::vector<MovedSeed>* received) {
    received->clear();
    const std::unordered_set<int> moved(moved_ks.begin(), moved_ks.end());

    Messages out;
    for (size_t i = 0; i < halo.askers.size(); i++) {
        std::vector<char>& msg = out.to(halo.askers[i]);
        for (uint64_t key : halo.sent[i]) {
            const int k = (int)(key >> 5);
            if (!moved.count(k)) continue;
            GhostAnswer ans;
            ans.k     = k;
            ans.shift = (int)(key & 31u);
            double s[3];
            shift_of_code(ans.shift, s);
            ans.p.x = mesh->seeds[k].x + s[0];
            ans.p.y = mesh->seeds[k].y + s[1];
#ifdef dim_3D
            ans.p.z = mesh->seeds[k].z + s[2];
#endif
            append(msg, ans);
        }
    }

    Messages in;
    partner_exchange(out, halo.asked, &in);

    for (size_t m = 0; m < in.ranks.size(); m++) {
        const int          owner = in.ranks[m];
        const size_t       n     = count_of<GhostAnswer>(in.data[m]);
        const GhostAnswer* as    = items_of<GhostAnswer>(in.data[m]);
        for (size_t i = 0; i < n; i++) {
            const auto it = halo.g_index.find(ghost_key(owner, as[i].k, as[i].shift));
            if (it == halo.g_index.end() || halo.g_slot[it->second] < 0) {
                exit_failure("HALO: rank %d moved cell %d, which rank %d does not hold as a ghost\n",
                             owner,
                             as[i].k,
                             decomp.rank);
            }
            halo.g_pos[it->second] = as[i].p;
            MovedSeed ms;
            ms.pos        = as[i].p;
            ms.ghost_slot = halo.g_slot[it->second];
            received->push_back(ms);
        }
    }
}

// the smallest timestep of all ranks
void halo_dt_allreduce(double* dt) {
#ifdef USE_MPI
    PROFILE_MPI("DT_ALLREDUCE");
    double local = *dt;
    MPI_Allreduce(&local, dt, 1, MPI_DOUBLE, MPI_MIN, decomp.comm);
#else
    (void)dt;
#endif
}

// one number summed over all ranks
void halo_sum_allreduce(double* v) {
#ifdef USE_MPI
    PROFILE_MPI("SUM_ALLREDUCE");
    double local = *v;
    MPI_Allreduce(&local, v, 1, MPI_DOUBLE, MPI_SUM, decomp.comm);
#else
    (void)v;
#endif
}
