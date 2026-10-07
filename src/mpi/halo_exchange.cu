// the used subset and the state exchanges with the partners of the build (included by halo.cu)

// exclusive scan of the n flags into pos; returns how many are set
static size_t scan_flags(size_t n) {
    if (n == 0) return 0;
    unsigned int*      flag = s_buffers.flag.data;
    unsigned int*      pos  = s_buffers.pos.data;
    const unsigned int last = flag[n - 1];
    parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_>("FLAG_SCAN", n, flag, pos, s_buffers.scan_scratch(n));
    return (size_t)pos[n - 1] + last;
}

// marks the slots behind a face, then tells every owner which of its cells are used here
void halo_build_used_subset(VMesh* mesh) {
    halo.used_subset_ready = 1;
    halo.state_send.clear();
    halo.state_recv.clear();
    if (halo.asked.empty() && halo.askers.empty()) return;

    PROFILE("HALO_USED_BUILD");
    const int      n_hydro = (int)mesh->n_hydro;
    const int      n_mpi   = halo.n_mpi_ghosts;
    unsigned char* used    = s_buffers.used_bitmap.fit((size_t)n_mpi);
    if (n_mpi > 0) {
        gpu_memset(used, 0, (size_t)n_mpi);
        const int* nc = mesh->neighbor_cell;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("BITMAP_MARK", mesh->num_faces, [=] HD(size_t f) {
            const int kn = nc[f];
            if (kn >= n_hydro && kn < n_hydro + n_mpi) used[kn - n_hydro] = 1;
        });
    }

    // the used ghosts in ghost order, then grouped by owner; within an owner they stay in slot order
    const size_t       n_g     = (size_t)halo.n_ghosts;
    const int*         g_slot  = halo.g_slot.data;
    const int*         g_owner = halo.g_owner.data;
    const GhostAnswer* g_ans   = halo.g_ans.data;
    unsigned int*      flag    = s_buffers.flag.fit(n_g);
    unsigned int*      pos     = s_buffers.pos.fit(n_g);
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("USED_FLAG", n_g, [=] HD(size_t g) {
        const int s = g_slot[g];
        flag[g]     = (s >= 0 && used[s]) ? 1u : 0u;
    });
    const size_t n_used = scan_flags(n_g);
    PairSort     sort   = s_buffers.sort.fit(n_used);
    {
        uint64_t*     keys = sort.keys;
        unsigned int* vals = sort.vals;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("USED_COMPACT", n_g, [=] HD(size_t g) {
            if (!flag[g]) return;
            keys[pos[g]] = (uint64_t)g_owner[g];
            vals[pos[g]] = (unsigned int)g;
        });
    }
    sort.sort("USED_SORT", n_used, rank_key_bits());

    halo.state_recv = with_partners(blocks_of_sorted_ranks(sort.keys, n_used, &s_buffers.run_scratch), halo.asked);
    const unsigned int* ghosts       = sort.vals;
    int*                cells        = s_buffers.cells.fit(n_used);
    int*                to_full_slot = halo.used_to_full_slot.fit(n_used);
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("USED_GATHER", n_used, [=] HD(size_t j) {
        const unsigned int g = ghosts[j];
        cells[j]             = g_ans[g].k;
        to_full_slot[j]      = g_slot[g];
    });

    // every owner learns which of its cells are used here, in that order
    partner_counts(halo.state_recv, halo.askers, &halo.state_send);
    exchange_items(
        cells, halo.state_recv, halo.used_export_indices.fit(halo.state_send.total), halo.state_send, sizeof(int));
}

#ifdef USE_MPI
static bool have_partners() {
    return halo.used_subset_ready && !(halo.asked.empty() && halo.askers.empty());
}

// one used cell's state to the askers, one used ghost's state from the asked; Item is what one cell sends,
// pack(k) makes it from cell k, unpack(slot, item) puts it into the ghost slot
template <typename Item, typename Pack, typename Unpack> static void exchange_state(Pack pack, Unpack unpack) {
    const size_t n_send  = halo.state_send.total;
    const size_t n_recv  = halo.state_recv.total;
    Item*        send    = (Item*)halo.sendbuf.fit(n_send * sizeof(Item));
    Item*        recv    = (Item*)halo.recvbuf.fit(n_recv * sizeof(Item));
    const int*   exports = halo.used_export_indices.data;
    const int*   slots   = halo.used_to_full_slot.data;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("PACK", n_send, [=] HD(size_t s) { send[s] = pack(exports[s]); });
    {
        PROFILE_MPI("WAIT");
        exchange_items(send, halo.state_send, recv, halo.state_recv, sizeof(Item));
    }
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("UNPACK", n_recv, [=] HD(size_t s) { unpack(slots[s], recv[s]); });
}

// one cell's primvars on the wire
struct HaloPrimCell {
    double     rho;
    POINT_TYPE v;
    double     E;
};
#endif

// state of the used ghosts
void halo_exchange_primvars(hydro::primvars* primvar) {
#ifndef USE_MPI
    (void)primvar;
#else
    if (!have_partners()) return;
    PROFILE("HALO_PRIM");
    exchange_state<HaloPrimCell>(
        [=] HD(int k) {
            HaloPrimCell c;
            c.rho = primvar->rho[k];
            c.v   = primvar->v[k];
            c.E   = primvar->E[k];
            return c;
        },
        [=] HD(int g, const HaloPrimCell& c) {
            primvar->rho_g[g] = c.rho;
            primvar->v_g[g]   = c.v;
            primvar->E_g[g]   = c.E;
        });
#endif
}

// their gradients, all components in one message
void halo_exchange_gradients(gradients::PrimGradients* grads) {
#ifndef USE_MPI
    (void)grads;
#else
    if (!have_partners()) return;
    PROFILE("HALO_GRAD");
    exchange_state<gradients::PrimGradient>([=] HD(int k) { return grads->load(k); },
                                            [=] HD(int g, const gradients::PrimGradient& c) {
                                                grads->rho_g[g] = c.rho;
                                                grads->vx_g[g]  = c.vx;
                                                grads->vy_g[g]  = c.vy;
#ifdef dim_3D
                                                grads->vz_g[g] = c.vz;
#endif
                                                grads->P_g[g]      = c.P;
                                                grads->anchor_g[g] = c.anchor;
                                            });
#endif
}

// and their mesh velocity
void halo_exchange_v_mesh(VMesh* mesh) {
#if !defined(USE_MPI) || !defined(MOVING_MESH)
    (void)mesh;
#else
    if (!have_partners()) return;
    PROFILE("HALO_VMESH");
    exchange_state<POINT_TYPE>([=] HD(int k) { return mesh->v_mesh[k]; },
                               [=] HD(int g, const POINT_TYPE& v) { mesh->v_mesh_g[g] = v; });
#endif
}

// where the centroid of each used ghost sits, relative to its seed
void halo_exchange_centroids(VMesh* mesh) {
#ifndef USE_MPI
    (void)mesh;
#else
    if (!have_partners()) return;
    PROFILE("HALO_COM");
    exchange_state<POINT_TYPE>([=] HD(int k) { return point_diff_periodic(mesh->com[k], mesh->seeds[k]); },
                               [=] HD(int g, const POINT_TYPE& d) { mesh->com_off_g[g] = d; });
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
    exchange_state<double>([=] HD(int k) { return mesh->volumes[k]; },
                           [=] HD(int g, const double& v) { mesh->volumes_g[g] = v; });
#endif
}
#endif

// flags the sent keys of other askers whose cell moved, with their place in pos; returns how many
static size_t flag_moved_exports(const std::vector<int>& moved_ks) {
    const size_t n_sent = halo.n_sent;
    if (moved_ks.empty() || n_sent == 0) return 0;
    std::vector<int> ks(moved_ks);
    std::sort(ks.begin(), ks.end());
    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    int* moved = s_buffers.moved.fit(ks.size());
    std::memcpy(moved, ks.data(), ks.size() * sizeof(int));

    const uint64_t* sent    = halo.sent.data;
    const size_t    n_moved = ks.size();
    const int       me      = decomp.rank;
    unsigned int*   flag    = s_buffers.flag.fit(n_sent);
    s_buffers.pos.fit(n_sent);
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("MOVED_FLAG", n_sent, [=] HD(size_t i) {
        const uint64_t key = sent[i];
        const int      k   = key_cell(key);
        const size_t   at  = lower_bound_of(moved, n_moved, k);
        flag[i]            = (key_rank(key) != me && at < n_moved && moved[at] == k) ? 1u : 0u;
    });
    return scan_flags(n_sent);
}

// cells another rank holds as a ghost, counted once per ghost
int halo_count_moved_exports(const std::vector<int>& moved_ks) {
    if (moved_ks.empty() || halo.askers.empty()) return 0;
    return (int)flag_moved_exports(moved_ks);
}

// every asker gets the new position of the cells it holds, every owner sends ours
void halo_exchange_moved_seeds(const VMesh* mesh, const std::vector<int>& moved_ks, std::vector<MovedSeed>* received) {
    received->clear();

    // the moved cells per asker, in the order they were sent
    const size_t    n_out = halo.askers.empty() ? 0 : flag_moved_exports(moved_ks);
    GhostAnswer*    out   = s_buffers.m_out.fit(n_out);
    uint64_t*       dest  = s_buffers.m_dest.fit(n_out);
    const uint64_t* sent  = halo.sent.data;
    if (n_out > 0) {
        const unsigned int* flag  = s_buffers.flag.data;
        const unsigned int* pos   = s_buffers.pos.data;
        const double3*      seeds = mesh->seeds;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("MOVED_PACK", halo.n_sent, [=] HD(size_t i) {
            if (!flag[i]) return;
            const uint64_t key = sent[i];
            GhostAnswer    a;
            a.k     = key_cell(key);
            a.shift = key_shift(key);
            double s[3];
            shift_of_code(a.shift, s);
            a.p.x = seeds[a.k].x + s[0];
            a.p.y = seeds[a.k].y + s[1];
#ifdef dim_3D
            a.p.z = seeds[a.k].z + s[2];
#endif
            out[pos[i]]  = a;
            dest[pos[i]] = (uint64_t)key_rank(key);
        });
    }

    const Blocks blocks = with_partners(blocks_of_sorted_ranks(dest, n_out, &s_buffers.run_scratch), halo.askers);
    Blocks       in;
    partner_counts(blocks, halo.asked, &in);
    GhostAnswer* moved_in = s_buffers.m_in.fit(in.total);
    exchange_items(out, blocks, moved_in, in, sizeof(GhostAnswer));
    if (in.total == 0) return;

    // the ghost behind every answer, from the ghosts sorted by key
    const size_t n_g  = (size_t)halo.n_ghosts;
    PairSort     sort = s_buffers.sort.fit(n_g);
    {
        const GhostAnswer* g_ans   = halo.g_ans.data;
        const int*         g_owner = halo.g_owner.data;
        uint64_t*          keys    = sort.keys;
        unsigned int*      vals    = sort.vals;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("GHOST_KEYS", n_g, [=] HD(size_t g) {
            keys[g] = ghost_key(g_owner[g], g_ans[g].k, g_ans[g].shift);
            vals[g] = (unsigned int)g;
        });
    }
    sort.sort("GHOST_SORT", n_g, 36 + rank_key_bits());
    const uint64_t*     sorted_keys = sort.keys;
    const unsigned int* sorted_vals = sort.vals;
    int*                ghost       = s_buffers.m_ghost.fit(in.total);
    for (size_t j = 0; j < in.ranks.size(); j++) {
        const int          owner = in.ranks[j];
        const GhostAnswer* as    = moved_in + in.offsets[j];
        int*               g_of  = ghost + in.offsets[j];
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("GHOST_FIND", in.counts[j], [=] HD(size_t i) {
            const uint64_t key = ghost_key(owner, as[i].k, as[i].shift);
            const size_t   at  = lower_bound_of(sorted_keys, n_g, key);
            g_of[i]            = (at < n_g && sorted_keys[at] == key) ? (int)sorted_vals[at] : -1;
        });
    }

    // the few moved ghosts go to the host for the repair
    for (size_t j = 0; j < in.ranks.size(); j++) {
        for (size_t i = in.offsets[j]; i < in.offsets[j] + in.counts[j]; i++) {
            const int g = ghost[i];
            if (g < 0 || halo.g_slot.data[g] < 0) {
                exit_failure("HALO: rank %d moved cell %d, which rank %d does not hold as a ghost\n",
                             in.ranks[j],
                             moved_in[i].k,
                             decomp.rank);
            }
            halo.g_ans.data[g].p = moved_in[i].p;
            MovedSeed ms;
            ms.pos        = moved_in[i].p;
            ms.ghost_slot = halo.g_slot.data[g];
            received->push_back(ms);
        }
    }
}
