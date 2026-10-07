// ghosts by request: the balls go out, the cells inside them come back (included by halo.cu)

// (owner, k, shift) of a ghost in one key; k needs 31 bits, the shift 5
HD inline uint64_t ghost_key(int owner, int k, int shift) {
    return ((uint64_t)owner << 36) | ((uint64_t)(unsigned int)k << 5) | (uint64_t)shift;
}
HD inline int key_rank(uint64_t key) {
    return (int)(key >> 36);
}
HD inline int key_cell(uint64_t key) {
    return (int)((key >> 5) & 0x7FFFFFFFull);
}
HD inline int key_shift(uint64_t key) {
    return (int)(key & 31u);
}

// adds r to a sorted list of ranks if it is not in there
static void add_partner(std::vector<int>* list, int r) {
    auto it = std::lower_bound(list->begin(), list->end(), r);
    if (it == list->end() || *it != r) list->insert(it, r);
}

void halo_begin_build() {
    halo.n_ghosts     = 0;
    halo.n_mpi_ghosts = 0;
    halo.n_sent       = 0;
    halo.asked.clear();
    halo.askers.clear();
    halo.used_subset_ready = 0;
    halo.state_send.clear();
    halo.state_recv.clear();
}

// a ball's image rarely touches many ranks; a short list keeps one from being asked twice
constexpr int SEEN_RANKS = 16;

// the queries of one ball: one per box shift whose image of the ball reaches into the box, to every rank
// whose keys that image touches; the part of the ball this rank holds itself needs no query
template <typename F>
HD inline void queries_of_ball(const POINT_TYPE& c, double r, const uint64_t* cuts, int nranks, int me, F emit) {
    // a shift of +1 is only worth a look if the ball leaves the box at the top, -1 at the bottom
    const double x[3] = {c.x,
                         c.y,
#ifdef dim_3D
                         c.z
#else
                         0.5
#endif
    };
    int lo[3], hi[3];
    for (int i = 0; i < 3; i++) {
        lo[i] = (x[i] - r < 0.0) ? -1 : 0;
        hi[i] = (x[i] + r > 1.0) ? 1 : 0;
    }
    for (int sx = lo[0]; sx <= hi[0]; sx++) {
        for (int sy = lo[1]; sy <= hi[1]; sy++) {
            for (int sz = lo[2]; sz <= hi[2]; sz++) {
                POINT_TYPE img;
                img.x = c.x - (double)sx;
                img.y = c.y - (double)sy;
#ifdef dim_3D
                img.z = c.z - (double)sz;
#endif
                if (dist2_to_cube(KeyCube{0, {0u, 0u, 0u}}, img) > r * r) continue;
                const int code = shift_code(sx, sy, sz);
                int       seen[SEEN_RANKS];
                int       n_seen = 0;
                for_each_rank_in_ball(img, r, cuts, nranks, [&](int o) {
                    for (int j = 0; j < n_seen; j++)
                        if (seen[j] == o) return true;
                    if (n_seen < SEEN_RANKS) seen[n_seen++] = o;
                    if (o == me && code == SHIFT_NONE) return true;
                    GhostQuery q;
                    q.c     = img;
                    q.r     = r;
                    q.shift = code;
                    emit(o, q);
                    return true;
                });
            }
        }
    }
}

// the queries of all balls in one array, grouped by target rank, ball order kept within a rank
static const GhostQuery*
build_queries(const POINT_TYPE* cell_pos, const int* cells, const double* radii, int nb, Blocks* out) {
    out->clear();
    if (nb == 0) return nullptr;
    const uint64_t* cuts   = decomp.cuts;
    const int       nranks = decomp.nranks;
    const int       me     = decomp.rank;
    unsigned int*   offset = s_buffers.q_offset.fit((size_t)nb + 1);

    // count, scan, then every ball writes its queries at its offset
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("COUNT", nb, [=] HD(int i) {
        unsigned int n = 0;
        queries_of_ball(cell_pos[cells[i]], radii[i], cuts, nranks, me, [&](int, const GhostQuery&) { n++; });
        offset[i] = n;
    });
    offset[nb] = 0;
    parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_>(
        "SCAN", (size_t)nb + 1, offset, offset, s_buffers.scan_scratch((size_t)nb + 1));
    const size_t total = offset[nb];
    if (total == 0) return nullptr;

    PairSort    sort  = s_buffers.sort.fit(total);
    GhostQuery* built = s_buffers.q_built.fit(total);
    {
        uint64_t*     dest  = sort.keys;
        unsigned int* order = sort.vals;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("FILL", nb, [=] HD(int i) {
            unsigned int at = offset[i];
            queries_of_ball(cell_pos[cells[i]], radii[i], cuts, nranks, me, [&](int o, const GhostQuery& q) {
                dest[at]  = (uint64_t)o;
                order[at] = at;
                built[at] = q;
                at++;
            });
        });
    }

    // grouped by target rank; the sort is stable, so ball order stays within a rank
    if (nranks > 1) sort.sort("SORT", total, rank_key_bits());
    const unsigned int* sorted_order = sort.vals;
    GhostQuery*         sorted       = s_buffers.q_sorted.fit(total);
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("GATHER", total, [=] HD(size_t j) { sorted[j] = built[sorted_order[j]]; });

    *out = blocks_of_sorted_ranks(sort.keys, total, &s_buffers.run_scratch);
    return sorted;
}

// cells per run of the hit mask; an asker's balls only touch the runs near it, so only those are read
constexpr int MARK_RUN = 256;

// marks the own cells in a ball and their run; the tree boxes only filter, a cell counts by its exact distance
HD inline void mark_cells_in_ball(const knn_problem* knn,
                                  int                n_hydro,
                                  const GhostQuery&  q,
                                  unsigned int*      mask,
                                  unsigned int*      run_hit,
                                  int*               stack_full) {
    const int n = knn->len_pts;
    if (n == 0) return;
    const double       r2  = q.r * q.r;
    const unsigned int bit = 1u << q.shift;
    if (n == 1) {
        const unsigned int orig = knn->d_permutation[0];
        if ((int)orig < n_hydro && knn::dist2_point(q.c, knn->d_stored_points[0]) <= r2) {
            portable_atomicOr(&mask[orig], bit);
            portable_atomicOr(&run_hit[orig / MARK_RUN], 1u);
        }
        return;
    }
    const TreeNode* nodes = knn->d_nodes;
    const int       leaf0 = n - 1;
    int             stack[knn::TREE_STACK];
    int             sp = 0;
    stack[sp++]        = 0;
    while (sp > 0) {
        const TreeNode& nd = nodes[stack[--sp]];
        for (int h = 0; h < 2; h++) {
            const int c = nd.child[h];
            if (c >= leaf0) {
                const unsigned int orig = knn->d_permutation[c - leaf0];
                if ((int)orig < n_hydro && knn::dist2_point(q.c, nd.lo[h]) <= r2) {
                    portable_atomicOr(&mask[orig], bit);
                    portable_atomicOr(&run_hit[orig / MARK_RUN], 1u);
                }
            } else if (knn::dist2_box(nd.lo[h], nd.hi[h], q.c) <= r2 * knn::TREE_PRUNE_SLACK) {
                if (sp == knn::TREE_STACK) {
                    *stack_full = 1;
                    return;
                }
                stack[sp++] = c;
            }
        }
    }
}

// the own cells inside the balls of one asker as keys (asker, k, shift) in that order, each once, and the
// answers; returns how many. The hit mask and the run flags are zero before and after
static size_t hits_of_asker(VMesh* mesh, const POINT_TYPE* cell_pos, const GhostQuery* qs, size_t nq, int asker) {
    const int n_hydro = (int)mesh->n_hydro;
    if (nq == 0 || n_hydro == 0) return 0;

    const knn_problem* knn        = mesh->knn;
    unsigned int*      mask       = s_buffers.hit_mask.data;
    unsigned int*      run_hit    = s_buffers.run_hit.data;
    int*               stack_full = s_buffers.stack_full.data;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "MARK", nq, [=] HD(size_t i) { mark_cells_in_ball(knn, n_hydro, qs[i], mask, run_hit, stack_full); });
    if (*stack_full) exit_failure("HALO: rank %d ran out of tree stack answering a ball\n", decomp.rank);

    // the runs with a hit in cell order; their flags go back to zero
    const size_t       n_runs   = (size_t)(n_hydro + MARK_RUN - 1) / MARK_RUN;
    unsigned int*      run_pos  = s_buffers.pos.fit(n_runs);
    unsigned int*      run_list = s_buffers.run_list.fit(n_runs);
    const unsigned int last_run = run_hit[n_runs - 1];
    parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_>(
        "RUN_SCAN", n_runs, run_hit, run_pos, s_buffers.scan_scratch(n_runs));
    const size_t n_touched = (size_t)run_pos[n_runs - 1] + last_run;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("RUN_LIST", n_runs, [=] HD(size_t r) {
        if (run_hit[r]) {
            run_list[run_pos[r]] = (unsigned int)r;
            run_hit[r]           = 0;
        }
    });
    if (n_touched == 0) return 0;

    // the hits of those runs, in cell order
    const size_t  m      = n_touched * MARK_RUN;
    unsigned int* offset = s_buffers.hit_offset.fit(m);
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("COUNT", m, [=] HD(size_t i) {
        const size_t k = (size_t)run_list[i / MARK_RUN] * MARK_RUN + i % MARK_RUN;
        offset[i]      = (k < (size_t)n_hydro) ? (unsigned int)portable_popcount(mask[k]) : 0u;
    });
    const unsigned int last_count = offset[m - 1];
    parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_>("SCAN", m, offset, offset, s_buffers.scan_scratch(m));
    const size_t total = (size_t)offset[m - 1] + last_count;

    uint64_t*    hits = s_buffers.hits.fit(total);
    GhostAnswer* ans  = s_buffers.hit_ans.fit(total);
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("FILL", m, [=] HD(size_t i) {
        const size_t k = (size_t)run_list[i / MARK_RUN] * MARK_RUN + i % MARK_RUN;
        if (k >= (size_t)n_hydro) return;
        unsigned int mm = mask[k];
        unsigned int at = offset[i];
        for (int shift = 0; mm != 0; shift++, mm >>= 1) {
            if (!(mm & 1u)) continue;
            double s[3];
            shift_of_code(shift, s);
            GhostAnswer a;
            a.k     = (int)k;
            a.shift = shift;
            a.p     = cell_pos[k];
            a.p.x   = a.p.x + s[0];
            a.p.y   = a.p.y + s[1];
#ifdef dim_3D
            a.p.z = a.p.z + s[2];
#endif
            hits[at] = ghost_key(asker, (int)k, shift);
            ans[at]  = a;
            at++;
        }
        mask[k] = 0;
    });
    return total;
}

// the hits this asker did not get earlier in the build go to the answers at a_at and to the pending keys;
// returns how many
static size_t keep_new_hits(size_t h, size_t a_at) {
    if (h == 0) return 0;
    const uint64_t* sent   = halo.sent.data;
    const size_t    n_sent = halo.n_sent;
    const uint64_t* hits   = s_buffers.hits.data;
    unsigned int*   flag   = s_buffers.flag.fit(h);
    unsigned int*   pos    = s_buffers.pos.fit(h);
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("NEW", h, [=] HD(size_t i) {
        const size_t j = lower_bound_of(sent, n_sent, hits[i]);
        flag[i]        = (j < n_sent && sent[j] == hits[i]) ? 0u : 1u;
    });
    parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_>("NEW_SCAN", h, flag, pos, s_buffers.scan_scratch(h));
    const size_t n_new = (size_t)pos[h - 1] + flag[h - 1];
    if (n_new == 0) return 0;

    const GhostAnswer* ans     = s_buffers.hit_ans.data;
    GhostAnswer*       a_out   = s_buffers.a_out.grow(a_at + n_new) + a_at;
    uint64_t*          pending = s_buffers.pending.grow(s_buffers.n_pending + n_new) + s_buffers.n_pending;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("KEEP", h, [=] HD(size_t i) {
        if (!flag[i]) return;
        a_out[pos[i]]   = ans[i];
        pending[pos[i]] = hits[i];
    });
    s_buffers.n_pending += n_new;
    return n_new;
}

// the pending keys into the sorted sent keys; neither has a key of the other, so every key knows its place
static void merge_pending_into_sent() {
    if (s_buffers.n_pending == 0) return;
    const size_t    n_a = halo.n_sent;
    const size_t    n_b = s_buffers.n_pending;
    const uint64_t* a   = halo.sent.data;
    const uint64_t* b   = s_buffers.pending.data;
    uint64_t*       out = s_buffers.sent_alt.fit(n_a + n_b);
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "MERGE_A", n_a, [=] HD(size_t i) { out[i + lower_bound_of(b, n_b, a[i])] = a[i]; });
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "MERGE_B", n_b, [=] HD(size_t j) { out[j + lower_bound_of(a, n_a, b[j])] = b[j]; });
    std::swap(halo.sent, s_buffers.sent_alt);
    halo.n_sent         = n_a + n_b;
    s_buffers.n_pending = 0;
}

// the answers that came back become ghosts in owner order, each owner's cells in the order it sent them
static void append_ghosts(const Blocks& a_in) {
    const size_t n_total = (size_t)halo.n_ghosts + a_in.total;
    halo.g_ans.grow(n_total);
    halo.g_owner.grow(n_total);
    halo.g_slot.grow(n_total);
    const int me   = decomp.rank;
    int       slot = halo.n_mpi_ghosts;
    for (size_t j = 0; j < a_in.ranks.size(); j++) {
        const size_t n = a_in.counts[j];
        if (n == 0) continue;
        const int          owner  = a_in.ranks[j];
        const int          slot0  = (owner == me) ? -1 : slot;
        const size_t       g0     = (size_t)halo.n_ghosts + a_in.offsets[j];
        const GhostAnswer* in     = s_buffers.a_in.data + a_in.offsets[j];
        GhostAnswer*       g_ans  = halo.g_ans.data + g0;
        int*               g_own  = halo.g_owner.data + g0;
        int*               g_slot = halo.g_slot.data + g0;
        parallel_for<_MPI_PACK_BLOCK_SIZE_>("APPEND", n, [=] HD(size_t i) {
            g_ans[i]  = in[i];
            g_own[i]  = owner;
            g_slot[i] = (slot0 < 0) ? -1 : slot0 + (int)i;
        });
        if (owner != me) slot += (int)n;
    }
    halo.n_ghosts     = (int)n_total;
    halo.n_mpi_ghosts = slot;
}

void halo_request_balls(VMesh* mesh, const POINT_TYPE* cell_pos, const int* cells, const double* radii, int nb) {
    PROFILE("HALO_REQUEST");
    const int me = decomp.rank;

    Blocks            q_out, q_in;
    const GhostQuery* q_send;
    {
        PROFILE("BUILD");
        q_send = build_queries(cell_pos, cells, radii, nb, &q_out);
    }
    {
        PROFILE("QUERIES");
        sparse_counts(q_out, &q_in);
        exchange_items(q_send, q_out, s_buffers.q_in.fit(q_in.total), q_in, sizeof(GhostQuery));
    }
    for (int r : q_out.ranks)
        if (r != me) add_partner(&halo.asked, r);
    for (int r : q_in.ranks)
        if (r != me) add_partner(&halo.askers, r);

    // answers: every query on the tree of this build, only own cells, each cell once per asker and build
    Blocks a_out;
    {
        PROFILE("ANSWER");
        const size_t n_hydro = (size_t)mesh->n_hydro;
        const size_t n_runs  = (n_hydro + MARK_RUN - 1) / MARK_RUN;
        if (n_hydro > 0) {
            gpu_memset(s_buffers.hit_mask.fit(n_hydro), 0, n_hydro * sizeof(unsigned int));
            gpu_memset(s_buffers.run_hit.fit(n_runs), 0, n_runs * sizeof(unsigned int));
        }
        *s_buffers.stack_full.fit(1) = 0;

        size_t a_total = 0;
        for (size_t m = 0; m < q_in.ranks.size(); m++) {
            const GhostQuery* qs = s_buffers.q_in.data + q_in.offsets[m];
            const size_t      h  = hits_of_asker(mesh, cell_pos, qs, q_in.counts[m], q_in.ranks[m]);
            const size_t      n  = keep_new_hits(h, a_total);
            a_out.add(q_in.ranks[m], n);
            a_total += n;
        }
        merge_pending_into_sent();
    }

    // every rank asked answers, possibly with nothing
    Blocks a_in;
    {
        PROFILE("ANSWERS");
        partner_counts(a_out, q_out.ranks, &a_in);
        exchange_items(s_buffers.a_out.data, a_out, s_buffers.a_in.fit(a_in.total), a_in, sizeof(GhostAnswer));
    }
    append_ghosts(a_in);
}

// the ghosts go after the cells: a copy of an own cell stands for that cell, any other ghost for its slot
void halo_write_ghosts(VMesh* mesh, POINT_TYPE* pts) {
    if (halo.n_mpi_ghosts > n_mpi_capacity) halo_grow_capacity(halo.n_mpi_ghosts);
    const int          n_hydro   = (int)mesh->n_hydro;
    const GhostAnswer* ans       = halo.g_ans.data;
    const int*         slot      = halo.g_slot.data;
    uint64_t*          ghost_ids = mesh->ghost_ids;
    double3*           seeds_g   = mesh->seeds_g;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("WRITE_GHOSTS", halo.n_ghosts, [=] HD(int g) {
        const POINT_TYPE p = ans[g].p;
        const int        s = slot[g];
        pts[n_hydro + g]   = p;
        ghost_ids[g]       = (s < 0) ? (uint64_t)ans[g].k : (uint64_t)(n_hydro + s);
        if (s < 0) return;
#ifdef dim_3D
        seeds_g[s] = double3{p.x, p.y, p.z};
#else
        seeds_g[s] = double3{p.x, p.y, 0.0};
#endif
    });
    mesh->n_mpi_ghosts = halo.n_mpi_ghosts;
}
