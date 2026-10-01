// ghosts by request: the balls go out, the cells inside them come back (included by halo.cu)

// (owner, k, shift) of a ghost in one key; k needs 31 bits, the shift 5
HD inline uint64_t ghost_key(int owner, int k, int shift) {
    return ((uint64_t)owner << 36) | ((uint64_t)(unsigned int)k << 5) | (uint64_t)shift;
}
HD inline uint64_t sent_key(int k, int shift) {
    return ((uint64_t)(unsigned int)k << 5) | (uint64_t)shift;
}

// adds r to a sorted list of ranks if it is not in there
static void add_partner(std::vector<int>* list, int r) {
    auto it = std::lower_bound(list->begin(), list->end(), r);
    if (it == list->end() || *it != r) list->insert(it, r);
}

void halo_begin_build() {
    halo.g_owner.clear();
    halo.g_k.clear();
    halo.g_shift.clear();
    halo.g_pos.clear();
    halo.g_slot.clear();
    halo.g_index.clear();
    halo.n_mpi_ghosts = 0;
    halo.asked.clear();
    halo.askers.clear();
    halo.sent.clear();
    halo.used_subset_ready = 0;
    halo.n_used_send       = 0;
    halo.n_used_recv       = 0;
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

// scratch of the queries, kept between rounds
static unsigned int* s_q_offset       = nullptr;
static unsigned int* s_q_scan_scratch = nullptr;
static size_t        s_q_balls_cap    = 0;
static uint64_t*     s_q_dest         = nullptr;
static uint64_t*     s_q_dest_alt     = nullptr;
static unsigned int* s_q_order        = nullptr;
static unsigned int* s_q_order_alt    = nullptr;
static unsigned int* s_q_sort_scratch = nullptr;
static GhostQuery*   s_q_built        = nullptr;
static GhostQuery*   s_q_sorted       = nullptr;
static size_t        s_q_cap          = 0;

// the queries of all balls in one array, grouped by target rank, ball order kept within a rank
static void build_queries(const POINT_TYPE* cell_pos, const int* cells, const double* radii, int nb, Messages* out) {
    out->clear();
    if (nb == 0) return;
    if ((size_t)nb + 1 > s_q_balls_cap) {
        if (s_q_offset) gpu_free(s_q_offset);
        if (s_q_scan_scratch) gpu_free(s_q_scan_scratch);
        s_q_balls_cap    = std::max((size_t)nb + 1, 2 * s_q_balls_cap);
        s_q_offset       = gpu_alloc<unsigned int>(s_q_balls_cap);
        s_q_scan_scratch = gpu_alloc<unsigned int>(scan_scratch_size(s_q_balls_cap, _MPI_PACK_BLOCK_SIZE_));
    }
    const uint64_t* cuts   = decomp.cuts;
    const int       nranks = decomp.nranks;
    const int       me     = decomp.rank;
    unsigned int*   offset = s_q_offset;

    // count, scan, then every ball writes its queries at its offset
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("COUNT", nb, [=] HD(int i) {
        unsigned int n = 0;
        queries_of_ball(cell_pos[cells[i]], radii[i], cuts, nranks, me, [&](int, const GhostQuery&) { n++; });
        offset[i] = n;
    });
    offset[nb] = 0;
    parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_>("SCAN", (size_t)nb + 1, offset, offset, s_q_scan_scratch);
    const size_t total = offset[nb];
    if (total == 0) return;

    if (total > s_q_cap) {
        for (void* p : {(void*)s_q_dest,
                        (void*)s_q_dest_alt,
                        (void*)s_q_order,
                        (void*)s_q_order_alt,
                        (void*)s_q_sort_scratch,
                        (void*)s_q_built,
                        (void*)s_q_sorted})
            if (p) gpu_free(p);
        s_q_cap          = std::max(total, 2 * s_q_cap);
        s_q_dest         = gpu_alloc<uint64_t>(s_q_cap);
        s_q_dest_alt     = gpu_alloc<uint64_t>(s_q_cap);
        s_q_order        = gpu_alloc<unsigned int>(s_q_cap);
        s_q_order_alt    = gpu_alloc<unsigned int>(s_q_cap);
        s_q_sort_scratch = gpu_alloc<unsigned int>(sort_scratch_size(s_q_cap));
        s_q_built        = gpu_alloc<GhostQuery>(s_q_cap);
        s_q_sorted       = gpu_alloc<GhostQuery>(s_q_cap);
    }
    uint64_t*     dest  = s_q_dest;
    unsigned int* order = s_q_order;
    GhostQuery*   built = s_q_built;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("FILL", nb, [=] HD(int i) {
        unsigned int at = offset[i];
        queries_of_ball(cell_pos[cells[i]], radii[i], cuts, nranks, me, [&](int o, const GhostQuery& q) {
            dest[at]  = (uint64_t)o;
            order[at] = at;
            built[at] = q;
            at++;
        });
    });

    // grouped by target rank; the sort is stable, so ball order stays within a rank
    if (nranks > 1) {
        int key_bits = 1;
        while ((1 << key_bits) < nranks)
            key_bits++;
        parallel_sort_pairs(
            "SORT", total, key_bits, s_q_dest, s_q_order, s_q_dest_alt, s_q_order_alt, s_q_sort_scratch);
    }
    const unsigned int* sorted_order = s_q_order;
    GhostQuery*         sorted       = s_q_sorted;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("GATHER", total, [=] HD(size_t j) { sorted[j] = built[sorted_order[j]]; });

    // only the finished messages go to the host
    std::vector<uint64_t>   h_dest(total);
    std::vector<GhostQuery> h_q(total);
    gpu_memcpy(h_dest.data(), s_q_dest, total * sizeof(uint64_t));
    gpu_memcpy(h_q.data(), s_q_sorted, total * sizeof(GhostQuery));
    for (size_t j = 0; j < total;) {
        size_t e = j;
        while (e < total && h_dest[e] == h_dest[j])
            e++;
        out->to((int)h_dest[j]).assign((const char*)&h_q[j], (const char*)&h_q[e]);
        j = e;
    }
}

// scratch of the answers: per own cell one bit per shift of an asker's balls, then the compacted list
static GhostQuery*   s_queries      = nullptr;
static size_t        s_queries_cap  = 0;
static unsigned int* s_hit_mask     = nullptr;
static unsigned int* s_hit_offset   = nullptr;
static unsigned int* s_scan_scratch = nullptr;
static uint64_t*     s_hits         = nullptr;
static GhostAnswer*  s_answers      = nullptr;
static size_t        s_cells_cap    = 0;
static size_t        s_hits_cap     = 0;
static int*          s_stack_full   = nullptr;

// marks the own cells in a ball; the tree boxes only filter, a cell counts by its exact distance
HD inline void
mark_cells_in_ball(const knn_problem* knn, int n_hydro, const GhostQuery& q, unsigned int* mask, int* stack_full) {
    const int n = knn->len_pts;
    if (n == 0) return;
    const double       r2  = q.r * q.r;
    const unsigned int bit = 1u << q.shift;
    if (n == 1) {
        const unsigned int orig = knn->d_permutation[0];
        if ((int)orig < n_hydro && knn::dist2_point(q.c, knn->d_stored_points[0]) <= r2)
            portable_atomicOr(&mask[orig], bit);
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
                if ((int)orig < n_hydro && knn::dist2_point(q.c, nd.lo[h]) <= r2) portable_atomicOr(&mask[orig], bit);
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

// the own cells inside the balls of one asker, as (k, shift) keys in that order, each once, and the answers
static void cells_in_balls(VMesh*                    mesh,
                           const POINT_TYPE*         cell_pos,
                           const GhostQuery*         qs,
                           size_t                    nq,
                           std::vector<uint64_t>*    found,
                           std::vector<GhostAnswer>* answers) {
    found->clear();
    answers->clear();
    const int n_hydro = (int)mesh->n_hydro;
    if (nq == 0 || n_hydro == 0) return;

    if (nq > s_queries_cap) {
        if (s_queries) gpu_free(s_queries);
        s_queries_cap = std::max(nq, 2 * s_queries_cap);
        s_queries     = gpu_alloc<GhostQuery>(s_queries_cap);
    }
    if ((size_t)n_hydro > s_cells_cap) {
        if (s_hit_mask) gpu_free(s_hit_mask);
        if (s_hit_offset) gpu_free(s_hit_offset);
        if (s_scan_scratch) gpu_free(s_scan_scratch);
        s_cells_cap    = std::max((size_t)n_hydro, 2 * s_cells_cap);
        s_hit_mask     = gpu_alloc<unsigned int>(s_cells_cap);
        s_hit_offset   = gpu_alloc<unsigned int>(s_cells_cap);
        s_scan_scratch = gpu_alloc<unsigned int>(scan_scratch_size(s_cells_cap, _MPI_PACK_BLOCK_SIZE_));
    }
    if (!s_stack_full) s_stack_full = gpu_calloc<int>(1);
    std::memcpy(s_queries, qs, nq * sizeof(GhostQuery));
    gpu_memset(s_hit_mask, 0, (size_t)n_hydro * sizeof(unsigned int));

    const knn_problem* knn        = mesh->knn;
    GhostQuery*        queries    = s_queries;
    unsigned int*      mask       = s_hit_mask;
    unsigned int*      offset     = s_hit_offset;
    int*               stack_full = s_stack_full;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "MARK", nq, [=] HD(size_t i) { mark_cells_in_ball(knn, n_hydro, queries[i], mask, stack_full); });
    if (*s_stack_full) exit_failure("HALO: rank %d ran out of tree stack answering a ball\n", decomp.rank);

    parallel_for<_MPI_PACK_BLOCK_SIZE_>(
        "COUNT", n_hydro, [=] HD(size_t k) { offset[k] = (unsigned int)portable_popcount(mask[k]); });
    parallel_exclusive_scan<_MPI_PACK_BLOCK_SIZE_>("SCAN", (size_t)n_hydro, offset, offset, s_scan_scratch);
    const size_t total = (size_t)offset[n_hydro - 1] + (size_t)portable_popcount(mask[n_hydro - 1]);
    if (total == 0) return;

    if (total > s_hits_cap) {
        if (s_hits) gpu_free(s_hits);
        if (s_answers) gpu_free(s_answers);
        s_hits_cap = std::max(total, 2 * s_hits_cap);
        s_hits     = gpu_alloc<uint64_t>(s_hits_cap);
        s_answers  = gpu_alloc<GhostAnswer>(s_hits_cap);
    }
    uint64_t*    hits = s_hits;
    GhostAnswer* ans  = s_answers;
    parallel_for<_MPI_PACK_BLOCK_SIZE_>("FILL", n_hydro, [=] HD(size_t k) {
        unsigned int m  = mask[k];
        unsigned int at = offset[k];
        for (int shift = 0; m != 0; shift++, m >>= 1) {
            if (!(m & 1u)) continue;
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
            hits[at] = sent_key((int)k, shift);
            ans[at]  = a;
            at++;
        }
    });
    found->resize(total);
    answers->resize(total);
    gpu_memcpy(found->data(), s_hits, total * sizeof(uint64_t));
    gpu_memcpy(answers->data(), s_answers, total * sizeof(GhostAnswer));
}

void halo_request_balls(VMesh* mesh, const POINT_TYPE* cell_pos, const int* cells, const double* radii, int nb) {
    PROFILE("HALO_REQUEST");
    const int me = decomp.rank;

    Messages q_out, q_in;
    {
        PROFILE("BUILD");
        build_queries(cell_pos, cells, radii, nb, &q_out);
    }
    {
        PROFILE("QUERIES");
        sparse_exchange(q_out, &q_in);
    }
    for (int r : q_out.ranks)
        if (r != me) add_partner(&halo.asked, r);

    // answers: every query on the tree of this build, only own cells, each cell once per asker and build
    Messages a_out;
    {
        PROFILE("ANSWER");
        for (size_t m = 0; m < q_in.ranks.size(); m++) {
            const int         src = q_in.ranks[m];
            const size_t      nq  = count_of<GhostQuery>(q_in.data[m]);
            const GhostQuery* qs  = items_of<GhostQuery>(q_in.data[m]);

            std::vector<uint64_t>    found;
            std::vector<GhostAnswer> answers;
            cells_in_balls(mesh, cell_pos, qs, nq, &found, &answers);

            // what this asker already has stays home
            std::vector<char>& msg = a_out.to(src);
            if (src == me) {
                msg.assign((const char*)answers.data(), (const char*)(answers.data() + answers.size()));
                continue;
            }
            auto         it = std::lower_bound(halo.askers.begin(), halo.askers.end(), src);
            const size_t a  = it - halo.askers.begin();
            if (it == halo.askers.end() || *it != src) {
                halo.askers.insert(it, src);
                halo.sent.insert(halo.sent.begin() + a, std::vector<uint64_t>());
            }
            std::vector<uint64_t>& sent = halo.sent[a];
            std::vector<uint64_t>  merged;
            merged.reserve(sent.size() + found.size());
            size_t j = 0;
            for (size_t i = 0; i < found.size(); i++) {
                while (j < sent.size() && sent[j] < found[i])
                    merged.push_back(sent[j++]);
                if (j < sent.size() && sent[j] == found[i]) continue;
                merged.push_back(found[i]);
                append(msg, answers[i]);
            }
            while (j < sent.size())
                merged.push_back(sent[j++]);
            sent.swap(merged);
        }
    }

    // every rank asked answers, possibly with nothing
    Messages a_in;
    {
        PROFILE("ANSWERS");
        partner_exchange(a_out, q_out.ranks, &a_in);
    }

    // new ghosts in owner order, each owner's cells in the order it sent them
    for (size_t m = 0; m < a_in.ranks.size(); m++) {
        const int          owner = a_in.ranks[m];
        const size_t       na    = count_of<GhostAnswer>(a_in.data[m]);
        const GhostAnswer* as    = items_of<GhostAnswer>(a_in.data[m]);
        for (size_t i = 0; i < na; i++) {
            const uint64_t key = ghost_key(owner, as[i].k, as[i].shift);
            if (halo.g_index.count(key)) continue;
            halo.g_index[key] = (int)halo.g_owner.size();
            halo.g_owner.push_back(owner);
            halo.g_k.push_back(as[i].k);
            halo.g_shift.push_back(as[i].shift);
            halo.g_pos.push_back(as[i].p);
            halo.g_slot.push_back(owner == me ? -1 : halo.n_mpi_ghosts++);
        }
    }
}

// the ghosts go after the cells: a copy of an own cell stands for that cell, any other ghost for its slot;
// written in three bulk copies
void halo_write_ghosts(VMesh* mesh, POINT_TYPE* pts, uint64_t* ghost_ids, int n_hydro) {
    if (halo.n_mpi_ghosts > n_mpi_capacity) halo_grow_capacity(halo.n_mpi_ghosts);
    const size_t          n_ghosts = halo.g_owner.size();
    std::vector<uint64_t> ids(n_ghosts);
    std::vector<double3>  slot_seeds((size_t)halo.n_mpi_ghosts);
    for (size_t g = 0; g < n_ghosts; g++) {
        const int slot = halo.g_slot[g];
        ids[g]         = (slot < 0) ? (uint64_t)halo.g_k[g] : (uint64_t)(n_hydro + slot);
        if (slot < 0) continue;
        const POINT_TYPE& p = halo.g_pos[g];
#ifdef dim_3D
        slot_seeds[slot] = double3{p.x, p.y, p.z};
#else
        slot_seeds[slot] = double3{p.x, p.y, 0.0};
#endif
    }
    if (n_ghosts > 0) {
        gpu_memcpy(pts + n_hydro, halo.g_pos.data(), n_ghosts * sizeof(POINT_TYPE));
        gpu_memcpy(ghost_ids, ids.data(), n_ghosts * sizeof(uint64_t));
    }
    if (!slot_seeds.empty()) gpu_memcpy(mesh->seeds_g, slot_seeds.data(), slot_seeds.size() * sizeof(double3));
    mesh->n_mpi_ghosts = halo.n_mpi_ghosts;
}
