// gathers the timers of all ranks and prints the tree (included by profiler.cu)

namespace {
#ifdef USE_MPI
    // names as one blob of null terminated strings
    std::vector<char> pack_names(const std::vector<std::string>& names) {
        std::vector<char> buf;
        for (const auto& n : names) {
            buf.insert(buf.end(), n.begin(), n.end());
            buf.push_back('\0');
        }
        return buf;
    }

    std::vector<std::string> unpack_names(const char* buf, int len) {
        std::vector<std::string> out;
        int                      start = 0;
        for (int i = 0; i < len; i++) {
            if (buf[i] == '\0') {
                if (i > start) out.emplace_back(buf + start, i - start);
                start = i + 1;
            }
        }
        return out;
    }
#endif

#ifdef USE_MPI
    std::vector<char>
    // allgather of byte blobs
    allgather_bytes(const std::vector<char>& mine, int nranks, std::vector<int>& lens, std::vector<int>& displs) {
        int my_len = (int)mine.size();
        lens.assign(nranks, 0);
        MPI_Allgather(&my_len, 1, MPI_INT, lens.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs.assign(nranks, 0);
        int total = 0;
        for (int r = 0; r < nranks; r++) {
            displs[r] = total;
            total += lens[r];
        }
        std::vector<char> all(total);
        MPI_Allgatherv(mine.data(), my_len, MPI_BYTE, all.data(), lens.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
        return all;
    }
#endif

    std::vector<std::vector<std::string>> allgather_timer_names(const std::vector<std::string>& my_names, int nranks) {
        std::vector<std::vector<std::string>> result(nranks);
#ifdef USE_MPI
        if (nranks > 1) {
            std::vector<int>        lens, displs;
            const std::vector<char> all = allgather_bytes(pack_names(my_names), nranks, lens, displs);
            for (int r = 0; r < nranks; r++) {
                result[r] = unpack_names(all.data() + displs[r], lens[r]);
            }
            return result;
        }
#endif
        result[0] = my_names;
        return result;
    }

    std::vector<std::vector<char>> allgather_timer_kinds(const std::vector<char>& my_kinds, int nranks) {
        std::vector<std::vector<char>> result(nranks);
#ifdef USE_MPI
        if (nranks > 1) {
            std::vector<int>        lens, displs;
            const std::vector<char> all = allgather_bytes(my_kinds, nranks, lens, displs);
            for (int r = 0; r < nranks; r++) {
                result[r].assign(all.begin() + displs[r], all.begin() + displs[r] + lens[r]);
            }
            return result;
        }
#endif
        result[0] = my_kinds;
        return result;
    }

    // one node of the printed tree
    struct TreeNode {
        std::string              full_path;
        std::string              leaf;
        char                     kind      = 'c';
        double                   cum_sum_s = 0.0;
        double                   imbalance = 1.0;
        std::vector<std::string> children;
    };

    // splits DOMAIN.SUB.LEAF into parent and leaf
    void split_path(const std::string& p, std::string& parent, std::string& leaf) {
        const auto pos = p.rfind('.');
        if (pos == std::string::npos) {
            parent.clear();
            leaf = p;
        } else {
            parent = p.substr(0, pos);
            leaf   = p.substr(pos + 1);
        }
    }

    // print a node and its children, largest first
    void print_subtree(std::ostream&                                    out,
                       const std::unordered_map<std::string, TreeNode>& nodes,
                       const std::string&                               path,
                       int                                              depth,
                       double                                           total_s) {
        auto it = nodes.find(path);
        if (it == nodes.end()) return;
        const TreeNode& n = it->second;

        const std::string indent(depth * 2, ' ');
        const double      pct_tot = (total_s > 0.0) ? 100.0 * n.cum_sum_s / total_s : 0.0;

        const std::string name = indent + n.leaf;
        const char*       tag  = (n.kind == 'm') ? "[mpi]" : (n.kind == 'g') ? "[gpu]" : "     ";

        char buf[256];
        std::snprintf(buf,
                      sizeof(buf),
                      "%-30s  %-5s  %7.3fs  %9.1f%%  %9.3f\n",
                      name.c_str(),
                      tag,
                      n.cum_sum_s,
                      pct_tot,
                      n.imbalance);
        out << buf;

        std::vector<const TreeNode*> kids;
        for (const auto& c : n.children) {
            auto cit = nodes.find(c);
            if (cit != nodes.end()) kids.push_back(&cit->second);
        }
        std::sort(
            kids.begin(), kids.end(), [](const TreeNode* a, const TreeNode* b) { return a->cum_sum_s > b->cum_sum_s; });
        for (const auto* k : kids)
            print_subtree(out, nodes, k->full_path, depth + 1, total_s);
    }

} // namespace

// gather the timers of all ranks, print the tree on rank 0
void Profiler::print_results() {
    drain_gpu_events(true);

    const int nranks = proteus_mpi::nranks();
    const int rank   = proteus_mpi::rank();

    std::vector<std::string> my_names;
    my_names.reserve(s_cum_us.size());
    for (const auto& kv : s_cum_us)
        my_names.push_back(kv.first);

    auto                  all_names = allgather_timer_names(my_names, nranks);
    std::set<std::string> union_names;
    for (const auto& v : all_names)
        for (const auto& n : v)
            union_names.insert(n);

    std::vector<std::string> ordered(union_names.begin(), union_names.end());
    const int                ntimers = (int)ordered.size();
    std::vector<double>      my_cum(ntimers, 0.0);
    {
        auto                                    live = collect_current();
        std::unordered_map<std::string, double> mine;
        for (const auto& r : live)
            mine[r.first] = r.second / 1e6;
        for (int i = 0; i < ntimers; i++) {
            auto it = mine.find(ordered[i]);
            if (it != mine.end()) my_cum[i] = it->second;
        }
    }

    // sum for the time, max for the imbalance
    std::vector<double> cum_sum = my_cum, cum_max = my_cum;
#ifdef USE_MPI
    if (nranks > 1 && ntimers > 0) {
        MPI_Allreduce(MPI_IN_PLACE, cum_sum.data(), ntimers, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, cum_max.data(), ntimers, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
#endif

    std::vector<char> my_kind(ntimers, 0), out_kind(ntimers, 0);
    for (int i = 0; i < ntimers; i++) {
        auto it = s_kind.find(ordered[i]);
        if (it != s_kind.end()) my_kind[i] = it->second;
    }
#ifdef USE_MPI
    if (nranks > 1 && ntimers > 0) {
        MPI_Allreduce(my_kind.data(), out_kind.data(), ntimers, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD);
    } else {
        out_kind = my_kind;
    }
#else
    out_kind = my_kind;
#endif

    // the reductions above need every rank
    if (rank != 0) return;

    std::unordered_map<std::string, TreeNode> nodes;
    for (int i = 0; i < ntimers; i++) {
        TreeNode n;
        n.full_path = ordered[i];
        std::string parent;
        split_path(n.full_path, parent, n.leaf);
        n.kind             = out_kind[i] ? out_kind[i] : 'c';
        n.cum_sum_s        = cum_sum[i];
        const double avg   = cum_sum[i] / (double)nranks;
        n.imbalance        = (avg > 0.0) ? cum_max[i] / avg : 1.0; // slowest rank over the average
        nodes[n.full_path] = n;
    }
    std::vector<std::string> roots;
    for (auto& kv : nodes) {
        std::string parent, leaf;
        split_path(kv.first, parent, leaf);
        if (parent.empty()) {
            roots.push_back(kv.first);
        } else {
            auto pit = nodes.find(parent);
            if (pit != nodes.end())
                pit->second.children.push_back(kv.first);
            else
                roots.push_back(kv.first);
        }
    }

    std::ostream&     out   = logging::root();
    constexpr int     WIDTH = 70;
    const std::string title = " Profiling Results ";
    const int         side  = (WIDTH - (int)title.size()) / 2;
    const int         rside = WIDTH - side - (int)title.size();
    out << "\n" << std::string(side, '=') << title << std::string(rside, '=') << "\n";

    const int total_threads = nranks * logging::omp_threads();
#ifdef CPU_DEBUG
    const int total_gpus = 0;
#else
    const int total_gpus = nranks;
#endif
    char left[64];
    std::snprintf(left, sizeof(left), "sum of %d ranks (%d threads, %d GPUs)", nranks, total_threads, total_gpus);
    char hdr[256];
    std::snprintf(hdr, sizeof(hdr), "%-37s  %8s  %10s  %9s\n", left, "time", "percentage", "imbalance");
    out << hdr;
    out << std::string(WIDTH, '-') << "\n";

    double total_s = 0.0;
    auto   it_tot  = nodes.find("TOTAL");
    if (it_tot != nodes.end()) total_s = it_tot->second.cum_sum_s;

    std::sort(roots.begin(), roots.end(), [&](const std::string& a, const std::string& b) {
        return nodes[a].cum_sum_s > nodes[b].cum_sum_s;
    });
    for (const auto& r : roots)
        print_subtree(out, nodes, r, 0, total_s);

    double mpi_total_s = 0.0, gpu_total_s = 0.0;
    for (const auto& kv : nodes) {
        if (kv.second.kind == 'm') mpi_total_s += kv.second.cum_sum_s;
        if (kv.second.kind == 'g') gpu_total_s += kv.second.cum_sum_s;
    }
    auto pct = [&](double v) { return (total_s > 0.0) ? 100.0 * v / total_s : 0.0; };
    out << std::string(WIDTH, '-') << "\n";
    char foot[256];
    std::snprintf(
        foot, sizeof(foot), "%-30s  %-5s  %7.3fs  %9.1f%%\n", "MPI total", "[mpi]", mpi_total_s, pct(mpi_total_s));
    out << foot;
    std::snprintf(foot,
                  sizeof(foot),
                  "%-30s  %-5s  %7.3fs  %9.1f%%\n",
                  "GPU device time total",
                  "[gpu]",
                  gpu_total_s,
                  pct(gpu_total_s));
    out << foot;
    out << std::string(WIDTH, '=') << "\n\n";
}
