// implements Profiler (profiler.h)

#include "../global/allvars.h"
#include "../io/h5.h"
#include "../mpi/mpi_compat.h"
#include "hdf5.h"
#include "profiler.h"
#include <algorithm>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>

#ifdef ENABLE_PROFILING

namespace {

    // ==========================================================
    // timers
    // ==========================================================

    thread_local std::vector<std::string> s_path_stack; // scope stack of this thread

    std::unordered_map<std::string, long long> s_cum_us; // microseconds per path, closed scopes only

    std::unordered_map<std::string, char> s_kind; // cpu, mpi or gpu per path

    std::unordered_map<std::string, long long> s_restart_baseline; // totals a restart started from

    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point>
        s_live_start; // start time of the open scopes

#ifdef CUDA_PROFILING
    // recycled events, one pair per kernel scope
    std::vector<cudaEvent_t> s_event_pool;

    struct GpuPending {
        cudaEvent_t start;
        cudaEvent_t stop;
    };
    // recorded, not read back yet
    std::unordered_map<std::string, std::deque<GpuPending>> s_pending_gpu;

    static cudaEvent_t acquire_event() {
        if (!s_event_pool.empty()) {
            cudaEvent_t e = s_event_pool.back();
            s_event_pool.pop_back();
            return e;
        }
        cudaEvent_t e;
        cudaEventCreate(&e);
        return e;
    }
    static void release_event(cudaEvent_t e) {
        s_event_pool.push_back(e);
    }
#endif

    // profile.hdf5 handles, closed by hand (see close_profile_log)
    hid_t s_file       = -1;
    bool  s_log_active = false;
    int   s_my_rank    = 0;
    int   s_nranks     = 1;

    hid_t   s_per_step = -1;
    hid_t   s_cum      = -1;
    hid_t   s_names    = -1;
    hid_t   s_kinds    = -1;
    hsize_t s_rows     = 0;

    std::unordered_map<std::string, size_t> s_timer_index; // row of a timer in the file

    std::vector<long long> s_prev_cum; // last cumulative per row, for the difference

    // what this rank reported last
    std::vector<std::string> s_sent_names;
    std::vector<char>        s_sent_kinds;
    std::vector<size_t>      s_sent_slots;

    // parent path plus this name
    static std::string build_full_path(const char* short_name) {
        if (s_path_stack.empty()) return std::string(short_name);
        return s_path_stack.back() + "." + short_name;
    }

    // kind as the 3 byte string in the file
    static const char* kind_str(char k) {
        switch (k) {
        case 'm':
            return "mpi";
        case 'g':
            return "gpu";
        default:
            return "cpu";
        }
    }

} // namespace

// start a timer, push the path; mpi wins over cpu
Profiler::Scope::Scope(const char* short_name, Kind kind) {
    m_path = build_full_path(short_name);
    s_path_stack.push_back(m_path);
    if (kind == Kind::mpi)
        s_kind[m_path] = 'm';
    else if (s_kind.find(m_path) == s_kind.end())
        s_kind[m_path] = 'c';
    s_live_start[m_path] = std::chrono::high_resolution_clock::now();
#ifdef CUDA_PROFILING
    nvtxRangePushA(m_path.c_str());
#endif
}

// add the elapsed time, pop the path
Profiler::Scope::~Scope() {
    const auto end = std::chrono::high_resolution_clock::now();
    auto       it  = s_live_start.find(m_path);
    if (it != s_live_start.end()) {
        s_cum_us[m_path] += std::chrono::duration_cast<std::chrono::microseconds>(end - it->second).count();
        s_live_start.erase(it);
    }
#ifdef CUDA_PROFILING
    nvtxRangePop();
#endif
    if (!s_path_stack.empty()) s_path_stack.pop_back();
}

// record the start event, no wall clock
Profiler::KernelScope::KernelScope(const char* short_name) {
    m_path = build_full_path(short_name);
    s_path_stack.push_back(m_path);
    s_kind[m_path] = 'g';
#ifdef CUDA_PROFILING
    nvtxRangePushA(m_path.c_str());
    cudaEvent_t e = acquire_event();
    cudaEventRecord(e, 0);
    m_start_event = (void*)e;
#endif
}

// record the stop event, read back later
Profiler::KernelScope::~KernelScope() {
#ifdef CUDA_PROFILING
    cudaEvent_t stop = acquire_event();
    cudaEventRecord(stop, 0);
    s_pending_gpu[m_path].push_back({(cudaEvent_t)m_start_event, stop});
    nvtxRangePop();
#endif
    if (!s_path_stack.empty()) s_path_stack.pop_back();
}

// TOTAL is a normal scope, open all run
static std::unique_ptr<Profiler::Scope> s_total_scope;

void Profiler::start_total_timer() {
    s_total_scope  = std::make_unique<Scope>("TOTAL");
    sim.wall_start = std::chrono::steady_clock::now();
}

void Profiler::stop_total_timer() {
    s_total_scope.reset();
}

double Profiler::total_seconds() {
    long long us = 0;
    auto      it = s_cum_us.find("TOTAL");
    if (it != s_cum_us.end()) us = it->second;
    auto live = s_live_start.find("TOTAL");
    if (live != s_live_start.end()) {
        us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() -
                                                                    live->second)
                  .count();
    }
    return us / 1e6;
}

// add up finished pairs, force_sync waits for the rest
void Profiler::drain_gpu_events(bool force_sync) {
#ifdef CUDA_PROFILING
    for (auto& kv : s_pending_gpu) {
        auto& q = kv.second;
        while (!q.empty()) {
            auto& pend = q.front();
            if (force_sync) {
                cudaEventSynchronize(pend.stop);
            } else {
                if (cudaEventQuery(pend.stop) != cudaSuccess) break;
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, pend.start, pend.stop);
            s_cum_us[kv.first] += (long long)((double)ms * 1000.0 + 0.5);
            release_event(pend.start);
            release_event(pend.stop);
            q.pop_front();
        }
    }
#else
    (void)force_sync;
#endif
}

// totals of every path, open scopes included
std::vector<std::pair<std::string, long long>> Profiler::collect_current() {
    drain_gpu_events(false);
    const auto end_time = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<std::string, long long>> rows;
    rows.reserve(s_cum_us.size() + s_live_start.size());

    for (const auto& kv : s_cum_us) {
        long long us   = kv.second;
        auto      live = s_live_start.find(kv.first);
        if (live != s_live_start.end()) {
            us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - live->second).count();
        }
        rows.emplace_back(kv.first, us);
    }
    for (const auto& kv : s_live_start) {
        if (s_cum_us.find(kv.first) != s_cum_us.end()) continue;
        long long us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - kv.second).count();
        rows.emplace_back(kv.first, us);
    }
    return rows;
}

// totals in seconds, as a snapshot stores them
std::unordered_map<std::string, double> Profiler::current_cumulative() {
    auto                                    rows = collect_current();
    std::unordered_map<std::string, double> out;
    for (const auto& r : rows)
        out[r.first] = r.second / 1e6;
    return out;
}

// restart: old totals back, TOTAL start shifted
void Profiler::seed_from_cumulative(const std::unordered_map<std::string, double>& cum_sec) {
    for (const auto& kv : cum_sec) {
        const long long us = (long long)(kv.second * 1e6 + 0.5);
        if (kv.first != "TOTAL") s_cum_us[kv.first] = us;
        s_restart_baseline[kv.first] = us;
    }
    auto it_cum = cum_sec.find("TOTAL");
    auto it_st  = s_live_start.find("TOTAL");
    if (it_cum != cum_sec.end() && it_st != s_live_start.end()) {
        const long long us = (long long)(it_cum->second * 1e6 + 0.5);
        it_st->second -= std::chrono::microseconds(us);
    }
}

namespace {
#ifdef USE_MPI
    // ==========================================================
    // report
    // ==========================================================

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

namespace {

    // ==========================================================
    // profile.hdf5
    // ==========================================================

    // chunk [16 steps, nranks, 256 timers], name 256 bytes
    constexpr hsize_t PROFILE_STEP_CHUNK  = 16;
    constexpr hsize_t PROFILE_TIMER_CHUNK = 256;
    constexpr size_t  PROFILE_NAME_LEN    = 256;
    constexpr size_t  PROFILE_KIND_LEN    = 3;

    // more than one rank: MPI-IO
    bool parallel_log() {
#ifdef USE_MPI
        return s_nranks > 1;
#else
        return false;
#endif
    }

    // string type of the two name lists
    h5::Type fixed_string(size_t len) {
        h5::Type t(H5Tcopy(H5T_C_S1));
        H5Tset_size(t, len);
        H5Tset_strpad(t, H5T_STR_NULLPAD);
        return t;
    }

    // [step, rank, timer] table, grows in both
    hid_t create_table(const char* name) {
        hsize_t   dims[3]  = {0, (hsize_t)s_nranks, 0};
        hsize_t   max[3]   = {H5S_UNLIMITED, (hsize_t)s_nranks, H5S_UNLIMITED};
        hsize_t   chunk[3] = {PROFILE_STEP_CHUNK, (hsize_t)s_nranks, PROFILE_TIMER_CHUNK};
        h5::Space space(H5Screate_simple(3, dims, max));
        h5::Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
        H5Pset_chunk(dcpl, 3, chunk);
        // the default fill would make all ranks zero the chunk first
        if (parallel_log()) H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
        return H5Dcreate(s_file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }

    // growing list of fixed length strings
    hid_t create_list(const char* name, size_t len) {
        hsize_t   dims = 0, max = H5S_UNLIMITED, chunk = PROFILE_TIMER_CHUNK;
        h5::Space space(H5Screate_simple(1, &dims, &max));
        h5::Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
        H5Pset_chunk(dcpl, 1, &chunk);
        h5::Type type = fixed_string(len);
        return H5Dcreate(s_file, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }

    // reads names or kinds back on a restart
    std::vector<std::string> read_list(hid_t dset, size_t len) {
        std::vector<std::string> out;
        h5::Space                space(H5Dget_space(dset));
        const hssize_t           n = H5Sget_simple_extent_npoints(space);
        if (n <= 0) return out;
        std::vector<char> buf((size_t)n * len);
        h5::Type          type = fixed_string(len);
        if (H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) < 0) return out;
        for (hssize_t i = 0; i < n; i++) {
            const char* s = buf.data() + (size_t)i * len;
            out.emplace_back(s, strnlen(s, len));
        }
        return out;
    }

    // both tables keep the same shape
    void resize_tables(hsize_t rows, hsize_t ntimers) {
        hsize_t dims[3] = {rows, (hsize_t)s_nranks, ntimers};
        H5Dset_extent(s_per_step, dims);
        H5Dset_extent(s_cum, dims);
    }

    // closed by hand, like the file
    void close_datasets() {
        for (hid_t* d : {&s_per_step, &s_cum, &s_names, &s_kinds}) {
            if (*d >= 0) H5Dclose(*d);
            *d = -1;
        }
    }

    // give a timer its row and starting value
    void register_timer(const std::string& name) {
        auto it             = s_restart_baseline.find(name);
        s_timer_index[name] = s_prev_cum.size();
        s_prev_cum.push_back(it != s_restart_baseline.end() ? it->second : 0);
    }

    // appended, so a row never moves, restart included
    void append_timers(const std::vector<std::string>& names, const std::vector<char>& kinds) {
        const hsize_t old_n = s_prev_cum.size();
        const hsize_t add   = names.size();
        const hsize_t total = old_n + add;

        std::vector<char> nbuf(add * PROFILE_NAME_LEN, '\0');
        std::vector<char> kbuf(add * PROFILE_KIND_LEN, '\0');
        for (size_t i = 0; i < add; i++) {
            std::memcpy(&nbuf[i * PROFILE_NAME_LEN], names[i].data(), names[i].size());
            std::memcpy(&kbuf[i * PROFILE_KIND_LEN], kind_str(kinds[i]), PROFILE_KIND_LEN);
        }

        h5::Plist dxpl(H5Pcreate(H5P_DATASET_XFER));
#ifdef USE_MPI
        if (parallel_log()) H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
#endif
        for (int k = 0; k < 2; k++) {
            const hid_t  dset = k ? s_kinds : s_names;
            const size_t len  = k ? PROFILE_KIND_LEN : PROFILE_NAME_LEN;
            H5Dset_extent(dset, &total);
            h5::Space fspace(H5Dget_space(dset));
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &old_n, NULL, &add, NULL);
            h5::Space mspace(H5Screate_simple(1, &add, NULL));
            h5::Type  type = fixed_string(len);
            H5Dwrite(dset, type, mspace, fspace, dxpl, k ? kbuf.data() : nbuf.data());
        }
        resize_tables(s_rows, total);
        for (const auto& n : names)
            register_timer(n);
    }

    // append every name the file does not have yet
    void add_new_timers(const std::vector<std::string>& my_names, const std::vector<char>& my_kinds) {
        const auto all_names = allgather_timer_names(my_names, s_nranks);
        const auto all_kinds = allgather_timer_kinds(my_kinds, s_nranks);

        std::map<std::string, char> fresh;
        for (int r = 0; r < s_nranks; r++) {
            for (size_t i = 0; i < all_names[r].size(); i++) {
                const std::string& n = all_names[r][i];
                if (s_timer_index.count(n)) continue;
                const char k  = (i < all_kinds[r].size()) ? all_kinds[r][i] : 'c';
                auto       it = fresh.find(n);
                if (it == fresh.end() || it->second == 'c') fresh[n] = k;
            }
        }
        if (fresh.empty()) return;

        std::vector<std::string> names;
        std::vector<char>        kinds;
        for (const auto& kv : fresh) {
            if (kv.first.size() > PROFILE_NAME_LEN) {
                proteus_mpi::exit_failure(
                    "PROFILER: timer name longer than %zu bytes: %s\n", PROFILE_NAME_LEN, kv.first.c_str());
            }
            names.push_back(kv.first);
            kinds.push_back(kv.second);
        }
        append_timers(names, kinds);
    }

    // this rank's [step, rank] block of one table
    void write_block(hid_t dset, int step, const std::vector<double>& values) {
        if (values.empty()) return;
        h5::Space fspace(H5Dget_space(dset));
        hsize_t   start[3] = {(hsize_t)step, (hsize_t)s_my_rank, 0};
        hsize_t   count[3] = {1, 1, values.size()};
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        h5::Space mspace(H5Screate_simple(3, count, NULL));
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, values.data());
    }

    // continue an existing log if layout and rank count match
    bool open_existing_log(int restart_step) {
        if (H5Lexists(s_file, "timer_names", H5P_DEFAULT) <= 0 || H5Lexists(s_file, "timer_kinds", H5P_DEFAULT) <= 0 ||
            H5Lexists(s_file, "per_step", H5P_DEFAULT) <= 0 || H5Lexists(s_file, "cumulative", H5P_DEFAULT) <= 0) {
            return false;
        }
        s_names    = H5Dopen(s_file, "timer_names", H5P_DEFAULT);
        s_kinds    = H5Dopen(s_file, "timer_kinds", H5P_DEFAULT);
        s_per_step = H5Dopen(s_file, "per_step", H5P_DEFAULT);
        s_cum      = H5Dopen(s_file, "cumulative", H5P_DEFAULT);
        bool ok    = s_names >= 0 && s_kinds >= 0 && s_per_step >= 0 && s_cum >= 0;

        hsize_t dims[3] = {0, 0, 0}, dims_cum[3] = {0, 0, 0};
        if (ok) {
            h5::Space sp(H5Dget_space(s_per_step));
            h5::Space sc(H5Dget_space(s_cum));
            ok = H5Sget_simple_extent_ndims(sp) == 3 && H5Sget_simple_extent_ndims(sc) == 3;
            if (ok) {
                H5Sget_simple_extent_dims(sp, dims, NULL);
                H5Sget_simple_extent_dims(sc, dims_cum, NULL);
            }
        }
        std::vector<std::string> names;
        if (ok) {
            names            = read_list(s_names, PROFILE_NAME_LEN);
            const auto kinds = read_list(s_kinds, PROFILE_KIND_LEN);
            ok = dims[1] == (hsize_t)s_nranks && dims[2] == names.size() && kinds.size() == names.size() &&
                 dims_cum[1] == dims[1] && dims_cum[2] == dims[2];
        }
        if (!ok) {
            close_datasets();
            return false;
        }

        for (const auto& n : names)
            register_timer(n);
        // cut the tables back to the resumed step
        s_rows = (hsize_t)restart_step;
        resize_tables(s_rows, names.size());
        return true;
    }

} // namespace

// continue on a restart, else a new file; MPI-IO from two ranks up
void Profiler::open_profile_log(const std::string& path, int restart_step) {
    s_my_rank    = proteus_mpi::rank();
    s_nranks     = proteus_mpi::nranks();
    s_log_active = true;

    h5::Plist fapl(H5Pcreate(H5P_FILE_ACCESS));
#ifdef USE_MPI
    if (parallel_log() && H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL) < 0) {
        proteus_mpi::exit_failure("PROFILER: could not select the MPI-IO driver for %s\n", path.c_str());
    }
#endif

    if (restart_step >= 0) {
        s_file = H5Fopen(path.c_str(), H5F_ACC_RDWR, fapl);
        if (s_file >= 0 && !open_existing_log(restart_step)) {
            logging::root() << "PROFILER: " << path << " has another layout or rank count. Starting a new log."
                            << std::endl;
            H5Fclose(s_file);
            s_file = -1;
        }
    }
    if (s_file < 0) {
        s_file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        if (s_file < 0) {
            logging::root() << "PROFILER: could not create " << path << ". Running without a profile log." << std::endl;
            s_log_active = false;
            return;
        }
        s_per_step = create_table("per_step");
        s_cum      = create_table("cumulative");
        s_names    = create_list("timer_names", PROFILE_NAME_LEN);
        s_kinds    = create_list("timer_kinds", PROFILE_KIND_LEN);
        s_rows     = 0;
    }

    H5Fflush(s_file, H5F_SCOPE_GLOBAL);
}

// closes the log at the end of the run
// by hand: a destructor would run after HDF5's atexit handler
void Profiler::close_profile_log() {
    if (!s_log_active) return;
    s_log_active = false;
    if (s_file < 0) return;
    close_datasets();
    H5Fclose(s_file);
    s_file = -1;
}

// error path: no HDF5 under MPI, H5Fclose is collective and would block
void Profiler::abort_profile_log() {

    if (parallel_log()) {
        s_log_active = false;
        return;
    }
    close_profile_log();
}

// one row per step: totals and the difference to the row before
void Profiler::log_timestep(int step) {
    if (!s_log_active) return;

    auto rows = collect_current();
    // sorted, so all ranks propose new timers in the same order
    std::sort(rows.begin(),
              rows.end(),
              [](const std::pair<std::string, long long>& a, const std::pair<std::string, long long>& b) {
                  return a.first < b.first;
              });
    std::vector<std::string> my_names;
    std::vector<char>        my_kinds;
    std::vector<long long>   my_vals;
    my_names.reserve(rows.size());
    my_kinds.reserve(rows.size());
    my_vals.reserve(rows.size());
    for (const auto& r : rows) {
        my_names.push_back(r.first);
        auto it = s_kind.find(r.first);
        my_kinds.push_back(it != s_kind.end() ? it->second : 'c');
        my_vals.push_back(r.second);
    }

    // growing the tables is collective, so all ranks must agree
    int changed = (my_names != s_sent_names || my_kinds != s_sent_kinds) ? 1 : 0;
#ifdef USE_MPI
    if (parallel_log()) MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (changed) {
        add_new_timers(my_names, my_kinds);
        s_sent_names = my_names;
        s_sent_kinds = my_kinds;
        s_sent_slots.resize(my_names.size());
        for (size_t j = 0; j < my_names.size(); j++)
            s_sent_slots[j] = s_timer_index[my_names[j]];
    }

    const size_t ntimers = s_prev_cum.size();
    if ((hsize_t)step + 1 > s_rows) {
        s_rows = (hsize_t)step + 1;
        resize_tables(s_rows, ntimers);
    }

    std::vector<long long> cum_us(ntimers, 0);
    for (size_t j = 0; j < my_vals.size(); j++)
        cum_us[s_sent_slots[j]] = my_vals[j];

    std::vector<double> per_step(ntimers), cumulative(ntimers);
    for (size_t i = 0; i < ntimers; i++) {
        per_step[i]   = (cum_us[i] - s_prev_cum[i]) / 1e6;
        cumulative[i] = cum_us[i] / 1e6;
        s_prev_cum[i] = cum_us[i];
    }
    write_block(s_per_step, step, per_step);
    write_block(s_cum, step, cumulative);

    H5Fflush(s_file, H5F_SCOPE_GLOBAL);
}

#endif

// peak memory of rank 0, GPU bytes in a CUDA build
void print_max_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {

        double rss_bytes = 0.0;
// ru_maxrss is bytes on macOS and kilobytes on linux
#if defined(__APPLE__) && defined(__MACH__)
        rss_bytes = static_cast<double>(usage.ru_maxrss);
#elif defined(__linux__)
        rss_bytes = static_cast<double>(usage.ru_maxrss) * 1024.0;
#else
        rss_bytes = static_cast<double>(usage.ru_maxrss);
#endif

        const long   pages     = sysconf(_SC_PHYS_PAGES);
        const long   page_size = sysconf(_SC_PAGE_SIZE);
        const double total_ram = (pages > 0 && page_size > 0) ? (double)pages * (double)page_size : 0.0;

        constexpr double MiB     = 1024.0 * 1024.0;
        const double     rss_mib = rss_bytes / MiB;
        const char*      tag     = proteus_mpi::nranks() > 1 ? " (rank 0)" : "";
        logging::root() << "MAIN: maximum CPU memory used" << tag << ": " << rss_mib << " MiB (" << total_ram / MiB
                        << " MiB total)" << std::endl;
    } else {
        std::cerr << "Error getting resource usage." << std::endl;
    }

#ifndef CPU_DEBUG
    size_t gpu_free  = 0;
    size_t gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    constexpr double MiB      = 1024.0 * 1024.0;
    const double     peak_mib = (double)g_gpu_bytes_peak() / MiB;
    const char*      tag      = proteus_mpi::nranks() > 1 ? " (rank 0)" : "";
    logging::root() << "MAIN: maximum GPU memory used" << tag << ": " << peak_mib << " MiB (" << (double)gpu_total / MiB
                    << " MiB total)" << std::endl;
#endif
}
