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

    // ---------- Path stack + per-timer state ----------------------------------

    // Thread-local scope stack. All current call sites fire from the main thread;
    // OpenMP regions inside a timed scope don't push their own timers. Keeping it
    // thread_local just future-proofs nested-thread use without giving up
    // hierarchy on the main path.
    thread_local std::vector<std::string> s_path_stack;

    // CPU/MPI: microseconds. GPU: microseconds derived from cudaEventElapsedTime.
    // All timers share this map; the kind tag tells you what the number means.
    std::unordered_map<std::string, long long> s_cum_us;

    // 'c' = cpu, 'm' = mpi, 'g' = gpu. Set on first Start; later Starts under the
    // same path don't downgrade an already-tagged timer.
    std::unordered_map<std::string, char> s_kind;

    std::unordered_map<std::string, long long> s_restart_baseline;

    // Live start times for currently-open scopes. CollectCurrent uses these to
    // extend long-running timers (TOTAL, HYDRO) to "now" each step.
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> s_live_start;

    // ---------- GPU event pool + pending queue --------------------------------

#ifdef CUDA_PROFILING
    // Recycled cudaEvent_t — created lazily, never destroyed (life of process).
    std::vector<cudaEvent_t> s_event_pool;

    struct GpuPending {
        cudaEvent_t start;
        cudaEvent_t stop;
    };
    // Per-timer queue of un-queried event pairs. Drained non-blocking at every
    // LogTimestep and force-synced once at end-of-run.
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

    // ---------- HDF5 state ----------------------------------------------------

    hid_t s_file       = -1;    // open on every rank
    bool  s_log_active = false; // true on every rank while profiling is open
    int   s_my_rank    = 0;
    int   s_nranks     = 1;

    hid_t   s_per_step = -1; // /per_step    [step, rank, timer]
    hid_t   s_cum      = -1; // /cumulative  [step, rank, timer]
    hid_t   s_names    = -1; // /timer_names [timer]
    hid_t   s_kinds    = -1; // /timer_kinds [timer]
    hsize_t s_rows     = 0;  // step extent of the two tables

    // The timer axis, identical on every rank: a timer's index is its row in /timer_names.
    std::unordered_map<std::string, size_t> s_timer_index;

    // This rank's cumulative microseconds at the previous LogTimestep, by timer index.
    std::vector<long long> s_prev_cum;

    // This rank's timer names and kinds as of the last time any rank's list changed (sorted by
    // name), and each one's index on the timer axis.
    std::vector<std::string> s_sent_names;
    std::vector<char>        s_sent_kinds;
    std::vector<size_t>      s_sent_slots;

    // ---------- Helpers -------------------------------------------------------

    // full_path = (stack top) + "." + short_name; top-level if stack empty.
    static std::string build_full_path(const char* short_name) {
        if (s_path_stack.empty()) return std::string(short_name);
        return s_path_stack.back() + "." + short_name;
    }

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

// ============================================================
// RAII scopes
// ============================================================

Profiler::Scope::Scope(const char* short_name) {
    m_path = build_full_path(short_name);
    s_path_stack.push_back(m_path);
    if (s_kind.find(m_path) == s_kind.end()) s_kind[m_path] = 'c';
    s_live_start[m_path] = std::chrono::high_resolution_clock::now();
#ifdef CUDA_PROFILING
    nvtxRangePushA(m_path.c_str());
#endif
}

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

Profiler::MpiScope::MpiScope(const char* short_name) {
    m_path = build_full_path(short_name);
    s_path_stack.push_back(m_path);
    s_kind[m_path]       = 'm';
    s_live_start[m_path] = std::chrono::high_resolution_clock::now();
#ifdef CUDA_PROFILING
    nvtxRangePushA(m_path.c_str());
#endif
}

Profiler::MpiScope::~MpiScope() {
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

Profiler::KernelScope::KernelScope(const char* short_name) {
    m_path = build_full_path(short_name);
    s_path_stack.push_back(m_path);
#ifdef CPU_DEBUG
    // CPU build: this region runs the equivalent CPU-side code, so report it as
    // cpu work timed by chrono. No GPU exists.
    s_kind[m_path]       = 'c';
    s_live_start[m_path] = std::chrono::high_resolution_clock::now();
#else
    // GPU build: this region brackets a CUDA kernel launch. CUDA_PROFILING
    // additionally records device-side events for accurate kernel timing.
    s_kind[m_path] = 'g';
#ifdef CUDA_PROFILING
    nvtxRangePushA(m_path.c_str());
    cudaEvent_t e = acquire_event();
    cudaEventRecord(e, 0);
    m_start_event = (void*)e;
#endif
#endif
}

Profiler::KernelScope::~KernelScope() {
#ifdef CPU_DEBUG
    const auto end = std::chrono::high_resolution_clock::now();
    auto       it  = s_live_start.find(m_path);
    if (it != s_live_start.end()) {
        s_cum_us[m_path] += std::chrono::duration_cast<std::chrono::microseconds>(end - it->second).count();
        s_live_start.erase(it);
    }
#else
#ifdef CUDA_PROFILING
    cudaEvent_t stop = acquire_event();
    cudaEventRecord(stop, 0);
    s_pending_gpu[m_path].push_back({(cudaEvent_t)m_start_event, stop});
    nvtxRangePop();
#endif
#endif
    if (!s_path_stack.empty()) s_path_stack.pop_back();
}

// ============================================================
// TOTAL root timer
// ============================================================

// TOTAL spans begrun() through endrun(), so it can't be a stack-RAII scope.
// Held on the heap here; StopTotalTimer destructs it, accumulating the final time.
static std::unique_ptr<Profiler::Scope> s_total_scope;

void Profiler::StartTotalTimer() {
    s_total_scope  = std::make_unique<Scope>("TOTAL");
    sim.wall_start = std::chrono::steady_clock::now(); // session wall clock for ETA + final runtime
}

void Profiler::StopTotalTimer() {
    s_total_scope.reset();
}

// cumulative TOTAL seconds for this rank (folds the live offset if still open).
// Includes runtime resumed from a restart, since SeedFromCumulative rewinds TOTAL's start.
double Profiler::TotalSeconds() {
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

// ============================================================
// GPU drain
// ============================================================

void Profiler::DrainGpuEvents(bool force_sync) {
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

// ============================================================
// CollectCurrent / CurrentCumulative
// ============================================================

std::vector<std::pair<std::string, long long>> Profiler::CollectCurrent() {
    DrainGpuEvents(/*force_sync=*/false);
    const auto endTime = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<std::string, long long>> rows;
    rows.reserve(s_cum_us.size() + s_live_start.size());

    // every accumulated timer, plus live offset if it's currently open
    for (const auto& kv : s_cum_us) {
        long long us   = kv.second;
        auto      live = s_live_start.find(kv.first);
        if (live != s_live_start.end()) {
            us += std::chrono::duration_cast<std::chrono::microseconds>(endTime - live->second).count();
        }
        rows.emplace_back(kv.first, us);
    }
    // open scopes that haven't accumulated anything yet
    for (const auto& kv : s_live_start) {
        if (s_cum_us.find(kv.first) != s_cum_us.end()) continue;
        long long us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - kv.second).count();
        rows.emplace_back(kv.first, us);
    }
    return rows;
}

std::unordered_map<std::string, double> Profiler::CurrentCumulative() {
    auto                                    rows = CollectCurrent();
    std::unordered_map<std::string, double> out;
    for (const auto& r : rows)
        out[r.first] = r.second / 1e6;
    return out;
}

// ============================================================
// SeedFromCumulative — restart resume
// ============================================================

void Profiler::SeedFromCumulative(const std::unordered_map<std::string, double>& cum_sec) {
    for (const auto& kv : cum_sec) {
        const long long us = (long long)(kv.second * 1e6 + 0.5);
        // For TOTAL we rewind its live start time below rather than seeding the
        // cumulative — otherwise we'd double-count once the live offset kicks in.
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

// ============================================================
// PrintResults — tree dump with cross-rank stats
// ============================================================

namespace {
#ifdef USE_MPI
    // Pack/unpack helpers reused for the end-of-run all-ranks name catalogue.
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
    // Every rank's buffer, concatenated in rank order. Rank r's bytes are [displs[r], displs[r] + lens[r]).
    std::vector<char>
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

    // Tree node for the printable output.
    struct TreeNode {
        std::string              full_path;
        std::string              leaf;
        char                     kind      = 'c';
        double                   cum_sum_s = 0.0; // summed across ranks — the headline number
        double                   imbalance = 1.0; // max / avg across ranks
        std::vector<std::string> children;        // full paths
    };

    // Split "A.B.C" → parent="A.B", leaf="C". Top-level: parent="", leaf=path.
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

    // Column widths chosen so the header titles ("time", "percentage", "imbalance") fit
    // exactly above their data columns. Tag column is fixed-width so [mpi]/[gpu] line up
    // and cpu entries get the same indentation.
    //
    //   <name 30>  <tag 5>  <time 8>  <pct 10>  <imbal 9>
    //   total: 30+2+5+2+8+2+10+2+9 = 70 chars.
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

        // children, sorted by summed time desc
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

void Profiler::PrintResults() {
    // 1) make sure GPU times are fully accounted for
    DrainGpuEvents(/*force_sync=*/true);

    const int nranks = proteus_mpi::nranks();
    const int rank   = proteus_mpi::rank();

    // 2) build the union name set across ranks
    std::vector<std::string> my_names;
    my_names.reserve(s_cum_us.size());
    for (const auto& kv : s_cum_us)
        my_names.push_back(kv.first);

    auto                  all_names = allgather_timer_names(my_names, nranks);
    std::set<std::string> union_names;
    for (const auto& v : all_names)
        for (const auto& n : v)
            union_names.insert(n);

    // 3) per-rank cum vectors aligned to the union order
    std::vector<std::string> ordered(union_names.begin(), union_names.end());
    const int                ntimers = (int)ordered.size();
    std::vector<double>      my_cum(ntimers, 0.0);
    {
        auto                                    live = CollectCurrent(); // live values, not just frozen s_cum_us
        std::unordered_map<std::string, double> mine;
        for (const auto& r : live)
            mine[r.first] = r.second / 1e6;
        for (int i = 0; i < ntimers; i++) {
            auto it = mine.find(ordered[i]);
            if (it != mine.end()) my_cum[i] = it->second;
        }
    }

    // 4) cross-rank sum + max reductions. The headline number is the sum over all
    // ranks; max feeds the imbalance = max / avg column (avg = sum / nranks).
    std::vector<double> cum_sum = my_cum, cum_max = my_cum;
#ifdef USE_MPI
    if (nranks > 1 && ntimers > 0) {
        MPI_Allreduce(MPI_IN_PLACE, cum_sum.data(), ntimers, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, cum_max.data(), ntimers, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
#endif

    // 5) gather kind tags from every rank too — a timer may exist on only some
    // ranks, so an Allreduce over a tiny byte-per-timer buffer is the easy way
    // to fill in a consistent kind label.
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

    if (rank != 0) return;

    // 6) build the tree (rank 0 only — values are the cross-rank sum / max)
    std::unordered_map<std::string, TreeNode> nodes;
    for (int i = 0; i < ntimers; i++) {
        TreeNode n;
        n.full_path = ordered[i];
        std::string parent;
        split_path(n.full_path, parent, n.leaf);
        n.kind             = out_kind[i] ? out_kind[i] : 'c';
        n.cum_sum_s        = cum_sum[i];
        const double avg   = cum_sum[i] / (double)nranks;
        n.imbalance        = (avg > 0.0) ? cum_max[i] / avg : 1.0;
        nodes[n.full_path] = n;
    }
    // wire parent → children
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
                roots.push_back(kv.first); // orphan — print at top level
        }
    }

    // 7) print
    std::ostream&     out   = logging::root();
    constexpr int     WIDTH = 70;
    const std::string title = " Profiling Results ";
    const int         side  = (WIDTH - (int)title.size()) / 2;
    const int         rside = WIDTH - side - (int)title.size();
    out << "\n" << std::string(side, '=') << title << std::string(rside, '=') << "\n";

    // "sum of N ranks (T threads, G GPUs)" sits in the name/tag area; column titles to its right.
    const int total_threads = nranks * logging::omp_threads();
#ifdef CPU_DEBUG
    const int total_gpus = 0;
#else
    const int total_gpus = nranks; // one GPU per rank
#endif
    char left[64];
    std::snprintf(left, sizeof(left), "sum of %d ranks (%d threads, %d GPUs)", nranks, total_threads, total_gpus);
    char hdr[256];
    std::snprintf(hdr, sizeof(hdr), "%-37s  %8s  %10s  %9s\n", left, "time", "percentage", "imbalance");
    out << hdr;
    out << std::string(WIDTH, '-') << "\n";

    // TOTAL anchors the % column.
    double total_s = 0.0;
    auto   it_tot  = nodes.find("TOTAL");
    if (it_tot != nodes.end()) total_s = it_tot->second.cum_sum_s;

    // sort roots by summed time desc
    std::sort(roots.begin(), roots.end(), [&](const std::string& a, const std::string& b) {
        return nodes[a].cum_sum_s > nodes[b].cum_sum_s;
    });
    for (const auto& r : roots)
        print_subtree(out, nodes, r, 0, total_s);

    // 8) total time spent in [mpi] / [gpu] leaves (summed across ranks)
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

// ============================================================
// HDF5 logging
// ============================================================

namespace {

    constexpr hsize_t PROFILE_STEP_CHUNK  = 16;  // rows per chunk
    constexpr hsize_t PROFILE_TIMER_CHUNK = 256; // timers per chunk
    constexpr size_t  PROFILE_NAME_LEN    = 256; // bytes per stored timer name
    constexpr size_t  PROFILE_KIND_LEN    = 3;   // "cpu" | "mpi" | "gpu"

    // True when the log is shared through MPI-IO. A single rank keeps the default driver.
    bool parallel_log() {
#ifdef USE_MPI
        return s_nranks > 1;
#else
        return false;
#endif
    }

    h5::Type fixed_string(size_t len) {
        h5::Type t(H5Tcopy(H5T_C_S1));
        H5Tset_size(t, len);
        H5Tset_strpad(t, H5T_STR_NULLPAD);
        return t;
    }

    // [step, rank, timer]. A chunk spans all ranks, so every rank's block lands in the same chunk
    // and all ranks touch the same chunk index entries.

    hid_t create_table(const char* name) {
        hsize_t   dims[3]  = {0, (hsize_t)s_nranks, 0};
        hsize_t   max[3]   = {H5S_UNLIMITED, (hsize_t)s_nranks, H5S_UNLIMITED};
        hsize_t   chunk[3] = {PROFILE_STEP_CHUNK, (hsize_t)s_nranks, PROFILE_TIMER_CHUNK};
        h5::Space space(H5Screate_simple(3, dims, max));
        h5::Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
        H5Pset_chunk(dcpl, 3, chunk);
        if (parallel_log()) H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
        return H5Dcreate(s_file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }

    hid_t create_list(const char* name, size_t len) {
        hsize_t   dims = 0, max = H5S_UNLIMITED, chunk = PROFILE_TIMER_CHUNK;
        h5::Space space(H5Screate_simple(1, &dims, &max));
        h5::Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
        H5Pset_chunk(dcpl, 1, &chunk);
        h5::Type type = fixed_string(len);
        return H5Dcreate(s_file, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }

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

    void resize_tables(hsize_t rows, hsize_t ntimers) {
        hsize_t dims[3] = {rows, (hsize_t)s_nranks, ntimers};
        H5Dset_extent(s_per_step, dims);
        H5Dset_extent(s_cum, dims);
    }

    void close_datasets() {
        for (hid_t* d : {&s_per_step, &s_cum, &s_names, &s_kinds}) {
            if (*d >= 0) H5Dclose(*d);
            *d = -1;
        }
    }

    // A timer's first cumulative value to diff against: its snapshot value on a restart, else 0.
    void register_timer(const std::string& name) {
        auto it             = s_restart_baseline.find(name);
        s_timer_index[name] = s_prev_cum.size();
        s_prev_cum.push_back(it != s_restart_baseline.end() ? it->second : 0);
    }

    // New timers go to the end of the timer axis, so no existing index ever moves.
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

    // Every rank learns every rank's timers and appends the unknown ones, all ranks in the same sorted
    // order. Kind rule: the first rank (in rank order) with a non-cpu kind wins.
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

    // One rank's row of one table: [step, my_rank, 0..ntimers).
    void write_block(hid_t dset, int step, const std::vector<double>& values) {
        if (values.empty()) return;
        h5::Space fspace(H5Dget_space(dset));
        hsize_t   start[3] = {(hsize_t)step, (hsize_t)s_my_rank, 0};
        hsize_t   count[3] = {1, 1, values.size()};
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        h5::Space mspace(H5Screate_simple(3, count, NULL));
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, values.data());
    }

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
        s_rows = (hsize_t)restart_step;
        resize_tables(s_rows, names.size());
        return true;
    }

} // namespace

void Profiler::OpenProfileLog(const std::string& path, int restart_step) {
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

    // so a rank that aborts before the first LogTimestep still leaves a readable file
    H5Fflush(s_file, H5F_SCOPE_GLOBAL);
}

void Profiler::CloseProfileLog() {
    if (!s_log_active) return;
    s_log_active = false;
    if (s_file < 0) return;
    close_datasets();
    H5Fclose(s_file);
    s_file = -1;
}

void Profiler::AbortProfileLog() {

    if (parallel_log()) {
        s_log_active = false;
        return;
    }
    CloseProfileLog();
}

void Profiler::LogTimestep(int step) {
    if (!s_log_active) return;

    // sorted by name, so the list only compares unequal to last step's when the timers changed
    auto rows = CollectCurrent();
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

    int changed = (my_names != s _sent_names || my_kinds != s_sent_kinds) ? 1 : 0;
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

    // This rank's block covers every timer on the axis, 0 for one it does not have.
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

    // flush each step so an MPI_Abort on any rank still leaves a readable file
    H5Fflush(s_file, H5F_SCOPE_GLOBAL);
}

#endif // ENABLE_PROFILING

// ============================================================
// Peak memory dump (unchanged)
// ============================================================

void print_max_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {

        double rssBytes = 0.0;
#if defined(__APPLE__) && defined(__MACH__)
        rssBytes = static_cast<double>(usage.ru_maxrss);
#elif defined(__linux__)
        rssBytes = static_cast<double>(usage.ru_maxrss) * 1024.0;
#else
        rssBytes = static_cast<double>(usage.ru_maxrss);
#endif

        const long   pages    = sysconf(_SC_PHYS_PAGES);
        const long   pageSize = sysconf(_SC_PAGE_SIZE);
        const double totalRam = (pages > 0 && pageSize > 0) ? (double)pages * (double)pageSize : 0.0;

        constexpr double MiB    = 1024.0 * 1024.0;
        const double     rssMiB = rssBytes / MiB;
        const char*      tag    = proteus_mpi::nranks() > 1 ? " (rank 0)" : "";
        logging::root() << "MAIN: maximum CPU memory used" << tag << ": " << rssMiB << " MiB (" << totalRam / MiB
                        << " MiB total)" << std::endl;
    } else {
        std::cerr << "Error getting resource usage." << std::endl;
    }

#ifndef CPU_DEBUG
    size_t gpu_free  = 0;
    size_t gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    constexpr double MiB     = 1024.0 * 1024.0;
    const double     peakMiB = (double)g_gpu_bytes_peak() / MiB;
    const char*      tag     = proteus_mpi::nranks() > 1 ? " (rank 0)" : "";
    logging::root() << "MAIN: maximum GPU memory used" << tag << ": " << peakMiB << " MiB (" << (double)gpu_total / MiB
                    << " MiB total)" << std::endl;
#endif
}
