#include "../global/allvars.h"
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

// Per-rank profile log written as a single HDF5 file. Layout:
//   /rank_<N>/per_step/<FULL_PATH>     1D extensible double, index i = step i, value = diff seconds
//   /rank_<N>/cumulative/<FULL_PATH>   1D extensible double, index i = step i, value = cum seconds
// Each dataset carries a string attribute @kind = "cpu" | "mpi" | "gpu" so the
// analyzer can color/filter without reparsing names. Every rank gathers its rows to rank 0,
// which writes the file with the serial HDF5 driver. Parallel-HDF5 was used here before but
// a shared file with per-rank datasets and per-step extends desyncs HDF5's collective
// metadata cache -> "H5Dset_extent ... MPI_Bcast ... Message truncated". The data is tiny,
// so serializing through rank 0 costs less than the collective metadata traffic it replaces.
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

    // Last-cumulative-at-LogTimestep cache (for the diff-per-step column). Keyed by
    // (rank, full_path): rank 0 holds the prev value for every rank's timers, since it
    // computes the per-step diff for all ranks from the gathered cumulative values.
    std::map<std::pair<int, std::string>, long long> s_prev_step_cum;

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

    hid_t s_file       = -1;    // valid only on rank 0 (serial-driver HDF5 file)
    bool  s_log_active = false; // true on every rank while profiling is open
    int   s_my_rank    = 0;
    int   s_nranks     = 1;

    // Handles to every (rank, full_path) dataset. Only rank 0 holds these -- it writes
    // every /rank_<N>/... dataset from the rows the other ranks ship it.
    struct DSetPair {
        hid_t per_step = -1;
        hid_t cum      = -1;
    };
    std::map<std::pair<int, std::string>, DSetPair> s_dsets;
    hsize_t                                         s_current_len = 0;

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

    static void write_kind_attr(hid_t dset, char kind) {
        const char* s = kind_str(kind);
        hid_t       t = H5Tcopy(H5T_C_S1);
        H5Tset_size(t, std::strlen(s));
        H5Tset_strpad(t, H5T_STR_NULLTERM);
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr  = H5Acreate(dset, "kind", t, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, t, s);
        H5Aclose(attr);
        H5Sclose(space);
        H5Tclose(t);
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
        s_prev_step_cum[{s_my_rank, kv.first}] = us;
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

    std::vector<std::vector<std::string>> allgather_timer_names(const std::vector<std::string>& my_names, int nranks) {
        std::vector<std::vector<std::string>> result(nranks);
#ifdef USE_MPI
        if (nranks > 1) {
            std::vector<char> my_buf = pack_names(my_names);
            int               my_len = (int)my_buf.size();
            std::vector<int>  lens(nranks, 0);
            MPI_Allgather(&my_len, 1, MPI_INT, lens.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::vector<int> displs(nranks, 0);
            int              total = 0;
            for (int r = 0; r < nranks; r++) {
                displs[r] = total;
                total += lens[r];
            }
            std::vector<char> all(total);
            MPI_Allgatherv(
                my_buf.data(), my_len, MPI_BYTE, all.data(), lens.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
            for (int r = 0; r < nranks; r++) {
                result[r] = unpack_names(all.data() + displs[r], lens[r]);
            }
            return result;
        }
#endif
        result[0] = my_names;
        return result;
    }

    // Per-rank (name, kind, cum_us) lists, valid only on rank 0 after gather_rows_to_root.
    struct GatheredRows {
        std::vector<std::vector<std::string>> names; // [rank][i]
        std::vector<std::vector<char>>        kinds; // [rank][i]
        std::vector<std::vector<long long>>   vals;  // [rank][i]  cumulative microseconds
    };

#ifdef USE_MPI
    // Gather every rank's rows to rank 0, the only writer. Non-root ranks get an empty result.
    GatheredRows gather_rows_to_root(const std::vector<std::string>& my_names,
                                     const std::vector<char>&        my_kinds,
                                     const std::vector<long long>&   my_vals,
                                     int                             nranks,
                                     int                             my_rank) {
        GatheredRows g;
        const int    my_count = (int)my_names.size();

        // per-rank timer counts (also the kind/val gatherv counts)
        std::vector<int> counts(nranks, 0);
        MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // names: null-separated byte blob, variable length per rank
        std::vector<char> name_buf = pack_names(my_names);
        const int         nlen     = (int)name_buf.size();
        std::vector<int>  nlens(nranks, 0);
        MPI_Gather(&nlen, 1, MPI_INT, nlens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> ndispls(nranks, 0), kdispls(nranks, 0);
        int              ntotal = 0, ktotal = 0;
        for (int r = 0; r < nranks; r++) {
            ndispls[r] = ntotal;
            ntotal += nlens[r];
            kdispls[r] = ktotal;
            ktotal += counts[r];
        }

        std::vector<char>      name_all(my_rank == 0 ? ntotal : 0);
        std::vector<char>      kind_all(my_rank == 0 ? ktotal : 0);
        std::vector<long long> val_all(my_rank == 0 ? ktotal : 0);
        MPI_Gatherv(name_buf.data(), nlen, MPI_BYTE, name_all.data(), nlens.data(), ndispls.data(), MPI_BYTE, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(my_kinds.data(), my_count, MPI_BYTE, kind_all.data(), counts.data(), kdispls.data(), MPI_BYTE, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(my_vals.data(), my_count, MPI_LONG_LONG, val_all.data(), counts.data(), kdispls.data(),
                    MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        if (my_rank != 0) return g;

        g.names.resize(nranks);
        g.kinds.resize(nranks);
        g.vals.resize(nranks);
        for (int r = 0; r < nranks; r++) {
            g.names[r] = unpack_names(name_all.data() + ndispls[r], nlens[r]);
            g.kinds[r].assign(kind_all.begin() + kdispls[r], kind_all.begin() + kdispls[r] + counts[r]);
            g.vals[r].assign(val_all.begin() + kdispls[r], val_all.begin() + kdispls[r] + counts[r]);
        }
        return g;
    }
#endif

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

    // Create one extensible 1D double dataset inside `group`. Collective in parallel HDF5.
    hid_t create_dataset(hid_t group, const std::string& name) {
        hsize_t initial[1] = {0};
        hsize_t maxdims[1] = {H5S_UNLIMITED};
        hid_t   space      = H5Screate_simple(1, initial, maxdims);
        hid_t   plist      = H5Pcreate(H5P_DATASET_CREATE);
        hsize_t chunk[1]   = {64};
        H5Pset_chunk(plist, 1, chunk);
        hid_t dset = H5Dcreate(group, name.c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Pclose(plist);
        H5Sclose(space);
        return dset;
    }

    void write_row(hid_t dset, hsize_t row_idx, double value) {
        hid_t   fspace   = H5Dget_space(dset);
        hsize_t start[1] = {row_idx};
        hsize_t count[1] = {1};
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        hid_t mspace = H5Screate_simple(1, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, &value);
        H5Sclose(mspace);
        H5Sclose(fspace);
    }

} // namespace

void Profiler::OpenProfileLog(const std::string& path, int restart_step) {
    s_my_rank    = proteus_mpi::rank();
    s_nranks     = proteus_mpi::nranks();
    s_log_active = true;

    // Only rank 0 owns the file; every other rank ships rows to it and never touches HDF5.
    if (s_my_rank != 0) return;

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);

    if (restart_step < 0) {
        s_file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    } else {
        s_file = H5Fopen(path.c_str(), H5F_ACC_RDWR, fapl);
        if (s_file < 0) {
            s_file       = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
            restart_step = -1;
        }
    }
    H5Pclose(fapl);

    const auto rg_path = [](int r) { return "/rank_" + std::to_string(r); };
    for (int r = 0; r < s_nranks; r++) {
        const std::string rg = rg_path(r);
        if (H5Lexists(s_file, rg.c_str(), H5P_DEFAULT) <= 0) {
            hid_t g = H5Gcreate(s_file, rg.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(g);
        }
        const std::string pg = rg + "/per_step";
        const std::string cg = rg + "/cumulative";
        if (H5Lexists(s_file, pg.c_str(), H5P_DEFAULT) <= 0) {
            hid_t g = H5Gcreate(s_file, pg.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(g);
        }
        if (H5Lexists(s_file, cg.c_str(), H5P_DEFAULT) <= 0) {
            hid_t g = H5Gcreate(s_file, cg.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(g);
        }
    }

    if (restart_step >= 0) {
        // Truncate to `restart_step` rows (not restart_step+1). The snapshot was
        // taken inside iteration N AFTER hydro_step ran but BEFORE LogTimestep(N)
        // was called, so the original profile.hdf5 had rows 0..N-1 at that point.
        // Dropping row N gives the same shape, lets us read row [restart_step-1]
        // for prev_step_cum seeding (so per_step diffs telescope across the
        // restart boundary), and avoids a costly H5Dread of a stale row.
        const hsize_t target = (hsize_t)restart_step;
        for (int r = 0; r < s_nranks; r++) {
            hid_t      gp = H5Gopen(s_file, (rg_path(r) + "/per_step").c_str(), H5P_DEFAULT);
            hid_t      gc = H5Gopen(s_file, (rg_path(r) + "/cumulative").c_str(), H5P_DEFAULT);
            H5G_info_t info;
            H5Gget_info(gp, &info);
            for (hsize_t i = 0; i < info.nlinks; i++) {
                char    nbuf[512];
                ssize_t nlen =
                    H5Lget_name_by_idx(gp, ".", H5_INDEX_NAME, H5_ITER_INC, i, nbuf, sizeof(nbuf), H5P_DEFAULT);
                if (nlen <= 0) continue;
                std::string name(nbuf);
                // Truncate, then close-and-reopen the dataset handle. Without the
                // reopen, the in-memory dataspace cache still reflects the pre-
                // truncate extent in parallel mode and subsequent H5Dwrites
                // silently miss for the first ~5 rows past the truncate boundary.
                // Caught by an explicit cum[restart_step+1..] == 0 invariant.
                hid_t d_ps = H5Dopen(gp, name.c_str(), H5P_DEFAULT);
                hid_t d_c  = H5Dopen(gc, name.c_str(), H5P_DEFAULT);
                H5Dset_extent(d_ps, &target);
                H5Dset_extent(d_c, &target);
                H5Dclose(d_ps);
                H5Dclose(d_c);
                DSetPair dp;
                dp.per_step        = H5Dopen(gp, name.c_str(), H5P_DEFAULT);
                dp.cum             = H5Dopen(gc, name.c_str(), H5P_DEFAULT);
                s_dsets[{r, name}] = dp;
            }
            H5Gclose(gp);
            H5Gclose(gc);
        }
        // Flush the truncation to file before subsequent extends/writes, so
        // dataset metadata is fully persisted and not just in the local cache.
        H5Fflush(s_file, H5F_SCOPE_GLOBAL);
        s_current_len = target;

        // Seed s_prev_step_cum from the file. SeedFromCumulative set it to
        // snap.attr[name], which equals the LIVE cum at snapshot time (mid-
        // iteration N). The cumulative dataset's last row holds the cum at
        // LogTimestep(N-1), which differs by iter_N's in-progress work. The
        // per-step diff telescopes only if prev_step_cum equals dataset row N-1.
        if (restart_step > 0) {
            const hsize_t prev_idx = (hsize_t)(restart_step - 1);
            for (auto& kv : s_dsets) {
                hid_t   fs     = H5Dget_space(kv.second.cum);
                hsize_t sel[1] = {prev_idx};
                hsize_t cnt[1] = {1};
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, sel, NULL, cnt, NULL);
                hid_t  mspace = H5Screate_simple(1, cnt, NULL);
                double v      = 0.0;
                H5Dread(kv.second.cum, H5T_NATIVE_DOUBLE, mspace, fs, H5P_DEFAULT, &v);
                H5Sclose(mspace);
                H5Sclose(fs);
                s_prev_step_cum[kv.first] = (long long)(v * 1e6 + 0.5);
            }
        } else {
            // restart from t=0 snapshot: no kept rows. Diffs are computed from 0.
            for (auto& kv : s_prev_step_cum)
                kv.second = 0;
        }
    }
}

void Profiler::CloseProfileLog() {
    if (!s_log_active) return;
    s_log_active = false;
    if (s_my_rank != 0 || s_file < 0) return;
    for (auto& kv : s_dsets) {
        if (kv.second.per_step >= 0) H5Dclose(kv.second.per_step);
        if (kv.second.cum >= 0) H5Dclose(kv.second.cum);
    }
    s_dsets.clear();
    H5Fclose(s_file);
    s_file        = -1;
    s_current_len = 0;
}

void Profiler::LogTimestep(int step) {
    if (!s_log_active) return;

    auto                     rows = CollectCurrent();
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

    // Every rank reaches this each step, so the gather stays in lockstep.
    GatheredRows g;
#ifdef USE_MPI
    if (s_nranks > 1) {
        g = gather_rows_to_root(my_names, my_kinds, my_vals, s_nranks, s_my_rank);
    } else
#endif
    {
        g.names = {my_names};
        g.kinds = {my_kinds};
        g.vals  = {my_vals};
    }

    if (s_my_rank != 0 || s_file < 0) return; // only rank 0 writes the file

    // Prefer any non-'c' tag: a rank that actually fired the scope (e.g. PROFILE_MPI("WAIT"))
    // overrides ranks that defaulted to 'c', so @kind is identical across ranks.
    std::set<std::pair<int, std::string>> needed;
    std::unordered_map<std::string, char> canonical_kind;
    for (int r = 0; r < s_nranks; r++) {
        const auto& names = g.names[r];
        const auto& kinds = g.kinds[r];
        for (size_t i = 0; i < names.size(); i++) {
            needed.insert({r, names[i]});
            const char k  = (i < kinds.size()) ? kinds[i] : 'c';
            auto       it = canonical_kind.find(names[i]);
            if (it == canonical_kind.end() || it->second == 'c') canonical_kind[names[i]] = k;
        }
    }

    for (const auto& key : needed) {
        if (s_dsets.count(key)) continue;
        const int          r  = key.first;
        const std::string& n  = key.second;
        hid_t              gp = H5Gopen(s_file, ("/rank_" + std::to_string(r) + "/per_step").c_str(), H5P_DEFAULT);
        hid_t              gc = H5Gopen(s_file, ("/rank_" + std::to_string(r) + "/cumulative").c_str(), H5P_DEFAULT);
        DSetPair           dp;
        dp.per_step = create_dataset(gp, n);
        dp.cum      = create_dataset(gc, n);
        auto kit    = canonical_kind.find(n);
        char kind   = (kit != canonical_kind.end()) ? kit->second : 'c';
        write_kind_attr(dp.per_step, kind);
        write_kind_attr(dp.cum, kind);
        if (s_current_len > 0) {
            H5Dset_extent(dp.per_step, &s_current_len);
            H5Dset_extent(dp.cum, &s_current_len);
        }
        s_dsets[key] = dp;
        H5Gclose(gp);
        H5Gclose(gc);
    }

    const hsize_t target = (hsize_t)(step + 1);
    if (target > s_current_len) {
        for (auto& kv : s_dsets) {
            H5Dset_extent(kv.second.per_step, &target);
            H5Dset_extent(kv.second.cum, &target);
        }
        s_current_len = target;
    }

    const hsize_t row_idx = (hsize_t)step;
    for (int r = 0; r < s_nranks; r++) {
        const auto& names = g.names[r];
        const auto& vals  = g.vals[r];
        for (size_t i = 0; i < names.size(); i++) {
            const auto      key    = std::make_pair(r, names[i]);
            const long long cumUs  = (i < vals.size()) ? vals[i] : 0;
            const long long prev   = s_prev_step_cum[key];
            const double    cumSec = cumUs / 1e6;
            const double    diff   = (cumUs - prev) / 1e6;
            s_prev_step_cum[key]   = cumUs;
            auto it                = s_dsets.find(key);
            if (it == s_dsets.end()) continue;
            write_row(it->second.per_step, row_idx, diff);
            write_row(it->second.cum, row_idx, cumSec);
        }
    }

    // flush each step so an MPI_Abort on any rank still leaves a readable file
    H5Fflush(s_file, H5F_SCOPE_GLOBAL);
}

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
