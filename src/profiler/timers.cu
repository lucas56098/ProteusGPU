// scopes, the TOTAL timer and the totals per path (included by profiler.cu)

namespace {

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

    // parent path plus this name
    static std::string build_full_path(const char* short_name) {
        if (s_path_stack.empty()) return std::string(short_name);
        return s_path_stack.back() + "." + short_name;
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
