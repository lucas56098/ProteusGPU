#include "../global/gpu_compat.h"
#include "../global/log.h"
#include "../mpi/mpi_compat.h"
#include "profiler.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>

// CPU wall-clock profiling (chrono-based, works in all modes)
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> Profiler::m_StartTimes;
std::unordered_map<std::string, long long>                                      Profiler::m_Timings;
std::unordered_map<std::string, long long>                                      Profiler::m_PrevStepCum;

// GPU event timing storage
std::unordered_map<std::string, double> Profiler::m_GpuTimings;
std::unordered_map<std::string, int>    Profiler::m_GpuCounts;
std::unordered_map<std::string, double> Profiler::m_GpuPrevStepCum;

std::unordered_map<std::string, Profiler::GpuEventPair> Profiler::m_GpuEvents;

void Profiler::StartTimer(const std::string& name) {
    NVTX_PUSH(name.c_str());
    m_StartTimes[name] = std::chrono::high_resolution_clock::now();
}

void Profiler::EndTimer(const std::string& name) {
    auto endTime   = std::chrono::high_resolution_clock::now();
    auto startTime = m_StartTimes[name];
    m_Timings[name] += std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    NVTX_POP();
}

// GPU event-based profiling (CUDA mode only)

void Profiler::StartGPU(const std::string& name) {
#ifdef CUDA_PROFILING
    NVTX_PUSH(name.c_str());
    auto& ev = m_GpuEvents[name];
    if (!ev.start) {
        cudaEventCreate((cudaEvent_t*)&ev.start);
        cudaEventCreate((cudaEvent_t*)&ev.stop);
    }
    cudaEventRecord((cudaEvent_t)ev.start, 0);
#else
    (void)name;
#endif
}

void Profiler::EndGPU(const std::string& name) {
#ifdef CUDA_PROFILING
    auto& ev = m_GpuEvents[name];
    cudaEventRecord((cudaEvent_t)ev.stop, 0);
    cudaEventSynchronize((cudaEvent_t)ev.stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, (cudaEvent_t)ev.start, (cudaEvent_t)ev.stop);
    m_GpuTimings[name] += (double)ms;
    m_GpuCounts[name]++;
    NVTX_POP();
#else
    (void)name;
#endif
}

// print combined results — per-rank timings but only rank 0's view is surfaced
void Profiler::PrintResults() {
    std::ostream& out = logging::root();
    out << "\n=== Profiling Results (Wall Clock Time, rank 0) ===\n";
    long long totalRuntime = 0;
    long long mpiCommTime  = 0;

    std::vector<std::pair<std::string, long long>> rows;
    rows.reserve(m_Timings.size() + 1);
    for (const auto& entry : m_Timings) {
        if (entry.first.rfind("MPI_", 0) == 0 && entry.first != "MPI_TOTAL") { mpiCommTime += entry.second; }
        if (entry.first == "TOTAL_RUNTIME") { totalRuntime = entry.second; }
        rows.emplace_back(entry.first, entry.second);
    }
    if (mpiCommTime > 0) { rows.emplace_back("MPI_TOTAL", mpiCommTime); }

    // sort by cumulative time descending
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& row : rows) {
        out << "[PROFILE] " << row.first << " took " << (row.second / 1e6) << "s\n";
    }

    out << "\nTOTAL_RUNTIME: " << (totalRuntime / 1e6) << "s\n";

    // GPU kernel timing breakdown
    if (!m_GpuTimings.empty()) {
        out << "\n=== GPU Kernel Profiling (CUDA Events, rank 0) ===\n";

        // sort by cumulative time for reading
        std::vector<std::pair<std::string, double>> sorted(m_GpuTimings.begin(), m_GpuTimings.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        double gpu_total_ms = 0.0;
        for (const auto& entry : sorted) {
            gpu_total_ms += entry.second;
        }

        for (const auto& entry : sorted) {
            int    calls = m_GpuCounts[entry.first];
            double pct   = (gpu_total_ms > 0.0) ? (entry.second / gpu_total_ms * 100.0) : 0.0;
            out << "[GPU] " << entry.first << ": " << entry.second << " ms"
                << " (" << calls << " calls, " << (entry.second / calls) << " ms/call"
                << ", " << pct << "%)\n";
        }
        out << "[GPU] TOTAL: " << gpu_total_ms << " ms\n";
    }

    out << "=========================\n";
}

// write per-timestep CSV record to out (typically the profile log file) — per-rank timings but only rank 0 writes
void Profiler::LogTimestep(int step, std::ostream& out) {

    out << "# timestep = " << step << " (CPU timers) (cum time [s], diff time [s]):\n";

    out << std::fixed << std::setprecision(6);

    // collect (name, cumUs) rows, injecting live values for long-lived timers and the synthesized MPI_TOTAL
    const auto endTime = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<std::string, long long>> rows;
    rows.reserve(m_Timings.size() + 3);

    // TOTAL_RUNTIME — m_Timings is 0 until endrun, so compute live from m_StartTimes.
    // ResumeFromLog rewinds m_StartTimes["TOTAL_RUNTIME"] so the live value includes any prior offset.
    {
        auto it = m_StartTimes.find("TOTAL_RUNTIME");
        if (it != m_StartTimes.end()) {
            rows.emplace_back(
                "TOTAL_RUNTIME",
                std::chrono::duration_cast<std::chrono::microseconds>(endTime - it->second).count());
        }
    }

    // HYDRO_MAIN — long-lived (entire hydro loop). m_Timings holds the seeded-from-restart portion;
    // add the currently-running portion live.
    {
        auto it = m_StartTimes.find("HYDRO_MAIN");
        if (it != m_StartTimes.end()) {
            const long long live = std::chrono::duration_cast<std::chrono::microseconds>(endTime - it->second).count();
            auto            it_t   = m_Timings.find("HYDRO_MAIN");
            const long long seeded = (it_t != m_Timings.end()) ? it_t->second : 0;
            rows.emplace_back("HYDRO_MAIN", seeded + live);
        }
    }

    // all other regular timers
    long long mpiCum = 0;
    for (const auto& entry : m_Timings) {
        if (entry.first == "HYDRO_MAIN") continue; // handled above
        if (entry.first.rfind("MPI_", 0) == 0 && entry.first != "MPI_TOTAL") { mpiCum += entry.second; }
        rows.emplace_back(entry.first, entry.second);
    }
    if (mpiCum > 0) { rows.emplace_back("MPI_TOTAL", mpiCum); }

    // sort by cumulative time descending
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    // emit rows; update per-step cumulative state
    for (const auto& row : rows) {
        const long long cumUs    = row.second;
        const long long prev     = m_PrevStepCum[row.first];
        const double    cumSec   = cumUs / 1e6;
        const double    diffSec  = (cumUs - prev) / 1e6;
        m_PrevStepCum[row.first] = cumUs;
        out << step << ", " << std::setw(16) << std::left << row.first << ", " << cumSec << ", " << diffSec << "\n";
    }

    // GPU kernel timing breakdown
    if (!m_GpuTimings.empty()) {
        out << "\n# timestep = " << step << " (GPU kernels) (time [ms], # calls, ms/call, fraction):\n";

        // sort by cumulative time for reading
        std::vector<std::pair<std::string, double>> sorted(m_GpuTimings.begin(), m_GpuTimings.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        double gpu_total_ms = 0.0;
        for (const auto& entry : sorted) {
            gpu_total_ms += entry.second;
        }

        out << step << std::setw(33) << std::left << ", gpu_total, " << gpu_total_ms << "\n";

        for (const auto& entry : sorted) {
            int    calls = m_GpuCounts[entry.first];
            double pct   = (gpu_total_ms > 0.0) ? (entry.second / gpu_total_ms * 100.0) : 0.0;
            out << step << ", " << std::setw(33) << std::left << entry.first << ", " << entry.second << ", " << calls
                << ", " << (entry.second / calls) << ", " << pct << "\n";
        }
    }

    out << "\n\n";
}

// restart: trim `path` to blocks with timestep <= step, seed in-memory counters from that block.
// Rank 0 only (matches FileLogger). No-op if `path` doesn't exist or has no surviving block.
void Profiler::ResumeFromLog(const std::string& path, int step) {
    if (!proteus_mpi::is_root()) return;

    std::ifstream in(path);
    if (!in.is_open()) return;

    auto trim = [](std::string& s) {
        size_t a = s.find_first_not_of(" \t\r");
        size_t b = s.find_last_not_of(" \t\r");
        s        = (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
    };

    std::ostringstream                         kept_buf;
    std::ostringstream                         cur_buf;
    int                                        cur_step = -1;
    std::unordered_map<std::string, long long> cur_cum;
    std::unordered_map<std::string, long long> final_cum;

    auto flush_block = [&]() {
        if (cur_step >= 0 && cur_step <= step) {
            kept_buf << cur_buf.str();
            for (const auto& kv : cur_cum) { final_cum[kv.first] = kv.second; }
        }
        cur_buf.str("");
        cur_buf.clear();
        cur_cum.clear();
        cur_step = -1;
    };

    std::string                  line;
    static const std::string     HDR = "# timestep = ";
    while (std::getline(in, line)) {
        if (line.compare(0, HDR.size(), HDR) == 0) {
            // header: "# timestep = <N> ..." — flush previous block, start new
            flush_block();
            try {
                cur_step = std::stoi(line.substr(HDR.size()));
            } catch (...) { cur_step = -1; }
            cur_buf << line << "\n";
        } else if (cur_step >= 0 && !line.empty() && (std::isdigit((unsigned char)line[0]) || line[0] == '-')) {
            // CSV data row: "<step>, <name>, <cum_seconds>, <diff_seconds>"
            cur_buf << line << "\n";
            std::stringstream ss(line);
            std::string       tok;
            std::getline(ss, tok, ','); // step
            std::getline(ss, tok, ','); // name
            trim(tok);
            const std::string name = tok;
            std::getline(ss, tok, ','); // cum
            trim(tok);
            try {
                double cum_sec  = std::stod(tok);
                cur_cum[name]   = (long long)(cum_sec * 1e6 + 0.5);
            } catch (...) {}
        } else {
            // GPU rows / blank lines / unknown — keep verbatim inside the current block, drop otherwise
            if (cur_step >= 0) cur_buf << line << "\n";
        }
    }
    flush_block();
    in.close();

    // rewrite the file with only the kept blocks
    std::ofstream out(path, std::ios::trunc);
    out << kept_buf.str();
    out.close();

    // seed in-memory counters; skip TOTAL_RUNTIME (rewound via m_StartTimes below) and
    // MPI_TOTAL (synthesized aggregate, not a real timer in m_Timings)
    for (const auto& kv : final_cum) {
        if (kv.first != "TOTAL_RUNTIME" && kv.first != "MPI_TOTAL") { m_Timings[kv.first] = kv.second; }
        m_PrevStepCum[kv.first] = kv.second;
    }
    // TOTAL_RUNTIME: rewind its start point so (now - start) includes the resumed time.
    // EndTimer in endrun() then accumulates into m_Timings normally.
    auto it = final_cum.find("TOTAL_RUNTIME");
    if (it != final_cum.end()) {
        auto& s = m_StartTimes["TOTAL_RUNTIME"];
        s       = s - std::chrono::microseconds(it->second);
    }
}

// peak CPU RSS (high-water mark from the OS) and peak GPU memory we ever held
void print_max_memory_usage() {

    // CPU side: peak resident set size from getrusage, total system RAM from sysconf
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {

        double rssBytes = 0.0;
#if defined(__APPLE__) && defined(__MACH__)
        rssBytes = static_cast<double>(usage.ru_maxrss); // macOS reports bytes
#elif defined(__linux__)
        rssBytes = static_cast<double>(usage.ru_maxrss) * 1024.0; // Linux reports KiB
#else
        rssBytes = static_cast<double>(usage.ru_maxrss); // fallback: assume bytes
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

    // GPU side: high-water mark of bytes we requested through gpu_alloc, total from cudaMemGetInfo
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