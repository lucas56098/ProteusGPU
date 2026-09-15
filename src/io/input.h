#ifndef INPUT_H
#define INPUT_H

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// hold IC data
struct ICHeader {
    std::string ic_filename;
    bool        restart_flag = false;
    int         dimension    = 0;

    uint64_t n_seeds  = 0;
    int64_t  n_global = 0;

    int knn_N_grid = 0;

#ifdef USE_MPI
    std::vector<int> decomp_splits[3];
#endif
};

struct ICData {
    std::vector<double> pos;    // dimension * n_seeds
    std::vector<double> rho;    // n_seeds
    std::vector<double> vel;    // dimension * n_seeds
    std::vector<double> energy; // n_seeds

#ifdef MOVING_MESH
    std::vector<double> v_mesh; // dimension * n_seeds
#endif

    // global cell ID
    std::vector<uint64_t> global_id;

    ICHeader header;
};

// snapshot header data for restarting
struct SnapshotHeader {
    double  t_sim    = 0.0; // simulation time at write
    int     step     = 0;   // step counter at write
    int64_t n_global = 0;   // total cell count (int64: 2000^3 already overflows int32)
    int     nranks   = 0;   // ranks the snapshot was written with
    int     rank     = 0;   // which rank wrote this file
    // /header/profiler attrs — per-rank cumulative seconds at snapshot time, used
    // by Profiler::seed_from_cumulative to restore in-memory state on restart.
    std::unordered_map<std::string, double> profiler_cum;
};

// loads parameter file, ICs, snapshots
class InputHandler {
  public:
    // read from parameter file
    bool load_parameters(const std::string& filename);

    std::string get_parameter(const std::string& key) const;
    double      get_parameter_double(const std::string& key) const;
    bool        has_parameter(const std::string& key) const; // for optional params (the getters throw)

    // load ic
    bool read_ic_file(const std::string& filename, ICData& ic_data);

    // peek IC header + total particle count without reading the bulk arrays (serial, sub-kB).
    // Used so begrun can size the decomposition before the field read in load_IC_fields.
    bool read_ic_header(const std::string& filename, ICHeader& header, uint64_t& n_total);

#ifdef USE_MPI
    // collective parallel-HDF5 read of rows [row_lo, row_lo + n_local) for pos/vel/rho/energy.
    // Every rank in MPI_COMM_WORLD must call with identical filename. Fills ic_data with this
    // rank's chunk only; global IDs assigned as row_lo + i (input-order).
    bool read_ic_chunk_parallel(const std::string& filename, ICData& ic_data, uint64_t row_lo, uint64_t n_local);
#endif

    // load snapshot
    bool       read_snapshot_file(const std::string& filename, ICData& ic_data, SnapshotHeader& snap);
    static int find_latest_snapshot(const std::string& dir, int nranks, int rank);

  private:
    std::map<std::string, std::string> parameters;
    std::string                        param_file_path;

    // helper
    std::string trim(const std::string& str);
};

#endif // INPUT_H
