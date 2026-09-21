#ifndef INPUT_H
#define INPUT_H

// Reads the parameter file, the initial conditions (IC) and restart snapshots.

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// storage for header and cell data from an IC or snapshot file; freed by begrun after first mesh build
struct ICHeader {
    std::string ic_filename;
    bool        restart_flag = false; // continue from latest snap
    int         dimension    = 0;

    uint64_t n_seeds  = 0; // cells on this rank
    int64_t  n_global = 0; // cells on all ranks

    int knn_N_grid = 0; // ngb grid size loaded from snap; 0 = estimate from cell count

#ifdef USE_MPI
    std::vector<int> decomp_splits[3]; // old domain decomp split table per axis (only loaded from snap)
#endif
};

struct ICData { // created once in global.cu
    std::vector<double> pos;
    std::vector<double> rho; // pos, vel, v_mesh have DIM x n_cells values
    std::vector<double> vel;
    std::vector<double> energy; // total energy per volume
#ifdef MOVING_MESH
    std::vector<double> v_mesh; // only used if restarting from snapshot
#endif

    ICHeader header;
};

struct SnapshotHeader {
    double                                  t_sim    = 0.0;
    int                                     step     = 0;
    int64_t                                 n_global = 0;
    int                                     nranks   = 0; // rank count the snapshot was written with
    int                                     rank     = 0; // rank that wrote this file
    std::unordered_map<std::string, double> profiler_cum; // cumulative seconds per timer path
};

// stores parameters and writes IC / snapshot into ic_data
class InputHandler {
  public:
    // store and afterwards access parameters
    bool load_parameters(const std::string& filename);

    std::string get_parameter(const std::string& key) const;
    double      get_parameter_double(const std::string& key) const;
    int         get_parameter_int(const std::string& key) const;
    bool        has_parameter(const std::string& key) const;

    void warn_unread_parameters() const;

    // load ic file into ic_data
    bool read_ic_file(const std::string& filename, ICData& ic_data);
    bool read_ic_header(const std::string& filename, ICHeader& header, uint64_t& n_total);
#ifdef USE_MPI
    bool read_ic_chunk_parallel(const std::string& filename, ICData& ic_data, uint64_t row_lo, uint64_t n_local);
#endif

    // find and load latest snapshot
    bool       read_snapshot_file(const std::string& filename, ICData& ic_data, SnapshotHeader& snap);
    static int find_latest_snapshot(const std::string& dir, int nranks, int rank);

  private:
    std::map<std::string, std::string> parameters; // key -> value
    mutable std::set<std::string>      read_keys;  // keys that have been used

    std::string param_file_path;

    std::string trim(const std::string& str);
};

#endif
