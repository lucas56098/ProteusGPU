#ifndef INPUT_H
#define INPUT_H

#include "hdf5.h"
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

    hsize_t n_seeds  = 0;
    int64_t n_global = 0;
};

struct ICData {
    std::vector<double> pos;    // dimension * n_seeds
    std::vector<double> rho;    // n_seeds
    std::vector<double> vel;    // dimension * n_seeds
    std::vector<double> energy; // n_seeds

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
    // by Profiler::SeedFromCumulative to restore in-memory state on restart.
    std::unordered_map<std::string, double> profiler_cum;
};

// loads parameter file, ICs, snapshots
class InputHandler {
  public:
    // read from parameter file
    bool loadParameters(const std::string& filename);

    std::string getParameter(const std::string& key) const;
    double      getParameterDouble(const std::string& key) const;
    bool        hasParameter(const std::string& key) const; // for optional params (the getters throw)

    // load ic
    bool readICFile(const std::string& filename, ICData& icData);

    // peek IC header + total particle count without reading the bulk arrays (serial, sub-kB).
    // Used so begrun can size the decomposition before the field read in load_IC_fields.
    bool readICHeader(const std::string& filename, ICHeader& header, hsize_t& n_total);

#ifdef USE_MPI
    // collective parallel-HDF5 read of rows [row_lo, row_lo + n_local) for pos/vel/rho/energy.
    // Every rank in MPI_COMM_WORLD must call with identical filename. Fills icData with this
    // rank's chunk only; global IDs assigned as row_lo + i (input-order).
    bool readICChunkParallel(const std::string& filename, ICData& icData, hsize_t row_lo, hsize_t n_local);
#endif

    // load snapshot
    bool       readSnapshotFile(const std::string& filename, ICData& icData, SnapshotHeader& snap);
    static int findLatestSnapshot(const std::string& dir, int nranks, int rank);

  private:
    std::map<std::string, std::string> parameters;
    std::string                        paramFilePath;

    // helper
    std::string trim(const std::string& str);
};

#endif // INPUT_H
