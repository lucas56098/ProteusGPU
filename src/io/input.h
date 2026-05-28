#ifndef INPUT_H
#define INPUT_H

#include "hdf5.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// structs to hold IC data read from HDF5 file
struct ICHeader {
    int dimension;
};

// snapshot header metadata returned by readSnapshotFile (used for restart bookkeeping)
struct SnapshotMeta {
    double t_sim    = 0.0; // simulation time at write
    int    step     = 0;   // step counter at write
    int    n_global = 0;   // total cell count across all ranks
    int    nranks   = 0;   // ranks the snapshot was written with
    int    rank     = 0;   // which rank wrote this file
};

struct ICData {
    std::vector<double>  pos;      // dimension * numSeeds
    std::vector<hsize_t> pos_dims; // [numSeeds, dimension]

    // hydro quantities
    std::vector<double> rho;    // numSeeds
    std::vector<double> vel;    // dimension * numSeeds
    std::vector<double> energy; // numSeeds

    // global cell ID, populated by proteus_mpi::distribute_ic_local at startup
    std::vector<uint64_t> global_id;

    ICHeader header;
};

// Input handler class for reading parameters and initial conditions
class InputHandler {
  private:
    std::map<std::string, std::string> parameters;
    std::string                        paramFilePath;

    // helper functions
    std::string trim(const std::string& str);

  public:
    InputHandler(const std::string& filename = "/ics/param.txt");

    // load parameters from parameter file
    bool loadParameters();

    // access parameters
    std::string getParameter(const std::string& key) const;
    double      getParameterDouble(const std::string& key) const;

    // read initial conditions from a HDF5 file
    bool readICFile(const std::string& filename, ICData& icData);

    // read a snapshot file into ICData (for restart); populates meta with header bookkeeping.
    // Requires the restart metadata attrs (n_global, nranks, rank) to be present.
    bool readSnapshotFile(const std::string& filename, ICData& icData, SnapshotMeta& meta);

    // find the latest snapshot N in a directory; matches "snapshot_N.hdf5" when nranks==1,
    // or "snapshot_N.<rank>.hdf5" otherwise. Returns N or -1 if none found.
    static int findLatestSnapshot(const std::string& dir, int nranks, int rank);
};

#endif // INPUT_H
