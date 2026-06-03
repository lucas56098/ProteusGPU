#ifndef INPUT_H
#define INPUT_H

#include "hdf5.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// hold IC data
struct ICHeader {
    int dimension;
};

struct ICData {
    std::vector<double>  pos;      // dimension * numSeeds
    std::vector<hsize_t> pos_dims; // [numSeeds, dimension]

    // hydro quantities
    std::vector<double> rho;    // numSeeds
    std::vector<double> vel;    // dimension * numSeeds
    std::vector<double> energy; // numSeeds

    // global cell ID, set in proteus_mpi::distribute_ic_local
    std::vector<uint64_t> global_id;

    ICHeader header;
};

// snapshot header data for restarting
struct SnapshotHeader {
    double t_sim    = 0.0; // simulation time at write
    int    step     = 0;   // step counter at write
    int    n_global = 0;   // total cell count
    int    nranks   = 0;   // ranks the snapshot was written with
    int    rank     = 0;   // which rank wrote this file
};

// loads parameter file, ICs, snapshots
class InputHandler {
  public:
    InputHandler(const std::string& filename = "/ics/param.txt");

    // read from parameter file
    bool loadParameters();

    std::string getParameter(const std::string& key) const;
    double      getParameterDouble(const std::string& key) const;

    // load ic
    bool readICFile(const std::string& filename, ICData& icData);

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
