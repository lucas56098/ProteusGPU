#ifndef INPUT_H
#define INPUT_H

#include "hdf5.h"
#include <map>
#include <string>
#include <vector>

// structs to hold IC data read from HDF5 file
struct ICHeader {
    int dimension;
};

struct ICData {
    std::vector<double>  seedpos;      // dimension * numSeeds
    std::vector<hsize_t> seedpos_dims; // [numSeeds, dimension]

    // hydro quantities
    std::vector<double> rho;    // numSeeds
    std::vector<double> vel;    // dimension * numSeeds
    std::vector<double> Energy; // numSeeds

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

    // read a snapshot file into ICData (for restart) and return the simulation time
    bool readSnapshotFile(const std::string& filename, ICData& icData, double& t_sim);

    // find the latest snapshot_N.hdf5 in a directory, return N (or -1 if none found)
    static int findLatestSnapshot(const std::string& dir);
};

#endif // INPUT_H
