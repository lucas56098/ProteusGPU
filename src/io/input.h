#ifndef INPUT_H
#define INPUT_H

#include <map>
#include <string>
#include <vector>

#ifdef USE_HDF5
#include "hdf5.h"

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
#endif

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
    int         getParameterInt(const std::string& key) const;
    double      getParameterDouble(const std::string& key) const;
    bool        getParameterBool(const std::string& key) const;
    bool        hasParameter(const std::string& key) const;

#ifdef USE_HDF5
    // read initial conditions from a HDF5 file
    bool readICFile(const std::string& filename, ICData& icData);

    // read a snapshot file into ICData (for restart) and return the simulation time
    bool readSnapshotFile(const std::string& filename, ICData& icData, double& t_sim);

    // find the latest snapshot_N.hdf5 in a directory, return N (or -1 if none found)
    static int findLatestSnapshot(const std::string& dir);
#endif
};

#endif // INPUT_H
