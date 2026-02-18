#ifndef INPUT_H
#define INPUT_H

#include <string>
#include <map>
#include <vector>

#ifdef USE_HDF5
#include "hdf5.h"
// structs to hoald IC data read from HDF5 file
struct ICHeader {
    int dimension; // 2D or 3D
    double extent; // box size (for now assume cubic)
};

struct ICData {
    std::vector<double> seedpos;  // dimension * numSeeds
    std::vector<hsize_t> seedpos_dims;  // [numSeeds, dimension]
    ICHeader header;
};
#endif

// Input handler class for reading parameters and initial conditions
class InputHandler {
private:
    std::map<std::string, std::string> parameters;
    std::string paramFilePath;

    // helper functions
    std::string trim(const std::string& str);

public:
    InputHandler(const std::string& filename = "param.txt");

    // load parameters from parameter file
    bool loadParameters();

    // access parameters
    std::string getParameter(const std::string& key) const;
    int getParameterInt(const std::string& key) const;
    double getParameterDouble(const std::string& key) const;
    bool getParameterBool(const std::string& key) const;

#ifdef USE_HDF5
    // read initial conditions from a HDF5 file
    bool readICFile(const std::string& filename, ICData& icData);
#endif
};

#endif // INPUT_H
