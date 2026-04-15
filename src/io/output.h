#ifndef OUTPUT_H
#define OUTPUT_H

#include "../global/allvars.h"
#include <chrono>
#include <string>
#include <vector>

struct VMesh;

#ifdef USE_HDF5
#include "hdf5.h"

// has to be improved once we are at a point that there actually is something to store...

// structs to prepare mesh data for writing to HDF5 file
struct MeshHeader {
    int    dimension = DIMENSION;
    double extent;
    int    n;
    int    k;
    int    nmax;
    int    seed;
};

struct MeshCellData {
    MeshHeader           header;
    std::vector<double>  seeds;      // numCells x dimension
    std::vector<hsize_t> seeds_dims; // [numCells, dimension]
    std::vector<double>  volumes;
    std::vector<int>     face_counts; // number of faces per cell
};

struct KNNData {
    int                       num_points;
    int                       k;
    std::vector<double>       points;       // num_points x 3 (x,y,z)
    std::vector<hsize_t>      points_dims;  // [num_points, 3]
    std::vector<unsigned int> nearest;      // num_points x k
    std::vector<hsize_t>      nearest_dims; // [num_points, k]
    std::vector<unsigned int> permutation;  // num_points
};
#endif

// output handler class for writing mesh files
class OutputHandler {
  private:
    std::string outputDirectory;

  public:
    OutputHandler(const std::string& outputDir = "./output/");

    bool        initialize(); // initalize output directory
    std::string getOutputDirectory() const { return outputDirectory; }

#ifdef USE_HDF5
    // wrapper to convert mesh into meshData and then store snapshot
    void snapshot(int snap_num, VMesh* mesh, const hydro::primvars* primvar, int n_hydro, double t_sim);

    // convert VMesh (for hydro computation) to MeshCellData (for output)
    void vmesh_to_meshdata(VMesh* mesh, MeshCellData& meshData);

    // write snapshot (mesh and hydro data) to HDF5 file
    bool writeSnapshot(const std::string&     filename,
                       const MeshCellData&    meshData,
                       const hydro::primvars* primvar,
                       int                    n_hydro,
                       double                 t_sim);

#endif
};

void print_log(int                                   step,
               std::chrono::steady_clock::time_point wall,
               double                                t_sim,
               double                                dt,
               double                                t_start,
               double                                t_end,
               int*                                  next_log);

#endif // OUTPUT_H
