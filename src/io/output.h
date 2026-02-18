#ifndef OUTPUT_H
#define OUTPUT_H

#include <string>
#include <vector>
#include "../global/allvars.h"

#ifdef USE_HDF5
#include "hdf5.h"

// has to be improved once we are at a point that there actually is something to store...

// structs to prepare mesh data for writing to HDF5 file
struct MeshHeader {
    int dimension = DIMENSION;
    double extent;
    int n;
    int k;
    int nmax;
    int seed;
    bool store_edge_coords = false;
};

struct MeshFaceData {
    std::vector<int> neighbor_cell;
    std::vector<double> normal;        // numFaces x dimension
    std::vector<hsize_t> normal_dims;  // [numFaces, dimension]
    std::vector<double> area;
    std::vector<double> edge_coords;   // all edge coords concatenated
    std::vector<hsize_t> edge_coords_dims;  // [totalVertices, dimension]
    std::vector<int> edge_coords_offsets;   // Number of vertices per face
};

struct MeshCellData {
    MeshHeader header;
    std::vector<int> cell_ids;
    std::vector<double> seeds;         // numCells x dimension
    std::vector<hsize_t> seeds_dims;   // [numCells, dimension]
    std::vector<double> volumes;
    MeshFaceData faces;
};
#endif

// output handler class for writing mesh files
class OutputHandler {
private:
    std::string outputDirectory;

public:
    OutputHandler(const std::string& outputDir = "./output/");

    bool initialize(); // initalize output directory
    std::string getOutputDirectory() const { return outputDirectory; }

#ifdef USE_HDF5
    // write mesh data to HDF5 file
    bool writeMeshFile(const std::string& filename, const MeshCellData& meshData);
#endif
};

#endif // OUTPUT_H
