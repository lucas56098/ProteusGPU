#ifndef OUTPUT_H
#define OUTPUT_H

#include "../global/allvars.h"
#include "hdf5.h"
#include <chrono>
#include <string>

struct VMesh;

// output handler class for writing snapshot files
class OutputHandler {
  private:
    std::string outputDirectory;

  public:
    OutputHandler(const std::string& outputDir = "./output/");

    bool        initialize(); // initalize output directory
    std::string getOutputDirectory() const { return outputDirectory; }

    // write snapshot (mesh + hydro) to HDF5. n_global/nranks/rank are baked into the header
    // so a same-decomposition restart can validate and skip IC redistribution.
    void snapshot(int snap_num, VMesh* mesh, const hydro::primvars* primvar, double t_sim, int step);
};

void print_log(int                                   step,
               std::chrono::steady_clock::time_point wall,
               double                                t_sim,
               double                                dt,
               double                                t_start,
               double                                t_end);

#endif // OUTPUT_H
