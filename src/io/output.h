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

    bool        initialize();
    std::string getOutputDirectory() const { return outputDirectory; }

    // write snapshot
    void snapshot(int snap_num, VMesh* mesh, const hydro::primvars* primvar, double t_sim, int step);
};

// runtime printout
void print_log(
    int step, std::chrono::steady_clock::time_point wall, double t_sim, double dt, double t_start, double t_end);

#endif // OUTPUT_H
