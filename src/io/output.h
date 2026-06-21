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
    void write_snapshot();
};

// runtime printout
void print_log();

#endif // OUTPUT_H
