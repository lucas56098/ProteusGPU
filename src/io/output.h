#ifndef OUTPUT_H
#define OUTPUT_H

#include "../global/allvars.h"
#include <chrono>
#include <string>

struct VMesh;

// output handler class for writing snapshot files
class OutputHandler {
  private:
    std::string output_directory;

  public:
    OutputHandler(const std::string& output_dir = "./output/");

    bool initialize();

    // write snapshot
    void write_snapshot();
};

// runtime printout
void print_log();

#endif // OUTPUT_H
