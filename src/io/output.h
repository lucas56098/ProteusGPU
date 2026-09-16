#ifndef OUTPUT_H
#define OUTPUT_H

#include "../global/allvars.h"
#include <string>

// stores the output directory and writes snapshot files (one per rank)
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
