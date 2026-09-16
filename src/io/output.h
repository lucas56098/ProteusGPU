#ifndef OUTPUT_H
#define OUTPUT_H

// Writes the snapshots and the per-step log line.

#include "../global/allvars.h"
#include <string>

// stores the output directory and writes snapshot files (one per rank)
class OutputHandler {
  private:
    std::string output_directory;

  public:
    OutputHandler(const std::string& output_dir = "./output/"); // created in globals.cu, dir set by begrun

    bool initialize();

    void write_snapshot();
};

// per-step log line
void print_log();

#endif
