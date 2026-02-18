#ifndef BEGRUN_H
#define BEGRUN_H
#include "../io/input.h"

#ifndef USE_HDF5
#error "Currently, HDF5 support is mandatory. Other formats may be added in the future. Please add USE_HDF5 to Config.sh and recompile."
#endif

namespace begrun {

void print_banner();
InputHandler loadInputFiles(int argc, char* argv[]);

} // namespace begrun

#endif // BEGRUN_H
