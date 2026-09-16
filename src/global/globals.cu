// defines the global singletons declared in globals.h and units.h

#include "../io/input.h"
#include "../io/output.h"
#include "globals.h"

InputHandler  input;
ICData        ic_data;
OutputHandler output;
SimState      sim  = {};
double        buff = 0.5; // ghost band width, begrun sets it from the global cell count
Units         units;