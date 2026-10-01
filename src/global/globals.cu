// defines the global singletons declared in globals.h and units.h

#include "../io/input.h"
#include "../io/output.h"
#include "globals.h"

InputHandler  input;
ICData        ic_data;
OutputHandler output;
SimState      sim = {};
Units         units;