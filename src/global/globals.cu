#include "../io/input.h"
#include "../io/output.h"
#include "globals.h"

InputHandler  input;
ICData        icData;
OutputHandler output;
SimState      sim  = {};
double        buff = 0.5; // will be reduced once IC loaded