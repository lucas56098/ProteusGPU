#include "allvars.h"
#include "../io/input.h"
#include "../io/output.h"

#if defined(CPU_DEBUG) && !defined(USE_OPENMP)
int threadId;
#endif
// structs for input, output and IC handling
InputHandler  input;
ICData        icData;
OutputHandler output;
double        buff    = 0.5; // will be reduced once IC loaded
double        _gamma_ = 5. / 3.;