#include "allvars.h"
#include "../io/input.h"
#include "../io/output.h"

#ifdef CPU_DEBUG
int3 blockId;
int3 threadId;
#endif
// structs for input, output and IC handling
InputHandler input;
ICData icData;
OutputHandler output;
double buff = 0.1;