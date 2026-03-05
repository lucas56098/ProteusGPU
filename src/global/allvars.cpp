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
double buff = (1./100.) * 4; // this has to be changed later, buff will have to be dynamical given mesh resolution? (idk like calc max dist between points somehow or so... or sth better :D)
double gamma = 5./3.;