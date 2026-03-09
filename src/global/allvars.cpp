#include "allvars.h"
#include "../io/input.h"
#include "../io/output.h"

#if defined(CPU_DEBUG) && !defined(USE_OPENMP)
int threadId;
#endif
// structs for input, output and IC handling
InputHandler input;
ICData icData;
OutputHandler output;
double buff = (1./100.) * 4; // this has to be changed later, buff will have to be dynamical given mesh resolution? (idk like calc max dist between points somehow or so... or sth better :D)
double _gamma_ = 5./3.;