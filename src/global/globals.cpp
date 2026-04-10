#include "globals.h"
#include "../io/input.h"
#include "../io/output.h"

#if defined(CPU_DEBUG) && !defined(USE_OPENMP)
int threadId;
#endif

InputHandler  input;
ICData        icData;
OutputHandler output;
double        buff      = 0.5; // will be reduced once IC loaded
double        gamma_eos = 5. / 3.;

double CellShapingSpeed  = 0.5;
double CellShapingFactor = 1.0;
