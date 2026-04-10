#ifndef GLOBALS_H
#define GLOBALS_H
#pragma once

// forward declarations for IO types
class InputHandler;
struct ICData;
class OutputHandler;

// global simulation state
extern InputHandler  input;
extern ICData        icData;
extern OutputHandler output;
extern double        buff;              // buffer for the periodic bc (box will then be 1 + 2*buff long)
extern double        gamma_eos;         // ideal gas constant
extern double        CellShapingSpeed;  // regularization speed fraction
extern double        CellShapingFactor; // regularization threshold in units of cell radius

#if defined(CPU_DEBUG) && !defined(USE_OPENMP)
extern int threadId;
#endif

#endif // GLOBALS_H
