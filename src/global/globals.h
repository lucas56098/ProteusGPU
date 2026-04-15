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
extern double        buff; // buffer for the periodic bc (box will be 1 + 2*buff long)

// compile-time physics constants (set via -D flags from Config.sh)
constexpr double gamma_eos         = (double)_GAMMA_EOS_;
constexpr double CellShapingSpeed  = (double)_CELL_SHAPING_SPEED_;
constexpr double CellShapingFactor = (double)_CELL_SHAPING_FACTOR_;

#endif // GLOBALS_H
