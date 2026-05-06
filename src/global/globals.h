#ifndef GLOBALS_H
#define GLOBALS_H
#pragma once

// Defaults for Config.sh compile-time constants. Any value set in
// Config.sh overrides these.

// voronoi compile time settings
#ifdef dim_2D
#ifndef _K_
#define _K_ 35
#endif
#ifndef _MAX_P_
#define _MAX_P_ 30
#endif
#ifndef _MAX_T_
#define _MAX_T_ 60
#endif
#ifndef _FAST_K_
#define _FAST_K_ 15
#endif
#ifndef _FAST_MAX_P_
#define _FAST_MAX_P_ 20
#endif
#ifndef _FAST_MAX_T_
#define _FAST_MAX_T_ 20
#endif
#ifndef _FACE_CAPACITY_MULT_
#define _FACE_CAPACITY_MULT_ 8
#endif
#else
#ifndef _K_
#define _K_ 190
#endif
#ifndef _MAX_P_
#define _MAX_P_ 50
#endif
#ifndef _MAX_T_
#define _MAX_T_ 96
#endif
#ifndef _FAST_K_
#define _FAST_K_ 35
#endif
#ifndef _FAST_MAX_P_
#define _FAST_MAX_P_ 30
#endif
#ifndef _FAST_MAX_T_
#define _FAST_MAX_T_ 60
#endif
#ifndef _FACE_CAPACITY_MULT_
#define _FACE_CAPACITY_MULT_ 17
#endif
#endif

// GPU kernel block sizes
#ifndef _VORO_BLOCK_SIZE_
#define _VORO_BLOCK_SIZE_ 64
#endif
#ifndef _KNN_BLOCK_SIZE_
#define _KNN_BLOCK_SIZE_ 256
#endif
#ifndef _GRAD_BLOCK_SIZE_
#define _GRAD_BLOCK_SIZE_ 256
#endif
#ifndef _HYDRO_BLOCK_SIZE_
#define _HYDRO_BLOCK_SIZE_ 256
#endif
#ifndef _MESH_BLOCK_SIZE_
#define _MESH_BLOCK_SIZE_ 256
#endif

// hydro / mesh
#ifndef _GAMMA_EOS_
#define _GAMMA_EOS_ 5./3.
#endif
#ifndef _CELL_SHAPING_SPEED_
#define _CELL_SHAPING_SPEED_ 0.7
#endif
#ifndef _CELL_SHAPING_FACTOR_
#define _CELL_SHAPING_FACTOR_ 0.2
#endif

// forward declarations for IO types
class InputHandler;
struct ICData;
class OutputHandler;

// global simulation state
extern InputHandler  input;
extern ICData        icData;
extern OutputHandler output;
extern double        buff; // buffer for the periodic bc (box will be 1 + 2*buff long)

// compile-time physics constants
constexpr double gamma_eos         = (double)_GAMMA_EOS_;
constexpr double CellShapingSpeed  = (double)_CELL_SHAPING_SPEED_;
constexpr double CellShapingFactor = (double)_CELL_SHAPING_FACTOR_;

#endif // GLOBALS_H
