#ifndef GLOBALS_H
#define GLOBALS_H
#pragma once
#include "log.h"
#include <chrono>
#include <cstddef>

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
#ifndef _MPI_PACK_BLOCK_SIZE_
#define _MPI_PACK_BLOCK_SIZE_ 256
#endif

// hydro / mesh
#ifndef _GAMMA_EOS_
#define _GAMMA_EOS_ 5. / 3.
#endif
#ifndef _CELL_SHAPING_SPEED_
#define _CELL_SHAPING_SPEED_ 0.7
#endif
#ifndef _CELL_SHAPING_FACTOR_
#define _CELL_SHAPING_FACTOR_ 0.2
#endif

// forward declarations
class InputHandler;
struct ICData;
class OutputHandler;
struct VMesh;
namespace hydro {
    struct primvars;
}
namespace gradients {
    struct PrimGradients;
}

// everything that lives across the main loop
struct SimState {
    // hydro + mesh
    size_t                    n_hydro;
    hydro::primvars*          primvar;  // current state (rho, v, E)
    hydro::primvars*          prim_new; // swap target each step
    gradients::PrimGradients* grads;    // per-step gradient scratch
    VMesh*                    mesh;
    double*                   dt;

    // running state
    double t_sim    = 0.0;
    int    snap_num = 0;
    int    step     = 0;
    double t_nextoutput;

    // run config
    double t_start;
    double t_end;
    double CFL;
    double output_dt;
    int    rebalance_interval;
    int    imbalance_log_interval;
    double imbalance_threshold;

    // wall-clock start; per-step profile log lives in profile.hdf5 (see Profiler::OpenProfileLog)
    std::chrono::steady_clock::time_point wall_start;
};

// globals
extern InputHandler  input;
extern ICData        icData;
extern OutputHandler output;
extern SimState      sim;
extern double        buff; // buffer for the periodic bc (box will be 1 + 2*buff long)

// compile-time physics constants
constexpr double gamma_eos         = (double)_GAMMA_EOS_;
constexpr double CellShapingSpeed  = (double)_CELL_SHAPING_SPEED_;
constexpr double CellShapingFactor = (double)_CELL_SHAPING_FACTOR_;
constexpr double PI                = 3.14159265358979323846;

#endif // GLOBALS_H
