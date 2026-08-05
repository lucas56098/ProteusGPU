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

// Wide "big" tier for the CPU fallback only (see BigConvexCell in voronoi/cell.h).
//
// _MAX_P_/_MAX_T_ above are bounded by the 8-bit index types the GPU tiers use: plane ids
// live in VERT_TYPE's uchar components and 255 is reserved as the "no such plane" sentinel,
// so _MAX_P_ <= 255, and Euler (V = 2F - 4) then caps a cell at ~129 faces. That is not a
// limit on how many faces a cell HAS -- it is a limit on how many plane slots the
// incremental construction burns, because a plane that clips the intermediate polytope
// keeps its slot even after a later, closer plane erases every face it contributed.
// A seed stranded in an evacuated cavity produces a long spike whose security radius is
// only reached after ~1e4 neighbours, burning far more slots than its 47 real faces.
//
// The big tier uses 32-bit indices so the CPU fallback can build such a cell exactly.
// Keep the pair Euler-consistent: _BIG_MAX_T_ >= 2 * _BIG_MAX_P_ - 4, or triangle overflow
// fires before plane overflow and the extra plane capacity is unreachable.
#ifndef _BIG_MAX_P_
#define _BIG_MAX_P_ 1024
#endif
#ifndef _BIG_MAX_T_
#define _BIG_MAX_T_ 2048
#endif

// size-equalizing mesh drift (VOL_REGULARIZE): fraction of the local signal speed the
// drift may use when a cell is smaller than the reference size
#ifndef _VOL_SHAPING_SPEED_
#define _VOL_SHAPING_SPEED_ 0.7
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

    // minimum specific internal energy (code units) enforced on the hydro update. 0 disables it.
    // Set from the cooling temperature floor (T_floor / C_T) when COOLING is compiled in.
    double min_egy_spec = 0.0;

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
constexpr double VolShapingSpeed   = (double)_VOL_SHAPING_SPEED_;
constexpr double PI                = 3.14159265358979323846;

#endif // GLOBALS_H
