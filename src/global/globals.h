#ifndef GLOBALS_H
#define GLOBALS_H
#pragma once

// Compile-time capacities and constants, plus the state that lives across the main loop.
#include "log.h"
#include <chrono>
#include <cstddef>

// Voronoi capacities per tier, separate for 2D and 3D; each one can be set in Config.sh
// _K_ neighbours searched, _MAX_P_ clip planes, _MAX_T_ cell vertices, _FAST_* the fast tier,
// _FACE_CAPACITY_MULT_ faces budgeted per cell
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

// wide tier of the CPU fallback
#ifndef _BIG_MAX_P_
#define _BIG_MAX_P_ 1024
#endif
#ifndef _BIG_MAX_T_
#define _BIG_MAX_T_ 2048
#endif

// share of the signal speed used to even out cell volumes (VOL_REGULARIZE)
#ifndef _VOL_SHAPING_SPEED_
#define _VOL_SHAPING_SPEED_ 0.7
#endif

// CUDA block sizes per kernel family
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

// adiabatic index of the ideal gas
#ifndef _GAMMA_EOS_
#define _GAMMA_EOS_ 5. / 3.
#endif
// Lloyd push: share of the sound speed, and how far the seed may sit from the centroid before it acts
#ifndef _CELL_SHAPING_SPEED_
#define _CELL_SHAPING_SPEED_ 0.7
#endif
#ifndef _CELL_SHAPING_FACTOR_
#define _CELL_SHAPING_FACTOR_ 0.2
#endif

#if !defined(ASTRO_PHYSICS) &&                                                                                         \
    (defined(NFW) || defined(HERNQUIST) || defined(SMBH) || defined(COOLING) || defined(SF_FEEDBACK) ||                \
     defined(AGN_THERMAL) || defined(AGN_KINETIC) || defined(LIMITERS))
#error "Config.sh: an astro sub-flag is set without ASTRO_PHYSICS."
#endif

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
    size_t                    n_hydro;  // local cells
    hydro::primvars*          primvar;  // current state
    hydro::primvars*          prim_new; // the step writes here, the pointers swap at the end
    gradients::PrimGradients* grads;
    VMesh*                    mesh;
    double*                   dt; // managed, written by calc_timestep

    double t_sim    = 0.0;
    int    snap_num = 0;
    int    step     = 0;
    double t_nextoutput; // time of the next snapshot

    double t_start; // time this run started at, used for the ETA
    double t_end;
    double CFL;
    double output_dt;
    // only read under USE_MPI
    int    rebalance_interval;
    int    imbalance_log_interval;
    double imbalance_threshold;

    double min_egy_spec = 0.0; // temperature floor as specific energy, set by cooling

    std::chrono::steady_clock::time_point wall_start; // wall clock at the start of the run
};

// defined in globals.cu
extern InputHandler  input;
extern ICData        ic_data;
extern OutputHandler output;
extern SimState      sim;
extern double        buff;

// run-wide constants from the values above
constexpr double gamma_eos         = (double)_GAMMA_EOS_;
constexpr double CellShapingSpeed  = (double)_CELL_SHAPING_SPEED_;
constexpr double CellShapingFactor = (double)_CELL_SHAPING_FACTOR_;
constexpr double VolShapingSpeed   = (double)_VOL_SHAPING_SPEED_;
constexpr double PI                = 3.14159265358979323846;

#endif
