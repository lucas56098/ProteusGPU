#ifndef BEGRUN_H
#define BEGRUN_H

// Starts the run and shuts it down.

// flag combinations that must not build, caught before anything else compiles
#if (!defined(dim_3D) && !defined(dim_2D)) || (defined(dim_3D) && defined(dim_2D))
#error "Choose a dimension in Config.sh: [dim_3D] OR [dim_2D]"
#endif
#if (!defined(CUDA) && !defined(CPU_DEBUG)) || (defined(CUDA) && defined(CPU_DEBUG))
#error "Choose a backend in Config.sh: [CUDA] OR [CPU_DEBUG]"
#endif
#if defined(CUDA_PROFILING) && !defined(CUDA)
#error "CUDA_PROFILING requires CUDA in Config.sh"
#endif
#if defined(CUDA_PROFILING) && !defined(ENABLE_PROFILING)
#error "CUDA_PROFILING requires ENABLE_PROFILING in Config.sh"
#endif

namespace begrun {

    // parameters, IC or snapshot, decomposition, hydro arrays and the first mesh
    void begrun(int argc, char* argv[]);
    // frees mesh and hydro, closes the profile log, prints the totals
    void endrun();

} // namespace begrun

#endif
