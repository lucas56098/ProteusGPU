#ifndef BEGRUN_H
#define BEGRUN_H

// Config.sh define checks
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

    // setup/end simulation run
    void begrun(int argc, char* argv[]);
    void endrun();

} // namespace begrun

#endif // BEGRUN_H
