#ifndef BEGRUN_H
#define BEGRUN_H
#include "../io/input.h"

// Config.sh define checks
#if (!defined(dim_3D) && !defined(dim_2D)) || (defined(dim_3D) && defined(dim_2D))
#error "Choose a dimension in Config.sh: [dim_3D] OR [dim_2D]"
#endif

namespace begrun {

    // setup/end simulation run
    void begrun(int argc, char* argv[]);
    void endrun();

    // helpers
    void         print_banner();
    void         log_run_mode();
    void         init_gpu();
    InputHandler load_params(int argc, char* argv[]);
    void         load_initial_conditions(int argc, char* argv[]);
    void         init_decomposition();
    void         init_hydro_and_mesh();
    void         init_run_config();
    void         free_initial_conditions();

} // namespace begrun

#endif // BEGRUN_H
