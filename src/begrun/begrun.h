#ifndef BEGRUN_H
#define BEGRUN_H
#include "../io/input.h"
#include "../io/output.h"

// Config.sh define checks
#if (!defined(dim_3D) && !defined(dim_2D)) || (defined(dim_3D) && defined(dim_2D))
#error "Choose a dimension in Config.sh: [dim_3D] OR [dim_2D]"
#endif

namespace begrun {

    // simulation start state
    struct StartState {
        double t_sim;
        int    snap_num;
    };

    // loads input parameters and IC
    StartState begrun(int argc, char* argv[]);

    // free IC data no longer needed
    void free_initial_conditions();

    // helpers
    void         print_banner();
    InputHandler load_params(int argc, char* argv[]);

} // namespace begrun

#endif // BEGRUN_H
