#ifndef BEGRUN_H
#define BEGRUN_H
#include "../io/input.h"
#include "../io/output.h"

// Config.sh define checks
#ifndef USE_HDF5
#error                                                                                                                 \
    "Currently, HDF5 support is mandatory. Other formats may be added in the future. Please add USE_HDF5 to Config.sh and recompile."
#endif
#if (!defined(dim_3D) && !defined(dim_2D)) || (defined(dim_3D) && defined(dim_2D))
#error "Choose a dimension in Config.sh: [dim_3D] OR [dim_2D]"
#endif

namespace begrun {

    // simulation start state returned by begrun
    struct StartState {
        double t_sim;
        int    snap_num;
    };

    // called in main: ./ProteusGPU [param.txt] [restart_flag]
    // restart_flag: 0 = fresh start (default), 1 = restart from latest snapshot
    StartState begrun(int argc, char* argv[]);
    void       free_initial_conditions();

    // helpers
    void         print_banner();
    InputHandler loadInputFiles(int argc, char* argv[]);

} // namespace begrun

#endif // BEGRUN_H
