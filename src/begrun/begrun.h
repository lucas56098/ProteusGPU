#ifndef BEGRUN_H
#define BEGRUN_H

// Config.sh define checks
#if (!defined(dim_3D) && !defined(dim_2D)) || (defined(dim_3D) && defined(dim_2D))
#error "Choose a dimension in Config.sh: [dim_3D] OR [dim_2D]"
#endif

namespace begrun {

    // setup/end simulation run
    void begrun(int argc, char* argv[]);
    void endrun();

} // namespace begrun

#endif // BEGRUN_H
