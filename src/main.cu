/*=========================================================================
        _____           _                    _____ _____  _    _
       |  __ \         | |                  / ____|  __ \| |  | |
       | |__) | __ ___ | |_ ___ _   _ ___  | |  __| |__) | |  | |
       |  ___/ '__/ _ \| __/ _ \ | | / __| | | |_ |  ___/| |  | |
       | |   | | | (_) | ||  __/ |_| \__ \ | |__| | |    | |__| |
       |_|   |_|  \___/ \__\___|\__,_|___/  \_____|_|     \____/

       GPU-accelerated moving mesh hydrodynamics for astrophysics
===========================================================================
Version: 0.8
Authors: Lucas Schleuss (MPA Garching), Dylan Nelson (ITA Heidelberg)
===========================================================================*/

#include "integrator/integrator.h"
#include "mpi/mpi.h"
#include "timers/timers.h"
#include <cstdlib>

int main(int argc, char* argv[]) {
    mpi::init(&argc, &argv);
    {
        const timers::Scope total("TOTAL");
        integrator::run(argc, argv);
    }
    mpi::finalize();
    return EXIT_SUCCESS;
}
