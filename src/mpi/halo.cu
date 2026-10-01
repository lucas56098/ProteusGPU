// ghost cells by request (halo.h)

#include "halo.h"

#include "decomp.h"
#include "global/allvars.h"
#include "global/structs.h"
#include "gradients/gradients.h"
#include "hydro/finite_volume_solver.h"
#include "knn/knn.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

#include "halo_packing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace proteus_mpi {

    MpiHalo halo                = {};
    int     n_mpi_capacity      = 0;
    int     n_local_initial_max = 0;
    double  alloc_growth        = 0.0;

    // clang-format off
    // one translation unit, so the include order matters
    #include "halo_init.cu"
    #include "halo_requests.cu"
    #include "halo_exchange.cu"
    // clang-format on

} // namespace proteus_mpi
