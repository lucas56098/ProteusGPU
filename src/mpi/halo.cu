#include "halo.h"

#include "decomp.h"
#include "global/structs.h"
#include "gradients/gradients.h"
#include "profiler/profiler.h"
#include "voronoi/voronoi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace proteus_mpi {

MpiHalo halo           = {};
int     n_mpi_capacity = 0;

// composed from sub-files in dependency order
#include "halo_internal.cu"   // shared low-level helpers
#include "halo_init.cu"       // halo_init / halo_free / halo_default_width
#include "halo_build.cu"      // halo_build_exports / halo_build_used_subset / halo_remap_export_indices
#include "halo_exchange.cu"   // halo_exchange_seeds / _primvars / _gradients / _v_mesh / halo_dt_allreduce

}  // namespace proteus_mpi
