#ifndef VORONOI_PERIODIC_H
#define VORONOI_PERIODIC_H

#include "../global/allvars.h"
#include "../voronoi/voronoi.h"

namespace voronoi {

    // periodic ghost generation + mesh rebuild over the extended [-buff, 1+buff]^d domain.
    void compute_periodic_mesh(VMesh*           mesh,
                               POINT_TYPE*      pts_data,
                               hsize_t          num_points,
                               hydro::primvars* primvar,
                               hydro::primvars* primvar_aux);

} // namespace voronoi
#endif // VORONOI_PERIODIC_H
