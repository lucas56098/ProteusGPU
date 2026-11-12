#pragma once

// building the open cells of one round, in tiers.cu (included by the .cu files of mesh)

#include "mesh/mesh.h"

namespace mesh {

    // a ball may not reach beyond this part of the box, where a cell would meet its own periodic copy
    constexpr double MAX_BALL_RADIUS = 0.49;

    // nearest neighbours a cell is first cut by, in every tier; the first guess of a ball reaches the farthest of them
    constexpr int NEAREST_NEIGHBOURS = (config::dim == 2) ? 15 : 35;

    // cells built per tier, not counting cells built again for want of room
    struct CellsPerTier {
        long long fast  = 0;
        long long slow  = 0;
        long long exact = 0;

        void add(const CellsPerTier& more) {
            fast += more.fast;
            slow += more.slow;
            exact += more.exact;
        }
    };

    // builds the open cells of the round; each ends certain or asks for more points; the run stops if one fails
    CellsPerTier build_open_cells(int                       n_cells,
                                  const domain::Domain&     domain,
                                  double                    drift_margin,
                                  const ghosts::Ghosts&     ghosts,
                                  const device::Array<Vec>& seed,
                                  device::Array<double>&    volume,
                                  device::Array<Vec>&       centroid_from_seed,
                                  Faces&                    faces,
                                  BuildWork&                work);

} // namespace mesh
