"""
Creates 3D Riemann Initial Conditions (IC) HDF5 file.

From Hoppe et al. (2024), see https://gitlab.lrz.de/nanoshock/riemann_cubes
supports: random mesh and perturbed cartesian

Runs in either mode:
  python create_riemann3d.py --n 50
  mpirun -np 4 python create_riemann3d.py --n 100
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)

# (rho, vx, vy, vz) per octant: the four back octants (z >= mid) then the four front ones,
# each running top-right, top-left, bottom-left, bottom-right. Pressure is uniform.
OCTANTS = [
    (1.0, 0.25, -0.25, -0.5),
    (0.5, 0.25, 0.25, -0.25),
    (2.0, -0.25, 0.25, 0.25),
    (0.5, -0.25, -0.25, -0.25),
    (0.5, -0.25, -0.5, 0.5),
    (2.0, -0.25, 0.5, -0.25),
    (0.5, 0.25, 0.5, 0.25),
    (1.0, 0.25, -0.5, -0.25),
]
PRESSURE = 1.0


def fill_riemann3d(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo + n_local)."""
    pos = seed_positions_slice(
        row_lo, n_local, args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    mid = 0.5 * args.extent
    masks = []
    for z_slab in (z >= mid, z < mid):
        masks += [
            (x >= mid) & (y >= mid) & z_slab,
            (x < mid) & (y > mid) & z_slab,
            (x < mid) & (y < mid) & z_slab,
            (x > mid) & (y < mid) & z_slab,
        ]

    rho = np.zeros(n_local, dtype=np.float64)
    vel = np.zeros((n_local, args.dimension), dtype=np.float64)

    for q, (rho_q, vx_q, vy_q, vz_q) in zip(masks, OCTANTS, strict=True):
        rho[q] = rho_q
        vel[q, 0] = vx_q
        vel[q, 1] = vy_q
        vel[q, 2] = vz_q

    energy = PRESSURE / (args.gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)
    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "riemann3d",
        default_n=50,
        default_mesh_mode="cartesian",
        fixed={"dimension": 3},
    )
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "riemann3d"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_riemann3d,
        args=args,
    )
