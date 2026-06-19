"""
Creates Sod's Shock Tube Initial Conditions (IC) HDF5 file.

Runs in either mode:
  python create_sod.py --n 50 --mesh_mode cartesian
  mpirun -np 4 python create_sod.py --n 200 --mesh_mode cartesian

The launcher decides — same script either way. See ics/common.py for the
slice/serial dispatch and bit-identical-across-launchers rules.
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)


def fill_sod(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo+n_local)."""
    n_global = args.n ** args.dimension
    pos = seed_positions_slice(
        row_lo, n_local, n_global,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )

    rho_left, p_left = 1.0, 1.0
    rho_right, p_right = 0.125, 0.1

    left_mask = pos[:, 0] < 0.5 * args.extent
    rho = np.where(left_mask, rho_left, rho_right)
    pressure = np.where(left_mask, p_left, p_right)

    vel = np.zeros((n_local, args.dimension), dtype=np.float64)
    energy = pressure / (args.gamma - 1.0)  # v = 0 so kinetic term is zero

    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "sod", default_n=50, default_dim=3, default_mesh_mode="cartesian",
    )
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "sod"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_sod,
        args=args,
    )
