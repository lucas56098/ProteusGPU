"""
Creates Kelvin-Helmholtz Initial Conditions (IC) HDF5 file.

Runs in either mode:
  python create_kh.py --n 70 --mesh_mode cartesian
  mpirun -np 4 python create_kh.py --n 200 --mesh_mode cartesian
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)


def fill_kh(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo+n_local).

    Two-shear-layer KH: high-density shear band at width centered on extent/2,
    with a sin perturbation in v_y localised at the two interfaces by Gaussians.
    """
    n_global = args.n ** args.dimension
    pos = seed_positions_slice(
        row_lo, n_local, n_global,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )

    x = pos[:, 0]
    y = pos[:, 1]

    y_low = args.extent / 2 - args.width / 2
    y_high = args.extent / 2 + args.width / 2
    sigma = 0.05 * args.extent / np.sqrt(2.0)

    inside_shear = (y > y_low) & (y < y_high)

    rho = np.where(inside_shear, 2.0, 1.0).astype(np.float64)
    u = np.where(inside_shear, 0.5, -0.5).astype(np.float64)
    v_pert = (
        0.1
        * np.sin(4.0 * np.pi * x / args.extent)
        * (np.exp(-((y - y_low) ** 2) / (2.0 * sigma * sigma))
           + np.exp(-((y - y_high) ** 2) / (2.0 * sigma * sigma)))
    )

    vel = np.zeros((n_local, args.dimension), dtype=np.float64)
    vel[:, 0] = u
    vel[:, 1] = v_pert
    # vel[:, 2] = 0 by construction in 3D

    pressure = np.full(n_local, 2.5, dtype=np.float64)
    energy = pressure / (args.gamma - 1.0) + 0.5 * rho * np.sum(vel ** 2, axis=1)

    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "kh", default_n=70, default_dim=3, default_mesh_mode="cartesian",
    )
    parser.add_argument("--width", type=float, default=0.5,
                        help="height of the central high-density shear band")
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "kh"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_kh,
        args=args,
    )
