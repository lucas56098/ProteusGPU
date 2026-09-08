#!/usr/bin/env python3
"""
Uniform medium at rest — the trivial test IC for ProteusGPU.

rho = 1, v = 0, energy = 1 everywhere, so nothing should happen: any drift is
mesh or scheme noise rather than physics.

Runs in either mode:
  python create.py --n 32 --dimension 2
  mpirun -np 4 python create.py --n 64
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)


def fill_test(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo + n_local)."""
    pos = seed_positions_slice(
        row_lo, n_local, args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )
    rho = np.ones(n_local, dtype=np.float64)
    vel = np.zeros((n_local, args.dimension), dtype=np.float64)
    energy = np.ones(n_local, dtype=np.float64)
    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "test", default_n=32, default_dim=2, default_mesh_mode="random",
    )
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "test"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_test,
        args=args,
    )
