"""
Creates Noh implosion Initial Conditions (IC) HDF5 file.

Cold gas flows radially toward the box centre at unit speed:
  - Uniform density rho_0 = 1
  - Negligible pressure p_0 ~ 0
  - Velocity v = -v_0 r_hat (zero in a cell sitting exactly at the centre)

An infinitely strong shock forms at the centre and moves outward at
v_s = (gamma - 1) v_0 / 2 (= 1/3 for gamma = 5/3). Behind it the gas is at rest
with rho = rho_0 ((gamma + 1) / (gamma - 1))^D (16 in 2D, 64 in 3D for
gamma = 5/3); ahead of it the infalling gas is compressed as
rho = rho_0 (1 + v_0 t / r)^(D - 1).
(Noh 1987; Springel 2010, Sec. 8.4).

The box is the periodic unit box, and there is no inflow imposed at the boundaries.
As a reuslt, a rarefaction wave propagates inward at ~v_0. The region of the box 
this covers is compromised by edge effects. However, the problem is self-similar, 
so the usual setup (box of 6, end at t = 2, shock radius 2/3) is equilvane to the 
setup here at t = 1/3, with shock radius 1/9, and the rarefaction still ~0.05 away.

Runs in either mode:
  python create_noh.py --n 256 --dimension 2
  mpirun -np 4 python create_noh.py --n 128 --dimension 3
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)


def fill_noh(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo + n_local)."""
    pos = seed_positions_slice(
        row_lo, n_local, args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )

    dr = pos - 0.5 * args.extent
    radius = np.linalg.norm(dr, axis=1)

    rho = np.full(n_local, args.rho_0, dtype=np.float64)

    vel = np.zeros((n_local, args.dimension), dtype=np.float64)
    nonzero_radius = radius > 0.0
    vel[nonzero_radius] = -args.v_0 * dr[nonzero_radius] / radius[nonzero_radius, None]

    pressure = np.full(n_local, args.p_0, dtype=np.float64)

    energy = pressure / (args.gamma - 1.0) + 0.5 * rho * np.sum(vel ** 2, axis=1)
    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "noh",
        default_n=256,
        default_dim=2,
        default_mesh_mode="cartesian",
        default_perturbation=0.01,
    )
    parser.add_argument("--rho_0", type=float, default=1.0)
    parser.add_argument("--v_0", type=float, default=1.0, help="radial infall speed")
    parser.add_argument("--p_0", type=float, default=1.0e-6)
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "noh"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_noh,
        args=args,
    )
