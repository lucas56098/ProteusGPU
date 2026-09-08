"""
Creates Quad Shock (2D Riemann) Initial Conditions (IC) HDF5 file.

note: conf == 1 (2) is Configuration 3 (5) of Kurganov & Tadmor (2022)
supports: random mesh and perturbed cartesian

Runs in either mode:
  python create_quadshock.py --n 200
  mpirun -np 4 python create_quadshock.py --n 400
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)

# (rho, vx, vy, pressure) per quadrant, ordered top-right, top-left, bottom-left, bottom-right
CONFIGS = {
    1: [(1.5, 0.0, 0.0, 1.5),
        (0.5323, 1.206, 0.0, 0.3),
        (0.138, 1.206, 1.206, 0.029),
        (0.5323, 0.0, 1.206, 0.3)],
    2: [(1.0, 0.75, -0.5, 1.0),
        (2.0, 0.75, 0.5, 1.0),
        (1.0, -0.75, 0.5, 1.0),
        (3.0, -0.75, -0.5, 1.0)],
}


def fill_quadshock(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo + n_local)."""
    pos = seed_positions_slice(
        row_lo, n_local, args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )

    x, y = pos[:, 0], pos[:, 1]
    mid = 0.5 * args.extent
    quadrants = [
        (x >= mid) & (y >= mid),
        (x < mid) & (y > mid),
        (x < mid) & (y < mid),
        (x > mid) & (y < mid),
    ]

    rho = np.zeros(n_local, dtype=np.float64)
    vel = np.zeros((n_local, args.dimension), dtype=np.float64)
    pressure = np.zeros(n_local, dtype=np.float64)

    for q, (rho_q, vx_q, vy_q, p_q) in zip(quadrants, CONFIGS[args.conf], strict=True):
        rho[q] = rho_q
        vel[q, 0] = vx_q
        vel[q, 1] = vy_q
        pressure[q] = p_q

    energy = pressure / (args.gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)
    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "quadshock",
        default_n=200,
        default_mesh_mode="cartesian",
        fixed={"dimension": 2},
    )
    parser.add_argument(
        "--conf", type=int, default=1, choices=[1, 2],
        help="Kurganov-Tadmor configuration: 1 -> conf 3, 2 -> conf 5",
    )
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, f"quadshock{args.conf}"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_quadshock,
        args=args,
    )
