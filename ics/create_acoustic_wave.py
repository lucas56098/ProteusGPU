"""
Creates Acoustic Wave Initial Conditions (IC) HDF5 file.

Linear right-traveling sound wave with delta_rho/rho = 1e-6 of unit wavelength
on a periodic unit box, with rho_0 = 1, P_0 = 3/5, gamma = 5/3 (so c_s = 1).
The wave is a plane wave along x. After t = L / c_s = 1, the analytic solution
returns to the IC.
(Stone et al. 2008; Springel 2010, Sec. 8.1).

Runs in either mode:
  python create_acoustic_wave.py --n 32 --dimension 3
  mpirun -np 4 python create_acoustic_wave.py --n 64
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)


def fill_acoustic_wave(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo + n_local)."""
    pos = seed_positions_slice(
        row_lo, n_local, args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )

    c_s = np.sqrt(args.gamma * args.p_0 / args.rho_0)
    delta_v = (args.delta_rho / args.rho_0) * c_s
    delta_p = c_s * c_s * args.delta_rho

    # plane wave along x with unit wavelength
    s = np.sin((2.0 * np.pi / args.extent) * pos[:, 0])

    rho = args.rho_0 + args.delta_rho * s
    pressure = args.p_0 + delta_p * s

    vel = np.zeros((n_local, args.dimension), dtype=np.float64)
    vel[:, 0] = delta_v * s

    energy = pressure / (args.gamma - 1.0) + 0.5 * rho * np.sum(vel ** 2, axis=1)
    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "acoustic_wave",
        default_n=32,
        default_dim=3,
        default_mesh_mode="cartesian",
        default_perturbation=0.00001,
    )
    parser.add_argument("--rho_0", type=float, default=1.0)
    parser.add_argument("--p_0", type=float, default=3.0 / 5.0)
    parser.add_argument("--delta_rho", type=float, default=1.0e-6)
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "acoustic_wave"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_acoustic_wave,
        args=args,
    )
