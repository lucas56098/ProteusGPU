"""
Creates Gresho vortex Initial Conditions (IC) HDF5 file.

supports: random mesh, perturbed cartesian, and polar ring

Runs in either mode:
  python create_gresho.py --n 800
  mpirun -np 4 python create_gresho.py --n 800
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)


def fill_gresho(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo + n_local)."""
    pos = seed_positions_slice(
        row_lo, n_local, args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )

    x = pos[:, 0] - 0.5 * args.extent
    y = pos[:, 1] - 0.5 * args.extent
    radius = np.sqrt(x**2 + y**2)
    xi = radius / args.extent

    inner = xi < 0.2
    mid = (xi >= 0.2) & (xi < 0.4)
    outer = xi >= 0.4

    rho = np.ones(n_local, dtype=np.float64)

    vrot = np.zeros(n_local, dtype=np.float64)
    vrot[inner] = 5.0 * xi[inner]
    vrot[mid] = 2.0 - 5.0 * xi[mid]

    vel = np.zeros((n_local, args.dimension), dtype=np.float64)
    nonzero_radius = radius > 0.0
    vel[nonzero_radius, 0] = vrot[nonzero_radius] * y[nonzero_radius] / radius[nonzero_radius]
    vel[nonzero_radius, 1] = -vrot[nonzero_radius] * x[nonzero_radius] / radius[nonzero_radius]

    pressure = np.zeros(n_local, dtype=np.float64)
    pressure[inner] = 5.0 + 12.5 * xi[inner] ** 2
    pressure[mid] = 9.0 + 12.5 * xi[mid] ** 2 - 20 * xi[mid] + 4 * np.log(xi[mid] / 0.2)
    pressure[outer] = 3.0 + 4.0 * np.log(2.0)

    energy = pressure / (args.gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)
    return pos, vel, rho, energy


if __name__ == "__main__":
    # Gresho is 2D-only.
    parser = build_arg_parser(
        "gresho",
        default_n=800,
        default_dim=2,
        allowed_dims=(2,),
        default_mesh_mode="polar_ring",
    )
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "gresho"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_gresho,
        args=args,
    )
