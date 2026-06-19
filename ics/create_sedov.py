"""
Creates Taylor-Sedov blast wave IC HDF5 file.

Point explosion in a uniform ambient medium:
  - Uniform density rho_0 = 1
  - Negligible pressure p_0 ~ 0
  - Large energy E_blast deposited in a small central region

Cartesian mesh only (energy deposition assumes structured cell volumes).

Runs in either mode:
  python create_sedov.py --n 45 --dimension 2
  mpirun -np 4 python create_sedov.py --n 200 --dimension 3
"""

import numpy as np

from common import (
    build_arg_parser,
    resolve_filename,
    seed_positions_slice,
    write_ic,
    mpi_runtime,
    collective_sum_int,
)


def fill_sedov(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo+n_local).

    The blast cell count and per-cell deposition need a global view: each rank
    counts its local blast cells, an Allreduce gives the global total, and the
    same E_per_cell value is then applied locally.
    """
    n_global = args.n ** args.dimension
    pos = seed_positions_slice(
        row_lo, n_local, n_global,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode="cartesian",
        perturbation=args.perturbation,
    )

    n_per_axis = int(round(n_global ** (1.0 / args.dimension)))
    dx = args.extent / n_per_axis
    r_blast = args.r_blast if args.r_blast is not None else 0.9 * dx

    center = 0.5 * args.extent
    radius = np.linalg.norm(pos - center, axis=1)
    blast_mask = radius < r_blast

    # global blast-cell count: each rank contributes its local count, all ranks
    # see the same global total via Allreduce. In serial collective_sum_int is
    # a no-op identity.
    comm, _rank, _nranks = mpi_runtime()
    n_blast_global = collective_sum_int(comm, int(np.sum(blast_mask)))
    if n_blast_global == 0:
        raise RuntimeError("No cells inside r_blast — increase r_blast or args.n.")

    cell_volume = (args.extent ** args.dimension) / n_global
    E_per_cell = args.E_blast / (n_blast_global * cell_volume)
    E_ambient = args.p_ambient / (args.gamma - 1.0)

    rho = np.full(n_local, 1.0, dtype=np.float64)
    vel = np.zeros((n_local, args.dimension), dtype=np.float64)
    energy = np.where(blast_mask, E_per_cell, E_ambient).astype(np.float64)

    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "sedov",
        default_n=45,
        default_dim=2,
        default_mesh_mode="cartesian",
        allowed_mesh_modes=("cartesian",),
        default_perturbation=0.01,
    )
    parser.add_argument("--E_blast", type=float, default=1.0)
    parser.add_argument(
        "--r_blast", type=float, default=None,
        help="blast radius (default: auto-derived from cell size)",
    )
    parser.add_argument("--p_ambient", type=float, default=1.0e-5)
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "sedov"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_sedov,
        args=args,
    )
