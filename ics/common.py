"""
Common helper utilities for IC generation.
"""

import argparse
import h5py
import numpy as np


_ALL_MESH_MODES = ("random", "cartesian", "polar_ring")

def build_arg_parser(
    name,
    *,
    description=None,
    default_n=64,
    default_dim=3,
    allowed_dims=(2, 3),
    default_mesh_mode="cartesian",
    allowed_mesh_modes=_ALL_MESH_MODES,
    default_perturbation=0.05,
    default_rng_seed=424242,
    default_extent=1.0,
    default_gamma=5.0 / 3.0,
):
    """Return an ArgumentParser with the shared IC-creation flags."""
    parser = argparse.ArgumentParser(description=description or f"Create {name} IC.")

    parser.add_argument(
        "--filename", type=str, default=None,
        help=f"output hdf5 path (default: IC_{name}_<D>D_N<n>.hdf5)",
    )
    parser.add_argument(
        "--n", type=int, default=default_n,
        help="cells per dimension (total seeds = n^dimension)",
    )
    parser.add_argument(
        "--dimension", type=int, default=default_dim, choices=list(allowed_dims),
    )
    parser.add_argument(
        "--mesh_mode", type=str, default=default_mesh_mode, choices=list(allowed_mesh_modes),
    )
    parser.add_argument(
        "--perturbation", type=float, default=default_perturbation,
        help="cartesian-grid jitter as fraction of cell size",
    )
    parser.add_argument("--rng_seed", type=int, default=default_rng_seed)
    parser.add_argument("--extent", type=float, default=default_extent)
    parser.add_argument("--gamma", type=float, default=default_gamma)

    return parser


def resolve_filename(args, name):
    """Pick args.filename if given, otherwise IC_<name>_<D>D_N<n>.hdf5."""
    if args.filename:
        return args.filename
    return f"IC_{name}_{args.dimension}D_N{args.n}.hdf5"


def seed_positions(
    num_seeds,
    dimension,
    extent=1.0,
    rng_seed=424242,
    mesh_mode="random",  # ["random", "cartesian"]
    perturbation=0.05,  # to perturb the cartesian grid
):
    assert mesh_mode in ("random", "cartesian", "polar_ring"), "Unknown 'mesh_mode'."

    rng = np.random.default_rng(rng_seed)

    if mesh_mode == "random":
        seedpos = rng.uniform(0.0, extent, size=(num_seeds, dimension)).astype(np.float64)

    elif mesh_mode == "cartesian":
        if dimension == 2:
            nx = int(round(np.sqrt(num_seeds)))
            ny = nx
            if nx * ny != num_seeds:
                raise ValueError("For cartesian 2D mesh, num_seeds must be a perfect square.")

            dx = extent / nx
            dy = extent / ny
            x1 = (np.arange(nx) + 0.5) * dx
            y1 = (np.arange(ny) + 0.5) * dy
            xx, yy = np.meshgrid(x1, y1, indexing="xy")
            seedpos = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
        else:
            n = int(round(num_seeds ** (1.0 / 3.0)))
            if n * n * n != num_seeds:
                raise ValueError("For cartesian 3D mesh, num_seeds must be a perfect cube.")

            dx = extent / n
            x1 = (np.arange(n) + 0.5) * dx
            xx, yy, zz = np.meshgrid(x1, x1, x1, indexing="xy")
            seedpos = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float64)

        # perturb the cartesian grid
        for i in range(dimension):
            seedpos[:, i] += rng.uniform(-perturbation * dx, perturbation * dx, size=num_seeds)

        # periodic wrap after perturbation
        seedpos %= extent

    elif mesh_mode == "polar_ring":
        seedpos = np.zeros((num_seeds, dimension), dtype=np.float64)
        seed_count = 0

        n_per_dim = int(round(num_seeds ** (1.0 / dimension)))
        d_ring = extent / n_per_dim
        extent_half = 0.5 * extent

        for ring_index in range(n_per_dim):
            n_cells_this_ring = max([1, int(round(2.0 * np.pi * ring_index))])

            phi = rng.uniform(0, 2.0 * np.pi)  # random starting angle
            dphi = (2.0 * np.pi) / n_cells_this_ring

            for _i in range(n_cells_this_ring):
                radius = d_ring * ring_index
                x = radius * np.sin(phi)
                y = radius * np.cos(phi)

                # only include the cell if within the box
                if -extent_half <= x < extent_half and -extent_half <= y < extent_half:
                    seedpos[seed_count] = [x, y]
                    seed_count += 1

                if seed_count == num_seeds:
                    break

                phi += dphi

        seedpos += 0.5 * extent  # shift to box coordinates

    return seedpos
