"""
Common helper utilities for IC generation.
"""

import h5py
import numpy as np


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
