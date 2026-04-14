"""
Creates Taylor-Sedov blast wave IC HDF5 file.

Point explosion in a uniform ambient medium:
  - Uniform density rho_0 = 1
  - Negligible pressure p_0 ~ 0
  - Large energy E_blast deposited in a small central region

Supports cartesian mesh IC only. (due to energy in)

"""

import h5py
import numpy as np
from common import seed_positions


def create_sedov(
    filename,
    num_seeds,
    dimension,
    extent=1.0,
    gamma=5.0 / 3.0,
    E_blast=1.0,
    r_blast=None,
    p_ambient=1.0e-5,
):

    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")

    print(f"Creating Sedov {dimension}D IC file: {filename}")
    print(f"  Total seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    print(f"  Gamma: {gamma}")
    print(f"  Mesh mode: {"cartesian"}")

    pos = seed_positions(num_seeds, dimension, extent=extent, mesh_mode="cartesian", perturbation=0.01)
    dx = extent / int(round(num_seeds ** (1.0 / dimension)))

    # energy injection radius
    if r_blast is None:
        r_blast = 0.9 * dx
    print(f"  r_blast: {r_blast:.6f}")

    # approximate cell volume (uniform grid estimate)
    cell_volume = (extent ** dimension) / num_seeds

    # distance from center
    center = 0.5 * extent
    dr = pos - center
    radius = np.sqrt(np.sum(dr**2, axis=1))

    # ic
    rho = np.full(num_seeds, 1.0, dtype=np.float64)
    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    E_ambient = p_ambient / (gamma - 1.0)
    Energy = np.full(num_seeds, E_ambient, dtype=np.float64)

    # deposit blast energy uniformly over cells inside r_blast
    blast_mask = radius < r_blast
    n_blast = np.sum(blast_mask)
    if n_blast == 0:
        raise RuntimeError("No cells inside r_blast — increase r_blast or num_seeds")

    E_per_cell = E_blast / (n_blast * cell_volume)
    Energy[blast_mask] = E_per_cell

    print(f"  Blast cells: {n_blast}")
    print(f"  E_blast: {E_blast}")
    print(f"  E_per_blast_cell (energy density): {E_per_cell:.6e}")

    # write Proteus HDF5
    with h5py.File(filename, "w") as f:
        header = f.create_group("header")
        header.attrs["dimension"] = dimension
        header.attrs["extent"] = extent
        header.attrs["gamma"] = gamma

        f.create_dataset("seedpos", data=pos)
        f.create_dataset("rho", data=rho)
        f.create_dataset("vel", data=vel)
        f.create_dataset("Energy", data=Energy)

    print(f"  Created {filename} with {num_seeds} cells\n")


if __name__ == "__main__":
    # Create 2D Sedov blast wave
    create_sedov("IC_sedov_2D.hdf5", num_seeds=45**2, dimension=2)

    # Create 3D Sedov blast wave
    #create_sedov("IC_sedov_3D.hdf5", num_seeds=64**3, dimension=3, mesh_mode="cartesian")
