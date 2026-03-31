"""
Creates Gresho vortex Initial Conditions (IC) HDF5 file.

supports: random mesh, perturbed cartesian, and polar ring
"""

import h5py
import numpy as np

from common import seed_positions


def create_gresho_vortex(filename, num_seeds, dimension, extent=1.0, gamma=5.0 / 3.0, mesh_mode="polar_ring"):
    """Create initial conditions for the Gresho vortex test problem."""
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")

    print(f"Creating Gresho vortex IC file: {filename}")
    print(f"  Total seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    print(f"  Gamma: {gamma}")
    print(f"  Mesh mode: {mesh_mode}")

    # Seedpoints
    seedpos = seed_positions(num_seeds, dimension, extent=extent, mesh_mode=mesh_mode)

    # set hydro states
    x = seedpos[:, 0] - 0.5 * extent
    y = seedpos[:, 1] - 0.5 * extent

    radius = np.sqrt(x**2 + y**2)
    xi = radius / extent

    region_1 = np.where(xi < 0.2)
    region_2 = np.where((xi >= 0.2) & (xi < 0.4))
    region_3 = np.where(xi >= 0.4)

    # set density
    rho = np.zeros(num_seeds, dtype="float32")
    rho += 1.0  # constant

    # set velocities
    vel = np.zeros((num_seeds, dimension), dtype="float32")
    vrot = np.zeros(num_seeds, dtype="float32")

    vrot[region_1] = 5.0 * xi[region_1]
    vrot[region_2] = 2.0 - 5.0 * xi[region_2]
    vrot[region_3] = 0.0

    nonzero_radius = radius > 0.0
    vel[nonzero_radius, 0] = vrot[nonzero_radius] * y[nonzero_radius] / radius[nonzero_radius]
    vel[nonzero_radius, 1] = -vrot[nonzero_radius] * x[nonzero_radius] / radius[nonzero_radius]

    # set energy (energy per volume)
    pressure = np.zeros(num_seeds, dtype="float64")

    pressure[region_1] = 5.0 + 12.5 * xi[region_1] ** 2
    pressure[region_2] = 9.0 + 12.5 * xi[region_2] ** 2 - 20 * xi[region_2] + 4 * np.log(xi[region_2] / 0.2)
    pressure[region_3] = 3.0 + 4.0 * np.log(2.0)

    Energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)

    print("\n  Initial state summary:")
    print(f"    rho range: [{rho.min():.6f}, {rho.max():.6f}]")
    print(f"    pressure range: [{pressure.min():.6f}, {pressure.max():.6f}]")
    print(f"    Energy range: [{Energy.min():.6f}, {Energy.max():.6f}]")

    # Write to HDF5
    with h5py.File(filename, "w") as f:
        header_group = f.create_group("header")
        header_group.attrs["dimension"] = dimension
        header_group.attrs["extent"] = extent
        header_group.attrs["gamma"] = gamma

        f.create_dataset("seedpos", data=seedpos)
        f.create_dataset("rho", data=rho)
        f.create_dataset("vel", data=vel)
        f.create_dataset("Energy", data=Energy)

    print(f"\nSuccessfully created {filename}\n")


if __name__ == "__main__":
    # 2D Gresho IC
    create_gresho_vortex("IC_gresho_2D.hdf5", num_seeds=800**2, dimension=2)

    # 3D Gresho IC
    # create_gresho_vortex("IC_gresho_3D.hdf5", num_seeds=150**3, dimension=3)
