"""
Creates Quad Shock 2 Initial Conditions (IC) HDF5 file.

supports: random mesh and perturbed cartesian
"""

import h5py
import numpy as np

from common import seed_positions


def create_quadshock2(
    filename,
    num_seeds,
    dimension,
    extent=1.0,
    gamma=7.0 / 5.0,
    mesh_mode="random",  # ["random", "cartesian"]
):
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")

    print(f"Creating Quad Shock 2 IC file: {filename}")
    print(f"  Total seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    print(f"  Gamma: {gamma}")
    print(f"  Mesh mode: {mesh_mode}")

    # Seedpoints
    seedpos = seed_positions(num_seeds, dimension, extent=extent, mesh_mode=mesh_mode)

    # set hydro states based on quadrant
    x = seedpos[:, 0]
    y = seedpos[:, 1]

    mid = 0.5 * extent

    # quadrant masks
    q1 = (x >= mid) & (y >= mid)  # top-right
    q2 = (x < mid) & (y > mid)   # top-left
    q3 = (x < mid) & (y < mid)   # bottom-left
    q4 = (x > mid) & (y < mid)   # bottom-right

    # density
    rho = np.empty(num_seeds, dtype=np.float64)
    rho[q1] = 1.0
    rho[q2] = 2.0
    rho[q3] = 1.0
    rho[q4] = 3.0

    # velocities
    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    vel[q1, 0] = 0.75;    vel[q1, 1] = -0.5
    vel[q2, 0] = 0.75;    vel[q2, 1] = 0.5
    vel[q3, 0] = -0.75;   vel[q3, 1] = 0.5
    vel[q4, 0] = -0.75;   vel[q4, 1] = -0.5

    # pressure (all quadrants have P = 1)
    pressure = np.ones(num_seeds, dtype=np.float64)

    # energy per volume: E = P/(gamma-1) + 0.5*rho*v^2
    Energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)

    print("\n  Initial state summary:")
    print(f"    rho range: [{rho.min():.6f}, {rho.max():.6f}]")
    print(f"    vel_x range: [{vel[:,0].min():.6f}, {vel[:,0].max():.6f}]")
    print(f"    vel_y range: [{vel[:,1].min():.6f}, {vel[:,1].max():.6f}]")
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
    # 2D Quad Shock 2 IC
    create_quadshock2("IC_quadshock2_2D.hdf5", num_seeds=800**2, dimension=2, mesh_mode="cartesian")

    # 3D Quad Shock 2 IC
    #create_quadshock2("IC_quadshock2_3D.hdf5", num_seeds=50**3, dimension=3, mesh_mode="cartesian")
