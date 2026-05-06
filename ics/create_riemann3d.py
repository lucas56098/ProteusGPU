"""
Creates 3D Riemann Initial Conditions (IC) HDF5 file.

From Hoppe et al. (2024), see https://gitlab.lrz.de/nanoshock/riemann_cubes
supports: random mesh and perturbed cartesian
"""

import h5py
import numpy as np

from common import seed_positions, build_arg_parser, resolve_filename

def create_riemann3d(
    filename,
    num_seeds,
    extent=1.0,
    gamma=5.0 / 3.0,
    mesh_mode="random",  # ["random", "cartesian"]
):
    dimension = 3

    print(f"Creating Riemann3D IC file: {filename}")
    print(f"  Total seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    print(f"  Gamma: {gamma}")
    print(f"  Mesh mode: {mesh_mode}")

    # Seedpoints
    seedpos = seed_positions(num_seeds, dimension, extent=extent, mesh_mode=mesh_mode)

    # set hydro states based on octant
    x = seedpos[:, 0]
    y = seedpos[:, 1]
    z = seedpos[:, 2]

    mid = 0.5 * extent

    # octant masks
    q1 = (x >= mid) & (y >= mid) & (z >= mid)  # top-right (back)
    q2 = (x < mid) & (y > mid) & (z >= mid)   # top-left (back)
    q3 = (x < mid) & (y < mid) & (z >= mid)   # bottom-left (back)
    q4 = (x > mid) & (y < mid) & (z >= mid)   # bottom-right (back)
    q5 = (x >= mid) & (y >= mid) & (z < mid)  # top-right (front)
    q6 = (x < mid) & (y > mid) & (z < mid)   # top-left (front)
    q7 = (x < mid) & (y < mid) & (z < mid)   # bottom-left (front)
    q8 = (x > mid) & (y < mid) & (z < mid)   # bottom-right (front)

    # allocate
    rho = np.zeros(num_seeds, dtype=np.float64)
    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    pressure = np.zeros(num_seeds, dtype=np.float64)

    # density
    rho[q1] = 1.0
    rho[q2] = 0.5
    rho[q3] = 2.0
    rho[q4] = 0.5
    rho[q5] = 0.5
    rho[q6] = 2.0
    rho[q7] = 0.5
    rho[q8] = 1.0

    # velocities
    vel[q1, 0] = 0.25;   vel[q1, 1] = -0.25;  vel[q1, 2] = -0.5
    vel[q2, 0] = 0.25;   vel[q2, 1] = 0.25;   vel[q2, 2] = -0.25
    vel[q3, 0] = -0.25;  vel[q3, 1] = 0.25;   vel[q3, 2] = 0.25
    vel[q4, 0] = -0.25;  vel[q4, 1] = -0.25;  vel[q4, 2] = -0.25
    vel[q5, 0] = -0.25;  vel[q5, 1] = -0.5;   vel[q5, 2] = 0.5
    vel[q6, 0] = -0.25;  vel[q6, 1] = 0.5;    vel[q6, 2] = -0.25
    vel[q7, 0] = 0.25;   vel[q7, 1] = 0.5;    vel[q7, 2] = 0.25
    vel[q8, 0] = 0.25;   vel[q8, 1] = -0.5;    vel[q8, 2] = -0.25

    # pressure
    pressure += 1.0 # uniform

    # energy per volume: E = P/(gamma-1) + 0.5*rho*v^2
    Energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)

    print("\n  Initial state summary:")
    print(f"    rho range: [{rho.min():.6f}, {rho.max():.6f}]")
    print(f"    vel_x range: [{vel[:,0].min():.6f}, {vel[:,0].max():.6f}]")
    print(f"    vel_y range: [{vel[:,1].min():.6f}, {vel[:,1].max():.6f}]")
    print(f"    vel_z range: [{vel[:,2].min():.6f}, {vel[:,2].max():.6f}]")
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
    # riemann3d is 3D-only.
    parser = build_arg_parser(
        "riemann3d",
        default_n=50,
        default_dim=3,
        allowed_dims=(3,),
        default_mesh_mode="cartesian",
    )
    args = parser.parse_args()

    create_riemann3d(
        filename=resolve_filename(args, "riemann3d"),
        num_seeds=args.n ** args.dimension,
        extent=args.extent,
        gamma=args.gamma,
        mesh_mode=args.mesh_mode,
    )
