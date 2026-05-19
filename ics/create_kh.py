"""
Creates Kelvin-Helmholtz Initial Conditions (IC) HDF5 file.

supports: random mesh and perturbed cartesian
"""

import h5py
import numpy as np

from common import seed_positions, build_arg_parser, resolve_filename


def create_kelvin_helmholtz(
    filename,
    num_seeds,
    dimension,
    extent=1.0,
    gamma=5.0 / 3.0,
    width=0.5,
    mesh_mode="random",  # ["random", "cartesian"]
):
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")

    print(f"Creating Kelvin-Helmholtz IC file: {filename}")
    print(f"  Total seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    print(f"  Gamma: {gamma}")
    print(f"  Mesh mode: {mesh_mode}")

    # Seedpoints
    pos = seed_positions(num_seeds, dimension, extent=extent, mesh_mode=mesh_mode)

    # set hydro states
    x = pos[:, 0]
    y = pos[:, 1]

    y_low = extent / 2 - width / 2
    y_high = extent / 2 + width / 2

    sigma = 0.05 * extent / np.sqrt(2.0)

    inside_shear = (y > y_low) & (y < y_high)

    # set density
    rho = np.where(inside_shear, 2.0, 1.0).astype(np.float64)

    # set velocities
    u = np.where(inside_shear, 0.5, -0.5).astype(np.float64)
    v_pert = (
        0.1
        * np.sin(4.0 * np.pi * x / extent)
        * (np.exp(-((y - y_low) ** 2) / (2.0 * sigma * sigma)) + np.exp(-((y - y_high) ** 2) / (2.0 * sigma * sigma)))
    )

    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    vel[:, 0] = u
    vel[:, 1] = v_pert
    if dimension == 3:
        vel[:, 2] = 0.0

    # set energy (energy per volume)
    pressure = np.full(num_seeds, 2.5, dtype=np.float64)
    energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)

    print("\n  Initial state summary:")
    print(f"    rho range: [{rho.min():.6f}, {rho.max():.6f}]")
    print(f"    u range: [{u.min():.6f}, {u.max():.6f}]")
    print(f"    v perturbation range: [{v_pert.min():.6f}, {v_pert.max():.6f}]")
    print(f"    energy range: [{energy.min():.6f}, {energy.max():.6f}]")

    # Write to HDF5
    with h5py.File(filename, "w") as f:
        header_group = f.create_group("header")
        header_group.attrs["dimension"] = dimension
        header_group.attrs["extent"] = extent
        header_group.attrs["gamma"] = gamma

        f.create_dataset("pos", data=pos)
        f.create_dataset("rho", data=rho)
        f.create_dataset("vel", data=vel)
        f.create_dataset("energy", data=energy)

    print(f"\nSuccessfully created {filename}\n")


if __name__ == "__main__":
    parser = build_arg_parser(
        "kh", default_n=70, default_dim=3, default_mesh_mode="cartesian",
    )
    args = parser.parse_args()

    create_kelvin_helmholtz(
        filename=resolve_filename(args, "kh"),
        num_seeds=args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        gamma=args.gamma,
        mesh_mode=args.mesh_mode,
    )
