"""
Creates Acoustic Wave Initial Conditions (IC) HDF5 file.

Linear right-traveling sound wave with delta_rho/rho = 1e-6 of unit wavelength
on a periodic unit box, with rho_0 = 1, P_0 = 3/5, gamma = 5/3 (so c_s = 1).
The wave is a plane wave along x. After t = L / c_s = 1, the analytic solution
returns to the IC.
(Stone et al. 2008; Springel 2010, Sec. 8.1). 

supports: random mesh and perturbed cartesian
"""

import h5py
import numpy as np

from common import seed_positions, build_arg_parser, resolve_filename


def create_acoustic_wave(
    filename,
    num_seeds,
    dimension,
    extent=1.0,
    gamma=5.0 / 3.0,
    rho_0=1.0,
    p_0=3.0 / 5.0,
    delta_rho=1.0e-6,
    mesh_mode="cartesian",
    perturbation=0.0,
):
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")

    c_s = np.sqrt(gamma * p_0 / rho_0)
    delta_v = (delta_rho / rho_0) * c_s
    delta_p = c_s * c_s * delta_rho

    print(f"Creating Acoustic Wave IC file: {filename}")
    print(f"  Total seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    print(f"  Gamma: {gamma}")
    print(f"  rho_0: {rho_0}, p_0: {p_0}, c_s: {c_s}")
    print(f"  Delta rho/rho: {delta_rho/rho_0}")
    print(f"  Mesh mode: {mesh_mode} (perturbation = {perturbation})")

    pos = seed_positions(
        num_seeds,
        dimension,
        extent=extent,
        mesh_mode=mesh_mode,
        perturbation=perturbation,
    )

    # plane wave along x with unit wavelength
    x = pos[:, 0]
    k = 2.0 * np.pi / extent
    s = np.sin(k * x)

    rho = rho_0 + delta_rho * s
    pressure = p_0 + delta_p * s

    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    vel[:, 0] = delta_v * s

    # energy per volume
    energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel ** 2, axis=1)

    print("\n  Initial state summary:")
    print(f"    rho range: [{rho.min():.12f}, {rho.max():.12f}]")
    print(f"    vx  range: [{vel[:, 0].min():.6e}, {vel[:, 0].max():.6e}]")
    print(f"    p   range: [{pressure.min():.12f}, {pressure.max():.12f}]")

    with h5py.File(filename, "w") as f:
        header_group = f.create_group("header")
        header_group.attrs["dimension"] = dimension

        mesh_group = f.create_group("mesh")
        mesh_group.create_dataset("pos", data=pos)

        hydro_group = f.create_group("hydro")
        hydro_group.create_dataset("rho", data=rho)
        hydro_group.create_dataset("vel", data=vel)
        hydro_group.create_dataset("energy", data=energy)

    print(f"\nSuccessfully created {filename}\n")


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

    create_acoustic_wave(
        filename=resolve_filename(args, "acoustic_wave"),
        num_seeds=args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        gamma=args.gamma,
        rho_0=args.rho_0,
        p_0=args.p_0,
        delta_rho=args.delta_rho,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )
