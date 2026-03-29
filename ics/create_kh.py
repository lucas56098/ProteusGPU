"""
Creates Kelvin-Helmholtz Initial Conditions (IC) HDF5 file.

supports: random mesh and perturbed cartesian
"""
import h5py
import numpy as np

def create_kelvin_helmholtz(
    filename,
    num_seeds,
    dimension,
    extent       = 1.0,
    gamma        = 5./3.,
    rng_seed     = 424242,
    mesh_mode    = "random",    # ["random", "cartesian"]
    perturbation = 0.05         # to perturb the cartesian grid
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

            # perturb the cartesian grid
            seedpos[:, 0] += rng.uniform(-perturbation * dx, perturbation * dx, size=num_seeds)
            seedpos[:, 1] += rng.uniform(-perturbation * dy, perturbation * dy, size=num_seeds)
        else:
            n = int(round(num_seeds ** (1.0 / 3.0)))
            if n * n * n != num_seeds:
                raise ValueError("For cartesian 3D mesh, num_seeds must be a perfect cube.")

            dx = extent / n
            x1 = (np.arange(n) + 0.5) * dx
            xx, yy, zz = np.meshgrid(x1, x1, x1, indexing="xy")
            seedpos = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float64)

            # perturb the cartesian grid
            seedpos[:, 0] += rng.uniform(-perturbation * dx, perturbation * dx, size=num_seeds)
            seedpos[:, 1] += rng.uniform(-perturbation * dx, perturbation * dx, size=num_seeds)
            seedpos[:, 2] += rng.uniform(-perturbation * dx, perturbation * dx, size=num_seeds)

        # periodic wrap after perturbation
        seedpos %= extent
    else:
        raise ValueError("mesh_mode must be 'random' or 'cartesian'")


    # set hydro states
    x = seedpos[:, 0]
    y = seedpos[:, 1]

    y_low = 0.25 * extent
    y_high = 0.75 * extent
    sigma = 0.05 * extent / np.sqrt(2.0)

    inside_shear = (y > y_low) & (y < y_high)

    # set density
    rho = np.where(inside_shear, 2.0, 1.0).astype(np.float64)
    
    # set velocities
    u = np.where(inside_shear, 0.5, -0.5).astype(np.float64)
    v_pert = 0.1 * np.sin(4.0 * np.pi * x / extent) * (np.exp(-((y - y_low) ** 2) / (2.0 * sigma * sigma)) + np.exp(-((y - y_high) ** 2) / (2.0 * sigma * sigma)))

    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    vel[:, 0] = u
    vel[:, 1] = v_pert
    if dimension == 3:
        vel[:, 2] = 0.0

    # set energy (energy per volume)
    pressure = np.full(num_seeds, 2.5, dtype=np.float64)
    Energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)

    print("\n  Initial state summary:")
    print(f"    rho range: [{rho.min():.6f}, {rho.max():.6f}]")
    print(f"    u range: [{u.min():.6f}, {u.max():.6f}]")
    print(f"    v perturbation range: [{v_pert.min():.6f}, {v_pert.max():.6f}]")
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
    # 2D KH ICcl
    create_kelvin_helmholtz("IC_kh_2D.hdf5", num_seeds=150**2, dimension=2)

    # 3D KH IC
    # create_kelvin_helmholtz("IC_kh_3D.hdf5", num_seeds=150**3, dimension=3)
