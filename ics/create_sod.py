"""
Creates Sod's Shock Tube Initial Conditions (IC) HDF5 file

supports: random mesh and perturbed cartesian
"""
import h5py
import numpy as np

def create_sod_shock_tube(
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
    
    print(f"Creating Sod shock tube IC file: {filename}")
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
    rho = np.zeros(num_seeds, dtype=np.float64)
    pressure = np.zeros(num_seeds, dtype=np.float64)    

    rho_left, p_left = 1.0, 1.0
    rho_right, p_right = 0.125, 0.1

    left_mask = seedpos[:, 0] < 0.5 * extent
    rho[left_mask] = rho_left
    pressure[left_mask] = p_left
    
    right_mask = ~left_mask
    rho[right_mask] = rho_right
    pressure[right_mask] = p_right
    
    # velocity is zero
    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    
    # energy from ideal gas equation (energy per volume)
    Energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)
    
    print(f"\n  Initial state:")
    print(f"    Left (x < {0.5 * extent}):  rho={rho_left}, p={p_left}, E={p_left/((gamma-1.0)):.6f}")
    print(f"    Right (x >= {0.5 * extent}): rho={rho_right}, p={p_right}, E={p_right/((gamma-1.0)):.6f}")
    
    # Write to HDF5
    with h5py.File(filename, 'w') as f:
        # Create header group and attributes
        header_group = f.create_group("header")
        header_group.attrs['dimension'] = dimension
        header_group.attrs['extent'] = extent
        header_group.attrs['gamma'] = gamma
        
        # Create datasets
        f.create_dataset("seedpos", data=seedpos)
        f.create_dataset("rho", data=rho)
        f.create_dataset("vel", data=vel)
        f.create_dataset("Energy", data=Energy)
    print(f"\nSuccessfully created {filename}\n")


if __name__ == "__main__":
    # Create 2D Sod shock tube
    create_sod_shock_tube("IC_sod_2D.hdf5", num_seeds=200**2, dimension=2)
    
    # Create 3D Sod shock tube
    #create_sod_shock_tube("IC_sod_3D.hdf5", num_seeds=150**3, dimension=3)
