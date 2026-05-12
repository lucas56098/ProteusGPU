"""
Creates Sod's Shock Tube Initial Conditions (IC) HDF5 file

supports: random mesh and perturbed cartesian
"""
import h5py
import numpy as np

from common import build_arg_parser, resolve_filename, seed_positions

def create_sod_shock_tube(
    filename, 
    num_seeds, 
    dimension, 
    extent       = 1.0,
    gamma        = 5./3.,
    mesh_mode    = "random",    # ["random", "cartesian"]
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
    pos = seed_positions(num_seeds, dimension, extent=extent, mesh_mode=mesh_mode)

    # set hydro states
    rho = np.zeros(num_seeds, dtype=np.float64)
    pressure = np.zeros(num_seeds, dtype=np.float64)    

    rho_left, p_left = 1.0, 1.0
    rho_right, p_right = 0.125, 0.1

    left_mask = pos[:, 0] < 0.5 * extent
    rho[left_mask] = rho_left
    pressure[left_mask] = p_left
    
    right_mask = ~left_mask
    rho[right_mask] = rho_right
    pressure[right_mask] = p_right
    
    # velocity is zero
    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    
    # energy from ideal gas equation (energy per volume)
    energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)
    
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
        f.create_dataset("pos", data=pos)
        f.create_dataset("rho", data=rho)
        f.create_dataset("vel", data=vel)
        f.create_dataset("energy", data=energy)

    print(f"\nSuccessfully created {filename}\n")


if __name__ == "__main__":
    parser = build_arg_parser(
        "sod", default_n=50, default_dim=3, default_mesh_mode="cartesian",
    )
    args = parser.parse_args()

    create_sod_shock_tube(
        filename=resolve_filename(args, "sod"),
        num_seeds=args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        gamma=args.gamma,
        mesh_mode=args.mesh_mode,
    )
