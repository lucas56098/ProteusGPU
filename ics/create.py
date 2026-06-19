#!/usr/bin/env python3
"""
Create test Initial Conditions (IC) HDF5 file for ProteusGPU
"""

import h5py
import numpy as np

from common import build_arg_parser, resolve_filename

def create_test_ic(filename="IC.hdf5", num_seeds=100, extent=1.0, dimension=3):
    """
    Create a test IC file with random seedpoints in [0, extent]^dimension
    """
    
    print(f"Creating test IC file: {filename}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    
    with h5py.File(filename, 'w') as f:
        header_group = f.create_group("header")
        header_group.attrs['dimension'] = dimension

        # Seed positions (num_seeds x dimension)
        rng = np.random.default_rng(424242)
        pos = rng.uniform(0, extent, size=(num_seeds, dimension)).astype(np.float64)

        mesh_group = f.create_group("mesh")
        mesh_group.create_dataset("pos", data=pos)

        print(f"  Created mesh/pos dataset: {pos.shape}")
        print(f"    Min values: {pos.min(axis=0)}")
        print(f"    Max values: {pos.max(axis=0)}")

        # Hydro quantities
        rho    = np.ones(num_seeds, dtype=np.float64)                  # uniform density
        vel    = np.zeros((num_seeds, dimension), dtype=np.float64)    # zero velocity
        energy = np.ones(num_seeds, dtype=np.float64)                  # uniform energy

        hydro_group = f.create_group("hydro")
        hydro_group.create_dataset("rho", data=rho)
        hydro_group.create_dataset("vel", data=vel)
        hydro_group.create_dataset("energy", data=energy)

        print(f"  Created hydro/rho dataset: {rho.shape}")
        print(f"  Created hydro/vel dataset: {vel.shape}")
        print(f"  Created hydro/energy dataset: {energy.shape}")
    
    print(f"Successfully created {filename}\n")

if __name__ == "__main__":
    parser = build_arg_parser(
        "test", default_n=32, default_dim=2, default_mesh_mode="random",
    )
    args = parser.parse_args()

    create_test_ic(
        filename=resolve_filename(args, "test"),
        num_seeds=args.n ** args.dimension,
        extent=args.extent,
        dimension=args.dimension,
    )
