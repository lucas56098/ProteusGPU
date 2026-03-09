#!/usr/bin/env python3
"""
Create test Initial Conditions (IC) HDF5 file for ProteusGPU
"""

import h5py
import numpy as np

def create_test_ic(filename="IC.hdf5", num_seeds=100, extent=1.0, dimension=3):
    """
    Create a test IC file with random seedpoints in [0, extent]^dimension
    """
    
    print(f"Creating test IC file: {filename}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    
    with h5py.File(filename, 'w') as f:
        # Create header group and attributes
        header_group = f.create_group("header")
        
        header_group.attrs['dimension'] = dimension
        header_group.attrs['extent'] = extent
        
        print(f"  Created header group with attributes")
        
        # Create seedpos dataset (num_seeds x dimension)
        rng = np.random.default_rng(424242)
        seedpos = rng.uniform(0, extent, size=(num_seeds, dimension)).astype(np.float64)
        f.create_dataset("seedpos", data=seedpos)
        
        print(f"  Created seedpos dataset: {seedpos.shape}")
        print(f"    Min values: {seedpos.min(axis=0)}")
        print(f"    Max values: {seedpos.max(axis=0)}")
        
        # Create hydro quantities
        rho = np.ones(num_seeds, dtype=np.float64)  # uniform density
        f.create_dataset("rho", data=rho)
        print(f"  Created rho dataset: {rho.shape}")
        
        vel = np.zeros((num_seeds, dimension), dtype=np.float64)  # zero velocity
        f.create_dataset("vel", data=vel)
        print(f"  Created vel dataset: {vel.shape}")
        
        Energy = np.ones(num_seeds, dtype=np.float64)  # uniform energy
        f.create_dataset("Energy", data=Energy)
        print(f"  Created Energy dataset: {Energy.shape}")
    
    print(f"Successfully created {filename}\n")

if __name__ == "__main__":
    # Create test IC file
    create_test_ic("IC.hdf5", num_seeds=1000, extent=1.0, dimension=2) # change dimension here :D
    #create_test_ic("IC.hdf5", num_seeds=15000, extent=1.0, dimension=3)
