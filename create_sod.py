#!/usr/bin/env python3
"""
Create Sod Shock Tube Initial Conditions (IC) HDF5 file for ProteusGPU
Standard test case for hydrodynamic solvers with shock, rarefaction, and contact discontinuity
"""

import h5py
import numpy as np
from scipy.spatial import Voronoi

def lloyd_regularization(seedpos, extent, num_iterations=5):
    """
    Perform Lloyd's algorithm for point set regularization.
    
    This creates a more uniform distribution by iteratively moving each point
    to the centroid of its Voronoi cell.
    
    Args:
        seedpos: Nx dimension array of point coordinates
        extent: Domain size [0, extent] in each direction
        num_iterations: Number of Lloyd iterations to perform
        
    Returns:
        Regularized seedpos array
    """
    seedpos = seedpos.copy()
    dimension = seedpos.shape[1]
    
    print(f"\n  Performing Lloyd regularization ({num_iterations} iterations)...")
    
    for iteration in range(num_iterations):
        # Compute Voronoi diagram
        vor = Voronoi(seedpos)
        
        # Move each point to centroid of its Voronoi cell
        new_seedpos = np.zeros_like(seedpos)
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            
            if -1 in region:
                # Cell is unbounded, keep point as is (or use neighbors' centroid)
                new_seedpos[i] = seedpos[i]
            else:
                # Cell is bounded, compute centroid
                vertices = vor.vertices[region]
                centroid = vertices.mean(axis=0)
                
                # Clamp to domain bounds
                centroid = np.clip(centroid, 0, extent)
                new_seedpos[i] = centroid
        
        seedpos = new_seedpos
        print(f"    Iteration {iteration + 1}/{num_iterations} completed")
    
    return seedpos


def create_sod_shock_tube(filename="IC.hdf5", num_seeds=100, extent=1.0, dimension=2, gamma=5./3.):
    """
    Create Sod shock tube initial conditions in the x-direction.
    
    The shock tube is initialized with a discontinuity at x=0.5:
    - Left state (x < 0.5):  rho=1.0, v=0, p=1.0
    - Right state (x >= 0.5): rho=0.125, v=0, p=0.1
    
    Seed points are regularized using Lloyd's algorithm for uniform distribution.
    
    Args:
        filename: Output HDF5 filename
        num_seeds: Total number of seed points (regularized via Lloyd's algorithm)
        extent: Domain size in each direction [0, extent]
        dimension: 2 or 3 (for 2D or 3D shock tube)
        gamma: Adiabatic index (default: 1.4 for air)
    """
    
    print(f"Creating Sod shock tube IC file: {filename}")
    print(f"  Total seeds: {num_seeds}")
    print(f"  Dimension: {dimension}")
    print(f"  Extent: {extent}")
    print(f"  Gamma: {gamma}")
    
    # Create random seedpoints
    seedpos = np.random.uniform(0, extent, size=(num_seeds, dimension)).astype(np.float64)
    
    # Apply Lloyd regularization for uniform distribution
    #seedpos = lloyd_regularization(seedpos, extent, num_iterations=5)
    
    # Shock position at x = 0.5
    shock_position = 0.5 * extent
    
    # Sod shock tube states
    rho_left, p_left = 1.0, 1.0
    rho_right, p_right = 0.125, 0.1
    v_left = 0.0
    v_right = 0.0
    
    # Assign properties based on x-coordinate
    rho = np.zeros(num_seeds, dtype=np.float64)
    pressure = np.zeros(num_seeds, dtype=np.float64)
    
    # Left of shock: high density and pressure
    left_mask = seedpos[:, 0] < shock_position
    rho[left_mask] = rho_left
    pressure[left_mask] = p_left
    
    # Right of shock: low density and pressure
    right_mask = ~left_mask
    rho[right_mask] = rho_right
    pressure[right_mask] = p_right
    
    # Velocity is zero everywhere (will develop during simulation)
    vel = np.zeros((num_seeds, dimension), dtype=np.float64)
    
    # Energy from ideal gas equation: E = p / ((gamma - 1) * rho)
    Energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel**2, axis=1)
    
    print(f"\n  Initial state:")
    print(f"    Left (x < {shock_position}):  rho={rho_left}, p={p_left}, E={p_left/((gamma-1.0)*rho_left):.6f}")
    print(f"    Right (x >= {shock_position}): rho={rho_right}, p={p_right}, E={p_right/((gamma-1.0)*rho_right):.6f}")
    
    # Write to HDF5
    with h5py.File(filename, 'w') as f:
        # Create header group and attributes
        header_group = f.create_group("header")
        header_group.attrs['dimension'] = dimension
        header_group.attrs['extent'] = extent
        header_group.attrs['gamma'] = gamma
        
        print(f"\n  Created header group with attributes")
        
        # Create datasets
        f.create_dataset("seedpos", data=seedpos)
        print(f"  Created seedpos dataset: {seedpos.shape}")
        print(f"    x range: [{seedpos[:, 0].min():.6f}, {seedpos[:, 0].max():.6f}]")
        
        f.create_dataset("rho", data=rho)
        print(f"  Created rho dataset: {rho.shape}")
        print(f"    Min: {rho.min():.6f}, Max: {rho.max():.6f}")
        print(f"    Left side mean: {rho[left_mask].mean():.6f}")
        print(f"    Right side mean: {rho[right_mask].mean():.6f}")
        
        f.create_dataset("vel", data=vel)
        print(f"  Created vel dataset: {vel.shape}")
        
        f.create_dataset("Energy", data=Energy)
        print(f"  Created Energy dataset: {Energy.shape}")
        print(f"    Min: {Energy.min():.6f}, Max: {Energy.max():.6f}")
        print(f"    Left side mean: {Energy[left_mask].mean():.6f}")
        print(f"    Right side mean: {Energy[right_mask].mean():.6f}")
    
    print(f"\nSuccessfully created {filename}\n")

if __name__ == "__main__":
    # Create 2D Sod shock tube
    create_sod_shock_tube("IC.hdf5", num_seeds=150**3, extent=1.0, dimension=3)
    
    # Uncomment for 3D shock tube
    # create_sod_shock_tube("IC.hdf5", num_seeds=8000, extent=1.0, dimension=3)
