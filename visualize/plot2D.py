#!/usr/bin/env python3
"""
Fast 2D Voronoi-like visualization using nearest-neighbor rasterization.
Instead of computing Voronoi polygons, each pixel is assigned to the nearest
seed point and colored by its hydro quantity. Handles ~1M cells efficiently.
"""

import argparse
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def plot_2d_fast(hdf5_file, quantity='rho', output_file=None, vmin=None, vmax=None,
                 resolution=1024, cmap_name='viridis', dpi=150, show_seeds=False):
    """
    Fast rasterized nearest-neighbor plot of 2D Voronoi mesh.

    Parameters
    ----------
    hdf5_file : str
        Path to mesh_output HDF5 file.
    quantity : str
        Hydro quantity to plot: 'rho', 'vel_mag', 'energy', 'pressure'.
    output_file : str or None
        Save path; if None, displays interactively.
    vmin, vmax : float or None
        Colorbar limits (defaults to data range).
    resolution : int
        Pixel resolution per axis (default 1024 → 1024x1024 image).
    cmap_name : str
        Matplotlib colormap name.
    dpi : int
        Output DPI when saving.
    show_seeds : bool
        Overlay seed points on the image.
    """
    t_start = time.perf_counter()

    with h5py.File(hdf5_file, 'r') as f:
        seeds = f['mesh/pos'][:]
        header = f['header']
        dimension = header.attrs['dimension']
        extent = header.attrs['extent']

        if dimension != 2:
            print(f"Error: This script is for 2D data only (got {dimension}D).")
            return False

        rho = f['hydro/rho'][:]
        vel = f['hydro/vel'][:]
        energy = f['hydro/energy'][:]

    # Compute quantity
    if quantity == 'rho':
        values = rho
        label = 'Density (ρ)'
    elif quantity == 'vel_mag':
        values = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        label = 'Velocity Magnitude |v|'
    elif quantity == 'vel_x':
        values = vel[:, 0]
        label = 'Velocity x'
    elif quantity == 'vel_y':
        values = vel[:, 1]
        label = 'Velocity y'
    elif quantity == 'energy':
        values = energy
        label = 'Energy'
    elif quantity == 'pressure':
        gamma = 5.0 / 3.0
        # Energy is the *total* energy density (internal + kinetic), so
        # pressure = (γ-1) * (E - 0.5 ρ |v|²); without the kinetic term we'd
        # just be plotting a scaled E.
        kinetic = 0.5 * rho * (vel[:, 0] ** 2 + vel[:, 1] ** 2)
        values = (gamma - 1.0) * (energy - kinetic)
        label = 'Pressure'
    else:
        raise ValueError(f"Unknown quantity: {quantity}")

    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    n_cells = len(seeds)
    print(f"Cells: {n_cells}")
    print(f"Domain: [0, {extent}]²")
    print(f"Plotting: {label}")
    print(f"Value range: [{values.min():.4e}, {values.max():.4e}]")
    print(f"Color range: [{vmin:.4e}, {vmax:.4e}]")
    print(f"Resolution: {resolution}x{resolution}")

    # --- Build KDTree from seed positions (2D) ---
    t0 = time.perf_counter()
    tree = cKDTree(seeds[:, :2])
    t1 = time.perf_counter()
    print(f"KDTree built in {t1 - t0:.3f}s")

    # --- Create pixel grid centers ---
    pixel_size = extent / resolution
    x_centers = np.linspace(pixel_size / 2, extent - pixel_size / 2, resolution)
    y_centers = np.linspace(pixel_size / 2, extent - pixel_size / 2, resolution)
    xv, yv = np.meshgrid(x_centers, y_centers)
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])

    # --- Query nearest seed for every pixel ---
    t0 = time.perf_counter()
    _, indices = tree.query(grid_points, workers=-1)
    t1 = time.perf_counter()
    print(f"Nearest-neighbor query in {t1 - t0:.3f}s")

    # --- Map values onto grid ---
    image = values[indices].reshape(resolution, resolution)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 9))

    cmap = plt.get_cmap(cmap_name)
    im = ax.imshow(image, origin='lower', extent=[0, extent, 0, extent],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
                   aspect='equal')

    if show_seeds:
        ax.scatter(seeds[:, 0], seeds[:, 1], s=0.1, c='white', edgecolors='none',
                   alpha=0.3, zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, extent)
    ax.set_ylim(0, extent)

    cbar = plt.colorbar(im, ax=ax, label=label)
    plt.title(f'{label}  ({n_cells} cells, {resolution}×{resolution} px)')

    t_total = time.perf_counter() - t_start
    print(f"Total time: {t_total:.3f}s")

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved to: {output_file}")
    else:
        plt.show()
    plt.close()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fast 2D Voronoi-like plot via nearest-neighbor rasterization')
    parser.add_argument('-i', '--input', type=str, default='../output/mesh_output.hdf5',
                        help='Input HDF5 file (default: ../output/mesh_output.hdf5)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output image file (if not specified, displays plot)')
    parser.add_argument('-q', '--quantity', type=str, default='rho',
                        choices=['rho', 'vel_mag', 'vel_x', 'vel_y', 'energy', 'pressure'],
                        help='Quantity to plot (default: rho)')
    parser.add_argument('-r', '--resolution', type=int, default=1024,
                        help='Pixel resolution per axis (default: 1024)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value for colorbar')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value for colorbar')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap name (default: viridis)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')
    parser.add_argument('--seeds', action='store_true',
                        help='Overlay seed points on the image')

    args = parser.parse_args()

    plot_2d_fast(args.input, args.quantity, args.output, args.vmin, args.vmax,
                 args.resolution, args.cmap, args.dpi, args.seeds)
