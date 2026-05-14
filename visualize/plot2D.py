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


def _rank_path(template, rank):
    """Insert `.rank_<rank>` before the `.hdf5` extension in `template`."""
    if template.endswith('.hdf5'):
        return f"{template[:-5]}.rank_{rank}.hdf5"
    return f"{template}.rank_{rank}.hdf5"


def load_snapshot(hdf5_file, n_ranks=None):
    """
    Load seeds + hydro fields from a single snapshot or a set of per-rank files.

    With n_ranks=None, opens `hdf5_file` directly. Otherwise treats it as a
    template and reads `<stem>.rank_<r>.hdf5` for r in range(n_ranks), then
    concatenates. dimension/extent are taken from rank 0.

    Returns (seeds, rho, vel, energy, dimension, extent, rank_ids), where
    rank_ids is an int array assigning each cell to its owning rank in the
    multi-rank case (None otherwise).
    """
    if n_ranks is None:
        files = [hdf5_file]
    else:
        files = [_rank_path(hdf5_file, r) for r in range(n_ranks)]

    seeds_parts, rho_parts, vel_parts, energy_parts = [], [], [], []
    counts = []
    dimension = None
    extent = None

    for fp in files:
        with h5py.File(fp, 'r') as f:
            if dimension is None:
                dimension = int(f['header'].attrs['dimension'])
                extent = float(f['header'].attrs['extent'])
            s = f['mesh/pos'][:]
            seeds_parts.append(s)
            counts.append(len(s))
            rho_parts.append(f['hydro/rho'][:])
            vel_parts.append(f['hydro/vel'][:])
            energy_parts.append(f['hydro/energy'][:])

    seeds = np.concatenate(seeds_parts, axis=0)
    rho = np.concatenate(rho_parts, axis=0)
    vel = np.concatenate(vel_parts, axis=0)
    energy = np.concatenate(energy_parts, axis=0)

    rank_ids = None
    if n_ranks is not None:
        rank_ids = np.repeat(np.arange(n_ranks, dtype=np.int32), counts)

    return seeds, rho, vel, energy, dimension, extent, rank_ids


def plot_2d_fast(hdf5_file, quantity='rho', output_file=None, vmin=None, vmax=None,
                 resolution=1024, cmap_name='viridis', dpi=150, show_seeds=False,
                 n_ranks=None, rank_colors=False):
    """
    Fast rasterized nearest-neighbor plot of 2D Voronoi mesh.

    Parameters
    ----------
    hdf5_file : str
        Path to mesh_output HDF5 file. When n_ranks is set, treated as a template:
        sibling files are derived by inserting `.rank_<r>` before the `.hdf5` extension.
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
    n_ranks : int or None
        If set, load from `n_ranks` per-rank files and concatenate.
    rank_colors : bool
        Color the seed overlay by owning rank. Requires show_seeds and n_ranks.
    """
    t_start = time.perf_counter()

    seeds, rho, vel, energy, dimension, extent, rank_ids = load_snapshot(
        hdf5_file, n_ranks)

    if dimension != 2:
        print(f"Error: This script is for 2D data only (got {dimension}D).")
        return False

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
    if n_ranks is not None:
        print(f"Multi-rank: concatenated {n_ranks} files")
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
        if rank_colors and rank_ids is not None:
            seed_cmap = plt.get_cmap('tab10' if n_ranks <= 10 else 'tab20')
            ax.scatter(seeds[:, 0], seeds[:, 1], s=4.0, c=rank_ids,
                       cmap=seed_cmap, vmin=0, vmax=max(n_ranks - 1, 1),
                       edgecolors='none', alpha=1.0, zorder=5)
        else:
            ax.scatter(seeds[:, 0], seeds[:, 1], s=0.1, c='white',
                       edgecolors='none', alpha=0.3, zorder=5)

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
                        help='Input HDF5 file (default: ../output/mesh_output.hdf5). '
                             'With -n, treated as a template; reads <stem>.rank_<r>.hdf5.')
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
    parser.add_argument('-n', '--n-ranks', type=int, default=None,
                        help='Load and concatenate this many per-rank files '
                             '(treats --input as a template; reads <stem>.rank_<r>.hdf5 '
                             'for r in 0..n-1). Omit for single-file mode.')
    parser.add_argument('--rank-colors', action='store_true',
                        help='Color the seed overlay by owning rank using a '
                             'categorical colormap. Requires --seeds and --n-ranks.')

    args = parser.parse_args()

    plot_2d_fast(args.input, args.quantity, args.output, args.vmin, args.vmax,
                 args.resolution, args.cmap, args.dpi, args.seeds,
                 args.n_ranks, args.rank_colors)
