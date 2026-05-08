#!/usr/bin/env python3
"""
Fast 2D Voronoi-like animation using nearest-neighbor rasterization.
Each frame loads one snapshot_*.hdf5 file from an input directory.
"""

import argparse
import re
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree


def compute_quantity(quantity, rho, vel, energy):
    if quantity == 'rho':
        values = rho
        label = 'Density (rho)'
    elif quantity == 'vel_mag':
        values = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
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
        raise ValueError(f'Unknown quantity: {quantity}')

    return values, label


def discover_snapshots(input_dir, pattern):
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f'Input path is not a directory: {input_dir}')

    files = list(input_path.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f'No files found in {input_dir} matching pattern "{pattern}"'
        )

    # Numeric sort by snapshot index for names like snapshot_32.hdf5.
    num_re = re.compile(r'.*?(\d+)\.hdf5$')

    def snapshot_key(path):
        m = num_re.match(path.name)
        if m:
            return int(m.group(1))
        return float('inf')

    files.sort(key=lambda p: (snapshot_key(p), p.name))
    return files


def read_snapshot(snapshot_path, quantity):
    with h5py.File(snapshot_path, 'r') as f:
        seeds = f['cells/seeds'][:]
        header = f['header']
        dimension = int(header.attrs['dimension'])
        extent = float(header.attrs['extent'])

        if dimension != 2:
            raise ValueError(
                f'{snapshot_path.name}: expected 2D data, got {dimension}D'
            )

        rho = f['hydro/rho'][:]
        vel = f['hydro/vel'][:]
        energy = f['hydro/Energy'][:]

    values, label = compute_quantity(quantity, rho, vel, energy)
    return seeds, values, label, extent


def build_frame_image(seeds, values, extent, resolution):
    tree = cKDTree(seeds[:, :2])

    pixel_size = extent / resolution
    x_centers = np.linspace(pixel_size / 2, extent - pixel_size / 2, resolution)
    y_centers = np.linspace(pixel_size / 2, extent - pixel_size / 2, resolution)
    xv, yv = np.meshgrid(x_centers, y_centers)
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])

    _, indices = tree.query(grid_points, workers=-1)
    image = values[indices].reshape(resolution, resolution)
    return image


def animate_2d_fast(
    input_dir,
    output_file,
    quantity='rho',
    pattern='snapshot_*.hdf5',
    vmin=None,
    vmax=None,
    resolution=1024,
    cmap_name='viridis',
    dpi=120,
    fps=10,
    show_seeds=True,
):
    t_start = time.perf_counter()

    snapshots = discover_snapshots(input_dir, pattern)
    print(f'Found {len(snapshots)} snapshots')
    print(f'First: {snapshots[0].name}')
    print(f'Last : {snapshots[-1].name}')

    # First pass: gather min/max, labels, and extents for stable color mapping.
    mins = []
    maxs = []
    extents = []
    for snap in snapshots:
        seeds, values, label, extent = read_snapshot(snap, quantity)
        mins.append(values.min())
        maxs.append(values.max())
        extents.append(extent)

    if len(set(np.round(extents, 12))) != 1:
        raise ValueError('Snapshots have inconsistent domain extents.')

    extent = extents[0]

    if vmin is None:
        vmin = float(np.min(mins))
    if vmax is None:
        vmax = float(np.max(maxs))

    print(f'Plotting: {label}')
    print(f'Global value range: [{np.min(mins):.4e}, {np.max(maxs):.4e}]')
    print(f'Color range      : [{vmin:.4e}, {vmax:.4e}]')
    print(f'Resolution       : {resolution}x{resolution}')
    print(f'FPS              : {fps}')

    # Use a high-resolution colormap for a smoother colorbar gradient.
    cmap = plt.get_cmap(cmap_name, 4096)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 9))
    initial = np.zeros((resolution, resolution), dtype=float)
    im = ax.imshow(
        initial,
        origin='lower',
        extent=[0, extent, 0, extent],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        aspect='equal',
    )

    seed_scatter = None
    if show_seeds:
        seed_scatter = ax.scatter(
            [], [], s=2.0, c='white', edgecolors='none', alpha=0.6, zorder=5
        )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, extent)
    ax.set_ylim(0, extent)
    cbar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar_mappable.set_array([])
    cbar = plt.colorbar(cbar_mappable, ax=ax, label=label)

    def update(frame_idx):
        snap = snapshots[frame_idx]
        t0 = time.perf_counter()
        seeds, values, _, _ = read_snapshot(snap, quantity)
        image = build_frame_image(seeds, values, extent, resolution)
        im.set_data(image)

        if seed_scatter is not None:
            seed_scatter.set_offsets(seeds[:, :2])

        ax.set_title(
            f'{label} | {snap.name} ({frame_idx + 1}/{len(snapshots)})'
        )
        dt = time.perf_counter() - t0
        print(f'Frame {frame_idx + 1:4d}/{len(snapshots)} in {dt:.3f}s')

        artists = [im]
        if seed_scatter is not None:
            artists.append(seed_scatter)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=1000 / max(1, fps),
        blit=False,
        repeat=False,
    )

    output_path = Path(output_file)
    suffix = output_path.suffix.lower()

    plt.tight_layout()
    if suffix == '.gif':
        print('Note: GIF is palette-limited (256 colors); MP4 gives smoother gradients.')
        writer = animation.PillowWriter(fps=fps)
        ani.save(output_file, writer=writer, dpi=dpi)
    elif suffix == '.mp4':
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(output_file, writer=writer, dpi=dpi)
    else:
        raise ValueError('Output file must end with .gif or .mp4')

    plt.close(fig)

    t_total = time.perf_counter() - t_start
    print(f'Saved animation: {output_file}')
    print(f'Total time: {t_total:.3f}s')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Fast 2D Voronoi-like animation via nearest-neighbor rasterization '
            'for snapshot_*.hdf5 files.'
        )
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='../output',
        help='Input folder containing snapshot files (default: ../output)',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='animation.mp4',
        help='Output animation file (.gif or .mp4)',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='snapshot_*.hdf5',
        help='Glob pattern for snapshot files (default: snapshot_*.hdf5)',
    )
    parser.add_argument(
        '-q',
        '--quantity',
        type=str,
        default='rho',
        choices=['rho', 'vel_mag', 'vel_x', 'vel_y', 'energy', 'pressure'],
        help='Quantity to plot (default: rho)',
    )
    parser.add_argument(
        '-r',
        '--resolution',
        type=int,
        default=1024,
        help='Pixel resolution per axis (default: 1024)',
    )
    parser.add_argument('--vmin', type=float, default=None, help='Colorbar min')
    parser.add_argument('--vmax', type=float, default=None, help='Colorbar max')
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='Colormap name (default: viridis)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=120,
        help='Output DPI (default: 120)',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Animation frames per second (default: 10)',
    )
    parser.add_argument(
        '--no-seeds',
        action='store_true',
        help='Disable seed point overlay',
    )

    args = parser.parse_args()

    animate_2d_fast(
        input_dir=args.input,
        output_file=args.output,
        quantity=args.quantity,
        pattern=args.pattern,
        vmin=args.vmin,
        vmax=args.vmax,
        resolution=args.resolution,
        cmap_name=args.cmap,
        dpi=args.dpi,
        fps=args.fps,
        show_seeds=not args.no_seeds,
    )
