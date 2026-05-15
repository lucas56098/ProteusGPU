#!/usr/bin/env python3
"""
Fast 2D Voronoi-like animation using nearest-neighbor rasterization.
Each frame loads one snapshot_*.hdf5 file (single-rank) or one set of
snapshot_*.<r>.hdf5 files (multi-rank, with --n-ranks) from an input
directory.
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


def _rank_path(rank0_path, rank):
    """Given a path to a `*.rank_0.hdf5` file, return the sibling `*.rank_<rank>.hdf5`."""
    name = rank0_path.name
    if '.rank_0.hdf5' not in name:
        raise ValueError(
            f'Expected a `.rank_0.hdf5` anchor file, got: {name}'
        )
    return rank0_path.with_name(name.replace('.rank_0.hdf5', f'.rank_{rank}.hdf5'))


def discover_snapshots(input_dir, pattern, n_ranks=None):
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f'Input path is not a directory: {input_dir}')

    # In rank-aware mode, enumerate frames via their rank-0 files unless the
    # user overrode --pattern themselves.
    if n_ranks is not None and pattern == 'snapshot_*.hdf5':
        pattern = 'snapshot_*.rank_0.hdf5'

    files = list(input_path.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f'No files found in {input_dir} matching pattern "{pattern}"'
        )

    # Numeric sort by snapshot index. Matches both `snapshot_32.hdf5` and
    # `snapshot_32.0.hdf5`-style names.
    num_re = re.compile(r'.*?(\d+)(?:\.\d+)?\.hdf5$')

    def snapshot_key(path):
        m = num_re.match(path.name)
        if m:
            return int(m.group(1))
        return float('inf')

    files.sort(key=lambda p: (snapshot_key(p), p.name))
    return files


def read_snapshot(snapshot_path, quantity, n_ranks=None):
    if n_ranks is None:
        files = [snapshot_path]
    else:
        files = [_rank_path(snapshot_path, r) for r in range(n_ranks)]

    seeds_parts, rho_parts, vel_parts, energy_parts = [], [], [], []
    counts = []
    extent = None

    for fp in files:
        with h5py.File(fp, 'r') as f:
            dimension = int(f['header'].attrs['dimension'])
            if dimension != 2:
                raise ValueError(
                    f'{Path(fp).name}: expected 2D data, got {dimension}D'
                )
            if extent is None:
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

    values, label = compute_quantity(quantity, rho, vel, energy)
    return seeds, values, label, extent, rank_ids


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
    n_ranks=None,
    rank_colors=False,
):
    t_start = time.perf_counter()

    snapshots = discover_snapshots(input_dir, pattern, n_ranks=n_ranks)
    print(f'Found {len(snapshots)} snapshots')
    print(f'First: {snapshots[0].name}')
    print(f'Last : {snapshots[-1].name}')
    if n_ranks is not None:
        print(f'Multi-rank mode: concatenating {n_ranks} files per frame')

    # First pass: gather min/max, labels, and extents for stable color mapping.
    mins = []
    maxs = []
    extents = []
    for snap in snapshots:
        seeds, values, label, extent, _ = read_snapshot(snap, quantity, n_ranks=n_ranks)
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
    use_rank_colors = show_seeds and rank_colors and n_ranks is not None
    if show_seeds:
        if use_rank_colors:
            seed_cmap = plt.get_cmap('tab10' if n_ranks <= 10 else 'tab20')
            # Initialize with one dummy point so the cmap+norm machinery is wired up;
            # set_offsets/set_array will overwrite it on the first frame.
            seed_scatter = ax.scatter(
                [0.0], [0.0], c=[0], s=4.0, cmap=seed_cmap,
                vmin=0, vmax=max(n_ranks - 1, 1),
                edgecolors='none', alpha=1.0, zorder=5,
            )
        else:
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
        seeds, values, _, _, rank_ids = read_snapshot(snap, quantity, n_ranks=n_ranks)
        image = build_frame_image(seeds, values, extent, resolution)
        im.set_data(image)

        if seed_scatter is not None:
            seed_scatter.set_offsets(seeds[:, :2])
            if use_rank_colors:
                seed_scatter.set_array(rank_ids)

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
        help='Glob pattern for snapshot files (default: snapshot_*.hdf5; '
             'auto-switched to snapshot_*.0.hdf5 when --n-ranks is set)',
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
    parser.add_argument(
        '-n',
        '--n-ranks',
        type=int,
        default=None,
        help=(
            'Load and concatenate this many per-rank files per frame. '
            'Frames are enumerated from the matching `*.0.hdf5` files; '
            'sibling rank files are derived by swapping `0` -> `<r>`. '
            'Omit for single-file mode.'
        ),
    )
    parser.add_argument(
        '--rank-colors',
        action='store_true',
        help=(
            'Color the seed overlay by owning rank using a categorical '
            'colormap. Requires --n-ranks; ignored if seeds are disabled.'
        ),
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
        n_ranks=args.n_ranks,
        rank_colors=args.rank_colors,
    )
