#!/usr/bin/env python3
"""
Single-frame counterpart to plot3D_anim.py: render one PNG of a 3D snapshot
from a fixed camera angle. Same kernel, same backends, same transfer function,
same gradient shading; just no orbit and no ffmpeg. The heavy lifting is
imported from plot3D_anim.

Example (GPU):
    python visualize/plot3D.py -i output/snapshot_1.hdf5 -o visualize/snap.png \
        --azim-deg 30 --elev-deg 20 --width 1920 --height 1080 --shade

Example (CPU debug):
    python visualize/plot3D.py -i output/snapshot_1.hdf5 -o visualize/snap.png \
        --cpu --width 320 --height 180 --n-steps 256
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

from plot3D_anim import (
    SnapshotProvider,
    add_common_render_args,
    orbit_camera,
    prepare_render_context,
    render_one_frame,
    select_backend,
)


def main():
    p = argparse.ArgumentParser(
        description='Single-PNG 3D Voronoi renderer. See plot3D_anim.py for '
                    'the orbit-animation version.')
    p.add_argument('-i', '--input', default='output/snapshot_1.hdf5',
                   help='HDF5 snapshot to render')
    p.add_argument('-o', '--output', default='visualize/render.png',
                   help='output PNG path')
    p.add_argument('--azim-deg', type=float, default=0.0,
                   help='camera azimuth around the box, in degrees '
                        '(0 = +X axis; matches phi=0 of the animation orbit)')
    p.add_argument('-n', '--n-ranks', type=int, default=None,
                   help='Load and concatenate this many per-rank files (treats '
                        '--input as a template; reads <stem>.rank_<r>.hdf5 for '
                        'r in 0..n-1). Omit for single-file mode.')

    add_common_render_args(p)
    args = p.parse_args()

    if args.color_quantity is None:
        args.color_quantity = args.quantity
    if args.color_quantity == args.quantity:
        if args.color_vmin is None:
            args.color_vmin = args.vmin
        if args.color_vmax is None:
            args.color_vmax = args.vmax

    # SnapshotProvider expects these attrs even in single-snap mode.
    args.start_snap = None
    args.end_snap = None

    backend = select_backend(args)

    print(f"[mode] single snapshot: {args.input}")
    if args.n_ranks is not None:
        print(f"[mode] multi-rank: concatenating {args.n_ranks} files")
    print(f"[fields] opacity='{args.quantity}'  color='{args.color_quantity}'")

    provider = SnapshotProvider(args)
    provider.load_arrays(provider.first_snap)
    extent = provider.extent

    print(f"[render][{backend.name}] image {args.width}x{args.height}, "
          f"{args.n_steps} steps/ray")
    ctx = prepare_render_context(provider, extent, args, backend)

    snap_idx = provider.first_snap
    snap = backend.fetch(provider, snap_idx)

    cam, fwd, right, up = orbit_camera(
        ctx['target'], ctx['radius'],
        np.deg2rad(args.elev_deg),
        np.deg2rad(args.azim_deg),
    )

    sim_t = provider._times.get(snap_idx, 0.0)
    t0 = time.time()
    frame = render_one_frame(backend, snap, extent, args, ctx,
                             cam, fwd, right, up, sim_t)
    print(f"[render] one frame in {time.time() - t0:.2f}s")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(frame).save(args.output)
    print(f"[render] saved → {args.output}")


if __name__ == '__main__':
    main()
