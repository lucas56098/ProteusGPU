#!/usr/bin/env python3
"""
Volume renderer + camera-orbit animation for 3D snapshots. For a single still
PNG (no orbit, no ffmpeg), see plot3D.py — it imports the helpers below.

Two backends, selected at runtime:
  * GPU (default): CUDA raycaster compiled via CuPy. At every ray step a
    spatial-hash NN lookup resolves the nearest Voronoi seed; the cell's value
    drives opacity and color directly.
  * CPU debug (--cpu): scipy cKDTree NN + numpy front-to-back composite.
    Much slower (50-200x); for correctness checks and machines without a GPU.
    Keep --width / --height / --frames small.

Both backends do *exact* Voronoi rendering — no auxiliary value-grid, no
trilinear blur across cell boundaries. The disk cache stores per-snapshot
seed positions and per-quantity values (small, shared across all renderings
of the same simulation), not a resampled grid.

Example (GPU):
    python visualize/plot3D_anim.py -i output/snapshot_1.hdf5 \
        -o visualize/render.mp4 --frames 240 --width 1280 --height 720

Example (time series, linear mapping, jet along y kept upright, physical labels):
    python visualize/plot3D_anim.py --snap-pattern output/snapshot_{}.hdf5 \
        --start-snap 0 --end-snap 239 --frames 240 --fps 15 \
        -q rho --color-quantity vel_mag --linear --vmin 0 --vmax 2.271 \
        --color-vmin 0 --color-vmax 0.6 --cmap afmhot --orbit-axis y \
        --brightness 13 --brightness-gamma 2.8 --smbh \
        --length-unit kpc --velocity-unit km/s --time-unit Gyr

Example (CPU debug):
    python visualize/plot3D_anim.py -i output/snapshot_1.hdf5 --cpu \
        --width 320 --height 180 --frames 8 --n-steps 256
"""

import argparse
import os
import subprocess
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _rank_path(template, rank):
    """Insert `.<rank>` before the `.hdf5` extension in `template`."""
    if template.endswith('.hdf5'):
        return f"{template[:-5]}.{rank}.hdf5"
    return f"{template}.{rank}.hdf5"


def load_snapshot(path, n_ranks=None):
    """Read one snapshot. With n_ranks set, `path` is treated as a template and
    sibling `<stem>.<r>.hdf5` files are concatenated (dimension/time are taken
    from rank 0). extent is the hardcoded [0,1]^D code domain."""
    files = [path] if n_ranks is None else [_rank_path(path, r) for r in range(n_ranks)]

    seeds_parts, rho_parts, vel_parts, E_parts = [], [], [], []
    dim = None
    extent = 1.0  # domain is hardcoded to [0,1]^D in the code
    sim_time = None

    units = None
    for fp in files:
        with h5py.File(fp, 'r') as f:
            if dim is None:
                dim = int(f['header'].attrs['dimension'])
                sim_time = float(f['header'].attrs['time'])
                units = _read_units(f['header'].attrs)
            seeds_parts.append(f['mesh/pos'][:])
            rho_parts.append(f['hydro/rho'][:])
            vel_parts.append(f['hydro/vel'][:])
            E_parts.append(f['hydro/energy'][:])

    if dim != 3:
        raise ValueError(f"need 3D snapshot, got {dim}D")

    seeds = np.concatenate(seeds_parts, axis=0)
    rho = np.concatenate(rho_parts, axis=0)
    vel = np.concatenate(vel_parts, axis=0)
    E = np.concatenate(E_parts, axis=0)
    return seeds, rho, vel, E, extent, sim_time, units


QUANTITIES = ('rho', 'vel_mag', 'energy', 'temperature', 'xray')


# Physical-unit display factors. Each entry is (unit-in-cgs, suffix).
# 'code' means "no conversion" — display value 1:1 in whatever the sim uses.
_LEN_CM   = {'code': None,  'pc':   3.0857e18, 'kpc':   3.0857e21, 'Mpc':   3.0857e24}
_VEL_CMPS = {'code': None,  'cm/s': 1.0,       'km/s':  1.0e5}
_TIME_S   = {'code': None,  's':    1.0,       'yr':    3.1557e7,  'Myr':   3.1557e13, 'Gyr': 3.1557e16}


def _read_units(header_attrs):
    """cgs unit factors from a snapshot header, or None when the header has
    none (the code writes them only with ASTRO_PHYSICS). Time unit is derived
    as UnitLength / UnitVelocity."""
    if 'UnitLength_in_cm' not in header_attrs or 'UnitVelocity_in_cm_per_s' not in header_attrs:
        return None
    u_len = float(header_attrs['UnitLength_in_cm'])
    u_vel = float(header_attrs['UnitVelocity_in_cm_per_s'])
    return dict(length_cm=u_len, velocity_cm_per_s=u_vel,
                time_s=u_len / u_vel if u_vel > 0 else 1.0)


def _display_factor(kind, choice, units):
    """Given a unit kind ('length'/'velocity'/'time'), a CLI choice, and the
    per-snapshot cgs unit dict, return (multiplicative_factor, suffix). A
    factor of 1.0 with '' suffix means 'display raw code values'."""
    if choice == 'code' or units is None:
        return 1.0, ''
    table, key = {
        'length':   (_LEN_CM,   'length_cm'),
        'velocity': (_VEL_CMPS, 'velocity_cm_per_s'),
        'time':     (_TIME_S,   'time_s'),
    }[kind]
    if choice not in table or table[choice] is None:
        return 1.0, ''
    return units[key] / table[choice], choice


def _extract_quantity(name, rho, vel, E):
    if name == 'rho':
        return rho
    if name == 'vel_mag':
        return np.sqrt((vel ** 2).sum(axis=1))
    if name == 'energy':
        return E
    if name == 'temperature':
        # E is total energy per volume: E = rho*u + 0.5*rho*|v|^2
        # Return specific internal energy u ∝ T (up to μ/k_B in code units).
        v2 = (vel ** 2).sum(axis=1)
        return (E - 0.5 * rho * v2) / np.maximum(rho, 1e-30)
    if name == 'xray':
        # Mock thermal-bremsstrahlung emissivity ∝ ρ² · √T.
        v2 = (vel ** 2).sum(axis=1)
        u = (E - 0.5 * rho * v2) / np.maximum(rho, 1e-30)
        return rho * rho * np.sqrt(np.maximum(u, 1e-30))
    raise ValueError(f"unknown quantity: {name}")


def build_transfer_function(name='fire', n=256, opacity_gamma=2.0,
                            tf_mu1=None, tf_s1=0.05,
                            tf_mu2=None, tf_s2=0.05):
    """Return [n,4] float32 RGBA LUT. Alpha here is the opacity multiplier κ
    (not the final composited alpha) — gets fed to (1 - exp(-κ*dt)).

    Opacity modes:
    - monotonic (default, both mu unset): κ(t) = t ** opacity_gamma
    - single Gaussian bump (only one mu set): κ(t) = exp(-(t-μ)²/(2σ²))
    - double Gaussian bump (both mu set): sum of the two Gaussians (additive
      where they overlap; each peaks at 1). Useful for showing two populations
      — e.g. μ1 near the cold-gas band, μ2 near the hot cocoon band.
    """
    t = np.linspace(0, 1, n)

    if name == 'fire':
        r = np.clip(2.0 * t - 0.1, 0, 1)
        g = np.clip(2.5 * t - 1.0, 0, 1)
        b = np.clip(3.0 * t - 2.3, 0, 1) + 0.35 * np.exp(-((t - 0.05) / 0.08)**2)
        b = np.clip(b, 0, 1)
    elif name == 'coolwarm_dark':
        # Diverging cmap with a *dark* middle (Berlin-style pivot). Blue half
        # samples GnBu, red half samples YlOrRd — so each side has its own
        # tonal variation instead of a flat blue↔red interpolation. The
        # brightness envelope (d^0.6, d = 2|t-0.5|) fades both sides to black
        # at the pivot so neutral gas disappears into the background.
        import matplotlib as mpl
        gnbu  = mpl.colormaps['GnBu']
        ylord = mpl.colormaps['YlOrRd']

        d = 2.0 * np.abs(t - 0.5)           # 1 at edges, 0 at pivot
        env = d ** 0.6                       # soft-shoulder brightness fade

        # Per-side palette sweep. Blue half skips GnBu's yellow-green bottom
        # (pos>=0.42) so near-pivot is a light teal-blue, edge is a saturated
        # blue but stops before near-black navy that would vanish on black bg.
        left  = t <= 0.5
        right = ~left
        pos = np.empty_like(t)
        pos[left]  = 0.65 + 0.30 * d[left]   # 0.65 near pivot (clean blue) → 0.95 (deep navy) at edge
        pos[right] = 0.20 + 0.55 * d[right]  # 0.20 near pivot (bright yellow) → 0.75 (fiery red-orange) at edge

        r = np.empty_like(t)
        g = np.empty_like(t)
        b = np.empty_like(t)

        for mask, palette in ((left, gnbu), (right, ylord)):
            rgba = palette(pos[mask])        # (N,4) samples of chosen palette
            r[mask] = rgba[:, 0] * env[mask]
            g[mask] = rgba[:, 1] * env[mask]
            b[mask] = rgba[:, 2] * env[mask]
    elif name == 'ice':
        r = np.clip(1.4 * t - 0.5, 0, 1)
        g = np.clip(1.2 * t - 0.05, 0, 1)
        b = np.clip(0.3 + 0.7 * t, 0, 1)
    elif name == 'aurora':
        r = np.clip(2.0 * t - 1.0, 0, 1)
        g = np.clip(1.6 * t + 0.05, 0, 1)
        b = np.clip(1.4 - 1.6 * t + 0.5 * np.exp(-((t-0.85)/0.1)**2), 0, 1)
    else:
        import matplotlib as mpl
        rgba = mpl.colormaps[name](t)
        r, g, b = rgba[:, 0], rgba[:, 1], rgba[:, 2]

    if tf_mu1 is None and tf_mu2 is None:
        a = t ** opacity_gamma
    else:
        a = np.zeros_like(t)
        if tf_mu1 is not None:
            a = a + np.exp(-((t - tf_mu1) ** 2) / (2.0 * max(tf_s1, 1e-4) ** 2))
        if tf_mu2 is not None:
            a = a + np.exp(-((t - tf_mu2) ** 2) / (2.0 * max(tf_s2, 1e-4) ** 2))
    return np.stack([r, g, b, a], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial-hash NN builder (host). K=1 specialization of the structure used by
# src/knn/knn.h: voxelize seeds into a G^3 uniform grid, then walk a
# ring-sorted offset table when querying. Fed to the GPU kernel as flat
# numpy arrays.
# ---------------------------------------------------------------------------

def build_spatial_hash(seeds, extent, G, R):
    """Return all the buffers the GPU NN kernel needs.

    Parameters
    ----------
    seeds   : (N, 3) seed positions in [0, extent]^3.
    extent  : box side length.
    G       : voxel grid resolution. Aim ~1 seed per voxel.
    R       : max ring radius for the offset table (in voxels).

    Returns a dict of host numpy arrays, ready to be cp.asarray'd:
      seeds_sorted   (3N,)   float32  — flat (sx,sy,sz,...) reordered by voxel
      perm           (N,)    int32    — perm[sorted] = original seed index
      voxel_count    (G^3,)  int32    — seeds per voxel
      voxel_ptr      (G^3,)  int32    — first seed offset in seeds_sorted per voxel
      cell_off_d{x,y,z} (M,) int32    — voxel-offset (dx,dy,dz), ring-sorted
      cell_off_dist2    (M,) float32  — squared world-units min-distance per offset
      G, voxel_size, N_offsets — scalar metadata
    """
    voxel_size = extent / G
    inv_vsize = 1.0 / voxel_size

    ix = np.clip(np.floor(seeds[:, 0] * inv_vsize).astype(np.int32), 0, G - 1)
    iy = np.clip(np.floor(seeds[:, 1] * inv_vsize).astype(np.int32), 0, G - 1)
    iz = np.clip(np.floor(seeds[:, 2] * inv_vsize).astype(np.int32), 0, G - 1)
    vidx = (ix * G + iy) * G + iz

    voxel_count = np.bincount(vidx, minlength=G**3).astype(np.int32)
    voxel_ptr = np.zeros(G**3, dtype=np.int32)
    voxel_ptr[1:] = np.cumsum(voxel_count[:-1])

    perm = np.argsort(vidx, kind='stable').astype(np.int32)
    seeds_sorted = np.ascontiguousarray(seeds[perm], dtype=np.float32)

    drange = np.arange(-R, R + 1, dtype=np.int32)
    DX, DY, DZ = np.meshgrid(drange, drange, drange, indexing='ij')
    dx = DX.ravel(); dy = DY.ravel(); dz = DZ.ravel()
    # worst-case min-distance from any point in origin voxel to any point in
    # the (dx,dy,dz)-offset voxel: per-axis max(0, |d| - 1) in voxel units.
    bx = np.maximum(np.abs(dx) - 1, 0).astype(np.float32)
    by = np.maximum(np.abs(dy) - 1, 0).astype(np.float32)
    bz = np.maximum(np.abs(dz) - 1, 0).astype(np.float32)
    dist2 = (bx * bx + by * by + bz * bz) * (voxel_size * voxel_size)
    sort_idx = np.argsort(dist2, kind='stable')
    dx = dx[sort_idx]; dy = dy[sort_idx]; dz = dz[sort_idx]
    dist2 = dist2[sort_idx].astype(np.float32)

    return dict(
        seeds_sorted=seeds_sorted.reshape(-1),
        perm=perm,
        voxel_count=voxel_count,
        voxel_ptr=voxel_ptr,
        cell_off_dx=dx.astype(np.int32),
        cell_off_dy=dy.astype(np.int32),
        cell_off_dz=dz.astype(np.int32),
        cell_off_dist2=dist2,
        G=int(G),
        voxel_size=float(voxel_size),
        N_offsets=int(len(dx)),
    )


# ---------------------------------------------------------------------------
# CUDA kernel. find_nearest_seed walks ring-sorted voxel offsets, early-
# terminating once the ring's lower-bound exceeds best-so-far. The raycast
# kernel does slab intersection, then a front-to-back emission-absorption
# composite where every step's value comes from the nearest Voronoi seed.
#
# Optional gradient shading (do_shade=1): at each step, take 6 extra NN
# lookups at p ± h*ê on the opacity field, build a log-space central-diff
# gradient, normalize, and shade RGB by ambient + (1-ambient)*max(0,-n·l).
# h must be on the order of the local cell size — the field is piecewise-
# constant per Voronoi cell, so a stencil smaller than that returns 0.
# ---------------------------------------------------------------------------

KERNEL_SRC = r"""
extern "C" {

__device__ inline int find_nearest_seed(
    float px, float py, float pz,
    const float* __restrict__ seeds_sorted,
    int G, float voxel_size,
    const int* __restrict__ voxel_count,
    const int* __restrict__ voxel_ptr,
    const int* __restrict__ cell_off_dx,
    const int* __restrict__ cell_off_dy,
    const int* __restrict__ cell_off_dz,
    const float* __restrict__ cell_off_dist2,
    int N_offsets
) {
    float inv_vsize = 1.f / voxel_size;
    int ix = (int)floorf(px * inv_vsize);
    int iy = (int)floorf(py * inv_vsize);
    int iz = (int)floorf(pz * inv_vsize);
    if (ix < 0) ix = 0; else if (ix >= G) ix = G - 1;
    if (iy < 0) iy = 0; else if (iy >= G) iy = G - 1;
    if (iz < 0) iz = 0; else if (iz >= G) iz = G - 1;

    int   best_idx = -1;
    float best_d2  = 1e30f;

    for (int k = 0; k < N_offsets; ++k) {
        if (cell_off_dist2[k] > best_d2) break;
        int nx = ix + cell_off_dx[k];
        int ny = iy + cell_off_dy[k];
        int nz = iz + cell_off_dz[k];
        if (nx < 0 || nx >= G || ny < 0 || ny >= G || nz < 0 || nz >= G) continue;
        int voxel = (nx * G + ny) * G + nz;
        int base = voxel_ptr[voxel];
        int n    = voxel_count[voxel];
        for (int s = 0; s < n; ++s) {
            int sidx = base + s;
            float sx = seeds_sorted[3 * sidx + 0];
            float sy = seeds_sorted[3 * sidx + 1];
            float sz = seeds_sorted[3 * sidx + 2];
            float ex = sx - px, ey = sy - py, ez = sz - pz;
            float d2 = ex * ex + ey * ey + ez * ez;
            if (d2 < best_d2) {
                best_d2  = d2;
                best_idx = sidx;
            }
        }
    }
    return best_idx;
}

__global__ void raycast_voronoi(
    const float* __restrict__ seeds_sorted,
    const float* __restrict__ values_op_sorted,
    const float* __restrict__ values_c_sorted,
    int G, float voxel_size,
    const int* __restrict__ voxel_count,
    const int* __restrict__ voxel_ptr,
    const int* __restrict__ cell_off_dx,
    const int* __restrict__ cell_off_dy,
    const int* __restrict__ cell_off_dz,
    const float* __restrict__ cell_off_dist2,
    int N_offsets,
    float ext,
    float cam_x, float cam_y, float cam_z,
    float fwd_x, float fwd_y, float fwd_z,
    float right_x, float right_y, float right_z,
    float up_x, float up_y, float up_z,
    float fov,
    int W, int H,
    int n_steps,
    // These are "map-space" (log(vmin/vmax) if log_map==1, else raw vmin/vmax).
    float log_vmin_op, float log_vmax_op,
    float log_vmin_c,  float log_vmax_c,
    int log_map,
    float opacity_scale,
    const float4* __restrict__ tf, int tf_n,
    float bg_r, float bg_g, float bg_b,
    float box_r, float box_g, float box_b,
    float brightness,
    float brightness_gamma,
    int do_shade,
    float light_x, float light_y, float light_z,
    float ambient, float shade_h,
    unsigned char* out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;

    float aspect = (float)W / (float)H;
    float u = (2.f * ((float)i + 0.5f) / (float)W - 1.f) * aspect * fov;
    float v = -(2.f * ((float)j + 0.5f) / (float)H - 1.f) * fov;

    float dx = fwd_x + u * right_x + v * up_x;
    float dy = fwd_y + u * right_y + v * up_y;
    float dz = fwd_z + u * right_z + v * up_z;
    float inv = rsqrtf(dx*dx + dy*dy + dz*dz);
    dx *= inv; dy *= inv; dz *= inv;

    // slab ray-box intersection on [0, ext]^3
    float t0 = -1e30f, t1 = 1e30f;
    float ox[3] = {cam_x, cam_y, cam_z};
    float od[3] = {dx, dy, dz};
    bool hit = true;
    for (int k = 0; k < 3; ++k) {
        if (fabsf(od[k]) < 1e-8f) {
            if (ox[k] < 0.f || ox[k] > ext) { hit = false; break; }
        } else {
            float ta = (0.f - ox[k]) / od[k];
            float tb = (ext - ox[k]) / od[k];
            if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
            if (ta > t0) t0 = ta;
            if (tb < t1) t1 = tb;
        }
    }
    int pix = 3 * (j * W + i);
    if (!hit || t0 >= t1 || t1 < 0.f) {
        out[pix+0] = (unsigned char)(255.f * bg_r);
        out[pix+1] = (unsigned char)(255.f * bg_g);
        out[pix+2] = (unsigned char)(255.f * bg_b);
        return;
    }
    if (t0 < 0.f) t0 = 0.f;

    float dt = (t1 - t0) / (float)n_steps;
    // historical opacity-density factor: keeps default --opacity-scale visually
    // calibrated to the old grid-based renderer with --grid 256.
    float ds = dt / ext * 256.f;

    float lx = 0.f, ly = 0.f, lz = 0.f;
    if (do_shade) {
        float lN = rsqrtf(light_x*light_x + light_y*light_y + light_z*light_z + 1e-20f);
        lx = light_x * lN; ly = light_y * lN; lz = light_z * lN;
    }

    float R = 0.f, Gc = 0.f, B = 0.f, A = 0.f;

    for (int s = 0; s < n_steps; ++s) {
        if (A > 0.995f) break;
        float t  = t0 + ((float)s + 0.5f) * dt;
        float px = cam_x + t * dx;
        float py = cam_y + t * dy;
        float pz = cam_z + t * dz;

        int seed_idx = find_nearest_seed(
            px, py, pz,
            seeds_sorted, G, voxel_size,
            voxel_count, voxel_ptr,
            cell_off_dx, cell_off_dy, cell_off_dz, cell_off_dist2,
            N_offsets
        );
        if (seed_idx < 0) continue;

        float val_op = values_op_sorted[seed_idx];
        float mv_op  = log_map ? logf(fmaxf(val_op, 1e-30f)) : val_op;
        float lv_op  = (mv_op - log_vmin_op) / (log_vmax_op - log_vmin_op);
        lv_op = fminf(fmaxf(lv_op, 0.f), 1.f);
        int   idx_op = (int)(lv_op * (float)(tf_n - 1));
        float alpha_factor = tf[idx_op].w;

        float val_c = values_c_sorted[seed_idx];
        float mv_c  = log_map ? logf(fmaxf(val_c, 1e-30f)) : val_c;
        float lv_c  = (mv_c - log_vmin_c) / (log_vmax_c - log_vmin_c);
        lv_c = fminf(fmaxf(lv_c, 0.f), 1.f);
        int   idx_c = (int)(lv_c * (float)(tf_n - 1));
        float4 c = tf[idx_c];

        float shade = 1.f;
        if (do_shade) {
            // Central diff in log opacity over a stencil large enough to cross
            // Voronoi cell boundaries (h ~ local cell size). Each axis costs 2
            // extra NN lookups via the same spatial hash.
            float vxp = values_op_sorted[find_nearest_seed(
                px + shade_h, py, pz, seeds_sorted, G, voxel_size,
                voxel_count, voxel_ptr, cell_off_dx, cell_off_dy, cell_off_dz,
                cell_off_dist2, N_offsets)];
            float vxm = values_op_sorted[find_nearest_seed(
                px - shade_h, py, pz, seeds_sorted, G, voxel_size,
                voxel_count, voxel_ptr, cell_off_dx, cell_off_dy, cell_off_dz,
                cell_off_dist2, N_offsets)];
            float vyp = values_op_sorted[find_nearest_seed(
                px, py + shade_h, pz, seeds_sorted, G, voxel_size,
                voxel_count, voxel_ptr, cell_off_dx, cell_off_dy, cell_off_dz,
                cell_off_dist2, N_offsets)];
            float vym = values_op_sorted[find_nearest_seed(
                px, py - shade_h, pz, seeds_sorted, G, voxel_size,
                voxel_count, voxel_ptr, cell_off_dx, cell_off_dy, cell_off_dz,
                cell_off_dist2, N_offsets)];
            float vzp = values_op_sorted[find_nearest_seed(
                px, py, pz + shade_h, seeds_sorted, G, voxel_size,
                voxel_count, voxel_ptr, cell_off_dx, cell_off_dy, cell_off_dz,
                cell_off_dist2, N_offsets)];
            float vzm = values_op_sorted[find_nearest_seed(
                px, py, pz - shade_h, seeds_sorted, G, voxel_size,
                voxel_count, voxel_ptr, cell_off_dx, cell_off_dy, cell_off_dz,
                cell_off_dist2, N_offsets)];
            float gxv = logf(fmaxf(vxp, 1e-30f)) - logf(fmaxf(vxm, 1e-30f));
            float gyv = logf(fmaxf(vyp, 1e-30f)) - logf(fmaxf(vym, 1e-30f));
            float gzv = logf(fmaxf(vzp, 1e-30f)) - logf(fmaxf(vzm, 1e-30f));
            float gN = rsqrtf(gxv*gxv + gyv*gyv + gzv*gzv + 1e-20f);
            float nx = gxv * gN, ny = gyv * gN, nz = gzv * gN;
            // gradient points up-rho; "surface" normal points down-rho
            float diff = fmaxf(0.f, -(nx*lx + ny*ly + nz*lz));
            shade = ambient + (1.f - ambient) * diff;
        }

        float a = 1.f - expf(-alpha_factor * opacity_scale * ds);
        float w = (1.f - A) * a;
        R  += w * c.x * shade;
        Gc += w * c.y * shade;
        B  += w * c.z * shade;
        A  += w;
    }

    R  += (1.f - A) * box_r;
    Gc += (1.f - A) * box_g;
    B  += (1.f - A) * box_b;

    R  = R  / (1.f + R)  * 1.2f;
    Gc = Gc / (1.f + Gc) * 1.2f;
    B  = B  / (1.f + B)  * 1.2f;

    R  *= brightness;
    Gc *= brightness;
    B  *= brightness;

    // display-gamma mid-tone lift: powf(x, 1/g). g==1 is a no-op; g>1 lifts
    // mid-tones without pulling highlights or shadows off their endpoints.
    if (fabsf(brightness_gamma - 1.f) > 1e-4f) {
        float inv_g = 1.f / brightness_gamma;
        R  = powf(fmaxf(R,  0.f), inv_g);
        Gc = powf(fmaxf(Gc, 0.f), inv_g);
        B  = powf(fmaxf(B,  0.f), inv_g);
    }

    R  = fminf(fmaxf(R,  0.f), 1.f);
    Gc = fminf(fmaxf(Gc, 0.f), 1.f);
    B  = fminf(fmaxf(B,  0.f), 1.f);
    out[pix+0] = (unsigned char)(255.f * R);
    out[pix+1] = (unsigned char)(255.f * Gc);
    out[pix+2] = (unsigned char)(255.f * B);
}

}
"""


# ---------------------------------------------------------------------------
# CPU debug raycaster: row-strip vectorized numpy compositor on top of a
# single batched cKDTree.query per strip. Mirrors the GPU kernel
# semantics — slab intersection, front-to-back composite, same TF.
# ---------------------------------------------------------------------------

CPU_TILE_H = 32  # rows per batched cKDTree.query; caps peak memory


def _cpu_raycast_strip(tree, values_op, values_c,
                       sample_pos, dt_over_ext, opacity_scale,
                       log_vmin_op, log_vmax_op, log_vmin_c, log_vmax_c,
                       tf_alpha, tf_rgb, tf_n,
                       box_bg, brightness=1.0, brightness_gamma=1.0,
                       shade=None, log_map=True):
    """Composite one strip given prebuilt sample positions. Returns (tile_h, W, 3)
    float32 in [0, 1] over hit pixels; non-hit values are arbitrary and must be
    masked by the caller. dt_over_ext is per-step distance / extent.

    If `shade` is given (dict with 'light', 'ambient', 'h_world'), 6 extra
    stencil queries on the opacity field are batched into the tree query and
    used to compute a log-space central-diff gradient → directional shade."""
    tile_h, W, n_steps, _ = sample_pos.shape

    if shade is not None:
        h = float(shade['h_world'])
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [+h,  0.0, 0.0], [-h,  0.0, 0.0],
            [0.0, +h,  0.0], [0.0, -h,  0.0],
            [0.0, 0.0, +h ], [0.0, 0.0, -h ],
        ], dtype=np.float32)  # (7, 3)
        # (tile_h, W, n_steps, 7, 3)
        stencil_pos = sample_pos[..., None, :] + offsets[None, None, None, :, :]
        flat_pos = stencil_pos.reshape(-1, 3)
        _, idx = tree.query(flat_pos, k=1, workers=-1)
        val_op_all = values_op[idx].reshape(tile_h, W, n_steps, 7)
        val_op = val_op_all[..., 0]
        idx_center = idx.reshape(tile_h, W, n_steps, 7)[..., 0]
        val_c = values_c[idx_center]
    else:
        flat_pos = sample_pos.reshape(-1, 3)
        _, idx = tree.query(flat_pos, k=1, workers=-1)
        val_op = values_op[idx].reshape(tile_h, W, n_steps)
        val_c  = values_c[idx].reshape(tile_h, W, n_steps)

    log_range_op = log_vmax_op - log_vmin_op
    log_range_c  = log_vmax_c  - log_vmin_c

    mv_op = np.log(np.maximum(val_op, 1e-30)) if log_map else val_op
    lv_op = (mv_op - log_vmin_op) / log_range_op
    lv_op = np.clip(lv_op, 0.0, 1.0)
    idx_op = (lv_op * (tf_n - 1)).astype(np.int32)
    alpha_factor = tf_alpha[idx_op]

    mv_c = np.log(np.maximum(val_c, 1e-30)) if log_map else val_c
    lv_c = (mv_c - log_vmin_c) / log_range_c
    lv_c = np.clip(lv_c, 0.0, 1.0)
    idx_c = (lv_c * (tf_n - 1)).astype(np.int32)
    c_rgb = tf_rgb[idx_c]  # (tile_h, W, n_steps, 3)

    if shade is not None:
        log_op = np.log(np.maximum(val_op_all, 1e-30))
        gxv = log_op[..., 1] - log_op[..., 2]
        gyv = log_op[..., 3] - log_op[..., 4]
        gzv = log_op[..., 5] - log_op[..., 6]
        gN = 1.0 / np.sqrt(gxv * gxv + gyv * gyv + gzv * gzv + 1e-20)
        nx = gxv * gN; ny = gyv * gN; nz = gzv * gN
        light = np.asarray(shade['light'], dtype=np.float32)
        lN = 1.0 / np.sqrt(float(np.dot(light, light)) + 1e-20)
        lx, ly, lz = light * lN
        diff = np.maximum(0.0, -(nx * lx + ny * ly + nz * lz))
        ambient = float(shade['ambient'])
        shade_arr = (ambient + (1.0 - ambient) * diff)[..., None]  # (...,n_steps,1)
        c_rgb = c_rgb * shade_arr

    # match GPU kernel: ds = (dt / extent) * 256, with the 256 a historical
    # opacity-density constant kept for visual parity with the old renderer.
    ds = dt_over_ext * 256.0
    a = 1.0 - np.exp(-alpha_factor * opacity_scale * ds[..., None])

    # Front-to-back composite via cumulative transmittance.
    # T[s] = product over s'<s of (1 - a[s']). Then w[s] = T[s] * a[s].
    one_minus_a = 1.0 - a
    cum = np.cumprod(one_minus_a, axis=-1)
    T = np.concatenate([np.ones((tile_h, W, 1), dtype=cum.dtype), cum[..., :-1]], axis=-1)
    w = T * a
    rgb = np.einsum('ijk,ijkc->ijc', w, c_rgb)
    A = np.sum(w, axis=-1)

    box_arr = np.asarray(box_bg, dtype=np.float32)
    rgb = rgb + (1.0 - A)[..., None] * box_arr
    rgb = rgb / (1.0 + rgb) * 1.2
    rgb = rgb * float(brightness)
    if abs(brightness_gamma - 1.0) > 1e-4:
        rgb = np.power(np.maximum(rgb, 0.0), 1.0 / float(brightness_gamma))
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


# ---------------------------------------------------------------------------
# Backends. Both expose the same small interface used by render_animation:
#   .upload_tf(tf_np)            — push the [n,4] RGBA LUT to wherever it lives.
#   .fetch(provider, snap_idx)   -> backend-specific snapshot data dict.
#   .render_frame(...)           -> uint8 [H,W,3] np.ndarray.
# ---------------------------------------------------------------------------

class GPUBackend:
    name = 'gpu'

    def __init__(self, args):
        import cupy as cp
        self.cp = cp
        self.module = cp.RawModule(code=KERNEL_SRC)
        self.kernel = self.module.get_function('raycast_voronoi')
        self.out = cp.empty((args.height, args.width, 3), dtype=cp.uint8)
        self.block = (16, 16, 1)
        self.grid_blocks = ((args.width + 15) // 16, (args.height + 15) // 16, 1)
        self.tf = None
        self.tf_n = 0

    def upload_tf(self, tf_np):
        self.tf = self.cp.asarray(tf_np.reshape(-1))
        self.tf_n = tf_np.shape[0]

    def fetch(self, provider, snap_idx):
        return provider.gpu_data(snap_idx)

    def render_frame(self, snap, extent,
                     cam, fwd, right, up, fov,
                     W, H, n_steps,
                     log_vmin_op, log_vmax_op, log_vmin_c, log_vmax_c,
                     opacity_scale, bg, box_bg, brightness=1.0,
                     brightness_gamma=1.0, shade=None,
                     log_map=True):
        if shade is None:
            do_shade, light, ambient, shade_h = 0, (0.0, 0.0, 1.0), 1.0, 0.0
        else:
            do_shade = 1
            light    = shade['light']
            ambient  = shade['ambient']
            shade_h  = shade['h_world']
        self.kernel(
            self.grid_blocks, self.block,
            (snap['seeds_sorted'],
             snap['values_op_sorted'], snap['values_c_sorted'],
             np.int32(snap['G']), np.float32(snap['voxel_size']),
             snap['voxel_count'], snap['voxel_ptr'],
             snap['cell_off_dx'], snap['cell_off_dy'], snap['cell_off_dz'],
             snap['cell_off_dist2'],
             np.int32(snap['N_offsets']),
             np.float32(extent),
             np.float32(cam[0]), np.float32(cam[1]), np.float32(cam[2]),
             np.float32(fwd[0]), np.float32(fwd[1]), np.float32(fwd[2]),
             np.float32(right[0]), np.float32(right[1]), np.float32(right[2]),
             np.float32(up[0]), np.float32(up[1]), np.float32(up[2]),
             np.float32(fov),
             np.int32(W), np.int32(H), np.int32(n_steps),
             np.float32(log_vmin_op), np.float32(log_vmax_op),
             np.float32(log_vmin_c),  np.float32(log_vmax_c),
             np.int32(1 if log_map else 0),
             np.float32(opacity_scale),
             self.tf, np.int32(self.tf_n),
             np.float32(bg[0]),     np.float32(bg[1]),     np.float32(bg[2]),
             np.float32(box_bg[0]), np.float32(box_bg[1]), np.float32(box_bg[2]),
             np.float32(brightness),
             np.float32(brightness_gamma),
             np.int32(do_shade),
             np.float32(light[0]), np.float32(light[1]), np.float32(light[2]),
             np.float32(ambient), np.float32(shade_h),
             self.out),
        )
        self.cp.cuda.runtime.deviceSynchronize()
        return self.cp.asnumpy(self.out)


class CPUBackend:
    name = 'cpu'

    def __init__(self, args):
        self.args = args
        self.tf = None
        self.out = np.empty((args.height, args.width, 3), dtype=np.uint8)

    def upload_tf(self, tf_np):
        self.tf = np.ascontiguousarray(tf_np)

    def fetch(self, provider, snap_idx):
        return provider.cpu_data(snap_idx)

    def render_frame(self, snap, extent,
                     cam, fwd, right, up, fov,
                     W, H, n_steps,
                     log_vmin_op, log_vmax_op, log_vmin_c, log_vmax_c,
                     opacity_scale, bg, box_bg, brightness=1.0,
                     brightness_gamma=1.0, shade=None,
                     log_map=True):
        tree      = snap['tree']
        values_op = snap['values_op']
        values_c  = snap['values_c']

        cam_arr = np.asarray(cam, dtype=np.float32)
        fwd_a   = np.asarray(fwd, dtype=np.float32)
        right_a = np.asarray(right, dtype=np.float32)
        up_a    = np.asarray(up, dtype=np.float32)

        aspect = W / H
        i_idx = np.arange(W, dtype=np.float32)
        j_idx = np.arange(H, dtype=np.float32)
        u = (2.0 * (i_idx + 0.5) / W - 1.0) * aspect * fov  # (W,)
        v = -(2.0 * (j_idx + 0.5) / H - 1.0) * fov          # (H,)

        # ray dirs: (H, W, 3)
        dirs = (fwd_a[None, None, :]
                + u[None, :, None] * right_a[None, None, :]
                + v[:, None, None] * up_a[None, None, :])
        dirs = (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).astype(np.float32)

        # slab intersection — NaN/inf from near-zero dirs propagate harmlessly
        # because the final hit mask is a strict t1 > t0 comparison.
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = (0.0 - cam_arr[None, None, :]) / dirs
            tb = (extent - cam_arr[None, None, :]) / dirs
        tlo = np.minimum(ta, tb)
        thi = np.maximum(ta, tb)
        t0 = np.maximum(np.maximum(tlo[..., 0], tlo[..., 1]), tlo[..., 2])
        t1 = np.minimum(np.minimum(thi[..., 0], thi[..., 1]), thi[..., 2])
        hit = (t1 > t0) & (t1 > 0.0)
        t0 = np.maximum(t0, 0.0)

        out = np.empty((H, W, 3), dtype=np.float32)
        out[..., 0] = bg[0]; out[..., 1] = bg[1]; out[..., 2] = bg[2]

        tf_alpha = np.ascontiguousarray(self.tf[:, 3])
        tf_rgb   = np.ascontiguousarray(self.tf[:, :3])
        tf_n     = self.tf.shape[0]

        s_arr = (np.arange(n_steps, dtype=np.float32) + 0.5)

        for r0 in range(0, H, CPU_TILE_H):
            r1 = min(r0 + CPU_TILE_H, H)

            tile_t0   = t0[r0:r1]
            tile_t1   = t1[r0:r1]
            tile_hit  = hit[r0:r1]
            tile_dirs = dirs[r0:r1]

            # sanitize non-hit rays so sample positions stay finite and the
            # KDTree query gets valid input. We mask the result back at the end.
            safe_t0 = np.where(tile_hit, tile_t0, 0.0).astype(np.float32)
            safe_dt = np.where(tile_hit, (tile_t1 - tile_t0) / n_steps, 0.0).astype(np.float32)

            t_s = safe_t0[..., None] + s_arr[None, None, :] * safe_dt[..., None]  # (tile_h, W, n_steps)
            sample_pos = (cam_arr[None, None, None, :]
                          + t_s[..., None] * tile_dirs[..., None, :]).astype(np.float32)

            rgb = _cpu_raycast_strip(
                tree, values_op, values_c,
                sample_pos, safe_dt / extent,
                opacity_scale,
                log_vmin_op, log_vmax_op, log_vmin_c, log_vmax_c,
                tf_alpha, tf_rgb, tf_n,
                box_bg, brightness=brightness, brightness_gamma=brightness_gamma,
                shade=shade, log_map=log_map,
            )
            out[r0:r1] = np.where(tile_hit[..., None], rgb, out[r0:r1])

        self.out[:] = (out * 255.0).clip(0, 255).astype(np.uint8)
        return self.out


_FONT_CANDIDATES = {
    'regular': [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/TTF/DejaVuSans.ttf',
    ],
    'bold': [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
    ],
}


def _load_font(weight, size):
    for path in _FONT_CANDIDATES[weight]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# 12 edges of the unit cube, expressed as pairs of corner indices in
# {0..7}, where bit 0=X, bit 1=Y, bit 2=Z.
_CUBE_EDGES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)


def _project_world_to_pixel(P, cam, fwd, right, up, fov, W, H):
    """Inverse of the kernel's ray equation: given a world point P, return
    (i, j) pixel coords (floats) or None if behind the camera."""
    rel = np.asarray(P, dtype=np.float32) - cam
    df = float(rel @ fwd)
    if df <= 1e-6:
        return None
    dr = float(rel @ right)
    du = float(rel @ up)
    aspect = W / H
    i = ((dr / df) / (aspect * fov) + 1.0) * W * 0.5 - 0.5
    j = (1.0 - (du / df) / fov) * H * 0.5 - 0.5
    return (i, j)


def draw_cube_edges_on_frame(frame_rgb, extent, cam, fwd, right, up, fov,
                             W, H, color=(0.1, 0.1, 0.1), alpha=1.0, width=1):
    """Overlay the 12 edges of the [0, extent]^3 cube onto an (H,W,3) uint8
    frame using the same camera projection as the volume raycaster. Edges
    whose endpoints are behind the camera are skipped."""
    corners = np.array([
        [(k & 1) * extent, ((k >> 1) & 1) * extent, ((k >> 2) & 1) * extent]
        for k in range(8)
    ], dtype=np.float32)
    pts = [_project_world_to_pixel(c, cam, fwd, right, up, fov, W, H)
           for c in corners]
    a = int(round(max(0.0, min(1.0, float(alpha))) * 255))
    rgba = (int(round(float(color[0]) * 255)),
            int(round(float(color[1]) * 255)),
            int(round(float(color[2]) * 255)), a)
    img = Image.fromarray(frame_rgb).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for a_idx, b_idx in _CUBE_EDGES:
        if pts[a_idx] is None or pts[b_idx] is None:
            continue
        d.line([pts[a_idx], pts[b_idx]], fill=rgba, width=max(1, int(width)))
    return np.asarray(Image.alpha_composite(img, overlay).convert('RGB'))


def draw_smbh_marker_on_frame(frame_rgb, cam, fwd, right, up, fov, W, H,
                              pos, size_px, edge_px,
                              fill_rgb=(0, 0, 0), edge_rgb=(255, 255, 255)):
    """Draw a filled circle at the SMBH world position with a contrasting ring.
    Coordinates use the same projection as the volume raycaster. No-op if the
    point is behind the camera or the projected pixel falls outside [-r, W+r]."""
    p = _project_world_to_pixel(pos, cam, fwd, right, up, fov, W, H)
    if p is None:
        return frame_rgb
    r_outer = max(1, int(size_px)) + max(0, int(edge_px))
    i, j = p
    if i < -r_outer or i > W + r_outer or j < -r_outer or j > H + r_outer:
        return frame_rgb
    img = Image.fromarray(frame_rgb).convert('RGBA')
    d = ImageDraw.Draw(img)
    if edge_px > 0:
        d.ellipse([i - r_outer, j - r_outer, i + r_outer, j + r_outer],
                  fill=(int(edge_rgb[0]), int(edge_rgb[1]), int(edge_rgb[2]), 255))
    r_inner = max(1, int(size_px))
    d.ellipse([i - r_inner, j - r_inner, i + r_inner, j + r_inner],
              fill=(int(fill_rgb[0]), int(fill_rgb[1]), int(fill_rgb[2]), 255))
    return np.asarray(img.convert('RGB'))


def draw_overlay(frame_rgb, sim_t, scale_bar_unit, scale_bar_pixels,
                 brand='ProteusGPU', brand_color=(200, 200, 205), brand_opacity=0.20,
                 colorbar=None,
                 length_factor=1.0, length_unit='',
                 time_factor=1.0, time_unit=''):
    """Burn a HUD onto an (H,W,3) uint8 RGB frame. Returns a new array."""
    H, W = frame_rgb.shape[:2]
    img = Image.fromarray(frame_rgb).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    margin = max(16, int(0.022 * H))

    f_time = _load_font('regular', max(16, int(0.030 * H)))
    t_val = sim_t * float(time_factor)
    t_suf = f' {time_unit}' if time_unit else ''
    t_text = f't = {t_val:.2f}{t_suf}'
    bbox = d.textbbox((0, 0), t_text, font=f_time)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((W - tw - margin, margin), t_text, font=f_time, fill=(255, 255, 255, 230))

    if colorbar is not None:
        cbar_w = max(20, int(0.025 * W))
        cbar_x1 = W - margin
        cbar_x0 = cbar_x1 - cbar_w
        cbar_y0 = margin + max(60, int(0.10 * H))
        cbar_y1 = H - max(110, int(0.16 * H))
        cbar_h = cbar_y1 - cbar_y0

    if colorbar is not None and cbar_h > 0 and cbar_w > 0:
        tf_rgb = colorbar['tf_rgb']
        strip = np.tile(tf_rgb[:, None, :], (1, cbar_w, 1))[::-1]
        bar_img = Image.fromarray(strip).resize((cbar_w, cbar_h), Image.BILINEAR)
        overlay.paste(bar_img.convert('RGBA'), (cbar_x0, cbar_y0))
        d.rectangle([cbar_x0, cbar_y0, cbar_x1, cbar_y1],
                    outline=(255, 255, 255, 200), width=1)

        f_title = _load_font('regular', max(16, int(0.028 * H)))
        title = colorbar['label']
        bbox = d.textbbox((0, 0), title, font=f_title)
        ttw, tth = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text((cbar_x1 - ttw, cbar_y0 - tth - 10),
               title, font=f_title, fill=(255, 255, 255, 230))

        f_tick = _load_font('regular', max(13, int(0.022 * H)))
        n_ticks = 5
        log_scale = colorbar.get('log', True)
        vmin, vmax = colorbar['vmin'], colorbar['vmax']
        for i in range(n_ticks):
            frac = i / (n_ticks - 1)
            y = cbar_y0 + int(frac * cbar_h)
            if log_scale and vmin > 0 and vmax > 0:
                lv0, lv1 = np.log(vmin), np.log(vmax)
                vv = np.exp(lv1 - frac * (lv1 - lv0))
            else:
                vv = vmax - frac * (vmax - vmin)
            label = f'{vv:.4g}'
            bbox = d.textbbox((0, 0), label, font=f_tick)
            ltw, lth = bbox[2] - bbox[0], bbox[3] - bbox[1]
            d.line([(cbar_x0 - 6, y), (cbar_x0 - 1, y)],
                   fill=(255, 255, 255, 200), width=1)
            d.text((cbar_x0 - ltw - 10, y - lth / 2),
                   label, font=f_tick, fill=(255, 255, 255, 230))

    f_scale = _load_font('regular', max(20, int(0.038 * H)))
    sb_val = scale_bar_unit * float(length_factor)
    sb_suf = f' {length_unit}' if length_unit else ''
    label = f'{sb_val:.4g}{sb_suf}'
    bbox = d.textbbox((0, 0), label, font=f_scale)
    lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bar_y = H - margin
    bar_x0 = margin
    bar_x1 = bar_x0 + scale_bar_pixels
    label_gap = max(14, int(0.022 * H))
    d.text((bar_x0 + (scale_bar_pixels - lw) / 2, bar_y - lh - label_gap),
           label, font=f_scale, fill=(255, 255, 255, 230))
    d.line([(bar_x0, bar_y), (bar_x1, bar_y)],
           fill=(255, 255, 255, 230), width=max(3, int(0.0055 * H)))

    f_brand = _load_font('bold', max(20, int(0.050 * H)))
    bbox = d.textbbox((0, 0), brand, font=f_brand)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    br, bg_, bb = (int(round(c)) for c in brand_color)
    ba = int(round(max(0.0, min(1.0, brand_opacity)) * 255))
    d.text((W - bw - margin, H - bh - margin - int(0.4 * bh)),
           brand, font=f_brand, fill=(br, bg_, bb, ba))

    out = Image.alpha_composite(img, overlay).convert('RGB')
    return np.asarray(out)


def camera_basis(cam, target, world_up=(0, 0, 1)):
    cam = np.asarray(cam, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(world_up, dtype=np.float32))
    right /= np.linalg.norm(right) + 1e-20
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up) + 1e-20
    return fwd, right, up


_AXIS_POLE = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}


def orbit_camera(target, radius, elev_rad, azim_rad, orbit_axis='z'):
    """Place a camera on the orbit sphere around `target` and return
    (cam, fwd, right, up). Used by both the animation (varying azim_rad over
    frames) and plot3D.py (fixed angle).

    `orbit_axis` picks which world axis is the orbit "pole" — the axis kept
    vertical in the frame as azim sweeps. elev is the angle above the
    equatorial plane (perpendicular to orbit_axis); azim wraps around it,
    starting from the next axis in cyclic order (z→x→y→z)."""
    ce, se = np.cos(elev_rad), np.sin(elev_rad)
    ca, sa = np.cos(azim_rad), np.sin(azim_rad)
    if orbit_axis == 'z':      # azim starts at +x, sweeps toward +y
        offset = np.array([ce * ca, ce * sa, se], dtype=np.float32)
    elif orbit_axis == 'x':    # azim starts at +y, sweeps toward +z
        offset = np.array([se, ce * ca, ce * sa], dtype=np.float32)
    elif orbit_axis == 'y':    # azim starts at +z, sweeps toward +x
        offset = np.array([ce * sa, se, ce * ca], dtype=np.float32)
    else:
        raise ValueError(f"orbit_axis must be one of x/y/z, got {orbit_axis!r}")
    cam = target + radius * offset
    fwd, right, up = camera_basis(cam, target, world_up=_AXIS_POLE[orbit_axis])
    return cam, fwd, right, up


def _percentile_range(values, vmin, vmax, pmin, pmax):
    flat = values[values > 0]
    if len(flat) == 0:
        flat = np.abs(values).ravel() + 1e-30
    if vmin is None:
        vmin = float(np.percentile(flat, pmin))
    if vmax is None:
        vmax = float(np.percentile(flat, pmax))
    return vmin, vmax


class SnapshotProvider:
    """Resolves snapshot index → seeds + per-quantity values, with a small
    disk cache (one .npy per quantity, plus seeds.npy) shared across all
    renderings of the same simulation. Backend-specific structures
    (cKDTree for CPU, spatial-hash dict for GPU) are built lazily on demand
    and pinned for the current snapshot."""

    def __init__(self, args):
        self.args = args
        self.extent = None
        self.units = None
        self._fields_share = (args.color_quantity == args.quantity)
        # snap_idx → simulation time
        self._times = {}
        # (snap_idx, quantity) → (min, max), kept across snap eviction
        self._stats = {}
        # cached host arrays for current "first snap" (used for vmin/vmax)
        self._first_seeds = None
        self._first_op = None
        self._first_c = None
        # CPU pinned snap: (snap_idx, dict)
        self._cpu_snap = None
        self._cpu_data = None
        # GPU pinned snap: (snap_idx, dict of cupy arrays)
        self._gpu_snap = None
        self._gpu_data = None

        if args.start_snap is None:
            self._mode = 'single'
            self._single_path = args.input
            self.first_snap = 0
        else:
            self._mode = 'range'
            assert args.end_snap is not None and args.end_snap >= args.start_snap
            self.first_snap = args.start_snap

    def _path_for(self, snap_idx):
        if self._mode == 'single':
            return self._single_path
        return self.args.snap_pattern.format(snap_idx)

    def snap_for_frame(self, f, total_frames):
        if self._mode == 'single':
            return 0
        n = self.args.end_snap - self.args.start_snap + 1
        hold = max(0, int(getattr(self.args, 'hold_frames', 0) or 0))
        evolve = max(1, total_frames - hold)
        if f >= evolve:
            return self.args.end_snap
        return self.args.start_snap + min(n - 1, int(f / evolve * n))

    def _cache_paths(self, snap_idx):
        """Return (seeds_path, op_path, c_path). None entries if caching disabled."""
        if self.args.no_cache:
            return None, None, None
        path = self._path_for(snap_idx)
        base = os.path.splitext(os.path.basename(path))[0]
        n_ranks = getattr(self.args, 'n_ranks', None)
        if n_ranks is not None:
            # Multi-rank concatenations are conceptually a different input than
            # the single-file path of the same name; namespace them so caches
            # from different --n-ranks runs don't collide.
            base = f'{base}_nranks{n_ranks}'
        cache_dir = os.path.join(os.path.dirname(path) or '.', '_render_cache')
        seeds_p = os.path.join(cache_dir, f'{base}_seeds.npy')
        op_p    = os.path.join(cache_dir, f'{base}_{self.args.quantity}.npy')
        c_p     = os.path.join(cache_dir, f'{base}_{self.args.color_quantity}.npy')
        return seeds_p, op_p, c_p

    def evict_disk_cache(self, snap_idx):
        seeds_p, op_p, c_p = self._cache_paths(snap_idx)
        for p in {seeds_p, op_p, c_p}:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    print(f"[cache] evicted {p}")
                except OSError as e:
                    print(f"[cache] failed to remove {p}: {e}")

    def load_arrays(self, snap_idx):
        """Return (seeds, values_op, values_c) as float32 numpy arrays. Reads
        from per-snapshot disk cache if all three pieces exist; otherwise loads
        the HDF5 and writes the cache."""
        seeds_p, op_p, c_p = self._cache_paths(snap_idx)
        path = self._path_for(snap_idx)

        seeds = values_op = values_c = None
        extent = sim_t = None

        cache_complete = (
            seeds_p and op_p and os.path.exists(seeds_p) and os.path.exists(op_p)
            and (self._fields_share or (c_p and os.path.exists(c_p)))
        )

        if cache_complete:
            seeds = np.load(seeds_p).astype(np.float32, copy=False)
            values_op = np.load(op_p).astype(np.float32, copy=False)
            values_c = values_op if self._fields_share else \
                       np.load(c_p).astype(np.float32, copy=False)
            with h5py.File(path, 'r') as fh:
                sim_t = float(fh['header'].attrs.get('time', 0.0))
                self.units = _read_units(fh['header'].attrs)
            extent = 1.0  # domain is hardcoded to [0,1]^D in the code
            print(f"[cache] loaded snap {snap_idx} from {os.path.dirname(seeds_p)}")
        else:
            n_ranks = getattr(self.args, 'n_ranks', None)
            full_seeds, rho, vel, E, extent, sim_t, units = load_snapshot(path, n_ranks=n_ranks)
            self.units = units
            seeds = full_seeds.astype(np.float32, copy=False)
            values_op = _extract_quantity(self.args.quantity, rho, vel, E)\
                            .astype(np.float32, copy=False)
            if self._fields_share:
                values_c = values_op
            else:
                values_c = _extract_quantity(self.args.color_quantity, rho, vel, E)\
                                .astype(np.float32, copy=False)
            print(f"[load] {path}  cells={len(seeds)}  extent={extent}  t={sim_t}")
            if seeds_p:
                os.makedirs(os.path.dirname(seeds_p), exist_ok=True)
                np.save(seeds_p, seeds)
                np.save(op_p, values_op)
                if not self._fields_share:
                    np.save(c_p, values_c)
                print(f"[cache] saved snap {snap_idx} → {os.path.dirname(seeds_p)}")

        self.extent = extent
        self._times[snap_idx] = sim_t
        key_op = (snap_idx, self.args.quantity)
        key_c  = (snap_idx, self.args.color_quantity)
        if key_op not in self._stats:
            self._stats[key_op] = (float(values_op.min()), float(values_op.max()))
        if key_c not in self._stats:
            self._stats[key_c] = (float(values_c.min()), float(values_c.max()))

        if snap_idx == self.first_snap and self._first_seeds is None:
            self._first_seeds = seeds
            self._first_op = values_op
            self._first_c = values_c
        return seeds, values_op, values_c

    def cpu_data(self, snap_idx):
        if self._cpu_snap == snap_idx:
            return self._cpu_data

        seeds, values_op, values_c = self.load_arrays(snap_idx)
        from scipy.spatial import cKDTree
        print(f"[cpu] building cKDTree over {len(seeds)} seeds (snap {snap_idx})...")
        t = time.time()
        tree = cKDTree(seeds)
        print(f"[cpu]   built in {time.time()-t:.2f}s")

        self._cpu_snap = snap_idx
        self._cpu_data = dict(seeds=seeds, values_op=values_op, values_c=values_c, tree=tree)
        return self._cpu_data

    def gpu_data(self, snap_idx):
        if self._gpu_snap == snap_idx:
            return self._gpu_data
        import cupy as cp
        if self._gpu_data is not None:
            self._gpu_data = None
            cp.get_default_memory_pool().free_all_blocks()

        seeds, values_op, values_c = self.load_arrays(snap_idx)
        N_cells = len(seeds)
        if self.args.knn_grid:
            G = self.args.knn_grid
        else:
            G = max(8, int(round(N_cells ** (1.0 / 3.0))))
            G = min(G, 256)
        R = min(max(1, G - 1), self.args.knn_search_radius)
        print(f"[gpu] spatial hash: N={N_cells}, G={G}, R={R} (snap {snap_idx})")
        t = time.time()
        sh = build_spatial_hash(seeds, self.extent, G, R)

        values_op_sorted = values_op[sh['perm']].astype(np.float32, copy=False)
        if self._fields_share:
            values_c_sorted = values_op_sorted
        else:
            values_c_sorted = values_c[sh['perm']].astype(np.float32, copy=False)
        print(f"[gpu]   built in {time.time()-t:.2f}s, {sh['N_offsets']} offsets")

        d_seeds = cp.asarray(sh['seeds_sorted'])
        d_op    = cp.asarray(values_op_sorted)
        d_c     = d_op if self._fields_share else cp.asarray(values_c_sorted)
        gpu = dict(
            seeds_sorted=d_seeds,
            values_op_sorted=d_op,
            values_c_sorted=d_c,
            voxel_count=cp.asarray(sh['voxel_count']),
            voxel_ptr=cp.asarray(sh['voxel_ptr']),
            cell_off_dx=cp.asarray(sh['cell_off_dx']),
            cell_off_dy=cp.asarray(sh['cell_off_dy']),
            cell_off_dz=cp.asarray(sh['cell_off_dz']),
            cell_off_dist2=cp.asarray(sh['cell_off_dist2']),
            G=sh['G'],
            voxel_size=sh['voxel_size'],
            N_offsets=sh['N_offsets'],
        )
        self._gpu_snap = snap_idx
        self._gpu_data = gpu
        return gpu


def prepare_render_context(provider, extent, args, backend):
    """One-time setup shared between plot3D_anim and plot3D: derive vmin/vmax
    from the first loaded snapshot, build + upload the transfer function,
    compute camera radius / fov, and pack the overlay metadata + shading dict
    so the per-frame renderer can stay flat. Returns a dict consumed by
    render_one_frame()."""
    op_first = provider._first_op
    c_first  = provider._first_c
    vmin_op, vmax_op = _percentile_range(op_first, args.vmin, args.vmax,
                                         args.pmin, args.pmax)
    vmin_c, vmax_c   = _percentile_range(c_first, args.color_vmin, args.color_vmax,
                                         args.pmin, args.pmax)
    print(f"[render] opacity '{args.quantity}'  vmin={vmin_op:.3e} vmax={vmax_op:.3e}"
          f"  ({'log' if args.log else 'linear'})")
    print(f"[render] color   '{args.color_quantity}'  vmin={vmin_c:.3e}  vmax={vmax_c:.3e}")
    if args.log:
        log_vmin_op, log_vmax_op = np.log(vmin_op), np.log(vmax_op)
        log_vmin_c,  log_vmax_c  = np.log(vmin_c),  np.log(vmax_c)
    else:
        log_vmin_op, log_vmax_op = float(vmin_op), float(vmax_op)
        log_vmin_c,  log_vmax_c  = float(vmin_c),  float(vmax_c)

    tf_np = build_transfer_function(
        args.cmap, opacity_gamma=args.opacity_gamma,
        tf_mu1=args.tf_mu1, tf_s1=args.tf_s1,
        tf_mu2=args.tf_mu2, tf_s2=args.tf_s2,
    )
    backend.upload_tf(tf_np)

    fov = np.tan(np.deg2rad(args.fov_deg) * 0.5)
    target = np.array([0.5 * extent] * 3, dtype=np.float32)
    half_diag = np.sqrt(3.0) / 2.0 * extent
    auto_radius = half_diag / np.tan(np.deg2rad(args.fov_deg) * 0.5)
    radius = args.radius_factor * auto_radius

    bg = tuple(float(c) for c in args.bg)
    box_bg = tuple(float(c) for c in args.box_bg)

    shade = None
    if args.shade:
        shade = dict(
            light=tuple(float(c) for c in args.light),
            ambient=float(args.ambient),
            h_world=float(args.shade_h_frac) * float(extent),
        )
        print(f"[shade] enabled: light={shade['light']} ambient={shade['ambient']:.2f} "
              f"h_world={shade['h_world']:.4g} (= {args.shade_h_frac:.4g} * extent)")

    # Physical-unit display factors (1.0/'' when --*-unit is 'code' or header
    # attrs missing). These convert values at DISPLAY time only — internal
    # math (kernel, TF, vmin/vmax) stays in code units.
    len_fac,  len_unit  = _display_factor('length',   args.length_unit,   provider.units)
    vel_fac,  vel_unit  = _display_factor('velocity', args.velocity_unit, provider.units)
    time_fac, time_unit = _display_factor('time',     args.time_unit,     provider.units)
    if provider.units is None and any(u != 'code' for u in
                                      (args.length_unit, args.velocity_unit, args.time_unit)):
        print("[units] snapshot header has no UnitLength_in_cm/UnitVelocity_in_cm_per_s "
              "(run without ASTRO_PHYSICS?) -> labels stay in code units")
    if provider.units is not None and (len_unit or vel_unit or time_unit):
        print(f"[units] header L={provider.units['length_cm']:.3e} cm  "
              f"V={provider.units['velocity_cm_per_s']:.3e} cm/s  "
              f"T={provider.units['time_s']:.3e} s")
        if len_unit:  print(f"[units] length:   1 code = {len_fac:.4g} {len_unit}")
        if vel_unit:  print(f"[units] velocity: 1 code = {vel_fac:.4g} {vel_unit}")
        if time_unit: print(f"[units] time:     1 code = {time_fac:.4g} {time_unit}")

    cbar_info = None
    if not args.no_overlay and not args.no_colorbar:
        cbar_tf_rgb = (np.clip(tf_np[:, :3], 0.0, 1.0) * 255).astype(np.uint8)
        cb_label = args.color_quantity
        cb_fac = 1.0
        # Only vel_mag gets a physical velocity unit for now (rho/energy/temp/xray
        # need extra conversions or μ; deliberately out of scope).
        if args.color_quantity == 'vel_mag' and vel_unit:
            cb_fac = vel_fac
            cb_label = f'vel_mag [{vel_unit}]'
        cbar_info = dict(tf_rgb=cbar_tf_rgb,
                         vmin=float(vmin_c) * cb_fac, vmax=float(vmax_c) * cb_fac,
                         label=cb_label, log=bool(args.log))

    pixels_per_box_unit = args.height / (2.0 * radius * fov)
    scale_bar_pixels = max(2, int(round(args.scale_bar_unit * pixels_per_box_unit)))
    if not args.no_overlay:
        print(f"[overlay] scale bar: {args.scale_bar_unit} box units = "
              f"{scale_bar_pixels} px")

    return dict(
        log_vmin_op=log_vmin_op, log_vmax_op=log_vmax_op,
        log_vmin_c=log_vmin_c,   log_vmax_c=log_vmax_c,
        vmin_op=vmin_op, vmax_op=vmax_op, vmin_c=vmin_c, vmax_c=vmax_c,
        log_map=bool(args.log),
        fov=fov, target=target, radius=radius,
        bg=bg, box_bg=box_bg, shade=shade,
        cbar_info=cbar_info, scale_bar_pixels=scale_bar_pixels,
        tf_np=tf_np,
        length_factor=len_fac,   length_unit=len_unit,
        time_factor=time_fac,    time_unit=time_unit,
    )


def render_one_frame(backend, snap, extent, args, ctx,
                     cam, fwd, right, up, sim_t):
    """Render a single uint8 (H,W,3) frame using a context built by
    prepare_render_context. Applies the HUD overlay if enabled."""
    frame = backend.render_frame(
        snap, extent,
        cam, fwd, right, up, ctx['fov'],
        args.width, args.height, args.n_steps,
        ctx['log_vmin_op'], ctx['log_vmax_op'],
        ctx['log_vmin_c'],  ctx['log_vmax_c'],
        args.opacity_scale, ctx['bg'], ctx['box_bg'],
        brightness=args.brightness, brightness_gamma=args.brightness_gamma,
        shade=ctx['shade'], log_map=ctx['log_map'],
    )
    if args.cube_edges:
        frame = draw_cube_edges_on_frame(
            frame, extent, cam, fwd, right, up, ctx['fov'],
            args.width, args.height,
            color=tuple(args.cube_edge_color),
            alpha=float(args.cube_edge_alpha),
            width=int(args.cube_edge_width),
        )
    if args.smbh:
        frame = draw_smbh_marker_on_frame(
            frame, cam, fwd, right, up, ctx['fov'],
            args.width, args.height,
            pos=tuple(args.smbh_pos),
            size_px=int(args.smbh_size),
            edge_px=int(args.smbh_edge_width),
            fill_rgb=tuple(args.smbh_fill),
            edge_rgb=tuple(args.smbh_edge),
        )
    if not args.no_overlay:
        frame = draw_overlay(frame, sim_t, args.scale_bar_unit,
                             ctx['scale_bar_pixels'],
                             brand=args.brand,
                             brand_color=tuple(args.brand_color),
                             brand_opacity=args.brand_opacity,
                             colorbar=ctx['cbar_info'],
                             length_factor=ctx['length_factor'],
                             length_unit=ctx['length_unit'],
                             time_factor=ctx['time_factor'],
                             time_unit=ctx['time_unit'])
    return frame


def render_animation(provider, extent, args, backend):
    print(f"[render][{backend.name}] image {args.width}x{args.height}, "
          f"{args.frames} frames, {args.n_steps} steps/ray")
    ctx = prepare_render_context(provider, extent, args, backend)
    elev = np.deg2rad(args.elev_deg)

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{args.width}x{args.height}', '-pix_fmt', 'rgb24',
        '-r', str(args.fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'medium', '-crf', str(args.crf),
        args.output
    ]
    print(f"[render] launching ffmpeg → {args.output}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    g_op_min = np.inf; g_op_max = -np.inf
    g_c_min  = np.inf; g_c_max  = -np.inf

    try:
        t_start = time.time()
        for f in range(args.frames):
            snap_idx = provider.snap_for_frame(f, args.frames)
            snap = backend.fetch(provider, snap_idx)

            op_min, op_max = provider._stats[(snap_idx, args.quantity)]
            c_min,  c_max  = provider._stats[(snap_idx, args.color_quantity)]
            g_op_min = min(g_op_min, op_min); g_op_max = max(g_op_max, op_max)
            g_c_min  = min(g_c_min,  c_min);  g_c_max  = max(g_c_max,  c_max)
            print(f"[f {f+1:4d}/{args.frames} s={snap_idx:>3}] "
                  f"{args.quantity}=[{op_min:.3e},{op_max:.3e}]  "
                  f"{args.color_quantity}=[{c_min:.3e},{c_max:.3e}]")

            hold = max(0, int(args.hold_frames or 0))
            evolve = max(1, args.frames - hold)
            if f < evolve:
                phi = 2 * np.pi * args.turns * f / evolve
            else:
                phi = 2 * np.pi * (args.turns
                                   + args.hold_turns * (f - evolve) / max(1, hold))
            cam, fwd, right, up = orbit_camera(ctx['target'], ctx['radius'], elev, phi,
                                               orbit_axis=args.orbit_axis)

            sim_t = provider._times.get(snap_idx, 0.0)
            frame = render_one_frame(backend, snap, extent, args, ctx,
                                     cam, fwd, right, up, sim_t)
            proc.stdin.write(frame.tobytes())

            if args.evict_cache:
                last_use = (f + 1 == args.frames or
                            provider.snap_for_frame(f + 1, args.frames) != snap_idx)
                if last_use:
                    provider.evict_disk_cache(snap_idx)

            if (f + 1) % 20 == 0 or f == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (f + 1) * (args.frames - f - 1)
                print(f"[render]   elapsed {elapsed:.1f}s  eta {eta:.1f}s")
    finally:
        proc.stdin.close()
        proc.wait()
    print(f"[render] done → {args.output}")
    print(f"[GLOBAL] {args.quantity:>10}: min={g_op_min:.6e}  max={g_op_max:.6e}")
    print(f"[GLOBAL] {args.color_quantity:>10}: min={g_c_min:.6e}  max={g_c_max:.6e}")
    print(f"[GLOBAL] rerun with: --vmin {g_op_min:.6e} --vmax {g_op_max:.6e} "
          f"--color-vmin {g_c_min:.6e} --color-vmax {g_c_max:.6e}")


def add_common_render_args(p):
    """Knobs shared between plot3D_anim.py (orbit animation) and plot3D.py
    (single PNG): fields, image size, camera framing, transfer function,
    shading, overlay, caching, backend selection. Each script owns its own
    -i / -o defaults and any mode-specific args (anim adds frames/turns/...,
    plot3D adds --azim-deg)."""
    p.add_argument('-q', '--quantity', default='rho', choices=QUANTITIES,
                   help='field driving OPACITY')
    p.add_argument('--color-quantity', default=None, choices=QUANTITIES,
                   help='field driving COLOR via the transfer function. '
                        'If unset, defaults to --quantity (single-field mode).')

    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--n-steps', type=int, default=512)
    p.add_argument('--fov-deg', type=float, default=35.0)
    p.add_argument('--elev-deg', type=float, default=20.0)
    p.add_argument('--orbit-axis', default='z', choices=('x', 'y', 'z'),
                   help="world axis kept vertical as the camera orbits. Use 'y' "
                        "for jets aligned with the y-axis so they stay upright.")
    p.add_argument('--radius-factor', type=float, default=1.15,
                   help='multiplier on auto-fit radius (1.0 = bounding sphere just fills view)')

    p.add_argument('--cmap', default='fire',
                   help='fire | ice | aurora | coolwarm_dark | <any matplotlib cmap>')
    p.add_argument('--opacity-scale', type=float, default=4.0,
                   help='optical depth multiplier; the kernel scales by a fixed '
                        'historical density factor of 256, so this default '
                        'matches the previous grid-based renderer at --grid 256.')
    p.add_argument('--opacity-gamma', type=float, default=2.0)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument('--log', dest='log', action='store_true', default=True,
                       help='map values to color/opacity in log-space (default). '
                            '--vmin/--vmax need to be > 0.')
    scale.add_argument('--linear', dest='log', action='store_false',
                       help='map values to color/opacity linearly between '
                            '--vmin and --vmax (and --color-vmin/--color-vmax).')
    p.add_argument('--tf-mu1', type=float, default=None,
                   help='center μ1 (in normalized value space [0,1]) of the first '
                        'opacity Gaussian. When set (with optional --tf-mu2), '
                        'replaces the default α = t^gamma. Values outside [0,1] '
                        'push the bump partly off-range.')
    p.add_argument('--tf-s1', type=float, default=0.05,
                   help='sigma σ1 of the first Gaussian (only used if --tf-mu1 set)')
    p.add_argument('--tf-mu2', type=float, default=None,
                   help='center μ2 of an optional second opacity Gaussian; enables '
                        'a two-population TF (e.g. cold+hot in the same frame)')
    p.add_argument('--tf-s2', type=float, default=0.05,
                   help='sigma σ2 of the second Gaussian (only used if --tf-mu2 set)')

    p.add_argument('--vmin', type=float, default=None,
                   help='opacity field vmin (auto from percentile if unset)')
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--color-vmin', type=float, default=None,
                   help='color field vmin (auto from percentile if unset)')
    p.add_argument('--color-vmax', type=float, default=None)
    p.add_argument('--pmin', type=float, default=1.0)
    p.add_argument('--pmax', type=float, default=99.5)

    p.add_argument('--brightness', type=float, default=1.0,
                   help='final-stage RGB multiplier applied after tonemap, '
                        'before the 0-1 clamp. >1 brightens, <1 darkens; '
                        'default 1.0 (no change).')
    p.add_argument('--brightness-gamma', type=float, default=1.0,
                   help='display-gamma mid-tone lift applied after --brightness. '
                        'out = in ** (1/g). g=1 (default) is no-op; g>1 brightens '
                        'mid-tones without pulling highlights or shadows off their '
                        'endpoints; g<1 does the opposite (mid-tones darker).')

    p.add_argument('--bg', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=('R', 'G', 'B'),
                   help='screen background where rays miss the cube')
    p.add_argument('--box-bg', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=('R', 'G', 'B'),
                   help='in-cube background composited where rays exit the '
                        'cube without saturating opacity (default black; pair '
                        'with --cube-edges to keep the cube silhouette).')

    p.add_argument('--no-cube-edges', dest='cube_edges', action='store_false',
                   help='disable the bounding-cube wireframe (drawn on by '
                        'default at low gray so the box silhouette stays '
                        'visible against the dark background)')
    p.set_defaults(cube_edges=True)
    p.add_argument('--cube-edge-color', type=float, nargs=3,
                   default=(0.1, 0.1, 0.1), metavar=('R', 'G', 'B'),
                   help='RGB color of the bounding cube edges, 0-1 per '
                        'channel (default 0.1 gray)')
    p.add_argument('--cube-edge-alpha', type=float, default=1.0,
                   help='opacity of the cube edges, 0-1 (default 1.0)')
    p.add_argument('--smbh', action='store_true',
                   help='draw a circular SMBH marker at --smbh-pos over the frame')
    p.add_argument('--smbh-pos', type=float, nargs=3, default=(0.5, 0.5, 0.5),
                   metavar=('X', 'Y', 'Z'),
                   help='SMBH world coordinates for the marker overlay')
    p.add_argument('--smbh-size', type=int, default=6,
                   help='radius of the filled core in pixels')
    p.add_argument('--smbh-edge-width', type=int, default=2,
                   help='thickness of the contrasting ring, in pixels')
    p.add_argument('--smbh-fill', type=int, nargs=3, default=(0, 0, 0),
                   metavar=('R', 'G', 'B'),
                   help='SMBH core color, 0..255 (default: black)')
    p.add_argument('--smbh-edge', type=int, nargs=3, default=(255, 255, 255),
                   metavar=('R', 'G', 'B'),
                   help='SMBH ring color, 0..255 (default: white)')
    p.add_argument('--cube-edge-width', type=int, default=1,
                   help='line width in pixels for the cube edges (default 1)')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--no-overlay', action='store_true',
                   help='disable HUD (time, length scale, brand, colorbar)')
    p.add_argument('--no-colorbar', action='store_true',
                   help='disable just the colorbar; keep other HUD elements')
    p.add_argument('--length-unit', default='code', choices=list(_LEN_CM.keys()),
                   help="physical length unit for the scale-bar label. 'code' "
                        "keeps values raw. Requires UnitLength_in_cm in the "
                        "snapshot header (present by default).")
    p.add_argument('--velocity-unit', default='code', choices=list(_VEL_CMPS.keys()),
                   help="physical velocity unit for the colorbar (only when "
                        "--color-quantity=vel_mag). Requires UnitVelocity_in_cm_per_s.")
    p.add_argument('--time-unit', default='code', choices=list(_TIME_S.keys()),
                   help="physical time unit for the t= overlay. Derived from "
                        "UnitLength/UnitVelocity in the header.")
    p.add_argument('--scale-bar-unit', type=float, default=0.1,
                   help='length scale label in box units (default 0.1)')
    p.add_argument('--brand', default='ProteusGPU',
                   help='bottom-right watermark text (empty string disables)')
    p.add_argument('--brand-color', type=int, nargs=3, default=(200, 200, 205),
                   metavar=('R', 'G', 'B'),
                   help='watermark color, 0-255 per channel (default: 200 200 205)')
    p.add_argument('--brand-opacity', type=float, default=0.20,
                   help='watermark opacity, 0.0-1.0 (default: 0.20)')

    p.add_argument('--shade', action='store_true',
                   help='enable gradient shading: 6 extra NN lookups per ray '
                        'step build a log-space density gradient that '
                        'modulates RGB by ambient + (1-ambient)*max(0,-n.l). '
                        'Adds ~5-7x kernel cost; off by default.')
    p.add_argument('--light', type=float, nargs=3, default=(0.4, 0.6, 0.7),
                   metavar=('X', 'Y', 'Z'),
                   help='light direction in world space (need not be normalized).')
    p.add_argument('--ambient', type=float, default=0.35,
                   help='ambient term for gradient shading (0=full shadow, '
                        '1=no shading); default 0.35.')
    p.add_argument('--shade-h-frac', type=float, default=0.01,
                   help='gradient stencil offset as fraction of box extent. '
                        'Must span ~one Voronoi cell or larger or the central '
                        'difference returns 0. Default 0.01.')

    p.add_argument('--cpu', action='store_true',
                   help='use the scipy cKDTree CPU raycaster instead of CUDA. '
                        'Much slower; intended for debugging or machines without a GPU.')
    p.add_argument('--knn-grid', type=int, default=None,
                   help='spatial-hash voxel grid resolution G (default: '
                        'auto ≈ N_cells^(1/3), clamped to [8, 256])')
    p.add_argument('--knn-search-radius', type=int, default=16,
                   help='max ring radius (in voxels) for the spatial-hash NN '
                        'offset table (default 16)')


def select_backend(args):
    """Instantiate GPU or CPU backend; exit cleanly if cupy is missing."""
    if args.cpu:
        return CPUBackend(args)
    try:
        import cupy  # noqa
    except ImportError:
        print("ERROR: cupy not installed. Install with e.g. "
              "`pip install cupy-cuda12x` matching your CUDA version, "
              "or pass --cpu to use the (slow) cKDTree CPU debug renderer.",
              file=sys.stderr)
        sys.exit(1)
    return GPUBackend(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', default='output/snapshot_1.hdf5',
                   help='single-snapshot input (ignored if --start-snap is set)')
    p.add_argument('-o', '--output', default='visualize/render.mp4')

    p.add_argument('--start-snap', type=int, default=None,
                   help='first snapshot index (enables time-evolving animation)')
    p.add_argument('--end-snap', type=int, default=None,
                   help='last snapshot index (inclusive)')
    p.add_argument('--snap-pattern', default='output/snapshot_{}.hdf5',
                   help='format string for snapshot path; {} is the index. '
                        'With --n-ranks, each formatted path is treated as a '
                        'single-file template; sibling .<r>.hdf5 files are '
                        'derived from it.')
    p.add_argument('-n', '--n-ranks', type=int, default=None,
                   help='Load and concatenate this many per-rank files per '
                        'snapshot (treats --input / --snap-pattern paths as '
                        'templates; reads <stem>.<r>.hdf5 for r in 0..n-1). '
                        'Omit for single-file mode.')

    p.add_argument('--frames', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--turns', type=float, default=1.0,
                   help='full rotations during the snapshot-evolution phase '
                        '(or the whole animation if --hold-frames is 0)')
    p.add_argument('--hold-frames', type=int, default=0,
                   help='after evolving through start..end snapshots over the '
                        'first (frames - hold-frames) frames, hold on '
                        '--end-snap for this many additional frames (still '
                        'rotating by --hold-turns). Default 0 (no hold phase).')
    p.add_argument('--hold-turns', type=float, default=1.0,
                   help='full rotations performed during the hold phase '
                        '(only used when --hold-frames > 0); default 1.0.')
    p.add_argument('--crf', type=int, default=18)
    p.add_argument('--evict-cache', action='store_true',
                   help='delete each snapshot\'s cached .npy files as soon '
                        'as the last frame referencing it has been rendered.')

    add_common_render_args(p)
    args = p.parse_args()

    if args.color_quantity is None:
        args.color_quantity = args.quantity

    if args.color_quantity == args.quantity:
        if args.color_vmin is None:
            args.color_vmin = args.vmin
        if args.color_vmax is None:
            args.color_vmax = args.vmax

    if (args.start_snap is None) ^ (args.end_snap is None):
        print("ERROR: --start-snap and --end-snap must be set together "
              "(or both omitted for single-snapshot mode).", file=sys.stderr)
        sys.exit(2)

    backend = select_backend(args)

    if args.hold_frames < 0:
        print("ERROR: --hold-frames must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.hold_frames >= args.frames:
        print("ERROR: --hold-frames must be < --frames "
              f"(got hold={args.hold_frames}, frames={args.frames})",
              file=sys.stderr)
        sys.exit(2)

    if args.start_snap is not None:
        n_snaps = args.end_snap - args.start_snap + 1
        evolve_frames = args.frames - args.hold_frames
        if args.hold_frames > 0:
            print(f"[mode] time-evolving: snaps {args.start_snap}..{args.end_snap} "
                  f"({n_snaps} snapshots) over {evolve_frames} evolve frames "
                  f"(~{evolve_frames / n_snaps:.1f} frames per snap), "
                  f"then hold on snap {args.end_snap} for {args.hold_frames} frames")
        else:
            print(f"[mode] time-evolving: snaps {args.start_snap}..{args.end_snap} "
                  f"({n_snaps} snapshots) over {args.frames} frames "
                  f"(~{args.frames / n_snaps:.1f} frames per snap)")
    else:
        print(f"[mode] single snapshot: {args.input}")
    if args.n_ranks is not None:
        print(f"[mode] multi-rank: concatenating {args.n_ranks} files per snapshot")
    print(f"[fields] opacity='{args.quantity}'  color='{args.color_quantity}'")

    provider = SnapshotProvider(args)
    provider.load_arrays(provider.first_snap)
    extent = provider.extent

    render_animation(provider, extent, args, backend)


if __name__ == '__main__':
    main()
