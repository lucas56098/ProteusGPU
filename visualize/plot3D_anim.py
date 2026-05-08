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


def load_snapshot(path):
    with h5py.File(path, 'r') as f:
        seeds = f['cells/seeds'][:]
        rho = f['hydro/rho'][:]
        vel = f['hydro/vel'][:]
        E = f['hydro/Energy'][:]
        dim = int(f['header'].attrs['dimension'])
        extent = float(f['header'].attrs['extent'])
        sim_time = float(f['header'].attrs['time'])
    if dim != 3:
        raise ValueError(f"need 3D snapshot, got {dim}D")
    return seeds, rho, vel, E, extent, sim_time


QUANTITIES = ('rho', 'vel_mag', 'energy')


def _extract_quantity(name, rho, vel, E):
    if name == 'rho':
        return rho
    if name == 'vel_mag':
        return np.sqrt((vel ** 2).sum(axis=1))
    if name == 'energy':
        return E
    raise ValueError(f"unknown quantity: {name}")


def build_transfer_function(name='fire', n=256, opacity_gamma=2.0):
    """Return [n,4] float32 RGBA LUT. Alpha here is the opacity multiplier κ
    (not the final composited alpha) — gets fed to (1 - exp(-κ*dt))."""
    t = np.linspace(0, 1, n)

    if name == 'fire':
        r = np.clip(2.0 * t - 0.1, 0, 1)
        g = np.clip(2.5 * t - 1.0, 0, 1)
        b = np.clip(3.0 * t - 2.3, 0, 1) + 0.35 * np.exp(-((t - 0.05) / 0.08)**2)
        b = np.clip(b, 0, 1)
    elif name == 'ice':
        r = np.clip(1.4 * t - 0.5, 0, 1)
        g = np.clip(1.2 * t - 0.05, 0, 1)
        b = np.clip(0.3 + 0.7 * t, 0, 1)
    elif name == 'aurora':
        r = np.clip(2.0 * t - 1.0, 0, 1)
        g = np.clip(1.6 * t + 0.05, 0, 1)
        b = np.clip(1.4 - 1.6 * t + 0.5 * np.exp(-((t-0.85)/0.1)**2), 0, 1)
    else:
        import matplotlib.cm as cm
        rgba = cm.get_cmap(name)(t)
        r, g, b = rgba[:, 0], rgba[:, 1], rgba[:, 2]

    a = t ** opacity_gamma
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
    float log_vmin_op, float log_vmax_op,
    float log_vmin_c,  float log_vmax_c,
    float opacity_scale,
    const float4* __restrict__ tf, int tf_n,
    float bg_r, float bg_g, float bg_b,
    float box_r, float box_g, float box_b,
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
        float lv_op  = (logf(fmaxf(val_op, 1e-30f)) - log_vmin_op) /
                       (log_vmax_op - log_vmin_op);
        lv_op = fminf(fmaxf(lv_op, 0.f), 1.f);
        int   idx_op = (int)(lv_op * (float)(tf_n - 1));
        float alpha_factor = tf[idx_op].w;

        float val_c = values_c_sorted[seed_idx];
        float lv_c  = (logf(fmaxf(val_c, 1e-30f)) - log_vmin_c) /
                      (log_vmax_c - log_vmin_c);
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
                       box_bg, shade=None):
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

    lv_op = (np.log(np.maximum(val_op, 1e-30)) - log_vmin_op) / log_range_op
    lv_op = np.clip(lv_op, 0.0, 1.0)
    idx_op = (lv_op * (tf_n - 1)).astype(np.int32)
    alpha_factor = tf_alpha[idx_op]

    lv_c = (np.log(np.maximum(val_c, 1e-30)) - log_vmin_c) / log_range_c
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
                     opacity_scale, bg, box_bg, shade=None):
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
             np.float32(opacity_scale),
             self.tf, np.int32(self.tf_n),
             np.float32(bg[0]),     np.float32(bg[1]),     np.float32(bg[2]),
             np.float32(box_bg[0]), np.float32(box_bg[1]), np.float32(box_bg[2]),
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
                     opacity_scale, bg, box_bg, shade=None):
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
                box_bg, shade=shade,
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


def draw_overlay(frame_rgb, sim_t, scale_bar_unit, scale_bar_pixels,
                 brand='ProteusGPU', brand_color=(200, 200, 205), brand_opacity=0.20,
                 colorbar=None):
    """Burn a HUD onto an (H,W,3) uint8 RGB frame. Returns a new array."""
    H, W = frame_rgb.shape[:2]
    img = Image.fromarray(frame_rgb).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    margin = max(16, int(0.022 * H))

    f_time = _load_font('regular', max(16, int(0.030 * H)))
    t_text = f't = {sim_t:.2f}'
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
            label = f'{vv:.2g}'
            bbox = d.textbbox((0, 0), label, font=f_tick)
            ltw, lth = bbox[2] - bbox[0], bbox[3] - bbox[1]
            d.line([(cbar_x0 - 6, y), (cbar_x0 - 1, y)],
                   fill=(255, 255, 255, 200), width=1)
            d.text((cbar_x0 - ltw - 10, y - lth / 2),
                   label, font=f_tick, fill=(255, 255, 255, 230))

    f_scale = _load_font('regular', max(20, int(0.038 * H)))
    label = f'{scale_bar_unit:g}'
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


def orbit_camera(target, radius, elev_rad, azim_rad):
    """Place a camera on the orbit sphere around `target` and return
    (cam, fwd, right, up). Used by both the animation (varying azim_rad over
    frames) and plot3D.py (fixed angle)."""
    cam = target + np.array([
        radius * np.cos(elev_rad) * np.cos(azim_rad),
        radius * np.cos(elev_rad) * np.sin(azim_rad),
        radius * np.sin(elev_rad),
    ], dtype=np.float32)
    fwd, right, up = camera_basis(cam, target)
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
        return self.args.start_snap + min(n - 1, int(f / total_frames * n))

    def _cache_paths(self, snap_idx):
        """Return (seeds_path, op_path, c_path). None entries if caching disabled."""
        if self.args.no_cache:
            return None, None, None
        path = self._path_for(snap_idx)
        base = os.path.splitext(os.path.basename(path))[0]
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
                extent = float(fh['header'].attrs['extent'])
                sim_t = float(fh['header'].attrs.get('time', 0.0))
            print(f"[cache] loaded snap {snap_idx} from {os.path.dirname(seeds_p)}")
        else:
            full_seeds, rho, vel, E, extent, sim_t = load_snapshot(path)
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
    print(f"[render] opacity '{args.quantity}'  vmin={vmin_op:.3e} vmax={vmax_op:.3e}")
    print(f"[render] color   '{args.color_quantity}'  vmin={vmin_c:.3e}  vmax={vmax_c:.3e}")
    log_vmin_op, log_vmax_op = np.log(vmin_op), np.log(vmax_op)
    log_vmin_c,  log_vmax_c  = np.log(vmin_c),  np.log(vmax_c)

    tf_np = build_transfer_function(args.cmap, opacity_gamma=args.opacity_gamma)
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

    cbar_info = None
    if not args.no_overlay and not args.no_colorbar:
        cbar_tf_rgb = (np.clip(tf_np[:, :3], 0.0, 1.0) * 255).astype(np.uint8)
        cbar_info = dict(tf_rgb=cbar_tf_rgb,
                         vmin=float(vmin_c), vmax=float(vmax_c),
                         label=args.color_quantity, log=True)

    pixels_per_box_unit = args.height / (2.0 * radius * fov)
    scale_bar_pixels = max(2, int(round(args.scale_bar_unit * pixels_per_box_unit)))
    if not args.no_overlay:
        print(f"[overlay] scale bar: {args.scale_bar_unit} box units = "
              f"{scale_bar_pixels} px")

    return dict(
        log_vmin_op=log_vmin_op, log_vmax_op=log_vmax_op,
        log_vmin_c=log_vmin_c,   log_vmax_c=log_vmax_c,
        vmin_op=vmin_op, vmax_op=vmax_op, vmin_c=vmin_c, vmax_c=vmax_c,
        fov=fov, target=target, radius=radius,
        bg=bg, box_bg=box_bg, shade=shade,
        cbar_info=cbar_info, scale_bar_pixels=scale_bar_pixels,
        tf_np=tf_np,
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
        args.opacity_scale, ctx['bg'], ctx['box_bg'], shade=ctx['shade'],
    )
    if not args.no_overlay:
        frame = draw_overlay(frame, sim_t, args.scale_bar_unit,
                             ctx['scale_bar_pixels'],
                             brand=args.brand,
                             brand_color=tuple(args.brand_color),
                             brand_opacity=args.brand_opacity,
                             colorbar=ctx['cbar_info'])
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

            phi = 2 * np.pi * args.turns * f / args.frames
            cam, fwd, right, up = orbit_camera(ctx['target'], ctx['radius'], elev, phi)

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
    p.add_argument('--radius-factor', type=float, default=1.15,
                   help='multiplier on auto-fit radius (1.0 = bounding sphere just fills view)')

    p.add_argument('--cmap', default='fire',
                   help='fire | ice | aurora | <any matplotlib cmap>')
    p.add_argument('--opacity-scale', type=float, default=4.0,
                   help='optical depth multiplier; the kernel scales by a fixed '
                        'historical density factor of 256, so this default '
                        'matches the previous grid-based renderer at --grid 256.')
    p.add_argument('--opacity-gamma', type=float, default=2.0)

    p.add_argument('--vmin', type=float, default=None,
                   help='opacity field vmin (auto from percentile if unset)')
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--color-vmin', type=float, default=None,
                   help='color field vmin (auto from percentile if unset)')
    p.add_argument('--color-vmax', type=float, default=None)
    p.add_argument('--pmin', type=float, default=1.0)
    p.add_argument('--pmax', type=float, default=99.5)

    p.add_argument('--bg', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=('R', 'G', 'B'),
                   help='screen background where rays miss the cube')
    p.add_argument('--box-bg', type=float, nargs=3, default=(0.04, 0.04, 0.07),
                   metavar=('R', 'G', 'B'),
                   help='in-cube background; kept slightly lighter than --bg '
                        'so the cube silhouette is visible')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--no-overlay', action='store_true',
                   help='disable HUD (time, length scale, brand, colorbar)')
    p.add_argument('--no-colorbar', action='store_true',
                   help='disable just the colorbar; keep other HUD elements')
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
                   help='format string for snapshot path; {} is the index')

    p.add_argument('--frames', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--turns', type=float, default=1.0,
                   help='full rotations across the animation')
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

    if args.start_snap is not None:
        n_snaps = args.end_snap - args.start_snap + 1
        print(f"[mode] time-evolving: snaps {args.start_snap}..{args.end_snap} "
              f"({n_snaps} snapshots) over {args.frames} frames "
              f"(~{args.frames / n_snaps:.1f} frames per snap)")
    else:
        print(f"[mode] single snapshot: {args.input}")
    print(f"[fields] opacity='{args.quantity}'  color='{args.color_quantity}'")

    provider = SnapshotProvider(args)
    provider.load_arrays(provider.first_snap)
    extent = provider.extent

    render_animation(provider, extent, args, backend)


if __name__ == '__main__':
    main()
