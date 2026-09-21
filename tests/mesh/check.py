#!/usr/bin/env python3
"""
Checks the Voronoi mesh a run wrote with OUTPUT_MESH.

Usage: check.py OUTPUT_DIR

For every snapshot in OUTPUT_DIR, with all rank files of one snapshot taken together:

  volume sum   the cell volumes add up to the box
  wall faces   no face lies on the box wall (neighbour -1)
  closed       every cell is closed: its area weighted face normals sum to zero
  both sides   a face between two cells of one rank is seen from both sides, same area
  scipy        every cell matches scipy.spatial.Voronoi of the same seeds in the periodic
               box: volume, centroid, and the area and normal of each face

The box is [0, 1) on every axis and periodic. Faces smaller than SMALL_FACE times the
surface of their cell are left out of the face comparison on both sides: there a
difference in rounding decides whether a face exists at all.

Exit status is non-zero if any snapshot fails.
"""

import glob
import os
import re
import sys

import h5py
import numpy as np
from scipy.spatial import Voronoi

TOL_VOLUME_SUM = 1e-10  # |sum V - 1|
TOL_VOLUME     = 1e-9   # |V - V_scipy| / V_scipy
TOL_CENTROID   = 1e-9   # |c - c_scipy| / mean spacing
TOL_CLOSED     = 1e-9   # |sum A n| / sum A
TOL_AREA       = 1e-7   # |A - A_scipy| / surface of the cell
TOL_NORMAL     = 1e-7   # 1 - n . n_scipy
SMALL_FACE     = 1e-7   # faces below this fraction of the cell surface are not compared
SHOW           = 5      # offending cells printed per check


# ============================================================
# reading
# ============================================================

def snapshot_files(out_dir):
    """{snapshot number: [file of rank 0, file of rank 1, ...]}"""
    groups = {}
    for path in glob.glob(os.path.join(out_dir, "snapshot_*.hdf5")):
        m = re.fullmatch(r"snapshot_(\d+)(?:\.(\d+))?\.hdf5", os.path.basename(path))
        if m:
            groups.setdefault(int(m.group(1)), []).append((int(m.group(2) or 0), path))
    return {s: [p for _, p in sorted(files)] for s, files in sorted(groups.items())}


def read_rank(path):
    with h5py.File(path, "r") as f:
        if "mesh/face_area" not in f:
            raise SystemExit(f"{path}: no mesh/face_area, the run was not built with OUTPUT_MESH")
        m = f["mesh"]
        return {
            "pos": m["pos"][:],
            "volume": m["volume"][:],
            "centroid": m["centroid"][:],
            "face_offset": m["face_offset"][:],
            "face_neighbor": m["face_neighbor"][:],
            "face_area": m["face_area"][:],
            "face_normal": m["face_normal"][:],
        }


# ============================================================
# reference: scipy on the same seeds
# ============================================================

def polygon_area_centroid(verts, normal):
    """area and centroid of a planar convex polygon, vertices in any order"""
    c0 = verts.mean(axis=0)
    u = verts[0] - c0
    u -= normal * np.dot(u, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rel = verts - c0
    order = np.argsort(np.arctan2(rel @ v, rel @ u))
    ring = verts[order]
    area, centroid = 0.0, np.zeros(3)
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        t = 0.5 * np.linalg.norm(np.cross(a - c0, b - c0))
        area += t
        centroid += t * (c0 + a + b) / 3.0
    return area, centroid / area


def reference_cells(pos):
    """per cell: volume, centroid and a list of (unit normal, area) faces"""
    n, dim = pos.shape
    spacing = n ** (-1.0 / dim)
    margin = min(0.5, 8.0 * spacing)

    # the seeds plus every periodic copy within the margin; the real seeds come first
    shifts = np.array(np.meshgrid(*[[0, -1, 1]] * dim, indexing="ij")).reshape(dim, -1).T
    pts, owner = [pos], [np.arange(n)]
    for s in shifts[1:]:
        p = pos + s
        keep = np.all((p > -margin) & (p < 1.0 + margin), axis=1)
        pts.append(p[keep])
        owner.append(np.arange(n)[keep])
    pts = np.concatenate(pts)
    vor = Voronoi(pts)

    for i in range(n):
        if -1 in vor.regions[vor.point_region[i]]:
            raise SystemExit(f"scipy: cell {i} is open, the periodic margin {margin} is too small")

    faces = [[] for _ in range(n)]
    volume = np.zeros(n)
    moment = np.zeros((n, dim))
    for (a, b), rv in zip(vor.ridge_points, vor.ridge_vertices):
        for c, o in ((a, b), (b, a)):
            if c >= n:
                continue
            d = pts[o] - pts[c]
            dist = np.linalg.norm(d)
            normal = d / dist
            verts = vor.vertices[rv]
            if dim == 2:
                area = np.linalg.norm(verts[1] - verts[0])
                # triangle seed, a, b
                tri = 0.5 * abs((verts[0, 0] - pts[c, 0]) * (verts[1, 1] - pts[c, 1]) -
                                (verts[0, 1] - pts[c, 1]) * (verts[1, 0] - pts[c, 0]))
                volume[c] += tri
                moment[c] += tri * (pts[c] + verts[0] + verts[1]) / 3.0
            else:
                area, g = polygon_area_centroid(verts, normal)
                # pyramid from the seed to the face
                pyr = area * 0.5 * dist / 3.0
                volume[c] += pyr
                moment[c] += pyr * (pts[c] + 0.75 * (g - pts[c]))
            faces[c].append((normal, area))
    return volume, moment / volume[:, None], faces


# ============================================================
# the checks
# ============================================================

def wrap(d):
    return d - np.round(d)


def check_snapshot(files):
    ranks = [read_rank(p) for p in files]
    dim = ranks[0]["pos"].shape[1]
    pos = np.concatenate([r["pos"] for r in ranks]) % 1.0
    n = len(pos)
    spacing = n ** (-1.0 / dim)

    problems = []
    worst = {}

    def report(check, lines):
        if lines:
            problems.append(f"{check}: {len(lines)} cell(s)")
            problems.extend("    " + s for s in lines[:SHOW])

    # volume sum
    vol_sum = sum(r["volume"].sum() for r in ranks)
    worst["sum V - 1"] = abs(vol_sum - 1.0)
    if abs(vol_sum - 1.0) > TOL_VOLUME_SUM:
        problems.append(f"volume sum: {vol_sum!r}, box is 1")

    # per rank: wall faces, closed cells, both sides
    walls, open_cells, one_sided = [], [], []
    worst_closed = 0.0
    for rk, r in enumerate(ranks):
        off, nb, area, normal = r["face_offset"], r["face_neighbor"], r["face_area"], r["face_normal"]
        n_loc = len(r["pos"])
        seen = {}
        for k in range(n_loc):
            lo, hi = off[k], off[k + 1]
            if np.any(nb[lo:hi] < 0):
                walls.append(f"rank {rk} cell {k} at {r['pos'][k]}: {int(np.sum(nb[lo:hi] < 0))} wall face(s)")
            surface = area[lo:hi].sum()
            closed = np.linalg.norm((area[lo:hi, None] * normal[lo:hi]).sum(axis=0)) / surface
            worst_closed = max(worst_closed, closed)
            if closed > TOL_CLOSED:
                open_cells.append(f"rank {rk} cell {k}: |sum A n| / sum A = {closed:.3e}")
            for f in range(lo, hi):
                if 0 <= nb[f] < n_loc:
                    seen[(k, int(nb[f]))] = area[f]
        for (k, j), a in seen.items():
            b = seen.get((j, k))
            if b is None:
                one_sided.append(f"rank {rk}: face {k} -> {j} has no face {j} -> {k}")
            elif abs(a - b) > TOL_AREA * max(a, b, 1e-300):
                one_sided.append(f"rank {rk}: face {k} <-> {j} area {a!r} vs {b!r}")
    worst["closed"] = worst_closed
    report("wall faces", walls)
    report("not closed", open_cells)
    report("both sides", one_sided)

    # scipy
    ref_vol, ref_cen, ref_faces = reference_cells(pos)
    vol = np.concatenate([r["volume"] for r in ranks])
    cen = np.concatenate([r["centroid"] for r in ranks])
    d_vol = np.abs(vol - ref_vol) / ref_vol
    d_cen = np.linalg.norm(wrap(cen - ref_cen), axis=1) / spacing
    worst["dV/V"] = d_vol.max()
    worst["dc/spacing"] = d_cen.max()
    report("scipy volume", [f"cell {i} at {pos[i]}: V {vol[i]!r} vs scipy {ref_vol[i]!r}"
                            for i in np.nonzero(d_vol > TOL_VOLUME)[0]])
    report("scipy centroid", [f"cell {i} at {pos[i]}: off by {d_cen[i]:.3e} spacings"
                              for i in np.nonzero(d_cen > TOL_CENTROID)[0]])

    bad_faces = []
    worst_area = worst_normal = 0.0
    i = 0
    for rk, r in enumerate(ranks):
        off, area, normal = r["face_offset"], r["face_area"], r["face_normal"]
        for k in range(len(r["pos"])):
            lo, hi = off[k], off[k + 1]
            surface = area[lo:hi].sum()
            mine = [(normal[f], area[f]) for f in range(lo, hi) if area[f] >= SMALL_FACE * surface]
            ref = [(nn, a) for nn, a in ref_faces[i] if a >= SMALL_FACE * surface]
            if len(mine) != len(ref):
                bad_faces.append(f"cell {i} at {pos[i]}: {len(mine)} faces, scipy {len(ref)}")
            else:
                ref_n = np.array([nn for nn, _ in ref])
                used = set()
                for nn, a in mine:
                    j = int(np.argmax(ref_n @ nn))
                    miss = 1.0 - float(ref_n[j] @ nn)
                    da = abs(a - ref[j][1]) / surface
                    worst_normal = max(worst_normal, miss)
                    worst_area = max(worst_area, da)
                    if j in used or miss > TOL_NORMAL or da > TOL_AREA:
                        bad_faces.append(f"cell {i} at {pos[i]}: face normal {nn} area {a!r}, "
                                         f"nearest scipy face {ref_n[j]} area {ref[j][1]!r}")
                        break
                    used.add(j)
            i += 1
    worst["dA/surface"] = worst_area
    worst["1 - n.n"] = worst_normal
    report("scipy faces", bad_faces)

    n_faces = sum(len(r["face_area"]) for r in ranks)
    return n, n_faces, worst, problems


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(sys.argv) == 2 else 2
    snaps = snapshot_files(sys.argv[1])
    if not snaps:
        print(f"no snapshots in {sys.argv[1]}")
        return 1

    failed = 0
    for s, files in snaps.items():
        n, n_faces, worst, problems = check_snapshot(files)
        detail = "  ".join(f"{k} {v:.1e}" for k, v in worst.items())
        print(f"snapshot_{s}: {n} cells, {n_faces} faces, {len(files)} file(s)  {detail}")
        if problems:
            failed += 1
            for p in problems:
                print("  " + p)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
