#!/usr/bin/env python3
"""Compare two run directories, bitwise.

Takes no tolerance: the point of this suite is that every configuration produces identical
bits, so "close" is a failure. Exits non-zero on any difference, and also on an empty
comparison -- a run that wrote nothing must not read as a pass.

The cells are compared in the order of their mesh/id, and the faces of a cell in their order,
each neighbour as its id; so runs on different numbers of ranks compare cell by cell. What
depends on the number of ranks by its nature is left out: header/nranks and mesh/face_offset.
"""
import sys
import glob
import os

import h5py
import numpy as np

MAX_REPORTED = 12
DIFFERS_WITH_THE_RANKS = ("header@nranks", "mesh/face_offset")


def snapshots(d):
    return sorted(glob.glob(os.path.join(d, "snapshot_*.hdf5")))


def attr_value(v):
    """Attribute payload as comparable bytes. Strings go through repr, since a
    variable-length string attribute has no meaningful tobytes()."""
    a = np.asarray(v)
    if a.dtype.kind in ("S", "U", "O"):
        return repr(a.tolist()).encode()
    return a.tobytes()


def attr_dtype(obj, key):
    """The dtype as stored in the file, not as numpy chooses to hand it back."""
    try:
        return str(obj.attrs.get_id(key).dtype)
    except Exception:
        return str(np.asarray(obj.attrs[key]).dtype)


def in_order_of_ids(datasets):
    """The cells sorted by mesh/id and the faces in the order of their cells; face_neighbor as ids."""
    if "mesh/id" not in datasets:
        return datasets
    ids = datasets["mesh/id"][2]
    order = np.argsort(ids, kind="stable")
    face_index = None
    if "mesh/face_offset" in datasets:
        first = datasets["mesh/face_offset"][2][:-1][order]
        count = datasets["mesh/n_faces"][2][order]
        start_in_order = np.cumsum(count) - count
        face_index = np.repeat(first - start_in_order, count) + np.arange(count.sum())
    ordered = {}
    for name, (dtype, shape, values) in datasets.items():
        if name in DIFFERS_WITH_THE_RANKS:
            continue
        if name.startswith("mesh/face_"):
            values = values[face_index]
            if name == "mesh/face_neighbor":
                values = ids[values]
        elif len(shape) > 0 and shape[0] == len(ids):
            values = values[order]
        ordered[name] = (dtype, values.shape, values)
    return ordered


def contents(path):
    """Flatten one HDF5 file, the cells in the order of their ids.

    datasets: {name: (dtype, shape, values)}
    attrs:    {"owner@key": (dtype, shape, bytes)}
    groups:   {name}
    """
    datasets, attrs, groups = {}, {}, set()
    with h5py.File(path, "r") as f:

        def take_attrs(name, obj):
            for k in obj.attrs:
                if f"{name}@{k}" in DIFFERS_WITH_THE_RANKS:
                    continue
                v = obj.attrs[k]
                attrs[f"{name}@{k}"] = (attr_dtype(obj, k), np.shape(v), attr_value(v))

        take_attrs("/", f)

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = (str(obj.dtype), obj.shape, obj[()])
            else:
                groups.add(name)
            take_attrs(name, obj)

        f.visititems(visit)
    return in_order_of_ids(datasets), attrs, groups


def describe_value_diff(label, x, y):
    """The old float reporting, kept: for a numeric array the magnitude of the
    disagreement is what tells you whether it is a real divergence or one late bit."""
    n = int(np.asarray(x != y).sum())
    if isinstance(x, np.ndarray) and x.dtype.kind == "f" and x.size:
        scale = max(float(np.abs(x).max()), 1e-300)
        worst = float(np.abs(x - y).max())
        return f"{label}  {n} of {x.size} values differ, max |d| {worst:.3e} (rel {worst / scale:.3e})"
    return f"{label}  {n} values differ"


def compare_files(fa, fb, bad):
    """Append one line to `bad` per difference. Returns how many things were checked."""
    tag = os.path.basename(fa)
    da, aa, ga = contents(fa)
    db, ab, gb = contents(fb)

    for what, sa, sb in (("datasets", set(da), set(db)),
                         ("attributes", set(aa), set(ab)),
                         ("groups", ga, gb)):
        only = sa ^ sb
        if only:
            bad.append(f"{tag}: {what} present in one run only: {sorted(only)[:6]}")
            return 0  # structure differs, per-item comparison would be noise

    for k in sorted(da):
        (ta, sha, xa), (tb, shb, xb) = da[k], db[k]
        if ta != tb:
            bad.append(f"{tag}:{k}  dtype {ta} vs {tb}")
        elif sha != shb:
            bad.append(f"{tag}:{k}  shape {sha} vs {shb}")
        elif not np.array_equal(xa, xb):
            bad.append(describe_value_diff(f"{tag}:{k}", xa, xb))

    for k in sorted(aa):
        (ta, sha, va), (tb, shb, vb) = aa[k], ab[k]
        if ta != tb:
            bad.append(f"{tag}:{k}  attr dtype {ta} vs {tb}")
        elif sha != shb:
            bad.append(f"{tag}:{k}  attr shape {sha} vs {shb}")
        elif va != vb:
            bad.append(f"{tag}:{k}  attr value differs")

    return len(da) + len(aa)


def main():
    if len(sys.argv) != 3:
        print("usage: compare.py <ref_dir> <new_dir>", file=sys.stderr)
        return 2
    a_dir, b_dir = sys.argv[1:]

    a_snaps, b_snaps = snapshots(a_dir), snapshots(b_dir)
    if not a_snaps or not b_snaps:
        print(f"no snapshots to compare (ref {len(a_snaps)}, new {len(b_snaps)})")
        return 1
    if len(a_snaps) != len(b_snaps):
        print(f"snapshot count differs: ref {len(a_snaps)}, new {len(b_snaps)}")
        return 1

    n_checked = 0
    bad = []
    for fa, fb in zip(a_snaps, b_snaps):
        n_checked += compare_files(fa, fb, bad)

    # only a real emptiness, not a structural mismatch that already recorded its finding
    if not n_checked and not bad:
        print("snapshots contained nothing to compare")
        return 1

    if bad:
        print(f"{len(bad)} difference(s):")
        for line in bad[:MAX_REPORTED]:
            print(f"  {line}")
        if len(bad) > MAX_REPORTED:
            print(f"  ... and {len(bad) - MAX_REPORTED} more")
        return 1

    print(f"{n_checked} datasets+attributes over {len(a_snaps)} snapshots, bitwise identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
