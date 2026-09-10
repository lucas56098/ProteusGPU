#!/usr/bin/env python3
"""Compare two run directories, bitwise.

Takes no tolerance: the point of this suite is that every configuration produces identical
bits, so "close" is a failure. Exits non-zero on any difference, and also on an empty
comparison -- a run that wrote nothing must not read as a pass.
"""
import sys
import glob
import os

import h5py
import numpy as np

SKIP_ATTRS_UNDER = "header/profiler"
MAX_REPORTED = 12


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


def contents(path, with_values=True):
    """Flatten one HDF5 file.

    datasets: {name: (dtype, shape, values or None)}
    attrs:    {"owner@key": (dtype, shape, bytes)}
    groups:   {name}
    """
    datasets, attrs, groups = {}, {}, set()
    with h5py.File(path, "r") as f:

        def take_attrs(name, obj):
            if name.startswith(SKIP_ATTRS_UNDER):
                return
            for k in obj.attrs:
                v = obj.attrs[k]
                attrs[f"{name}@{k}"] = (attr_dtype(obj, k), np.shape(v), attr_value(v))

        take_attrs("/", f)

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = (str(obj.dtype), obj.shape, obj[()] if with_values else None)
            else:
                groups.add(name)
            take_attrs(name, obj)

        f.visititems(visit)
    return datasets, attrs, groups


def describe_value_diff(label, x, y):
    """The old float reporting, kept: for a numeric array the magnitude of the
    disagreement is what tells you whether it is a real divergence or one late bit."""
    n = int(np.asarray(x != y).sum())
    if isinstance(x, np.ndarray) and x.dtype.kind == "f" and x.size:
        scale = max(float(np.abs(x).max()), 1e-300)
        worst = float(np.abs(x - y).max())
        return f"{label}  {n} of {x.size} values differ, max |d| {worst:.3e} (rel {worst / scale:.3e})"
    return f"{label}  {n} values differ"


def compare_files(fa, fb, with_values, bad):
    """Append one line to `bad` per difference. Returns how many things were checked."""
    tag = os.path.basename(fa)
    da, aa, ga = contents(fa, with_values)
    db, ab, gb = contents(fb, with_values)

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
        elif with_values and not np.array_equal(xa, xb):
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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_profile = "--profile" in sys.argv[1:]
    if len(args) != 2:
        print("usage: compare.py <ref_dir> <new_dir> [--profile]", file=sys.stderr)
        return 2
    a_dir, b_dir = args

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
        n_checked += compare_files(fa, fb, True, bad)

    # only a real emptiness, not a structural mismatch that already recorded its finding
    if not n_checked and not bad:
        print("snapshots contained nothing to compare")
        return 1

    n_prof = 0
    if do_profile:
        pa, pb = os.path.join(a_dir, "profile.hdf5"), os.path.join(b_dir, "profile.hdf5")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print("--profile given but profile.hdf5 is missing "
                  f"(ref {os.path.exists(pa)}, new {os.path.exists(pb)}) -- "
                  "is ENABLE_PROFILING set for these builds?")
            return 1
        # structure only: with_values=False leaves the timings out, they are wall clock
        n_prof = compare_files(pa, pb, False, bad)
        if not n_prof and not bad:
            print("profile.hdf5 contained nothing to compare")
            return 1

    if bad:
        print(f"{len(bad)} difference(s):")
        for line in bad[:MAX_REPORTED]:
            print(f"  {line}")
        if len(bad) > MAX_REPORTED:
            print(f"  ... and {len(bad) - MAX_REPORTED} more")
        return 1

    msg = f"{n_checked} datasets+attributes over {len(a_snaps)} snapshots, bitwise identical"
    if do_profile:
        msg += f"; profile.hdf5 structure matches ({n_prof} objects)"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
