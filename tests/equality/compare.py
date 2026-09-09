#!/usr/bin/env python3
"""Compare two snapshot directories dataset by dataset, bitwise.

Takes no tolerance: the point of this suite is that CPU and GPU produce identical bits,
so "close" is a failure. Exits non-zero on any difference, and also on an empty
comparison -- a run that wrote nothing must not read as a pass.
"""
import sys
import glob
import os

import h5py
import numpy as np


def snapshots(d):
    return sorted(glob.glob(os.path.join(d, "snapshot_*.hdf5")))


def datasets(f):
    out = {}
    f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    a_snaps, b_snaps = snapshots(a_dir), snapshots(b_dir)

    if not a_snaps or not b_snaps:
        print(f"no snapshots to compare (cpu {len(a_snaps)}, gpu {len(b_snaps)})")
        return 1
    if len(a_snaps) != len(b_snaps):
        print(f"snapshot count differs: cpu {len(a_snaps)}, gpu {len(b_snaps)}")
        return 1

    n_ds = 0
    bad = []
    for fa, fb in zip(a_snaps, b_snaps, strict=True):
        with h5py.File(fa) as ha, h5py.File(fb) as hb:
            da, db = datasets(ha), datasets(hb)
            only = set(da) ^ set(db)
            if only:
                print(f"{os.path.basename(fa)}: datasets differ between runs: {sorted(only)}")
                return 1
            for k in sorted(da):
                n_ds += 1
                x, y = da[k], db[k]
                if np.array_equal(x, y):
                    continue
                if np.shape(x) != np.shape(y):
                    # one backend produced a different number of cells: report it plainly
                    # rather than letting the comparison broadcast and throw
                    bad.append(f"{os.path.basename(fa)}:{k}  shape {np.shape(x)} vs {np.shape(y)}")
                    continue
                n = int(np.asarray(x != y).sum())
                if isinstance(x, np.ndarray) and x.dtype.kind == "f":
                    scale = max(float(np.abs(x).max()), 1e-300)
                    worst = float(np.abs(x - y).max())
                    bad.append(f"{os.path.basename(fa)}:{k}  {n} of {x.size} values differ, "
                               f"max |d| {worst:.3e} (rel {worst / scale:.3e})")
                else:
                    bad.append(f"{os.path.basename(fa)}:{k}  {n} values differ")

    if not n_ds:
        print("snapshots contained no datasets")
        return 1

    if bad:
        print(f"{len(bad)} of {n_ds} datasets differ:")
        for line in bad[:12]:
            print(f"  {line}")
        if len(bad) > 12:
            print(f"  ... and {len(bad) - 12} more")
        return 1

    print(f"{n_ds} datasets over {len(a_snaps)} snapshots, bitwise identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
