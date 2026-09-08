"""Shared helpers for the convergence cases under tests/convergence/.

A case is a directory holding two files: case.sh says what to run, check.py says what
to measure. The runner builds the code, does the runs, and writes a manifest naming one
output directory per resolution; check.py reads it, computes an error norm for each, and
reports the observed convergence order.

A case never has to know how it was built or launched -- load() hides the per-rank files
an MPI run produces, so the same check.py serves every variant.
"""

import glob
import json
import os
import sys

import h5py
import numpy as np


def case_dir():
    """Directory of the check.py being run."""
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def load_manifest(path=None):
    """The runner's manifest, from argv[1] unless given explicitly."""
    with open(path or sys.argv[1]) as f:
        return json.load(f)


def _snapshot_files(outdir):
    """Every file belonging to the last snapshot in outdir, across ranks."""
    files = glob.glob(os.path.join(outdir, "snapshot_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no snapshot written in {outdir}")

    # snapshot_<n>.hdf5 serially, snapshot_<n>.<rank>.hdf5 under MPI
    def number(path):
        return int(os.path.basename(path).split("_")[1].split(".")[0])

    last = max(number(p) for p in files)
    return sorted(p for p in files if number(p) == last)


def load(outdir):
    """Fields of the last snapshot, with an MPI run's per-rank files concatenated."""
    fields = {"pos": [], "rho": [], "vel": [], "energy": [], "volume": []}
    time = None
    for path in _snapshot_files(outdir):
        with h5py.File(path, "r") as f:
            time = float(f["header"].attrs["time"])
            if "volume" not in f["mesh"]:
                raise KeyError(f"{path} has no mesh/volume -- build with OUTPUT_MESH")
            fields["pos"].append(f["mesh/pos"][...])
            fields["rho"].append(f["hydro/rho"][...])
            fields["vel"].append(f["hydro/vel"][...])
            fields["energy"].append(f["hydro/energy"][...])
            fields["volume"].append(f["mesh/volume"][...])

    data = {k: np.concatenate(v) for k, v in fields.items()}
    data["time"] = time
    return data


def l1(values, exact, volume, mask=None):
    """Volume-weighted L1 error. Cells are not equal size, least of all on a moving mesh."""
    if mask is not None:
        values, exact, volume = values[mask], exact[mask], volume[mask]
    return float(np.sum(np.abs(values - exact) * volume) / np.sum(volume))


def report(errors, min_order, label="L1"):
    """Print the error ladder with observed orders. True if every step clears min_order.

    Order comes from the actual resolution ratio, so a ladder need not double.
    """
    resolutions = sorted(errors)
    print(f"        {'n':>6}  {label:>13}  {'order':>6}")
    ok = True
    prev_n = prev_e = None
    for n in resolutions:
        e = errors[n]
        if prev_e is None:
            print(f"        {n:6d}  {e:13.4e}  {'-':>6}")
        else:
            order = float(np.log(prev_e / e) / np.log(n / prev_n))
            below = order < min_order
            ok = ok and not below
            note = f"   below {min_order}" if below else ""
            print(f"        {n:6d}  {e:13.4e}  {order:6.2f}{note}")
        prev_n, prev_e = n, e

    if not ok:
        print(f"        expected order >= {min_order} at every step")
    return ok
