"""Gresho vortex: L1 of the azimuthal velocity against the initial profile.

The vortex is a steady solution, so the exact answer at any time is the profile itself.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402


def exact_v_phi(xi):
    """Rotation curve from ics/create_gresho.py: linear rise, linear fall, then still."""
    return np.where(xi < 0.2, 5.0 * xi, np.where(xi < 0.4, 2.0 - 5.0 * xi, 0.0))


def main():
    manifest = common.load_manifest()

    errors = {}
    for run in manifest["runs"]:
        d = common.load(run["outdir"])
        x, y = d["pos"][:, 0] - 0.5, d["pos"][:, 1] - 0.5
        r = np.sqrt(x * x + y * y)

        v_phi = np.zeros_like(r)
        turning = r > 0.0
        # same sign convention as the IC: v = v_phi * (y, -x) / r
        v_phi[turning] = (d["vel"][turning, 0] * y[turning]
                          - d["vel"][turning, 1] * x[turning]) / r[turning]

        errors[run["n"]] = common.l1(v_phi, exact_v_phi(r), d["volume"])

    return 0 if common.report(errors, manifest["min_order"], "L1(v_phi)") else 1


if __name__ == "__main__":
    sys.exit(main())
