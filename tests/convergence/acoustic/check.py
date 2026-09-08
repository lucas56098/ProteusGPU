"""Linear sound wave: L1 of density against the travelling-wave solution."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

# must match the defaults in ics/create_acoustic_wave.py
RHO_0, P_0, DELTA_RHO = 1.0, 3.0 / 5.0, 1.0e-6


def main():
    manifest = common.load_manifest()
    c_s = np.sqrt(manifest["gamma"] * P_0 / RHO_0)

    errors = {}
    for run in manifest["runs"]:
        d = common.load(run["outdir"])
        x = d["pos"][:, 0]
        exact = RHO_0 + DELTA_RHO * np.sin(2.0 * np.pi * (x - c_s * d["time"]))
        errors[run["n"]] = common.l1(d["rho"], exact, d["volume"])

    return 0 if common.report(errors, manifest["min_order"], "L1(rho)") else 1


if __name__ == "__main__":
    sys.exit(main())
