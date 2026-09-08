"""Sod shock tube: L1 of density against the exact Riemann solution."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common    # noqa: E402
import riemann   # noqa: E402

# must match ics/create_sod.py
LEFT, RIGHT, X0 = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.5
SAFETY = 1.05   # widen the excluded edges slightly; the order is flat well past this


def main():
    manifest = common.load_manifest()
    g = manifest["gamma"]
    speed = riemann.max_wave_speed(LEFT, RIGHT, g)

    errors = {}
    for run in manifest["runs"]:
        d = common.load(run["outdir"])
        x, t = d["pos"][:, 0], d["time"]

        # the box is periodic, so the wrap-around jump runs its own fan in from each edge
        pad = SAFETY * speed * t
        clean = (x > pad) & (x < 1.0 - pad)
        if clean.mean() < 0.4:
            print(f"        only {clean.mean()*100:.0f}% of the box is uncontaminated at "
                  f"t={t:.3f}; lower CASE_TIME_END")
            return 1

        exact, _, _ = riemann.sample((x[clean] - X0) / t, LEFT, RIGHT, g)
        errors[run["n"]] = common.l1(d["rho"][clean], exact, d["volume"][clean])

    print(f"        comparing |x-0.5| < {0.5 - SAFETY * speed * t:.3f}, "
          f"outside the wrap-around fans")
    return 0 if common.report(errors, manifest["min_order"], "L1(rho)") else 1


if __name__ == "__main__":
    sys.exit(main())
