#!/usr/bin/env python3

import sys
import os
import statistics

import h5py


def totals_of(run_dir):
    """{scope path: seconds} for one run, combined across ranks with max."""
    path = os.path.join(run_dir, "profile.hdf5")
    out = {}
    with h5py.File(path, "r") as f:
        for rank in f:
            grp = f[rank].get("cumulative")
            if grp is None:
                continue
            for scope, ds in grp.items():
                v = float(ds[()][-1])
                out[scope] = max(out.get(scope, 0.0), v)
    if not out:
        raise ValueError(f"no cumulative scopes in {path}")
    return out


def collect(dirs):
    """{scope: [seconds per run]}, keeping only scopes present in every run."""
    per_run = [totals_of(d) for d in dirs]
    common = set(per_run[0])
    for r in per_run[1:]:
        common &= set(r)
    return {s: [r[s] for r in per_run] for s in common}


def spread(vals):
    """max-min as a fraction of the median -- how much to trust the median."""
    m = statistics.median(vals)
    return (max(vals) - min(vals)) / m if m > 0 else 0.0


def band(vals):
    """max-min in seconds: how far this scope moves when nothing has changed."""
    return max(vals) - min(vals)


def main():
    if "--" not in sys.argv:
        print("usage: compare_profiles.py <ref_dir>... -- <new_dir>... [--top N]", file=sys.stderr)
        return 2
    args = sys.argv[1:]
    top = 12
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i : i + 2]

    min_pct = None
    if "--min-pct" in args:
        i = args.index("--min-pct")
        min_pct = float(args[i + 1])
        del args[i : i + 2]
    cut = args.index("--")
    ref_dirs, new_dirs = args[:cut], args[cut + 1 :]
    if not ref_dirs or not new_dirs:
        print("need at least one run directory on each side", file=sys.stderr)
        return 2

    ref, new = collect(ref_dirs), collect(new_dirs)
    shared = sorted(set(ref) & set(new))
    if not shared:
        print("no scopes in common -- were both sides built with ENABLE_PROFILING?", file=sys.stderr)
        return 2

    run_total = statistics.median(ref["TOTAL"]) if "TOTAL" in ref else 0.0

    rows = []
    for s in shared:
        a, b = statistics.median(ref[s]), statistics.median(new[s])
        
        noise = max(band(ref[s]), band(new[s]))
        
        if min_pct is None:
            real = abs(b - a) > noise and abs(b - a) >= 0.001 * run_total
        else:
            real = a > 0 and abs(100.0 * (b - a) / a) >= min_pct and abs(b - a) >= 0.01 * run_total
        rows.append((s, a, b, b - a, max(spread(ref[s]), spread(new[s])), real))

    total = [r for r in rows if r[0] == "TOTAL"]
    rest = sorted((r for r in rows if r[0] != "TOTAL"), key=lambda r: -abs(r[3]))

    print(f"    {'scope':<30}{'ref':>9}{'new':>9}{'delta':>10}{'pct':>8}{'spread':>8}")
    any_noise = False
    for name, a, b, d, sp, real in total + rest[:top]:
        pct = (100.0 * d / a) if a > 0 else 0.0
        label = "TOTAL (wall)" if name == "TOTAL" else name[len("TOTAL.") :] if name.startswith("TOTAL.") else name
        flag = "" if real else " ~"
        any_noise = any_noise or not real
        print(f"    {label:<30}{a:>8.3f}s{b:>8.3f}s{d:>+9.3f}s{pct:>+7.1f}%{100 * sp:>7.1f}%{flag}")
    if any_noise and min_pct is None:
        print("\n    ~ delta is inside that scope's run-to-run scatter, or under 0.1% of the")
        print("      run: not a finding, whatever the percentage says. More --reps narrows it.")
    elif any_noise:
        print(f"\n    ~ under {min_pct:.0f}%, or under 1% of the run. One run per side measures no")
        print("      scatter, so this only catches something badly wrong. Drop --quick to measure.")
    print(f"\n    {len(shared)} scopes compared, {len(ref_dirs)} vs {len(new_dirs)} runs each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
