#!/usr/bin/env python3
"""Checks that every #include in src/ follows the module table below.

    check_dependencies.py <src dir>
        exits with 1 if any include breaks a rule
    check_dependencies.py --self-test <src dir>
        exits with 1 unless the errors found are exactly the lines marked "// forbidden"

Rules, for a file of module m (its folder under src/; files directly in src/ belong to main):
    "m/..."         allowed
    "x/..."         allowed only if m may use x, and then only the public header "x/x.h"
    "file.h"        a private header, which must be in the same folder
    "...cu"         never
    "../..."        never
A folder in src/ that is not a module of the table is an error as well.

The table: "May use" is transitive. A layer name stands for every module of that layer, "all below" for every
module above in the table. A module may only use modules that come before it, so there are no cycles.
"""

import re
import sys
from pathlib import Path

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
FORBIDDEN_MARK = "// forbidden"

# the modules from the bottom up: layer, module, what it may use
MODULE_TABLE = [
    ("core", "config", ""),
    ("core", "timers", "config"),
    ("core", "mpi", "config, timers"),
    ("core", "logging", "mpi"),
    ("core", "math", "config"),
    ("core", "device", "logging, timers"),
    ("core", "profile", "timers, mpi, logging, device"),
    ("libraries", "seed", "math"),
    ("libraries", "keys", "math"),
    ("libraries", "ngbtree", "keys, device"),
    ("libraries", "voronoi_cell", "ngbtree"),
    ("libraries", "gas", "math"),
    ("libraries", "riemann", "gas"),
    ("ranks", "domain", "keys, mpi, device"),
    ("ranks", "ghosts", "domain, ngbtree, mpi, device"),
    ("ranks", "migration", "domain, mpi, device"),
    ("mesh", "mesh", "voronoi_cell, ghosts, seed, device, timers"),
    ("time", "timesteps", "mesh, ghosts"),
    ("physics", "mesh_motion", "gas, mesh"),
    ("physics", "gradients", "gas, mesh, timesteps"),
    ("physics", "hydro", "riemann, gradients, timesteps"),
    ("physics", "signals", "gas, mesh"),
    ("physics", "astro", "gas, timesteps, hydro"),
    ("app", "cells", "physics, time, migration"),
    ("app", "params", "all below"),
    ("app", "io", "all below"),
    ("app", "simulation", "all below"),
    ("app", "begrun", "all below"),
    ("app", "integrator", "all below"),
    ("app", "main", "all below"),
]


def main():
    args = sys.argv[1:]
    self_test = args[:1] == ["--self-test"]
    if self_test:
        args = args[1:]
    if len(args) != 1:
        sys.exit(__doc__)
    src = Path(args[0])

    may_use = read_module_table()
    errors = check_includes(src, may_use)

    ok = check_self_test(src, errors) if self_test else report(errors)
    sys.exit(0 if ok else 1)


# ==========================================================
# the module table
# ==========================================================


def read_module_table():
    """Every module of the table, with the set of modules it may use, directly or through others."""
    layers = {}
    for layer, module, _ in MODULE_TABLE:
        layers.setdefault(layer, []).append(module)

    direct = {}
    for index, (_, module, cell) in enumerate(MODULE_TABLE):
        earlier = [m for _, m, _ in MODULE_TABLE[:index]]
        direct[module] = expand(cell, module, earlier, layers)
    return {module: reachable(module, direct) for module in direct}


def expand(cell, module, earlier, layers):
    """The modules one "may use" text names; each of them must come before the module in the table."""
    if not cell:
        return set()
    names = set()
    for token in (t.strip() for t in cell.split(",")):
        if token == "all below":
            names.update(earlier)
        elif token in layers:
            names.update(layers[token])
        else:
            names.add(token)
    later = names - set(earlier)
    if later:
        sys.exit(f"{module} may only use modules above it in the table, not {', '.join(sorted(later))}")
    return names


def reachable(module, direct):
    """Every module that module uses, directly or through other modules."""
    found, todo = set(), list(direct[module])
    while todo:
        other = todo.pop()
        if other not in found:
            found.add(other)
            todo.extend(direct[other])
    return found


# ==========================================================
# the includes
# ==========================================================


def check_includes(src, may_use):
    """Every include that breaks a rule, as (file, line number, message)."""
    errors = []
    for path in sorted(p for p in src.rglob("*") if p.suffix in (".cu", ".h")):
        module = module_of(path, src)
        if module not in may_use:
            errors.append((path, 1, f"folder {module} is not a module of the table"))
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            match = INCLUDE.match(line)
            if match:
                message = include_error(match.group(1), path, module, may_use)
                if message:
                    errors.append((path, number, message))
    return errors


def module_of(path, src):
    parts = path.relative_to(src).parts
    return parts[0] if len(parts) > 1 else "main"


def include_error(target, path, module, may_use):
    """Why a file of module may not include target; None if it may."""
    if target.endswith(".cu"):
        return f'includes a .cu file: "{target}"'
    if "../" in target:
        return f'includes a path with "../": "{target}"'
    if "/" not in target:
        if not (path.parent / target).is_file():
            return f'"{target}" is not a header in the folder of this file'
        return None

    other = target.split("/")[0]
    if other == module:
        return None
    if other not in may_use:
        return f'{other} is not a module of the table: "{target}"'
    if other not in may_use[module]:
        return f'{module} may not use {other}: "{target}"'
    if target != f"{other}/{other}.h":
        return f'only the public header "{other}/{other}.h" of {other} may be included: "{target}"'
    return None


# ==========================================================
# the result
# ==========================================================


def report(errors):
    for path, number, message in errors:
        print(f"{path}:{number}: {message}")
    if errors:
        print(f"dependency check: {len(errors)} include(s) break the module table")
    else:
        print("dependency check: every include follows the module table")
    return not errors


def check_self_test(src, errors):
    """True if the errors are exactly the lines marked // forbidden."""
    marked = set()
    for path in sorted(p for p in src.rglob("*") if p.suffix in (".cu", ".h")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if FORBIDDEN_MARK in line:
                marked.add((path, number))
    found = {(path, number) for path, number, _ in errors}

    for path, number in sorted(marked - found):
        print(f"{path}:{number}: marked forbidden, but the check did not find it")
    for path, number in sorted(found - marked):
        print(f"{path}:{number}: found by the check, but not marked forbidden")
    if marked == found:
        print(f"dependency check self-test: all {len(marked)} forbidden includes found, nothing else")
    return marked == found


if __name__ == "__main__":
    main()
