#!/usr/bin/env bash
# Proteus CPU/GPU bitwise equality
# Deliberately NOT CUDA_FAST_MATH
# Scope: ASTRO_PHYSICS off (inside are still atomics making stuff non deterministic)

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/.." && pwd)"
CASES_DIR="$SELF_DIR/equality"

usage() {
    cat <<'USAGE'
Usage: tests/cpu_gpu_equality.sh [options]

Builds each case for CPU and GPU, runs both on the same IC, and compares every snapshot
dataset bitwise. A difference means the backends have diverged numerically, which breaks
the guarantee that a GPU result can be reproduced on a laptop.

Options:
  --only NAME    run just one case (the directory name)
  --list         print the cases that would run, and exit
  --keep         keep builds, ICs and snapshots instead of deleting them
  -h, --help     show this message

To add a case, create tests/equality/<name>/ with a case.sh setting CASE_DESC, CASE_DIM,
CASE_FLAGS, CASE_N, CASE_TIME_END and CASE_IC. The runner needs no edit, and cases sharing
a (dim, flags) pair share their builds, which is most of the runtime -- the whole suite is
well under a minute.

Exit status is non-zero if any case fails. Without nvcc every case is skipped and counted
separately: a skip is not a pass.
USAGE
}

LIST=0; KEEP=0; ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --only)    ONLY="${2:-}"; shift ;;
        --list)    LIST=1 ;;
        --keep)    KEEP=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd "$REPO"
[ -d "$CASES_DIR" ] || { echo "missing cases directory: $CASES_DIR" >&2; exit 2; }

case "$(uname -s)" in
    Linux)  SYSTYPE=Ubuntu ;;
    Darwin) SYSTYPE=macOS ;;
    *) echo "cannot guess SYSTYPE for $(uname -s)" >&2; exit 2 ;;
esac

CASES=()
for d in "$CASES_DIR"/*/; do
    name="$(basename "$d")"
    [ -f "$d/case.sh" ] || continue
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
    CASES+=("$name")
done
[ "${#CASES[@]}" -gt 0 ] || { echo "no cases found in $CASES_DIR" >&2; exit 2; }

have() { command -v "$1" >/dev/null 2>&1; }
CAP_NVCC=no; have nvcc && CAP_NVCC=yes

printf '\033[1mCPU/GPU bitwise equality\033[0m\n'
printf '  cases      %s\n' "${CASES[*]}"
printf '  nvcc       %s\n' "$CAP_NVCC"
printf '  gpu build  CUDA without CUDA_FAST_MATH (strict div/sqrt, no FTZ)\n'
printf '\n'

if [ "$LIST" -eq 1 ]; then
    for c in "${CASES[@]}"; do
        # shellcheck disable=SC1090
        ( . "$CASES_DIR/$c/case.sh"
          printf '  %-12s %-32s dim=%s  n=%s  t_end=%s  flags="%s"\n' \
                 "$c" "$CASE_DESC" "$CASE_DIM" "$CASE_N" "$CASE_TIME_END" "$CASE_FLAGS" )
    done
    exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/proteus-equality.XXXXXX")"
cleanup() { if [ "$KEEP" -eq 1 ]; then echo "kept: $WORK"; else rm -rf "$WORK"; fi; }
trap cleanup EXIT

build_for() {   # dim backend flags -> prints exec path, or nothing on failure
    local dim="$1" backend="$2" flags="$3" key dir cfg
    key="${dim}d_${backend}_$(echo "$flags" | tr ' ' '_')"
    dir="$WORK/build_$key"
    if [ -x "$dir/ProteusGPU" ]; then printf '%s' "$dir/ProteusGPU"; return 0; fi
    cfg="$WORK/config_$key.sh"
    {
        printf 'dim_%sD\n' "$dim"
        if [ "$backend" = gpu ]; then printf 'CUDA\n'; else printf 'CPU_DEBUG\nUSE_OPENMP\n'; fi
        printf 'OUTPUT_MESH\n'   # compares cell volumes too, not just the hydro state
        for f in $flags; do printf '%s\n' "$f"; done
    } > "$cfg"
    if timeout 2400 make -s "SYSTYPE=$SYSTYPE" "CONFIG=$cfg" \
            "BUILD_DIR=$dir" "EXEC=$dir/ProteusGPU" >"$WORK/build_$key.log" 2>&1 \
            && [ -x "$dir/ProteusGPU" ]; then
        printf '%s' "$dir/ProteusGPU"
        return 0
    fi
    return 1
}

n_pass=0; n_fail=0; n_skip=0
FAILED=()

for case_name in "${CASES[@]}"; do
    # shellcheck disable=SC1090
    . "$CASES_DIR/$case_name/case.sh"

    if [ "$CAP_NVCC" = no ]; then
        printf '\033[33mSKIP\033[0m  %-12s (needs nvcc)\n' "$case_name"
        n_skip=$((n_skip + 1))
        continue
    fi

    start=$SECONDS
    ic="$WORK/$case_name/ic.hdf5"
    mkdir -p "$WORK/$case_name"
    # shellcheck disable=SC2086
    if ! timeout 900 python3 $REPO/$CASE_IC --n "$CASE_N" --filename "$ic" \
            >"$WORK/$case_name/ic.log" 2>&1; then
        printf '\033[31mFAIL\033[0m  %-12s IC generation\n' "$case_name"
        tail -3 "$WORK/$case_name/ic.log" | sed 's/^/        /'
        FAILED+=("$case_name (IC)"); n_fail=$((n_fail + 1)); KEEP=1
        continue
    fi

    failed=""
    for backend in cpu gpu; do
        exe="$(build_for "$CASE_DIM" "$backend" "$CASE_FLAGS")"
        if [ -z "$exe" ]; then
            failed="$backend build"
            tail -4 "$WORK/build_${CASE_DIM}d_${backend}_$(echo "$CASE_FLAGS" | tr ' ' '_').log" \
                > "$WORK/$case_name/why" 2>/dev/null
            break
        fi
        out="$WORK/$case_name/$backend"
        mkdir -p "$out"
        printf 'ic_file = %s\noutput_directory = %s/\ntime_end = %s\noutput_dt = %s\nCFL_frac = 0.3\n' \
            "$ic" "$out" "$CASE_TIME_END" "$CASE_TIME_END" > "$out/param.txt"
        if ! timeout 3600 "$exe" "$out/param.txt" >"$out/run.log" 2>&1; then
            failed="$backend run"
            tail -4 "$out/run.log" > "$WORK/$case_name/why" 2>/dev/null
            break
        fi
    done

    dt=$((SECONDS - start))
    if [ -n "$failed" ]; then
        printf '\033[31mFAIL\033[0m  %-12s %4ds  %s\n' "$case_name" "$dt" "$failed"
        [ -f "$WORK/$case_name/why" ] && sed 's/^/        /' "$WORK/$case_name/why"
        FAILED+=("$case_name ($failed)"); n_fail=$((n_fail + 1)); KEEP=1
        continue
    fi

    steps=$(grep -o 'Finished after [0-9]* steps' "$WORK/$case_name/cpu/run.log" | grep -o '[0-9]*')
    if timeout 900 python3 "$CASES_DIR/compare.py" \
            "$WORK/$case_name/cpu" "$WORK/$case_name/gpu" >"$WORK/cmp.out" 2>&1; then
        printf '\033[32mPASS\033[0m  %-12s %4ds  %s steps, %s\n' \
            "$case_name" "$dt" "${steps:-?}" "$(cat "$WORK/cmp.out")"
        n_pass=$((n_pass + 1))
    else
        printf '\033[31mFAIL\033[0m  %-12s %4ds  %s steps\n' "$case_name" "$dt" "${steps:-?}"
        sed 's/^/        /' "$WORK/cmp.out"
        FAILED+=("$case_name"); n_fail=$((n_fail + 1)); KEEP=1
    fi
done

printf '\n────────────────────────────────────────\n'
printf ' passed   %d\n' "$n_pass"
printf ' failed   %d\n' "$n_fail"
if [ "$n_skip" -gt 0 ]; then
    printf ' \033[33mskipped  %d  (NOT tested — a skip is not a pass)\033[0m\n' "$n_skip"
else
    printf ' skipped  0\n'
fi
printf '────────────────────────────────────────\n'

if [ "$n_fail" -gt 0 ]; then
    printf '\nfailed:\n'
    for f in "${FAILED[@]}"; do printf '  %s\n' "$f"; done
    exit 1
fi
exit 0
