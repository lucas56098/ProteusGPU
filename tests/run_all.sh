#!/usr/bin/env bash
# Proteus: run every test suite.
#
# Defaults to the widest coverage each suite offers -- the full compile matrix including
# CUDA, the full IC rank ladder, convergence under MPI and GPU, and CPU/GPU equality.
# Anything the machine cannot do is skipped by the suite that owns that decision, and
# skips are reported separately: a skip is never a pass.
#
# Suites run cheapest-first so a compile break surfaces in seconds rather than after the
# convergence runs. Every suite runs even if an earlier one fails, because "what else is
# broken" is usually the more useful answer.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'USAGE'
Usage: tests/run_all.sh [options]

Runs every suite at its widest setting. Expect roughly 10-12 minutes on a workstation
with a GPU; the compile matrix and the convergence runs dominate.

Options:
  --no-cuda      skip everything needing a GPU: the matrix's cuda tier, the convergence
                 gpu variant, and the CPU/GPU equality suite entirely
  --list         print the suites and the exact commands that would run, then exit
  -h, --help     show this message

Exit status is non-zero if any suite fails.
USAGE
}

NO_CUDA=0; LIST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --no-cuda) NO_CUDA=1 ;;
        --list)    LIST=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# --cuda is the only axis --no-cuda changes, so build the arguments once rather than
# keeping two copies of the table in sync
MATRIX_ARGS="--full"
CONV_ARGS="--mpi"
if [ "$NO_CUDA" -eq 0 ]; then
    MATRIX_ARGS="$MATRIX_ARGS --cuda"
    CONV_ARGS="$CONV_ARGS --cuda"
fi

# name | script | args | needs a GPU to be worth running at all
SUITES=(
    "compile matrix|build_matrix.sh|$MATRIX_ARGS|no"
    "IC invariance|ic_invariance.sh||no"
    "CPU/GPU equality|cpu_gpu_equality.sh||yes"
    "hydro convergence|run_convergence.sh|$CONV_ARGS|no"
)

if [ "$LIST" -eq 1 ]; then
    printf '\033[1mProteus test suites\033[0m%s\n\n' "$([ "$NO_CUDA" -eq 1 ] && echo '  (--no-cuda)')"
    for entry in "${SUITES[@]}"; do
        IFS='|' read -r name script args gpu <<< "$entry"
        if [ "$NO_CUDA" -eq 1 ] && [ "$gpu" = yes ]; then
            printf '  %-20s \033[33mskipped\033[0m  (--no-cuda)\n' "$name"
        else
            printf '  %-20s tests/%s %s\n' "$name" "$script" "$args"
        fi
    done
    exit 0
fi

n_pass=0; n_fail=0; n_skip=0
FAILED=()
total_start=$SECONDS

for entry in "${SUITES[@]}"; do
    IFS='|' read -r name script args gpu <<< "$entry"

    printf '\n\033[1m════ %s ════\033[0m\n\n' "$name"

    if [ "$NO_CUDA" -eq 1 ] && [ "$gpu" = yes ]; then
        printf '\033[33mSKIPPED\033[0m  needs a GPU, and --no-cuda was given\n'
        n_skip=$((n_skip + 1))
        continue
    fi

    start=$SECONDS
    # shellcheck disable=SC2086
    "$SELF_DIR/$script" $args
    rc=$?
    dt=$((SECONDS - start))

    if [ "$rc" -eq 0 ]; then
        printf '\n\033[32m%s passed\033[0m  (%ds)\n' "$name" "$dt"
        n_pass=$((n_pass + 1))
    else
        printf '\n\033[31m%s FAILED\033[0m  (%ds, exit %d)\n' "$name" "$dt" "$rc"
        FAILED+=("$name")
        n_fail=$((n_fail + 1))
    fi
done

total=$((SECONDS - total_start))
printf '\n════════════════════════════════════════\n'
printf ' suites passed   %d\n' "$n_pass"
printf ' suites failed   %d\n' "$n_fail"
if [ "$n_skip" -gt 0 ]; then
    printf ' \033[33msuites skipped  %d  (NOT tested — a skip is not a pass)\033[0m\n' "$n_skip"
else
    printf ' suites skipped  0\n'
fi
printf ' total time      %dm%02ds\n' "$((total / 60))" "$((total % 60))"
printf '════════════════════════════════════════\n'

if [ "$n_fail" -gt 0 ]; then
    printf '\nfailed suites:\n'
    for f in "${FAILED[@]}"; do printf '  %s\n' "$f"; done
    exit 1
fi
exit 0
