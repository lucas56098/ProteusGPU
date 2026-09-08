#!/usr/bin/env bash
# Proteus IC launcher-invariance check
#
# Every IC script must produce the same file whether it is run serially or under
# mpirun, at any rank count. A slice that depends on how the work was divided is a
# silent bug: the run succeeds and the IC is wrong.
#
# Scripts are discovered, not listed.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage: tests/ic_invariance.sh [options]

Generates every IC twice -- once with plain python3, once under mpirun -- and compares
the results bit for bit. Rank counts double from 1 up to the core count.

An IC script is any git-tracked ics/create*.py that calls write_ic(), so a new IC is
covered as soon as it is committed. Untracked scripts are ignored.

Options:
  --n N            cells per dimension, passed to every script (default: 8)
  --max-ranks N    stop the rank ladder at N (default: the machine's core count)
  --list           print what would run and exit
  --keep           keep the generated ICs instead of deleting them
  -h, --help       show this message

Exit status is non-zero if any comparison fails. Checks needing a capability this machine
lacks (mpirun, MPI-enabled h5py) are skipped and counted separately -- a skip is not a pass.
USAGE
}

N_DEFAULT=8; MAX_RANKS=0; LIST=0; KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --n)          N_DEFAULT="${2:-}"; shift ;;
        --max-ranks)  MAX_RANKS="${2:-}"; shift ;;
        --list)       LIST=1 ;;
        --keep)       KEEP=1 ;;
        -h|--help)    usage; exit 0 ;;
        *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd "$REPO"
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository: $REPO" >&2; exit 2; }

# ============================================================
# Discovery: tracked ics/create*.py that call write_ic()
# ============================================================
SCRIPTS=()
while IFS= read -r f; do
    [ -n "$f" ] && SCRIPTS+=("$f")
done < <(git ls-files 'ics/create*.py' | xargs -r grep -l 'write_ic' | sort)

[ "${#SCRIPTS[@]}" -gt 0 ] || { echo "no IC scripts found under ics/" >&2; exit 2; }

# ============================================================
# Capabilities
# ============================================================
have() { command -v "$1" >/dev/null 2>&1; }

NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
CAP_MPI=no; CAP_H5MPI=no
have mpirun && CAP_MPI=yes
if [ "$CAP_MPI" = yes ]; then
    # Debian's h5py picks its build from the launcher environment, so probe under mpirun
    if timeout 120 mpirun -np 2 python3 -c \
        'import h5py, sys; sys.exit(0 if h5py.get_config().mpi else 1)' >/dev/null 2>&1; then
        CAP_H5MPI=yes
    fi
fi

CAP="$NPROC"
[ "$MAX_RANKS" -gt 0 ] && [ "$MAX_RANKS" -lt "$CAP" ] && CAP="$MAX_RANKS"
RANKS=(); r=1
while [ "$r" -le "$CAP" ]; do RANKS+=("$r"); r=$((r * 2)); done

# nproc is not the number of slots mpirun will give us: OpenMPI counts physical cores, so
# a machine with SMT reports twice what it will launch, and containers or batch allocations
# restrict it further. Probe downward for the largest rank count that actually starts. The
# ladder is monotone, so this normally costs one failed probe.
SLOT_LIMIT=""
if [ "$CAP_MPI" = yes ]; then
    for (( i = ${#RANKS[@]} - 1; i >= 0; i-- )); do
        if timeout 120 mpirun -np "${RANKS[i]}" true >/dev/null 2>&1; then
            SLOT_LIMIT="${RANKS[i]}"
            break
        fi
    done
    if [ -z "$SLOT_LIMIT" ]; then
        CAP_MPI=no
    elif [ "$SLOT_LIMIT" -lt "${RANKS[-1]}" ]; then
        TRIMMED=()
        for r in "${RANKS[@]}"; do [ "$r" -le "$SLOT_LIMIT" ] && TRIMMED+=("$r"); done
        RANKS=("${TRIMMED[@]}")
    fi
fi

printf 'IC launcher-invariance\n'
printf '  scripts    %d (tracked, calling write_ic)\n' "${#SCRIPTS[@]}"
printf '  ranks      %s   (cores: %s)\n' "${RANKS[*]}" "$NPROC"
printf '  mpirun     %s' "$CAP_MPI"
[ -n "$SLOT_LIMIT" ] && [ "$SLOT_LIMIT" -lt "$NPROC" ] && printf '   (%s launchable slots)' "$SLOT_LIMIT"
printf '\n'
printf '  h5py MPI   %s\n' "$CAP_H5MPI"
[ "$CAP_MPI" = no ] && printf '  \033[33mno mpirun: every multi-rank check will be skipped\033[0m\n'
[ "$CAP_MPI" = yes ] && [ "$CAP_H5MPI" = no ] && \
    printf '  \033[33mh5py has no MPI build: checks above 1 rank will be skipped\033[0m\n'
printf '\n'

if [ "$LIST" -eq 1 ]; then
    for s in "${SCRIPTS[@]}"; do printf '  %-28s --n %s\n' "$(basename "$s")" "$N_DEFAULT"; done
    exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/proteus-ic-invariance.XXXXXX")"
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$WORK"; }
trap cleanup EXIT

# compare the payload of two IC files, printing what differs
compare_ic() {
    python3 - "$1" "$2" <<'PY'
import sys, h5py, numpy as np
a, b = h5py.File(sys.argv[1]), h5py.File(sys.argv[2])
bad = []
for k in ("mesh/pos", "hydro/rho", "hydro/vel", "hydro/energy"):
    if k not in a or k not in b:
        bad.append(f"{k} missing"); continue
    x, y = a[k][...], b[k][...]
    if x.shape != y.shape:
        bad.append(f"{k} {x.shape} vs {y.shape}")
    elif not np.array_equal(x, y):
        n = int(np.count_nonzero(x != y))
        bad.append(f"{k} differs in {n} of {x.size} values")
print("; ".join(bad))
sys.exit(1 if bad else 0)
PY
}

n_pass=0; n_fail=0; n_skip=0
FAILED=()
idx=0

for script in "${SCRIPTS[@]}"; do
    idx=$((idx + 1))
    base="$(basename "$script")"
    args="--n $N_DEFAULT"
    dir="$WORK/$base"; mkdir -p "$dir"
    label="[$idx/${#SCRIPTS[@]}]"
    start=$SECONDS

    # reference: plain python3, no launcher in the environment
    # shellcheck disable=SC2086
    if ! (cd "$dir" && timeout 600 python3 "$REPO/$script" $args --filename "$dir/ref.hdf5") \
            >"$dir/ref.log" 2>&1; then
        printf '%s \033[31mFAIL\033[0m  %-26s  python3 run failed\n' "$label" "$base"
        sed 's/^/           /' "$dir/ref.log" | tail -3
        FAILED+=("$base (python3)"); n_fail=$((n_fail + 1)); KEEP=1
        continue
    fi

    cells="$(python3 -c \
        "import h5py,sys; print(h5py.File(sys.argv[1])['hydro/rho'].shape[0])" \
        "$dir/ref.hdf5" 2>/dev/null)"
    if [ -z "$cells" ] || [ "$cells" -lt 1 ]; then
        printf '%s \033[31mFAIL\033[0m  %-26s  reference run produced no cells\n' "$label" "$base"
        FAILED+=("$base (empty reference)"); n_fail=$((n_fail + 1)); KEEP=1
        continue
    fi

    ok=1; detail=""; done_ranks=()
    for r in "${RANKS[@]}"; do
        if [ "$CAP_MPI" = no ] || { [ "$r" -gt 1 ] && [ "$CAP_H5MPI" = no ]; }; then
            n_skip=$((n_skip + 1)); continue
        fi
        out="$dir/np$r.hdf5"
        # shellcheck disable=SC2086
        if ! (cd "$dir" && timeout 600 mpirun -np "$r" python3 "$REPO/$script" $args --filename "$out") \
                >"$dir/np$r.log" 2>&1; then
            ok=0; detail="-np $r failed to run"; n_fail=$((n_fail + 1)); break
        fi
        if ! diff_out="$(compare_ic "$dir/ref.hdf5" "$out")"; then
            ok=0; detail="-np $r: $diff_out"; n_fail=$((n_fail + 1)); break
        fi
        n_pass=$((n_pass + 1)); done_ranks+=("$r")
    done

    dt=$((SECONDS - start))
    if [ "$ok" -eq 1 ]; then
        if [ "${#done_ranks[@]}" -eq 0 ]; then
            printf '%s \033[33mSKIP\033[0m  %-26s %3ds  (no rank could be tested)\n' "$label" "$base" "$dt"
        else
            printf '%s \033[32mPASS\033[0m  %-26s %3ds  ranks %s\n' \
                "$label" "$base" "$dt" "$(IFS=,; echo "${done_ranks[*]}")"
        fi
    else
        printf '%s \033[31mFAIL\033[0m  %-26s %3ds  %s\n' "$label" "$base" "$dt" "$detail"
        [ -s "$dir/np$r.log" ] && tail -3 "$dir/np$r.log" | sed 's/^/           /'
        FAILED+=("$base ($detail)"); KEEP=1
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
    printf '\ngenerated ICs kept in %s\n' "$WORK"
    exit 1
fi
exit 0
