#!/usr/bin/env bash
# Proteus hydro convergence tests
#
# A test is any directory under tests/convergence/ holding a case.sh and a check.py.
# case.sh says what to run, check.py says what to measure -- neither knows how the code
# was built or launched, so every case runs under every variant for free.
#
# Static vs moving mesh, CPU vs GPU and serial vs MPI are compile-time flags, so this
# builds its own binaries rather than reusing one. OUTPUT_MESH is always on: without cell
# volumes an L1 norm on a moving mesh is wrong.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/.." && pwd)"
CASES_DIR="$SELF_DIR/convergence"

usage() {
    cat <<'USAGE'
Usage: tests/run_convergence.sh [options]

Runs each convergence case at several resolutions and checks that the error falls at the
expected rate. A case that converges too slowly is a broken scheme, even when the code
runs to completion and the pictures look fine.

Options:
  --mpi          also run the moving-mesh variant under mpirun -np 4
  --cuda         also run the moving-mesh variant on the GPU (needs nvcc)
  --only NAME    run just one case (the directory name)
  --list         print the cases and variants that would run, and exit
  --keep         keep the builds, ICs and snapshots instead of deleting them
  -h, --help     show this message

Variants:
  static     no MOVING_MESH -- the mesh stays put
  moving     MOVING_MESH
  mpi        MOVING_MESH + USE_MPI, launched with mpirun -np 4   (--mpi)
  gpu        MOVING_MESH + CUDA                                  (--cuda)

To add a test, create tests/convergence/<name>/ with a case.sh and a check.py. The runner
needs no edit. case.sh sets CASE_DIM, CASE_RESOLUTIONS, CASE_TIME_END, CASE_IC and
CASE_MIN_ORDER; check.py reads the manifest it is handed and exits non-zero if the
measured order is short of CASE_MIN_ORDER.

Exit status is non-zero if any case fails. Variants needing a capability this machine
lacks are skipped and counted separately -- a skip is not a pass.
USAGE
}

WITH_MPI=0; WITH_CUDA=0; LIST=0; KEEP=0; ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --mpi)     WITH_MPI=1 ;;
        --cuda)    WITH_CUDA=1 ;;
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

# ============================================================
# Cases and capabilities
# ============================================================
CASES=()
for d in "$CASES_DIR"/*/; do
    name="$(basename "$d")"
    [ -f "$d/case.sh" ] && [ -f "$d/check.py" ] || continue
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
    CASES+=("$name")
done
[ "${#CASES[@]}" -gt 0 ] || { echo "no cases found in $CASES_DIR" >&2; exit 2; }

have() { command -v "$1" >/dev/null 2>&1; }

CAP_MPI=no; CAP_NVCC=no
if have mpicxx && have mpirun; then
    for h in /usr/include/hdf5/openmpi/hdf5.h /opt/homebrew/opt/hdf5-mpi/include/hdf5.h; do
        [ -f "$h" ] && CAP_MPI=yes && break
    done
fi
have nvcc && CAP_NVCC=yes

VARIANTS=(static moving)
[ "$WITH_MPI" -eq 1 ] && VARIANTS+=(mpi)
[ "$WITH_CUDA" -eq 1 ] && VARIANTS+=(gpu)

printf 'Hydro convergence\n'
printf '  cases      %s\n' "${CASES[*]}"
printf '  variants   %s\n' "${VARIANTS[*]}"
[ "$WITH_MPI" -eq 1 ] && printf '  mpi        %s\n' "$CAP_MPI"
[ "$WITH_CUDA" -eq 1 ] && printf '  nvcc       %s\n' "$CAP_NVCC"
printf '\n'

if [ "$LIST" -eq 1 ]; then
    for c in "${CASES[@]}"; do
        # shellcheck disable=SC1090
        ( . "$CASES_DIR/$c/case.sh"
          printf '  %-12s %-34s dim=%s  n=%s  t_end=%s  order>=%s\n' \
                 "$c" "$CASE_DESC" "$CASE_DIM" "$CASE_RESOLUTIONS" "$CASE_TIME_END" "$CASE_MIN_ORDER" )
    done
    exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/proteus-convergence.XXXXXX")"
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$WORK"; }
trap cleanup EXIT

# ============================================================
# Builds, shared between cases of the same dimension and variant
# ============================================================
write_config() {   # path dim variant
    {
        printf 'dim_%sD\n' "$2"
        # fast math is what a production GPU run uses, and LAUNCH_BOUNDS only keeps its
        # occupancy target with it on -- testing without it tests a config nobody runs
        if [ "$3" = gpu ]; then printf 'CUDA\nCUDA_FAST_MATH\n'; else printf 'CPU_DEBUG\n'; fi
        printf 'USE_OPENMP\n'
        printf 'OUTPUT_MESH\n'   # cell volumes: an L1 norm without them is wrong
        if [ "$3" != static ]; then printf 'MOVING_MESH\n'; fi
        if [ "$3" = mpi ]; then printf 'USE_MPI\n'; fi
    } > "$1"
}

build_for() {   # dim variant -> prints exec path, or nothing on failure
    local dim="$1" var="$2" key dir
    key="${dim}d_${var}"
    dir="$WORK/build_$key"
    if [ -x "$dir/ProteusGPU" ]; then printf '%s' "$dir/ProteusGPU"; return 0; fi
    write_config "$WORK/config_$key.sh" "$dim" "$var"
    if timeout 2400 make -s "SYSTYPE=$SYSTYPE" "CONFIG=$WORK/config_$key.sh" \
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
    printf '\033[1m%s\033[0m  — %s\n' "$case_name" "$CASE_DESC"

    for variant in "${VARIANTS[@]}"; do
        if { [ "$variant" = mpi ] && [ "$CAP_MPI" = no ]; } || \
           { [ "$variant" = gpu ] && [ "$CAP_NVCC" = no ]; }; then
            printf '  \033[33mSKIP\033[0m  %-8s (capability missing)\n' "$variant"
            n_skip=$((n_skip + 1))
            continue
        fi

        exe="$(build_for "$CASE_DIM" "$variant")"
        if [ -z "$exe" ]; then
            printf '  \033[31mFAIL\033[0m  %-8s build failed\n' "$variant"
            tail -4 "$WORK/build_${CASE_DIM}d_${variant}.log" | sed 's/^/          /'
            FAILED+=("$case_name/$variant (build)"); n_fail=$((n_fail + 1))
            continue
        fi

        launch=""
        [ "$variant" = mpi ] && launch="mpirun -np 4"

        start=$SECONDS
        runs=""; failed_run=""
        for n in $CASE_RESOLUTIONS; do
            out="$WORK/$case_name/$variant/n$n"
            mkdir -p "$out"
            ic="$WORK/$case_name/ic_n$n.hdf5"
            if [ ! -f "$ic" ]; then
                # shellcheck disable=SC2086
                if ! timeout 900 python3 $REPO/$CASE_IC --n "$n" --filename "$ic" \
                        >"$out/ic.log" 2>&1; then
                    failed_run="IC generation at n=$n"; break
                fi
            fi
            printf 'ic_file = %s\noutput_directory = %s/\ntime_end = %s\noutput_dt = %s\nCFL_frac = 0.3\nrebalance_interval = 10\nimbalance_log_interval = 1000\nimbalance_threshold = 1.10\n' \
                "$ic" "$out" "$CASE_TIME_END" "$CASE_TIME_END" > "$out/param.txt"
            # shellcheck disable=SC2086
            if ! timeout 3600 $launch "$exe" "$out/param.txt" >"$out/run.log" 2>&1; then
                failed_run="run at n=$n"; break
            fi
            runs="$runs{\"n\": $n, \"outdir\": \"$out\"},"
        done
        dt=$((SECONDS - start))

        if [ -n "$failed_run" ]; then
            printf '  \033[31mFAIL\033[0m  %-8s %4ds  %s\n' "$variant" "$dt" "$failed_run"
            [ -f "$out/run.log" ] && tail -3 "$out/run.log" | sed 's/^/          /'
            FAILED+=("$case_name/$variant ($failed_run)"); n_fail=$((n_fail + 1)); KEEP=1
            continue
        fi

        manifest="$WORK/$case_name/$variant/manifest.json"
        printf '{"case": "%s", "variant": "%s", "gamma": 1.6666666666666667, "min_order": %s, "runs": [%s]}\n' \
            "$case_name" "$variant" "$CASE_MIN_ORDER" "${runs%,}" > "$manifest"

        if timeout 900 python3 "$CASES_DIR/$case_name/check.py" "$manifest" >"$WORK/check.out" 2>&1; then
            printf '  \033[32mPASS\033[0m  %-8s %4ds\n' "$variant" "$dt"
            sed 's/^/    /' "$WORK/check.out"
            n_pass=$((n_pass + 1))
        else
            printf '  \033[31mFAIL\033[0m  %-8s %4ds\n' "$variant" "$dt"
            sed 's/^/    /' "$WORK/check.out"
            FAILED+=("$case_name/$variant"); n_fail=$((n_fail + 1)); KEEP=1
        fi
    done
    printf '\n'
done

printf '────────────────────────────────────────\n'
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
    printf '\nbuilds and snapshots kept in %s\n' "$WORK"
    exit 1
fi
exit 0
