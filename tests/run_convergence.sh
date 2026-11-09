#!/usr/bin/env bash
# Runs every case of tests/convergence/ at its resolutions, on the CPU and on the GPU, and lets its check.py measure
# the order at which the error falls with the resolution. Every build has a moving mesh and individual timesteps.
#
#   tests/run_convergence.sh               every case, on CPU and GPU
#   tests/run_convergence.sh --cpu         only on the CPU, for machines without a GPU
#   tests/run_convergence.sh --only sod    one case
#   tests/run_convergence.sh --keep        keeps the builds, initial conditions and snapshots, and says where
#   tests/run_convergence.sh --switch ENABLE_PROFILING   every build also with this Config.sh switch
#
# A case is a folder with case.sh, which sets CASE_DESC, CASE_DIM, CASE_RESOLUTIONS, CASE_TIME_END, CASE_IC and
# CASE_MIN_ORDER, and check.py, which reads the list of runs and fails if an order is below CASE_MIN_ORDER.
# Exits with 1 if a case fails.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASES="$ROOT/tests/convergence"
WORK="$(mktemp -d)"

# the adiabatic index of the builds: Config.sh without ADIABATIC_INDEX
GAMMA=1.6666666666666667

cpu_only=0
only=""
keep=0
switches=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu_only=1 ;;
        --switch) switches="$switches ${2:-}"; shift ;;
        --only) only="${2:-}"; shift ;;
        --keep) keep=1 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
    shift
done
if [ $keep = 1 ]; then trap 'echo "kept in $WORK"' EXIT; else trap 'rm -rf "$WORK"' EXIT; fi

devices="cpu gpu"
[ $cpu_only = 1 ] && devices="cpu"

# the executable for a dimension and a device, built once into its own folder; prints its path. The snapshots hold the
# mesh, as the checks weigh the error of every cell with its volume
executable() {
    local name="${1}d_$2"
    if [ ! -x "$WORK/build_$name/Proteus" ]; then
        {
            echo "dim_${1}D"
            [ "$2" = gpu ] && echo CUDA
            echo USE_OPENMP
            echo MOVING_MESH
            echo INDIVIDUAL_TIMESTEPS
            echo OUTPUT_DEBUG
            for switch in $switches; do echo "$switch"; done
        } > "$WORK/$name.sh"
        make -C "$ROOT" -s CONFIG="$WORK/$name.sh" BUILD_DIR="$WORK/build_$name" EXEC="$WORK/build_$name/Proteus" \
            < /dev/null > "$WORK/build_$name.log" 2>&1 || return 1
    fi
    echo "$WORK/build_$name/Proteus"
}

passed=0
failed=0
for case_folder in "$CASES"/*/; do
    case_name="$(basename "$case_folder")"
    [ -f "$case_folder/case.sh" ] || continue
    [ -n "$only" ] && [ "$case_name" != "$only" ] && continue
    . "$case_folder/case.sh"
    echo "$case_name: $CASE_DESC"

    for device in $devices; do
        problem=""
        runs=""
        exe="$(executable "$CASE_DIM" "$device")" || problem="the build, see $WORK/build_${CASE_DIM}d_$device.log"

        for n in $CASE_RESOLUTIONS; do
            [ -n "$problem" ] && break
            ic="$WORK/$case_name/ic_$n.hdf5"
            out="$WORK/$case_name/$device/n$n"
            mkdir -p "$out"

            # the initial conditions once per resolution, for every device
            if [ ! -f "$ic" ] && ! python3 "$ROOT"/$CASE_IC --n "$n" --filename "$ic" > "$out/ic.log" 2>&1; then
                problem="the initial conditions at n = $n, see $out/ic.log"
                break
            fi

            # one snapshot at the start, one at the end
            printf 'ic_file = %s\noutput_directory = %s/\ntime_end = %s\noutput_dt = %s\nCFL_frac = 0.3\n' \
                "$ic" "$out" "$CASE_TIME_END" "$CASE_TIME_END" > "$out/param.txt"
            printf 'memory_limit_in_gb = 4\n' >> "$out/param.txt"
            if ! "$exe" "$out/param.txt" > "$out/run.log" 2>&1; then
                problem="the run at n = $n, see $out/run.log"
                break
            fi
            runs="$runs{\"n\": $n, \"outdir\": \"$out\"},"
        done

        check_log="$WORK/$case_name/$device/check.log"
        if [ -z "$problem" ]; then
            manifest="$WORK/$case_name/$device/manifest.json"
            printf '{"case": "%s", "variant": "%s", "gamma": %s, "min_order": %s, "runs": [%s]}\n' \
                "$case_name" "$device" "$GAMMA" "$CASE_MIN_ORDER" "${runs%,}" > "$manifest"
            python3 "$case_folder/check.py" "$manifest" > "$check_log" 2>&1 || problem="the order"
        fi

        # a failure keeps everything, to be looked at
        if [ -z "$problem" ]; then
            echo "  ok    $device"
            passed=$((passed + 1))
        else
            echo "  FAIL  $device: $problem"
            failed=$((failed + 1))
            trap 'echo "kept in $WORK"' EXIT
        fi
        [ -f "$check_log" ] && cat "$check_log"
    done
done

echo "convergence: $passed passed, $failed failed"
[ $failed = 0 ]
