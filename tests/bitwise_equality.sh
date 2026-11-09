#!/usr/bin/env bash
# Runs every case of tests/equality/ in several ways that must give the same snapshots, bit for bit: on the CPU with
# one thread (the reference), with many threads, the same again, on the GPU with one and with many CPU threads, and
# with MPI on 1, 2, 4 and 7 ranks, on the CPU and on the GPU (the ranks share the GPU). The restarts go on from the
# snapshot halfway of the reference, on the CPU and with MPI on 3 ranks. Every build has a moving mesh and individual
# timesteps.
#
#   tests/bitwise_equality.sh               every case
#   tests/bitwise_equality.sh --cpu         without the GPU runs, for machines without a GPU
#   tests/bitwise_equality.sh --no-mpi      without the MPI runs, for machines without MPI and parallel HDF5
#   tests/bitwise_equality.sh --only sod2d  one case
#   tests/bitwise_equality.sh --keep        keeps the builds, initial conditions and snapshots, and says where
#   tests/bitwise_equality.sh --switch ENABLE_PROFILING   every build also with this Config.sh switch
#
# A case is a folder with case.sh, which sets CASE_DESC, CASE_DIM, CASE_N, CASE_TIME_END and CASE_IC, and may set
# CASE_SWITCHES, more Config.sh switches for its builds, and CASE_PARAMETERS, more lines of param.txt. compare.py
# compares every dataset and attribute of every snapshot with those of the reference, the cells matched by their id.
# Exits with 1 if a run fails or differs.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASES="$ROOT/tests/equality"
WORK="$(mktemp -d)"

cpu_only=0
with_mpi=1
only=""
keep=0
switches=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu_only=1 ;;
        --no-mpi) with_mpi=0 ;;
        --switch) switches="$switches ${2:-}"; shift ;;
        --only) only="${2:-}"; shift ;;
        --keep) keep=1 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
    shift
done
if [ $keep = 1 ]; then trap 'echo "kept in $WORK"' EXIT; else trap 'rm -rf "$WORK"' EXIT; fi

# the runs, as name:build:ranks:threads; the first one is the reference. A build is cpu or gpu, or mpi_cpu or mpi_gpu
# with USE_MPI, whose runs go through mpirun with 2 threads per rank; ranks 0 is a run without mpirun
threads=$(nproc)
[ "$threads" -gt 8 ] && threads=8
runs="cpu_t1:cpu:0:1 cpu_t$threads:cpu:0:$threads cpu_again:cpu:0:$threads restart_cpu:cpu:0:$threads"
[ $cpu_only = 0 ] && runs="$runs gpu_t1:gpu:0:1 gpu_t$threads:gpu:0:$threads"
if [ $with_mpi = 1 ]; then
    for n_ranks in 1 2 4 7; do
        runs="$runs mpi_cpu_$n_ranks:mpi_cpu:$n_ranks:2"
        [ $cpu_only = 0 ] && runs="$runs mpi_gpu_$n_ranks:mpi_gpu:$n_ranks:2"
    done
    runs="$runs restart_mpi_cpu_3:mpi_cpu:3:2"
fi

# the executable for a dimension and a build, made once into its own folder; prints its path. The snapshots hold the
# mesh, so that its volumes and faces are compared too
executable() {
    local tag
    tag="$(echo $CASE_SWITCHES | tr ' ' '_' | tr 'A-Z' 'a-z')"
    local name="${1}d_$2${tag:+_$tag}"
    if [ ! -x "$WORK/build_$name/Proteus" ]; then
        {
            echo "dim_${1}D"
            [[ "$2" == *gpu ]] && echo CUDA
            echo USE_OPENMP
            [[ "$2" == mpi_* ]] && echo USE_MPI
            echo MOVING_MESH
            echo INDIVIDUAL_TIMESTEPS
            echo OUTPUT_DEBUG
            for switch in $switches $CASE_SWITCHES; do echo "$switch"; done
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
    CASE_SWITCHES=""
    CASE_PARAMETERS=""
    . "$case_folder/case.sh"
    echo "$case_name: $CASE_DESC"

    ic="$WORK/$case_name/ic.hdf5"
    mkdir -p "$WORK/$case_name"
    if ! python3 "$ROOT"/$CASE_IC --n "$CASE_N" --filename "$ic" > "$WORK/$case_name/ic.log" 2>&1; then
        echo "  FAIL  the initial conditions, see $WORK/$case_name/ic.log"
        failed=$((failed + 1))
        trap 'echo "kept in $WORK"' EXIT
        continue
    fi

    reference=""
    for run in $runs; do
        IFS=: read -r name build n_ranks n_threads <<< "$run"
        out="$WORK/$case_name/$name"
        mkdir -p "$out"
        problem=""
        result=""

        # three snapshots: at the start, halfway and at the end
        half_time=$(awk -v t="$CASE_TIME_END" 'BEGIN { printf "%.17g", t / 2 }')
        printf 'ic_file = %s\noutput_directory = %s/\ntime_end = %s\noutput_dt = %s\nCFL_frac = 0.3\n' \
            "$ic" "$out" "$CASE_TIME_END" "$half_time" > "$out/param.txt"
        printf 'memory_limit_in_gb = 1\n' >> "$out/param.txt"
        [ -n "$CASE_PARAMETERS" ] && printf '%s\n' "$CASE_PARAMETERS" >> "$out/param.txt"

        # a restart starts in a folder with the snapshots of the reference up to the one halfway
        restart=()
        if [[ "$name" == restart_* ]]; then
            cp "$reference/snapshot_0.hdf5" "$reference/snapshot_1.hdf5" "$out/"
            restart=(1)
        fi

        launch=()
        [ "$n_ranks" -gt 0 ] && launch=(mpirun --oversubscribe -x OMP_NUM_THREADS -np "$n_ranks")
        if ! exe="$(executable "$CASE_DIM" "$build")"; then
            problem="the build, see the build logs in $WORK"
        elif ! OMP_NUM_THREADS=$n_threads ${launch[@]+"${launch[@]}"} "$exe" "$out/param.txt" ${restart[@]+"${restart[@]}"} \
            > "$out/run.log" 2>&1; then
            problem="the run, see $out/run.log"
        elif [ -z "$reference" ]; then
            reference="$out"
            result="the reference"
        elif ! result="$(python3 "$CASES/compare.py" "$reference" "$out" 2>&1)"; then
            problem="$result"
        fi

        # a failure keeps everything, to be looked at
        if [ -z "$problem" ]; then
            printf '  ok    %-10s %s\n' "$name" "$result"
            passed=$((passed + 1))
        else
            printf '  FAIL  %-10s %s\n' "$name" "$problem"
            failed=$((failed + 1))
            trap 'echo "kept in $WORK"' EXIT
            [ -z "$reference" ] && break
        fi
    done
done

echo "bitwise equality: $passed passed, $failed failed"
[ $failed = 0 ]
