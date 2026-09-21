#!/usr/bin/env bash
# Proteus mesh check
#
# Runs each case for a few steps with OUTPUT_MESH and checks the mesh in every snapshot
# against its invariants and against scipy.spatial.Voronoi of the same seeds. The hydro
# suites only see a wrong mesh through the flow it produces; this one looks at the cells.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/.." && pwd)"
CASES_DIR="$SELF_DIR/mesh"

usage() {
    cat <<'USAGE'
Usage: tests/mesh_check.sh [options]

Runs each case for a few steps and checks the mesh of every snapshot:

  volume sum   the cell volumes add up to the box
  wall faces   no face lies on the box wall
  closed       every cell is closed: its area weighted face normals sum to zero
  both sides   a face between two cells of one rank is seen from both sides, same area
  scipy        every cell matches scipy.spatial.Voronoi of the same seeds in the periodic
               box: volume, centroid, and the area and normal of each face

Each case runs in every launch configuration, on the CPU and on the GPU:

  serial       built without USE_MPI
  mpi2         mpirun -np 2
  mpi4         mpirun -np 4
  mpi9         mpirun -np 9, 2D cases only, CPU only: the first rank count where the halo
               takes the neighbour collective path

Options:
  --only NAME      run just one case (the directory name)
  --no-cuda        skip the GPU builds
  --max-ranks N    leave out the launch configurations above N ranks
  --list           print what would run, and exit
  --keep           keep builds, ICs and snapshots instead of deleting them
  -h, --help       show this message

To add a case, create tests/mesh/<name>/ with a case.sh setting CASE_DESC, CASE_DIM,
CASE_FLAGS, CASE_N, CASE_TIME_END and CASE_IC. The runner needs no edit. Use seeds that are
not on a lattice: on an exact lattice the Voronoi cells are degenerate and the two codes may
split them differently.

Exit status is non-zero if any check fails. Configurations needing a capability this
machine lacks (nvcc, mpirun, parallel HDF5, scipy) are skipped and counted separately -- a
skip is never a pass.
USAGE
}

LIST=0; KEEP=0; ONLY=""; MAX_RANKS=0; NO_CUDA=0
while [ $# -gt 0 ]; do
    case "$1" in
        --only)      ONLY="${2:-}"; shift ;;
        --max-ranks) MAX_RANKS="${2:-}"; shift ;;
        --no-cuda)   NO_CUDA=1 ;;
        --list)      LIST=1 ;;
        --keep)      KEEP=1 ;;
        -h|--help)   usage; exit 0 ;;
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

# ============================================================
# Capabilities
# ============================================================
have() { command -v "$1" >/dev/null 2>&1; }

if ! python3 -c 'import scipy.spatial, h5py' >/dev/null 2>&1; then
    printf '\033[33mSKIPPED\033[0m  python3 cannot import scipy.spatial and h5py: the mesh is NOT checked\n'
    exit 0
fi

CAP_NVCC=no; have nvcc && CAP_NVCC=yes
[ "$NO_CUDA" -eq 1 ] && CAP_NVCC=no
CAP_MPI=no;  have mpirun && have mpicxx && CAP_MPI=yes
CAP_HDF5_MPI=no
for d in /usr/include/hdf5/openmpi /opt/homebrew/opt/hdf5-mpi/include "${HDF5_HOME:-}/include"; do
    [ -f "$d/hdf5.h" ] && CAP_HDF5_MPI=yes && break
done
[ "$CAP_HDF5_MPI" = no ] && CAP_MPI=no   # USE_MPI needs parallel HDF5 to read the IC

# config spec: "label ranks use_mpi scope threads"; scope all = every case and backend,
# 2dcpu = 2D cases on the CPU only
CONFIGS=("serial 1 0 all 2")
if [ "$CAP_MPI" = yes ]; then
    for spec in "mpi2 2 1 all 2" "mpi4 4 1 all 2" "mpi9 9 1 2dcpu 1"; do
        set -- $spec
        [ "$MAX_RANKS" -gt 0 ] && [ "$2" -gt "$MAX_RANKS" ] && continue
        timeout 120 mpirun -np "$2" true >/dev/null 2>&1 && CONFIGS+=("$spec")
    done
fi

BACKENDS=(cpu)
[ "$CAP_NVCC" = yes ] && BACKENDS+=(gpu)

printf '\033[1mProteus mesh check\033[0m\n'
printf '  cases      %s\n' "${CASES[*]}"
printf '  configs    %s\n' "$(for g in "${CONFIGS[@]}"; do set -- $g; printf '%s ' "$1"; done)"
printf '  backends   %s\n' "${BACKENDS[*]}"
printf '\n'
if [ "$CAP_NVCC" = no ]; then
    reason="nvcc missing"; [ "$NO_CUDA" -eq 1 ] && reason="--no-cuda"
    printf '  \033[33m%s: the GPU build is NOT checked\033[0m\n\n' "$reason"
fi
if [ "$CAP_MPI" = no ]; then
    printf '  \033[33mmpirun or parallel HDF5 missing: the MPI mesh is NOT checked\033[0m\n\n'
fi

# does this config run this case on this backend
applies() {   # scope dim backend
    [ "$1" = all ] && return 0
    [ "$1" = 2dcpu ] && [ "$2" = 2 ] && [ "$3" = cpu ]
}

if [ "$LIST" -eq 1 ]; then
    n_runs=0
    for c in "${CASES[@]}"; do
        # shellcheck disable=SC1090
        ( . "$CASES_DIR/$c/case.sh"
          printf '  %-12s %-36s dim=%s  n=%s  t_end=%s  flags="%s"\n' \
                 "$c" "$CASE_DESC" "$CASE_DIM" "$CASE_N" "$CASE_TIME_END" "$CASE_FLAGS" )
        c_dim="$( . "$CASES_DIR/$c/case.sh"; printf '%s' "$CASE_DIM" )"
        for g in "${CONFIGS[@]}"; do
            set -- $g
            for b in "${BACKENDS[@]}"; do applies "$4" "$c_dim" "$b" && n_runs=$((n_runs + 1)); done
        done
    done
    printf '\n  %d case(s): %d runs\n' "${#CASES[@]}" "$n_runs"
    exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/proteus-mesh.XXXXXX")"
cleanup() { if [ "$KEEP" -eq 1 ]; then echo "kept: $WORK"; else rm -rf "$WORK"; fi; }
trap cleanup EXIT

# ============================================================
# Builds, shared across cases
# ============================================================
build_for() {   # dim backend use_mpi flags -> prints exec path, or nothing on failure
    local dim="$1" backend="$2" use_mpi="$3" flags="$4" key dir cfg
    key="${dim}d_${backend}_mpi${use_mpi}_$(echo "$flags" | tr ' ' '_')"
    dir="$WORK/build_$key"
    [ -e "$dir/failed" ] && return 1
    if [ -x "$dir/ProteusGPU" ]; then printf '%s' "$dir/ProteusGPU"; return 0; fi
    cfg="$WORK/config_$key.sh"
    {
        printf 'dim_%sD\n' "$dim"
        if [ "$backend" = gpu ]; then printf 'CUDA\n'; else printf 'CPU_DEBUG\n'; fi
        printf 'USE_OPENMP\n'
        printf 'OUTPUT_MESH\n'
        [ "$use_mpi" -eq 1 ] && printf 'USE_MPI\n'
        for f in $flags; do printf '%s\n' "$f"; done
    } > "$cfg"
    if timeout 2400 make -s "SYSTYPE=$SYSTYPE" "CONFIG=$cfg" \
            "BUILD_DIR=$dir" "EXEC=$dir/ProteusGPU" >"$WORK/build_$key.log" 2>&1 \
            && [ -x "$dir/ProteusGPU" ]; then
        printf '%s' "$dir/ProteusGPU"
        return 0
    fi
    mkdir -p "$dir"; touch "$dir/failed"
    return 1
}

n_pass=0; n_fail=0; n_skip=0
FAILED=()

fail() {   # case label why-file message
    printf '\033[31mFAIL\033[0m  %-12s %-11s %s\n' "$1" "$2" "$4"
    [ -f "$3" ] && sed 's/^/        /' "$3"
    FAILED+=("$1/$2 ($4)"); n_fail=$((n_fail + 1)); KEEP=1
}

for case_name in "${CASES[@]}"; do
    # shellcheck disable=SC1090
    . "$CASES_DIR/$case_name/case.sh"

    mkdir -p "$WORK/$case_name"
    ic="$WORK/$case_name/ic.hdf5"
    # shellcheck disable=SC2086
    if ! timeout 900 python3 $REPO/$CASE_IC --n "$CASE_N" --filename "$ic" \
            >"$WORK/$case_name/ic.log" 2>&1; then
        tail -3 "$WORK/$case_name/ic.log" > "$WORK/$case_name/why"
        fail "$case_name" "" "$WORK/$case_name/why" "IC generation"
        continue
    fi

    for cfg_spec in "${CONFIGS[@]}"; do
        set -- $cfg_spec
        g_label="$1"; g_ranks="$2"; g_mpi="$3"; g_scope="$4"; g_threads="$5"

        for backend in "${BACKENDS[@]}"; do
            applies "$g_scope" "$CASE_DIM" "$backend" || continue
            label="$g_label/$backend"
            out="$WORK/$case_name/${g_label}_$backend"
            why="$out/why"
            mkdir -p "$out"
            start=$SECONDS

            exe="$(build_for "$CASE_DIM" "$backend" "$g_mpi" "$CASE_FLAGS")"
            if [ -z "$exe" ]; then
                key="${CASE_DIM}d_${backend}_mpi${g_mpi}_$(echo "$CASE_FLAGS" | tr ' ' '_')"
                tail -4 "$WORK/build_$key.log" > "$why" 2>/dev/null
                fail "$case_name" "$label" "$why" "build"
                continue
            fi

            # two snapshots: the first mesh, and the one after the last step
            printf 'ic_file = %s\noutput_directory = %s/\ntime_end = %s\noutput_dt = %s\nCFL_frac = 0.3\nalloc_growth = 2.0\n' \
                "$ic" "$out" "$CASE_TIME_END" "$CASE_TIME_END" > "$out/param.txt"
            printf 'rebalance_interval = 10\nimbalance_log_interval = 1000\nimbalance_threshold = 1.10\n' \
                >> "$out/param.txt"

            if [ "$g_mpi" -eq 1 ]; then
                launch=(mpirun --bind-to none -np "$g_ranks" "$exe" "$out/param.txt")
            else
                launch=("$exe" "$out/param.txt")
            fi
            if ! OMP_NUM_THREADS="$g_threads" timeout 1800 "${launch[@]}" >"$out/run.log" 2>&1; then
                tail -4 "$out/run.log" > "$why"
                fail "$case_name" "$label" "$why" "run"
                continue
            fi
            steps=$(grep -a -o 'Finished after [0-9]* steps' "$out/run.log" | grep -o '[0-9]*')

            if ! timeout 1800 python3 "$CASES_DIR/check.py" "$out" >"$out/check.log" 2>&1; then
                cp "$out/check.log" "$why"
                fail "$case_name" "$label" "$why" "mesh check"
                continue
            fi
            printf '\033[32mPASS\033[0m  %-12s %-11s %4ds  %s steps, %s snapshots\n' \
                "$case_name" "$label" "$((SECONDS - start))" "${steps:-?}" "$(grep -c '^snapshot_' "$out/check.log")"
            n_pass=$((n_pass + 1))
        done
    done
done

# axes we could not test at all, reported so a green run is never mistaken for full coverage
[ "$CAP_NVCC" = no ] && n_skip=$((n_skip + ${#CASES[@]}))
[ "$CAP_MPI"  = no ] && n_skip=$((n_skip + ${#CASES[@]}))

printf '\n────────────────────────────────────────\n'
printf ' passed   %d\n' "$n_pass"
printf ' failed   %d\n' "$n_fail"
if [ "$n_skip" -gt 0 ]; then
    printf ' \033[33mskipped  %d axis-case(s)  (NOT tested — a skip is not a pass)\033[0m\n' "$n_skip"
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
