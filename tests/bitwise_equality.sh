#!/usr/bin/env bash
# Proteus bitwise reproducibility
#
# One reference run per launch configuration; every variant of it must reproduce the
# reference bit for bit. The axes are the ones production actually varies: CPU vs GPU,
# rank count, OpenMP thread count, and simply running the same thing twice.
#
# Deliberately NOT CUDA_FAST_MATH.
# Scope: ASTRO_PHYSICS off (cold_mass still sums doubles under an atomic).

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/.." && pwd)"
CASES_DIR="$SELF_DIR/equality"

usage() {
    cat <<'USAGE'
Usage: tests/bitwise_equality.sh [options]

Runs each case in several configurations that MUST agree bit for bit, and compares every
snapshot dataset. Not "close": identical. A difference means a result cannot be reproduced
on different hardware, at a different rank count, or even on a rerun.

Within one launch configuration (a fixed rank count), these must all match:

  cpu/t1       CPU_DEBUG, 1 OpenMP thread        <- the reference
  cpu/tN       CPU_DEBUG, N threads              catches thread-order dependence
  cpu/rerun    CPU_DEBUG, N threads, again       catches run-to-run races
  gpu/t1       CUDA, 1 host thread               catches backend divergence
  gpu/tN       CUDA, N host threads              the MPI halo build is host-side OpenMP
  cpu/restart  CPU_DEBUG, resumed mid-run        catches state the snapshot fails to restore

and that whole set is repeated per launch configuration:

  serial       built without USE_MPI, run directly
  mpi2         built with USE_MPI, mpirun -np 2
  mpi4         built with USE_MPI, mpirun -np 4

Different rank counts are NOT compared against each other, and that is deliberate: a
different decomposition renumbers cells and reorders each cell's neighbour list, so the
sums legitimately differ. Cross-rank-count reproducibility is a separate, much stronger
property this code does not claim.

Options:
  --only NAME      run just one case (the directory name)
  --no-cuda        skip the GPU variants; the rank and thread axes still run
  --max-ranks N    cap the rank ladder (default: as many as mpirun will give us, up to 4)
  --threads N      the multi-thread count to test against 1 (default: min(8, nproc))
  --list           print what would run, and exit
  --keep           keep builds, ICs and snapshots instead of deleting them
  -h, --help       show this message

To add a case, create tests/equality/<name>/ with a case.sh setting CASE_DESC, CASE_DIM,
CASE_FLAGS, CASE_N, CASE_TIME_END and CASE_IC. The runner needs no edit. Cases sharing a
(dim, flags) pair share their builds, which is most of the runtime.

Exit status is non-zero if any comparison fails. Configurations needing a capability this
machine lacks (nvcc, mpirun, parallel HDF5) are skipped and counted separately -- a skip is
never a pass.
USAGE
}

LIST=0; KEEP=0; ONLY=""; MAX_RANKS=0; THREADS_N=0; NO_CUDA=0
while [ $# -gt 0 ]; do
    case "$1" in
        --only)      ONLY="${2:-}"; shift ;;
        --max-ranks) MAX_RANKS="${2:-}"; shift ;;
        --threads)   THREADS_N="${2:-}"; shift ;;
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

CAP_NVCC=no; have nvcc && CAP_NVCC=yes
[ "$NO_CUDA" -eq 1 ] && CAP_NVCC=no
CAP_MPI=no;  have mpirun && have mpicxx && CAP_MPI=yes
CAP_HDF5_MPI=no
for d in /usr/include/hdf5/openmpi /opt/homebrew/opt/hdf5-mpi/include "${HDF5_HOME:-}/include"; do
    [ -f "$d/hdf5.h" ] && CAP_HDF5_MPI=yes && break
done
[ "$CAP_HDF5_MPI" = no ] && CAP_MPI=no   # USE_MPI needs parallel HDF5 to read the IC

NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
if [ "$THREADS_N" -le 0 ]; then
    THREADS_N=8
    [ "$NPROC" -lt 8 ] && THREADS_N="$NPROC"
fi

# Rank ladder. nproc counts hardware threads while OpenMPI hands out cores, so ask mpirun
# what it will actually launch rather than trusting the core count.
RANK_LIST=()
if [ "$CAP_MPI" = yes ]; then
    cap=4
    [ "$MAX_RANKS" -gt 0 ] && [ "$MAX_RANKS" -lt "$cap" ] && cap="$MAX_RANKS"
    for r in 2 4; do
        [ "$r" -gt "$cap" ] && continue
        if timeout 120 mpirun -np "$r" true >/dev/null 2>&1; then RANK_LIST+=("$r"); fi
    done
fi

# group spec: "label ranks use_mpi"
CONFIGS=("serial 1 0")
for r in "${RANK_LIST[@]:-}"; do
    [ -n "$r" ] && CONFIGS+=("mpi$r $r 1")
done

# variant spec: "label backend threads"
VARIANTS=("cpu/t1 cpu 1")
[ "$THREADS_N" -gt 1 ] && VARIANTS+=("cpu/t$THREADS_N cpu $THREADS_N")
VARIANTS+=("cpu/rerun cpu $THREADS_N")
VARIANTS+=("cpu/restart cpu $THREADS_N restart")
if [ "$CAP_NVCC" = yes ]; then
    VARIANTS+=("gpu/t1 gpu 1")
    [ "$THREADS_N" -gt 1 ] && VARIANTS+=("gpu/t$THREADS_N gpu $THREADS_N")
fi

printf '\033[1mProteus bitwise reproducibility\033[0m\n'
printf '  cases      %s\n' "${CASES[*]}"
printf '  configs    %s\n' "$(for g in "${CONFIGS[@]}"; do set -- $g; printf '%s ' "$1"; done)"
printf '  variants   %s\n' "$(for v in "${VARIANTS[@]}"; do set -- $v; printf '%s ' "$1"; done)"
printf '  nvcc       %s\n' "$CAP_NVCC"
printf '  mpi        %s   (cores: %s, threads tested: 1 and %s)\n' "$CAP_MPI" "$NPROC" "$THREADS_N"
printf '  gpu build  CUDA without CUDA_FAST_MATH (strict div/sqrt, no FTZ)\n'
printf '\n'

if [ "$CAP_NVCC" = no ]; then
    reason="nvcc missing"; [ "$NO_CUDA" -eq 1 ] && reason="--no-cuda"
    printf '  \033[33m%s: the CPU/GPU axis is NOT tested\033[0m\n\n' "$reason"
fi
if [ "$CAP_MPI" = no ]; then
    printf '  \033[33mmpirun or parallel HDF5 missing: the rank axis is NOT tested\033[0m\n\n'
fi

if [ "$LIST" -eq 1 ]; then
    for c in "${CASES[@]}"; do
        # shellcheck disable=SC1090
        ( . "$CASES_DIR/$c/case.sh"
          printf '  %-12s %-32s dim=%s  n=%s  t_end=%s  flags="%s"\n' \
                 "$c" "$CASE_DESC" "$CASE_DIM" "$CASE_N" "$CASE_TIME_END" "$CASE_FLAGS" )
    done
    printf '\n  %d case(s) x %d config(s) x %d variant(s) = %d runs\n' \
        "${#CASES[@]}" "${#CONFIGS[@]}" "${#VARIANTS[@]}" \
        "$(( ${#CASES[@]} * ${#CONFIGS[@]} * ${#VARIANTS[@]} ))"
    exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/proteus-bitwise.XXXXXX")"
cleanup() { if [ "$KEEP" -eq 1 ]; then echo "kept: $WORK"; else rm -rf "$WORK"; fi; }
trap cleanup EXIT

# ============================================================
# Builds, shared across cases and variants
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
        printf 'USE_OPENMP\n'    # the halo build is host-side OpenMP even in a CUDA build
        printf 'OUTPUT_MESH\n'      # compares cell volumes and face data, not just hydro state
        printf 'ENABLE_PROFILING\n'
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

for case_name in "${CASES[@]}"; do
    # shellcheck disable=SC1090
    . "$CASES_DIR/$case_name/case.sh"

    mkdir -p "$WORK/$case_name"
    ic="$WORK/$case_name/ic.hdf5"
    # shellcheck disable=SC2086
    if ! timeout 900 python3 $REPO/$CASE_IC --n "$CASE_N" --filename "$ic" \
            >"$WORK/$case_name/ic.log" 2>&1; then
        printf '\033[31mFAIL\033[0m  %-12s %-8s IC generation\n' "$case_name" ""
        tail -3 "$WORK/$case_name/ic.log" | sed 's/^/        /'
        FAILED+=("$case_name (IC)"); n_fail=$((n_fail + 1)); KEEP=1
        continue
    fi

    for cfg_spec in "${CONFIGS[@]}"; do
        set -- $cfg_spec
        g_label="$1"; g_ranks="$2"; g_mpi="$3"

        start=$SECONDS
        ref_dir=""; ref_label=""; ref_backend=""; problem=""; compared=0; steps=""

        for variant in "${VARIANTS[@]}"; do
            set -- $variant
            v_label="$1"; v_backend="$2"; v_threads="$3"; v_mode="${4:-run}"

            exe="$(build_for "$CASE_DIM" "$v_backend" "$g_mpi" "$CASE_FLAGS")"
            if [ -z "$exe" ]; then
                problem="$v_label build"
                key="${CASE_DIM}d_${v_backend}_mpi${g_mpi}_$(echo "$CASE_FLAGS" | tr ' ' '_')"
                tail -4 "$WORK/build_$key.log" > "$WORK/$case_name/why" 2>/dev/null
                break
            fi

            out="$WORK/$case_name/$g_label/$(echo "$v_label" | tr '/' '_')"
            mkdir -p "$out"
            
            half_dt=$(awk -v t="$CASE_TIME_END" 'BEGIN{printf "%.17g", t/2}')
            printf 'ic_file = %s\noutput_directory = %s/\ntime_end = %s\noutput_dt = %s\nCFL_frac = 0.3\n' \
                "$ic" "$out" "$CASE_TIME_END" "$half_dt" > "$out/param.txt"
            printf 'rebalance_interval = 10\nimbalance_log_interval = 1000\nimbalance_threshold = 1.10\n' \
                >> "$out/param.txt"

            restart_flag=()
            if [ "$v_mode" = restart ]; then
                last=$(ls "$ref_dir"/snapshot_*.hdf5 2>/dev/null \
                       | sed 's/.*snapshot_\([0-9]*\).*/\1/' | sort -n | tail -1)
                if [ -z "$last" ] || [ "$last" -lt 1 ]; then
                    problem="$v_label needs >=2 reference snapshots"; break
                fi
                
                copy_fail=""
                for k in $(seq 0 $((last - 1))); do
                    if [ "$g_mpi" -eq 1 ]; then
                        for r in $(seq 0 $((g_ranks - 1))); do
                            cp "$ref_dir/snapshot_$k.$r.hdf5" "$out/" || copy_fail="snapshot_$k.$r"
                        done
                    else
                        cp "$ref_dir/snapshot_$k.hdf5" "$out/" || copy_fail="snapshot_$k"
                    fi
                done
                if [ -n "$copy_fail" ]; then
                    problem="$v_label could not seed $copy_fail"; break
                fi
                
                [ -f "$ref_dir/profile.hdf5" ] && cp "$ref_dir/profile.hdf5" "$out/"
                restart_flag=(1)
            fi

            # --bind-to none so OpenMP threads actually spread; the default binds each rank
            # to one core and the thread axis silently stops testing anything
            if [ "$g_mpi" -eq 1 ]; then
                launch=(mpirun --bind-to none -np "$g_ranks" "$exe" "$out/param.txt"
                        ${restart_flag[@]+"${restart_flag[@]}"})
            else
                launch=("$exe" "$out/param.txt" ${restart_flag[@]+"${restart_flag[@]}"})
            fi
            if ! OMP_NUM_THREADS="$v_threads" timeout 3600 "${launch[@]}" >"$out/run.log" 2>&1; then
                problem="$v_label run"
                tail -4 "$out/run.log" > "$WORK/$case_name/why" 2>/dev/null
                break
            fi

            if [ -z "$ref_dir" ]; then
                ref_dir="$out"; ref_label="$v_label"; ref_backend="$v_backend"
                steps=$(grep -o 'Finished after [0-9]* steps' "$out/run.log" | grep -o '[0-9]*')
                continue
            fi

            cmp_opts=()
            [ "$v_backend" = "$ref_backend" ] && cmp_opts=(--profile)
            if ! timeout 900 python3 "$CASES_DIR/compare.py" "$ref_dir" "$out" \
                    ${cmp_opts[@]+"${cmp_opts[@]}"} >"$WORK/cmp.out" 2>&1; then
                problem="$v_label != $ref_label"
                cp "$WORK/cmp.out" "$WORK/$case_name/why"
                break
            fi
            compared=$((compared + 1))
        done

        dt=$((SECONDS - start))
        if [ -n "$problem" ]; then
            printf '\033[31mFAIL\033[0m  %-12s %-7s %4ds  %s\n' "$case_name" "$g_label" "$dt" "$problem"
            [ -f "$WORK/$case_name/why" ] && sed 's/^/        /' "$WORK/$case_name/why"
            FAILED+=("$case_name/$g_label ($problem)"); n_fail=$((n_fail + 1)); KEEP=1
        else
            printf '\033[32mPASS\033[0m  %-12s %-7s %4ds  %s steps, %d variant(s) match %s\n' \
                "$case_name" "$g_label" "$dt" "${steps:-?}" "$compared" "$ref_label"
            n_pass=$((n_pass + 1))
        fi
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
