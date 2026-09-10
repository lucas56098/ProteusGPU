#!/usr/bin/env bash
# Proteus head-to-head performance comparison

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage: tests/benchmark.sh [options]

Benchmarks two versions of Proteus and reports timings from the profiler.

Options:
  --ref SPEC       baseline: any commit-ish, or WORKTREE (default: HEAD)
  --new SPEC       comparison: any commit-ish, or WORKTREE (default: WORKTREE)
  --n N            cells per side
  --time-end T     simulated end time (default: 0.02)
  --reps R         pairs per order; 2*R runs per build per mode (default: 2)
  --quick          one run per side, forward order only
  --only MODE      restrict to one mode, repeatable: cpu | mpi | gpu | gpumpi
  --ranks N        ranks for mpi (default 4) and gpumpi (default 2)
  --top N          scopes to show per mode, largest absolute change first (default: 12)
  --keep           keep the builds, IC and profiles
  --list           print the plan and exit without building or running
  -h, --help       show this message

Modes:
  cpu      CPU_DEBUG + OpenMP
  mpi      CPU_DEBUG + OpenMP + USE_MPI, under mpirun
  gpu      CUDA + CUDA_FAST_MATH
  gpumpi   CUDA + CUDA_FAST_MATH + USE_MPI, under mpirun

USAGE
}

main() {
    REF="HEAD"; NEW="WORKTREE"; N=100; TEND=0.02; REPS=2; TOP=12
    RANKS_CPU=4; RANKS_GPU=2; KEEP=0; LIST=0; QUICK=0; PASSES=2; ONLY=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --ref)      REF="${2:?}"; shift ;;
            --new)      NEW="${2:?}"; shift ;;
            --n)        N="${2:?}"; shift ;;
            --time-end) TEND="${2:?}"; shift ;;
            --reps)     REPS="${2:?}"; shift ;;
            --only)     ONLY+=("${2:?}"); shift ;;
            --ranks)    RANKS_CPU="${2:?}"; RANKS_GPU="${2:?}"; shift ;;
            --top)      TOP="${2:?}"; shift ;;
            --quick)    QUICK=1; REPS=1; PASSES=1 ;;
            --keep)     KEEP=1 ;;
            --list)     LIST=1 ;;
            -h|--help)  usage; exit 0 ;;
            *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
        esac
        shift
    done
    [ "$REPS" -ge 1 ] || { echo "--reps must be at least 1" >&2; exit 2; }

    cd "$REPO"
    case "$(uname -s)" in
        Linux)  SYSTYPE=Ubuntu ;;
        Darwin) SYSTYPE=macOS ;;
        *) echo "cannot guess SYSTYPE for $(uname -s)" >&2; exit 2 ;;
    esac

    have() { command -v "$1" >/dev/null 2>&1; }
    CAP_NVCC=no; have nvcc && CAP_NVCC=yes
    CAP_MPI=no;  have mpirun && have mpicxx && CAP_MPI=yes
    for d in /usr/include/hdf5/openmpi /opt/homebrew/opt/hdf5-mpi/include "${HDF5_HOME:-}/include"; do
        [ -f "$d/hdf5.h" ] || continue
        break
    done

    # mode spec: "label flags ranks"; ranks 1 means no mpirun
    ALL_MODES=(
        "cpu|CPU_DEBUG USE_OPENMP|1|"
        "mpi|CPU_DEBUG USE_OPENMP USE_MPI|$RANKS_CPU|mpi"
        "gpu|CUDA CUDA_FAST_MATH USE_OPENMP|1|nvcc"
        "gpumpi|CUDA CUDA_FAST_MATH USE_OPENMP USE_MPI|$RANKS_GPU|nvcc mpi"
    )

    MODES=(); SKIPPED=()
    for spec in "${ALL_MODES[@]}"; do
        IFS='|' read -r label flags ranks needs <<< "$spec"
        if [ "${#ONLY[@]}" -gt 0 ]; then
            keep=0
            for o in "${ONLY[@]}"; do [ "$o" = "$label" ] && keep=1; done
            [ "$keep" -eq 1 ] || continue
        fi
        miss=""
        for cap in $needs; do
            [ "$cap" = nvcc ] && [ "$CAP_NVCC" = no ] && miss="nvcc"
            [ "$cap" = mpi  ] && [ "$CAP_MPI"  = no ] && miss="${miss:+$miss + }mpirun"
        done
        if [ -n "$miss" ]; then SKIPPED+=("$label (needs $miss)"); continue; fi
        MODES+=("$spec")
    done
    [ "${#MODES[@]}" -gt 0 ] || { echo "no runnable modes selected" >&2; exit 2; }

    for spec in "$REF" "$NEW"; do
        [ "$spec" = WORKTREE ] && continue
        git rev-parse --verify --quiet "${spec}^{commit}" >/dev/null \
            || { echo "not a commit: $spec" >&2; exit 2; }
    done
    describe() {
        if [ "$1" = WORKTREE ]; then
            local d; d="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
            printf 'WORKTREE (uncommitted, on %s)' "$d"
        else
            printf '%s (%s)' "$(git rev-parse --short "$1")" "$1"
        fi
    }

    CELLS=$((N * N * N))
    RUNS_PER_MODE=$((2 * PASSES * REPS))
    GPU_EST=$(( CELLS * 19 / 10000 ))   # ~1.9 KB/cell measured on this code at 100^3

    printf '\033[1mProteus performance comparison\033[0m\n'
    printf '  ref        %s\n' "$(describe "$REF")"
    printf '  new        %s\n' "$(describe "$NEW")"
    printf '  problem    3D acoustic wave, n=%s (%s cells), time_end=%s\n' "$N" "$CELLS" "$TEND"
    printf '  modes      %s\n' "$(for m in "${MODES[@]}"; do printf '%s ' "${m%%|*}"; done)"
    if [ "$QUICK" -eq 1 ]; then
        printf '  repeats    \033[33m--quick: 1 run per side, one order -> %s runs per mode\033[0m\n' "$RUNS_PER_MODE"
        printf '             \033[33msanity check only; nothing under 25%% is reported\033[0m\n'
    else
        printf '  repeats    %s pairs per order -> %s runs per mode\n' "$REPS" "$RUNS_PER_MODE"
    fi
    for s in "${SKIPPED[@]:-}"; do [ -n "$s" ] && printf '  \033[33mskipped    %s\033[0m\n' "$s"; done
    if [ "$CAP_NVCC" = yes ]; then
        gpu_total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)"
        if [ "${gpu_total:-0}" -gt 0 ] && [ "$GPU_EST" -gt $((gpu_total * 85 / 100)) ]; then
            printf '  \033[33mwarning    GPU modes need ~%s MiB of %s MiB; lower --n if they thrash\033[0m\n' \
                "$GPU_EST" "$gpu_total"
        fi
    fi
    
    NPROC="$(nproc 2>/dev/null || echo 1)"
    LOAD1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
    if [ "$(printf '%.0f' "${LOAD1:-0}" 2>/dev/null || echo 0)" -gt $((NPROC / 4)) ]; then
        printf '  \033[33mwarning    load average is %s on %s cores -- results will be noisy,\033[0m\n' \
            "$LOAD1" "$NPROC"
        printf '  \033[33m           wait for the machine to go idle before trusting anything\033[0m\n'
    fi
    printf '\n'

    if [ "$LIST" -eq 1 ]; then
        for spec in "${MODES[@]}"; do
            IFS='|' read -r label flags ranks needs <<< "$spec"
            launch="direct"; [ "$ranks" -gt 1 ] && launch="mpirun -np $ranks"
            printf '  %-8s %-46s %s\n' "$label" "dim_3D MOVING_MESH ENABLE_PROFILING $flags" "$launch"
        done
        printf '\n  %d mode(s) x %d runs = %d runs, plus %d builds\n' \
            "${#MODES[@]}" "$RUNS_PER_MODE" "$(( ${#MODES[@]} * RUNS_PER_MODE ))" "$(( ${#MODES[@]} * 2 ))"
        exit 0
    fi

    WORK="$(mktemp -d "${TMPDIR:-/tmp}/proteus-bench.XXXXXX")"
    cleanup() { if [ "$KEEP" -eq 1 ]; then echo "kept: $WORK"; else rm -rf "$WORK"; fi; }
    trap cleanup EXIT

    # ============================================================
    # Trees: a commit is exported read-only, WORKTREE is used in place
    # ============================================================
    resolve_tree() {   # spec label -> prints a source dir
        local spec="$1" label="$2" dir
        if [ "$spec" = WORKTREE ]; then printf '%s' "$REPO"; return 0; fi
        dir="$WORK/tree_$label"
        mkdir -p "$dir"
        git archive "$spec" | tar -x -C "$dir" || return 1
        printf '%s' "$dir"
    }

    TREE_REF="$(resolve_tree "$REF" ref)" || { echo "could not export $REF" >&2; exit 1; }
    TREE_NEW="$(resolve_tree "$NEW" new)" || { echo "could not export $NEW" >&2; exit 1; }

    build_one() {   # tree label mode flags -> prints exec path
        local tree="$1" label="$2" mode="$3" flags="$4" dir cfg
        dir="$WORK/build_${label}_${mode}"
        cfg="$WORK/config_${mode}.sh"
        {
            printf 'dim_3D\nMOVING_MESH\nENABLE_PROFILING\n'
            for f in $flags; do printf '%s\n' "$f"; done
        } > "$cfg"
        if ( cd "$tree" && timeout 2400 make -s "SYSTYPE=$SYSTYPE" "CONFIG=$cfg" \
                "BUILD_DIR=$dir" "EXEC=$dir/ProteusGPU" ) >"$WORK/build_${label}_${mode}.log" 2>&1 \
                && [ -x "$dir/ProteusGPU" ]; then
            printf '%s' "$dir/ProteusGPU"
            return 0
        fi
        return 1
    }

    # ============================================================
    # One shared IC, so neither build is measured against different input
    # ============================================================
    IC="$WORK/ic.hdf5"
    printf 'generating IC (%s cells) ... ' "$CELLS"
    if ! timeout 3600 python3 "$REPO/ics/create_acoustic_wave.py" --n "$N" --dimension 3 \
            --filename "$IC" >"$WORK/ic.log" 2>&1; then
        printf '\033[31mfailed\033[0m\n'; tail -5 "$WORK/ic.log" | sed 's/^/  /'; exit 1
    fi
    printf 'ok\n\n'

    run_one() {
        local exe="$1" ranks="$2" tag="$3"
        local out="$WORK/run_$tag"
        local threads=$((NPROC / ranks))
        [ "$threads" -lt 1 ] && threads=1
        rm -rf "$out"; mkdir -p "$out"
        {
            printf 'ic_file = %s\noutput_directory = %s/\ntime_end = %s\noutput_dt = 1e30\n' "$IC" "$out" "$TEND"
            printf 'CFL_frac = 0.3\nrebalance_interval = 10\nimbalance_log_interval = 1000000\n'
            printf 'imbalance_threshold = 1.10\n'
        } > "$out/param.txt"
        local -a launch
        if [ "$ranks" -gt 1 ]; then
            launch=(mpirun --bind-to none -np "$ranks" "$exe" "$out/param.txt")
        else
            launch=("$exe" "$out/param.txt")
        fi
        OMP_NUM_THREADS="$threads" timeout 7200 "${launch[@]}" >"$out/run.log" 2>&1 || return 1
        [ -f "$out/profile.hdf5" ] || return 1
        printf '%s' "$out"
    }

    n_ok=0; n_bad=0
    MODES_LEFT=${#MODES[@]}
    FAILED=()

    for spec in "${MODES[@]}"; do
        IFS='|' read -r mode flags ranks needs <<< "$spec"

        exe_ref="$(build_one "$TREE_REF" ref "$mode" "$flags")"
        exe_new="$(build_one "$TREE_NEW" new "$mode" "$flags")"
        if [ -z "$exe_ref" ] || [ -z "$exe_new" ]; then
            side=new; [ -z "$exe_ref" ] && side=ref
            printf '\033[31mFAIL\033[0m  %-8s %s build\n' "$mode" "$side"
            tail -4 "$WORK/build_${side}_${mode}.log" 2>/dev/null | sed 's/^/        /'
            FAILED+=("$mode ($side build)"); n_bad=$((n_bad + 1)); KEEP=1
            continue
        fi

        # interleaved, and the second half runs each pair in the opposite order
        ref_dirs=(); new_dirs=(); broke=""; done_runs=0
        printf '  %-8s %s threads x %s rank(s), %s runs\n' \
            "$mode" "$((NPROC / ranks))" "$ranks" "$RUNS_PER_MODE"
        for ((r = 1; r <= PASSES * REPS; r++)); do
            if [ "$r" -le "$REPS" ]; then order=(ref new); else order=(new ref); fi
            for side in "${order[@]}"; do
                eval "exe=\$exe_$side"
                t0=$SECONDS
                d="$(run_one "$exe" "$ranks" "${mode}_${side}_$r")" || { broke="$side run $r"; break; }
                dt=$((SECONDS - t0))
                done_runs=$((done_runs + 1))
                if [ "$side" = ref ]; then ref_dirs+=("$d"); else new_dirs+=("$d"); fi
                
                if [ "$done_runs" -eq 1 ]; then
                    printf '  %-8s   run %s/%s %-4s %4ds   (this mode ~%dm, all remaining ~%dm)\n' \
                        "$mode" "$done_runs" "$RUNS_PER_MODE" "$side" "$dt" \
                        "$((dt * RUNS_PER_MODE / 60))" "$((dt * RUNS_PER_MODE * MODES_LEFT / 60))"
                else
                    printf '  %-8s   run %s/%s %-4s %4ds\n' "$mode" "$done_runs" "$RUNS_PER_MODE" "$side" "$dt"
                fi
            done
            [ -n "$broke" ] && break
        done
        MODES_LEFT=$((MODES_LEFT - 1))
        if [ -n "$broke" ]; then
            printf '\033[31mFAIL\033[0m  %-8s %s\n' "$mode" "$broke"
            tail -4 "$WORK/run_${mode}_"*"/run.log" 2>/dev/null | tail -4 | sed 's/^/        /'
            FAILED+=("$mode ($broke)"); n_bad=$((n_bad + 1)); KEEP=1
            continue
        fi

        steps=$(grep -oE 'Finished after [0-9]+ steps' "${ref_dirs[0]}/run.log" | grep -oE '[0-9]+')
        launch="direct"; [ "$ranks" -gt 1 ] && launch="-np $ranks"
        printf '\033[1m%s\033[0m  (%s, %s steps, %s runs each)\n' "$mode" "$launch" "${steps:-?}" "${#ref_dirs[@]}"
        cmp_args=(--top "$TOP")
        [ "$QUICK" -eq 1 ] && cmp_args+=(--min-pct 25)
        if timeout 600 python3 "$SELF_DIR/bench/compare_profiles.py" "${cmp_args[@]}" \
                "${ref_dirs[@]}" -- "${new_dirs[@]}"; then
            n_ok=$((n_ok + 1))
        else
            FAILED+=("$mode (comparison)"); n_bad=$((n_bad + 1)); KEEP=1
        fi
        printf '\n'
    done

    printf '────────────────────────────────────────\n'
    printf ' modes compared  %d\n' "$n_ok"
    printf ' modes failed    %d\n' "$n_bad"
    if [ "${#SKIPPED[@]}" -gt 0 ] && [ -n "${SKIPPED[0]:-}" ]; then
        printf ' \033[33mmodes skipped   %d\033[0m\n' "${#SKIPPED[@]}"
    fi
    printf '────────────────────────────────────────\n'
    printf ' ref  %s\n new  %s\n' "$(describe "$REF")" "$(describe "$NEW")"

    if [ "$n_bad" -gt 0 ]; then
        printf '\nfailed:\n'
        for f in "${FAILED[@]}"; do printf '  %s\n' "$f"; done
        exit 1
    fi
    exit 0

}

main "$@"