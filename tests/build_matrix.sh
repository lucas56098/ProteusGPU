# Proteus compile matrix
# The configurations themselves are set in tests/configs.txt, one per line
#
# Compile-only. No running is done.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/.." && pwd)"
CONFIGS="$SELF_DIR/configs.txt"

usage() {
    cat <<'USAGE'
Usage: tests/build_matrix.sh [options]

Builds Proteus in many Config.sh flag combinations to catch configuration-specific
breaks. Configurations are listed in tests/configs.txt -- to add one, copy a line
there and edit it.

Options:
  --full           also build the 'full' tier (~47 configs instead of ~16)
  --cuda           also build the 'cuda' tier (requires nvcc)
  --list           print the selected configurations and exit, without building
  --keep           keep the temporary build tree instead of deleting it
  --no-audit       skip the flag-coverage check
  -j, --jobs N     make parallelism per configuration (default: the Makefile's own -j)
  --systype NAME   override SYSTYPE (default: Ubuntu on Linux, macOS on Darwin)
  -h, --help       show this message

Tiers:
  basic      always built
  full       --full
  cuda       --cuda
  mustfail   always built; a Makefile guard must REJECT these, so a successful
             build is the failure

Warnings are errors by default. Use --no-werror when a
new compiler introduces a diagnostic you have not addressed yet.

Before building anything, a coverage audit greps src/ for every #ifdef flag and fails
if one is never compiled by any configuration in tests/configs.txt.

Exit status is non-zero if any runnable configuration fails. Configurations needing a
capability this machine lacks (nvcc, mpicxx, parallel HDF5) are skipped.
USAGE
}

FULL=0; WITH_CUDA=0; LIST=0; KEEP=0; AUDIT=1; WERROR=1; SYSTYPE=""; JOBS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --full)      FULL=1 ;;
        --cuda)      WITH_CUDA=1 ;;
        --list)      LIST=1 ;;
        --keep)      KEEP=1 ;;
        --no-audit)  AUDIT=0 ;;
        --no-werror) WERROR=0 ;;
        --systype)   SYSTYPE="${2:-}"; shift ;;
        -j|--jobs)   JOBS="${2:-}"; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd "$REPO"
[ -f "$CONFIGS" ] || { echo "missing configuration list: $CONFIGS" >&2; exit 2; }

if [ -z "$SYSTYPE" ]; then
    case "$(uname -s)" in
        Linux)  SYSTYPE=Ubuntu ;;
        Darwin) SYSTYPE=macOS ;;
        *) echo "cannot guess SYSTYPE for $(uname -s); pass --systype" >&2; exit 2 ;;
    esac
fi

# ============================================================
# Capability detection
# ============================================================
have() { command -v "$1" >/dev/null 2>&1; }

CAP_CXX=no; CAP_HDF5=no; CAP_HDF5_MPI=no; CAP_MPI=no; CAP_NVCC=no; CAP_GPU=no
CXX_VER=""; GPU_NAME=""

if have g++ || have g++-15; then CAP_CXX=yes; CXX_VER="$(g++ --version 2>/dev/null | head -1)"; fi
for d in /usr/include/hdf5/serial /opt/homebrew/opt/hdf5/include "${HDF5_HOME:-}/include" "${EBROOTHDF5:-}/include"; do
    [ -f "$d/hdf5.h" ] && CAP_HDF5=yes && break
done
for d in /usr/include/hdf5/openmpi /opt/homebrew/opt/hdf5-mpi/include; do
    [ -f "$d/hdf5.h" ] && CAP_HDF5_MPI=yes && break
done
have mpicxx && CAP_MPI=yes
have nvcc   && CAP_NVCC=yes
if have nvidia-smi; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    [ -n "$GPU_NAME" ] && CAP_GPU=yes
fi

# ============================================================
# Load tests/configs.txt
# ============================================================
NAMES=(); FLAGS=(); TIERS=()
while read -r tier name rest; do
    case "$tier" in
        basic|mustfail) ;;
        full) [ "$FULL" -eq 1 ] || continue ;;
        cuda) [ "$WITH_CUDA" -eq 1 ] || continue ;;
        *) printf 'configs.txt: unknown tier "%s" for %s\n' "$tier" "$name" >&2; exit 2 ;;
    esac
    TIERS+=("$tier"); NAMES+=("$name"); FLAGS+=("$rest")
done < <(grep -vE '^[[:space:]]*(#|$)' "$CONFIGS")

[ "${#NAMES[@]}" -gt 0 ] || { echo "no configurations selected" >&2; exit 2; }

# Requirements are derived from the flags
needs_of() {
    local f mpi=0 cuda=0
    for f in $1; do
        case "${f%%=*}" in
            USE_MPI) mpi=1 ;;
            CUDA)    cuda=1 ;;
        esac
    done
    if [ "$cuda" -eq 1 ] && [ "$mpi" -eq 1 ]; then echo "cuda+mpi"
    elif [ "$cuda" -eq 1 ]; then echo cuda
    elif [ "$mpi" -eq 1 ]; then echo mpi
    fi
}

# ============================================================
# Report
# ============================================================
printf '\n\033[1mProteus compile matrix\033[0m  (systype=%s, tiers: basic%s%s + mustfail)\n' \
    "$SYSTYPE" "$([ "$FULL" -eq 1 ] && echo ' + full')" "$([ "$WITH_CUDA" -eq 1 ] && echo ' + cuda')"
printf '\ncapabilities:\n'
printf '  %-16s %s\n' "host compiler" "$([ "$CAP_CXX" = yes ] && echo "yes  ${CXX_VER}" || echo NO)"
printf '  %-16s %s\n' "serial HDF5"   "$CAP_HDF5"
printf '  %-16s %s\n' "parallel HDF5" "$CAP_HDF5_MPI"
printf '  %-16s %s\n' "mpicxx"        "$CAP_MPI"
printf '  %-16s %s\n' "nvcc"          "$CAP_NVCC"
printf '  %-16s %s\n' "GPU"           "$([ "$CAP_GPU" = yes ] && echo "yes  $GPU_NAME" || echo 'no (not needed to compile)')"

if [ "$LIST" -eq 1 ]; then
    printf '\n%d configurations selected:\n' "${#NAMES[@]}"
    for i in "${!NAMES[@]}"; do
        printf '  %-9s %-30s %s\n' "${TIERS[$i]}" "${NAMES[$i]}" "${FLAGS[$i]}"
    done
    exit 0
fi

# ============================================================
# Coverage audit (ensures all existing configs have to be tested)
# ============================================================
AUDIT_IGNORE="__CUDA_ARCH__ __CUDACC_VER_MAJOR__ __cplusplus _OPENMP M_PI NDEBUG DRY_RUN GIT_COMMIT GIT_DIFFSTAT
__APPLE__ __MACH__ __linux__ __unix__ __has_include MPIX_CUDA_AWARE_SUPPORT
PROTEUS_HAS_MPIX_QUERY_CUDA RUN_MODE DIMENSION"

run_audit() {
    declare -A USED_IN DERIVED_OF COVERED
    local tier name rest f kind flag file srcs ok s holes=0
    local ignore=" $(printf '%s ' $AUDIT_IGNORE)"

    while read -r tier name rest; do
        for f in $rest; do COVERED["${f%%=*}"]=1; done
    done < <(grep -vE '^[[:space:]]*(#|$)' "$CONFIGS")

    while read -r kind flag file srcs; do
        case "$kind" in
            USE)     [ -z "${USED_IN[$flag]:-}" ] && USED_IN["$flag"]="$file" ;;
            DERIVED) DERIVED_OF["$flag"]="$file $srcs" ;;
        esac
    done < <(find src -name '*.cu' -o -name '*.h' | sort | xargs awk '
        FNR == 1 { prev = "" }
        {
            line = $0
            s = line
            while (match(s, /#[ \t]*(ifdef|ifndef)[ \t]+[A-Za-z_][A-Za-z0-9_]*/)) {
                t = substr(s, RSTART, RLENGTH); sub(/.*[ \t]/, "", t)
                print "USE " t " " FILENAME
                s = substr(s, RSTART + RLENGTH)
            }
            s = line
            while (match(s, /defined[ \t]*\([ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*\)/)) {
                t = substr(s, RSTART, RLENGTH); gsub(/[^A-Za-z0-9_]/, "", t); sub(/^defined/, "", t)
                print "USE " t " " FILENAME
                s = substr(s, RSTART + RLENGTH)
            }
            if (line ~ /^[ \t]*#[ \t]*define[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]*$/ &&
                prev ~ /^[ \t]*#[ \t]*(if|elif)/ && prev ~ /defined/) {
                name = line; sub(/^[ \t]*#[ \t]*define[ \t]+/, "", name); gsub(/[ \t]/, "", name)
                out = ""; s = prev
                while (match(s, /defined[ \t]*\([ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*\)/)) {
                    t = substr(s, RSTART, RLENGTH); gsub(/[^A-Za-z0-9_]/, "", t); sub(/^defined/, "", t)
                    out = out " " t
                    s = substr(s, RSTART + RLENGTH)
                }
                print "DERIVED " name " " FILENAME out
            }
            if (line !~ /^[ \t]*$/) prev = line
        }')

    for flag in "${!USED_IN[@]}"; do
        case "$ignore" in *" $flag "*) continue ;; esac
        case "$flag" in *_H) continue ;; esac
        [ -n "${COVERED[$flag]:-}" ] && continue
        if [ -n "${DERIVED_OF[$flag]:-}" ]; then
            ok=0
            for s in ${DERIVED_OF[$flag]#* }; do [ -n "${COVERED[$s]:-}" ] && ok=1; done
            [ "$ok" -eq 1 ] && continue
        fi
        if [ "$holes" -eq 0 ]; then
            printf '\n\033[31mcoverage audit failed\033[0m — these flags are used in src/ but no\n'
            printf 'configuration in tests/configs.txt ever compiles them:\n\n'
        fi
        printf '  \033[31m%-28s\033[0m %s\n' "$flag" "${USED_IN[$flag]}"
        holes=$((holes + 1))
    done

    if [ "$holes" -gt 0 ]; then
        printf '\nCode behind an uncompiled flag is never checked, so the matrix would report\n'
        printf 'success while src/ does not build. Add a line to tests/configs.txt covering\n'
        printf 'each flag above, or --no-audit to bypass.\n\n'
        return 1
    fi
    printf '\ncoverage audit: \033[32mok\033[0m (every #ifdef flag in src/ is built by some configuration)\n'
    return 0
}

if [ "$AUDIT" -eq 1 ]; then
    run_audit || exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/proteus-matrix.XXXXXX")"
cleanup() { if [ "$KEEP" -eq 1 ]; then echo "build tree kept: $WORK"; else rm -rf "$WORK"; fi; }
trap cleanup EXIT

# ============================================================
# Build
# ============================================================
n_pass=0; n_fail=0; n_skip=0
FAILED=()
printf '\n'

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; flags="${FLAGS[$i]}"; tier="${TIERS[$i]}"
    need="$(needs_of "$flags")"
    idx="$(printf '%2d/%d' $((i + 1)) "${#NAMES[@]}")"

    reason=""
    case "$need" in
        mpi)      { [ "$CAP_MPI" = yes ] && [ "$CAP_HDF5_MPI" = yes ]; } || reason="needs mpicxx + parallel HDF5" ;;
        cuda)     [ "$CAP_NVCC" = yes ] || reason="needs nvcc" ;;
        cuda+mpi) { [ "$CAP_NVCC" = yes ] && [ "$CAP_MPI" = yes ] && [ "$CAP_HDF5_MPI" = yes ]; } || reason="needs nvcc + mpicxx + parallel HDF5" ;;
    esac
    [ -z "$reason" ] && [ "$CAP_HDF5" = no ] && reason="needs HDF5"

    if [ -n "$reason" ]; then
        printf '[%s] \033[33mSKIP\033[0m  %-30s (%s)\n' "$idx" "$name" "$reason"
        n_skip=$((n_skip + 1)); continue
    fi

    b="$WORK/$name"; mkdir -p "$b"
    printf '%s\n' $flags > "$b/Config.sh"

    t0=$SECONDS
    make ${JOBS:+-j$JOBS} CONFIG="$b/Config.sh" SYSTYPE="$SYSTYPE" WERROR="$WERROR" \
         BUILD_DIR="$b" EXEC="$b/ProteusGPU" > "$b/build.log" 2>&1
    rc=$?
    dt=$((SECONDS - t0))

    warn=$(grep -cE 'warning:|warning #' "$b/build.log" 2>/dev/null)
    warn=${warn:-0}

    ok=0
    if [ "$tier" = mustfail ]; then
        [ "$rc" -ne 0 ] && [ ! -x "$b/ProteusGPU" ] && ok=1
    else
        [ "$rc" -eq 0 ] && [ -x "$b/ProteusGPU" ] && ok=1
    fi

    if [ "$ok" -eq 1 ]; then
        extra=""
        if [ "$tier" = mustfail ]; then
            extra=" (guard fired, as expected)"
        elif [ "$warn" -gt 0 ]; then
            extra=" \033[33m${warn} warning(s)\033[0m"
        fi
        printf '[%s] \033[32mPASS\033[0m  %-30s %3ds%b\n' "$idx" "$name" "$dt" "$extra"
        n_pass=$((n_pass + 1))
    else
        if [ "$tier" = mustfail ]; then
            printf '[%s] \033[31mFAIL\033[0m  %-30s %3ds  (guard did NOT fire — invalid config built)\n' "$idx" "$name" "$dt"
        else
            printf '[%s] \033[31mFAIL\033[0m  %-30s %3ds\n' "$idx" "$name" "$dt"
            grep -m3 -E 'error:|Error [0-9]+|\*\*\*' "$b/build.log" | sed 's/^/           /'
        fi
        FAILED+=("$name")
        n_fail=$((n_fail + 1))
        KEEP=1   # preserve logs when something failed
    fi
done

# ============================================================
# Summary
# ============================================================
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
    printf '\nfailed configurations:\n'
    for f in "${FAILED[@]}"; do printf '  %s   (log: %s/%s/build.log)\n' "$f" "$WORK" "$f"; done
    exit 1
fi
exit 0
