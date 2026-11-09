#!/usr/bin/env bash
# Builds Proteus and runs its tests once for every line of tests/build_configs.txt, each in its own build folder.
#
#   tests/build_matrix.sh          every line
#   tests/build_matrix.sh --cpu    only the lines without CUDA, for machines without a GPU
#
# A configuration fails if the build or a test fails, or if any warning shows up. Exits with 1 if one fails.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS="$ROOT/tests/build_configs.txt"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cpu_only=0
[ "${1:-}" = "--cpu" ] && cpu_only=1

passed=0
failed=0
skipped=0
while read -r name switches; do
    case "$name" in '' | '#'*) continue ;; esac

    if [ $cpu_only = 1 ] && [[ " $switches " == *" CUDA "* ]]; then
        echo "skip  $name"
        skipped=$((skipped + 1))
        continue
    fi

    # the switches of this line as a Config.sh of its own
    printf '%s\n' $switches > "$WORK/$name.sh"
    log="$WORK/$name.log"
    make -C "$ROOT" -s all test CONFIG="$WORK/$name.sh" BUILD_DIR="$WORK/build_$name" EXEC="$WORK/proteus_$name" \
        < /dev/null > "$log" 2>&1
    status=$?

    if [ $status = 0 ] && ! grep -qi "warning" "$log"; then
        echo "ok    $name"
        passed=$((passed + 1))
    else
        echo "FAIL  $name"
        tail -n 30 "$log"
        failed=$((failed + 1))
    fi
done < "$CONFIGS"

echo "build matrix: $passed passed, $failed failed, $skipped skipped"
[ $failed = 0 ]
