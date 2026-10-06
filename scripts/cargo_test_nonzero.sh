#!/usr/bin/env bash
# Run `cargo test <args>` and fail when it executed zero tests.
#
# A test filter that matches nothing, a feature that compiles a module out, or
# a renamed module all leave `cargo test` green with "0 passed"; this wrapper
# turns that into a failure. Used by ci.yml and scripts/preflight.sh with the
# same arguments.
#
# usage: scripts/cargo_test_nonzero.sh <cargo test arguments...>
set -euo pipefail
log=$(mktemp)
trap 'rm -f "$log"' EXIT
cargo test "$@" 2>&1 | tee "$log"
passed=$(grep -oE 'test result: ok\. [0-9]+ passed' "$log" | awk '{s += $4} END {print s + 0}')
echo "tests passed in total: $passed"
if [ "$passed" -eq 0 ]; then
  echo "error: cargo test $* executed zero tests" >&2
  exit 1
fi
