#!/usr/bin/env bash
# Run GPU unit tests with the adapter required, and fail unless they really ran.
#
# Unit tests take their device from `device::test_device()`, which skips the
# test when no adapter is present. With TRT_REQUIRE_GPU=1 (set here) a missing
# adapter panics instead. This wrapper also fails when
#   - no test acquired a device (`[gpu] test device acquired` count is 0), or
#   - any test still reports a skip (`[gpu] test skipped` count is not 0), or
#   - zero tests passed.
# Used by ci.yml and scripts/preflight.sh with the same arguments.
#
# usage: scripts/cargo_test_gpu.sh <cargo test arguments...>
#        scripts/cargo_test_gpu.sh --self-test <cargo test arguments...>
#   --self-test hides the adapter (TRT_TEST_FORCE_NO_ADAPTER=1) and passes only
#   when the run fails with the TRT_REQUIRE_GPU panic, i.e. checks that the
#   requirement has teeth.
set -euo pipefail
self_test=0
if [ "${1:-}" = "--self-test" ]; then
  self_test=1
  shift
fi
log=$(mktemp)
trap 'rm -f "$log"' EXIT
export TRT_REQUIRE_GPU=1
if [ "$self_test" -eq 1 ]; then
  if TRT_TEST_FORCE_NO_ADAPTER=1 cargo test "$@" 2>&1 | tee "$log"; then
    echo "error: self-test: the run passed with the adapter hidden" >&2
    exit 1
  fi
  hits=$(grep -c 'TRT_REQUIRE_GPU is set but no GPU adapter is available' "$log" || true)
  echo "self-test: runs failed on the missing adapter: $hits"
  if [ "$hits" -eq 0 ]; then
    echo "error: self-test: the run failed for another reason" >&2
    exit 1
  fi
  exit 0
fi
cargo test "$@" 2>&1 | tee "$log"
passed=$(grep -oE 'test result: ok\. [0-9]+ passed' "$log" | awk '{s += $4} END {print s + 0}')
acquired=$(grep -c '\[gpu\] test device acquired' "$log" || true)
skipped=$(grep -c '\[gpu\] test skipped' "$log" || true)
echo "tests passed: $passed, GPU devices acquired: $acquired, GPU skips: $skipped"
if [ "$passed" -eq 0 ]; then
  echo "error: cargo test $* executed zero tests" >&2
  exit 1
fi
if [ "$acquired" -eq 0 ]; then
  echo "error: cargo test $* acquired no GPU device" >&2
  exit 1
fi
if [ "$skipped" -ne 0 ]; then
  echo "error: cargo test $* skipped $skipped GPU test(s)" >&2
  exit 1
fi
