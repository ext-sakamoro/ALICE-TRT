#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push` (ci.yml). Every command
# below is the one CI runs, with the same arguments; a step this script does not
# cover is a step that can only fail remotely. When a workflow step is added,
# add it here in the same commit.
#
# usage: scripts/preflight.sh [--quick]
#   (none)   every static gate + every test suite ci.yml runs
#   --quick  every static gate + the device-free lib tests (cpu-test job)
#
# Steps CI runs that this script does not reproduce:
#   - .github/actions/sibling-deps: clones ALICE-ML / ALICE-Physics next to the
#     checkout and strips alice-sdf / alice-db from Cargo.toml. Locally the
#     sibling checkouts (../ALICE-ML, ../ALICE-Physics, ../ALICE-SDF,
#     ../ALICE-DB and its own siblings) must exist so Cargo can resolve
#     Cargo.toml unmodified.
#   - fix128-gpu-matrix / fix128-physics-solver-matrix "Install Vulkan / Mesa":
#     Linux runner only. The GPU test steps below run on whatever adapter this
#     machine has (Metal / Vulkan / DX12); the other two adapters in the CI
#     matrix (Vulkan lavapipe, DX12 WARP) can only be exercised on CI.
#   - msrv: needs the 1.87 toolchain (`rustup toolchain install 1.87`); run
#     here when it is installed, otherwise reported as skipped.
#   - security-audit.yml (cargo audit / deny / machete) is not part of this
#     script; it runs on dependency changes and weekly.
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
case "${1:-}" in
  --quick) quick=1 ;;
  "") ;;
  *) echo "usage: scripts/preflight.sh [--quick]" >&2; exit 2 ;;
esac

# the feature set every CI job that builds the crate uses (sdf / db / cuda are
# not built in CI; python is linted only)
CI_FEATURES='ffi,view,voice,fix128-arithmetic,physics-solver'

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

need actionlint "brew install actionlint"
need python3 "Python 3.9+"
has_toolchain 1.92.0 || { echo "missing toolchain 1.92.0 (rustup toolchain install 1.92.0)" >&2; exit 1; }

step "ci.yml / fmt"
cargo fmt -- --check

step "ci.yml / actionlint"
actionlint .github/workflows/*.yml

step "ci.yml / docs-lint"
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check

step "ci.yml / clippy (default features, all targets)"
cargo +1.92.0 clippy --all-targets -- -D warnings

# pyo3 0.22 knows CPython up to 3.13; with a newer local interpreter the
# forward-compatibility switch lets the extension module type-check (CI pins 3.12)
step "ci.yml / clippy (every feature CI builds, all targets)"
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
  cargo +1.92.0 clippy --all-targets --features "ffi,python,view,voice,fix128-arithmetic,physics-solver" -- -D warnings

step "ci.yml / doc (RUSTDOCFLAGS=-Dwarnings)"
RUSTDOCFLAGS="-Dwarnings" cargo +1.92.0 doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo +1.92.0 doc --lib --no-deps --features "$CI_FEATURES"

step "ci.yml / msrv (rust-version = 1.87)"
if has_toolchain 1.87; then
  cargo +1.87 check --lib
  cargo +1.87 check --lib --features "$CI_FEATURES"
else
  echo "skip: toolchain 1.87 not installed (rustup toolchain install 1.87); CI runs this check" >&2
fi

step "ci.yml / cpu-test (device-free modules + doctests, zero executed is red)"
bash scripts/cargo_test_nonzero.sh --lib --features "view,voice" -- constraint_graph:: kernel:: view_bridge:: voice_bridge::
bash scripts/cargo_test_nonzero.sh --doc --features "ffi,view,voice"

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (GPU test suites skipped)"; exit 0
fi

step "ci.yml / fix128-gpu-matrix: build"
cargo +1.92.0 build --lib --features fix128-arithmetic

step "ci.yml / fix128-gpu-matrix: Fix128 CPU reference"
cargo +1.92.0 test --lib --features fix128-arithmetic fix128::tests::fix128_gpu_

step "ci.yml / fix128-gpu-matrix: WGSL shader source symbol coverage"
cargo +1.92.0 test --lib --features fix128-arithmetic fix128::tests::wgsl_

step "ci.yml / fix128-gpu-matrix: Fix128 GPU dispatch"
cargo +1.92.0 test --lib --features fix128-arithmetic fix128 -- --nocapture --test-threads=1

step "ci.yml / fix128-gpu-matrix: analytic oracles"
cargo +1.92.0 test --test analytic_oracle --features fix128-arithmetic,voice,view -- --test-threads=1

step "ci.yml / fix128-physics-solver-matrix: build"
cargo +1.92.0 build --lib --features fix128-arithmetic,physics-solver

step "ci.yml / fix128-physics-solver-matrix: physics-solver byte-exact CPU-GPU goldens"
cargo +1.92.0 test --lib --features fix128-arithmetic,physics-solver -- --nocapture --test-threads=1

echo; echo "preflight OK"
