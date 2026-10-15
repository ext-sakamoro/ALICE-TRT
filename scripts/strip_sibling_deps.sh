#!/usr/bin/env bash
# Remove the optional sibling path dependencies that CI does not clone from
# Cargo.toml (run from the repository root, on a CI checkout only).
#
#   scripts/strip_sibling_deps.sh                  drop alice-sdf
#   scripts/strip_sibling_deps.sh --drop-physics   also drop alice-physics and
#                                                  the features that need it
#
# alice-sdf pulls further sibling path dependencies that CI does not check
# out, and Cargo resolves every path dependency, optional or not, before
# building. The `sdf` feature is therefore not built in CI. alice-db is a
# crates.io dependency and is left in place (the `db` feature is built).
set -euo pipefail

drop_physics=0
case "${1:-}" in
  --drop-physics) drop_physics=1 ;;
  "") ;;
  *) echo "usage: $0 [--drop-physics]" >&2; exit 2 ;;
esac

# `sed -i.bak` works on both BSD (macOS) and GNU sed
sed -i.bak '/^alice-sdf = {/d' Cargo.toml
# the feature stays declared but empty, so `#[cfg(feature = "sdf")]` is still a
# known cfg value (check-cfg) and the gated code is simply not built
sed -i.bak 's/^sdf = \["dep:alice-sdf"\].*/sdf = []/' Cargo.toml
if [ "$drop_physics" = 1 ]; then
  sed -i.bak '/^alice-physics = {/d' Cargo.toml
  sed -i.bak 's/^physics = \["dep:alice-physics"\].*/physics = []/; s/^physics-solver = .*/physics-solver = []/' Cargo.toml
fi
rm -f Cargo.toml.bak

# a pattern that stopped matching leaves the dependency in place; fail here
# rather than later with an unresolved path
if grep -q '^alice-sdf = {' Cargo.toml; then echo "strip failed: alice-sdf still declared" >&2; exit 1; fi
# alice-db must stay a registry dependency: a path form would not resolve on CI
grep -q '^alice-db = { version = ' Cargo.toml || { echo "alice-db is not a crates.io dependency" >&2; exit 1; }
grep -q '^db = \["dep:alice-db"\]' Cargo.toml || { echo "feature db does not enable alice-db" >&2; exit 1; }
if [ "$drop_physics" = 1 ] && grep -q '^alice-physics = {' Cargo.toml; then
  echo "strip failed: alice-physics still declared" >&2; exit 1
fi
grep -q '^sdf = \[\]$' Cargo.toml || { echo "strip failed: feature sdf not emptied" >&2; exit 1; }
if [ "$drop_physics" = 1 ]; then
  for name in physics physics-solver; do
    grep -q "^$name = \[\]$" Cargo.toml || { echo "strip failed: feature $name not emptied" >&2; exit 1; }
  done
fi
echo "--- Cargo.toml after strip ---"
grep -E '^(alice-|\[|physics|sdf|db) ' Cargo.toml || true
