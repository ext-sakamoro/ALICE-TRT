#!/usr/bin/env bash
# Remove the optional sibling path dependencies that CI does not clone from
# Cargo.toml (run from the repository root, on a CI checkout only).
#
#   scripts/strip_sibling_deps.sh                  drop alice-sdf / alice-db
#   scripts/strip_sibling_deps.sh --drop-physics   also drop alice-physics and
#                                                  the features that need it
#
# alice-sdf and alice-db each pull further sibling path dependencies
# (alice-db -> alice-analytics, alice-crypto, ...) that CI does not check out,
# and Cargo resolves every path dependency, optional or not, before building.
# The `sdf` and `db` features are therefore not built in CI.
set -euo pipefail

drop_physics=0
case "${1:-}" in
  --drop-physics) drop_physics=1 ;;
  "") ;;
  *) echo "usage: $0 [--drop-physics]" >&2; exit 2 ;;
esac

# `sed -i.bak` works on both BSD (macOS) and GNU sed
sed -i.bak '/^alice-sdf = {/d; /^alice-db = {/d' Cargo.toml
# the features stay declared but empty, so `#[cfg(feature = "sdf")]` / `"db"` are
# still known cfg values (check-cfg) and the gated code is simply not built
sed -i.bak 's/^sdf = \["dep:alice-sdf"\].*/sdf = []/; s/^db = \["dep:alice-db"\].*/db = []/' Cargo.toml
if [ "$drop_physics" = 1 ]; then
  sed -i.bak '/^alice-physics = {/d' Cargo.toml
  sed -i.bak 's/^physics = \["dep:alice-physics"\].*/physics = []/; s/^physics-solver = .*/physics-solver = []/' Cargo.toml
fi
rm -f Cargo.toml.bak

# a pattern that stopped matching leaves the dependency in place; fail here
# rather than later with an unresolved path
for name in alice-sdf alice-db; do
  if grep -q "^$name = {" Cargo.toml; then echo "strip failed: $name still declared" >&2; exit 1; fi
done
if [ "$drop_physics" = 1 ] && grep -q '^alice-physics = {' Cargo.toml; then
  echo "strip failed: alice-physics still declared" >&2; exit 1
fi
for name in sdf db; do
  grep -q "^$name = \[\]$" Cargo.toml || { echo "strip failed: feature $name not emptied" >&2; exit 1; }
done
if [ "$drop_physics" = 1 ]; then
  for name in physics physics-solver; do
    grep -q "^$name = \[\]$" Cargo.toml || { echo "strip failed: feature $name not emptied" >&2; exit 1; }
  done
fi
echo "--- Cargo.toml after strip ---"
grep -E '^(alice-|\[|physics|sdf|db) ' Cargo.toml || true
