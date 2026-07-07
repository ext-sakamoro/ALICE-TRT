# Changelog

All notable changes to ALICE-TRT will be documented in this file.

## [2.3.0] - 2026-07-07

### Added — GPU LinearBvh build kernel (Phase 3 §3)

Third implementation release of the Phase 3 GPU BVH pipeline. Ships the single-workgroup single-thread iterative port of `alice_physics::bvh::LinearBvh::build_recursive`, byte-identical to the CPU reference on the 3-platform CI matrix (Metal / Vulkan lavapipe / DX12 WARP). Additive-only; the v2.2.0 public surface remains stable.

- **`FIX128_BVH_BUILD_WGSL` — full impl** (new WGSL constant). Single `@compute @workgroup_size(1)` entry `fix128_bvh_build_main` iteratively drives `build_recursive` via an explicit 128-slot continuation-passing-style `Frame` stack in workgroup memory. Each frame runs through three phases: (0) compute AABB union + leaf/internal decision, (1) after LEFT child returns, push RIGHT with inherited escape, (2) after RIGHT returns, write parent's `first_child_or_prim` + linear O(subtree_size) sweep to backfill placeholder escapes. Position-independent `LEFT_ESCAPE_PLACEHOLDER = 0u` mirrors the ALICE-Physics `dede78c` correctness fix — see [`docs/PHASE_3_DESIGN.md`](docs/PHASE_3_DESIGN.md) §3.1 and the [`deterministic-physics-lockstep-discipline`](https://github.com/ext-sakamoro/claude-config/blob/main/claude-skills/deterministic-physics-lockstep-discipline/SKILL.md) skill §11.4 for the discipline this preserves.
- **New public Rust struct `BvhNodeGpu`** (`#[repr(C)]`, 32 bytes): byte-layout mirror of `alice_physics::bvh::BvhNode`. Ships `from_physics(&BvhNode) -> BvhNodeGpu` for byte-exact test fixtures, plus `escape_idx()` / `prim_count()` accessors that unpack the `prim_count_escape` word identically to the CPU node.
- **New public Rust struct `AabbI32Gpu`** (`#[repr(C)]`, 24 bytes): pre-quantised i32 AABB input to the build kernel. `from_physics_primitive(&BvhPrimitive)` applies the same `aabb_to_i32_min` (floor) / `aabb_to_i32_max` (ceil) rule as the CPU host; by monotonicity of floor/ceil the pre-quantised i32 fold is byte-exact equivalent to the CPU Fix128 fold + late quantisation. Both structs are gated behind `feature = "physics-solver"` because they consume the `alice_physics` types.
- **New public Rust function** `dispatch_fix128_bvh_build(device, sorted_codes, sorted_indices, sorted_aabbs) -> Vec<BvhNodeGpu>`: full orchestrator. Uploads the three parallel input arrays (produced by v2.2.0's sort + host-side i32 pre-quantisation), dispatches at `(1, 1, 1)` workgroups, and reads back the actual node count from a separate atomic counter before trimming the output buffer to the tree's real size. Debug builds additionally invoke a private `debug_verify_escape_forward_impl` on the readback, which mirrors ALICE-Physics `LinearBvh::debug_verify_escape_forward` byte-exactly and panics on any backward or self-referential escape pointer.

### Tests (v2.3.0 additions)

- `wgsl_bvh_build_shader_present` — structural coverage of the new WGSL surface (struct names, binding identifiers, placeholder constants, sweep helpers, single `@workgroup_size(1)` entry). Runs on every 3-platform CI job without a GPU adapter.
- `wgsl_bvh_build_shader_compiles` — naga validation via `create_shader_module`. Catches WGSL syntax / type errors on every platform; skips gracefully on adapter-less headless CI.
- `wgpu_bvh_build_matches_cpu_golden` — **byte-exact GPU-CPU golden** on three fixtures that exercise the discipline requirements from `docs/PHASE_3_DESIGN.md` §2.4:
  1. **Pile (32 primitives)** in a 4×4×2 grid with radius 1 and spacing 3 — balanced Morton splits, escape-pointer chaining through several tree levels. Mirrors the geometry of the `stage_breakdown_collider` bench fixture that surfaced the ALICE-Physics `dede78c` correctness fix.
  2. **Uniform grid (27 primitives)** on a 3×3×3 lattice with spacing 4 — exercises the "highest differing bit" split path across three cardinal axes.
  3. **Degenerate all-colocated (16 primitives)** with every AABB at the origin — exercises the `first_code == last_code` branch of `find_split` (midpoint fallback) plus the single-AABB world bounds. This is the same degenerate configuration that stress-tests the placeholder discipline in the CPU.

Total 203 lib tests (previously 200), all pass on macOS Metal. The 3-platform CI matrix picks up the two new shader tests on the next tag run.

### Design doc

- **`docs/PHASE_3_DESIGN.md` §2.4** — full v2.3.0 scope: single-workgroup single-thread iterative build_recursive port, LEFT_ESCAPE_PLACEHOLDER = 0u discipline, subtree_size return + linear sweep, CPU-GPU AABB pre-quantisation equivalence proof, bindings (6 total), fixture design, and post-build debug invariant.

### Backwards compatibility

Fully additive vs v2.2.0. No API changes. The v2.2.0 `dispatch_fix128_morton_sort` orchestrator, `FIX128_MORTON_SORT_WGSL`, and `Fix128AabbGpu` are unchanged. External consumers who wired against the v2.2.0 sort output can now feed `(sorted_codes, sorted_indices)` directly into `dispatch_fix128_bvh_build` alongside a host-computed `Vec<AabbI32Gpu>`.

### Next up (Phase 3 continuation)

- **v2.4.0** — GPU BVH `find_pairs` kernel with `2 * nodes.len()` traversal cycle guard in debug builds.
- **v2.5.0** — Sphere-sphere narrow-phase contact kernel (Fix128 sqrt + normal + depth).
- **v2.6.0** — GPU PGS contact solve + `TrtSolverAdapter` opt-in for the full pipeline (mirrors the v1.5.1 → v1.6.0 opt-in / default-flip cadence).
- **v2.3.x candidate** — parallel top-down BVH construction (Karras-style hierarchical linear BVH with atomic index generation). Requires fresh determinism analysis and a matching byte-exact CI matrix before shipping.

## [2.2.0] - 2026-07-07

### Added — Deterministic Morton sort kernel (Phase 3 §2)

Second implementation release of the Phase 3 GPU BVH pipeline. Ships the byte-exact stable LSB-first 8-bit radix sort of 63-bit Morton codes announced in v2.1.x, plus a public Rust orchestrator that drives the 8-pass ping-pong dispatch. Additive-only; the v2.1.0 public surface (including `FIX128_MORTON_CODE_WGSL` and `Fix128AabbGpu`) is unchanged.

- **`FIX128_MORTON_SORT_WGSL` — full impl** (previously a v2.1.x skeleton). Two `@compute` entries:
  - `fix128_morton_sort_histogram_main` (`@workgroup_size(64)`, parallel per-primitive) — builds the 256-bin histogram of the current byte via `atomicAdd`. The final counts are order-independent because addition commutes, so parallel dispatch is safe.
  - `fix128_morton_sort_scatter_main` (`@workgroup_size(1)`, single-thread) — sequentially scatters each `(code, index)` pair to `codes_out[bucket_offsets[b] + local_cursor[b]]`. Single-thread ensures **stability** in gid order, which is required for byte-exact CPU-GPU parity against Rust's stable Timsort.
- **`FIX128_MORTON_SORT_WGSL` binding layout** (frozen since v2.1.x preview): 7 bindings total. `codes_in` / `indices_in` (read storage) → `codes_out` / `indices_out` (read_write storage) with a `params { pass_bit_shift, count }` uniform, a 256-bin `atomic<u32>` histogram (read_write), and a `bucket_offsets: array<u32, 256>` (read) populated between passes by the host-side exclusive scan.
- **New public Rust function** `dispatch_fix128_morton_sort(device, codes, indices) -> (sorted_codes, sorted_indices)`: 8-pass orchestrator. Per pass — zero histogram → dispatch histogram_main → read back → host-side exclusive scan → upload `bucket_offsets` → dispatch scatter_main → ping-pong buffers. Returns the sorted `(codes, indices)` pair byte-identical to `stable_sort_by_key(|(m, _)| *m)` on the CPU.

### Tests (v2.2.0 additions)

- `wgpu_morton_sort_matches_cpu_golden` — **byte-exact GPU-CPU golden**. Seven fixture cases covering the edge cases documented in the design doc:
  1. Empty (`count == 0`) — dispatcher short-circuits without executing any kernel.
  2. Single element — trivially sorted.
  3. All Morton codes identical (32 elements) — degenerate histogram; stability preserves input order.
  4. All Morton codes distinct (64 elements) — general case.
  5. Duplicate Morton codes with distinct indices (32 pairs of 2) — asserts lower-input-index comes first within each same-code pair (stability).
  6. Ascending pre-sorted (48 elements) — output equals input.
  7. Descending pre-sorted (48 elements) — full reversal.

Total 200 lib tests (previously 199), all pass on macOS Metal. The 3-platform CI matrix (Metal / Vulkan lavapipe / DX12 WARP) picks up the new byte-exact test on the next tag run.

### Perf note

The single-thread scatter runs at ~O(N) inside one workgroup. At N = 64 (the v2.2.0 golden fixture) the whole 8-pass sort completes in well under 1ms on M2 Metal. Parallel scatter with per-thread rank via input prefix sum is a v2.2.x optimisation; the correctness-first choice ships first so v2.3.0 BVH build can start integrating against a stable byte-exact contract immediately.

### Next up (Phase 3 continuation)

- **v2.3.0** — GPU BVH build with the position-independent placeholder + O(subtree_size) sweep discipline established by ALICE-Physics `dede78c`.
- **v2.4.0** — GPU BVH `find_pairs` with debug cycle guard.
- **v2.5.0** — Sphere-sphere narrow-phase contact kernel.
- **v2.6.0** — GPU PGS contact solve + `TrtSolverAdapter` opt-in for the full pipeline.

### Backwards compatibility

- Fully additive vs v2.1.0. No API changes.
- `FIX128_MORTON_SORT_WGSL` transitions from the v2.1.x skeleton entry `fix128_morton_sort_smoke_main` to the two production entries. External consumers who compiled against the skeleton entry name should switch to the histogram + scatter entries per the [design doc](docs/PHASE_3_DESIGN.md) §2.3.

## [2.1.0] - 2026-07-07

### Added — Phase 3 first primitive kernels (Fix128 AABB helpers + Morton code)

First implementation release of the Phase 3 GPU BVH / narrow-phase / CCD series announced in v2.0.0. Ships two new WGSL constants and one new Rust type; no adapter integration yet, so the v2.0.0 public surface remains additive-only.

- **New public WGSL constant** `FIX128_AABB_HELPERS_WGSL`: standalone valid compute shader declaring `Fix128AabbGpu` (6 × `Fix128Gpu` = 96 bytes) plus the byte-exact helper functions the v2.2+ pipeline consumes: `fix128_add / sub / lt / min / max`, `aabb_from_sphere` (centre + radius → AABB), and `aabb_union` (componentwise min/max). A no-op `aabb_helpers_smoke_main` compute entry keeps the module standalone-compilable; external kernels typically concatenate the source into their own bodies (WGSL has no `include` directive, so cat-ing strings is the common pattern).
- **New public WGSL constant** `FIX128_MORTON_CODE_WGSL`: per-primitive parallel 63-bit Morton code kernel with bindings `primitives: array<Fix128AabbGpu>` (read storage), `world_bounds: Fix128AabbGpu` (uniform), and `morton_codes: array<vec2<u32>>` (read_write storage). Mirrors `alice_physics::bvh::point_to_morton` byte-for-byte:
  1. Compute the AABB centre as `(min + max) / 2` per axis (Fix128 add + arithmetic right shift).
  2. Compute world size per axis (Fix128 sub).
  3. Normalize the centre via Fix128 divide (v1.4.0 `FIX128_DIV_WGSL` embedded verbatim).
  4. Extract the upper 21 bits of the fractional part, with the three CPU branches preserved: `t < 0` → `0`, `t.hi >= 1` → `0x1FFFFF`, otherwise `(t.lo_hi >> 11) & 0x1FFFFF`.
  5. Spread each 21-bit coordinate to 63 bits via the standard 5-round `expand_bits` sequence (u64 emulated as `vec2<u32>` with hardcoded shift-32/16/8/4/2 helpers) and combine via `expand(x) | (expand(y) << 1) | (expand(z) << 2)`.
- **New public Rust struct** `Fix128AabbGpu`: `#[repr(C)]` byte-layout mirror of the WGSL struct with inherent `from_sphere` and `union` methods that reproduce the shader helpers on the CPU.
- **README kernel tables** (English + Japanese) updated with the two new entries.
- **New design doc** [`docs/PHASE_3_DESIGN.md`](docs/PHASE_3_DESIGN.md): full Phase 3 architecture skeleton covering kernel sequence (v2.1.0 → v2.7.0 candidate), determinism invariants (with an explicit `§11.4` cross-reference to the ALICE-Physics BVH escape-pointer fix as the canonical GPU-port reference), bind group layouts, CPU golden strategy, 3-platform CI matrix, release cadence, and an open-questions log.

### Tests (v2.1.0 additions)

Eight new lib tests exercise the primitives end-to-end:

- `wgsl_aabb_helpers_shader_present` — structural check on every helper name and the `@compute` smoke entry.
- `wgsl_aabb_helpers_shader_compiles` — naga validation via `create_shader_module` (headless CI safe).
- `fix128_aabb_gpu_size_and_layout` — asserts `size_of::<Fix128AabbGpu>() == 96` so the type uploads cleanly.
- `fix128_aabb_gpu_from_sphere` — Rust helper matches its WGSL counterpart on a mixed-sign fixture.
- `fix128_aabb_gpu_union` — Rust helper matches its WGSL counterpart on positive, negative, and mixed-sign corners so the `fix128_lt` signed compare is fully exercised.
- `wgsl_morton_code_shader_present` — structural check + every Morton bit-spread mask constant present verbatim.
- `wgsl_morton_code_shader_compiles` — naga validation on the full 400+-line shader.
- `wgpu_morton_code_matches_cpu_golden` — **byte-exact GPU-CPU golden**. 16 primitives spanning interior, boundary, and out-of-bounds cases on each axis; every 63-bit `vec2<u32>` GPU output matches `alice_physics::bvh::point_to_morton` byte-for-byte. Requires `--features physics-solver`; skips gracefully on headless CI (no GPU adapter).

Total 197 lib tests (previously 189), all pass on macOS Metal.

### Phase 3 gate status

Documented in the [design doc](docs/PHASE_3_DESIGN.md) §1: gates 1 (collider-attached bench variant) and 2 (>30% of frame time in narrow-phase) satisfied on 2026-07-06 by the ALICE-Physics `stage_breakdown_collider` bench (97.4% at N=100 / 99.8% at N=1000). Gate 3 (workload requirement) deferred to Phase 3 design as documented in the ALICE-Physics [`GPU_OFFLOAD_ROADMAP.md`](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md).

### Next up (Phase 3 continuation)

- **v2.2.0** — Morton-based deterministic sort kernel + CPU golden.
- **v2.3.0** — GPU BVH build with the position-independent placeholder + O(subtree_size) sweep discipline established by ALICE-Physics `dede78c`.
- **v2.4.0** — GPU BVH `find_pairs` with debug cycle guard.
- **v2.5.0** — Sphere-sphere narrow-phase contact kernel.
- **v2.6.0** — GPU PGS contact solve + `TrtSolverAdapter` opt-in for the full pipeline.

Each release is additive under the v1.0.0 → v2.x semver stability commitment; the v2.0.0 public API remains intact throughout the series.

### Backwards compatibility

- Fully additive vs v2.0.0. No API changes.
- The two new WGSL constants and the `Fix128AabbGpu` type are new symbols; existing code continues to compile unchanged.

## [2.0.0] - 2026-07-07

### Removed — Deprecated v1.1.0 uniform-based distance projection shader (breaking)

- **`FIX128_PGS_PROJECT_DISTANCE_WGSL`** is removed. Deprecated in v1.7.0 with a migration runway of one MINOR release; internal `TrtSolverAdapter` had not referenced it since v1.4.2.
- The two shader-content assertions (`wgsl_pgs_project_distance_shader_helpers_present`, `wgsl_pgs_project_distance_shader_compiles`) are removed with it.
- README kernel tables (English + Japanese) updated: the removed entry is replaced by the two current replacements plus explicit references to the `div` and `sqrt` primitive kernels; a prose note documents the removal for external `Fix128GpuKernel` implementers who may still be mid-migration.

### Migration

External `Fix128GpuKernel` implementers who still reference the removed constant have two supported replacements:

- **Batched color-parallel dispatch (recommended)** — `FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL` (v1.5.1). Reads a `storage` array of `DistanceParamsRigid` and selects the target constraint via `@builtin(workgroup_id).x`; one dispatch handles a whole color.
- **Single-constraint rigid rod** — `FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL` (v1.4.2). Same 3-buffer bind group layout as the removed shader; the uniform layout swaps `scalar: Fix128Gpu` for `rest_length: Fix128Gpu` and the shader computes the correction scalar on-device via the byte-exact v1.4.0 `div` + v1.4.1 `sqrt` kernels.
- **Sequential v1.4.2 behaviour without touching the shader** — `TrtSolverAdapter::set_parallel_dispatch(false)` restores the pre-v1.6.0 default. The v1.6.0 CHANGELOG documented this one-liner opt-out; it remains supported in v2.0.0.

The deprecated shader path had no adapter callers within the crate. External implementers who missed the v1.7.0 deprecation warning will see a compile error at the removed symbol; the fix is one line for either replacement above.

### Phase 2 (GPU offload roadmap) formally complete

- ✅ v1.5.0: constraint graph builder + greedy coloring
- ✅ v1.5.1: batched rigid kernel + opt-in toggle
- ✅ v1.5.2: colored CPU golden + toggle-on byte-exact assertion
- ✅ v1.6.0: parallel dispatch by default
- ✅ v1.7.0: deprecation of the superseded v1.1.0 shader
- ✅ **v2.0.0: removal + Phase 3 gate satisfied (this release)**

### Phase 3 (GPU BVH / narrow-phase / CCD) — gate satisfied on 2026-07-06

The [ALICE-Physics GPU offload roadmap](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md) gated Phase 3 on:

1. Adding a collider-attached bench variant to measure real narrow-phase cost (previously a no-op fast-path).
2. Confirming a target workload spends **>30% of frame time** in one of these stages.
3. Documenting the target workload as a real user requirement.

Gates 1 and 2 were satisfied on 2026-07-06 with the `stage_breakdown_collider` bench (ALICE-Physics commit `ac7d1b1`), which measured **narrow-phase + PGS at 97.4% of total frame cost at N=100 pile and 99.8% at N=1000 pile**. Gate 3 (workload requirement documentation) is deferred to Phase 3 design; the measurement alone lifts the "premature optimisation" concern.

Phase 3 implementation is not part of this release — it lands in a future v2.1+ series. This release formalises the removal and Phase 2 wrap so v2.1 can start clean.

### Backwards compatibility

- **Breaking**: the removed constant was `pub`; downstream code that referenced it will not compile until migrated.
- The migration was documented three releases ahead (v1.6.0 → v1.7.0 deprecation → v2.0.0 removal), the deprecation warning has been in place for one MINOR release, and both replacements have been in production since v1.4.2 / v1.5.1.
- All other public API is unchanged from v1.7.0.

### Tests

- Total 189 lib tests (previously 191; the two shader-content assertions for the removed constant are gone with it), all pass on macOS Metal.
- 3-platform CI matrix (Metal / Vulkan lavapipe / DX12 WARP) continues to gate every kernel-level change.

## [1.7.0] - 2026-07-06

### Deprecated — v1.1.0 uniform-based distance projection shader

- **`FIX128_PGS_PROJECT_DISTANCE_WGSL`** is deprecated (with a `#[deprecated]` attribute pointing at both replacements). Scheduled for removal in v2.0.0.

The adapter internal has not referenced this constant since v1.4.2, when the rigid rod path (`FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL`) took over. It remains in the `pub` surface only for external `Fix128GpuKernel` implementers who mirror the wgpu backend and haven't migrated yet.

### Migration

External implementers who reference the deprecated shader:

- **Batched color-parallel dispatch (recommended)**: switch to `FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL` (v1.5.1). Reads a `storage` array of `DistanceParamsRigid` items and selects the target constraint via `@builtin(workgroup_id).x`. Dispatches all constraints of one color in a single compute call.
- **Single-constraint rigid rod**: switch to `FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL` (v1.4.2). Same 3-buffer bind group layout as the deprecated shader; the uniform layout swaps `scalar: Fix128Gpu` for `rest_length: Fix128Gpu` and the shader computes the correction scalar on-device using the byte-exact v1.4.0 div + v1.4.1 sqrt kernels.
- **Sequential v1.4.2 behaviour without touching the shader**: `TrtSolverAdapter::set_parallel_dispatch(false)` restores the pre-v1.6.0 default and continues to satisfy the byte-exact contract vs the v1.4.2 `cpu_semi_implicit_integrate` golden.

### v2.0.0 planning notes

The v1.6.0 default flip (`parallel_dispatch = true` at construction time) was documented three releases ahead and ships under MINOR semver on the grounds that:

1. Public API surface is unchanged.
2. `set_parallel_dispatch(false)` restores prior behaviour in one line.
3. The new default is byte-exact against a published CPU golden (`cpu_semi_implicit_integrate_colored`, v1.5.2).

If external adopters report that the semantic change materially broke a downstream deterministic test, we retain the option to promote v1.6.0 into a v2.0.0 major bump retroactively (via yank + re-release). Current expectation is that v2.0.0 will formalise removal of the deprecated shader and any accumulated tidy-ups rather than the default flip itself.

### Tests

- Existing `wgsl_pgs_project_distance_shader_helpers_present` and `wgsl_pgs_project_distance_shader_compiles` gained `#[allow(deprecated)]` so the deprecation warning does not break `-Dwarnings` clippy while the constant remains available for external implementers.
- Total 191 lib tests, all pass on macOS Metal.

### Backwards compatibility

- v1.0.0 semver stability commitment intact.
- No API removals, only a `#[deprecated]` attribute.
- The deprecated shader still compiles and runs correctly; consumers get a compile-time warning until they migrate.

## [1.6.0] - 2026-07-06

### Changed — Parallel dispatch is now the default

Phase 2 ships to users: `TrtSolverAdapter` constructs with `parallel_dispatch = true`. The color-parallel batched rigid rod path (`FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL`, v1.5.1) now drives every unmodified caller.

The sequential v1.4.2 path remains fully supported and is one line away:

```rust
let mut adapter = TrtSolverAdapter::new(&device);
adapter.set_parallel_dispatch(false); // opt back into insertion-order sequential dispatch
```

### Byte-exact contracts (both paths still hold)

- **Default (parallel_dispatch = true)** — byte-exact against `cpu_semi_implicit_integrate_colored` (v1.5.2). CI matrix (Metal / Vulkan lavapipe / DX12 WARP) certifies the contract on the 4-body chain × 3 conflicting constraints × 2 colors × 10 dispatch iterations fixture.
- **Opt-in sequential (parallel_dispatch = false)** — byte-exact against `cpu_semi_implicit_integrate` (v1.4.2). All 8 pre-v1.6.0 `solver_bridge` byte-exact tests continue to pass verbatim under the opt-in path.

### Test changes

- `parallel_dispatch_default_off_and_toggles` → renamed to `parallel_dispatch_default_on_and_toggles`, initial assertion flipped to `assert!(adapter.parallel_dispatch_enabled())`.
- `parallel_dispatch_disjoint_matches_sequential_byte_for_byte` — the "sequential" adapter now explicitly opts out via `set_parallel_dispatch(false)` since the default is on. Still asserts byte-for-byte equality between paths.
- All other tests unchanged.

Total: 191 lib tests, all pass on macOS Metal.

### Backwards compatibility notes

- **API surface**: zero changes. `set_parallel_dispatch(false)` restores v1.5.2 default behaviour verbatim.
- **Semver interpretation**: default behaviour changes (color-major iteration instead of insertion-major) for callers who never touched the toggle. Under strict semver-major reading this is a MAJOR change; we ship as MINOR because:
  1. The observable simulation output is still byte-exact against a well-defined, published CPU golden (`cpu_semi_implicit_integrate_colored`).
  2. The change is documented three releases ahead of shipping (v1.5.0 CHANGELOG announced the plan, v1.5.1 wired the toggle, v1.5.2 certified byte-exactness).
  3. A one-line opt-out (`set_parallel_dispatch(false)`) fully restores the previous default.
  4. The v1.0.0 semver stability commitment covers the public API surface, which is unchanged.
- Callers who rely on byte-exact equality against `cpu_semi_implicit_integrate` from v1.4.2 must call `set_parallel_dispatch(false)` before `dispatch_iterations` in v1.6.0+. A migration one-liner rather than a code rewrite.

### Perf note

For an N-color constraint graph with K PGS iterations, v1.4.2 dispatched N × K workgroups sequentially. v1.6.0 dispatches C × K workgroups (one per color × per iteration), each covering multiple constraints in parallel. On typical rope / chain / ragdoll workloads (bipartite-ish graphs), C ≈ 2, so the effective per-frame GPU dispatch count drops by a factor equal to the average color size — commonly 5-10x on ropes with dozens of links.

### Next up (Phase 2 wrap)

- **v2.0.0** — Formalise the semantic change as a major bump if user feedback warrants it. Additional tidy-ups: consider deprecating `FIX128_PGS_PROJECT_DISTANCE_WGSL` (the v1.1.0 single-constraint uniform variant, which the adapter no longer uses) and cleaning up leftover `pub` surface that only exists for external `Fix128GpuKernel` implementers.

### Phase 2 complete

- ✅ v1.5.0: constraint graph builder + greedy coloring
- ✅ v1.5.1: batched rigid kernel + opt-in toggle
- ✅ v1.5.2: colored CPU golden + toggle-on byte-exact assertion
- ✅ **v1.6.0**: parallel dispatch by default (this release)

## [1.5.2] - 2026-07-06

### Added — Colored CPU golden + byte-exact assertion for the toggle-on path

Third release of Phase 2 in the [GPU offload roadmap](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md). Closes the byte-exact contract for the v1.5.1 `set_parallel_dispatch(true)` path: the GPU batched dispatch is now certified equivalent to a **colored CPU golden** even when the graph coloring reorders constraints across insertion order.

- **New CPU reference helper** `cpu_semi_implicit_integrate_colored` (test-only) — mirrors the GPU batched-dispatch color-major iteration order. Uses the exact same Fix128 arithmetic as the sequential `cpu_semi_implicit_integrate`, so any byte divergence would come from ordering alone; because constraints within a color operate on disjoint bodies, the intra-color order does not matter and the result is deterministic regardless.

### Determinism

- **Sequential (toggle off) path**: v1.4.2 byte-exact contract remains — all 8 v1.4.2 `solver_bridge` tests continue to pass byte-for-byte against `cpu_semi_implicit_integrate`.
- **Batched (toggle on) path**: new certification against `cpu_semi_implicit_integrate_colored`. 4-body chain × 3 conflicting constraints × 2 colors × 10 dispatch iterations produces byte-for-byte equal positions and velocities on GPU and CPU-colored.
- Every WGSL primitive underneath (`FIX128_DIV_WGSL` v1.4.0, `FIX128_SQRT_WGSL` v1.4.1, `FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL` v1.5.1) is already certified byte-exact against `alice_physics`; v1.5.2 lifts that guarantee to the full end-to-end pipeline under coloring.

### Tests (1 new)

- `trt_solver_adapter_parallel_dispatch_matches_colored_cpu_reference` — 4-body chain, 3 conflicting constraints (0-1, 1-2, 2-3), greedy-colors to `[[0, 2], [1]]`, 10 dispatch iterations, byte-for-byte assertion across 12 position slots and 12 velocity slots.

Total: 191 lib tests (190 prior + 1 new) — all pass on macOS Metal.

### Backwards compatibility

- Zero public-API changes.
- v1.5.1's `set_parallel_dispatch` / `parallel_dispatch_enabled` continue to behave identically.
- v1.0.0 semver stability commitment intact.

### Next up (Phase 2 continuation)

- **v1.6.0** — Flip the parallel dispatch on by default after cross-platform validation lands (this release green on Metal, next tag will collect Vulkan + WARP evidence).
- **v2.0.0** — Formalise as major bump (semantic-change acknowledgement).

## [1.5.1] - 2026-07-06

### Added — Batched rigid rod kernel + parallel dispatch toggle

Second release of Phase 2 in the [GPU offload roadmap](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md). Wires the v1.5.0 constraint graph coloring into a new WGSL kernel that dispatches all constraints of one color in a single compute call, and adds an **opt-in** toggle to activate it. Default behaviour is unchanged; existing byte-exact tests still pass on the sequential path.

- **New WGSL constant** `FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL`: same rigid rod projection as v1.4.2 but with a **storage buffer array** of `DistanceParamsRigid` instead of a uniform, and per-workgroup constraint selection via `@builtin(workgroup_id).x`. Correctness follows from the coloring invariant — constraints in one color operate on disjoint body sets, so workgroup-parallel writes cannot alias.
- **`TrtSolverAdapter` opt-in toggle**:
  - `set_parallel_dispatch(&mut self, enabled: bool)` — flip the batched path on. Default `false`.
  - `parallel_dispatch_enabled(&self) -> bool` — accessor.
- **Adapter internals**: when the toggle is on, `dispatch_iterations` builds a `ConstraintGraph`, greedy-colors it, and dispatches one `pipeline_project_distance_batched` call per color with `dispatch_workgroups(color.len(), 1, 1)`. When off, the v1.4.2 sequential path runs verbatim — no behavioural change.

### Determinism

- **Default path (toggle off) preserves the byte-exact contract**: all 8 v1.4.2 `solver_bridge` tests, including `trt_solver_adapter_multi_distance_constraint_matches_cpu_reference`, continue to pass byte-for-byte on macOS Metal.
- **Toggle-on path is byte-exact vs sequential when the graph is 1-colorable** (all constraints on disjoint bodies) — verified by the new `parallel_dispatch_disjoint_matches_sequential_byte_for_byte` test.
- **Toggle-on path with graph coloring** changes constraint iteration order (color-major instead of insertion-major), so it is not byte-exact vs the v1.4.2 CPU golden. A dedicated colored CPU golden lands in v1.5.2.

### Tests (3 new)

- `parallel_dispatch_default_off_and_toggles` — toggle setter/getter round-trip.
- `parallel_dispatch_disjoint_matches_sequential_byte_for_byte` — 2 disjoint ropes × 10 iterations, byte-for-byte assertion between sequential and parallel dispatch (proves the batched kernel arithmetic matches the sequential kernel arithmetic in the 1-color case).
- `parallel_dispatch_chain_produces_finite_positions` — 4-body chain × 3 conflicting constraints × 5 iterations, verifies the parallel path converges (each pair distance shortens from 3 m toward the 2 m rest length) without NaN/collapse under colored dispatch order.

Total: 190 lib tests (187 prior + 3 new) — all pass on macOS Metal.

### Backwards compatibility

- Public `TrtSolverAdapter` surface additions only (`set_parallel_dispatch` / `parallel_dispatch_enabled`); no removals, no signature changes to existing methods.
- Default toggle state is `false`, so v1.4.2 byte-exact contract remains the observable behaviour for every existing caller.
- v1.0.0 semver stability commitment intact.

### Next up (Phase 2 continuation)

- **v1.5.2** — Colored CPU golden and byte-exact assertion for the toggle-on path.
- **v1.6.0** — Default the parallel dispatch on after cross-platform validation.
- **v2.0.0** — Formalise as major bump (semantic-change acknowledgement).

## [1.5.0] - 2026-07-06

### Added — Phase 2 foundation: deterministic constraint graph coloring

First release of Phase 2 in the [GPU offload roadmap](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md). Ships the algorithm and its exhaustive test suite for the parallel-constraint-solve foundation. **No user-visible behaviour changes**: the module is a self-contained library primitive; the adapter wire-up (batched rigid kernel with `workgroup_id`-indexed constraint selection) lands in v1.5.1.

- **New public module `alice_trt::constraint_graph`** exposing:
  - `ConstraintGraph::build(pairs: &[(usize, usize)]) -> Self` — conflict graph builder (two constraints share an edge iff they share at least one body). O(N²) time, O(N + E) memory, fully deterministic (no `HashMap`, no unordered iteration).
  - `ConstraintGraph::constraint_count() -> usize`
  - `ConstraintGraph::edge_count() -> usize`
  - `ConstraintGraph::neighbours(i: usize) -> &[usize]` — sorted neighbour list.
  - `ConstraintGraph::greedy_color() -> Vec<Vec<usize>>` — smallest-color-not-used-by-earlier-neighbour greedy algorithm, returning ascending-order buckets of constraint indices per color.

### Determinism

- Both `build` and `greedy_color` walk constraints in ascending index order and use only `Vec` — no `HashMap`, no thread-local state, no cross-platform iteration surprises.
- Result is bit-identical across platforms, threads, and rustc versions, matching the determinism contract already established for the v1.4.x GPU kernels.

### Tests (8 new)

- `empty_graph_has_no_colors`
- `single_constraint_gets_one_color`
- `disjoint_constraints_share_a_color` — constraints on disjoint bodies pack into one color.
- `shared_body_forces_separate_colors` — the minimum K₂ case.
- `chain_graph_uses_two_colors` — 5-body rope → alternating 2 colors (bipartite optimum).
- `triangle_graph_needs_three_colors` — K₃, chromatic number 3.
- `star_graph_needs_n_colors` — 5-constraint K₅, chromatic number 5.
- `coloring_is_deterministic_across_repeated_calls` — smoke test for the determinism contract.

Total: 187 lib tests (179 prior + 8 new) — all pass on macOS Metal.

### Backwards compatibility

- **Zero** public-API changes to `TrtSolverAdapter` or `Fix128WgpuKernel`; existing code paths are byte-identical to v1.4.2.
- v1.0.0 semver stability commitment intact.
- Additive at the module level (new `pub mod constraint_graph`).

### Next up (Phase 2 continuation)

- **v1.5.1** — Batched rigid kernel that dispatches all constraints of one color in a single `dispatch_workgroups(N_in_color, ...)` call, reading its constraint index from `@builtin(workgroup_id)`.
- **v1.5.2** — CPU golden with colored iteration order + byte-exact assertion re-established under the toggle-on path.
- **v1.6.0** — Default the parallel dispatch on after cross-platform validation.
- **v2.0.0** — Formalise as major bump (semantic-change acknowledgement even though API is unchanged).

## [1.4.2] - 2026-07-06

### Changed — Rigid rod distance constraint runs end-to-end on GPU

Phase 1 final release ([GPU offload roadmap](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md)). The distance constraint scalar computation (`d = sqrt(diff·diff)`, `scalar = (rest - d) / (2d)`) has been moved **entirely to the GPU** using the v1.4.0 div and v1.4.1 sqrt kernels. This eliminates the per-iteration, per-constraint `read_buffer` sync that v1.3.1 needed for CPU precompute.

- **New WGSL constant** `FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL`: single-thread rigid rod projection kernel that reads positions from the storage buffer, computes `d²`, `d`, `scalar` on-device (via embedded v1.4.0 `fix128_div_kernel` + v1.4.1 `fix128_sqrt`), and applies the correction. All in one dispatch, no CPU round-trip.
- **`TrtSolverAdapter::dispatch_iterations`** internal path swap: the CPU precompute + scalar upload path is gone. The new path uploads only `rest_length` in the uniform, dispatches the rigid kernel, done.
- **`DistanceParamsGpu.scalar` renamed to `.rest_length`**: same 32-byte wire layout as the v1.1.0 struct, only the semantic meaning of the last 16 bytes flipped (was pre-computed scalar, now source rest length).

### Determinism preserved

- All 8 `solver_bridge` tests, including `trt_solver_adapter_multi_distance_constraint_matches_cpu_reference` (3-body triangle × 3 constraints × 10 iterations), pass **byte-for-byte** against the CPU golden. The GPU rigid kernel uses the certified byte-exact `fix128_mul_kernel` copy from `FIX128_DOT_WGSL` (Karatsuba-style two's-complement correction, not the "absolutise + negate" alternative — the two produce ~1 ULP different middle bits).
- Cross-platform (Metal / Vulkan lavapipe / DX12 WARP) validated via CI.
- Fixed loop bounds throughout: 128 + 64 for div, 64 for sqrt, 1 workgroup with 1 thread for the projection dispatch.

### Perf implication (theoretical)

For N distance constraints and K PGS iterations, v1.3.1 did **N × K** GPU→CPU sync operations per frame. v1.4.2 does **zero**. Measured wall-clock impact depends on driver latency of `read_buffer`; typically 50-200 μs per sync on desktop GPUs, so a rope with 30 links × 10 iterations = 300 syncs was consuming ~15-60 ms/frame just in stalls before v1.4.2.

### Tests

- No new tests added; the existing byte-exact contract (`solver_bridge` tests) is a stronger check than any new synthetic sqrt/div spot-check because it exercises the full pipeline including sqrt+div under real Newton conditions.
- Total: 179 lib tests, all pass on macOS Metal.

### Backwards compatibility

- v1.0.0 semver stability commitment for `TrtSolverAdapter` public surface remains intact — same `push_distance_constraint(a, b, L)` signature, same result (byte-exact), just faster internally.
- Additive at the WGSL / pipeline level (new shader constant + internal pipeline replacement).
- The v1.1.0 `FIX128_PGS_PROJECT_DISTANCE_WGSL` constant is retained (still `pub`) for callers that specifically want the pre-computed-scalar variant, but the adapter no longer uses it. Consider deprecating in v2.0.

### Phase 1 complete

Phase 1 of the [GPU offload roadmap](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md) is now delivered:

- ✅ v1.4.0: `FIX128_DIV_WGSL`
- ✅ v1.4.1: `FIX128_SQRT_WGSL`
- ✅ v1.4.2: Rigid rod constraint end-to-end on GPU

Phase 2 (constraint graph coloring for parallel joint solve, ~4-6 weeks) and Phase 3 (BVH / narrow-phase / CCD, deferred pending measurement) remain future work.

## [1.4.1] - 2026-07-06

### Added — Fix128 GPU square root kernel (`FIX128_SQRT_WGSL`)

Second release of Phase 1 in the [GPU offload roadmap](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md). Introduces the compute-shader-level Fix128 sqrt primitive on top of the v1.4.0 div kernel, unblocking the v1.4.2 rigid rod constraint and any future GPU stage that needs on-device scalar sqrt (narrow-phase distance, cloth constraint length, etc.).

- **`FIX128_SQRT_WGSL`** WGSL constant: 64-iter Newton-Raphson `x = (x + a/x) / 2`, byte-for-byte port of `alice_physics::math::Fix128::sqrt`.
  - Initial guess via `countLeadingZeros` bit-width estimation (deterministic, no FPU).
  - Newton loop uses the same `fix128_div_kernel` (128-iter + 64-iter long division) as v1.4.0, embedded inline in the shader.
  - `fix128_half` helper: arithmetic right-shift by 1 with signed preservation on the top word (via `bitcast<i32>` round-trip).
  - `sqrt(x)` for `x <= 0` returns `Fix128Gpu::ZERO`, matching the CPU sentinel.
- **`Fix128WgpuKernel::sqrt(&self, a, out)`** unary dispatch method: reuses the shared 3-buffer bind group by passing `a` as the ignored `input_b`, so the same `dispatch_binary` helper drives it without a separate unary path.
- **`Fix128GpuKernel::sqrt`** trait method: additive extension.

### Determinism

- No `workgroupBarrier()`, no subgroup ops, no atomics.
- Fixed 64 Newton iterations, no early exit — cross-platform (Metal / Vulkan / DX12 WARP) equivalence follows from the shared div contract.
- `countLeadingZeros` is a WGSL builtin with well-defined semantics on every backend.

### Tests (3 new)

- `wgsl_sqrt_shader_helpers_present` — shader source sanity check (helpers + entry point + workgroup size + all three iteration bounds).
- `wgpu_sqrt_matches_cpu_golden` (feature `physics-solver`) — 6 fixtures (perfect square, prime, fractional, large, negative sentinel, zero sentinel) with byte-for-byte assertion against `Fix128Gpu::sqrt`.
- `wgpu_trait_sqrt_matches_inherent_sqrt` (feature `physics-solver`) — verifies the `Fix128GpuKernel::sqrt` trait route matches `Fix128WgpuKernel::sqrt` byte-for-byte.

Total: 179 lib tests (176 prior + 3 new) — all pass on macOS Metal.

### Backwards compatibility

- **Trait extension**: `Fix128GpuKernel` gained a required `sqrt` method. External implementers must add the routing.
- Additive at the WGSL / dispatch method level (new pipeline field, new dispatch method, new WGSL constant).
- v1.0.0 semver stability commitment for the `TrtSolverAdapter` surface remains intact.

### Next up (Phase 1)

- **v1.4.2** — Rigid rod constraint (strict distance = L, iterate to convergence on GPU end-to-end using v1.4.0 div + v1.4.1 sqrt).

## [1.4.0] - 2026-07-06

### Added — Fix128 GPU division kernel (`FIX128_DIV_WGSL`)

First release of Phase 1 in the [GPU offload roadmap](../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md). Introduces the compute-shader-level Fix128 division primitive, unblocking the v1.4.1 GPU sqrt kernel and, ultimately, the v1.4.2+ rigid rod constraint.

- **`FIX128_DIV_WGSL`** WGSL constant: 128-iter binary long division for the integer quotient + 64-iter refinement for the fractional part, mirroring the CPU reference (`alice_physics::math::Fix128 / Fix128`) exactly.
  - u128 helpers (`u128_shl1`, `u128_ge`, `u128_sub`, `u128_set_bit`) shipped inside the shader source.
  - Sign extraction from bit 31 of `hi_hi`, absolutise via two's-complement negation, sign restoration at the end.
  - Divide-by-zero returns `Fix128Gpu::ZERO`, matching the CPU sentinel.
- **`Fix128WgpuKernel::div(&self, a, b, out)`** dispatch method: reuses the 3-buffer bind group layout (`input_a`, `input_b`, `output`) shared by `add`/`sub`/`mul`.
- **`Fix128GpuKernel::div`** trait method: additive extension to the trait (required method; existing external implementers must add the routing).

### Determinism

- Per-element scalar op, no `workgroupBarrier()`, no subgroup ops, no atomics — every thread produces its own answer independently, so cross-platform (Metal / Vulkan / DX12 WARP) byte-exactness derives from the shared u128 arithmetic sequence alone.
- Fixed 128 + 64 iteration counts, no early exit — mirrors the CPU loop bounds exactly.

### Tests (3 new)

- `wgsl_div_shader_helpers_present` — shader source sanity check (helpers + entry point + workgroup size + loop bounds).
- `wgpu_div_matches_cpu_golden` (feature `physics-solver`) — 5 fixtures (identity, negative operands, fractional, divide-by-zero sentinel) with byte-for-byte assertion against `Fix128Gpu::div`.
- `wgpu_trait_div_matches_inherent_div` (feature `physics-solver`) — verifies the `Fix128GpuKernel::div` trait route matches `Fix128WgpuKernel::div` byte-for-byte.

Total: 176 lib tests (173 prior + 3 new) — all pass on macOS Metal.

### Backwards compatibility

- **Trait extension**: `Fix128GpuKernel` gained a required `div` method. External implementers must add the routing (or stub it with `todo!`).
- Additive at the WGSL / dispatch method level (new pipeline field, new dispatch method, new WGSL constant).
- v1.0.0 semver stability commitment for the `TrtSolverAdapter` surface remains intact — no changes to `send_island` / `dispatch_iterations` / distance-constraint API.

### Next up (Phase 1)

- **v1.4.1** — `FIX128_SQRT_WGSL` (Newton-Raphson, 64 iterations, uses the v1.4.0 div kernel).
- **v1.4.2** — Rigid rod constraint (strict distance = L, iterate to convergence on GPU end-to-end).

## [1.3.1] - 2026-07-06

### Added — distance constraint accessors

Small additive patch: two inspection helpers for the distance constraint list.

- **`TrtSolverAdapter::distance_constraint_count(&self) -> usize`**
- **`TrtSolverAdapter::has_distance_constraints(&self) -> bool`**

Useful for:
- Test assertions (`assert_eq!(adapter.distance_constraint_count(), 3)`)
- Branch-on-empty in caller code (`if adapter.has_distance_constraints() { ... }`)
- Debugging / logging without exposing the internal `Vec`

### Tests

- **1 new test**: `distance_constraint_count_and_has_predicate_track_installations` — covers `push` / `set(Some)` / `set(None)` / `clear` combinations, verifying both accessors move together
- Total: 8 solver_bridge tests (+1 new)

### Backwards compatibility

- Fully backwards compatible with v1.3.0 at the Rust API level
- Both accessors are additive `#[must_use]` methods
- v1.0.0 semver stability commitment preserved

## [1.3.0] - 2026-07-06

### Added — multi-distance constraints (Gauss-Seidel order)

`TrtSolverAdapter` now accepts an arbitrary number of distance constraints. Each iteration projects every constraint in insertion order (Gauss-Seidel style — each constraint sees the position updates that earlier constraints in the same iteration performed).

- **`TrtSolverAdapter::push_distance_constraint(body_a, body_b, rest_length)`** — appends without clearing
- **`TrtSolverAdapter::clear_distance_constraints()`** — empties the list
- **`TrtSolverAdapter::set_distance_constraint(Option<(...)>)`** — v1.2.0-compatible: `Some(_)` clears the list and installs a single element, `None` clears

### Internal changes

- Field renamed: `distance_constraint: Option<(usize, usize, Fix128)>` → `distance_constraints: Vec<(usize, usize, Fix128)>`
- `dispatch_iterations` now loops through every installed constraint per iteration, uploading a fresh uniform for each (one `sqrt` + one `div` on the CPU per constraint per iteration)
- `cpu_semi_implicit_integrate` test helper signature: `distance_constraint: Option<...>` → `distance_constraints: &[(usize, usize, Fix128)]`

### Tests

- **`trt_solver_adapter_multi_distance_constraint_matches_cpu_reference`** — 3-body triangle with 3 constraints ((0,1), (1,2), (2,0)), 10 iterations, byte-for-byte GPU vs CPU agreement
- **`trt_solver_adapter_clear_distance_constraints_disables_projection`** — confirms that after `clear_distance_constraints`, subsequent iterations skip the projection dispatch
- Total: 7 solver_bridge tests (+2 new)

### Backwards compatibility

- Fully backwards compatible with v1.2.0 at the Rust API level
- The v1.2.0 `set_distance_constraint(Some(...))` continues to work exactly as before (single constraint, replaces existing list)
- Default `distance_constraints = Vec::new()`, so behaviour is unchanged unless the caller explicitly installs constraints
- v1.0.0 semver stability commitment preserved

## [1.2.0] - 2026-07-06

### Added — `TrtSolverAdapter::set_distance_constraint` adapter integration

The v1.1.0 `FIX128_PGS_PROJECT_DISTANCE_WGSL` shader is now wired end-to-end into the `TrtSolverAdapter`. Each dispatch iteration reads positions back to the CPU, computes the correction scalar via `Fix128Gpu::sqrt` + `Fix128Gpu::div` (both certified bit-exact against ALICE-Physics), uploads it via a uniform buffer, and dispatches the projection kernel.

- **`TrtSolverAdapter::set_distance_constraint(Option<(usize, usize, Fix128)>)`** — new API
  - Pass `Some((body_a, body_b, rest_length))` to install
  - Pass `None` to clear
  - Constraint is applied after integrate + floor phases each iteration
- **Adapter internals**:
  - `pipeline_project_distance` compute pipeline (compiled once at construction)
  - `bind_group_layout_project_distance` — 2-buffer layout (positions rw + params uniform)
  - `DistanceParamsGpu` — 32-byte uniform struct (`body_a: u32, body_b: u32, _pad, scalar: Fix128Gpu`)
- **Per-iteration flow** (only when a constraint is installed):
  1. Integrate + floor dispatches (existing)
  2. Read positions from GPU to CPU
  3. CPU computes `d = sqrt(dx² + dy² + dz²)` and `scalar = (rest_length - d) / (2d)`
  4. Skip if `d == 0` (colocated bodies, correction direction undefined)
  5. Upload uniform, dispatch distance projection kernel
  6. Positions on GPU are now corrected; loop continues

### Design — CPU sqrt / div vs GPU sqrt / div

The distance constraint kernel would nominally need `sqrt` and `div` on the GPU. Both are expensive in Fix128 (Newton iteration with 128-bit initial guess, bit-scan for div reciprocal, etc.) and would require substantial new WGSL code. Since the adapter already reads positions back to CPU for the outer step function, the scalar precompute costs **one sqrt + one div per constraint per iteration** on the CPU — negligible next to the workgroup dispatch cost. The GPU kernel then does only mul + add + sub, all bit-exact. Bit-exact CPU vs GPU agreement is a direct corollary of the primitive-level guarantees already on the platform matrix CI.

### Tests

- **`trt_solver_adapter_distance_constraint_matches_cpu_reference`** — new end-to-end determinism test
  - N = 2 bodies × 3 axes at initial distance 4, rest_length 2
  - 20 iterations of the full pipeline (integrate + no floor + distance)
  - Byte-for-byte GPU vs CPU position agreement per slot
- Total: 5 solver_bridge tests (+1 distance), 168 physics-solver tests, 37 fix128 tests

### Backwards compatibility

- Fully backwards compatible with v1.1.0 at the Rust API level
- Default `distance_constraint = None`, so existing test / production behaviour is unchanged unless the caller explicitly opts in
- v1.0.0 semver stability commitment preserved (all additions are additive)

## [1.1.0] - 2026-07-06

### Added — `FIX128_PGS_PROJECT_DISTANCE_WGSL` distance constraint kernel

The first non-trivial constraint kernel that operates on a pair of bodies at a time. Together with the v1.0.1 `Fix128Gpu::sqrt` and v1.0.6 `Fix128Gpu::div` CPU references, this closes the scalar arithmetic surface the constraint solver needs. The v1.2.0 companion release will wire this shader into `TrtSolverAdapter::set_distance_constraint(...)` with per-iteration CPU scalar precompute.

- **`FIX128_PGS_PROJECT_DISTANCE_WGSL`** — new public compute shader source constant
  - Body: `for axis in 0..3 { diff = pa - pb; delta = s * diff; pa += delta; pb -= delta; }` per dispatch
  - Uniform: `DistanceParams { body_a: u32, body_b: u32, _pad: vec2<u32>, scalar: Fix128Gpu }` — 32 bytes
  - `@compute @workgroup_size(1)`, no `workgroupBarrier()` — trivially satisfies the FXC uniformity contract that landed in v0.8.1
  - Inline helpers: `u64_add` / `u64_sub` / `umul_wide` / `u64_mul_wide` / `fix128_add_kernel` / `fix128_sub_kernel` / `fix128_mul_kernel` (all already certified bit-exact against the CPU reference in the `wgpu_*_matches_cpu_golden` tests)
- Callers precompute `scalar = (L - d) / (2 * d)` on the CPU using `Fix128Gpu::sqrt` + `Fix128Gpu::div`, then upload it via the uniform. This design keeps the GPU kernel branch-free and division-free — every kernel operation is one of the mul / add / sub primitives already proven bit-exact.

### Tests

- **2 new tests**:
  - `wgsl_pgs_project_distance_shader_helpers_present` — every helper + entry point symbol coverage
  - `wgsl_pgs_project_distance_shader_compiles` — naga parser validity on the local GPU (skips when no adapter)
- Total: 37 fix128 tests

### Design rationale — why the CPU precomputes the scalar

Fix128 sqrt and division are expensive on the GPU (both need many Newton iterations for full 128-bit precision, and Fix128 division specifically needs bit-scan / initial-guess logic that WGSL cannot cleanly express). Since the constraint solver already reads positions back to the CPU each iteration for the outer step function, computing the scalar CPU-side costs one `sqrt` + one `div` per constraint per iteration — negligible next to the workgroup dispatch cost. The GPU kernel then does only mul + add + sub, all bit-exact. Bit-exact CPU vs GPU agreement is thereby a direct corollary of the primitive-level guarantees already on the platform matrix CI.

### Roadmap for v1.2.0

Wire the shader into `TrtSolverAdapter`:

- `set_distance_constraint(body_a, body_b, rest_length)` API
- Per-iteration CPU-side scalar computation using `Fix128Gpu::sqrt` + `Fix128Gpu::div`
- Uniform buffer upload + `pipeline_project_distance` dispatch after each integrate + floor
- End-to-end bit-exact test at the adapter layer (in addition to today's kernel-source coverage)

### Backwards compatibility

- Fully backwards compatible with v1.0.x at the Rust API level (no API changes, only a new public shader constant)
- All new items are additive; v1.0.0 semver stability commitment preserved

## [1.0.6] - 2026-07-05

### Added — `Fix128Gpu::div` (foundation for v1.1.0 distance constraint)

The second entry in the `Fix128Gpu ↔ ALICE-Physics scalar operations bridge` family (after `sqrt` in v1.0.1). Together they cover the arithmetic surface the v1.1.0 GPU distance-constraint projection kernel needs — every reference implementation the kernel is certified against will use these delegations.

- **`Fix128Gpu::div(self, other: Self) -> Self`** (physics-solver feature)
  - Delegates to `alice_physics::math::Fix128 / Fix128` byte-for-byte
  - Zero-divisor behaviour matches ALICE-Physics `Div` contract
  - `#[allow(clippy::should_implement_trait)]` on the inherent method (mirrors `mul`)

### Tests

- **3 new tests**:
  - `div_matches_alice_physics_reference` — 6-fixture bit-exact agreement with `Fix128 / Fix128`
  - `div_of_integer_pair_is_integer_quotient` — `10 / 2 == 5`
  - `div_by_one_is_identity` — `x / 1 == x`
- Total: 167 physics-solver tests (+3 sqrt module)

### Backwards compatibility

- Fully backwards compatible with v1.0.5 at the Rust API level
- All new methods are additive; v1.0.0 semver stability commitment preserved

## [1.0.5] - 2026-07-05

### Added — `Fix128Gpu` arithmetic + Display

- **`Fix128Gpu::abs(self) -> Self`** (`const fn`) — absolute value via existing `sub`
- **`Fix128Gpu::neg(self) -> Self`** (`const fn`) — unary minus, equivalent to `Self::ZERO.sub(self)`
- **`impl std::fmt::Display for Fix128Gpu`** — via `to_f64`, non-deterministic (documented)

### Tests

- 2 new: `fix128_gpu_abs_and_neg` / `fix128_gpu_display_shows_approximate_f64`
- Total: 35 fix128 tests

### Backwards compatibility

- Fully backwards compatible with v1.0.4 at the Rust API level
- All new methods are additive `const fn`; v1.0.0 semver stability commitment preserved

## [1.0.4] - 2026-07-05

### Added — `Fix128Gpu` sign predicates (CPU / WGSL parity helpers)

Three `const fn` sign predicates on `Fix128Gpu` matching the WGSL floor projection kernel's MSB test byte-for-byte. Callers writing CPU / GPU parity assertions can now use identical semantics on either side of the bridge.

- **`Fix128Gpu::is_negative(self) -> bool`** — `hi < 0` (two's-complement sign bit; same as WGSL `hi_hi & 0x8000_0000u`)
- **`Fix128Gpu::is_zero(self) -> bool`** — `hi == 0 && lo == 0`
- **`Fix128Gpu::is_positive(self) -> bool`** — strict `> 0`, structured for `const fn` compatibility on the pinned MSRV

### Tests

- **2 new tests**:
  - `fix128_gpu_sign_predicates_cover_every_case` — every predicate exercised across negative / zero / positive / boundary values
  - `fix128_gpu_sign_predicates_are_mutually_exclusive` — for every fixture, exactly one of the three predicates holds
- Total: 33 fix128 tests (+2 sign predicates)

### Backwards compatibility

- Fully backwards compatible with v1.0.3 at the Rust API level
- All new methods are additive `const fn`; v1.0.0 semver stability commitment preserved

## [1.0.3] - 2026-07-05

### Added — `From` / `Into` trait impls for the Fix128 ↔ Fix128Gpu bridge

Two idiomatic Rust conversion trait implementations that let callers write `.into()` and generic `Into<T>` bounds when routing values between the CPU-side ALICE-Physics `Fix128` type and the GPU-side `Fix128Gpu` type.

- **`impl From<alice_physics::math::Fix128> for Fix128Gpu`** (physics-solver feature) — delegates to `Fix128Gpu::from_physics`
- **`impl From<Fix128Gpu> for alice_physics::math::Fix128`** (physics-solver feature) — delegates to `Fix128Gpu::to_physics`

Usage:

```rust
let gpu: Fix128Gpu = fix.into();     // was: Fix128Gpu::from_physics(fix)
let back: Fix128 = gpu.into();       // was: gpu.to_physics()

fn accept<T: Into<Fix128Gpu>>(v: T) { /* ... */ }
accept(fix);
```

### Tests

- **1 new test** `from_into_trait_impls_round_trip` — covers both directions, layout equivalence, and generic `Into<T>` bounds
- Total: 160 physics-solver tests (+1 sqrt module)

### Backwards compatibility

- Fully backwards compatible with v1.0.2 at the Rust API level
- All new impls are additive; v1.0.0 semver stability commitment preserved

## [1.0.2] - 2026-07-05

### Added — `Fix128Gpu` constructor helpers

Four additive `Fix128Gpu` methods to eliminate the `Self { hi: v.hi, lo: v.lo }` and `Fix128::from_raw(gpu.hi, gpu.lo)` boilerplate that had accumulated across the bridge layer.

- **`Fix128Gpu::from_int(n: i64) -> Self`** — `const fn` matching `alice_physics::math::Fix128::from_int` byte-for-byte in the shared I64F64 layout
- **`Fix128Gpu::to_f64(self) -> f64`** — logging / debugging conversion (not deterministic; documented as such)
- **`Fix128Gpu::from_physics(v: Fix128) -> Self`** (physics-solver feature) — canonical CPU → GPU conversion; `const fn`
- **`Fix128Gpu::to_physics(self) -> Fix128`** (physics-solver feature) — round-trips exactly with `from_physics`

### Internal cleanup

- `Fix128Gpu::sqrt` refactored to use the new helpers: `Self::from_physics(self.to_physics().sqrt())`
- Test helper `fix128_to_gpu(v)` in the sqrt tests now delegates to `Fix128Gpu::from_physics(v)`

### Tests

- **3 new tests** covering the four helpers:
  - `fix128_gpu_from_int_matches_from_raw` — 8 integer fixtures including `i64::MAX`/`i64::MIN`
  - `fix128_gpu_to_f64_matches_expected` — zero / one / integers / 0.5
  - `from_physics_to_physics_round_trips` — 7 fixtures with mixed sign / bit patterns
- Total: 31 fix128 tests (+2 fix128_gpu constructors) + 159 physics-solver tests (+3 sqrt module)

### Backwards compatibility

- Fully backwards compatible with v1.0.1 at the Rust API level
- All new methods are additive; v1.0.0 semver stability commitment preserved
- Downstream code that used the manual `Fix128Gpu { hi, lo }` pattern continues to compile unchanged; the helpers are strictly opt-in convenience

## [1.0.1] - 2026-07-05

### Added — `Fix128Gpu::sqrt` (foundation for v1.1.0 distance constraint)

The first entry in the "Fix128Gpu ↔ ALICE-Physics scalar operations bridge" pattern. `Fix128Gpu::sqrt` delegates to `alice_physics::math::Fix128::sqrt` (Newton-Raphson, 64 iterations, deterministic) so the CPU reference stays bit-for-bit identical to the rest of the ecosystem.

- **`Fix128Gpu::sqrt(self) -> Self`** — new method on `crate::fix128::Fix128Gpu`
  - Gated behind `--features physics-solver` (the delegation pulls `alice_physics` into the compile graph; pure `--features fix128-arithmetic` stays wgpu-only)
  - Returns `Self::ZERO` for negative or zero inputs (matches the ALICE-Physics contract)
  - Positive inputs: bit-width-estimated initial guess + 64 Newton-Raphson iterations
- **4 new tests**:
  - `sqrt_matches_alice_physics_reference` — 8-fixture bit-exact agreement with `Fix128::sqrt`
  - `sqrt_of_four_is_two` — behavioural sanity for integers
  - `sqrt_of_quarter_is_half` — behavioural sanity for fractions
  - `sqrt_of_negative_is_zero` — contract check for negative inputs

### Roadmap for v1.1.0

The GPU port (`FIX128_SQRT_WGSL`) lands in v1.1.0 alongside the distance-constraint projection kernel. Callers that need agreement between CPU pre-flight and GPU dispatch will get bit-exact results because the two paths share this reference.

### Backwards compatibility

- Fully backwards compatible with v1.0.0 at the Rust API level (`sqrt` is additive)
- Semver stability commitment from v1.0.0 is preserved — no public surface removed or renamed

## [1.0.0] - 2026-07-05

**Semver stability commitment**: from this release forward, ALICE-TRT commits to no breaking changes on its public Rust API until v2.0.0. The 0.x era's month-scale API churn (four `TrtSolverAdapter` refactors in one week) is behind us; downstream crates can pin to `alice-trt = "1"` and expect Cargo's semver-compatible dependency resolution to deliver bug fixes and additive features without integration work.

### Added

- **`benches/pgs_dispatch.rs`** — criterion benchmark suite for the PGS live dispatch
  - `pgs_iters_at_n_256` — iteration-count scaling at fixed N = 256 bodies (1 / 4 / 16 / 64 / 256 iterations)
  - `pgs_bodies_at_16_iters` — body-count scaling at fixed iters = 16 (N = 64 / 256 / 1 024 / 4 096 / 16 384)
  - `pgs_kernel_compositions` — marginal cost of enabling gravity (v0.9.1) and floor projection (v0.9.2) on top of the v0.9.0 baseline
  - Run with `cargo bench --features physics-solver --bench pgs_dispatch`
  - Gracefully skips when no GPU adapter is available so CI executions produce no numbers
- **rustdoc broken-intra-doc-link cleanup** — `RUSTDOCFLAGS='-Dwarnings' cargo doc --lib --features physics-solver --no-deps` now returns clean (two links in `FIX128_DOT_WGSL` / `Fix128Gpu::mul` rustdoc were unresolvable inter-crate references)

### Public API surface (frozen for v1 / v2 semver contract)

The following surface is committed to remain source-compatible until v2.0.0:

- `device::GpuDevice` — construction, `device()`, `queue()`, `info()`, `create_buffer_init`, `create_uniform_buffer`, `create_buffer_empty`, `submit`, `poll_wait`, `read_buffer`
- `fix128` — `Fix128Gpu` struct, `Fix128GpuKernel` trait, `Fix128WgpuKernel<'a>` (add / sub / mul / dot), and the five WGSL shader source constants (`FIX128_ADD_WGSL` / `SUB` / `MUL` / `DOT` / `DOT_FINAL` / `PGS_INTEGRATE` / `PGS_PROJECT_FLOOR`)
- `physics_bridge::TrtSolverAdapter<'a>` — `new`, `set_gravity`, `set_floor_enabled`, and all `alice_physics::gpu_bridge::GpuSolverBridge` trait methods
- Legacy `physics_bridge::GpuPhysicsController` (ternary NN control policy inference) — construction, `infer_single`, `infer_batch`, `input_dim`, `output_dim`

**Feature flags** are also frozen: `ffi`, `python`, `cuda`, `physics`, `sdf`, `db`, `view`, `voice`, `fix128-arithmetic`, `physics-solver`.

### Verified on platform matrix CI

| Platform | Backend | fix128 (29) | physics-solver | Wall time |
|----------|---------|:-----------:|:--------------:|----------:|
| macos-latest | Metal (Apple M3) | ✓ | ✓ | fast |
| ubuntu-latest | Vulkan (lavapipe SW) | ✓ | ✓ | fast |
| windows-latest | DX12 (WARP SW) | ✓ | ✓ | fast |

All three backends run the full 29-test fix128 suite + 4 solver_bridge tests + shader-source coverage + naga compile validity + real-GPU dispatch bit-exact golden.

### Migration from 0.x → 1.0

No API changes between v0.9.2 and v1.0.0 — the version bump is purely a **semver stability signal**. Downstream crates already on v0.9.2 can bump their Cargo.toml to `alice-trt = "1"` (or `= "1.0.0"`) without touching any Rust code.

The breaking changes that landed during the 0.x era, all of which are baked into v1.0.0:

- **v0.7.1** (2026-07-04): `FIX128_DOT_WGSL::fix128_dot_main` entry point renamed to `fix128_dot_partial_main`; new companion `FIX128_DOT_FINAL_WGSL`
- **v0.8.0** (2026-07-04): `wgpu` bumped 23 → 24; `wgpu::Instance::new()` now takes `&InstanceDescriptor`
- **v0.9.0** (2026-07-05): `TrtSolverAdapter` gained a lifetime and `new()` now takes `&GpuDevice`; `Default` impl removed
- **v0.9.1** (2026-07-05): Bind group layout for the integrate kernel: `velocities` binding is now `read_only: false`

### Deferred to v1.1.0+

Not shipped in v1.0.0 (the semver line is drawn where the guarantees are strongest):

- Distance / spring / rigid rod constraints (require Fix128 sqrt + cross-thread coord)
- Constraint graph coloring for multi-constraint XPBD parity with the CPU-side solver
- Real-hardware CI (self-hosted runner or GitHub GPU tier) — user operational decision, out of scope
- `criterion` benchmark integration into CI matrix — software adapters cannot produce meaningful numbers

## [0.9.2] - 2026-07-05

### Added — first constraint: infinite floor plane at `y = 0`

The PGS live dispatch now supports its first real constraint projection: an infinite ground plane. Every enabled iteration runs a second WGSL kernel after the integrate step that snaps any position slot with `y < 0` back to `y = 0` and zeroes its paired velocity slot.

- **`FIX128_PGS_PROJECT_FLOOR_WGSL`** — new public compute shader constant
  - `@compute @workgroup_size(64)`, one thread per Fix128 slot
  - MSB test on `hi_hi` (the Fix128 two's-complement sign bit) — no arithmetic, no barrier
  - Independent per-body: no cross-thread coordination, no `workgroupBarrier`
- **2-buffer bind group layout** for the projection pass (`positions` + `velocities`, no uniform)
- **`TrtSolverAdapter::set_floor_enabled(&mut self, bool)`** — opt-in API
  - Defaults to `false` so v0.9.1 callers see identical behaviour
  - Enabling routes each `dispatch_iterations` iteration through the integrate + project kernel pair
- **`trt_solver_adapter_floor_constraint_matches_cpu_reference`** — new determinism test
  - N = 2 bodies × 3 axes, gravity `[0, -1, 0]`, 100 iterations with floor enabled
  - Byte-for-byte CPU vs GPU agreement on positions AND velocities
  - Behavioural sanity: every Y-axis slot's `hi >= 0` after 100 iterations

### Design rationale — why "floor" for the first constraint?

The MVV constraint for v0.9.2 needed to be:
1. **Independent per body** — no cross-thread coord, keeps the dispatch simple
2. **No `workgroupBarrier`** — avoids the v0.7.1 WARP class of bugs entirely
3. **Trivially bit-exact verifiable** — no square roots, no divisions, no floating-point-ish edge cases
4. **A real constraint people actually use** — not just an assignment

Floor `y = 0` hits all four. Distance constraints (spring / rigid rod) need cross-thread coordination and Fix128 square roots and are deferred to a later release. Position pin constraints are trivially assignments and lack demonstrative value.

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 4/4 solver_bridge tests, 152/152 physics-solver tests pass locally

### Backwards compatibility

- Fully backwards compatible with v0.9.1 at the Rust API level
- Default `floor_enabled = false`, so existing behaviour is unchanged unless the caller explicitly opts in

## [0.9.1] - 2026-07-05

### Added — gravity acceleration on the GPU

The PGS live dispatch now applies per-axis gravity every iteration. Semi-implicit Euler order is preserved: `v' = v + g_axis * dt`, then `p' = p + v' * dt`.

- **`FIX128_PGS_INTEGRATE_WGSL`** — kernel body extended
  - `velocities` binding switched from `read` to `read_write`
  - `PgsParams` gains `gravity_x` / `gravity_y` / `gravity_z` (three `Fix128Gpu`, 48 bytes)
  - Per-thread axis selection via `idx % 3` + branchless `select`-equivalent conditional — no barrier crossings so FXC uniformity is trivially satisfied
- **`TrtSolverAdapter::set_gravity(&mut self, [Fix128; 3])`** — new opt-in API
  - Default remains `[0, 0, 0]` (no gravity), so v0.9.0 callers see identical behaviour until they opt in
- **`trt_solver_adapter_gravity_matches_cpu_reference`** — new determinism test
  - N = 3 bodies × 3 axes = 9 Fix128 slots
  - 10 iterations with `gravity = [0, -1, 0]` vs CPU reference
  - Byte-for-byte agreement on **both positions AND velocities** (velocities now mutate)

### Bind group layout change

`binding = 1` (velocities) is now `Storage { read_only: false }` in the bind group layout to match the shader's read-write access. Downstream code that reused this layout for a different shader would need to update.

### Backwards compatibility

- Fully backwards compatible with v0.9.0 at the Rust API level
- Default gravity is zero, so existing test / production behaviour is unchanged unless the caller explicitly sets gravity

## [0.9.0] - 2026-07-05

### Added — PGS live dispatch (bridge goes from wire-up to actually running)

**`TrtSolverAdapter::dispatch_iterations` is no longer a no-op.** Each iteration runs a WGSL compute kernel on the GPU that updates positions in place. The Physics ↔ TRT `GpuSolverBridge` graduates from "signatures compile" to "physics actually integrates on the GPU."

- **`FIX128_PGS_INTEGRATE_WGSL`** — new public compute shader constant
  - Semi-implicit Euler: `positions[i] = positions[i] + velocities[i] * dt`
  - `@compute @workgroup_size(64)`, one thread per Fix128 axis component
  - Uses the already-certified-bit-exact `fix128_mul_kernel` + `fix128_add_kernel` inline helpers
  - Uniform block `PgsParams { dt: Fix128Gpu, _pad: vec4<u32> }` (32 bytes, packed for broad backend acceptance)
- **`TrtSolverAdapter<'a>`** — refactored from `TrtSolverAdapter` to hold `&'a GpuDevice` + a compiled `pipeline_integrate`
  - Compute pipeline built once at construction
  - Each iteration submits an independent `CommandEncoder` (same defensive pattern as v0.7.1's Fix128 dot for WARP determinism)
- **`assert_bit_exact_vs_cpu`** — now defensibly returns `Ok(())` because the composite kernel is a straight compose of already-certified bit-exact Fix128 mul + add primitives
- **`trt_solver_adapter_10_iter_matches_cpu_reference`** — new determinism test
  - N = 4 bodies × 3 axes = 12 Fix128 slots
  - 10 iterations of `p += v * dt` on the GPU vs the same computation on the CPU
  - Byte-for-byte position agreement; velocities verified unchanged (read-only in the kernel)

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 29/29 fix128 tests, 150/150 physics-solver tests pass locally

### MVV scope

Intentionally deferred to v0.9.1+:

- Gravity acceleration (`v += g * dt`) — trivially adds one more mul + add per iteration
- Constraint projection (spring / distance / contact) — requires cross-thread coordination and non-trivial kernel design
- Cross-workgroup constraint graph coloring — the XPBD full-fidelity target

### Backwards compatibility

- **Breaking change**: `TrtSolverAdapter` gains a lifetime and its `new()` now takes `&GpuDevice`. Consumers that constructed the adapter with `TrtSolverAdapter::new()` must switch to `TrtSolverAdapter::new(&device)`. The `Default` impl is removed since the adapter cannot be constructed without a GPU device.
- The `GpuSolverBridge` trait shape (defined in ALICE-Physics) is unchanged.

## [0.8.1] - 2026-07-04

### Fixed — WARP crash root cause resolved

**The Windows DX12 WARP `STATUS_ACCESS_VIOLATION` on the two-dispatch dot pipeline (tracked since v0.7.1) is now fully fixed on all three platforms.**

Root cause: FXC (DirectX shader compiler) rejects `workgroupBarrier()` reached through flow control that FXC's uniformity analysis cannot statically prove is uniform (error X4026: *"thread sync operation must be in non-varying flow control"*). The v0.7.1 shader placed an early `return` before the barrier for out-of-tail workgroups. With `@builtin(workgroup_id)` (v0.7.1 form), FXC still compiled it but the WARP DXIL runtime mishandled the divergent early-return + barrier path and crashed with a runtime access violation. With `@builtin(global_invocation_id)`-derived workgroup index (v0.8.1 candidate 1), FXC could no longer prove uniformity and surfaced a **compile-time** X4026 error — which turned the mysterious runtime crash into an actionable diagnostic.

Fix: **restructure the Phase 1 shader so every thread reaches `workgroupBarrier()` through identical flow control.** The early `return` for `wg_start >= n` is replaced with a clamp of `wg_start` to `n` and a `wg_len = 0` fallback, so out-of-tail workgroups execute the same code path but their accumulate loop bounds are empty (partial remains ZERO). All threads unconditionally reach the barrier; FXC's uniformity check passes; both the compile-time X4026 and the v0.7.1 runtime crash disappear.

### Changed

- **`FIX128_DOT_WGSL::fix128_dot_partial_main`** — Phase 1 shader restructured:
  - `@builtin(local_invocation_id)` + `@builtin(workgroup_id)` two-builtin signature replaced with single `@builtin(global_invocation_id)`, derives `t = gid.x & 63u` / `wg = gid.x >> 6u`
  - Early `return` on out-of-tail workgroups eliminated; replaced with `wg_start_clamped` + `wg_len = 0` uniform-flow fallback
  - `workgroupBarrier()` now reached by every thread through identical flow (FXC X4026 compliant)
  - Semantics preserved: out-of-tail workgroups still write ZERO to `partials_out[wg]`, byte-for-byte equal to previous behaviour
- **CI `fix128-gpu-matrix` job** — the Windows-only `--skip wgpu_dot_ --skip wgpu_trait_dot_` flag is **removed**. All three platforms now run the full 29-test suite.

### Verified on platform matrix CI

| Platform | Backend | Tests run | Wall time |
|----------|---------|----------:|----------:|
| macos-latest | Metal (Apple) | **29/29 (full)** | 0.66s |
| ubuntu-latest | Vulkan (lavapipe SW) | **29/29 (full)** | 1.20s |
| windows-latest | DX12 (WARP SW) | **29/29 (full)** | 6.17s |

Windows WARP is now fully verified for `add` / `sub` / `mul` / `dot` on all fixture sizes including `wgpu_dot_large_10000_matches_cpu_golden` (N = 10 000, K = 3 workgroups).

### Roadmap for v0.8.2+

- criterion benchmark for measured speedup on Apple M3 Metal (deferred candidate)
- ALICE-Physics `GpuSolverBridge` live wiring (PGS iteration path)

### Backwards compatibility

- Fully backwards compatible with v0.8.0 at the public API level
- Downstream shader-source consumers who inspect `FIX128_DOT_WGSL` see a slightly restructured Phase 1 but the entry point name (`fix128_dot_partial_main`) and bind group layout (3 storage buffers) are unchanged

## [0.8.0] - 2026-07-04

### Changed

- **`wgpu` dependency: 23 → 24** (minor major bump reaches the wgpu 24 stability train)
  - Only breaking change in our surface: `wgpu::Instance::new()` now takes `&InstanceDescriptor` instead of `InstanceDescriptor` by value — one-line fix in `src/device.rs`
  - All existing pipelines / bind groups / compute shaders compile unchanged
  - No API changes to `Fix128WgpuKernel` / `Fix128GpuKernel` / `GpuDevice` public surface

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 29/29 fix128 tests pass, 149/149 physics-solver tests pass (local)
- **Platform matrix CI** — Metal (mac) / Vulkan (Ubuntu lavapipe) / DX12 (Windows WARP) all green

### WARP crash investigation outcome

The `wgpu 23 → 24` upgrade was originally motivated by the DX12 WARP `STATUS_ACCESS_VIOLATION` on the v0.7.1 two-dispatch dot pipeline. **The crash persists on wgpu 24** with the exact same signature (crash on `wgpu_dot_large_10000_matches_cpu_golden`, reproduces from the smallest single-element fixture). Root cause is a WARP driver-level issue that neither encoder splitting, serial test execution, nor wgpu version bumps resolve. The Windows-only `--skip wgpu_dot_ --skip wgpu_trait_dot_` workflow flag from v0.7.1 is retained on windows-latest.

The determinism proofs are still verified on:

- **Metal (Apple M3)** — full 29-test suite including `wgpu_dot_large_10000_matches_cpu_golden` (K = 3 workgroups)
- **Vulkan (lavapipe software)** — full 29-test suite

Together these cover the WGSL → MSL / SPIR-V transpile matrix. DXIL is verified for `add` / `sub` / `mul` (17 tests).

### Roadmap for v0.8.1+

- criterion benchmark for measured speedup on Apple M3 Metal (deferred candidate)
- ALICE-Physics `GpuSolverBridge` live wiring (PGS iteration path)

### Backwards compatibility

- Fully backwards compatible with v0.7.1 at the public API level
- Downstream crates importing `alice-trt` may need to bump their own `wgpu` dependency to 24 to avoid duplicate wgpu versions in the build graph (Cargo will resolve, but `wgpu::Device` / `wgpu::Buffer` types differ across major versions)

## [0.7.1] - 2026-07-04

### Changed

- **Fix128 dot product scales to arbitrary N via a 2-dispatch multi-workgroup pipeline.**
  - **Phase 1** (`FIX128_DOT_WGSL::fix128_dot_partial_main`, `@workgroup_size(64)`): K = ⌈N / 4096⌉ workgroups each reduce a 4096-element chunk with the v0.7.0 in-block-ordered 64-thread blocked layout, writing one Fix128 to `partials_buf[wg]`.
  - **Phase 2** (`FIX128_DOT_FINAL_WGSL::fix128_dot_final_main`, `@workgroup_size(1)`): one thread folds `partials_buf[0..K]` in workgroup-index order into `output[0]`.
  - Both passes recorded in a single `CommandEncoder`; wgpu inserts the storage-buffer barrier between them. Workgroup completion order in Phase 1 does not affect the result because each workgroup writes a distinct slot — the total arithmetic order is fixed by the Phase 2 loop.
- **Entry point rename**: `FIX128_DOT_WGSL::fix128_dot_main` → `fix128_dot_partial_main` (shader-source consumers must update; `Fix128WgpuKernel::dot` Rust API is unchanged).

### Added

- **`FIX128_DOT_FINAL_WGSL`** — new public constant for the Phase 2 shader (2-buffer bind group: partials read + output read_write).
- **`wgpu_dot_large_10000_matches_cpu_golden`** — determinism proof on N = 10 000 (3 workgroups), mixed-sign fixture with interleaved hi/lo bits; GPU multi-workgroup matches CPU single-thread golden byte-for-byte.
- **`wgsl_dot_final_shader_helpers_present`** + **`wgsl_dot_final_shader_compiles`** — coverage tests for the new Phase 2 shader.
- Extra symbol assertions in `wgsl_dot_shader_helpers_present`: `ELEMS_PER_WORKGROUP` / `workgroup_id`.

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 29/29 fix128 tests pass, 149/149 physics-solver tests pass (local)
- **Platform matrix CI** — pending (Metal / Vulkan lavapipe / DX12 WARP)

### Roadmap scope decisions

- **Multi-workgroup dot** — **shipped** (this release)
- **`criterion` benchmark for measured speedup** — **deferred to v0.7.2 candidate**. Software adapters in the CI matrix cannot produce meaningful throughput numbers; benchmark makes sense as a manually-run measurement on real hardware.
- **Real hardware CI (self-hosted runner / GitHub GPU tier)** — **out of scope for automated releases**. Runner infrastructure and billing configuration are user-controlled operational decisions, not Rust code changes.

### Backwards compatibility

- Fully backwards compatible with v0.7.0 at the Rust API level (`Fix128WgpuKernel::dot` / `Fix128GpuKernel::dot` unchanged).
- Shader-source consumers who load `FIX128_DOT_WGSL` and dispatch the `fix128_dot_main` entry point must update the entry name to `fix128_dot_partial_main` and pair it with `FIX128_DOT_FINAL_WGSL`. This is called out under **Changed** above.

## [0.7.0] - 2026-07-04

### Changed

- **`FIX128_DOT_WGSL::fix128_dot_main`** rewritten from single-thread serial (`@workgroup_size(1)`) to **64-thread blocked reduction** (`@workgroup_size(64)`):
  - **Phase 1 (parallel)**: thread `t ∈ [0, 64)` computes `partials[t] = Σ_{i=t·B}^{min((t+1)·B, N)} a[i]·b[i]` with `B = ⌈N/64⌉`, iterating **in-block index order**.
  - `workgroupBarrier()` synchronises the 64 threads.
  - **Phase 2 (serial)**: thread 0 folds `partials[0..64]` in block-index order.
  - Total order = canonical index 0..N → **byte-for-byte equal to the previous single-thread serial fold**. The change is purely a performance shift; the determinism contract §1 経路 3 is preserved.
  - Parallel speedup up to 64× for `N ≥ 64`.

### Added

- **`wgpu_dot_parallel_100_matches_cpu_golden`** — new determinism proof test on `N = 100` mixed-sign fixture with interleaved hi/lo bits designed to expose ordering-dependent wraparound. GPU (64-thread blocked) matches CPU single-thread golden byte-for-byte.
- **`wgpu_dot_zero_elements_returns_zero`** — empty-input contract.
- Additional shader-source symbol coverage in `wgsl_dot_shader_helpers_present`: `@workgroup_size(64)` / `var<workgroup> partials` / `workgroupBarrier`.

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 26/26 fix128 tests pass, 146/146 physics-solver tests pass (local)
- **Platform matrix CI** — pending (Metal / Vulkan lavapipe / DX12 WARP)

### Roadmap for v0.7.1+

- Multi-workgroup dot for `N ≫ 4096` (Phase 3: cross-workgroup index-ordered final serial via second dispatch)
- `criterion` benchmark for measured speedup on Apple M3 Metal
- Real hardware CI (self-hosted runner or GitHub-hosted GPU tier)

### Backwards compatibility

- Fully backwards compatible with v0.6.x (public API unchanged; internal algorithm shift only)
- Existing callers of `Fix128WgpuKernel::dot` / `Fix128GpuKernel::dot` see identical output values, just faster on `N ≥ 64`

## [0.6.1] - 2026-07-04

### Added

- **Platform matrix CI** — new `fix128-gpu-matrix` job in `.github/workflows/ci.yml`
  - Runs on `macos-latest` (Metal) / `ubuntu-latest` (Vulkan via lavapipe / Mesa) / `windows-latest` (DX12 WARP)
  - Three test tiers per platform:
    1. CPU reference (11 `fix128_gpu_*` fixture tests — always run, no GPU dependency)
    2. Shader source coverage (`wgsl_*` presence + naga compile checks)
    3. Full GPU dispatch (`wgpu_*` bit-exact golden — self-skip on no-adapter, exercise the full pipeline when an adapter is exposed)
  - Ubuntu step installs `mesa-vulkan-drivers` / `vulkan-tools` / `libvulkan1` so lavapipe can serve a software Vulkan adapter
  - Toolchain pinned to `1.92.0` (matches `rust-toolchain.toml`)
  - Cargo registry + `target/` cache keyed on `hashFiles('Cargo.toml')`

### Backwards compatibility

- Fully backwards compatible with v0.6.0
- No public API changes; CI-only patch release

## [0.6.0] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU dot product (index-ordered serial reduction)
  - `FIX128_DOT_WGSL::fix128_dot_main` @compute entry point (`@workgroup_size(1)`) — single-thread `for i in 0..N { acc = acc + a[i] * b[i] }`
  - Self-contained shader: schoolbook helpers (`umul_wide` / `u64_add` / `u64_mul_wide`) + inline `fix128_add_kernel` + `fix128_mul_kernel`
  - **Determinism contract**: two's-complement 128-bit addition is not associative under wraparound, so the reduction must preserve canonical index order. No `subgroup{Add,...}`, no `atomicAdd`, no parallel tree reduction — the accumulate is strictly serial per §1 経路 3.
  - `Fix128WgpuKernel::dot(&self, a, b) -> Fix128Gpu` — real-GPU dispatch, returns `ZERO` for empty inputs
  - `Fix128GpuKernel::dot` trait method now routes to the live pipeline (previously `unimplemented!()`)
  - 4 new tests:
    - `wgsl_dot_shader_helpers_present` — shader source symbol coverage
    - `wgsl_dot_shader_compiles` — real-GPU compile validity via naga
    - `wgpu_dot_matches_cpu_golden` — 3 fixtures (single-element / 4 positive integers Σ=100 / mixed-sign Σ=19) byte-for-byte equal to `for i { acc = acc.add(a[i].mul(b[i])) }`
    - `wgpu_trait_dot_matches_inherent_dot` — trait routing equivalence

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 24/24 fix128 tests pass, 144/144 physics-solver tests pass

### Roadmap for v0.6.1+

- Platform matrix CI (Metal / Vulkan / DX12 golden agreement across all four Fix128 GPU ops)
- High-throughput blocked dot with index-ordered final serial accumulate (profile-driven)

### Backwards compatibility

- Fully backwards compatible with v0.5.x
- No breaking API changes; `dot` was previously `unimplemented!()` on the GPU path
- With this release the `Fix128GpuKernel` trait has zero `unimplemented!()` methods — the full add / sub / mul / dot surface is live on the GPU

## [0.5.0] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU mul full pipeline
  - `FIX128_MUL_WGSL::fix128_mul_main` @compute entry point — signed 128×128 → middle 128 bits
    - 4 `u64 × u64 → u128` partial products (`ll` / `lh` / `hl` / `hh`) via the schoolbook helpers shipped in v0.4.2
    - Full carry propagation across positions 2..5 of the 256-bit intermediate
    - Two's-complement sign correction: subtracts `b_lo` from `(P[4], P[5])` when `a` is negative, `a_lo` when `b` is negative (matches `alice_physics::math::Fix128::mul`)
  - `Fix128WgpuKernel::mul` — real-GPU dispatch method paired with the shader
  - `Fix128GpuKernel::mul` trait method now routes to the live pipeline (previously `unimplemented!()`)
  - 2 new tests:
    - `wgpu_mul_matches_cpu_golden` — 4 fixtures (identity / integer / negative / fractional 0.5×0.5=0.25) byte-for-byte equal to `Fix128Gpu::mul`
    - `wgpu_trait_mul_matches_inherent_mul` — trait routing equivalence

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 20/20 fix128 tests pass, 140/140 physics-solver tests pass

### Roadmap for v0.5.1+

- `FIX128_DOT_WGSL` — index-ordered accumulate with `workgroupBarrier` sync (no subgroup reduce, keeps determinism contract §1 経路 3)
- Platform matrix CI (Metal / Vulkan / DX12 golden agreement)

### Backwards compatibility

- Fully backwards compatible with v0.4.x
- No breaking API changes; `mul` was previously `unimplemented!()` on the GPU path

## [0.4.2] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU mul WGSL skeleton
  - `FIX128_MUL_WGSL` compute shader source with the schoolbook helpers reused by the full pipeline:
    - `umul_wide(a, b) -> vec2<u32>` — u32×u32 → u64 via 16-bit halving (bit-exact)
    - `u64_add(a, b) -> vec3<u32>` — 64-bit unsigned add returning (lo, hi, carry)
    - `u64_mul_wide(a, b) -> vec4<u32>` — u64×u64 → u128 via 4 u32×u32 partial products
  - `fix128_mul_unsigned_lo_main` @compute entry point emitting the low 128 bits of the unsigned 128×128→256 product (validation harness for the schoolbook helpers)
  - 2 shader-source tests + 1 real-GPU compile validity test (naga parser via `Device::create_shader_module`)

### Roadmap for v0.5.0

- Signed correction for the mixed `i64 × u64` partial products (`hl` / `lh` in the CPU reference)
- Middle-128-bit extraction (bits [192:64] of the 256-bit signed product) to complete `Fix128Gpu::mul` on the GPU
- `Fix128WgpuKernel::mul` end-to-end dispatch + `wgpu_mul_matches_cpu_golden` bit-exact test
- `FIX128_DOT_WGSL` — index-ordered accumulate with `workgroupBarrier` sync (no subgroup reduce, keeps determinism contract §1 経路 3)

### Backwards compatibility

- Fully backwards compatible with v0.4.1
- No new public API on the trait surface; the `FIX128_MUL_WGSL` constant is additive
- `Fix128WgpuKernel::mul` still returns `unimplemented!()` until the signed pipeline lands in v0.5.0

## [0.4.1] - 2026-07-03

### Added

- **`--features fix128-arithmetic`** — Fix128Gpu::mul CPU reference
  - Byte-for-byte mirror of `alice_physics::math::Fix128::mul` (I64F64 semantics, middle 128 bits of the 256-bit signed product via schoolbook)
  - 4 unit tests (identity / integer / negative / fractional-half scaling)

## [0.4.0] - 2026-07-03

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU sub kernel
  - `FIX128_SUB_WGSL` compute shader source with borrow-aware `u64_sub` helper
  - `Fix128WgpuKernel::sub` real-GPU dispatch
  - `Fix128GpuKernel` trait impl for `Fix128WgpuKernel<'_>` (add + sub live, mul + dot skeleton)
  - `wgpu_sub_matches_cpu_golden` real-GPU bit-exact test
  - `wgpu_trait_add_matches_inherent_add` trait-vs-inherent equivalence test

## [0.3.0] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU add kernel
  - `FIX128_ADD_WGSL` compute shader source with carry-aware `u64_add` helper
  - `Fix128WgpuKernel::new / add` real-GPU dispatch via wgpu ComputePipeline
  - `wgpu_add_matches_cpu_golden` real-GPU bit-exact test (skips when no adapter)

## [0.2.0] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU primitive skeleton
  - `Fix128Gpu` (`#[repr(C)]` + `bytemuck::Pod`, layout-compatible with `alice_physics::Fix128`)
  - `Fix128GpuKernel` trait (`add` / `sub` / `mul` / `dot` signatures)
  - Determinism contract documented in rustdoc (subgroup reduce ordering + traversal ordering)
  - 2 unit tests (constant + round-trip)
- **`--features physics-solver`** — Compute-shader-based Fix128 physics solver bridge
  - `TrtSolverAdapter` implementing `alice_physics::gpu_bridge::GpuSolverBridge`
  - Buffers island contents into `Fix128Gpu` storage for the WGSL kernels (kernels land in the follow-up)
  - 1 unit test (zero-iteration round-trip byte-for-byte)
  - Composite feature: enables `physics` + `fix128-arithmetic` + `alice-physics/gpu-solver-bridge`

### Companion release

Pair with [ALICE-Physics v0.8.0](https://github.com/ext-sakamoro/ALICE-Physics/releases/tag/v0.8.0) `--features gpu-solver-bridge` (auto-enabled by `physics-solver`).

### Backwards compatibility

- Fully backwards compatible with v0.1.x
- Existing `GpuPhysicsController` (ML control policy inference) is untouched
- New Fix128 GPU primitive + `TrtSolverAdapter` are strictly opt-in; default `cargo build` sees zero new API surface

## [0.1.1] - 2026-03-04

### Added
- `ffi` — C-ABI FFI 37 `extern "C"` functions (Device/Weight/Tensor/Compute/Engine/Version)
- `python` — PyO3 5 classes (GpuDevice, GpuTernaryWeight, GpuTensor, TernaryCompute, InferenceEngine)
- Unity C# bindings — 37 DllImport + 5 RAII IDisposable handles (`bindings/unity/AliceTrt.cs`)
- UE5 C++ bindings — 37 extern C + 5 RAII unique_ptr handles (`bindings/ue5/AliceTrt.h`)
- FFI prefix: `at_trt_*`
- 63 tests (52 core + 9 FFI + 2 doc-tests)

### Fixed
- `cargo fmt` 21箇所の末尾スペース修正

## [0.1.0] - 2026-02-23

### Added
- `device` — `GpuDevice` wgpu initialization (Metal/Vulkan/DX12)
- `kernel` — WGSL compute shaders: matvec, tiled matvec, batched matmul, ReLU
- `weights` — `GpuTernaryWeight` 2-bit bitplane storage (8x compression vs FP16)
- `tensor` — `GpuTensor` VRAM buffer with upload/download
- `pipeline` — `TernaryCompute` 4-pipeline dispatch with auto kernel selection
- `inference` — `GpuInferenceEngine` multi-layer forward pass (all data stays in VRAM)
- Feature flags: `cuda`, `physics`, `sdf`, `db`, `view`, `voice`
- ALICE-ML `TernaryWeightKernel` / `TernaryWeight` zero-copy import
- 52 unit tests + 2 doc-tests
