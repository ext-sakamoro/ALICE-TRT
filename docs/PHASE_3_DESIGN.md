# Phase 3 GPU BVH / narrow-phase / CCD — Design Document

**Status**: 2026-07-07 initial draft
**Companion**: [`../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md`](../../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md)
**Gate satisfaction**: Gates 1 and 2 satisfied on 2026-07-06 (`stage_breakdown_collider` measured 97.4% / 99.8% of frame cost in narrow-phase + PGS at N=100 / N=1000 pile — well above the 30% threshold).

## §1 Overview

Phase 3 lifts the remaining CPU-side per-frame work — BVH broad-phase pair generation, sphere-sphere narrow-phase contact detection, and PGS contact solve — onto the GPU on top of the existing v1.4-v2.0 constraint bridge. The v2.0.0 release closed Phase 2 (parallel constraint dispatch) and formalised the Phase 3 gate; this document is the design skeleton for the v2.1+ release series that implements the kernel work.

### Goals

- **Bit-exact CPU-GPU parity** for every stage. Every new kernel ships with a CPU golden test that produces identical byte output for identical input, on the 3-platform CI matrix (Metal / Vulkan lavapipe / DX12 WARP).
- **Same v1.x semver posture**. Each release adds new WGSL constants under fresh names; existing public API remains stable. Breaking changes only land at a formal MAJOR bump.
- **Same determinism guardrails** as Phase 1 / Phase 2. See §3.
- **Same 3-platform CI matrix** as v0.8.1 → v2.0.0. WARP driver behaviour is checked from day one for every kernel.

### Non-goals

- **CCD (continuous collision detection)** — outside Phase 3 scope for the initial series. CCD only benefits a small subset of bodies (`enable_ccd(id)`) and can layer on top of the discrete narrow-phase kernels later.
- **Non-sphere colliders** — the ALICE-Physics `detect_collisions` path only understands sphere colliders today; Phase 3 mirrors that. Capsule / box / arbitrary SDF colliders are Phase 4.
- **Cross-node distributed BVH** — single-device GPU only. Lockstep rollback determinism still applies for per-device state.
- **API breakage from v2.0.0** — every new kernel is additive; the existing `TrtSolverAdapter` public surface remains intact.

## §2 Kernel Sequence

The CPU flow in `ALICE-Physics/src/solver.rs::detect_collisions` runs:

1. Build `primitives: Vec<BvhPrimitive>` from sphere-radiused bodies (linear O(N)).
2. `LinearBvh::build(primitives)` — compute world bounds → Morton codes → sort → recursive tree build with escape pointers.
3. `bvh.find_pairs()` — for each primitive, walk the tree emitting overlapping primitive pairs.
4. Per-pair narrow-phase (sphere-sphere distance, contact normal / point / depth).
5. PGS contact constraint solve inside the substep loop.

Phase 3 lifts stages 2-5 onto the GPU while stage 1 remains on the CPU (uploading the flat primitive buffer each frame is cheap and matches the existing PGS bridge upload pattern).

### 2.1 Kernel plan (release-by-release)

| Release | Kernel | Purpose | Determinism reference |
|---|---|---|---|
| **v2.1.0** | `FIX128_AABB_HELPERS_WGSL` | `Fix128AabbGpu` struct + `aabb_from_sphere` / `aabb_union` / `aabb_center` shader helpers | Same Fix128 primitives as v1.4.x |
| **v2.1.0** | `FIX128_MORTON_CODE_WGSL` | Per-primitive Morton code (63-bit u64 emulated as `vec2<u32>`) matching `ALICE-Physics::bvh::point_to_morton` byte-for-byte | New; CPU golden = `point_to_morton` |
| v2.2.0 | `FIX128_MORTON_SORT_WGSL` | Deterministic sort of primitives by Morton code (radix sort, workgroup-size-invariant) | New; CPU golden = `primitives.sort_by_key(morton)` |
| v2.3.0 | `FIX128_BVH_BUILD_WGSL` | LinearBvh build with escape pointers, following the position-independent placeholder + O(subtree_size) sweep pattern from ALICE-Physics `dede78c` (see §3) | ALICE-Physics `LinearBvh::build_recursive` |
| v2.4.0 | `FIX128_BVH_FIND_PAIRS_WGSL` | Stackless traversal + pair emission with debug cycle guard | ALICE-Physics `find_pairs` |
| v2.5.0 | `FIX128_SPHERE_SPHERE_CONTACT_WGSL` | Per-pair sphere-sphere contact (distance via Fix128 sqrt + normal + depth) | New; CPU golden = `detect_collisions` narrow-phase branch |
| v2.6.0 | `FIX128_PGS_CONTACT_SOLVE_WGSL` | PGS contact impulse computation + position correction | New; CPU golden = ALICE-Physics `solve_contact_constraints` |

Each release is a self-contained additive step. The v2.1.0 primitives (AABB helpers + Morton) unblock every subsequent step. The v2.6.0 release closes the Phase 3 loop end-to-end and lets ALICE-Physics `detect_collisions` opt into GPU dispatch via a new toggle on `TrtSolverAdapter`.

### 2.2 v2.1.0 scope (this release)

- **New public WGSL constants** in `alice_trt::fix128`:
  - `FIX128_AABB_HELPERS_WGSL`
  - `FIX128_MORTON_CODE_WGSL`
- **New public Rust struct** `Fix128AabbGpu` (`#[repr(C)]`, byte-layout mirror of the WGSL struct).
- **Unit tests**:
  - `wgsl_aabb_helpers_shader_present` (structural: helper fn names + struct definitions in the WGSL string)
  - `wgsl_aabb_helpers_shader_compiles` (naga validation, headless-CI-safe)
  - `wgsl_morton_code_shader_present`
  - `wgsl_morton_code_shader_compiles`
  - `morton_code_cpu_gpu_golden` (bit-exact vs ALICE-Physics `point_to_morton` on a 16-primitive fixture — headless CI adapter check, GPU dispatch + read back + compare)
- **Adapter integration**: none yet; the constants are exposed for external `Fix128GpuKernel` implementers and for the v2.2+ series to consume. The v2.0.0 `TrtSolverAdapter` public surface is unchanged.

Subsequent releases (v2.2+) layer on top without breaking v2.1's contracts.

## §3 Determinism Invariants

Every Phase 3 kernel MUST preserve the five determinism-breaking routes catalogued in the [`deterministic-physics-lockstep-discipline`](https://github.com/ext-sakamoro/claude-config/blob/main/claude-skills/deterministic-physics-lockstep-discipline/SKILL.md) skill:

1. **Broad-phase precision** — every AABB stays in Fix128Gpu space. No f32 intermediate for AABB overlap tests.
2. **CORDIC / sqrt** — Newton-Raphson from v1.4.1 already ships; kernel callers use the byte-exact primitive.
3. **SIMD / dispatch order** — no cross-lane sum reordering, no cross-workgroup non-deterministic reduce. Primitive order is determined by `body_id` at upload time and preserved through the pipeline.
4. **Rollback snapshot delta** — no new state on the GPU that isn't derivable from the CPU-side `PhysicsWorld` snapshot. GPU buffers are re-derived from state each frame.
5. **Thread / workgroup traversal order** — the BVH build's escape pointer placement (§3.1 below) is position-independent and single-dispatch. Any parallelisation is deferred to a later kernel with fresh determinism analysis.

### 3.1 GPU BVH construction — §11.4 discipline applied

The July 2026 ALICE-Physics CPU BVH fix ([`dede78c`](https://github.com/ext-sakamoro/ALICE-Physics/commit/dede78c)) established the canonical reference implementation for escape-pointer BVH construction:

- **Reserved magic value placeholder** — index `0` is safe because the tree root is at 0 and escape pointers strictly move forward; no legitimate escape target is ever `0`. **The GPU port MUST use the same convention** (`LEFT_ESCAPE_PLACEHOLDER: u32 = 0u`).
- **Subtree_size return + linear sweep** — `build_recursive` returns the number of nodes pushed by the current call. After building the left subtree, the caller sweeps `nodes[left_idx..left_idx + left_size]` in one linear pass, replacing every placeholder with the real right sibling index. **The GPU port MUST use the same O(subtree_size) sweep pattern** — no leftmost-spine recursion, no position-dependent `root + 1` matching.
- **`debug_assert!` on build completion** — every node's escape pointer is either `ESCAPE_NONE` or strictly forward. **The GPU port MUST emit an equivalent invariant** via a debug-only readback pass on the flat node buffer, gated on `#[cfg(debug_assertions)]` in the Rust adapter.
- **Traversal cycle guard** — the stackless traversal's node-visit counter is capped at `2 * nodes.len()` in debug builds; exceeding this panics with cycle location. **The GPU port MUST include the same guard** in the debug adapter path (`RUST_BACKTRACE=1` visibility on WARP panics has already proven its worth in the v0.7.1 → v0.8.1 investigation).

The CPU implementation at [`ALICE-Physics/src/bvh.rs`](../../ALICE-Physics/src/bvh.rs) is the specification. Any deviation is a bug in the port.

### 3.2 Morton coding — bit-exact CPU-GPU parity

The v2.1.0 Morton kernel MUST match `ALICE-Physics::bvh::point_to_morton` byte-for-byte on a shared fixture. The CPU implementation splits the coordinate normalisation into three cases (`t < 0` → 0, `t.hi >= 1` → `0x1FFFFF`, else `(t.lo >> 43) & 0x1FFFFF`); the GPU implementation MUST reproduce all three branches without floating point.

- `Fix128::div` (v1.4.0 GPU kernel) provides the byte-exact `t = (point - bounds.min) / size` computation.
- The `t.lo >> 43` extraction accesses the high 21 bits of the low 64-bit half of the Fix128 fractional part. On GPU, this is `t.lo_hi >> 11` (the top 32 bits of `t.lo`, shifted right by 43-32 = 11 bits).
- The 63-bit Morton output is packed as `vec2<u32>` (low 32 bits + high 31 bits). The `expand_bits` implementation is straightforward u64 bit-shift arithmetic, emulated as two u32 halves in WGSL.

Corner cases:

- **Zero-size axis** (`size.x.is_zero()`) → coordinate = 0. Kernel MUST short-circuit before the divide to avoid a div-by-zero shader panic.
- **Negative t** (point outside bounds on the low side) → coordinate = 0. Kernel checks `t.is_negative()` before extraction.
- **`t.hi >= 1`** (point outside bounds on the high side) → coordinate = `0x1FFFFF`. Kernel checks the high-half sign bit / value before extraction.

## §4 Bind Group Layouts

Layout convention for Phase 3 kernels, consistent with v1.4.2's rigid rod shader:

- `@group(0) @binding(0)` — primary storage buffer (per-primitive or per-node data, read or read_write).
- `@group(0) @binding(1)` — uniform buffer (kernel parameters: world bounds, primitive count, etc.).
- `@group(0) @binding(2)` — auxiliary storage (output pair list, output Morton codes, etc.).

### 4.1 v2.1.0 layouts

**`FIX128_MORTON_CODE_WGSL`**:

- `@group(0) @binding(0) var<storage, read>       primitives: array<Fix128AabbGpu>` — input primitive AABBs.
- `@group(0) @binding(1) var<uniform>             world_bounds: Fix128AabbGpu` — pre-computed world AABB (or its own dispatch stage computes this).
- `@group(0) @binding(2) var<storage, read_write> morton_codes: array<vec2<u32>>` — output 63-bit Morton codes packed as (low u32, high u32).

Workgroup size: `@workgroup_size(64)` with `dispatch_workgroups(ceil(N / 64), 1, 1)`. Each thread handles one primitive independently — no shared state, no cross-thread reads. This is the trivial parallelism case.

Dispatch determinism: identical (primitive_index, thread_id) pairs across every run because both are derived from body IDs and the uniform WGSL fixed-size workgroup layout.

## §5 CPU Golden Strategy

Every kernel ships with a bit-exact CPU golden. Pattern:

1. **CPU baseline** — the ALICE-Physics function that the kernel is porting (e.g., `point_to_morton`).
2. **Fixture** — a small (10-20 elements) deterministic input that exercises the kernel's decision branches.
3. **GPU dispatch** — one dispatch through the kernel with the fixture as storage buffer input.
4. **Readback** — `Buffer::map_read` the output buffer into CPU memory.
5. **Byte-for-byte compare** — `assert_eq!` on the raw byte slices. No epsilon, no approximate.

For headless CI without a GPU adapter, the test skips gracefully (following the pattern of `wgsl_pgs_project_distance_shader_compiles`). The 3-platform CI matrix runs the full path.

Fixtures for v2.1.0:

- `Fix128AabbGpu` helpers: 4 hand-constructed AABBs exercising union, intersection edges, sphere-derived AABBs.
- Morton code: 16 primitives with mixed coordinate distributions (interior points, boundary points, out-of-bounds low and high on each axis, zero-size world axis). The 16-element choice is small enough to hand-verify and large enough to exercise the branching.

## §6 3-Platform CI Matrix

Every new v2.1+ kernel must pass on:

- **macOS Metal** — primary developer platform. Fastest to iterate.
- **Ubuntu Vulkan (lavapipe)** — software rasterizer. Catches Mesa-side driver interpretation of WGSL edge cases.
- **Windows DX12 (WARP)** — Microsoft's software rasterizer. Historically the strictest FXC-side; caught the X4026 crash in v0.7.1 → v0.8.1.

The workflow file `.github/workflows/*.yml` already has the matrix in place from v0.8.1 onward; v2.1+ additions require no new CI infrastructure.

Kernel-specific CI gates:

- **Structural test** (`_shader_helpers_present`) — pure string search, runs on every platform without a GPU adapter.
- **Compile test** (`_shader_compiles`) — `naga::valid` + `create_shader_module`, requires an adapter but skips gracefully if none.
- **Golden test** (`_cpu_gpu_golden`) — full dispatch + readback + compare. Requires an adapter. Skipped on headless CI.

## §7 Release Cadence

- **v2.1.0** (this session or next) — Fix128 AABB helpers + Morton code kernel + golden test. Additive; existing surface unchanged.
- **v2.2.0** — Morton-based deterministic sort kernel + golden.
- **v2.3.0** — GPU BVH build. Heaviest kernel; may split into v2.3.0 (build_recursive equivalent) and v2.3.1 (escape pointer sweep). §11.4 discipline applied throughout.
- **v2.4.0** — GPU BVH find_pairs. Debug cycle guard on the adapter path.
- **v2.5.0** — Sphere-sphere narrow-phase contact.
- **v2.6.0** — GPU PGS contact solve + `TrtSolverAdapter` opt-in for the full pipeline.
- **v2.7.0 candidate** — Adapter default-on for narrow-phase (analogous to v1.6.0's parallel dispatch default flip).

Each release ships MINOR-only unless a formal breaking change is required. The v1.0.0 → v2.0.0 semver stability commitment covers the entire v2.x series.

## §8 Open Questions / Decisions Log

The following decisions may need revisiting during implementation:

1. **63-bit Morton u64 emulation** — packed as `vec2<u32>` (low u32, high u32) at output; internal computation uses full 64-bit arithmetic via existing Fix128 u32-pair helpers. Committed 2026-07-07; may revisit if WGSL performance shows the u64 emulation dominates dispatch cost (unlikely at N ≤ 10000).

2. **Sort algorithm for v2.2.0** — deterministic radix sort with fixed workgroup size, OR host-side sort with CPU→GPU upload each frame. Committed to radix for scale; may revisit if adapter measurements at small N show host sort is cheaper per-frame.

3. **BVH build parallelisation for v2.3.0** — initial implementation is single-dispatch sequential (mirrors CPU `build_recursive`). Parallel construction requires fresh determinism analysis and is deferred to v2.4+ if measurements warrant.

4. **Narrow-phase pair batching for v2.5.0** — one pair per workgroup vs many pairs per workgroup vs one dispatch for all pairs. Committed to one pair per thread (analogous to v1.5.1 batched rigid rod); may revisit if occupancy analysis warrants.

5. **PGS contact solve iteration count for v2.6.0** — matches CPU-side `config.iterations` (default 4). Fixed iteration count is required for determinism; no early-exit even if converged.

6. **`TrtSolverAdapter` opt-in vs opt-out for v2.6.0+** — mirrors the v1.5.1 → v1.6.0 pattern: land as opt-in, flip default in a later release after cross-platform validation.

New decisions get logged here with a date and rationale.

## §9 References

- Companion roadmap: [`../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md`](../../ALICE-Physics/docs/GPU_OFFLOAD_ROADMAP.md)
- CPU BVH implementation: [`../../ALICE-Physics/src/bvh.rs`](../../ALICE-Physics/src/bvh.rs)
- CPU BVH July 2026 correctness fix: ALICE-Physics commit `dede78c`
- Determinism discipline: [`deterministic-physics-lockstep-discipline`](https://github.com/ext-sakamoro/claude-config/blob/main/claude-skills/deterministic-physics-lockstep-discipline/SKILL.md) §11.4
- Precedent releases: v1.4.2 (rigid rod on-device sqrt/div), v1.5.1 (batched dispatch), v1.6.0 (default flip), v2.0.0 (formal Phase 2 wrap).
