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

### 2.3 v2.2.0 scope (upcoming: Morton sort kernel)

The v2.2.0 release ships the **deterministic Morton sort kernel** that consumes the v2.1.0 `FIX128_MORTON_CODE_WGSL` output and produces the same sorted (Morton-code, primitive-index) pair sequence that `ALICE-Physics::LinearBvh::build` obtains from `primitives.sort_by_key(|p| p.morton)` on the CPU. The sorted output is the direct input for the v2.3.0 BVH build kernel.

#### Algorithm choice — LSB-first 8-bit radix, one pass per dispatch

Deterministic sorts on the GPU face a design trade-off between throughput and determinism guarantees. For Phase 3 we prioritise **guaranteed byte-exact CPU-GPU parity across every 3-platform CI run** over raw throughput, and choose LSB-first 8-bit radix sort with one compute dispatch per radix pass:

- **8-bit radix** — each pass processes 8 bits of the 63-bit Morton code, so **8 dispatches** cover the whole key space (the top bit of the 63-bit code is always 0, so 8 × 8 = 64 covers everything with one wasted pass, or exactly 8 with a `if pass == 7 { mask = 0x7F }` guard in the shader).
- **LSB-first** — process bits 0-7 first, then bits 8-15, and so on. LSB-first radix is naturally stable: two keys with the same current-byte value keep their relative order from the previous pass. This is exactly what CPU `sort_by_key` produces (Rust's `sort_by_key` uses a stable sort — Timsort — under the hood).
- **One dispatch per pass** — each pass is a self-contained compute dispatch: (1) build the 256-bin histogram of current-byte values, (2) exclusive-scan the histogram to bucket offsets, (3) scatter each input key/index pair to its target position in the output arrays. Ping-pong between two `(codes, indices)` buffer pairs across the 8 passes.

The alternative (single-dispatch multi-pass with workgroup-scoped tiling) is faster but introduces cross-workgroup ordering hazards that require careful `workgroupBarrier` placement and open the v0.8.1 FXC X4026 uniformity crash risk on WARP. Explicitly out of scope for v2.2.0; may revisit in a v2.2.x optimisation pass once the byte-exact contract is locked.

#### Determinism proof sketch

Each pass is deterministic because:

1. The histogram step reads every input exactly once and increments a fixed-size (256-entry) counter array via `atomicAdd`. The set of increments is fully determined by the input, not the order in which they occur.
2. The exclusive-scan step is deterministic by construction (single-threaded prefix sum, or well-known deterministic parallel scan primitives — details in the v2.2.0 impl).
3. The scatter step reads each input exactly once and writes it to a position derived deterministically from `(bucket_offset, input_index_in_bucket)`. Because LSB-first radix processes buckets in ascending value order and the input-index-in-bucket is monotonic within each thread's assigned range, the output is uniquely determined by the input.

Across the 8 passes, the ping-pong preserves the invariant "buffer A holds the state after the last-even pass, buffer B after the last-odd pass". The final result is byte-identical to `stable_sort_by_key(morton)` on the CPU.

#### Bindings (per pass)

- `@group(0) @binding(0) var<storage, read>       codes_in:    array<vec2<u32>>` — input 63-bit Morton codes as `(low32, high32)`.
- `@group(0) @binding(1) var<storage, read>       indices_in:  array<u32>` — input primitive indices (parallel to codes).
- `@group(0) @binding(2) var<storage, read_write> codes_out:   array<vec2<u32>>` — output codes.
- `@group(0) @binding(3) var<storage, read_write> indices_out: array<u32>` — output indices.
- `@group(0) @binding(4) var<uniform>             params: SortPassParams` — `{ pass_bit_shift: u32, count: u32 }`. On pass `p`, `pass_bit_shift = p * 8u` and the shader extracts `(code >> pass_bit_shift) & 0xFFu` as the histogram key.
- `@group(0) @binding(5) var<storage, read_write> histogram: array<atomic<u32>, 256>` — 256-bin histogram, zeroed by the CPU between passes.

Between passes the CPU (a) swaps `(codes_in, codes_out)` and `(indices_in, indices_out)` bindings, (b) zeroes the histogram buffer, (c) updates `params.pass_bit_shift`. This is cheap host-side book-keeping; the compute work stays fully on the GPU.

#### CPU golden strategy

The v2.2.0 CPU golden mirrors the CPU implementation:

```rust
let mut cpu_pairs: Vec<(u64, u32)> = /* ... zip morton and index ... */;
cpu_pairs.sort_by_key(|(m, _)| *m); // Timsort — stable
```

The GPU test dispatches 8 passes (or 7 with the top-bit guard), reads back the sorted `(codes_out, indices_out)` pair, and asserts byte-for-byte equality against `cpu_pairs`.

#### Edge cases covered by the v2.2.0 fixture

- **Empty array** — `count = 0`, all passes short-circuit, output is empty.
- **Single element** — `count = 1`, all 8 passes leave it in place.
- **All codes identical** — degenerate histogram (one bucket = N, others = 0). Output preserves insertion order because LSB-first + stable.
- **All codes distinct** — general case.
- **Duplicate Morton codes** (same AABB centre from two different bodies) — stability guarantees the pair with the lower primitive index comes first. Test asserts this explicitly.
- **Ascending pre-sorted input** — output equals input; no swaps.
- **Descending pre-sorted input** — full reversal; exercises every bucket-crossing.

The fixture size for the v2.2.0 golden is **64 primitives** — small enough to hand-verify and large enough to exceed the workgroup size (64) so cross-workgroup ordering is exercised.

#### Skeleton in v2.1.x (this session)

The `FIX128_MORTON_SORT_WGSL` constant lands in this session as a **skeleton preview** — struct definitions + bindings + a no-op smoke entry point, with the impl body marked `// TODO(v2.2.0-impl)`. Naga validation passes; no `#[deprecated]` attribute; no adapter integration. External `Fix128GpuKernel` implementers can start compiling against the binding-layout contract now and swap in the v2.2.0 impl body when it ships.

The version stays at 2.1.0 — a skeleton is not a functional promise, so no MINOR bump. The `Unreleased` section of `CHANGELOG.md` documents the skeleton so the v2.2.0 release notes can pick up the diff cleanly.

### 2.4 v2.3.0 scope (upcoming: GPU BVH build kernel)

The v2.3.0 release ships the **GPU LinearBvh build kernel** that consumes the v2.2.0 sorted `(codes, indices)` output plus a parallel array of pre-quantised i32 primitive AABBs, and produces a `nodes: Vec<BvhNodeGpu>` sequence **byte-identical** to `ALICE-Physics::bvh::LinearBvh::build(primitives).nodes` on the CPU. This is the heaviest kernel in the Phase 3 series: it ports the recursive `build_recursive` from `ALICE-Physics/src/bvh.rs` verbatim while preserving the §3.1 escape-pointer discipline established by ALICE-Physics `dede78c`.

#### Algorithm choice — single-workgroup single-thread iterative build_recursive port

The CPU `build_recursive` is a recursive function that produces a depth-first pre-order tree with escape pointers. On the GPU we prioritise **byte-exact CPU-GPU parity** over throughput, and port the algorithm as a **single-workgroup single-thread iterative build** using an explicit continuation-passing-style stack:

- **Single workgroup, single thread** (`@workgroup_size(1)`, dispatched with `dispatch_workgroups(1, 1, 1)`) — the entire build runs sequentially in one invocation. This eliminates every cross-thread ordering hazard that could break byte-exact parity: no `atomicAdd` for node insertion, no cross-workgroup scan for subtree sizes, no thread-local speculative construction. The node buffer's write order is exactly the CPU's `nodes.push()` order.
- **Explicit stack in `var<function>` array** — WGSL has no recursion; we simulate `build_recursive` with a `Frame` struct that records `{ start, end, escape_idx, phase, node_idx, mid, left_idx, left_size }` and a fixed-size stack (`MAX_STACK_DEPTH = 128`). Each frame has three phases: (0) initial visit (compute AABB, possibly push leaf and return), (1) after left child returns (save `left_size`, push right child frame), (2) after right child returns (write parent's `first_child_or_prim`, sweep left subtree for placeholder backfill, pop frame). Return values propagate via a single u32 "return register" written by the popping frame and read by the parent frame in phase 1 or 2.
- **`LEFT_ESCAPE_PLACEHOLDER = 0u`** — identical to the CPU convention. Index 0 is always the tree root; escape pointers strictly move forward; no legitimate escape target is ever 0. The GPU port MUST NOT use a position-dependent placeholder like `left_idx + 1` (see §3.1 and skill §11.4).
- **Subtree-size return + linear sweep** — the popping frame writes `1 + left_size + right_size` (internal) or `1` (leaf) into the return register. When phase 2 completes, the caller sweeps `nodes[left_idx..left_idx + left_size]` in a single `for` loop, replacing every `escape_idx() == LEFT_ESCAPE_PLACEHOLDER` node with the real `right_idx = left_idx + left_size`. No leftmost-spine recursion; no `old_escape == root + 1` matching.

The alternative (parallel top-down BVH construction via Karras-style hierarchical linear BVH with atomic index generation) is 10-100x faster on large inputs but requires fresh determinism analysis and would ship as v2.3.x optimisation. The v2.3.0 correctness-first choice lets v2.4.0 (find_pairs) integrate against a stable byte-exact contract from day one.

#### Determinism proof sketch

Each step of the iterative build is deterministic because:

1. **AABB union in i32 domain** — the AABB for `primitives[start..end]` is the componentwise `min` / `max` of pre-quantised i32 values. i32 min/max is commutative and associative in the total order, so the linear-scan order matches the CPU `aabb.union(&prim.aabb)` fold order exactly. **Byte-exact by construction.** (The CPU version does the union in Fix128 space then quantises to i32 via floor/ceil; because floor and ceil are monotonic — `min(floor(a), floor(b)) = floor(min(a, b))` — pre-quantising on the CPU host side and doing i32 min/max on the GPU is byte-exact equivalent to the CPU fold. See §5 CPU golden fixture note.)
2. **Split search on Morton codes** — `find_split` scans `first_code`, `last_code`, computes the highest differing bit via `clz` (WGSL `countLeadingZeros`), and binary-searches for the split. All operations are on 64-bit integers emulated as `vec2<u32>`; the CPU golden uses u64 native arithmetic; the byte pattern of the result index is identical.
3. **Node push order** — single-thread iterative build visits frames in the same order the CPU recursion does: parent, left subtree (fully), right subtree (fully). The `nodes_out` write index matches the CPU `nodes.push()` sequence one-for-one.
4. **Placeholder backfill sweep** — after the left subtree completes, the sweep visits `nodes_out[left_idx..left_idx + left_size]` in strict ascending index order. Only `escape_idx() == LEFT_ESCAPE_PLACEHOLDER` nodes are touched. This mirrors the CPU sweep exactly.

Together with the single-thread dispatch, every write to `nodes_out` occurs in the same order and with the same byte content as the CPU `Vec<BvhNode>`. The final `nodes_out[0..node_count]` slice is byte-identical to the CPU `LinearBvh::build(primitives).nodes` sequence.

#### Bindings (v2.3.0 layout — frozen at this release)

- `@group(0) @binding(0) var<storage, read>       sorted_codes:   array<vec2<u32>>` — 63-bit Morton codes in sorted order from v2.2.0, packed as `(low32, high32)`. Used only by `find_split` to locate the highest differing bit.
- `@group(0) @binding(1) var<storage, read>       sorted_indices: array<u32>` — parallel primitive indices in sorted order. Copied verbatim into `BvhNodeGpu::first_child_or_prim` for leaves. Also passed back to the caller as the resulting `primitives` field of `LinearBvh` (via the sorted_indices input buffer being reused for the caller's `LinearBvh::primitives` Vec).
- `@group(0) @binding(2) var<storage, read>       sorted_aabbs:   array<AabbI32>` — pre-quantised i32 primitive AABBs in sorted order. The Rust adapter derives these from `BvhPrimitive.aabb` via `aabb_to_i32_min` / `aabb_to_i32_max` before dispatch.
- `@group(0) @binding(3) var<uniform>             params:         BuildParams` — `{ count: u32, leaf_max: u32 }`. `leaf_max` is fixed to 4 (matches CPU `if count <= 4`). Reserved for a future dispatch-time tuning knob without changing the bind group layout.
- `@group(0) @binding(4) var<storage, read_write> nodes_out:      array<BvhNodeGpu>` — output tree nodes. Capacity MUST be at least `2 * count` (worst-case for the recursion + leaf_max=4 gives fewer than 2N nodes for N ≥ 4).
- `@group(0) @binding(5) var<storage, read_write> node_count_out: array<atomic<u32>, 1>` — atomic counter written once with the total node count. Read back by the Rust adapter to trim `nodes_out` to the actual size. `atomic<u32>` is used for buffer-layout consistency with other Phase 3 kernels; only a single write happens.

Where `AabbI32 { min_x: i32, min_y: i32, min_z: i32, max_x: i32, max_y: i32, max_z: i32 }` (24 bytes, matches the Rust `[i32; 3]` + `[i32; 3]` layout) and `BvhNodeGpu { aabb_min_{x,y,z}: i32, first_child_or_prim: u32, aabb_max_{x,y,z}: i32, prim_count_escape: u32 }` (32 bytes, byte-identical to CPU `BvhNode`).

#### CPU golden strategy

The v2.3.0 CPU golden constructs the tree end-to-end via the ALICE-Physics reference implementation:

```rust
use alice_physics::bvh::{BvhPrimitive, LinearBvh, BvhNode};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};

let primitives: Vec<BvhPrimitive> = /* fixture */;
let cpu_bvh = LinearBvh::build(primitives);   // canonical byte-exact reference

// GPU dispatch: pass sorted (codes, indices, aabbs) derived from the
// same fixture (sorting via the v2.2.0 dispatch or the CPU
// stable_sort_by_key for the fixture's Morton codes — both produce
// identical inputs).
let gpu_nodes = dispatch_fix128_bvh_build(&device, /* ... */);

assert_eq!(bytemuck::bytes_of(cpu_bvh.nodes.as_slice()),
           bytemuck::bytes_of(gpu_nodes.as_slice()));   // byte-exact
```

The byte-exact assert compares the raw byte slices via `bytemuck::cast_slice`. No epsilon, no field-by-field compare — the byte patterns must match exactly.

#### Edge cases covered by the v2.3.0 fixture

- **Pile** (32 primitives): tightly packed spheres at slightly offset positions in a 4×4×2 pile. Exercises balanced Morton splits + escape pointer chaining through several levels. Mirrors the `stage_breakdown_collider` bench fixture that surfaced the `dede78c` correctness fix.
- **Uniform grid** (27 primitives): 3×3×3 lattice on integer coordinates. Exercises the "highest differing bit" split path across three cardinal axes and forces balanced tree depth.
- **Degenerate all-colocated** (16 primitives): all AABBs at the origin. Exercises the `first_code == last_code` branch of `find_split` (falls back to `(start + end) / 2`) plus the AABB-union no-op path. This is the same degenerate configuration that stress-tests the CPU recursion for placeholder collision (skill §11.4).

Fixture sizes ≤ 32 are chosen so the whole tree fits comfortably inside the stack depth budget (max recursion depth ≈ 5 for balanced N=32) and the byte-exact readback stays well under the 200KB / dispatch limit imposed by lavapipe.

#### Post-build debug invariant (Rust adapter)

Under `#[cfg(debug_assertions)]`, the Rust adapter runs the same `debug_verify_escape_forward` check as the CPU after readback:

```rust
#[cfg(debug_assertions)]
fn debug_verify_escape_forward(nodes: &[BvhNodeGpu]) -> bool {
    for (i, node) in nodes.iter().enumerate() {
        let esc = node.escape_idx();
        if esc == ESCAPE_NONE { continue; }
        if (esc as usize) <= i { return false; }   // backward or self = illegal
    }
    true
}
```

Any backward escape pointer indicates a bug in the port — either a placeholder was not backfilled, or the sweep visited the wrong range. The check is O(N) and runs only in debug builds.

### 2.5 v2.4.0 scope (upcoming: GPU BVH find_pairs kernel)

The v2.4.0 release ships the **GPU BVH `find_pairs` kernel** that consumes the v2.3.0 `Vec<BvhNodeGpu>` output plus the v2.2.0 sorted primitive indices, and produces a `Vec<(u32, u32)>` **byte-identical** to `alice_physics::bvh::LinearBvh::find_pairs()` on the CPU. This closes the broad-phase pipeline: after v2.4.0 the caller can go BvhPrimitive input → Morton sort → BVH build → pair list entirely on the GPU (minus the final `sort_unstable` + `dedup` that stay host-side per §CPU golden below).

#### Algorithm choice — single-workgroup single-thread stackless traversal

The CPU `find_pairs` iterates each primitive in `bvh.primitives` order and calls `query_callback(&self.bounds, |prim_j| { if prim_i < prim_j { pairs.push(...) } })`. The v2.4.0 port mirrors this exactly under the same discipline as v2.3.0:

- **Single workgroup, single thread** (`@workgroup_size(1)`, dispatched with `dispatch_workgroups(1, 1, 1)`) — the outer `for &prim_i in &self.primitives` loop runs sequentially on one invocation, and every inner tree walk shares that single thread's atomic emission counter. This is the same correctness-first choice as v2.3.0 and v2.2.0's scatter kernel: eliminates every cross-thread ordering hazard that could break byte-exact parity.
- **Stackless traversal via escape pointers** — the inner query mirrors the CPU `query_callback` byte-for-byte: `idx = 0`; loop while `idx != ESCAPE_NONE && idx < node_count`; on AABB-hit-and-leaf emit each primitive and jump to `escape_idx()`; on AABB-hit-and-internal descend to `first_child_or_prim`; on AABB-miss skip via `escape_idx()`. Because the CPU passes `self.bounds` (the world AABB) as the query, every `intersects_i32` check passes and the walk visits every internal + leaf node in flat-array order. The escape pointer is still exercised on the leaf → next-sibling transition.
- **`atomicAdd`-based pair emission** — pair output uses `atomicAdd(&counters[0], 1u)` to reserve a slot in `pairs_out`. The single-thread dispatch means only one invocation ever increments the counter, so `atomicAdd` is functionally equivalent to a plain increment; the atomic wrapper is kept for buffer-layout consistency with other Phase 3 kernels and to make future v2.4.x parallelisation a drop-in change. The `prim_i < prim_j` filter is applied at emission time (same as the CPU), so total emissions equal the final unique-pair count (no duplicates possible).
- **Debug cycle guard (§11.4)** — per-primitive visit counter is capped at `2 * node_count`. On overflow, the kernel sets `atomicStore(&counters[1], 1u)` and breaks the current inner loop (production behaviour matches CPU: unbounded loop would OOM, we prefer graceful truncation + Rust-side panic). The Rust adapter reads `counters[1]` under `#[cfg(debug_assertions)]` and panics if non-zero. This is the mechanism the [design doc §3.1](#31-gpu-bvh-construction--114-discipline-applied) mandates for every Phase 3 traversal kernel.

The alternative (per-primitive parallel dispatch — one thread per outer-loop iteration) is a v2.4.x optimisation. It requires each thread to reserve a contiguous output range via prefix-sum on the per-primitive pair count, which is a two-dispatch pattern. The v2.4.0 correctness-first choice is single-thread; parallel dispatch lands additively once the byte-exact contract is locked.

#### Bindings (v2.4.0 layout — frozen at this release)

- `@group(0) @binding(0) var<storage, read>       nodes:        array<BvhNodeGpu>` — v2.3.0 build output.
- `@group(0) @binding(1) var<storage, read>       primitives:   array<u32>` — sorted primitive body indices from v2.2.0 (the `sorted_indices` output; corresponds to CPU `LinearBvh::primitives`).
- `@group(0) @binding(2) var<uniform>             params:       FindPairsParams` — `{ node_count: u32, prim_count: u32, max_pairs: u32, _pad: u32 }` (16-byte aligned). `max_pairs` bounds the output buffer for overflow detection.
- `@group(0) @binding(3) var<uniform>             world_bounds: AabbI32` — pre-quantised world AABB (`from_physics_aabb` on the Rust adapter side). Consumed as the query AABB for every inner traversal — matches CPU `query_callback(&self.bounds, ...)`.
- `@group(0) @binding(4) var<storage, read_write> pairs_out:    array<vec2<u32>>` — output pairs. Element `i` = `vec2(prim_i, prim_j)`.
- `@group(0) @binding(5) var<storage, read_write> counters:     array<atomic<u32>, 2>` — `[0]` = emitted pair count; `[1]` = cycle-guard overflow flag (0 = OK, 1 = at least one prim_i's traversal exceeded the visit cap).

#### CPU golden strategy

The v2.4.0 CPU golden mirrors the CPU implementation end-to-end and applies the same host-side finaliser to both outputs before compare:

```rust
use alice_physics::bvh::LinearBvh;

let cpu_bvh = LinearBvh::build(fixture.clone());
let cpu_pairs = cpu_bvh.find_pairs();   // already sort_unstable + dedup applied

let gpu_nodes = dispatch_fix128_bvh_build(&device, &sorted_codes, &sorted_indices, &sorted_aabbs);
let mut gpu_pairs = dispatch_fix128_bvh_find_pairs(
    &device,
    &gpu_nodes,
    &sorted_indices,
    &AabbI32Gpu::from_physics_aabb(&cpu_bvh.bounds),
);
gpu_pairs.sort_unstable();
gpu_pairs.dedup();

assert_eq!(gpu_pairs, cpu_pairs);   // byte-exact
```

The host-side `sort_unstable` + `dedup` on the readback matches the CPU's own tail steps in `find_pairs`. GPU-side sort is deferred to a v2.4.x optimisation once the byte-exact contract is locked; the host tail is O(N² log N²) for pair counts of `N * (N-1) / 2`, which stays well inside the frame budget for N ≤ 10000.

#### Edge cases covered by the v2.4.0 fixture

Same three fixtures as v2.3.0 to exercise the discipline end-to-end:

1. **Pile (32 primitives, 4×4×2 grid, spacing 3, radius 1)** — `32 * 31 / 2 = 496` pairs. Exercises the balanced tree walk over many escape-pointer jumps.
2. **Uniform grid (27 primitives, 3×3×3, spacing 4, radius 1)** — `27 * 26 / 2 = 351` pairs. Exercises the "highest differing bit" split path via the walk.
3. **Degenerate all-colocated (16 primitives at origin)** — `16 * 15 / 2 = 120` pairs. Exercises the midpoint-fallback split path plus the all-AABBs-identical-with-world-bounds edge case.

Every fixture asserts `assert_eq!(gpu_pairs, cpu_pairs)` on the final host-sorted-and-deduped `Vec<(u32, u32)>`. Cycle-overflow flag is asserted zero under `#[cfg(debug_assertions)]` on the Rust adapter path.

#### Post-dispatch debug invariant (Rust adapter)

Under `#[cfg(debug_assertions)]`, the Rust adapter reads `counters[1]` and panics on cycle overflow:

```rust
#[cfg(debug_assertions)]
{
    assert!(
        cycle_overflow == 0,
        "BVH find_pairs cycle guard triggered: traversal visited > 2 * node_count nodes for at least one primitive. This indicates a malformed tree with backward escape pointers (skill §11.4)."
    );
}
```

Any overflow means the tree has a backward or self-referential escape pointer that slipped through v2.3.0's `debug_verify_escape_forward` check — an escalated bug report, not a benign fixture.

### 2.6 v2.5.0 scope (upcoming: sphere-sphere narrow-phase contact kernel)

The v2.5.0 release ships the **GPU sphere-sphere narrow-phase contact kernel** — a pure geometric map from `(pairs, positions, radii)` to a `Vec<ContactGpu>` list, byte-identical to the corresponding stanza of `alice_physics::solver::PhysicsWorld::detect_collisions` (the sphere-sphere `delta.normalize_with_length()` → depth/normal/point_a/point_b block) on the CPU. This closes the broad-phase → narrow-phase pipeline; v2.6.0 will consume this contact list for the GPU PGS solve.

#### Scope carve-out — geometric map only

The CPU `detect_collisions` mixes several concerns:

1. Filter check (`CollisionFilter::can_collide`) using per-body bit masks.
2. Static-static skip (`bodies[a].is_static() && bodies[b].is_static()`).
3. Both-sleeping skip (`islands.is_sleeping(a) && islands.is_sleeping(b)`).
4. **Sphere-sphere geometric test** (delta, length, depth, normal, point_a, point_b).
5. Event reporting, wake-on-contact, sensor branching, material-parameterised constraint construction.

Only stage 4 is a pure `Vec3Fix` / `Fix128` numeric map that admits a clean GPU port. Stages 1-3 depend on per-body state (filter masks, motion type flags, island sleep bits) that lives on the CPU and would drag in a much wider bind-group surface; stage 5 is a CPU-orchestrated fanout of side effects (event bus, ledger, cache write). The v2.5.0 GPU kernel accepts a pre-filtered `pairs` list from the caller (typically the v2.4.0 output, possibly post-filtered on CPU) and emits contacts strictly for the geometric map. The caller re-applies stage 5 on the readback.

This mirrors the same scope decision as the v1.6.0 batched rigid-rod projection: GPU handles the deterministic numeric hot path, CPU handles the state-dependent orchestration.

#### Algorithm — CPU parity per-pair test

Per pair `(a, b)` from the input list:

```rust
let delta = positions[b] - positions[a];                    // 3× Fix128 sub
let (normal, dist) = delta.normalize_with_length();         // 3× mul + 2× add + 1× sqrt + 1× div + 3× mul
let combined_radius = radii[a] + radii[b];                  // 1× add
if dist < combined_radius && !dist.is_zero() {              // lt + is_zero
    let depth = combined_radius - dist;                     // 1× sub
    let point_a = positions[a] + normal * radii[a];         // 3× mul + 3× add
    let point_b = positions[b] - normal * radii[b];         // 3× mul + 3× sub
    emit Contact { body_a: a, body_b: b, depth, normal, point_a, point_b };
}
```

`Vec3Fix::normalize_with_length` expands to `length_squared = dot(delta, delta); len = length_squared.sqrt(); inv_len = ONE / len; (delta * inv_len, len)`, which is what the WGSL port composes byte-for-byte from the certified v0.3.0 (add/sub/mul), v1.4.0 (div), and v1.4.1 (sqrt) primitives inlined per the house style. Total ~19 Fix128 ops per pair: 7 sub / 6 add / 12 mul / 1 div / 1 sqrt / 1 lt / 1 is_zero.

#### Bindings (v2.5.0 layout — frozen at this release)

- `@group(0) @binding(0) var<storage, read>       pairs:         array<vec2<u32>>` — pre-filtered pair list. Typically the v2.4.0 output, but the kernel treats each element as an independent geometric test — the caller is free to inject synthetic pairs (e.g., persistent-manifold hints from the contact cache).
- `@group(0) @binding(1) var<storage, read>       positions:     array<Vec3FixGpu>` — per-body position. Element `i` corresponds to body id `i`; the pair `(a, b)` indexes this array directly.
- `@group(0) @binding(2) var<storage, read>       radii:         array<Fix128Gpu>` — per-body sphere radius. Same indexing as `positions`.
- `@group(0) @binding(3) var<uniform>             params:        SphereContactParams` — `{ pair_count: u32, max_contacts: u32, _pad0: u32, _pad1: u32 }` (16-byte aligned).
- `@group(0) @binding(4) var<storage, read_write> contacts_out:  array<ContactGpu>` — output contacts. Element `i` = `{ body_a, body_b, depth, normal, point_a, point_b }`.
- `@group(0) @binding(5) var<storage, read_write> contact_count: array<atomic<u32>, 1>` — emitted contact count, incremented by the single-thread dispatch via `atomicAdd` for buffer-layout consistency (see v2.4.0 §2.5 rationale).

Where `Vec3FixGpu { x: Fix128Gpu, y: Fix128Gpu, z: Fix128Gpu }` (48 bytes) and `ContactGpu { body_a: u32, body_b: u32, _pad0: u32, _pad1: u32, depth: Fix128Gpu, normal: Vec3FixGpu, point_a: Vec3FixGpu, point_b: Vec3FixGpu }` (176 bytes, 16-byte aligned at the `depth` field).

#### CPU golden strategy

The v2.5.0 golden replays the CPU sphere-sphere block directly rather than driving `PhysicsWorld::detect_collisions` (which entangles the stage 1-3 filters). This isolates the geometric map for byte-exact comparison:

```rust
let mut cpu_contacts: Vec<ContactGpu> = Vec::new();
for &(a, b) in &pairs {
    let pa = positions_physics[a as usize];
    let pb = positions_physics[b as usize];
    let ra = radii_physics[a as usize];
    let rb = radii_physics[b as usize];
    let delta = pb - pa;
    let (normal, dist) = delta.normalize_with_length();
    let combined = ra + rb;
    if dist < combined && !dist.is_zero() {
        let depth = combined - dist;
        let point_a = pa + normal * ra;
        let point_b = pb - normal * rb;
        cpu_contacts.push(ContactGpu {
            body_a: a, body_b: b, _pad0: 0, _pad1: 0,
            depth: depth.into(), normal: normal.into(),
            point_a: point_a.into(), point_b: point_b.into(),
        });
    }
}

let gpu_contacts = dispatch_fix128_sphere_sphere_contact(&device, &pairs, &positions_gpu, &radii_gpu);
assert_eq!(gpu_contacts, cpu_contacts);   // byte-exact
```

Both sides emit in `pairs` iteration order (no host-side sort), so the byte-exact assert covers ordering as well as per-field content.

#### Edge cases covered by the v2.5.0 fixture

Three new fixtures specifically exercise the geometric branches:

1. **Overlap pile (32 bodies, 4×4×2 grid, spacing 1.5, radius 1)** — face-adjacent pairs at distance 1.5 collide (sum radius 2.0 > 1.5); face-diagonal pairs at distance ≈2.12 do not. Exercises both the collision and no-collision paths, plus non-trivial normal directions.
2. **Chain (6 bodies at x = 0, 2, 4, 6, 8, 10, radius 1.1)** — adjacent pairs at distance 2 collide (sum 2.2 > 2); non-adjacent pairs do not. Every collision has `normal = (1, 0, 0)` and `depth = 0.2`, which pins the byte-exact expected values.
3. **Zero-distance degenerate (4 bodies all at origin, radius 1)** — every pair has `dist == 0` and is filtered out by the `!dist.is_zero()` guard. Zero contacts emitted; exercises the `is_zero` short-circuit that would otherwise divide by zero in `inv_len = ONE / dist`.

Every fixture asserts `assert_eq!(gpu_contacts, cpu_contacts)` on the full `Vec<ContactGpu>` (each 176 bytes) via `#[derive(PartialEq, Eq)]`.

#### Kernel size + house style

The v2.5.0 shader inlines every Fix128 primitive it needs — add / sub / mul / div / sqrt / lt / is_zero / half / neg / abs / u64 + u128 helpers — plus the `Vec3FixGpu` / `ContactGpu` structs plus the sphere-sphere composition. Estimated ~500 lines total, matching the Phase 2 batched rigid-rod kernel weight class. This is deliberate: the WGSL "no include" constraint means every shader is standalone, and copying certified primitives keeps each kernel independently auditable without cross-shader coupling.

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
