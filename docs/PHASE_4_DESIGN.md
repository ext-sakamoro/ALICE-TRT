# ALICE-TRT Phase 4 — GPU joint solver design

**Status**: v3.1.0 ships ball-socket only. Hinge / Fixed / Slider / Spring / D6 / ConeTwist land as coordinated minor releases (v3.2 through v3.7) as demand justifies each.

**Motivation**: Phase 3 (v2.2 – v2.7) moved broad-phase, narrow-phase, and PGS contact solve to the GPU. That covered the contact hot path but left `alice_physics::joint::solve_joints` running entirely on the CPU. Every rigid-body sim that uses ragdolls, ropes, hinges, or constraint chains still pays a CPU tax proportional to joint count on every substep. Phase 4 closes that gap.

## §1 — Quaternion primitives (shipped v3.1.0)

- **`QuatFixGpu`** — 64-byte byte-layout mirror of `alice_physics::math::QuatFix`. `#[repr(C)]` with four `Fix128Gpu` fields `(x, y, z, w)` at offsets 0 / 16 / 32 / 48.
- **`FIX128_QUAT_MUL_WGSL`** — byte-exact port of `QuatFix::mul` (16 Fix128 muls + 12 add/sub per quaternion, left-to-right associativity).
- **`FIX128_QUAT_ROTATE_VEC_WGSL`** — byte-exact port of `QuatFix::rotate_vec` (`q * v * q_conjugate`; two quat_mul + one conjugate, ~32 muls + ~24 add/sub per rotation).
- **Rust orchestrators**: `dispatch_fix128_quat_mul` / `dispatch_fix128_quat_rotate_vec` — golden-test entry points; the ball-socket kernel inlines the algorithm directly for efficiency.
- **Byte-exact goldens** (v3.1.0 phase 1 commit `b3dc3a0`):
  - `wgpu_quat_mul_matches_cpu_golden` — identity × identity, identity × arbitrary, arbitrary × arbitrary.
  - `wgpu_quat_rotate_vec_matches_cpu_golden` — identity rotation preserves vector, arbitrary rotation, any rotation of zero returns zero.

## §2 — Ball-socket joint solver (shipped v3.1.0)

### Algorithm

Byte-exact port of `alice_physics::joint::solve_ball_joint`:

```text
for each joint in joints:
    anchor_a = position_a + rotate_vec(rotation_a, local_anchor_a)
    anchor_b = position_b + rotate_vec(rotation_b, local_anchor_b)
    delta    = anchor_b - anchor_a
    distance = sqrt(dot(delta, delta))
    if distance == 0: skip
    compliance_term = compliance / dt_sq        // dt_sq = dt * dt precomputed on CPU
    w_sum           = inv_mass_a + inv_mass_b + compliance_term
    if w_sum == 0: skip
    inv_w_sum  = 1 / w_sum
    lambda     = distance * inv_w_sum
    normal     = delta * (1 / distance)         // component-wise
    correction = normal * lambda
    if inv_mass_a != 0: position_a += correction * inv_mass_a
    if inv_mass_b != 0: position_b -= correction * inv_mass_b
```

### Dispatch shape

`@compute @workgroup_size(1)` + `dispatch_workgroups(1, 1, 1)`. Single-workgroup single-thread iterates the joint list top to bottom, matching the CPU `for joint in joints` order. Joint `i+1` observes position updates from joint `i` when they share a body, which is standard sequential Gauss-Seidel semantics.

Unlike PGS contact solve (which iterates `config.iterations` times per substep), joint solve runs **exactly once per substep** — the correction is a Baumgarte-stabilised projection, not a Gauss-Seidel iteration. `dispatch_joint_solve_iteration(dt)` is called once and the compliance stabilisation term (`compliance / dt_sq`) provides the softness knob.

### Bindings

- `@group(0) @binding(0)` — `joints: array<BallSocketJointGpu>` (read).
- `@group(0) @binding(1)` — `body_positions: array<Vec3FixGpu>` (read_write). In-place update.
- `@group(0) @binding(2)` — `body_rotations: array<QuatFixGpu>` (read).
- `@group(0) @binding(3)` — `body_inv_masses: array<Fix128Gpu>` (read).
- `@group(0) @binding(4)` — `params: BallSocketSolveParams { joint_count, _pad0/1/2, dt_sq }` (uniform).

Precomputing `dt_sq = dt * dt` on the CPU saves one Fix128 mul per joint inside the kernel. The rest of the compliance term (`compliance / dt_sq`) still runs per joint on the GPU because `compliance` is a per-joint field.

### Non-Ball joint variants

`TrtSolverAdapter::send_joints` **panics** with a "not implemented" message when the joint list contains any variant other than `Joint::Ball`. This is intentional fail-fast per the CLAUDE.md「仮実装完了偽装の禁止ルール」 — silent skip would ship a bridge that produces wrong physics without complaining. Callers with mixed joint types should either:

1. Detach the bridge (`world.take_gpu_solver_bridge()`) before calling `step`, so joint solve falls back to CPU which handles all variants; or
2. Wait for the coordinated minor releases that add each variant's GPU kernel (see roadmap below).

### Byte-exact goldens

Three tests in `physics_bridge::solver_bridge::tests`:

- **`ball_socket_joint_solve_single_joint_matches_cpu_golden`** — 2 bodies 1 unit apart, zero local anchors. Exercises the core code path (quaternion rotation of zero → zero, sqrt, correction).
- **`ball_socket_joint_solve_chain_4_body_3_joint_matches_cpu_golden`** — 4 bodies at x = 0, 2, 5, 8 with joints (0-1), (1-2), (2-3). Verifies sequential dispatch preserves the CPU insertion-major order across bodies that appear in multiple joints.
- **`ball_socket_joint_solve_disconnected_2_island_matches_cpu_golden`** — 4 bodies in 2 disjoint pairs. Verifies the second joint's solve does not interfere with the first's updates.

### alice-physics v0.12.0 coordination

- `GpuSolverBridge` trait gains `send_joints` / `send_body_rotations` / `dispatch_joint_solve_iteration` (default `panic!`).
- `PhysicsWorld::solve_joints_with_bridge<B>` — explicit-control entry point that uploads bodies + joints + rotations, dispatches, reads back positions.
- `PhysicsWorld::solve_joints_dispatch(dt)` — private helper called from `substep` / `substep_batched` / `substep_with_bridge`. Auto-routes through the installed bridge if any; falls back to the free-function `solve_joints` otherwise.

Installing a `TrtSolverAdapter` (v3.1.0+) on `PhysicsWorld::set_gpu_solver_bridge` now auto-routes **both** contact solve (v0.11.0) **and** joint solve (v0.12.0) on subsequent `step` / `substep` calls.

## §3 — Forward roadmap

Each variant follows the same v3.1.0 pattern: WGSL kernel byte-exact vs the CPU reference, 3 byte-exact goldens, adapter fail-fast on other variants until the corresponding release ships. Estimated per-variant effort assumes the ball-socket infrastructure (adapter joint pipeline state, upload / dispatch / readback plumbing, trait method plumbing) is reused.

### v3.2.0 — HingeJoint (1 DOF rotational + 3 DOF positional)

**Reference**: `alice_physics::joint::solve_hinge_joint`. Adds an angular constraint on top of the ball-socket positional constraint. Requires cross product and angular compliance, plus `body.inv_inertia.length()` (a per-body scalar the adapter needs to upload alongside positions/rotations/inv_masses).

**New Rust struct**: `HingeJointGpu` — includes `local_axis_a`, `local_axis_b`, `angular_compliance`.

**Estimated effort**: 8-10h. Angular compliance kernel adds ~150 WGSL lines.

### v3.3.0 — FixedJoint (0 DOF, weld constraint)

**Reference**: `alice_physics::joint::solve_fixed_joint`. Positional constraint + full angular alignment (three orthogonal axis constraints instead of one). Simpler algorithm than hinge because there's no free axis to preserve.

**New Rust struct**: `FixedJointGpu` — no axis field, but stores the initial relative orientation to constrain against.

**Estimated effort**: 6-8h.

### v3.4.0 — SliderJoint (1 DOF translational + 3 DOF rotational)

**Reference**: `alice_physics::joint::solve_slider_joint`. Constrains bodies along a shared axis with free rotation. Analogous to hinge but the DOF is translational instead of rotational.

**Estimated effort**: 8-10h.

### v3.5.0 — SpringJoint (compliant distance constraint)

**Reference**: `alice_physics::joint::solve_spring_joint`. Distance constraint with rest length and target compliance. Effectively a compliant ball-socket with a non-zero rest length; the kernel is a ~50-line adaptation of the v3.1.0 ball-socket kernel.

**Estimated effort**: 4-6h (cheapest of the follow-up variants because the delta from v3.1.0 is small).

### v3.6.0 — D6Joint (6-DOF configurable)

**Reference**: `alice_physics::joint::solve_d6_joint`. General configurable joint with per-axis linear and angular limits. The most complex to port — the CPU reference walks each of 6 axes conditionally based on limit mode (locked / limited / free).

**Estimated effort**: 16-20h. May be split into multiple sub-releases (v3.6.0-alpha for pure locked, v3.6.0-beta for limited, v3.6.0 for free).

### v3.7.0 — ConeTwistJoint (ragdoll)

**Reference**: `alice_physics::joint::solve_cone_twist_joint`. Ball-socket + cone limit + twist limit. Common in ragdoll simulations. Combines the v3.1.0 ball-socket kernel with two additional angular constraint kernels (cone limit + twist).

**Estimated effort**: 10-14h.

## §4 — Non-goals for v3.1.0

- **Batched dispatch**: v3.1.0 ships sequential single-thread dispatch. A colour-parallel batched variant (analogous to v2.8.0 for contact solve) is a candidate for v3.1.x once profiling identifies joint-solve as a bottleneck. Not shipped in the initial cut because joint counts in typical rigid-body sims (ragdolls: 15-25 joints, ropes: 20-100 joints) are well below the crossover point where dispatch overhead dominates.
- **Break-force checking**: `BallJoint::break_force` is uploaded through the wire but the v3.1.0 kernel does not enforce it. Break checking runs on the CPU after readback — `alice_physics::solver::solve_joints_breakable` handles this on the CPU side, and callers who need break enforcement should route through that entry point instead of relying on the bridge.
- **Angular constraints on ball-socket**: `BallJoint` has no angular constraint by design (that's the "socket" — free rotation). Callers who need positional + angular constraint should use `HingeJoint` (single-axis) or `FixedJoint` (full lock), both shipping in later minor releases.

## §5 — Related documents

- `CHANGELOG.md` §[3.1.0] — release notes with commit hashes and test counts.
- alice-physics `CHANGELOG.md` §[0.12.0] — coordinated trait extension.
- `docs/MIGRATION_v3.md` — v2 → v3 migration guide (Arc<GpuDevice> refactor from v3.0.0 — unchanged in v3.1.0).
- `~/claude-config/memory/project_alice_trt_roadmap_post_v2_7_1.md` §Tier C — the original design brief that motivates Phase 4.
