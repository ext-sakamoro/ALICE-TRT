# ALICE-TRT v3.0.0 Migration Guide

**From**: v2.x (any 2.x release, most recently v2.8.1)
**To**: v3.0.0
**Impact**: One breaking change to `TrtSolverAdapter::new` and the adapter type parameter. Mechanical, ~28 call-site update in the average deployment.

## What changed

`TrtSolverAdapter` no longer carries a lifetime and no longer borrows the `GpuDevice`. It now owns an `Arc<GpuDevice>` internally.

| API | v2.x | v3.0.0 |
|---|---|---|
| Type | `TrtSolverAdapter<'a>` | `TrtSolverAdapter` |
| Field | `device: &'a GpuDevice` | `device: Arc<GpuDevice>` |
| Constructor | `pub fn new(device: &'a GpuDevice) -> Self` | `pub fn new(device: Arc<GpuDevice>) -> Self` |
| Trait impl | `impl GpuSolverBridge for TrtSolverAdapter<'_>` | `impl GpuSolverBridge for TrtSolverAdapter` |
| Send + Sync | Requires manual analysis | Statically guaranteed (compile-time asserted in the lib test suite) |

All other public API (`send_island` / `dispatch_iterations` / `recv_island`, distance-constraint helpers, `set_parallel_dispatch`, `set_parallel_contact_solve`, the whole `GpuSolverBridge` trait impl) is unchanged and works the same way at the same call sites.

## Why the change

- **Enables the alice-physics v0.11.0 auto-routing pattern.** `PhysicsWorld::set_gpu_solver_bridge` requires `Box<dyn GpuSolverBridge + Send + Sync>`. That bound is impossible to satisfy with a borrowed `&'a GpuDevice` because the box would need to outlive whatever holds the reference. `Arc<GpuDevice>` makes the adapter `'static` + `Send + Sync`, so it fits.
- **Removes the lifetime infection from downstream types.** In v2.x, any struct that stored a `TrtSolverAdapter<'a>` had to propagate the lifetime up. `Arc<GpuDevice>` breaks that chain — downstream types can store `TrtSolverAdapter` in a `Box`, `Arc`, or plain field with no lifetime bound.
- **Better fits the "game-engine wrapper" use case.** Game engines typically own the `GpuDevice` centrally and hand out cheap `Arc` clones to subsystems (physics adapter, renderer, audio raymarch, etc.). v2.x forced these consumers to keep the `GpuDevice` alive in the exact scope where the adapter lived; v3.0.0 lets them share it.

## Mechanical upgrade

### Pattern 1 — single adapter per test / fixture

**Before (v2.x)**:

```rust
let device = match GpuDevice::new() {
    Ok(d) => d,
    Err(_) => return, // No GPU on this CI runner
};
let mut adapter = TrtSolverAdapter::new(&device);
```

**After (v3.0.0)**:

```rust
let device = match GpuDevice::new() {
    Ok(d) => d,
    Err(_) => return,
};
let device = std::sync::Arc::new(device);
let mut adapter = TrtSolverAdapter::new(std::sync::Arc::clone(&device));
```

The extra `Arc::new` line and the `Arc::clone` at the constructor are the only diffs. The rest of the test body is identical.

### Pattern 2 — multiple adapters reusing the same device

Common in comparative tests (sequential vs batched, toggle-on vs toggle-off).

**Before (v2.x)**:

```rust
let device = GpuDevice::new().unwrap();
let mut seq = TrtSolverAdapter::new(&device);
let mut par = TrtSolverAdapter::new(&device);
```

**After (v3.0.0)**:

```rust
let device = std::sync::Arc::new(GpuDevice::new().unwrap());
let mut seq = TrtSolverAdapter::new(std::sync::Arc::clone(&device));
let mut par = TrtSolverAdapter::new(std::sync::Arc::clone(&device));
```

Each `Arc::clone` is a single atomic-refcount increment — no `GpuDevice` copies, no shader recompilation, no pipeline recreation. The two adapters share the same underlying device exactly as in v2.x.

### Pattern 3 — benchmark helper taking device by reference

**Before (v2.x)**:

```rust
fn run_pgs(device: &GpuDevice, /* ... */) {
    let mut adapter = TrtSolverAdapter::new(device);
    /* ... */
}

// caller
let Ok(device) = GpuDevice::new() else { return; };
run_pgs(&device, /* ... */);
```

**After (v3.0.0)**:

```rust
fn run_pgs(device: &std::sync::Arc<GpuDevice>, /* ... */) {
    let mut adapter = TrtSolverAdapter::new(std::sync::Arc::clone(device));
    /* ... */
}

// caller
let Ok(device) = GpuDevice::new() else { return; };
let device = std::sync::Arc::new(device);
run_pgs(&device, /* ... */);
```

The helper signature gains `&Arc<GpuDevice>` (borrow of the Arc, no clone at the call boundary). The `Arc::clone` inside `run_pgs` costs one atomic increment per adapter, which is negligible vs the buffer uploads and dispatch that dominate every benchmark iteration.

### Pattern 4 — game-engine style shared device across subsystems

Not present in the ALICE-TRT test suite but the canonical use case for v3.0.0:

```rust
// v3.0.0 game engine host
struct Engine {
    device: std::sync::Arc<GpuDevice>,
    physics: TrtSolverAdapter,          // no lifetime!
    renderer: MyRenderer,               // uses Arc::clone(&device) internally
    // ...
}

impl Engine {
    fn new() -> Result<Self, DeviceError> {
        let device = std::sync::Arc::new(GpuDevice::new()?);
        Ok(Self {
            physics: TrtSolverAdapter::new(std::sync::Arc::clone(&device)),
            renderer: MyRenderer::new(std::sync::Arc::clone(&device))?,
            device,
        })
    }
}
```

## alice-physics coordination

v3.0.0 is coordinated with **alice-physics v0.11.0**, which introduces the field-based auto-routing:

```rust
use alice_physics::{PhysicsWorld, PhysicsConfig};
use alice_trt::physics_bridge::TrtSolverAdapter;

let device = std::sync::Arc::new(alice_trt::GpuDevice::new()?);
let adapter = TrtSolverAdapter::new(std::sync::Arc::clone(&device));

let mut world = PhysicsWorld::new(PhysicsConfig::default());
world.set_gpu_solver_bridge(Some(Box::new(adapter)));

// Every subsequent step now transparently routes contact-solve
// through the GPU adapter. No `_with_bridge` call needed.
let dt = alice_physics::Fix128::from_ratio(1, 60);
world.step(dt);
```

The `Box::new(adapter)` here upcasts to `Box<dyn GpuSolverBridge + Send + Sync>` — the two bounds are what v3.0.0 delivers.

Callers who prefer explicit control can continue to use the v0.10.0 `step_with_bridge` helper without installing the bridge on the world; the helper ignores the installed bridge and uses the borrowed one passed in.

## Send + Sync verification

v3.0.0 includes a compile-time assertion in the lib test suite:

```rust
// physics_bridge::solver_bridge::tests
#[test]
fn trt_solver_adapter_is_send_and_sync_v3() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TrtSolverAdapter>();
}
```

If a future field addition breaks Send or Sync, this test fails at compile time. That is the earliest possible signal — the failure surfaces on the developer's local `cargo test` before CI, and the message points directly at the offending field.

## What is NOT changed

- Every non-`new` method on `TrtSolverAdapter` has the same signature and behaviour.
- The `GpuSolverBridge` trait impl surface is unchanged.
- All existing tests other than the ~22 test call sites that construct a `TrtSolverAdapter` (which are all in `src/physics_bridge.rs` `mod tests`) work bit-for-bit identical.
- All 221 v2.8.1 byte-exact CPU-GPU goldens pass on v3.0.0 (+1 new Send + Sync compile-time test = 222 total).
- The v2.8.1 3-platform physics-solver CI lane is unchanged and confirms the byte-exact contract survives the refactor.

## Follow-up idioms

- Prefer `Arc<GpuDevice>` at the boundary of any struct that stores an adapter. Storing `TrtSolverAdapter` inside your struct is now free of lifetime propagation.
- Wrap `GpuDevice::new()` in `Arc::new(...)` at the top of your factory or engine setup once, then hand out clones to every subsystem that needs GPU access — physics, renderer, audio, whatever.
- If you need to swap or reset the device (unusual — most apps have one for the process lifetime), `take_gpu_solver_bridge()` on the world and rebuild the adapter with a fresh `Arc`.

## Related documents

- `CHANGELOG.md` §[3.0.0] — full release notes with commit hash, test count, and CI matrix status.
- alice-physics `CHANGELOG.md` §[0.11.0] — coordinated companion release.
- `~/claude-config/memory/project_alice_trt_roadmap_post_v2_7_1.md` §Tier S — the original design brief that this release delivers.
