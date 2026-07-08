//! Fix128 PGS live dispatch benchmarks (v1.0.0).
//!
//! Measures throughput of the [`TrtSolverAdapter`] on the local GPU
//! across three axes:
//!
//! - **Iteration count** — how many `dispatch_iterations` cycles per
//!   second at a fixed body count. This proves that the per-iteration
//!   round-trip overhead (encoder submit + `poll_wait` + buffer readback)
//!   does not dominate the compute for typical solver step counts.
//! - **Body count** — how throughput scales as N grows. On real GPU
//!   hardware the kernel should stay compute-bound well before the
//!   dispatch overhead starts to matter.
//! - **Kernel composition** — the marginal cost of enabling gravity
//!   (v0.9.1) and floor projection (v0.9.2) on top of the v0.9.0
//!   `p += v * dt` baseline.
//!
//! Skips gracefully when no GPU adapter is available, so it can also
//! run on CI without emitting throughput numbers.
//!
//! Run with:
//!
//! ```text
//! cargo bench --features physics-solver --bench pgs_dispatch
//! ```
//!
//! Numbers are meaningful only on real hardware. Software adapters
//! (WARP / lavapipe) will produce much larger wall-clock times that
//! are not representative.

#![cfg(feature = "physics-solver")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use alice_physics::gpu_bridge::GpuSolverBridge;
use alice_physics::math::Fix128;
use alice_trt::physics_bridge::TrtSolverAdapter;
use alice_trt::GpuDevice;

fn build_island(n: usize) -> (Vec<[Fix128; 3]>, Vec<[Fix128; 3]>) {
    let positions: Vec<[Fix128; 3]> = (0..n)
        .map(|i| {
            [
                Fix128::from_int(i as i64),
                Fix128::from_int(1 + (i as i64 % 5)),
                Fix128::from_int(-(i as i64 % 7)),
            ]
        })
        .collect();
    let velocities: Vec<[Fix128; 3]> = (0..n)
        .map(|i| {
            [
                Fix128::from_int(i as i64 % 3),
                Fix128::ZERO,
                Fix128::from_int(-(i as i64 % 2)),
            ]
        })
        .collect();
    (positions, velocities)
}

fn bench_iters(c: &mut Criterion) {
    let Ok(device) = GpuDevice::new() else {
        eprintln!("No GPU available, skipping PGS dispatch benchmarks");
        return;
    };
    let device = std::sync::Arc::new(device);
    let mut group = c.benchmark_group("pgs_iters_at_n_256");

    let (positions, velocities) = build_island(256);
    let dt = Fix128::from_ratio(1, 60);

    for iters in [1u32, 4, 16, 64, 256] {
        group.bench_with_input(BenchmarkId::new("iters", iters), &iters, |b, &iters| {
            b.iter(|| {
                let mut adapter = TrtSolverAdapter::new(std::sync::Arc::clone(&device));
                adapter.send_island(&positions, &velocities);
                adapter.dispatch_iterations(iters, dt);
                black_box(&adapter);
            });
        });
    }
    group.finish();
}

fn bench_bodies(c: &mut Criterion) {
    let Ok(device) = GpuDevice::new() else {
        return;
    };
    let device = std::sync::Arc::new(device);
    let mut group = c.benchmark_group("pgs_bodies_at_16_iters");

    let dt = Fix128::from_ratio(1, 60);
    for n in [64usize, 256, 1024, 4096, 16384] {
        let (positions, velocities) = build_island(n);
        group.bench_with_input(BenchmarkId::new("bodies", n), &n, |b, _n| {
            b.iter(|| {
                let mut adapter = TrtSolverAdapter::new(std::sync::Arc::clone(&device));
                adapter.send_island(&positions, &velocities);
                adapter.dispatch_iterations(16, dt);
                black_box(&adapter);
            });
        });
    }
    group.finish();
}

fn bench_kernel_compositions(c: &mut Criterion) {
    let Ok(device) = GpuDevice::new() else {
        return;
    };
    let device = std::sync::Arc::new(device);
    let mut group = c.benchmark_group("pgs_kernel_compositions");

    let (positions, velocities) = build_island(1024);
    let dt = Fix128::from_ratio(1, 60);

    // v0.9.0 baseline — no gravity, no floor: pure `p += v * dt`.
    group.bench_function("v0_9_0_baseline_p_plus_v_dt", |b| {
        b.iter(|| {
            let mut adapter = TrtSolverAdapter::new(std::sync::Arc::clone(&device));
            adapter.send_island(&positions, &velocities);
            adapter.dispatch_iterations(16, dt);
            black_box(&adapter);
        });
    });

    // v0.9.1 — gravity added (velocities become read/write).
    group.bench_function("v0_9_1_with_gravity", |b| {
        b.iter(|| {
            let mut adapter = TrtSolverAdapter::new(std::sync::Arc::clone(&device));
            adapter.set_gravity([Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO]);
            adapter.send_island(&positions, &velocities);
            adapter.dispatch_iterations(16, dt);
            black_box(&adapter);
        });
    });

    // v0.9.2 — gravity + floor: two kernels per iteration.
    group.bench_function("v0_9_2_with_gravity_and_floor", |b| {
        b.iter(|| {
            let mut adapter = TrtSolverAdapter::new(std::sync::Arc::clone(&device));
            adapter.set_gravity([Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO]);
            adapter.set_floor_enabled(true);
            adapter.send_island(&positions, &velocities);
            adapter.dispatch_iterations(16, dt);
            black_box(&adapter);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_iters,
    bench_bodies,
    bench_kernel_compositions
);
criterion_main!(benches);
