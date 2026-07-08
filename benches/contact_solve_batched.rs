//! v2.8.0 Fix128 PGS contact-solve batched (colour-parallel) dispatch
//! benchmarks.
//!
//! Compares the v2.6.0 sequential path (single-workgroup single-thread
//! `for i in 0..count` loop) against the v2.8.0 batched path (one
//! workgroup per constraint, dispatched one colour at a time). Both
//! paths process the same input fixture; the batched path pays an
//! extra O(N²) `ConstraintGraph::build` + greedy colouring cost every
//! iteration in exchange for GPU parallelism within a colour.
//!
//! Two topologies are covered:
//!
//! - **chain** — bipartite chain of `N` constraints connecting `N+1`
//!   bodies in sequence. Greedy colouring yields exactly 2 colours,
//!   so batched dispatches ≈ `N/2` workgroups per colour.
//! - **random-graph** — a deterministic pseudo-random constraint
//!   graph with target density that greedy-colours into ≈ 5 buckets.
//!   Representative of ragdoll and multi-body contact pileups.
//!
//! Each topology is measured at `N ∈ {100, 1000, 10000}` constraints
//! for 4 PGS iterations (typical solver setting). Skips gracefully if
//! no GPU adapter is available.
//!
//! Run with:
//!
//! ```text
//! cargo bench --features physics-solver --bench contact_solve_batched
//! ```
//!
//! Numbers are meaningful only on real hardware; software adapters
//! (WARP / lavapipe) will produce non-representative wall-clock times.

#![cfg(feature = "physics-solver")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use alice_physics::collider::Contact;
use alice_physics::gpu_bridge::GpuSolverBridge;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::ContactConstraint;
use alice_trt::physics_bridge::TrtSolverAdapter;
use alice_trt::GpuDevice;

const ITERS_PER_STEP: u32 = 4;

/// Build a bipartite chain fixture with `n_constraints` constraints
/// spanning `n_constraints + 1` bodies. Constraint `i` connects body
/// `i` to body `i + 1`, so the greedy colouring places even-indexed
/// constraints in colour 0 and odd-indexed constraints in colour 1.
fn build_chain(n_constraints: usize) -> (Vec<[Fix128; 3]>, Vec<Fix128>, Vec<ContactConstraint>) {
    let n_bodies = n_constraints + 1;
    let positions: Vec<[Fix128; 3]> = (0..n_bodies)
        .map(|i| [Fix128::from_int(i as i64 * 2), Fix128::ZERO, Fix128::ZERO])
        .collect();
    let inv_masses: Vec<Fix128> = (0..n_bodies).map(|_| Fix128::from_int(1)).collect();
    let constraints: Vec<ContactConstraint> = (0..n_constraints)
        .map(|i| ContactConstraint {
            body_a: i,
            body_b: i + 1,
            contact: Contact {
                depth: Fix128::from_ratio(2, 10),
                normal: Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            },
            friction: Fix128::from_ratio(3, 10),
            restitution: Fix128::from_ratio(2, 10),
            cached_lambda: Fix128::ZERO,
        })
        .collect();
    (positions, inv_masses, constraints)
}

/// Deterministic splitmix64 PRNG, seeded from the caller. Same seed
/// yields the same fixture across benchmark reruns and platforms.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a pseudo-random constraint graph on `n_bodies` bodies with
/// `n_constraints` edges. Each edge picks two distinct bodies via
/// splitmix64. The resulting graph is dense enough that greedy
/// colouring typically produces 4-6 buckets for the sizes used here.
fn build_random_graph(
    n_bodies: usize,
    n_constraints: usize,
    seed: u64,
) -> (Vec<[Fix128; 3]>, Vec<Fix128>, Vec<ContactConstraint>) {
    let positions: Vec<[Fix128; 3]> = (0..n_bodies)
        .map(|i| {
            [
                Fix128::from_int(i as i64 % 32),
                Fix128::from_int((i as i64 / 32) % 32),
                Fix128::from_int(i as i64 / 1024),
            ]
        })
        .collect();
    let inv_masses: Vec<Fix128> = (0..n_bodies).map(|_| Fix128::from_int(1)).collect();

    let mut prng_state = seed;
    let n_bodies_u64 = n_bodies as u64;
    let constraints: Vec<ContactConstraint> = (0..n_constraints)
        .map(|_| {
            let mut a = (splitmix64(&mut prng_state) % n_bodies_u64) as usize;
            let mut b = (splitmix64(&mut prng_state) % n_bodies_u64) as usize;
            if a == b {
                b = (a + 1) % n_bodies;
            }
            if b < a {
                std::mem::swap(&mut a, &mut b);
            }
            ContactConstraint {
                body_a: a,
                body_b: b,
                contact: Contact {
                    depth: Fix128::from_ratio(2, 10),
                    normal: Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
                    point_a: Vec3Fix::ZERO,
                    point_b: Vec3Fix::ZERO,
                },
                friction: Fix128::from_ratio(3, 10),
                restitution: Fix128::from_ratio(2, 10),
                cached_lambda: Fix128::ZERO,
            }
        })
        .collect();
    (positions, inv_masses, constraints)
}

/// Drive the adapter through the `GpuSolverBridge` trait for
/// `ITERS_PER_STEP` PGS iterations, returning nothing (readback is
/// included in the timing to represent a full frame cost).
fn run_pgs(
    device: &std::sync::Arc<GpuDevice>,
    parallel: bool,
    constraints: &[ContactConstraint],
    positions: &[[Fix128; 3]],
    inv_masses: &[Fix128],
) {
    let mut adapter = TrtSolverAdapter::new(std::sync::Arc::clone(device));
    adapter.set_parallel_contact_solve(parallel);
    let warm_start_factor = Fix128::from_ratio(85, 100);
    let bridge: &mut dyn GpuSolverBridge = &mut adapter;
    bridge.send_contact_constraints(constraints);
    bridge.send_body_state(positions, inv_masses);
    for _ in 0..ITERS_PER_STEP {
        bridge.dispatch_contact_solve_iteration(warm_start_factor);
    }
    let mut out_c: Vec<ContactConstraint> = constraints.to_vec();
    let mut out_p: Vec<[Fix128; 3]> = positions.to_vec();
    bridge.recv_contact_constraints(&mut out_c);
    bridge.recv_body_positions(&mut out_p);
    black_box(&out_c);
    black_box(&out_p);
}

fn bench_chain(c: &mut Criterion) {
    let Ok(device) = GpuDevice::new() else {
        eprintln!("No GPU available, skipping contact_solve_batched chain bench");
        return;
    };
    let device = std::sync::Arc::new(device);
    let mut group = c.benchmark_group("contact_solve_chain");

    for n_constraints in [100usize, 1000, 10000] {
        let (positions, inv_masses, constraints) = build_chain(n_constraints);

        group.bench_with_input(
            BenchmarkId::new("sequential", n_constraints),
            &n_constraints,
            |b, _| {
                b.iter(|| {
                    run_pgs(&device, false, &constraints, &positions, &inv_masses);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("batched", n_constraints),
            &n_constraints,
            |b, _| {
                b.iter(|| {
                    run_pgs(&device, true, &constraints, &positions, &inv_masses);
                });
            },
        );
    }
    group.finish();
}

fn bench_random_graph(c: &mut Criterion) {
    let Ok(device) = GpuDevice::new() else {
        return;
    };
    let device = std::sync::Arc::new(device);
    let mut group = c.benchmark_group("contact_solve_random_graph");

    for n_constraints in [100usize, 1000, 10000] {
        // Body count sized so the graph is dense enough to force
        // several colours but not so tight that greedy hits the
        // chromatic-number ceiling.
        let n_bodies = n_constraints / 5 + 10;
        let (positions, inv_masses, constraints) =
            build_random_graph(n_bodies, n_constraints, 0xA11CE_DEAD_BEEF);

        group.bench_with_input(
            BenchmarkId::new("sequential", n_constraints),
            &n_constraints,
            |b, _| {
                b.iter(|| {
                    run_pgs(&device, false, &constraints, &positions, &inv_masses);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("batched", n_constraints),
            &n_constraints,
            |b, _| {
                b.iter(|| {
                    run_pgs(&device, true, &constraints, &positions, &inv_masses);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_chain, bench_random_graph);
criterion_main!(benches);
