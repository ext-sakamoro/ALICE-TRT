//! GPU Ternary MatMul Benchmarks
//!
//! Compares GPU (ALICE-TRT) vs CPU (ALICE-ML) inference throughput.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_gpu_matvec(c: &mut Criterion) {
    let device = match alice_trt::GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("No GPU available, skipping GPU benchmarks: {e}");
            return;
        }
    };

    let compute = alice_trt::TernaryCompute::new(&device);

    let mut group = c.benchmark_group("gpu_matvec");

    for size in [64, 256, 512, 1024, 2048, 4096] {
        let values: Vec<i8> = (0..size * size).map(|i| (i % 3) as i8 - 1).collect();
        let input_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();

        let gpu_weights = alice_trt::GpuTernaryWeight::from_ternary(&device, &values, size, size);
        let gpu_input = alice_trt::GpuTensor::from_f32(&device, &input_data, &[size]);

        group.bench_with_input(
            BenchmarkId::new("gpu", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let output = compute.matvec(&device, &gpu_input, &gpu_weights);
                    device.poll_wait();
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

fn bench_cpu_vs_gpu(c: &mut Criterion) {
    let device = match alice_trt::GpuDevice::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    let compute = alice_trt::TernaryCompute::new(&device);

    let mut group = c.benchmark_group("cpu_vs_gpu");

    for size in [256, 1024, 4096] {
        let values: Vec<i8> = (0..size * size).map(|i| (i % 3) as i8 - 1).collect();
        let input_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();

        // GPU
        let gpu_weights = alice_trt::GpuTernaryWeight::from_ternary(&device, &values, size, size);
        let gpu_input = alice_trt::GpuTensor::from_f32(&device, &input_data, &[size]);

        group.bench_with_input(
            BenchmarkId::new("gpu", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let output = compute.matvec(&device, &gpu_input, &gpu_weights);
                    device.poll_wait();
                    black_box(output);
                });
            },
        );

        // CPU (ALICE-ML)
        let cpu_weights = alice_ml::TernaryWeight::from_ternary(&values, size, size);
        let mut cpu_output = vec![0.0f32; size];

        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            &size,
            |b, _| {
                b.iter(|| {
                    alice_ml::ternary_matvec(
                        black_box(&input_data),
                        &cpu_weights,
                        &mut cpu_output,
                    );
                    black_box(&cpu_output);
                });
            },
        );
    }

    group.finish();
}

fn bench_gpu_batch(c: &mut Criterion) {
    let device = match alice_trt::GpuDevice::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    let compute = alice_trt::TernaryCompute::new(&device);

    let mut group = c.benchmark_group("gpu_batch_matmul");

    let size = 512;
    let values: Vec<i8> = (0..size * size).map(|i| (i % 3) as i8 - 1).collect();
    let gpu_weights = alice_trt::GpuTernaryWeight::from_ternary(&device, &values, size, size);

    for batch_size in [1, 4, 16, 64, 256] {
        let input_data: Vec<f32> = (0..batch_size * size)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let gpu_input = alice_trt::GpuTensor::from_f32(&device, &input_data, &[batch_size, size]);

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let output = compute.matmul_batch(
                        &device,
                        &gpu_input,
                        &gpu_weights,
                        batch_size,
                    );
                    device.poll_wait();
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gpu_matvec, bench_cpu_vs_gpu, bench_gpu_batch);
criterion_main!(benches);
