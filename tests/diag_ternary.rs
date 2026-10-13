//! Diagnostic-only (not for main): one ternary GPU op on one shape per process,
//! selected by TRT_DIAG_OUT / TRT_DIAG_INP / TRT_DIAG_OP, with step markers on
//! stderr so a process crash shows the last step reached.

use alice_trt::prelude::*;
use alice_trt::TernaryCompute;
use std::io::Write;

fn step(s: &str) {
    eprintln!("[diag] {s}");
    std::io::stderr().flush().ok();
}

#[test]
fn diag_one_op() {
    let out: usize = std::env::var("TRT_DIAG_OUT").unwrap().parse().unwrap();
    let inp: usize = std::env::var("TRT_DIAG_INP").unwrap().parse().unwrap();
    let op = std::env::var("TRT_DIAG_OP").unwrap();
    step(&format!("device {out}x{inp} {op}"));
    let dev = GpuDevice::new().expect("adapter");
    step("compute");
    let compute = TernaryCompute::new(&dev);
    let w: Vec<i8> = (0..out * inp).map(|i| [(-1i8), 0, 1][i % 3]).collect();
    let x: Vec<f32> = (0..inp).map(|i| (i % 7) as f32 - 3.0).collect();
    step("weights");
    let weights = GpuTernaryWeight::from_ternary(&dev, &w, out, inp);
    let input = GpuTensor::from_f32(&dev, &x, &[inp]);
    match op.as_str() {
        "matvec" => {
            step("matvec dispatch");
            let y = compute.matvec(&dev, &input, &weights);
            step("poll");
            dev.poll_wait();
            step("download");
            let y = y.download(&dev);
            step(&format!("done len {}", y.len()));
        }
        "batch" => {
            let batch = 5;
            let xb: Vec<f32> = (0..batch * inp).map(|i| (i % 5) as f32 - 2.0).collect();
            let input_b = GpuTensor::from_f32(&dev, &xb, &[batch, inp]);
            step("matmul_batch dispatch");
            let y = compute.matmul_batch(&dev, &input_b, &weights, batch);
            step("poll");
            dev.poll_wait();
            step("download");
            let y = y.download(&dev);
            step(&format!("done len {}", y.len()));
        }
        _ => panic!("op"),
    }
}
