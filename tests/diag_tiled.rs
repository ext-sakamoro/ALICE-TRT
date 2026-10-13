//! Diagnostic-only (not for main): variants of the tiled ternary matvec kernel,
//! one per process via TRT_DIAG_VARIANT, to find which construct faults.

use alice_trt::kernel::TERNARY_MATVEC_TILED_SHADER as V1;
use alice_trt::prelude::*;
use std::io::Write;

fn step(s: &str) {
    eprintln!("[diag] {s}");
    std::io::stderr().flush().ok();
}

const MINIMAL: &str = r"
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial_sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    partial_sums[lid.x] = f32(lid.x);
    workgroupBarrier();
    if (lid.x == 0u) { output[0] = partial_sums[255]; }
}
";

fn variant(name: &str) -> String {
    let v1 = V1.to_string();
    match name {
        "v1" => v1,
        // early return removed: out-of-range workgroups compute nothing, write guarded
        "v2" => v1
            .replace("    let row = wgid.x;\n    if (row >= params.out_features) {\n        return;\n    }\n", "    let row = min(wgid.x, params.out_features - 1u);\n"),
        // one barrier, thread 0 sums all partials (no barrier inside a loop)
        "v3" => {
            let start = v1.find("    // Parallel reduction").unwrap();
            let end = v1.find("    // Thread 0 writes final result").unwrap();
            let mut s = v1.clone();
            s.replace_range(start..end, "");
            s.replace(
                "        output[row] = partial_sums[0] * params.scale;",
                "        var t: f32 = 0.0;\n        for (var i: u32 = 0u; i < 256u; i = i + 1u) { t = t + partial_sums[i]; }\n        output[row] = t * params.scale;",
            )
        }
        // workgroup of 64 with a 64-slot array, strides of 64
        "v5" => v1
            .replace("array<f32, 256>", "array<f32, 64>")
            .replace("@workgroup_size(256)", "@workgroup_size(64)")
            .replace("col = col + 256u;", "col = col + 64u;")
            .replace("var stride: u32 = 128u;", "var stride: u32 = 32u;"),
        "minimal" => MINIMAL.to_string(),
        _ => panic!("variant"),
    }
}

#[test]
fn diag_tiled_variant() {
    let name = std::env::var("TRT_DIAG_VARIANT").unwrap();
    if name != "v1" && name != "minimal" {
        assert_ne!(
            variant(&name),
            V1,
            "variant {name} did not change the kernel"
        );
    }
    let (out, inp) = (3u32, 1024u32);
    let words = inp.div_ceil(32);
    step(&format!("device {name}"));
    let dev = GpuDevice::new().expect("adapter");
    let d = dev.device();
    let src = variant(&name);
    step("shader");
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("diag"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    step("pipeline");
    let pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("diag"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let x: Vec<f32> = (0..inp).map(|i| (i % 7) as f32 - 3.0).collect();
    let plus: Vec<u32> = (0..out * words)
        .map(|i| 0x5555_5555u32.rotate_left(i))
        .collect();
    let minus: Vec<u32> = plus.iter().map(|p| !p & 0x0F0F_0F0F).collect();
    let params: [u32; 8] = [inp, out, words, 1.0f32.to_bits(), 1, 0, 0, 0];
    let b_in = dev.create_buffer_init("in", bytemuck::cast_slice(&x));
    let b_p = dev.create_buffer_init("p", bytemuck::cast_slice(&plus));
    let b_m = dev.create_buffer_init("m", bytemuck::cast_slice(&minus));
    let b_out = dev.create_buffer_empty("out", u64::from(out) * 4);
    let b_par = dev.create_uniform_buffer("par", bytemuck::cast_slice(&params));
    let layout = pipe.get_bind_group_layout(0);
    let all = [
        (0u32, &b_in),
        (1, &b_p),
        (2, &b_m),
        (3, &b_out),
        (4, &b_par),
    ];
    let used: &[u32] = if name == "minimal" {
        &[3]
    } else {
        &[0, 1, 2, 3, 4]
    };
    let entries: Vec<wgpu::BindGroupEntry> = all
        .iter()
        .filter(|(b, _)| used.contains(b))
        .map(|(b, buf)| wgpu::BindGroupEntry {
            binding: *b,
            resource: buf.as_entire_binding(),
        })
        .collect();
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    });
    let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(out, 1, 1);
    }
    step("submit");
    dev.submit(enc);
    step("poll");
    dev.poll_wait();
    step("download");
    let bytes = dev.read_buffer(&b_out, u64::from(out) * 4);
    let y: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    step(&format!("done {y:?}"));
}
