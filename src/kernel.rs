//! WGSL Compute Shaders: The Trojan Horse Kernels
//!
//! **The Hack**: Weights live as 2-bit packed bitplanes in VRAM.
//! On-the-fly decompression in registers. The GPU thinks it's doing
//! normal FP32 math. It doesn't know the weights are {-1, 0, +1}.
//!
//! Memory bandwidth reduction: 10x vs FP16, 5x vs INT8.

/// GPU-side parameters (must match WGSL struct layout exactly)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    pub in_features: u32,
    pub out_features: u32,
    pub words_per_row: u32,
    pub scale: f32,
    pub batch_size: u32,
    pub _pad: [u32; 3],
}

/// Ternary MatVec: One thread per output row (simple, low-latency)
///
/// Best for: small-to-medium layers (out_features < 4096)
///
/// Strategy:
/// - Each thread computes one output element
/// - Loops over all input columns
/// - Branchless: `acc += x * (f32(is_plus) - f32(is_minus))`
/// - The GPU's FMA unit processes {-1, 0, +1} × input at full throughput
pub const TERNARY_MATVEC_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    words_per_row: u32,
    scale: f32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> plus_bits: array<u32>;
@group(0) @binding(2) var<storage, read> minus_bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.out_features) {
        return;
    }

    var acc: f32 = 0.0;
    let row_offset = row * params.words_per_row;

    for (var w: u32 = 0u; w < params.words_per_row; w = w + 1u) {
        let pw = plus_bits[row_offset + w];
        let mw = minus_bits[row_offset + w];
        let base = w * 32u;

        // Unrolled inner loop: 32 weights per u32 word
        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
            let col = base + b;
            if (col >= params.in_features) {
                break;
            }

            let x = input[col];
            let is_plus = (pw >> b) & 1u;
            let is_minus = (mw >> b) & 1u;

            // Branchless ternary multiply-add
            // GPU sees: acc += x * weight  where weight ∈ {-1, 0, +1}
            // FMA unit fires at full throughput. No divergence.
            acc = acc + x * (f32(is_plus) - f32(is_minus));
        }
    }

    output[row] = acc * params.scale;
}
"#;

/// Ternary MatVec Tiled: Workgroup-cooperative with parallel reduction
///
/// Best for: large layers (out_features >= 4096, in_features >= 1024)
///
/// Strategy:
/// - Each workgroup handles one output row
/// - 256 threads split the input columns (stride pattern)
/// - Partial sums reduced via shared memory (log2 steps)
/// - Thread 0 writes final result
pub const TERNARY_MATVEC_TILED_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    words_per_row: u32,
    scale: f32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> plus_bits: array<u32>;
@group(0) @binding(2) var<storage, read> minus_bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    if (row >= params.out_features) {
        return;
    }

    let tid = lid.x;
    var local_acc: f32 = 0.0;
    let row_offset = row * params.words_per_row;

    // Strided column processing: thread i handles columns i, i+256, i+512, ...
    var col = tid;
    loop {
        if (col >= params.in_features) {
            break;
        }

        let word_idx = col / 32u;
        let bit_pos = col % 32u;

        let pw = plus_bits[row_offset + word_idx];
        let mw = minus_bits[row_offset + word_idx];

        let is_plus = (pw >> bit_pos) & 1u;
        let is_minus = (mw >> bit_pos) & 1u;

        local_acc = local_acc + input[col] * (f32(is_plus) - f32(is_minus));

        col = col + 256u;
    }

    // Store partial sum to shared memory
    partial_sums[tid] = local_acc;
    workgroupBarrier();

    // Parallel reduction (log2(256) = 8 steps)
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            partial_sums[tid] = partial_sums[tid] + partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes final result
    if (tid == 0u) {
        output[row] = partial_sums[0] * params.scale;
    }
}
"#;

/// Batched Ternary MatMul: One thread per (batch, row) pair
///
/// Computes: output[b][row] = sum_col(input[b][col] * weight[row][col]) * scale
///
/// Dispatch: (ceil(out_features/256), batch_size, 1)
pub const TERNARY_MATMUL_BATCH_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    words_per_row: u32,
    scale: f32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> plus_bits: array<u32>;
@group(0) @binding(2) var<storage, read> minus_bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let batch = gid.y;

    if (row >= params.out_features || batch >= params.batch_size) {
        return;
    }

    var acc: f32 = 0.0;
    let row_offset = row * params.words_per_row;
    let input_offset = batch * params.in_features;

    for (var w: u32 = 0u; w < params.words_per_row; w = w + 1u) {
        let pw = plus_bits[row_offset + w];
        let mw = minus_bits[row_offset + w];
        let base = w * 32u;

        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
            let col = base + b;
            if (col >= params.in_features) {
                break;
            }

            let x = input[input_offset + col];
            let is_plus = (pw >> b) & 1u;
            let is_minus = (mw >> b) & 1u;

            acc = acc + x * (f32(is_plus) - f32(is_minus));
        }
    }

    output[batch * params.out_features + row] = acc * params.scale;
}
"#;

/// ReLU activation (in-place on storage buffer)
pub const RELU_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    data[idx] = max(data[idx], 0.0);
}
"#;

/// ReLU params (matches WGSL struct)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReluParams {
    pub len: u32,
    pub _pad: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_params_size() {
        // Must be 32 bytes (multiple of 16 for uniform alignment)
        assert_eq!(core::mem::size_of::<GpuParams>(), 32);
    }

    #[test]
    fn test_relu_params_size() {
        assert_eq!(core::mem::size_of::<ReluParams>(), 16);
    }

    #[test]
    fn test_gpu_params_zeroed() {
        use bytemuck::Zeroable;
        let params = GpuParams::zeroed();
        assert_eq!(params.in_features, 0);
        assert_eq!(params.out_features, 0);
        assert_eq!(params.scale, 0.0);
    }
}
