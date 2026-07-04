//! Fix128 GPU primitive skeleton (physics-solver support, opt-in).
//!
//! Provides [`Fix128Gpu`] (`int64x2` hi/lo representation) and the
//! [`Fix128GpuKernel`] trait for compute-shader-level Fix128
//! `add / sub / mul / dot` operations. The WGSL shader body and full
//! arithmetic implementation are scheduled for follow-up commits; this
//! skeleton pins the public API so downstream `physics_bridge` code
//! and external `alice-physics` `GpuSolverBridge` integrations can
//! compile against a stable surface.
//!
//! # Determinism contract
//!
//! Implementations **must** guarantee bit-exact results across:
//! - GPU vendor (NVIDIA / AMD / Intel Arc / Apple Silicon)
//! - Backend (Metal / Vulkan / DX12)
//! - Workgroup / dispatch sizes
//!
//! This requires strict `workgroupBarrier` synchronisation, index-
//! ordered subgroup reductions, and forbidding any implicit `atomicAdd`
//! path in the WGSL shader. See the private
//! [`deterministic-physics-lockstep-discipline`](https://github.com/ext-sakamoro/claude-config/blob/main/claude-skills/deterministic-physics-lockstep-discipline/SKILL.md)
//! skill §1 経路 3 (SIMD / subgroup reduce ordering) and §1 経路 5
//! (thread traversal ordering) for the reference reasoning.
//!
//! # Feature
//!
//! Enable with `--features fix128-arithmetic`. The
//! [`Fix128GpuKernel`] trait itself does not depend on any GPU
//! backend so consumers can stub it out for testing.

/// GPU-friendly Fix128 representation.
///
/// `hi` holds the sign-extended integer part (`i64`) and `lo` the
/// 64-bit unsigned fractional low half — this matches the layout of
/// [`alice_physics::math::Fix128`] when compiled with `--features
/// std`. Kept `#[repr(C)]` so the type can be uploaded as a plain
/// storage buffer without further conversion.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Fix128Gpu {
    /// Sign-extended integer part (`I64F64` high word).
    pub hi: i64,
    /// Unsigned fractional low half (`I64F64` low word).
    pub lo: u64,
}

impl Fix128Gpu {
    /// Additive identity.
    pub const ZERO: Self = Self { hi: 0, lo: 0 };
    /// Multiplicative identity (one whole unit, zero fractional).
    pub const ONE: Self = Self { hi: 1, lo: 0 };

    /// Construct a `Fix128Gpu` from raw `(hi, lo)` halves.
    ///
    /// Callers are responsible for ensuring the pair encodes a valid
    /// `I64F64` value.
    #[must_use]
    pub const fn from_raw(hi: i64, lo: u64) -> Self {
        Self { hi, lo }
    }

    /// Reference `add` — Fix128 128-bit two's-complement addition with
    /// carry propagation from `lo` into `hi`. This is the golden
    /// implementation the WGSL kernel must reproduce byte-for-byte
    /// (both operate on the same `#[repr(C)] { hi: i64, lo: u64 }`
    /// storage layout).
    #[must_use]
    pub const fn add(self, other: Self) -> Self {
        let (lo, carry) = self.lo.overflowing_add(other.lo);
        // Wrap-around signed hi add (two's complement) plus carry from lo.
        let hi = self.hi.wrapping_add(other.hi).wrapping_add(carry as i64);
        Self { hi, lo }
    }

    /// Reference `sub` — mirrors `add` for the subtractive case; the
    /// WGSL kernel must reproduce this byte-for-byte.
    #[must_use]
    pub const fn sub(self, other: Self) -> Self {
        let (lo, borrow) = self.lo.overflowing_sub(other.lo);
        let hi = self.hi.wrapping_sub(other.hi).wrapping_sub(borrow as i64);
        Self { hi, lo }
    }
}

// ---------------------------------------------------------------------------
// WGSL shader source (Fix128 add) — golden reference for the follow-up
// wgpu dispatch integration. The shader operates on the same
// `#[repr(C)] Fix128Gpu { hi: i64, lo: u64 }` layout that the Rust
// side ships, treated as four little-endian `u32` fields per element.
// ---------------------------------------------------------------------------

/// WGSL compute shader source for `Fix128 add`. Bind group 0 exposes
/// three storage buffers `(a, b, out)` of `Fix128Gpu` arrays. The
/// entry point is `@compute @workgroup_size(64) fn fix128_add_main`.
///
/// This constant is the golden reference that a wgpu dispatch backend
/// will consume in the follow-up; the CPU reference is
/// [`Fix128Gpu::add`], and the two implementations must produce
/// byte-identical results on every platform.
pub const FIX128_ADD_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

@group(0) @binding(0) var<storage, read>       input_a: array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read>       input_b: array<Fix128Gpu>;
@group(0) @binding(2) var<storage, read_write> output:  array<Fix128Gpu>;

// 64-bit unsigned add, returning (sum_lo, sum_hi, carry_out).
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

fn fix128_add(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    // Lo half (unsigned 64-bit add).
    let lo_res        = u64_add(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let carry_from_lo = lo_res.z;

    // Hi half (unsigned 64-bit add, interpreted as two's complement signed).
    let hi_res        = u64_add(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    // Fold the carry from the lo half into the hi total.
    let hi_with_carry = u64_add(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(carry_from_lo, 0u));

    var out: Fix128Gpu;
    out.hi_lo = hi_with_carry.x;
    out.hi_hi = hi_with_carry.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

@compute @workgroup_size(64)
fn fix128_add_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_a)) { return; }
    output[i] = fix128_add(input_a[i], input_b[i]);
}
"#;

/// WGSL compute shader source for `Fix128 sub` — mirrors
/// [`FIX128_ADD_WGSL`] with borrow propagation replacing carry
/// propagation. Same bind group layout (3 storage buffers,
/// `Fix128Gpu` arrays); entry point is
/// `@compute @workgroup_size(64) fn fix128_sub_main`.
///
/// The CPU reference is [`Fix128Gpu::sub`]; the two implementations
/// must produce byte-identical results on every platform.
pub const FIX128_SUB_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

@group(0) @binding(0) var<storage, read>       input_a: array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read>       input_b: array<Fix128Gpu>;
@group(0) @binding(2) var<storage, read_write> output:  array<Fix128Gpu>;

// 64-bit unsigned sub, returning (diff_lo, diff_hi, borrow_out).
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let diff_lo = a.x - b.x;
    let borrow1 = select(0u, 1u, a.x < b.x);
    let mid     = a.y - b.y;
    let borrow2 = select(0u, 1u, a.y < b.y);
    let diff_hi = mid - borrow1;
    let borrow3 = select(0u, 1u, mid < borrow1);
    return vec3<u32>(diff_lo, diff_hi, borrow2 + borrow3);
}

fn fix128_sub(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    // Lo half (unsigned 64-bit sub).
    let lo_res         = u64_sub(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let borrow_from_lo = lo_res.z;

    // Hi half (unsigned 64-bit sub, interpreted as two's complement signed).
    let hi_res         = u64_sub(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    // Fold the borrow from the lo half into the hi total.
    let hi_with_borrow = u64_sub(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(borrow_from_lo, 0u));

    var out: Fix128Gpu;
    out.hi_lo = hi_with_borrow.x;
    out.hi_hi = hi_with_borrow.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

@compute @workgroup_size(64)
fn fix128_sub_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_a)) { return; }
    output[i] = fix128_sub(input_a[i], input_b[i]);
}
"#;

// ---------------------------------------------------------------------------
// wgpu dispatch backend (Fix128 add / sub) — pairs with the WGSL shaders above.
// ---------------------------------------------------------------------------

/// wgpu-based dispatch backend for the Fix128 GPU kernels.
///
/// Owns the `ComputePipeline` + `BindGroupLayout` compiled from
/// [`FIX128_ADD_WGSL`]. Constructed once per `GpuDevice`; each
/// `add` call uploads the two input slices, dispatches the compute
/// pipeline, and reads the results back into the caller-owned
/// output slice.
///
/// # Determinism
/// - Workgroup size is fixed at 64, dispatched in ascending
///   `global_invocation_id` order (skill §1 経路 5).
/// - No `atomicAdd` / subgroup reductions are used inside the shader
///   (skill §1 経路 3); each output index is computed by a single
///   thread from a single pair of inputs.
/// - Rust-side `bytemuck::cast_slice` preserves the little-endian
///   `#[repr(C)] { hi: i64, lo: u64 }` layout, so the shader sees the
///   same bit pattern as the CPU reference.
pub struct Fix128WgpuKernel<'a> {
    device: &'a crate::device::GpuDevice,
    pipeline_add: wgpu::ComputePipeline,
    pipeline_sub: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl<'a> Fix128WgpuKernel<'a> {
    /// Compile the WGSL shader and build the compute pipeline against
    /// the supplied `GpuDevice`. This is cheap enough to be done at
    /// scene load; the resulting kernel can be reused for every
    /// `add` dispatch.
    #[must_use]
    pub fn new(device: &'a crate::device::GpuDevice) -> Self {
        let shader_add = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_add_shader"),
                source: wgpu::ShaderSource::Wgsl(FIX128_ADD_WGSL.into()),
            });
        let shader_sub = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_sub_shader"),
                source: wgpu::ShaderSource::Wgsl(FIX128_SUB_WGSL.into()),
            });

        let bind_group_layout =
            device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fix128_add_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("fix128_add_pl"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline_add =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fix128_add_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_add,
                    entry_point: Some("fix128_add_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let pipeline_sub =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fix128_sub_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_sub,
                    entry_point: Some("fix128_sub_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        Self {
            device,
            pipeline_add,
            pipeline_sub,
            bind_group_layout,
        }
    }

    /// Internal helper — dispatches the given pipeline against the
    /// two input slices and reads the result back into `out`.
    fn dispatch_binary(
        &self,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
        a: &[Fix128Gpu],
        b: &[Fix128Gpu],
        out: &mut [Fix128Gpu],
    ) {
        assert_eq!(
            a.len(),
            b.len(),
            "Fix128 dispatch: input slices must match length"
        );
        assert_eq!(
            a.len(),
            out.len(),
            "Fix128 dispatch: output slice length mismatch"
        );
        if a.is_empty() {
            return;
        }

        let byte_size = (a.len() * std::mem::size_of::<Fix128Gpu>()) as u64;
        let buf_a = self
            .device
            .create_buffer_init(&format!("{label}_a"), bytemuck::cast_slice(a));
        let buf_b = self
            .device
            .create_buffer_init(&format!("{label}_b"), bytemuck::cast_slice(b));
        let buf_out = self
            .device
            .create_buffer_empty(&format!("{label}_out"), byte_size);

        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{label}_bg")),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_out.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("{label}_enc")),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label}_pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((a.len() as u32) + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.device.submit(encoder);
        self.device.poll_wait();

        let raw = self.device.read_buffer(&buf_out, byte_size);
        let result: &[Fix128Gpu] = bytemuck::cast_slice(&raw);
        out.copy_from_slice(result);
    }

    /// Dispatch the Fix128 `add` kernel. `a`, `b`, and `out` must
    /// have identical lengths; the caller owns the output slice.
    pub fn add(&self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        self.dispatch_binary(&self.pipeline_add, "fix128_add", a, b, out);
    }

    /// Dispatch the Fix128 `sub` kernel. `a`, `b`, and `out` must
    /// have identical lengths.
    pub fn sub(&self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        self.dispatch_binary(&self.pipeline_sub, "fix128_sub", a, b, out);
    }
}

// ---------------------------------------------------------------------------
// Fix128GpuKernel trait bridge (add/sub live via wgpu; mul/dot pending)
// ---------------------------------------------------------------------------

impl Fix128GpuKernel for Fix128WgpuKernel<'_> {
    fn add(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        Fix128WgpuKernel::add(self, a, b, out);
    }

    fn sub(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        Fix128WgpuKernel::sub(self, a, b, out);
    }

    fn mul(&mut self, _a: &[Fix128Gpu], _b: &[Fix128Gpu], _out: &mut [Fix128Gpu]) {
        unimplemented!(
            "Fix128 mul: WGSL kernel + Fix128WgpuKernel::mul scheduled for the next release"
        );
    }

    fn dot(&mut self, _a: &[Fix128Gpu], _b: &[Fix128Gpu], _out: &mut Fix128Gpu) {
        unimplemented!(
            "Fix128 dot: WGSL kernel + Fix128WgpuKernel::dot scheduled for the next release"
        );
    }
}

/// GPU kernel dispatch trait for Fix128 element-wise arithmetic.
///
/// Backends implementing this trait own a compute pipeline that
/// materialises the four operations below. Element ordering of both
/// inputs and the output slice must be preserved (index `i` in the
/// output corresponds to inputs at index `i`) to satisfy the skill
/// §1 経路 5 traversal-order contract.
///
/// # Slice length contract
///
/// - `add / sub / mul`: `a.len() == b.len() == out.len()`
/// - `dot`: `a.len() == b.len()`; the single-element accumulator is
///   summed in ascending index order.
///
/// # Skeleton
///
/// The trait is intentionally minimal for the initial rollout. The
/// bodies are stubbed with `todo!` in the skeleton so implementers can
/// wire the dispatch layer against a stable signature; the WGSL
/// arithmetic will land in a follow-up commit.
pub trait Fix128GpuKernel {
    /// Element-wise addition: `out[i] = a[i] + b[i]`.
    fn add(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]);

    /// Element-wise subtraction: `out[i] = a[i] - b[i]`.
    fn sub(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]);

    /// Element-wise multiplication (Fix128 semantics): `out[i] = a[i] * b[i]`.
    fn mul(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]);

    /// Fix128 dot product: `out = Σ a[i] * b[i]` accumulated in
    /// ascending index order.
    fn dot(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut Fix128Gpu);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix128_gpu_constants_are_wellformed() {
        assert_eq!(Fix128Gpu::ZERO.hi, 0);
        assert_eq!(Fix128Gpu::ZERO.lo, 0);
        assert_eq!(Fix128Gpu::ONE.hi, 1);
        assert_eq!(Fix128Gpu::ONE.lo, 0);
    }

    #[test]
    fn fix128_gpu_from_raw_round_trips() {
        let v = Fix128Gpu::from_raw(42, 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(v.hi, 42);
        assert_eq!(v.lo, 0xDEAD_BEEF_CAFE_BABE);
    }

    /// `add` with no carry into `hi` — the golden reference must
    /// leave `hi` untouched.
    #[test]
    fn fix128_gpu_add_no_carry() {
        let a = Fix128Gpu::from_raw(1, 0x0000_0000_0000_0001);
        let b = Fix128Gpu::from_raw(2, 0x0000_0000_0000_0002);
        let r = a.add(b);
        assert_eq!(r.hi, 3);
        assert_eq!(r.lo, 0x0000_0000_0000_0003);
    }

    /// `add` with `lo` overflow — carry must propagate into `hi`.
    #[test]
    fn fix128_gpu_add_carry_into_hi() {
        let a = Fix128Gpu::from_raw(1, u64::MAX);
        let b = Fix128Gpu::from_raw(0, 1);
        let r = a.add(b);
        assert_eq!(r.hi, 2);
        assert_eq!(r.lo, 0);
    }

    /// `add` on the negative side (two's complement) — verifies the
    /// wrap-around semantics that the WGSL kernel must reproduce.
    #[test]
    fn fix128_gpu_add_twos_complement_negative() {
        // (-1).add(1) == 0 in two's complement, regardless of the
        // fractional half.
        let a = Fix128Gpu::from_raw(-1, u64::MAX);
        let b = Fix128Gpu::from_raw(0, 1);
        let r = a.add(b);
        assert_eq!(r.hi, 0);
        assert_eq!(r.lo, 0);
    }

    /// `sub` with no borrow — the golden reference must leave `hi`
    /// untouched.
    #[test]
    fn fix128_gpu_sub_no_borrow() {
        let a = Fix128Gpu::from_raw(5, 10);
        let b = Fix128Gpu::from_raw(2, 3);
        let r = a.sub(b);
        assert_eq!(r.hi, 3);
        assert_eq!(r.lo, 7);
    }

    /// `sub` with `lo` underflow — borrow must propagate out of `hi`.
    #[test]
    fn fix128_gpu_sub_borrow_from_hi() {
        let a = Fix128Gpu::from_raw(3, 0);
        let b = Fix128Gpu::from_raw(0, 1);
        let r = a.sub(b);
        assert_eq!(r.hi, 2);
        assert_eq!(r.lo, u64::MAX);
    }

    /// The WGSL shader source is a non-empty compile-time constant.
    /// Follow-up wgpu integration will feed it through
    /// `Device::create_shader_module`.
    #[test]
    fn wgsl_add_shader_is_present() {
        assert!(FIX128_ADD_WGSL.contains("fix128_add_main"));
        assert!(FIX128_ADD_WGSL.contains("@compute"));
        assert!(FIX128_ADD_WGSL.contains("workgroup_size(64)"));
    }

    /// The Fix128 sub shader source is a non-empty compile-time
    /// constant with the required entry point.
    #[test]
    fn wgsl_sub_shader_is_present() {
        assert!(FIX128_SUB_WGSL.contains("fix128_sub_main"));
        assert!(FIX128_SUB_WGSL.contains("@compute"));
        assert!(FIX128_SUB_WGSL.contains("u64_sub"));
    }

    /// GPU dispatch bit-exact contract: for every input pair, the
    /// wgpu backend must produce the exact same `hi` / `lo` pair as
    /// the CPU reference [`Fix128Gpu::add`]. Fixtures cover the three
    /// carry-propagation regimes exercised by the CPU tests above:
    /// no carry, carry into hi, and two's complement negative.
    ///
    /// Skips when no GPU adapter is available (headless CI).
    #[test]
    fn wgpu_add_matches_cpu_golden() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // No GPU adapter — headless CI, skip.
        };
        let kernel = Fix128WgpuKernel::new(&device);

        let a = vec![
            Fix128Gpu::from_raw(1, 0x0000_0000_0000_0001),
            Fix128Gpu::from_raw(1, u64::MAX),
            Fix128Gpu::from_raw(-1, u64::MAX),
            Fix128Gpu::from_raw(0, 0xDEAD_BEEF_CAFE_BABE),
        ];
        let b = vec![
            Fix128Gpu::from_raw(2, 0x0000_0000_0000_0002),
            Fix128Gpu::from_raw(0, 1),
            Fix128Gpu::from_raw(0, 1),
            Fix128Gpu::from_raw(0, 0x1111_2222_3333_4444),
        ];
        let mut out = vec![Fix128Gpu::ZERO; 4];

        kernel.add(&a, &b, &mut out);

        for i in 0..a.len() {
            let cpu_ref = a[i].add(b[i]);
            assert_eq!(
                out[i].hi, cpu_ref.hi,
                "hi mismatch at i={i}: GPU {} vs CPU {}",
                out[i].hi, cpu_ref.hi
            );
            assert_eq!(
                out[i].lo, cpu_ref.lo,
                "lo mismatch at i={i}: GPU {:#x} vs CPU {:#x}",
                out[i].lo, cpu_ref.lo
            );
        }
    }

    /// GPU dispatch bit-exact contract for `sub` — mirrors the add
    /// golden test but exercises borrow propagation instead of
    /// carry propagation.
    #[test]
    fn wgpu_sub_matches_cpu_golden() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let kernel = Fix128WgpuKernel::new(&device);

        let a = vec![
            Fix128Gpu::from_raw(5, 10),
            Fix128Gpu::from_raw(3, 0),
            Fix128Gpu::from_raw(0, 0),
            Fix128Gpu::from_raw(-1, u64::MAX),
        ];
        let b = vec![
            Fix128Gpu::from_raw(2, 3),
            Fix128Gpu::from_raw(0, 1),
            Fix128Gpu::from_raw(0, 1),
            Fix128Gpu::from_raw(-2, 0),
        ];
        let mut out = vec![Fix128Gpu::ZERO; 4];

        kernel.sub(&a, &b, &mut out);

        for i in 0..a.len() {
            let cpu_ref = a[i].sub(b[i]);
            assert_eq!(
                out[i].hi, cpu_ref.hi,
                "sub hi mismatch at i={i}: GPU {} vs CPU {}",
                out[i].hi, cpu_ref.hi
            );
            assert_eq!(
                out[i].lo, cpu_ref.lo,
                "sub lo mismatch at i={i}: GPU {:#x} vs CPU {:#x}",
                out[i].lo, cpu_ref.lo
            );
        }
    }

    /// The `Fix128GpuKernel` trait bridge routes `add` / `sub` to the
    /// live wgpu pipelines while `mul` / `dot` remain
    /// `unimplemented!`. Verifies the trait method matches
    /// [`Fix128WgpuKernel::add`] byte-for-byte.
    #[test]
    fn wgpu_trait_add_matches_inherent_add() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut kernel = Fix128WgpuKernel::new(&device);

        let a = vec![
            Fix128Gpu::from_raw(1, u64::MAX),
            Fix128Gpu::from_raw(-3, 42),
        ];
        let b = vec![Fix128Gpu::from_raw(0, 1), Fix128Gpu::from_raw(5, 100)];
        let mut via_inherent = vec![Fix128Gpu::ZERO; 2];
        let mut via_trait = vec![Fix128Gpu::ZERO; 2];

        // Inherent method
        Fix128WgpuKernel::add(&kernel, &a, &b, &mut via_inherent);
        // Trait method
        <Fix128WgpuKernel<'_> as Fix128GpuKernel>::add(&mut kernel, &a, &b, &mut via_trait);

        assert_eq!(via_inherent, via_trait);
    }
}
