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
}
