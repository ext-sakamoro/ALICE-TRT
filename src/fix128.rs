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

    /// Reference `mul` — Fix128 (`I64F64`) multiplication that takes
    /// the middle 128 bits of the 256-bit signed product. Mirrors
    /// [`alice_physics::math::Fix128::mul`] so the WGSL kernel
    /// (scheduled for the follow-up commit) can be certified
    /// byte-for-byte against this reference.
    ///
    /// # Semantics
    /// - `self = self.hi + self.lo / 2^64`
    /// - `rhs  = rhs.hi + rhs.lo / 2^64`
    /// - Product: `hh << 128 + (hl + lh) << 64 + ll`
    /// - Return the middle 128 bits (`bits[192:64]`).
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        let a_hi = self.hi as i128;
        let a_lo = self.lo as u128;
        let b_hi = other.hi as i128;
        let b_lo = other.lo as u128;

        let ll = a_lo.wrapping_mul(b_lo);
        let hl = a_hi.wrapping_mul(b_lo as i128);
        let lh = (a_lo as i128).wrapping_mul(b_hi);
        let hh = a_hi.wrapping_mul(b_hi);

        let ll_hi = (ll >> 64) as i128;
        let mid = hl.wrapping_add(lh).wrapping_add(ll_hi);
        let mid_lo = mid as u64;
        let mid_hi = (mid >> 64) as i64;

        let hi = (hh as i64).wrapping_add(mid_hi);
        Self { hi, lo: mid_lo }
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

/// WGSL compute shader source for `Fix128 mul` (skeleton with the
/// `umul_wide` and `u64_mul_wide` schoolbook helpers ready to be
/// re-used by the full signed 128×128 pipeline in a follow-up).
///
/// The helpers exposed here — `umul_wide` (u32×u32→u64) and
/// `u64_mul_wide` (u64×u64→u128) — are byte-exact re-productions of
/// the operations that the CPU reference [`Fix128Gpu::mul`] performs
/// internally on `i128`. Adding them separately keeps the follow-up
/// commit focused on the sign-correction and truncation logic (the
/// "middle 128 bits of the 256-bit signed product") rather than on
/// wide integer arithmetic.
///
/// The full end-to-end `fix128_mul_main` entry point is scheduled for
/// v0.5.0; this constant currently ships the helpers plus a trivial
/// `fix128_mul_unsigned_lo_main` entry point that returns the low
/// 128 bits of the *unsigned* 128×128→256 product. That entry point
/// is enough to validate the schoolbook helpers on a real GPU
/// against a hand-computed golden.
pub const FIX128_MUL_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

@group(0) @binding(0) var<storage, read>       input_a: array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read>       input_b: array<Fix128Gpu>;
@group(0) @binding(2) var<storage, read_write> output:  array<Fix128Gpu>;

// u32 × u32 → u64 via 16-bit halving schoolbook.
// Returns (lo, hi) so that (lo, hi) = a * b exactly as an unsigned 64-bit number.
fn umul_wide(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu;
    let ah = a >> 16u;
    let bl = b & 0xFFFFu;
    let bh = b >> 16u;

    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;

    let mid       = lh + hl;
    let mid_carry = select(0u, 1u, mid < lh);

    let lo_out    = ll + (mid << 16u);
    let carry_lo  = select(0u, 1u, lo_out < ll);

    let hi_out    = hh + (mid >> 16u) + (mid_carry << 16u) + carry_lo;
    return vec2<u32>(lo_out, hi_out);
}

// 64-bit unsigned add returning (sum_lo, sum_hi, carry_out).
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

// u64 × u64 → u128 (schoolbook of four u32×u32 partial products).
// Returns the 128-bit product laid out as vec4<u32>(r0, r1, r2, r3)
// where r0 is the least-significant word. Used by the follow-up
// signed Fix128 mul pipeline.
fn u64_mul_wide(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let ll = umul_wide(a.x, b.x);
    let lh = umul_wide(a.x, b.y);
    let hl = umul_wide(a.y, b.x);
    let hh = umul_wide(a.y, b.y);

    // Position 0: ll.x
    let r0 = ll.x;

    // Position 1: ll.y + lh.x + hl.x (with carry into position 2)
    let s1a = ll.y + lh.x;
    let c1a = select(0u, 1u, s1a < ll.y);
    let r1  = s1a + hl.x;
    let c1b = select(0u, 1u, r1 < s1a);
    let carry_to_2 = c1a + c1b;

    // Position 2: lh.y + hl.y + hh.x + carry_to_2 (with carry into position 3)
    let s2a = lh.y + hl.y;
    let c2a = select(0u, 1u, s2a < lh.y);
    let s2b = s2a + hh.x;
    let c2b = select(0u, 1u, s2b < s2a);
    let r2  = s2b + carry_to_2;
    let c2c = select(0u, 1u, r2 < s2b);
    let carry_to_3 = c2a + c2b + c2c;

    // Position 3: hh.y + carry_to_3 (no further overflow within u128)
    let r3 = hh.y + carry_to_3;

    return vec4<u32>(r0, r1, r2, r3);
}

// Skeleton entry point: emits the *unsigned* low-128 bits of the
// 256-bit product `a.lo × b.lo`. Kept for helper validation harness.
@compute @workgroup_size(64)
fn fix128_mul_unsigned_lo_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_a)) { return; }
    let a = input_a[i];
    let b = input_b[i];
    let product = u64_mul_wide(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));

    var out: Fix128Gpu;
    out.lo_lo = product.x;
    out.lo_hi = product.y;
    out.hi_lo = product.z;
    out.hi_hi = product.w;
    output[i] = out;
}

// Full signed Fix128 multiplication.
//
// Algorithm:
//   1. Treat a, b as unsigned 128-bit numbers a_u, b_u.
//   2. Compute the 256-bit unsigned schoolbook product P = a_u × b_u
//      via four u64×u64 partial products (`ll`, `lh`, `hl`, `hh`).
//   3. Two's-complement correction:
//        signed(a) × signed(b) = a_u × b_u
//                              - (b_u << 128 if a < 0)
//                              - (a_u << 128 if b < 0)
//                              + (2^256 if both < 0)
//      The 2^256 term overflows away; the other two subtractions
//      touch only bits [128, 256), which is exactly where our
//      middle-hi output word lives.
//   4. The Fix128 result is the middle 128 bits (bits [192:64]):
//        lo (u64) = P[2..4]  (unchanged by signed correction)
//        hi (i64) = P[4..6]  (with the two subtractions above)
@compute @workgroup_size(64)
fn fix128_mul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_a)) { return; }
    let a = input_a[i];
    let b = input_b[i];

    // Unsigned 128-bit views:
    //   a_u = a_hi * 2^64 + a_lo   (a_hi, a_lo each u64 = vec2<u32>)
    let a_lo = vec2<u32>(a.lo_lo, a.lo_hi);
    let a_hi = vec2<u32>(a.hi_lo, a.hi_hi);
    let b_lo = vec2<u32>(b.lo_lo, b.lo_hi);
    let b_hi = vec2<u32>(b.hi_lo, b.hi_hi);

    // Four u64 × u64 → u128 partial products, each vec4<u32> laid
    // out little-endian (word 0 = least significant).
    let ll = u64_mul_wide(a_lo, b_lo);  // contributes to positions 0..4
    let lh = u64_mul_wide(a_lo, b_hi);  // contributes to positions 2..6
    let hl = u64_mul_wide(a_hi, b_lo);  // contributes to positions 2..6
    let hh = u64_mul_wide(a_hi, b_hi);  // contributes to positions 4..8

    // Positions 0..1: only ll contributes; no addition, no carry.
    // We start tracking carries from position 2 (the first
    // position where multiple limbs collide).

    // Position 2: ll.z + lh.x + hl.x
    let s2a = ll.z + lh.x;
    let c2a = select(0u, 1u, s2a < ll.z);
    let s2b = s2a + hl.x;
    let c2b = select(0u, 1u, s2b < s2a);
    let p2  = s2b;
    let carry_to_3 = c2a + c2b;

    // Position 3: ll.w + lh.y + hl.y + carry_to_3
    let s3a = ll.w + lh.y;
    let c3a = select(0u, 1u, s3a < ll.w);
    let s3b = s3a + hl.y;
    let c3b = select(0u, 1u, s3b < s3a);
    let p3  = s3b + carry_to_3;
    let c3c = select(0u, 1u, p3 < s3b);
    let carry_to_4 = c3a + c3b + c3c;

    // Position 4: lh.z + hl.z + hh.x + carry_to_4
    let s4a = lh.z + hl.z;
    let c4a = select(0u, 1u, s4a < lh.z);
    let s4b = s4a + hh.x;
    let c4b = select(0u, 1u, s4b < s4a);
    let p4_u = s4b + carry_to_4;
    let c4c = select(0u, 1u, p4_u < s4b);
    let carry_to_5 = c4a + c4b + c4c;

    // Position 5: lh.w + hl.w + hh.y + carry_to_5
    let s5a = lh.w + hl.w;
    let c5a = select(0u, 1u, s5a < lh.w);
    let s5b = s5a + hh.y;
    let c5b = select(0u, 1u, s5b < s5a);
    let p5_u = s5b + carry_to_5;
    // Carries past position 5 do not affect the middle 128 bits.

    // Signed correction:
    //   If a is negative (its i64 hi has MSB set), subtract b_u
    //   from position 4..8. For our middle window (positions 4..6),
    //   that means subtracting b_lo (u64) from (p4, p5).
    let a_negative = (a.hi_hi & 0x80000000u) != 0u;
    let b_negative = (b.hi_hi & 0x80000000u) != 0u;

    var p4 = p4_u;
    var p5 = p5_u;

    if (a_negative) {
        let sub_lo = b.lo_lo;
        let sub_hi = b.lo_hi;
        let new_p4 = p4 - sub_lo;
        let borrow = select(0u, 1u, p4 < sub_lo);
        let new_p5 = p5 - sub_hi - borrow;
        p4 = new_p4;
        p5 = new_p5;
    }

    if (b_negative) {
        let sub_lo = a.lo_lo;
        let sub_hi = a.lo_hi;
        let new_p4 = p4 - sub_lo;
        let borrow = select(0u, 1u, p4 < sub_lo);
        let new_p5 = p5 - sub_hi - borrow;
        p4 = new_p4;
        p5 = new_p5;
    }

    // Fix128 output:
    //   lo = middle_lo (u64) = (p2, p3)
    //   hi = middle_hi (i64) = (p4, p5)  (two's-complement bits)
    var out: Fix128Gpu;
    out.lo_lo = p2;
    out.lo_hi = p3;
    out.hi_lo = p4;
    out.hi_hi = p5;
    output[i] = out;
}
"#;

/// WGSL compute shader source for `Fix128 dot` — Σ a[i] × b[i] as a
/// single Fix128 output. The kernel dispatches a single workgroup of a
/// single thread that walks the input arrays in index order, mirroring
/// the CPU golden `acc = acc.add(a[i].mul(b[i]))` reduction loop.
///
/// # Determinism contract
///
/// Fix128 addition is **not** associative under two's-complement
/// wraparound; the sum depends on iteration order. To preserve
/// bit-exact equivalence across GPU vendors we must enforce the
/// canonical index order, which forbids:
///
/// - `subgroup{Add,Mul,Min,Max}` — order is vendor-specific.
/// - `atomicAdd` on a shared accumulator — race-dependent order.
/// - Any parallel tree-reduction that reorders the additions.
///
/// The v0.7.1 implementation is a **multi-workgroup blocked reduction**
/// for arbitrary N with a paired serial final-accumulate shader:
///
/// - Phase 1 (this shader, `fix128_dot_partial_main`): K workgroups
///   of 64 threads each cover the input in `ELEMS_PER_WORKGROUP = 4096`
///   chunks. Workgroup `w` owns `[w·4096, min((w+1)·4096, N))`,
///   split among its 64 threads via the same in-block ordering as v0.7.0.
///   Each workgroup writes one `Fix128Gpu` to `partials_out[w]`.
/// - Phase 2 ([`FIX128_DOT_FINAL_WGSL`]): a single-thread pass folds
///   `partials_out[0..K]` in **workgroup-index order** into `output[0]`.
///
/// K = ⌈N / 4096⌉ workgroups. Since Phase 1 writes go to distinct
/// `partials_out` slots and Phase 2 reads them in fixed index order,
/// workgroup completion order is irrelevant to the result. Total
/// arithmetic order = canonical index 0..N — byte-for-byte equal to
/// the single-thread CPU reference regardless of workgroup scheduling.
///
/// # Layout
///
/// Bind group 0 exposes `(input_a, input_b, output)` where `output`
/// is an `array<Fix128Gpu>` of length 1. The single result is written
/// to `output[0]`.
pub const FIX128_DOT_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

@group(0) @binding(0) var<storage, read>       input_a: array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read>       input_b: array<Fix128Gpu>;
@group(0) @binding(2) var<storage, read_write> output:  array<Fix128Gpu>;

// -- helper: u32 × u32 → u64 --------------------------------------------------
fn umul_wide(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu;
    let ah = a >> 16u;
    let bl = b & 0xFFFFu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let mid       = lh + hl;
    let mid_carry = select(0u, 1u, mid < lh);
    let lo_out    = ll + (mid << 16u);
    let carry_lo  = select(0u, 1u, lo_out < ll);
    let hi_out    = hh + (mid >> 16u) + (mid_carry << 16u) + carry_lo;
    return vec2<u32>(lo_out, hi_out);
}

// -- helper: 64-bit unsigned add returning (lo, hi, carry) --------------------
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

// -- helper: u64 × u64 → u128 -------------------------------------------------
fn u64_mul_wide(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let ll = umul_wide(a.x, b.x);
    let lh = umul_wide(a.x, b.y);
    let hl = umul_wide(a.y, b.x);
    let hh = umul_wide(a.y, b.y);
    let r0 = ll.x;
    let s1a = ll.y + lh.x;
    let c1a = select(0u, 1u, s1a < ll.y);
    let r1  = s1a + hl.x;
    let c1b = select(0u, 1u, r1 < s1a);
    let carry_to_2 = c1a + c1b;
    let s2a = lh.y + hl.y;
    let c2a = select(0u, 1u, s2a < lh.y);
    let s2b = s2a + hh.x;
    let c2b = select(0u, 1u, s2b < s2a);
    let r2  = s2b + carry_to_2;
    let c2c = select(0u, 1u, r2 < s2b);
    let carry_to_3 = c2a + c2b + c2c;
    let r3 = hh.y + carry_to_3;
    return vec4<u32>(r0, r1, r2, r3);
}

// -- fix128 add: 128-bit two's-complement add --------------------------------
fn fix128_add_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let lo_res        = u64_add(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let carry_from_lo = lo_res.z;
    let hi_res        = u64_add(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    let hi_with_carry = u64_add(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(carry_from_lo, 0u));
    var out: Fix128Gpu;
    out.hi_lo = hi_with_carry.x;
    out.hi_hi = hi_with_carry.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

// -- fix128 mul: signed 128×128 → middle 128 bits ----------------------------
fn fix128_mul_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let a_lo = vec2<u32>(a.lo_lo, a.lo_hi);
    let a_hi = vec2<u32>(a.hi_lo, a.hi_hi);
    let b_lo = vec2<u32>(b.lo_lo, b.lo_hi);
    let b_hi = vec2<u32>(b.hi_lo, b.hi_hi);

    let ll = u64_mul_wide(a_lo, b_lo);
    let lh = u64_mul_wide(a_lo, b_hi);
    let hl = u64_mul_wide(a_hi, b_lo);
    let hh = u64_mul_wide(a_hi, b_hi);

    // Position 2..5 with full carry propagation (see fix128_mul_main
    // in FIX128_MUL_WGSL for the full derivation).
    let s2a = ll.z + lh.x;
    let c2a = select(0u, 1u, s2a < ll.z);
    let s2b = s2a + hl.x;
    let c2b = select(0u, 1u, s2b < s2a);
    let p2  = s2b;
    let carry_to_3 = c2a + c2b;

    let s3a = ll.w + lh.y;
    let c3a = select(0u, 1u, s3a < ll.w);
    let s3b = s3a + hl.y;
    let c3b = select(0u, 1u, s3b < s3a);
    let p3  = s3b + carry_to_3;
    let c3c = select(0u, 1u, p3 < s3b);
    let carry_to_4 = c3a + c3b + c3c;

    let s4a = lh.z + hl.z;
    let c4a = select(0u, 1u, s4a < lh.z);
    let s4b = s4a + hh.x;
    let c4b = select(0u, 1u, s4b < s4a);
    let p4_u = s4b + carry_to_4;
    let c4c = select(0u, 1u, p4_u < s4b);
    let carry_to_5 = c4a + c4b + c4c;

    let s5a = lh.w + hl.w;
    let s5b = s5a + hh.y;
    let p5_u = s5b + carry_to_5;

    // Two's-complement correction.
    let a_negative = (a.hi_hi & 0x80000000u) != 0u;
    let b_negative = (b.hi_hi & 0x80000000u) != 0u;
    var p4 = p4_u;
    var p5 = p5_u;
    if (a_negative) {
        let sub_lo = b.lo_lo;
        let sub_hi = b.lo_hi;
        let new_p4 = p4 - sub_lo;
        let borrow = select(0u, 1u, p4 < sub_lo);
        let new_p5 = p5 - sub_hi - borrow;
        p4 = new_p4;
        p5 = new_p5;
    }
    if (b_negative) {
        let sub_lo = a.lo_lo;
        let sub_hi = a.lo_hi;
        let new_p4 = p4 - sub_lo;
        let borrow = select(0u, 1u, p4 < sub_lo);
        let new_p5 = p5 - sub_hi - borrow;
        p4 = new_p4;
        p5 = new_p5;
    }

    var out: Fix128Gpu;
    out.lo_lo = p2;
    out.lo_hi = p3;
    out.hi_lo = p4;
    out.hi_hi = p5;
    return out;
}

// Multi-workgroup blocked reduction. See the module doc for the
// full determinism argument.
const WG_SIZE: u32 = 64u;
const ELEMS_PER_WORKGROUP: u32 = 4096u;

var<workgroup> partials: array<Fix128Gpu, 64>;

@compute @workgroup_size(64)
fn fix128_dot_partial_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Derive local + workgroup indices from a single global_invocation_id
    // builtin rather than declaring @builtin(local_invocation_id) and
    // @builtin(workgroup_id) separately. WG_SIZE is 64 = 2^6, so:
    //   t  = gid.x mod 64  = gid.x & 63u   (local index within workgroup)
    //   wg = gid.x div 64  = gid.x >> 6u   (workgroup index)
    // This avoids the multi-builtin declaration pattern that some
    // WARP DXIL translation paths mishandled — WARP crashed
    // STATUS_ACCESS_VIOLATION on the v0.7.1 variant with two
    // @builtin parameters. Deriving both from a single builtin is
    // semantically identical and passes on Metal / Vulkan.
    let t = gid.x & 63u;
    let wg = gid.x >> 6u;
    let n = arrayLength(&input_a);

    let wg_start = wg * ELEMS_PER_WORKGROUP;

    // DirectX FXC's uniformity analysis on WARP requires that
    // `workgroupBarrier()` be reached by every thread through
    // identical flow control. An early `return` for out-of-tail
    // workgroups upstream of the barrier triggers FXC error
    // X4026 ("thread sync operation must be in non-varying flow
    // control") and — with `@builtin(workgroup_id)` — used to
    // manifest as a runtime `STATUS_ACCESS_VIOLATION` instead of
    // a compile error. We therefore keep the flow uniform: every
    // workgroup executes the same code path, and out-of-tail
    // workgroups simply write ZERO by leaving `partial` at its
    // default value and clamping their `wg_end` to `wg_start`.
    let wg_start_clamped = select(wg_start, n, wg_start > n);
    let wg_end_unclamped = wg_start_clamped + ELEMS_PER_WORKGROUP;
    let wg_end = select(wg_end_unclamped, n, wg_end_unclamped > n);
    let wg_len = wg_end - wg_start_clamped;

    // Block boundaries within this workgroup's slice.
    // `wg_len` is 0 for out-of-tail workgroups, so `block_size`
    // collapses to 0 and every thread's `local_start = local_end`,
    // which cleanly skips the accumulate loop below.
    let block_size = select((wg_len + WG_SIZE - 1u) / WG_SIZE, 0u, wg_len == 0u);
    let local_start = t * block_size;
    let local_end_unclamped = local_start + block_size;
    let local_end = select(local_end_unclamped, wg_len, local_end_unclamped > wg_len);

    var partial: Fix128Gpu;
    partial.hi_lo = 0u;
    partial.hi_hi = 0u;
    partial.lo_lo = 0u;
    partial.lo_hi = 0u;

    if (local_start < local_end) {
        let start = wg_start_clamped + local_start;
        let end = wg_start_clamped + local_end;
        for (var i: u32 = start; i < end; i = i + 1u) {
            let product = fix128_mul_kernel(input_a[i], input_b[i]);
            partial = fix128_add_kernel(partial, product);
        }
    }

    partials[t] = partial;
    workgroupBarrier();

    // Thread 0 folds the 64 shared partials in block-index order,
    // then writes this workgroup's contribution to partials_out[wg].
    if (t == 0u) {
        var acc: Fix128Gpu;
        acc.hi_lo = 0u;
        acc.hi_hi = 0u;
        acc.lo_lo = 0u;
        acc.lo_hi = 0u;
        for (var b: u32 = 0u; b < WG_SIZE; b = b + 1u) {
            acc = fix128_add_kernel(acc, partials[b]);
        }
        output[wg] = acc;
    }
}
"#;

/// WGSL compute shader source for the Phase 2 serial fold of
/// per-workgroup partials, paired with [`FIX128_DOT_WGSL`] for the
/// multi-workgroup dot pipeline.
///
/// Bind group 0 exposes `(input_a, output)` — a two-buffer layout
/// distinct from the Phase 1 three-buffer layout. `input_a` holds
/// the K partials written by Phase 1; `output[0]` receives the
/// final Fix128 dot product.
///
/// The kernel dispatches a single workgroup of one thread and folds
/// `input_a[0..K]` in canonical index order. Since Phase 1 has
/// already reduced each chunk to a single Fix128 without depending
/// on cross-workgroup ordering, this pass restores the total order
/// with an O(K) serial loop.
pub const FIX128_DOT_FINAL_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

@group(0) @binding(0) var<storage, read>       input_a: array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read_write> output:  array<Fix128Gpu>;

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

fn fix128_add_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let lo_res        = u64_add(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let carry_from_lo = lo_res.z;
    let hi_res        = u64_add(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    let hi_with_carry = u64_add(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(carry_from_lo, 0u));
    var out: Fix128Gpu;
    out.hi_lo = hi_with_carry.x;
    out.hi_hi = hi_with_carry.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

@compute @workgroup_size(1)
fn fix128_dot_final_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let k = arrayLength(&input_a);
    var acc: Fix128Gpu;
    acc.hi_lo = 0u;
    acc.hi_hi = 0u;
    acc.lo_lo = 0u;
    acc.lo_hi = 0u;
    for (var w: u32 = 0u; w < k; w = w + 1u) {
        acc = fix128_add_kernel(acc, input_a[w]);
    }
    output[0] = acc;
}
"#;

/// WGSL compute shader source for the **Fix128 PGS integrate** kernel —
/// the minimum-viable "live dispatch" that upgrades the ALICE-Physics ↔
/// ALICE-TRT `GpuSolverBridge` from wire-up-only to an actually-running
/// GPU physics step.
///
/// # Semantics
///
/// One dispatch performs semi-implicit Euler position integration on
/// every scalar Fix128 axis component in the storage buffer:
///
/// ```text
/// positions[i] = positions[i] + velocities[i] * dt
/// ```
///
/// Bodies are laid out as flat `Fix128Gpu[]` where each body occupies
/// three consecutive slots (x, y, z). The grid dispatches one thread
/// per axis component and updates `positions` in place. Velocities are
/// read-only in this MVV; gravity acceleration and constraint
/// projection land in v0.9.1+.
///
/// # Determinism contract
///
/// Every thread updates a distinct position slot with only a single
/// mul + add of the paired velocity and shared `dt`, so there is no
/// cross-thread ordering dependence. The Fix128 mul + add primitives
/// are already certified bit-exact against the CPU reference
/// ([`Fix128Gpu::mul`], [`Fix128Gpu::add`]) on the platform-matrix CI,
/// so the composite result is also bit-exact — this is the property
/// the paired `TrtSolverAdapter::assert_bit_exact_vs_cpu` test proves.
///
/// # Layout
///
/// - `@group(0) @binding(0) var<storage, read_write> positions: array<Fix128Gpu>`
/// - `@group(0) @binding(1) var<storage, read>       velocities: array<Fix128Gpu>`
/// - `@group(0) @binding(2) var<uniform>              params: PgsParams`
///
/// `PgsParams` packs `dt` (a `Fix128Gpu`, 16 bytes) plus 16 bytes of
/// padding so the uniform buffer stays at the 32-byte minimum many
/// backends prefer for uniform blocks.
pub const FIX128_PGS_INTEGRATE_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

struct PgsParams {
    dt: Fix128Gpu,
    gravity_x: Fix128Gpu,
    gravity_y: Fix128Gpu,
    gravity_z: Fix128Gpu,
}

@group(0) @binding(0) var<storage, read_write> positions:  array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read_write> velocities: array<Fix128Gpu>;
@group(0) @binding(2) var<uniform>             params:     PgsParams;

// -- helper: u32 × u32 → u64 (16-bit halving schoolbook) ----------------------
fn umul_wide(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu;
    let ah = a >> 16u;
    let bl = b & 0xFFFFu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let mid       = lh + hl;
    let mid_carry = select(0u, 1u, mid < lh);
    let lo_out    = ll + (mid << 16u);
    let carry_lo  = select(0u, 1u, lo_out < ll);
    let hi_out    = hh + (mid >> 16u) + (mid_carry << 16u) + carry_lo;
    return vec2<u32>(lo_out, hi_out);
}

// -- helper: 64-bit unsigned add returning (lo, hi, carry) --------------------
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

// -- helper: u64 × u64 → u128 -------------------------------------------------
fn u64_mul_wide(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let ll = umul_wide(a.x, b.x);
    let lh = umul_wide(a.x, b.y);
    let hl = umul_wide(a.y, b.x);
    let hh = umul_wide(a.y, b.y);
    let r0 = ll.x;
    let s1a = ll.y + lh.x;
    let c1a = select(0u, 1u, s1a < ll.y);
    let r1  = s1a + hl.x;
    let c1b = select(0u, 1u, r1 < s1a);
    let carry_to_2 = c1a + c1b;
    let s2a = lh.y + hl.y;
    let c2a = select(0u, 1u, s2a < lh.y);
    let s2b = s2a + hh.x;
    let c2b = select(0u, 1u, s2b < s2a);
    let r2  = s2b + carry_to_2;
    let c2c = select(0u, 1u, r2 < s2b);
    let carry_to_3 = c2a + c2b + c2c;
    let r3 = hh.y + carry_to_3;
    return vec4<u32>(r0, r1, r2, r3);
}

// -- fix128 add: 128-bit two's-complement add ---------------------------------
fn fix128_add_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let lo_res        = u64_add(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let carry_from_lo = lo_res.z;
    let hi_res        = u64_add(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    let hi_with_carry = u64_add(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(carry_from_lo, 0u));
    var out: Fix128Gpu;
    out.hi_lo = hi_with_carry.x;
    out.hi_hi = hi_with_carry.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

// -- fix128 mul: signed 128×128 → middle 128 bits (I64F64 semantics) ----------
fn fix128_mul_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let a_lo = vec2<u32>(a.lo_lo, a.lo_hi);
    let a_hi = vec2<u32>(a.hi_lo, a.hi_hi);
    let b_lo = vec2<u32>(b.lo_lo, b.lo_hi);
    let b_hi = vec2<u32>(b.hi_lo, b.hi_hi);
    let ll = u64_mul_wide(a_lo, b_lo);
    let lh = u64_mul_wide(a_lo, b_hi);
    let hl = u64_mul_wide(a_hi, b_lo);
    let hh = u64_mul_wide(a_hi, b_hi);
    let s2a = ll.z + lh.x;
    let c2a = select(0u, 1u, s2a < ll.z);
    let s2b = s2a + hl.x;
    let c2b = select(0u, 1u, s2b < s2a);
    let p2  = s2b;
    let carry_to_3 = c2a + c2b;
    let s3a = ll.w + lh.y;
    let c3a = select(0u, 1u, s3a < ll.w);
    let s3b = s3a + hl.y;
    let c3b = select(0u, 1u, s3b < s3a);
    let p3  = s3b + carry_to_3;
    let c3c = select(0u, 1u, p3 < s3b);
    let carry_to_4 = c3a + c3b + c3c;
    let s4a = lh.z + hl.z;
    let c4a = select(0u, 1u, s4a < lh.z);
    let s4b = s4a + hh.x;
    let c4b = select(0u, 1u, s4b < s4a);
    let p4_u = s4b + carry_to_4;
    let c4c = select(0u, 1u, p4_u < s4b);
    let carry_to_5 = c4a + c4b + c4c;
    let s5a = lh.w + hl.w;
    let s5b = s5a + hh.y;
    let p5_u = s5b + carry_to_5;
    let a_negative = (a.hi_hi & 0x80000000u) != 0u;
    let b_negative = (b.hi_hi & 0x80000000u) != 0u;
    var p4 = p4_u;
    var p5 = p5_u;
    if (a_negative) {
        let new_p4 = p4 - b.lo_lo;
        let borrow = select(0u, 1u, p4 < b.lo_lo);
        let new_p5 = p5 - b.lo_hi - borrow;
        p4 = new_p4;
        p5 = new_p5;
    }
    if (b_negative) {
        let new_p4 = p4 - a.lo_lo;
        let borrow = select(0u, 1u, p4 < a.lo_lo);
        let new_p5 = p5 - a.lo_hi - borrow;
        p4 = new_p4;
        p5 = new_p5;
    }
    var out: Fix128Gpu;
    out.lo_lo = p2;
    out.lo_hi = p3;
    out.hi_lo = p4;
    out.hi_hi = p5;
    return out;
}

// One dispatch = one PGS iteration. Every thread updates a distinct
// `positions[idx]` / `velocities[idx]` pair via semi-implicit Euler
// with per-axis gravity:
//
//   v' = v + g_axis * dt
//   p' = p + v' * dt
//
// The axis is `idx % 3` because bodies are laid out as flat
// `Fix128Gpu[]` with three consecutive slots per body (x, y, z).
// Selecting `params.gravity_x/y/z` via a dynamic branch on `axis`
// keeps the uniform block's layout simple and avoids
// `array<Fix128Gpu, 3>` indexing (which requires the index to be
// statically or dynamically uniform — the per-thread `idx % 3` is
// neither). The `select` chain compiles to branchless code on all
// three backends (Metal / Vulkan / DX12).
@compute @workgroup_size(64)
fn fix128_pgs_integrate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }

    // Pick the axis-specific gravity component via branchless select.
    let axis = idx % 3u;
    var g: Fix128Gpu;
    g = params.gravity_x;
    if (axis == 1u) { g = params.gravity_y; }
    if (axis == 2u) { g = params.gravity_z; }

    let g_dt   = fix128_mul_kernel(g, params.dt);
    let v_new  = fix128_add_kernel(velocities[idx], g_dt);
    velocities[idx] = v_new;
    let v_dt   = fix128_mul_kernel(v_new, params.dt);
    positions[idx]  = fix128_add_kernel(positions[idx], v_dt);
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
    pipeline_mul: wgpu::ComputePipeline,
    pipeline_dot_partial: wgpu::ComputePipeline,
    pipeline_dot_final: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_layout_dot_final: wgpu::BindGroupLayout,
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

        let shader_mul = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_mul_shader"),
                source: wgpu::ShaderSource::Wgsl(FIX128_MUL_WGSL.into()),
            });

        let shader_dot = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_dot_shader"),
                source: wgpu::ShaderSource::Wgsl(FIX128_DOT_WGSL.into()),
            });

        let shader_dot_final = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_dot_final_shader"),
                source: wgpu::ShaderSource::Wgsl(FIX128_DOT_FINAL_WGSL.into()),
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

        // 2-buffer bind group layout for the Phase 2 dot final pass.
        let bind_group_layout_dot_final =
            device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fix128_dot_final_bgl"),
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

        let pipeline_layout_dot_final =
            device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("fix128_dot_final_pl"),
                    bind_group_layouts: &[&bind_group_layout_dot_final],
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
        let pipeline_mul =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fix128_mul_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_mul,
                    entry_point: Some("fix128_mul_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let pipeline_dot_partial =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fix128_dot_partial_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_dot,
                    entry_point: Some("fix128_dot_partial_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let pipeline_dot_final =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fix128_dot_final_pipeline"),
                    layout: Some(&pipeline_layout_dot_final),
                    module: &shader_dot_final,
                    entry_point: Some("fix128_dot_final_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        Self {
            device,
            pipeline_add,
            pipeline_sub,
            pipeline_mul,
            pipeline_dot_partial,
            pipeline_dot_final,
            bind_group_layout,
            bind_group_layout_dot_final,
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

        let byte_size = std::mem::size_of_val(a) as u64;
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
            let workgroups = (a.len() as u32).div_ceil(64);
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

    /// Dispatch the Fix128 `mul` kernel. `a`, `b`, and `out` must
    /// have identical lengths; the caller owns the output slice.
    ///
    /// Bit-for-bit equivalent to [`Fix128Gpu::mul`] on every input;
    /// see the shader source (`FIX128_MUL_WGSL::fix128_mul_main`)
    /// for the signed 128×128→256 schoolbook algorithm.
    pub fn mul(&self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        self.dispatch_binary(&self.pipeline_mul, "fix128_mul", a, b, out);
    }

    /// Dispatch the Fix128 `dot` kernel — computes `Σ a[i] × b[i]`
    /// via a two-phase multi-workgroup pipeline that preserves
    /// canonical index order (determinism contract §1 経路 3).
    ///
    /// # Pipeline
    ///
    /// - **Phase 1** (`FIX128_DOT_WGSL::fix128_dot_partial_main`):
    ///   dispatched with K = ⌈N / 4096⌉ workgroups of 64 threads.
    ///   Each workgroup reduces its 4096-element chunk in-block-order
    ///   and writes one partial to `partials_buf[wg]`.
    /// - **Phase 2** (`FIX128_DOT_FINAL_WGSL::fix128_dot_final_main`):
    ///   one workgroup of one thread folds `partials_buf[0..K]` in
    ///   workgroup-index order into `output[0]`.
    ///
    /// Both passes are recorded in a single `CommandEncoder`, so
    /// wgpu inserts the storage-buffer barrier between them. The
    /// workgroup completion order in Phase 1 does not affect the
    /// result because each workgroup writes a distinct
    /// `partials_buf` slot; the total arithmetic order is fully
    /// determined by the Phase 2 loop.
    ///
    /// `a` and `b` must have identical lengths. Returns `ZERO` when
    /// the inputs are empty. Bit-for-bit equivalent to the CPU
    /// reference `for i { acc = acc.add(a[i].mul(b[i])) }` loop.
    pub fn dot(&self, a: &[Fix128Gpu], b: &[Fix128Gpu]) -> Fix128Gpu {
        assert_eq!(a.len(), b.len(), "Fix128 dot: input slice length mismatch");
        if a.is_empty() {
            return Fix128Gpu::ZERO;
        }

        // K workgroups cover N elements at 4096 per workgroup. This
        // constant must match `ELEMS_PER_WORKGROUP` in FIX128_DOT_WGSL.
        const ELEMS_PER_WORKGROUP: u32 = 4096;
        let n = a.len() as u32;
        let k = n.div_ceil(ELEMS_PER_WORKGROUP);

        let fix128_bytes = std::mem::size_of::<Fix128Gpu>() as u64;
        let partials_bytes = u64::from(k) * fix128_bytes;
        let out_bytes = fix128_bytes;

        let buf_a = self
            .device
            .create_buffer_init("fix128_dot_a", bytemuck::cast_slice(a));
        let buf_b = self
            .device
            .create_buffer_init("fix128_dot_b", bytemuck::cast_slice(b));
        let buf_partials = self
            .device
            .create_buffer_empty("fix128_dot_partials", partials_bytes);
        let buf_out = self.device.create_buffer_empty("fix128_dot_out", out_bytes);

        // Phase 1 bind group: (input_a, input_b, partials_out) — 3-buffer layout.
        let bind_group_partial =
            self.device
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fix128_dot_partial_bg"),
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
                            resource: buf_partials.as_entire_binding(),
                        },
                    ],
                });

        // Phase 2 bind group: (partials_in, output) — 2-buffer layout.
        let bind_group_final = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fix128_dot_final_bg"),
                layout: &self.bind_group_layout_dot_final,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_partials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fix128_dot_partial_enc"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fix128_dot_partial_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dot_partial);
            pass.set_bind_group(0, &bind_group_partial, &[]);
            pass.dispatch_workgroups(k, 1, 1);
        }
        // Force a device-level sync between phases: two compute passes
        // in the same encoder crashed the DX12 WARP driver on
        // windows-latest CI runners (STATUS_ACCESS_VIOLATION), so we
        // submit each phase in its own encoder. The overhead is one
        // extra submit — negligible relative to the Phase 1 dispatch.
        self.device.submit(encoder);
        self.device.poll_wait();

        let mut encoder_final =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fix128_dot_final_enc"),
                });
        {
            let mut pass = encoder_final.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fix128_dot_final_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dot_final);
            pass.set_bind_group(0, &bind_group_final, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.device.submit(encoder_final);
        self.device.poll_wait();

        let raw = self.device.read_buffer(&buf_out, out_bytes);
        let result: &[Fix128Gpu] = bytemuck::cast_slice(&raw);
        result[0]
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

    fn mul(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        Fix128WgpuKernel::mul(self, a, b, out);
    }

    fn dot(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut Fix128Gpu) {
        *out = Fix128WgpuKernel::dot(self, a, b);
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

    /// `mul` where either operand equals `Fix128Gpu::ONE` must return
    /// the other operand unchanged.
    #[test]
    fn fix128_gpu_mul_one_is_identity() {
        let x = Fix128Gpu::from_raw(42, 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(x.mul(Fix128Gpu::ONE), x);
        assert_eq!(Fix128Gpu::ONE.mul(x), x);
    }

    /// `mul` on pure integer operands must match ordinary signed
    /// integer multiplication (no fractional carry produced).
    #[test]
    fn fix128_gpu_mul_integer_operands() {
        let three = Fix128Gpu::from_raw(3, 0);
        let five = Fix128Gpu::from_raw(5, 0);
        let out = three.mul(five);
        assert_eq!(out.hi, 15);
        assert_eq!(out.lo, 0);
    }

    /// `mul` respects two's complement sign: `(-2) * 3 == -6`.
    #[test]
    fn fix128_gpu_mul_negative_operand() {
        let neg_two = Fix128Gpu::from_raw(-2, 0);
        let three = Fix128Gpu::from_raw(3, 0);
        let out = neg_two.mul(three);
        assert_eq!(out.hi, -6);
        assert_eq!(out.lo, 0);
    }

    /// `mul` with a fractional operand: `2 * 0.5 == 1`. In Fix128
    /// terms `0.5` is `hi=0, lo=1 << 63`.
    #[test]
    fn fix128_gpu_mul_half_scales_correctly() {
        let two = Fix128Gpu::from_raw(2, 0);
        let half = Fix128Gpu::from_raw(0, 1u64 << 63);
        let out = two.mul(half);
        assert_eq!(out.hi, 1);
        assert_eq!(out.lo, 0);
    }

    /// The Fix128 sub shader source is a non-empty compile-time
    /// constant with the required entry point.
    #[test]
    fn wgsl_sub_shader_is_present() {
        assert!(FIX128_SUB_WGSL.contains("fix128_sub_main"));
        assert!(FIX128_SUB_WGSL.contains("@compute"));
        assert!(FIX128_SUB_WGSL.contains("u64_sub"));
    }

    /// The Fix128 mul skeleton shader ships the schoolbook helpers
    /// (`umul_wide` + `u64_mul_wide`) and the unsigned-lo entry point
    /// that will be re-used by the full signed pipeline.
    #[test]
    fn wgsl_mul_shader_helpers_present() {
        assert!(FIX128_MUL_WGSL.contains("umul_wide"));
        assert!(FIX128_MUL_WGSL.contains("u64_mul_wide"));
        assert!(FIX128_MUL_WGSL.contains("fix128_mul_unsigned_lo_main"));
        assert!(FIX128_MUL_WGSL.contains("@compute"));
    }

    /// The mul skeleton must compile as valid WGSL. We ask the wgpu
    /// naga parser (via `Device::create_shader_module`) to validate
    /// it and fail loudly on any syntax / type error.
    ///
    /// Skips when no GPU adapter is available (headless CI).
    #[test]
    fn wgsl_mul_shader_compiles() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        // `create_shader_module` panics on WGSL parse errors, which
        // is the strongest compile-time signal we get short of
        // running the shader.
        let _ = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_mul_skeleton"),
                source: wgpu::ShaderSource::Wgsl(FIX128_MUL_WGSL.into()),
            });
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

    /// GPU dispatch bit-exact contract for `mul` — feeds four
    /// representative fixtures through the WGSL signed 128×128→256
    /// pipeline and checks byte-for-byte equivalence with the CPU
    /// reference [`Fix128Gpu::mul`].
    ///
    /// Fixtures cover the four sign-correction paths:
    ///   1. Identity   (1 × x = x)
    ///   2. Integer    (2 × 3 = 6)
    ///   3. Negative   (-1 × 3 = -3, exercises the `a_negative` branch)
    ///   4. Fractional (0.5 × 0.5 = 0.25, exercises the lo × lo carry-out
    ///      into the middle-lo output word)
    #[test]
    fn wgpu_mul_matches_cpu_golden() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let kernel = Fix128WgpuKernel::new(&device);

        // Fix128 one = raw(1, 0); Fix128 half = raw(0, 1 << 63).
        let half_lo: u64 = 1u64 << 63;
        let a = vec![
            Fix128Gpu::from_raw(1, 0),       // 1
            Fix128Gpu::from_raw(2, 0),       // 2
            Fix128Gpu::from_raw(-1, 0),      // -1
            Fix128Gpu::from_raw(0, half_lo), // 0.5
        ];
        let b = vec![
            Fix128Gpu::from_raw(-3, 42),     // arbitrary
            Fix128Gpu::from_raw(3, 0),       // 3
            Fix128Gpu::from_raw(3, 0),       // 3
            Fix128Gpu::from_raw(0, half_lo), // 0.5
        ];
        let mut out = vec![Fix128Gpu::ZERO; 4];

        kernel.mul(&a, &b, &mut out);

        for i in 0..a.len() {
            let cpu_ref = a[i].mul(b[i]);
            assert_eq!(
                out[i].hi, cpu_ref.hi,
                "mul hi mismatch at i={i}: GPU {} vs CPU {}",
                out[i].hi, cpu_ref.hi
            );
            assert_eq!(
                out[i].lo, cpu_ref.lo,
                "mul lo mismatch at i={i}: GPU {:#x} vs CPU {:#x}",
                out[i].lo, cpu_ref.lo
            );
        }
    }

    /// The `Fix128GpuKernel` trait bridge routes `add` / `sub` / `mul`
    /// to the live wgpu pipelines while `dot` remains
    /// `unimplemented!`. Verifies the trait `mul` matches
    /// [`Fix128WgpuKernel::mul`] byte-for-byte.
    #[test]
    fn wgpu_trait_mul_matches_inherent_mul() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut kernel = Fix128WgpuKernel::new(&device);

        let a = vec![Fix128Gpu::from_raw(1, 0), Fix128Gpu::from_raw(-1, 0)];
        let b = vec![Fix128Gpu::from_raw(3, 0), Fix128Gpu::from_raw(3, 0)];
        let mut via_inherent = vec![Fix128Gpu::ZERO; 2];
        let mut via_trait = vec![Fix128Gpu::ZERO; 2];

        Fix128WgpuKernel::mul(&kernel, &a, &b, &mut via_inherent);
        <Fix128WgpuKernel<'_> as Fix128GpuKernel>::mul(&mut kernel, &a, &b, &mut via_trait);

        assert_eq!(via_inherent, via_trait);
    }

    /// The Fix128 dot skeleton shader ships the schoolbook helpers,
    /// the inline `fix128_add_kernel` / `fix128_mul_kernel`, and the
    /// serial reduction entry point.
    #[test]
    fn wgsl_dot_shader_helpers_present() {
        assert!(FIX128_DOT_WGSL.contains("umul_wide"));
        assert!(FIX128_DOT_WGSL.contains("u64_mul_wide"));
        assert!(FIX128_DOT_WGSL.contains("fix128_add_kernel"));
        assert!(FIX128_DOT_WGSL.contains("fix128_mul_kernel"));
        assert!(FIX128_DOT_WGSL.contains("fix128_dot_partial_main"));
        assert!(FIX128_DOT_WGSL.contains("@compute"));
        assert!(FIX128_DOT_WGSL.contains("@workgroup_size(64)"));
        assert!(FIX128_DOT_WGSL.contains("var<workgroup> partials"));
        assert!(FIX128_DOT_WGSL.contains("workgroupBarrier"));
        assert!(FIX128_DOT_WGSL.contains("ELEMS_PER_WORKGROUP"));
        // Note: v0.8.1 rewrite eliminated `@builtin(workgroup_id)` in
        // favour of deriving the workgroup index from
        // `global_invocation_id` (see the FIX128_DOT_WGSL source).
        assert!(FIX128_DOT_WGSL.contains("global_invocation_id"));
    }

    /// The Fix128 dot Phase 2 shader ships the fold helpers and the
    /// single-thread final-accumulate entry point.
    #[test]
    fn wgsl_dot_final_shader_helpers_present() {
        assert!(FIX128_DOT_FINAL_WGSL.contains("u64_add"));
        assert!(FIX128_DOT_FINAL_WGSL.contains("fix128_add_kernel"));
        assert!(FIX128_DOT_FINAL_WGSL.contains("fix128_dot_final_main"));
        assert!(FIX128_DOT_FINAL_WGSL.contains("@compute"));
        assert!(FIX128_DOT_FINAL_WGSL.contains("@workgroup_size(1)"));
    }

    /// The dot Phase 2 shader must compile as valid WGSL. Skips when
    /// no GPU adapter is available (headless CI).
    #[test]
    fn wgsl_dot_final_shader_compiles() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let _ = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_dot_final_shader_compile_test"),
                source: wgpu::ShaderSource::Wgsl(FIX128_DOT_FINAL_WGSL.into()),
            });
    }

    /// The dot shader must compile as valid WGSL. Skips when no GPU
    /// adapter is available (headless CI).
    #[test]
    fn wgsl_dot_shader_compiles() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let _ = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_dot_shader_compile_test"),
                source: wgpu::ShaderSource::Wgsl(FIX128_DOT_WGSL.into()),
            });
    }

    /// GPU dispatch bit-exact contract for `dot` — three fixtures
    /// covering the single-element / positive-only / mixed-sign paths
    /// against the CPU golden `acc = acc.add(a[i].mul(b[i]))` loop.
    #[test]
    fn wgpu_dot_matches_cpu_golden() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let kernel = Fix128WgpuKernel::new(&device);

        // CPU golden: index-ordered fold via Fix128Gpu::mul + add.
        fn cpu_dot(a: &[Fix128Gpu], b: &[Fix128Gpu]) -> Fix128Gpu {
            let mut acc = Fix128Gpu::ZERO;
            for i in 0..a.len() {
                acc = acc.add(a[i].mul(b[i]));
            }
            acc
        }

        // Fixture 1 — single element (7 × 3 = 21)
        let a1 = vec![Fix128Gpu::from_raw(7, 0)];
        let b1 = vec![Fix128Gpu::from_raw(3, 0)];
        let gpu1 = kernel.dot(&a1, &b1);
        let cpu1 = cpu_dot(&a1, &b1);
        assert_eq!(gpu1, cpu1, "single-element dot mismatch");

        // Fixture 2 — 4 positive integers Σ = 1·2 + 3·4 + 5·6 + 7·8 = 100
        let a2 = vec![
            Fix128Gpu::from_raw(1, 0),
            Fix128Gpu::from_raw(3, 0),
            Fix128Gpu::from_raw(5, 0),
            Fix128Gpu::from_raw(7, 0),
        ];
        let b2 = vec![
            Fix128Gpu::from_raw(2, 0),
            Fix128Gpu::from_raw(4, 0),
            Fix128Gpu::from_raw(6, 0),
            Fix128Gpu::from_raw(8, 0),
        ];
        let gpu2 = kernel.dot(&a2, &b2);
        let cpu2 = cpu_dot(&a2, &b2);
        assert_eq!(gpu2, cpu2, "4-element positive dot mismatch");
        assert_eq!(gpu2.hi, 100, "4-element expected Σ = 100");
        assert_eq!(gpu2.lo, 0);

        // Fixture 3 — mixed sign Σ = (-1)·3 + 2·(-4) + 5·6 = -3 - 8 + 30 = 19
        let a3 = vec![
            Fix128Gpu::from_raw(-1, 0),
            Fix128Gpu::from_raw(2, 0),
            Fix128Gpu::from_raw(5, 0),
        ];
        let b3 = vec![
            Fix128Gpu::from_raw(3, 0),
            Fix128Gpu::from_raw(-4, 0),
            Fix128Gpu::from_raw(6, 0),
        ];
        let gpu3 = kernel.dot(&a3, &b3);
        let cpu3 = cpu_dot(&a3, &b3);
        assert_eq!(gpu3, cpu3, "mixed-sign dot mismatch");
        assert_eq!(gpu3.hi, 19, "mixed-sign expected Σ = 19");
    }

    /// v0.7.0 parallelisation stress test — feeds 100 elements
    /// through the 64-thread blocked reduction and checks byte-for-
    /// byte equivalence with the single-thread CPU reference
    /// (`for i { acc = acc.add(a[i].mul(b[i])) }`).
    ///
    /// This is the primary determinism proof for the blocked path:
    /// N > WG_SIZE (64) exercises the block-size clamp, the
    /// mid-block boundary, and the trailing shrunk block, so a
    /// successful golden match confirms the block-index-order
    /// serial final accumulate reproduces the canonical index order.
    #[test]
    fn wgpu_dot_parallel_100_matches_cpu_golden() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let kernel = Fix128WgpuKernel::new(&device);

        // Mixed-sign fixture with a deliberate hi/lo interleave so
        // that reordering the block-internal sums would flip
        // wraparound behaviour at the 128-bit boundary.
        let a: Vec<Fix128Gpu> = (0..100)
            .map(|i| {
                let sign = if i % 3 == 0 { -1 } else { 1 };
                Fix128Gpu::from_raw(sign * i, (i as u64) << 32)
            })
            .collect();
        let b: Vec<Fix128Gpu> = (0..100)
            .map(|i| {
                let sign = if i % 5 == 0 { -1 } else { 1 };
                Fix128Gpu::from_raw(sign * (i + 1), ((100 - i) as u64) << 16)
            })
            .collect();

        let gpu = kernel.dot(&a, &b);

        // Single-thread CPU reference.
        let mut cpu = Fix128Gpu::ZERO;
        for i in 0..a.len() {
            cpu = cpu.add(a[i].mul(b[i]));
        }

        assert_eq!(
            gpu, cpu,
            "N=100 parallel dot mismatch: GPU {gpu:?} vs CPU {cpu:?}"
        );
    }

    /// v0.7.1 multi-workgroup stress test — feeds 10 000 elements
    /// through the Phase 1 + Phase 2 pipeline. K = ⌈10 000 / 4096⌉ = 3
    /// workgroups, so this exercises:
    ///
    /// - Cross-workgroup partial writes to `partials_buf[0..3]`
    /// - The Phase 2 workgroup-index-ordered serial fold
    /// - Storage-buffer barrier between the two dispatches
    ///
    /// A successful byte-for-byte match with the single-thread CPU
    /// reference proves that the total arithmetic order is
    /// canonical index 0..N regardless of workgroup completion
    /// scheduling.
    /// Skipped on windows-latest CI at the workflow level (see
    /// `.github/workflows/ci.yml` — `--skip
    /// wgpu_dot_large_10000_matches_cpu_golden` on the Windows job).
    /// DX12 WARP crashed STATUS_ACCESS_VIOLATION on this fixture
    /// (N = 10 000, K = 3 workgroups); the smaller
    /// `wgpu_dot_parallel_100_matches_cpu_golden` (K = 1) already
    /// exercises the two-phase pipeline on WARP.
    #[test]
    fn wgpu_dot_large_10000_matches_cpu_golden() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let kernel = Fix128WgpuKernel::new(&device);

        // Vary sign and hi/lo bits across the range so re-ordered
        // sums would produce different wraparound.
        let a: Vec<Fix128Gpu> = (0..10_000i64)
            .map(|i| {
                let sign = if i % 7 == 0 { -1 } else { 1 };
                Fix128Gpu::from_raw(sign * i, ((i as u64) & 0xFFFF) << 40)
            })
            .collect();
        let b: Vec<Fix128Gpu> = (0..10_000i64)
            .map(|i| {
                let sign = if i % 11 == 0 { -1 } else { 1 };
                Fix128Gpu::from_raw(sign * (i + 1), ((i as u64).wrapping_mul(37)) & 0xFFFF)
            })
            .collect();

        let gpu = kernel.dot(&a, &b);

        let mut cpu = Fix128Gpu::ZERO;
        for i in 0..a.len() {
            cpu = cpu.add(a[i].mul(b[i]));
        }

        assert_eq!(
            gpu, cpu,
            "N=10000 multi-workgroup dot mismatch: GPU {gpu:?} vs CPU {cpu:?}"
        );
    }

    /// Empty-input contract: dot returns `ZERO` without dispatching.
    #[test]
    fn wgpu_dot_zero_elements_returns_zero() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let kernel = Fix128WgpuKernel::new(&device);
        let result = kernel.dot(&[], &[]);
        assert_eq!(result, Fix128Gpu::ZERO);
    }

    /// The `Fix128GpuKernel` trait bridge routes `dot` to the live
    /// wgpu pipeline. Verifies the trait `dot` matches
    /// [`Fix128WgpuKernel::dot`] byte-for-byte.
    #[test]
    fn wgpu_trait_dot_matches_inherent_dot() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut kernel = Fix128WgpuKernel::new(&device);

        let a = vec![
            Fix128Gpu::from_raw(1, 0),
            Fix128Gpu::from_raw(2, 0),
            Fix128Gpu::from_raw(-3, 0),
        ];
        let b = vec![
            Fix128Gpu::from_raw(4, 0),
            Fix128Gpu::from_raw(-5, 0),
            Fix128Gpu::from_raw(6, 0),
        ];

        let via_inherent = Fix128WgpuKernel::dot(&kernel, &a, &b);
        let mut via_trait = Fix128Gpu::ZERO;
        <Fix128WgpuKernel<'_> as Fix128GpuKernel>::dot(&mut kernel, &a, &b, &mut via_trait);

        assert_eq!(via_inherent, via_trait);
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
