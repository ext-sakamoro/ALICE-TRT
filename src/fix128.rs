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

impl std::fmt::Display for Fix128Gpu {
    /// Formats as the `f64` approximation so log lines and panic
    /// messages remain human-readable. Uses the same non-deterministic
    /// [`Self::to_f64`] path — never rely on `Display` output inside
    /// determinism-sensitive assertions.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f64())
    }
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

    /// Construct a `Fix128Gpu` from a signed integer.
    ///
    /// The result has `hi = n` and `lo = 0`, matching
    /// `alice_physics::math::Fix128::from_int` byte-for-byte in the
    /// I64F64 layout that both types share.
    #[must_use]
    pub const fn from_int(n: i64) -> Self {
        Self { hi: n, lo: 0 }
    }

    /// True when the Fix128 value is strictly less than zero.
    ///
    /// Uses the sign bit of the high half (`hi < 0` in two's-
    /// complement) — the same test the WGSL floor projection kernel
    /// performs via `hi_hi & 0x8000_0000u`. Keeping the CPU and GPU
    /// paths in lockstep lets tests assert identical behaviour on
    /// either side of the bridge.
    #[must_use]
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.hi < 0
    }

    /// True when the Fix128 value is exactly zero (both `hi` and
    /// `lo` halves are zero).
    #[must_use]
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.hi == 0 && self.lo == 0
    }

    /// True when the Fix128 value is strictly greater than zero.
    ///
    /// The unusual `else if` structure (rather than
    /// `!self.is_negative() && !self.is_zero()`) keeps the method
    /// `const fn`-compatible on the current MSRV; `const fn` bodies
    /// cannot call other trait methods yet.
    #[must_use]
    #[inline]
    pub const fn is_positive(self) -> bool {
        if self.hi > 0 {
            true
        } else if self.hi < 0 {
            false
        } else {
            self.lo > 0
        }
    }

    /// Absolute value.
    ///
    /// Delegates to the existing `sub` primitive so the two's-
    /// complement handling stays in one place: for negative values,
    /// the result is `Self::ZERO.sub(self)`. Positive values and
    /// zero are returned unchanged.
    #[must_use]
    #[inline]
    pub const fn abs(self) -> Self {
        if self.is_negative() {
            Self::ZERO.sub(self)
        } else {
            self
        }
    }

    /// Unary minus (arithmetic negation).
    ///
    /// Equivalent to `Self::ZERO.sub(self)` — kept as a first-class
    /// method so callers can spell it in the natural way rather than
    /// constructing the zero themselves. `const fn` compatible.
    #[must_use]
    #[inline]
    pub const fn neg(self) -> Self {
        Self::ZERO.sub(self)
    }

    /// Approximate `f64` conversion for logging / debugging only.
    ///
    /// **Not deterministic** — floating-point rounding depends on the
    /// host CPU's rounding mode. Use this to eyeball values in tests
    /// and print statements, never inside determinism-sensitive code
    /// paths. For the deterministic conversion the ecosystem relies
    /// on, go through
    /// `alice_physics::math::Fix128::from_raw(gpu.hi, gpu.lo)`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        // Value = hi + lo / 2^64; construct as f64 in one expression
        // so the compiler can fold the divisor into the constant pool.
        (self.hi as f64) + (self.lo as f64) / ((1u128 << 64) as f64)
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
    /// `alice_physics::math::Fix128::mul` so the WGSL kernel
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

/// WGSL compute shader source for **`Fix128 div`** — element-wise
/// division `out[i] = a[i] / b[i]` on the GPU (v1.4.0).
///
/// # Algorithm
///
/// Faithful port of the CPU reference (`impl Div for alice_physics::Fix128`
/// in `src/math.rs`), split into three phases per element:
///
/// 1. **Sign extraction** — extract the sign bit of both operands from
///    bit 31 of `hi_hi` (two's-complement MSB of the `i64 hi` field),
///    then absolutise both operands via two's-complement negation.
/// 2. **Integer quotient** — 128-iteration binary long division of the
///    absolute `a_full` by `b_full` (both u128 as `vec4<u32>` with `.x`
///    the least-significant word), producing a 128-bit integer quotient
///    (`quot_hi`) and remainder (`rem`). Only the low 64 bits of
///    `quot_hi` map to `result.hi` (the CPU code does `hi: quot_hi as i64`).
/// 3. **Fractional quotient** — 64-iteration bit-by-bit refinement of
///    the remainder, mirroring the CPU loop exactly (`overflow_bit ||
///    r >= b_full` → subtract & set bit).
/// 4. **Sign restoration** — `neg(result)` if operand signs differed.
///
/// # Determinism contract
///
/// - No `workgroupBarrier()` (per-element scalar op, no cross-thread
///   dependency).
/// - Uniform flow gate: divisor-is-zero returns the same 0 value on
///   every thread and every platform (guard the entire body with an
///   early return **only** at the outermost level, all subsequent
///   loops run for a fixed 128 + 64 iteration count without early exit).
/// - No subgroup ops, no atomics; each thread produces its own answer
///   independently.
///
/// # Layout
///
/// Same 3-buffer binding as [`FIX128_ADD_WGSL`] (`input_a`, `input_b`,
/// `output` all `array<Fix128Gpu>` of equal length). Entry point
/// `@compute @workgroup_size(64) fn fix128_div_main`.
///
/// # CPU reference
///
/// [`Fix128Gpu::div`] delegates to `alice_physics::math::Fix128 / Fix128`
/// which uses the same algorithm; every GPU-produced element must equal
/// the CPU result byte-for-byte (verified in the sibling
/// `fix128_div_gpu_matches_cpu_reference` integration test).
pub const FIX128_DIV_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

@group(0) @binding(0) var<storage, read>       input_a: array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read>       input_b: array<Fix128Gpu>;
@group(0) @binding(2) var<storage, read_write> output:  array<Fix128Gpu>;

// ---- 64-bit helpers (reused from add/sub/mul patterns) ----

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

// ---- Two's complement negation on the packed Fix128Gpu ----
fn fix128_neg(x: Fix128Gpu) -> Fix128Gpu {
    // ~x + 1 across all 128 bits.
    let inv_lo_lo = ~x.lo_lo;
    let inv_lo_hi = ~x.lo_hi;
    let inv_hi_lo = ~x.hi_lo;
    let inv_hi_hi = ~x.hi_hi;

    // Add 1 to the least-significant word, propagating carry.
    let s0 = inv_lo_lo + 1u;
    let c0 = select(0u, 1u, s0 < inv_lo_lo);
    let s1 = inv_lo_hi + c0;
    let c1 = select(0u, 1u, s1 < inv_lo_hi);
    let s2 = inv_hi_lo + c1;
    let c2 = select(0u, 1u, s2 < inv_hi_lo);
    let s3 = inv_hi_hi + c2;

    var out: Fix128Gpu;
    out.lo_lo = s0;
    out.lo_hi = s1;
    out.hi_lo = s2;
    out.hi_hi = s3;
    return out;
}

fn fix128_is_zero(x: Fix128Gpu) -> bool {
    return (x.hi_lo | x.hi_hi | x.lo_lo | x.lo_hi) == 0u;
}

fn fix128_is_negative(x: Fix128Gpu) -> bool {
    // Two's complement sign bit = bit 31 of hi_hi (MSB of the `i64 hi` field).
    return ((x.hi_hi >> 31u) & 1u) == 1u;
}

fn fix128_abs(x: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_negative(x)) {
        return fix128_neg(x);
    }
    return x;
}

// ---- u128 helpers as vec4<u32> (LSB word first) ----

fn u128_shl1(x: vec4<u32>) -> vec4<u32> {
    let c0 = x.x >> 31u;
    let c1 = x.y >> 31u;
    let c2 = x.z >> 31u;
    return vec4<u32>(x.x << 1u, (x.y << 1u) | c0, (x.z << 1u) | c1, (x.w << 1u) | c2);
}

fn u128_ge(a: vec4<u32>, b: vec4<u32>) -> bool {
    if (a.w != b.w) { return a.w > b.w; }
    if (a.z != b.z) { return a.z > b.z; }
    if (a.y != b.y) { return a.y > b.y; }
    return a.x >= b.x;
}

// Unsigned subtract, assuming a >= b (caller-enforced).
fn u128_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    // word 0
    let d0        = a.x - b.x;
    let borrow0   = select(0u, 1u, a.x < b.x);
    // word 1 (subtract b.y, then propagate borrow0)
    let m1        = a.y - b.y;
    let borrow1a  = select(0u, 1u, a.y < b.y);
    let d1        = m1 - borrow0;
    let borrow1b  = select(0u, 1u, m1 < borrow0);
    let borrow1   = borrow1a + borrow1b;
    // word 2
    let m2        = a.z - b.z;
    let borrow2a  = select(0u, 1u, a.z < b.z);
    let d2        = m2 - borrow1;
    let borrow2b  = select(0u, 1u, m2 < borrow1);
    let borrow2   = borrow2a + borrow2b;
    // word 3 (top; caller guarantees no final borrow-out because a >= b)
    let m3        = a.w - b.w;
    let d3        = m3 - borrow2;
    return vec4<u32>(d0, d1, d2, d3);
}

// Set bit at position `bit_pos` (0..128) of a vec4<u32> u128 value.
fn u128_set_bit(x: vec4<u32>, bit_pos: u32) -> vec4<u32> {
    let word  = bit_pos >> 5u;             // /32
    let shift = bit_pos & 31u;             // %32
    let mask  = 1u << shift;
    var out = x;
    // Straight-line switch (per-thread divergence is fine; no barrier).
    if      (word == 0u) { out.x = out.x | mask; }
    else if (word == 1u) { out.y = out.y | mask; }
    else if (word == 2u) { out.z = out.z | mask; }
    else                 { out.w = out.w | mask; }
    return out;
}

fn fix128_zero() -> Fix128Gpu {
    return Fix128Gpu(0u, 0u, 0u, 0u);
}

fn fix128_div(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    // Mirrors CPU: zero divisor OR a=0 → return ZERO.
    if (fix128_is_zero(b)) { return fix128_zero(); }

    let neg_a = fix128_is_negative(a);
    let neg_b = fix128_is_negative(b);
    let result_neg = neg_a != neg_b;

    let abs_a = fix128_abs(a);
    let abs_b = fix128_abs(b);

    // Pack Fix128Gpu into a vec4<u32> u128 (LSB word first).
    let a_full = vec4<u32>(abs_a.lo_lo, abs_a.lo_hi, abs_a.hi_lo, abs_a.hi_hi);
    let b_full = vec4<u32>(abs_b.lo_lo, abs_b.lo_hi, abs_b.hi_lo, abs_b.hi_hi);

    // Phase 1: 128-iter binary long division for the integer quotient.
    // Processes bits of `a_full` from MSB (bit 127) down to LSB (bit 0),
    // shifting each into the running remainder. When remainder >= divisor
    // subtract and set the corresponding bit of the quotient.
    var q_int = vec4<u32>(0u);
    var r_int = vec4<u32>(0u);
    var d     = a_full;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let msb = d.w >> 31u;
        r_int   = u128_shl1(r_int);
        r_int.x = r_int.x | msb;
        d       = u128_shl1(d);
        if (u128_ge(r_int, b_full)) {
            r_int = u128_sub(r_int, b_full);
            q_int = u128_set_bit(q_int, u32(127 - i));
        }
    }

    // Phase 2: 64-iter refinement for the fractional quotient.
    // CPU: `overflow_bit = r >> 127; r <<= 1; if overflow_bit || r >= b_full { r -= b_full; quot_lo |= 1 << i; }`
    var quot_lo_lo: u32 = 0u;
    var quot_lo_hi: u32 = 0u;
    var r = r_int;
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let overflow_bit = r.w >> 31u;
        r = u128_shl1(r);
        if (overflow_bit != 0u || u128_ge(r, b_full)) {
            r = u128_sub(r, b_full);
            if (i < 32) {
                quot_lo_lo = quot_lo_lo | (1u << u32(i));
            } else {
                quot_lo_hi = quot_lo_hi | (1u << u32(i - 32));
            }
        }
    }

    // Assemble result: low 64 bits of q_int → result.hi (CPU `hi: quot_hi as i64`).
    var result: Fix128Gpu;
    result.hi_lo = q_int.x;
    result.hi_hi = q_int.y;
    result.lo_lo = quot_lo_lo;
    result.lo_hi = quot_lo_hi;

    if (result_neg) { return fix128_neg(result); }
    return result;
}

@compute @workgroup_size(64)
fn fix128_div_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_a)) { return; }
    output[i] = fix128_div(input_a[i], input_b[i]);
}
"#;

/// WGSL compute shader source for **`Fix128 sqrt`** — element-wise
/// square root `out[i] = sqrt(a[i])` on the GPU (v1.4.1).
///
/// # Algorithm
///
/// Byte-for-byte port of the CPU reference (`Fix128::sqrt` in
/// `alice_physics::math`):
///
/// 1. **Zero / negative check** — return `Fix128Gpu::ZERO` if input is
///    `<= 0` (matches CPU).
/// 2. **Initial guess** — deterministic bit-width estimation via
///    `countLeadingZeros`. If `sig_bits` is the position of the highest
///    set bit, the initial `x` is placed at bit `(sig_bits + 63) / 2`
///    of the Fix128 result (a decent power-of-two starting point that
///    converges quickly regardless of input magnitude).
/// 3. **Newton-Raphson** — 64 iterations of `x = (x + a/x) / 2`, using
///    the v1.4.0 GPU division kernel (`fix128_div_kernel`) for `a/x`
///    and a straight bit-shift for the `/ 2`. Fixed iteration count
///    (no early exit) preserves determinism.
///
/// # Determinism contract
///
/// - No `workgroupBarrier()`, no subgroup ops, no atomics.
/// - Each thread is independent — a fixed 64-iteration Newton loop
///   over per-element Fix128 arithmetic, with no cross-thread traffic.
/// - Uses the shared `countLeadingZeros` builtin plus the byte-exact
///   long-division routine from v1.4.0, so cross-platform (Metal /
///   Vulkan / DX12 WARP) equivalence follows from the div contract.
///
/// # Layout
///
/// 3-buffer binding matching [`FIX128_ADD_WGSL`] for pipeline reuse:
/// `input_a` (the value to sqrt), `input_b` (unused, but bound so the
/// existing `dispatch_binary` helper can drive this kernel without a
/// separate unary path), `output`. Entry point
/// `@compute @workgroup_size(64) fn fix128_sqrt_main`.
///
/// # CPU reference
///
/// `Fix128Gpu::sqrt` (in `physics_bridge`) delegates to
/// `alice_physics::math::Fix128::sqrt`, so every GPU-produced element
/// must equal the CPU result byte-for-byte (verified in the sibling
/// `wgpu_sqrt_matches_cpu_golden` integration test).
pub const FIX128_SQRT_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

@group(0) @binding(0) var<storage, read>       input_a: array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read>       input_b: array<Fix128Gpu>;
@group(0) @binding(2) var<storage, read_write> output:  array<Fix128Gpu>;

// ---- Same helpers as FIX128_DIV_WGSL (inlined per-shader per house style) ----

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

fn fix128_neg(x: Fix128Gpu) -> Fix128Gpu {
    let inv_lo_lo = ~x.lo_lo;
    let inv_lo_hi = ~x.lo_hi;
    let inv_hi_lo = ~x.hi_lo;
    let inv_hi_hi = ~x.hi_hi;
    let s0 = inv_lo_lo + 1u;
    let c0 = select(0u, 1u, s0 < inv_lo_lo);
    let s1 = inv_lo_hi + c0;
    let c1 = select(0u, 1u, s1 < inv_lo_hi);
    let s2 = inv_hi_lo + c1;
    let c2 = select(0u, 1u, s2 < inv_hi_lo);
    let s3 = inv_hi_hi + c2;

    var out: Fix128Gpu;
    out.lo_lo = s0;
    out.lo_hi = s1;
    out.hi_lo = s2;
    out.hi_hi = s3;
    return out;
}

fn fix128_is_zero(x: Fix128Gpu) -> bool {
    return (x.hi_lo | x.hi_hi | x.lo_lo | x.lo_hi) == 0u;
}

fn fix128_is_negative(x: Fix128Gpu) -> bool {
    return ((x.hi_hi >> 31u) & 1u) == 1u;
}

fn fix128_abs(x: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_negative(x)) {
        return fix128_neg(x);
    }
    return x;
}

fn fix128_zero() -> Fix128Gpu {
    return Fix128Gpu(0u, 0u, 0u, 0u);
}

// Divide by 2 (arithmetic shift right by 1, sign-preserving on the top word).
fn fix128_half(x: Fix128Gpu) -> Fix128Gpu {
    var out: Fix128Gpu;
    out.lo_lo = (x.lo_lo >> 1u) | ((x.lo_hi & 1u) << 31u);
    out.lo_hi = (x.lo_hi >> 1u) | ((x.hi_lo & 1u) << 31u);
    out.hi_lo = (x.hi_lo >> 1u) | ((x.hi_hi & 1u) << 31u);
    // Signed shift on the top word preserves the sign bit (i32 >> 1 is
    // arithmetic shift in WGSL). Round-trip through i32 to opt into it.
    out.hi_hi = bitcast<u32>(bitcast<i32>(x.hi_hi) >> 1u);
    return out;
}

// ---- u128 helpers (same as FIX128_DIV_WGSL) ----

fn u128_shl1(x: vec4<u32>) -> vec4<u32> {
    let c0 = x.x >> 31u;
    let c1 = x.y >> 31u;
    let c2 = x.z >> 31u;
    return vec4<u32>(x.x << 1u, (x.y << 1u) | c0, (x.z << 1u) | c1, (x.w << 1u) | c2);
}

fn u128_ge(a: vec4<u32>, b: vec4<u32>) -> bool {
    if (a.w != b.w) { return a.w > b.w; }
    if (a.z != b.z) { return a.z > b.z; }
    if (a.y != b.y) { return a.y > b.y; }
    return a.x >= b.x;
}

fn u128_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let d0        = a.x - b.x;
    let borrow0   = select(0u, 1u, a.x < b.x);
    let m1        = a.y - b.y;
    let borrow1a  = select(0u, 1u, a.y < b.y);
    let d1        = m1 - borrow0;
    let borrow1b  = select(0u, 1u, m1 < borrow0);
    let borrow1   = borrow1a + borrow1b;
    let m2        = a.z - b.z;
    let borrow2a  = select(0u, 1u, a.z < b.z);
    let d2        = m2 - borrow1;
    let borrow2b  = select(0u, 1u, m2 < borrow1);
    let borrow2   = borrow2a + borrow2b;
    let m3        = a.w - b.w;
    let d3        = m3 - borrow2;
    return vec4<u32>(d0, d1, d2, d3);
}

fn u128_set_bit(x: vec4<u32>, bit_pos: u32) -> vec4<u32> {
    let word  = bit_pos >> 5u;
    let shift = bit_pos & 31u;
    let mask  = 1u << shift;
    var out = x;
    if      (word == 0u) { out.x = out.x | mask; }
    else if (word == 1u) { out.y = out.y | mask; }
    else if (word == 2u) { out.z = out.z | mask; }
    else                 { out.w = out.w | mask; }
    return out;
}

// Same 128 + 64 iter long division as FIX128_DIV_WGSL::fix128_div.
fn fix128_div_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_zero(b)) { return fix128_zero(); }

    let neg_a = fix128_is_negative(a);
    let neg_b = fix128_is_negative(b);
    let result_neg = neg_a != neg_b;

    let abs_a = fix128_abs(a);
    let abs_b = fix128_abs(b);

    let a_full = vec4<u32>(abs_a.lo_lo, abs_a.lo_hi, abs_a.hi_lo, abs_a.hi_hi);
    let b_full = vec4<u32>(abs_b.lo_lo, abs_b.lo_hi, abs_b.hi_lo, abs_b.hi_hi);

    var q_int = vec4<u32>(0u);
    var r_int = vec4<u32>(0u);
    var d     = a_full;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let msb = d.w >> 31u;
        r_int   = u128_shl1(r_int);
        r_int.x = r_int.x | msb;
        d       = u128_shl1(d);
        if (u128_ge(r_int, b_full)) {
            r_int = u128_sub(r_int, b_full);
            q_int = u128_set_bit(q_int, u32(127 - i));
        }
    }

    var quot_lo_lo: u32 = 0u;
    var quot_lo_hi: u32 = 0u;
    var r = r_int;
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let overflow_bit = r.w >> 31u;
        r = u128_shl1(r);
        if (overflow_bit != 0u || u128_ge(r, b_full)) {
            r = u128_sub(r, b_full);
            if (i < 32) {
                quot_lo_lo = quot_lo_lo | (1u << u32(i));
            } else {
                quot_lo_hi = quot_lo_hi | (1u << u32(i - 32));
            }
        }
    }

    var result: Fix128Gpu;
    result.hi_lo = q_int.x;
    result.hi_hi = q_int.y;
    result.lo_lo = quot_lo_lo;
    result.lo_hi = quot_lo_hi;

    if (result_neg) { return fix128_neg(result); }
    return result;
}

// ---- sqrt ----

fn fix128_sqrt(a: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_negative(a) || fix128_is_zero(a)) {
        return fix128_zero();
    }

    // Initial guess: bit-width estimation.
    // (CPU: `sig_bits = if hi > 0 { 128 - lz(hi as u64) } else { 64 - lz(lo) }`.)
    var sig_bits: u32;
    if (a.hi_hi != 0u || a.hi_lo != 0u) {
        var lz: u32;
        if (a.hi_hi != 0u) { lz = countLeadingZeros(a.hi_hi); }
        else               { lz = 32u + countLeadingZeros(a.hi_lo); }
        sig_bits = 128u - lz;
    } else {
        var lz: u32;
        if (a.lo_hi != 0u) { lz = countLeadingZeros(a.lo_hi); }
        else               { lz = 32u + countLeadingZeros(a.lo_lo); }
        sig_bits = 64u - lz;
    }
    let result_bit = (sig_bits + 63u) / 2u;

    var x: Fix128Gpu;
    x.hi_lo = 0u;
    x.hi_hi = 0u;
    x.lo_lo = 0u;
    x.lo_hi = 0u;
    if (result_bit >= 64u) {
        let shift_raw = result_bit - 64u;
        let shift = min(shift_raw, 62u);
        if (shift < 32u) {
            x.hi_lo = 1u << shift;
        } else {
            x.hi_hi = 1u << (shift - 32u);
        }
    } else {
        if (result_bit < 32u) {
            x.lo_lo = 1u << result_bit;
        } else {
            x.lo_hi = 1u << (result_bit - 32u);
        }
    }

    // Newton-Raphson: x = (x + a/x) / 2, fixed 64 iterations.
    for (var i: i32 = 0; i < 64; i = i + 1) {
        let div = fix128_div_kernel(a, x);
        let sum = fix128_add_kernel(x, div);
        x = fix128_half(sum);
    }

    return x;
}

@compute @workgroup_size(64)
fn fix128_sqrt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_a)) { return; }
    output[i] = fix128_sqrt(input_a[i]);
}
"#;

/// WGSL compute shader source for `Fix128 dot` — Σ `a[i] × b[i]` as a
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

/// WGSL compute shader source for the **Fix128 PGS floor projection**
/// kernel — v0.9.2's minimum-viable constraint solver.
///
/// # Semantics
///
/// For every Y-axis position slot (`idx % 3 == 1`), if the Fix128
/// value is negative (i.e. the body has passed through the ground
/// plane `y = 0`), the slot and its paired velocity slot are both
/// snapped to zero. All other slots are left untouched.
///
/// This is the classical "floor constraint" from position-based
/// dynamics: the world has an infinite ground plane at `y = 0` and
/// bodies cannot penetrate it. Zeroing the velocity prevents the
/// well-known "sinking into the floor" oscillation that would
/// otherwise occur if gravity kept pulling the body down while the
/// position stayed clamped.
///
/// # Determinism
///
/// Each thread reads and writes a single, distinct slot. There is
/// no `workgroupBarrier()`, no cross-thread dependency, and no
/// reduction — so uniformity concerns (which caused the WARP crash
/// fixed in v0.8.1) do not apply. The signed-negative check is a
/// single MSB test of `hi_hi` and cannot diverge across backends.
///
/// # Layout
///
/// - `@group(0) @binding(0) var<storage, read_write> positions: array<Fix128Gpu>`
/// - `@group(0) @binding(1) var<storage, read_write> velocities: array<Fix128Gpu>`
///
/// No uniform block is needed — the floor value is hard-coded at
/// `y = 0` for the MVV. A follow-up release can add a
/// `floor_y: Fix128Gpu` uniform for configurable floor heights.
pub const FIX128_PGS_PROJECT_FLOOR_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

@group(0) @binding(0) var<storage, read_write> positions:  array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read_write> velocities: array<Fix128Gpu>;

@compute @workgroup_size(64)
fn fix128_pgs_project_floor_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }

    // The MSB of `hi_hi` is the sign bit of the Fix128 two's-complement
    // 128-bit integer. A set bit here means the value is < 0.
    let is_y_axis    = (idx % 3u) == 1u;
    let is_negative  = (positions[idx].hi_hi & 0x80000000u) != 0u;

    if (is_y_axis && is_negative) {
        positions[idx].hi_lo  = 0u;
        positions[idx].hi_hi  = 0u;
        positions[idx].lo_lo  = 0u;
        positions[idx].lo_hi  = 0u;
        velocities[idx].hi_lo = 0u;
        velocities[idx].hi_hi = 0u;
        velocities[idx].lo_lo = 0u;
        velocities[idx].lo_hi = 0u;
    }
}
"#;

/// WGSL compute shader source for **rigid rod distance constraint**
/// (v1.4.2) — projects two bodies to satisfy `|p_a - p_b| = rest_length`
/// with the scalar computed **entirely on the GPU**, eliminating the
/// per-iteration CPU round-trip that the v1.1.0 uniform-based
/// PGS distance shader required (removed in v2.0.0).
///
/// # Algorithm (single thread, single workgroup)
///
/// 1. Read `pa` (3 Fix128 components) and `pb` from the positions storage
///    buffer.
/// 2. `diff = pa - pb` (component-wise Fix128 sub).
/// 3. `d² = dx*dx + dy*dy + dz*dz` (3 muls + 2 adds).
/// 4. `d = sqrt(d²)` — 64-iter Newton, embedded from v1.4.1.
/// 5. If `d != 0` (uniform branch, single thread):
///    - `scalar = (rest_length - d) / (d + d)` — one div, one add, one sub.
///    - `positions[a*3 + i] += scalar * diff[i]` for `i ∈ {0,1,2}`.
///    - `positions[b*3 + i] -= scalar * diff[i]`.
///
/// The whole projection is **byte-for-byte equivalent** to the v1.1.0 +
/// CPU-precompute path used by v1.3.1 `dispatch_iterations`, because the
/// intermediate arithmetic uses the same v1.4.0 `fix128_div_kernel` and
/// v1.4.1 `fix128_sqrt` primitives that are already certified byte-exact
/// against `alice_physics::math::Fix128 / Fix128` and `Fix128::sqrt`.
///
/// # Determinism contract
///
/// - Single-thread `@compute @workgroup_size(1)`; no barrier, no subgroup
///   ops, no atomics.
/// - Fixed loop bounds (128 + 64 for div, 64 for sqrt) — no early exit.
/// - Uniform-flow discipline preserved: the `d != 0` guard is a single
///   uniform branch, safe under FXC X4026.
///
/// # Layout
///
/// - `@group(0) @binding(0)` positions: `array<Fix128Gpu>` — read-write.
/// - `@group(0) @binding(1)` params: uniform
///   `DistanceParamsRigid { body_a: u32, body_b: u32, _pad: vec2<u32>, rest_length: Fix128Gpu }`
///   (same 32-byte layout as the v1.1.0 `DistanceParams`, only the
///   semantic meaning of the second field changes — `rest_length` in
///   the rigid variant, `scalar` in the v1.1.0 variant).
pub const FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

struct DistanceParamsRigid {
    body_a: u32,
    body_b: u32,
    _pad:   vec2<u32>,
    rest_length: Fix128Gpu,
}

@group(0) @binding(0) var<storage, read_write> positions: array<Fix128Gpu>;
@group(0) @binding(1) var<uniform>              params:   DistanceParamsRigid;

// ---- 64-bit helpers ----

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

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let lo_sub  = a.x - b.x;
    let borrow1 = select(0u, 1u, a.x < b.x);
    let mid     = a.y - b.y;
    let borrow2 = select(0u, 1u, a.y < b.y);
    let hi_sub  = mid - borrow1;
    let borrow3 = select(0u, 1u, mid < borrow1);
    return vec3<u32>(lo_sub, hi_sub, borrow2 + borrow3);
}

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

// ---- Fix128 basic ops ----

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

fn fix128_sub_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let lo_res       = u64_sub(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let borrow_from_lo = lo_res.z;
    let hi_res       = u64_sub(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    let hi_with_borrow = u64_sub(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(borrow_from_lo, 0u));

    var out: Fix128Gpu;
    out.hi_lo = hi_with_borrow.x;
    out.hi_hi = hi_with_borrow.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

// Signed Fix128 mul — byte-exact copy of the certified kernel in
// FIX128_DOT_WGSL / FIX128_PGS_INTEGRATE_WGSL (unsigned schoolbook +
// two's-complement correction, matching CPU `Fix128 * Fix128`).
fn fix128_mul_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let a_lo = vec2<u32>(a.lo_lo, a.lo_hi);
    let a_hi = vec2<u32>(a.hi_lo, a.hi_hi);
    let b_lo = vec2<u32>(b.lo_lo, b.lo_hi);
    let b_hi = vec2<u32>(b.hi_lo, b.hi_hi);

    let ll = u64_mul_wide(a_lo, b_lo);
    let lh = u64_mul_wide(a_lo, b_hi);
    let hl = u64_mul_wide(a_hi, b_lo);
    let hh = u64_mul_wide(a_hi, b_hi);

    // Position 2..5 with full carry propagation.
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

// ---- Common Fix128 sign / zero helpers ----

fn fix128_neg(x: Fix128Gpu) -> Fix128Gpu {
    let inv_lo_lo = ~x.lo_lo;
    let inv_lo_hi = ~x.lo_hi;
    let inv_hi_lo = ~x.hi_lo;
    let inv_hi_hi = ~x.hi_hi;
    let s0 = inv_lo_lo + 1u;
    let c0 = select(0u, 1u, s0 < inv_lo_lo);
    let s1 = inv_lo_hi + c0;
    let c1 = select(0u, 1u, s1 < inv_lo_hi);
    let s2 = inv_hi_lo + c1;
    let c2 = select(0u, 1u, s2 < inv_hi_lo);
    let s3 = inv_hi_hi + c2;
    var out: Fix128Gpu;
    out.lo_lo = s0;
    out.lo_hi = s1;
    out.hi_lo = s2;
    out.hi_hi = s3;
    return out;
}

fn fix128_is_zero(x: Fix128Gpu) -> bool {
    return (x.hi_lo | x.hi_hi | x.lo_lo | x.lo_hi) == 0u;
}

fn fix128_is_negative(x: Fix128Gpu) -> bool {
    return ((x.hi_hi >> 31u) & 1u) == 1u;
}

fn fix128_abs(x: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_negative(x)) { return fix128_neg(x); }
    return x;
}

fn fix128_zero() -> Fix128Gpu {
    return Fix128Gpu(0u, 0u, 0u, 0u);
}

fn fix128_half(x: Fix128Gpu) -> Fix128Gpu {
    var out: Fix128Gpu;
    out.lo_lo = (x.lo_lo >> 1u) | ((x.lo_hi & 1u) << 31u);
    out.lo_hi = (x.lo_hi >> 1u) | ((x.hi_lo & 1u) << 31u);
    out.hi_lo = (x.hi_lo >> 1u) | ((x.hi_hi & 1u) << 31u);
    out.hi_hi = bitcast<u32>(bitcast<i32>(x.hi_hi) >> 1u);
    return out;
}

// ---- u128 helpers (for div) ----

fn u128_shl1(x: vec4<u32>) -> vec4<u32> {
    let c0 = x.x >> 31u;
    let c1 = x.y >> 31u;
    let c2 = x.z >> 31u;
    return vec4<u32>(x.x << 1u, (x.y << 1u) | c0, (x.z << 1u) | c1, (x.w << 1u) | c2);
}

fn u128_ge(a: vec4<u32>, b: vec4<u32>) -> bool {
    if (a.w != b.w) { return a.w > b.w; }
    if (a.z != b.z) { return a.z > b.z; }
    if (a.y != b.y) { return a.y > b.y; }
    return a.x >= b.x;
}

fn u128_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let d0        = a.x - b.x;
    let borrow0   = select(0u, 1u, a.x < b.x);
    let m1        = a.y - b.y;
    let borrow1a  = select(0u, 1u, a.y < b.y);
    let d1        = m1 - borrow0;
    let borrow1b  = select(0u, 1u, m1 < borrow0);
    let borrow1   = borrow1a + borrow1b;
    let m2        = a.z - b.z;
    let borrow2a  = select(0u, 1u, a.z < b.z);
    let d2        = m2 - borrow1;
    let borrow2b  = select(0u, 1u, m2 < borrow1);
    let borrow2   = borrow2a + borrow2b;
    let m3        = a.w - b.w;
    let d3        = m3 - borrow2;
    return vec4<u32>(d0, d1, d2, d3);
}

fn u128_set_bit(x: vec4<u32>, bit_pos: u32) -> vec4<u32> {
    let word  = bit_pos >> 5u;
    let shift = bit_pos & 31u;
    let mask  = 1u << shift;
    var out = x;
    if      (word == 0u) { out.x = out.x | mask; }
    else if (word == 1u) { out.y = out.y | mask; }
    else if (word == 2u) { out.z = out.z | mask; }
    else                 { out.w = out.w | mask; }
    return out;
}

// ---- div + sqrt (from v1.4.0 / v1.4.1, byte-exact against alice_physics) ----

fn fix128_div_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_zero(b)) { return fix128_zero(); }

    let neg_a = fix128_is_negative(a);
    let neg_b = fix128_is_negative(b);
    let result_neg = neg_a != neg_b;

    let abs_a = fix128_abs(a);
    let abs_b = fix128_abs(b);

    let a_full = vec4<u32>(abs_a.lo_lo, abs_a.lo_hi, abs_a.hi_lo, abs_a.hi_hi);
    let b_full = vec4<u32>(abs_b.lo_lo, abs_b.lo_hi, abs_b.hi_lo, abs_b.hi_hi);

    var q_int = vec4<u32>(0u);
    var r_int = vec4<u32>(0u);
    var d     = a_full;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let msb = d.w >> 31u;
        r_int   = u128_shl1(r_int);
        r_int.x = r_int.x | msb;
        d       = u128_shl1(d);
        if (u128_ge(r_int, b_full)) {
            r_int = u128_sub(r_int, b_full);
            q_int = u128_set_bit(q_int, u32(127 - i));
        }
    }

    var quot_lo_lo: u32 = 0u;
    var quot_lo_hi: u32 = 0u;
    var r = r_int;
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let overflow_bit = r.w >> 31u;
        r = u128_shl1(r);
        if (overflow_bit != 0u || u128_ge(r, b_full)) {
            r = u128_sub(r, b_full);
            if (i < 32) {
                quot_lo_lo = quot_lo_lo | (1u << u32(i));
            } else {
                quot_lo_hi = quot_lo_hi | (1u << u32(i - 32));
            }
        }
    }

    var result: Fix128Gpu;
    result.hi_lo = q_int.x;
    result.hi_hi = q_int.y;
    result.lo_lo = quot_lo_lo;
    result.lo_hi = quot_lo_hi;

    if (result_neg) { return fix128_neg(result); }
    return result;
}

fn fix128_sqrt(a: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_negative(a) || fix128_is_zero(a)) { return fix128_zero(); }

    var sig_bits: u32;
    if (a.hi_hi != 0u || a.hi_lo != 0u) {
        var lz: u32;
        if (a.hi_hi != 0u) { lz = countLeadingZeros(a.hi_hi); }
        else               { lz = 32u + countLeadingZeros(a.hi_lo); }
        sig_bits = 128u - lz;
    } else {
        var lz: u32;
        if (a.lo_hi != 0u) { lz = countLeadingZeros(a.lo_hi); }
        else               { lz = 32u + countLeadingZeros(a.lo_lo); }
        sig_bits = 64u - lz;
    }
    let result_bit = (sig_bits + 63u) / 2u;

    var x: Fix128Gpu;
    x.hi_lo = 0u;
    x.hi_hi = 0u;
    x.lo_lo = 0u;
    x.lo_hi = 0u;
    if (result_bit >= 64u) {
        let shift_raw = result_bit - 64u;
        let shift = min(shift_raw, 62u);
        if (shift < 32u) { x.hi_lo = 1u << shift; }
        else             { x.hi_hi = 1u << (shift - 32u); }
    } else {
        if (result_bit < 32u) { x.lo_lo = 1u << result_bit; }
        else                  { x.lo_hi = 1u << (result_bit - 32u); }
    }

    for (var i: i32 = 0; i < 64; i = i + 1) {
        let div = fix128_div_kernel(a, x);
        let sum = fix128_add_kernel(x, div);
        x = fix128_half(sum);
    }
    return x;
}

// ---- Main: rigid rod projection ----

@compute @workgroup_size(1)
fn fix128_pgs_project_distance_rigid_main() {
    let a = params.body_a;
    let b = params.body_b;
    let rest = params.rest_length;

    // Read positions (3 axes each).
    let pa0 = positions[a * 3u + 0u];
    let pa1 = positions[a * 3u + 1u];
    let pa2 = positions[a * 3u + 2u];
    let pb0 = positions[b * 3u + 0u];
    let pb1 = positions[b * 3u + 1u];
    let pb2 = positions[b * 3u + 2u];

    // diff = pa - pb.
    let dx = fix128_sub_kernel(pa0, pb0);
    let dy = fix128_sub_kernel(pa1, pb1);
    let dz = fix128_sub_kernel(pa2, pb2);

    // d_sq = dx*dx + dy*dy + dz*dz.
    let dx_sq = fix128_mul_kernel(dx, dx);
    let dy_sq = fix128_mul_kernel(dy, dy);
    let dz_sq = fix128_mul_kernel(dz, dz);
    let d_sq  = fix128_add_kernel(fix128_add_kernel(dx_sq, dy_sq), dz_sq);

    // d = sqrt(d_sq).
    let d = fix128_sqrt(d_sq);

    // If d is zero, skip projection (colocated bodies — direction undefined).
    if (!fix128_is_zero(d)) {
        // scalar = (rest - d) / (d + d).
        let numerator   = fix128_sub_kernel(rest, d);
        let denominator = fix128_add_kernel(d, d);
        let scalar      = fix128_div_kernel(numerator, denominator);

        // Apply: positions[a] += scalar * diff, positions[b] -= scalar * diff.
        let delta_x = fix128_mul_kernel(scalar, dx);
        let delta_y = fix128_mul_kernel(scalar, dy);
        let delta_z = fix128_mul_kernel(scalar, dz);

        positions[a * 3u + 0u] = fix128_add_kernel(pa0, delta_x);
        positions[a * 3u + 1u] = fix128_add_kernel(pa1, delta_y);
        positions[a * 3u + 2u] = fix128_add_kernel(pa2, delta_z);
        positions[b * 3u + 0u] = fix128_sub_kernel(pb0, delta_x);
        positions[b * 3u + 1u] = fix128_sub_kernel(pb1, delta_y);
        positions[b * 3u + 2u] = fix128_sub_kernel(pb2, delta_z);
    }
}
"#;

/// WGSL compute shader source for the **batched rigid rod distance
/// constraint** (v1.5.1). Same body as
/// [`FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL`] except the params bind
/// point is a **storage buffer array** and each workgroup selects its
/// constraint from `@builtin(workgroup_id).x`.
///
/// This is Phase 2's one-dispatch-per-color kernel: given a color of
/// K body-disjoint constraints, the caller dispatches
/// `dispatch_workgroups(K, 1, 1)` and every workgroup applies its own
/// constraint concurrently. Correctness follows from the coloring
/// invariant (no two constraints in the same color share a body, so
/// there is no race on `positions`).
///
/// # Layout
///
/// - `@group(0) @binding(0)` positions: `array<Fix128Gpu>` — read-write.
/// - `@group(0) @binding(1)` params: `array<DistanceParamsRigid>` —
///   **storage, read-only** (was uniform in the v1.4.2 single-constraint
///   variant). One element per constraint in the color; workgroup `w`
///   reads `params[w]`.
///
/// # Determinism
///
/// The workgroup-parallel dispatch is safe by construction: constraints
/// in a color operate on disjoint body sets, so the memory writes never
/// alias. Every element of `params` is applied independently, and the
/// per-workgroup arithmetic is identical to
/// [`FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL`] (which is already
/// byte-exact vs the CPU golden). No workgroupBarrier, no subgroup ops,
/// no atomics — cross-platform equivalence follows from the shared v1.4.0
/// div + v1.4.1 sqrt contract.
pub const FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

struct DistanceParamsRigid {
    body_a: u32,
    body_b: u32,
    _pad:   vec2<u32>,
    rest_length: Fix128Gpu,
}

@group(0) @binding(0) var<storage, read_write> positions: array<Fix128Gpu>;
@group(0) @binding(1) var<storage, read>       params:    array<DistanceParamsRigid>;

// ---- Same helpers as FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL ----

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

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let lo_sub  = a.x - b.x;
    let borrow1 = select(0u, 1u, a.x < b.x);
    let mid     = a.y - b.y;
    let borrow2 = select(0u, 1u, a.y < b.y);
    let hi_sub  = mid - borrow1;
    let borrow3 = select(0u, 1u, mid < borrow1);
    return vec3<u32>(lo_sub, hi_sub, borrow2 + borrow3);
}

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

fn fix128_sub_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let lo_res         = u64_sub(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let borrow_from_lo = lo_res.z;
    let hi_res         = u64_sub(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    let hi_with_borrow = u64_sub(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(borrow_from_lo, 0u));

    var out: Fix128Gpu;
    out.hi_lo = hi_with_borrow.x;
    out.hi_hi = hi_with_borrow.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

// Byte-exact copy of the certified fix128_mul_kernel (Karatsuba + two's-comp).
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

fn fix128_neg(x: Fix128Gpu) -> Fix128Gpu {
    let inv_lo_lo = ~x.lo_lo;
    let inv_lo_hi = ~x.lo_hi;
    let inv_hi_lo = ~x.hi_lo;
    let inv_hi_hi = ~x.hi_hi;
    let s0 = inv_lo_lo + 1u;
    let c0 = select(0u, 1u, s0 < inv_lo_lo);
    let s1 = inv_lo_hi + c0;
    let c1 = select(0u, 1u, s1 < inv_lo_hi);
    let s2 = inv_hi_lo + c1;
    let c2 = select(0u, 1u, s2 < inv_hi_lo);
    let s3 = inv_hi_hi + c2;
    var out: Fix128Gpu;
    out.lo_lo = s0;
    out.lo_hi = s1;
    out.hi_lo = s2;
    out.hi_hi = s3;
    return out;
}

fn fix128_is_zero(x: Fix128Gpu) -> bool {
    return (x.hi_lo | x.hi_hi | x.lo_lo | x.lo_hi) == 0u;
}

fn fix128_is_negative(x: Fix128Gpu) -> bool {
    return ((x.hi_hi >> 31u) & 1u) == 1u;
}

fn fix128_abs(x: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_negative(x)) { return fix128_neg(x); }
    return x;
}

fn fix128_zero() -> Fix128Gpu {
    return Fix128Gpu(0u, 0u, 0u, 0u);
}

fn fix128_half(x: Fix128Gpu) -> Fix128Gpu {
    var out: Fix128Gpu;
    out.lo_lo = (x.lo_lo >> 1u) | ((x.lo_hi & 1u) << 31u);
    out.lo_hi = (x.lo_hi >> 1u) | ((x.hi_lo & 1u) << 31u);
    out.hi_lo = (x.hi_lo >> 1u) | ((x.hi_hi & 1u) << 31u);
    out.hi_hi = bitcast<u32>(bitcast<i32>(x.hi_hi) >> 1u);
    return out;
}

fn u128_shl1(x: vec4<u32>) -> vec4<u32> {
    let c0 = x.x >> 31u;
    let c1 = x.y >> 31u;
    let c2 = x.z >> 31u;
    return vec4<u32>(x.x << 1u, (x.y << 1u) | c0, (x.z << 1u) | c1, (x.w << 1u) | c2);
}

fn u128_ge(a: vec4<u32>, b: vec4<u32>) -> bool {
    if (a.w != b.w) { return a.w > b.w; }
    if (a.z != b.z) { return a.z > b.z; }
    if (a.y != b.y) { return a.y > b.y; }
    return a.x >= b.x;
}

fn u128_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let d0        = a.x - b.x;
    let borrow0   = select(0u, 1u, a.x < b.x);
    let m1        = a.y - b.y;
    let borrow1a  = select(0u, 1u, a.y < b.y);
    let d1        = m1 - borrow0;
    let borrow1b  = select(0u, 1u, m1 < borrow0);
    let borrow1   = borrow1a + borrow1b;
    let m2        = a.z - b.z;
    let borrow2a  = select(0u, 1u, a.z < b.z);
    let d2        = m2 - borrow1;
    let borrow2b  = select(0u, 1u, m2 < borrow1);
    let borrow2   = borrow2a + borrow2b;
    let m3        = a.w - b.w;
    let d3        = m3 - borrow2;
    return vec4<u32>(d0, d1, d2, d3);
}

fn u128_set_bit(x: vec4<u32>, bit_pos: u32) -> vec4<u32> {
    let word  = bit_pos >> 5u;
    let shift = bit_pos & 31u;
    let mask  = 1u << shift;
    var out = x;
    if      (word == 0u) { out.x = out.x | mask; }
    else if (word == 1u) { out.y = out.y | mask; }
    else if (word == 2u) { out.z = out.z | mask; }
    else                 { out.w = out.w | mask; }
    return out;
}

fn fix128_div_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_zero(b)) { return fix128_zero(); }

    let neg_a = fix128_is_negative(a);
    let neg_b = fix128_is_negative(b);
    let result_neg = neg_a != neg_b;

    let abs_a = fix128_abs(a);
    let abs_b = fix128_abs(b);

    let a_full = vec4<u32>(abs_a.lo_lo, abs_a.lo_hi, abs_a.hi_lo, abs_a.hi_hi);
    let b_full = vec4<u32>(abs_b.lo_lo, abs_b.lo_hi, abs_b.hi_lo, abs_b.hi_hi);

    var q_int = vec4<u32>(0u);
    var r_int = vec4<u32>(0u);
    var d     = a_full;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let msb = d.w >> 31u;
        r_int   = u128_shl1(r_int);
        r_int.x = r_int.x | msb;
        d       = u128_shl1(d);
        if (u128_ge(r_int, b_full)) {
            r_int = u128_sub(r_int, b_full);
            q_int = u128_set_bit(q_int, u32(127 - i));
        }
    }

    var quot_lo_lo: u32 = 0u;
    var quot_lo_hi: u32 = 0u;
    var r = r_int;
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let overflow_bit = r.w >> 31u;
        r = u128_shl1(r);
        if (overflow_bit != 0u || u128_ge(r, b_full)) {
            r = u128_sub(r, b_full);
            if (i < 32) {
                quot_lo_lo = quot_lo_lo | (1u << u32(i));
            } else {
                quot_lo_hi = quot_lo_hi | (1u << u32(i - 32));
            }
        }
    }

    var result: Fix128Gpu;
    result.hi_lo = q_int.x;
    result.hi_hi = q_int.y;
    result.lo_lo = quot_lo_lo;
    result.lo_hi = quot_lo_hi;

    if (result_neg) { return fix128_neg(result); }
    return result;
}

fn fix128_sqrt(a: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_negative(a) || fix128_is_zero(a)) { return fix128_zero(); }

    var sig_bits: u32;
    if (a.hi_hi != 0u || a.hi_lo != 0u) {
        var lz: u32;
        if (a.hi_hi != 0u) { lz = countLeadingZeros(a.hi_hi); }
        else               { lz = 32u + countLeadingZeros(a.hi_lo); }
        sig_bits = 128u - lz;
    } else {
        var lz: u32;
        if (a.lo_hi != 0u) { lz = countLeadingZeros(a.lo_hi); }
        else               { lz = 32u + countLeadingZeros(a.lo_lo); }
        sig_bits = 64u - lz;
    }
    let result_bit = (sig_bits + 63u) / 2u;

    var x: Fix128Gpu;
    x.hi_lo = 0u;
    x.hi_hi = 0u;
    x.lo_lo = 0u;
    x.lo_hi = 0u;
    if (result_bit >= 64u) {
        let shift_raw = result_bit - 64u;
        let shift = min(shift_raw, 62u);
        if (shift < 32u) { x.hi_lo = 1u << shift; }
        else             { x.hi_hi = 1u << (shift - 32u); }
    } else {
        if (result_bit < 32u) { x.lo_lo = 1u << result_bit; }
        else                  { x.lo_hi = 1u << (result_bit - 32u); }
    }

    for (var i: i32 = 0; i < 64; i = i + 1) {
        let div = fix128_div_kernel(a, x);
        let sum = fix128_add_kernel(x, div);
        x = fix128_half(sum);
    }
    return x;
}

// ---- Main: batched rigid rod projection (one workgroup per constraint) ----

@compute @workgroup_size(1)
fn fix128_pgs_project_distance_batched_main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = wg_id.x;
    if (idx >= arrayLength(&params)) { return; }

    let p = params[idx];
    let a = p.body_a;
    let b = p.body_b;
    let rest = p.rest_length;

    let pa0 = positions[a * 3u + 0u];
    let pa1 = positions[a * 3u + 1u];
    let pa2 = positions[a * 3u + 2u];
    let pb0 = positions[b * 3u + 0u];
    let pb1 = positions[b * 3u + 1u];
    let pb2 = positions[b * 3u + 2u];

    let dx = fix128_sub_kernel(pa0, pb0);
    let dy = fix128_sub_kernel(pa1, pb1);
    let dz = fix128_sub_kernel(pa2, pb2);

    let dx_sq = fix128_mul_kernel(dx, dx);
    let dy_sq = fix128_mul_kernel(dy, dy);
    let dz_sq = fix128_mul_kernel(dz, dz);
    let d_sq  = fix128_add_kernel(fix128_add_kernel(dx_sq, dy_sq), dz_sq);
    let d = fix128_sqrt(d_sq);

    if (!fix128_is_zero(d)) {
        let numerator   = fix128_sub_kernel(rest, d);
        let denominator = fix128_add_kernel(d, d);
        let scalar      = fix128_div_kernel(numerator, denominator);

        let delta_x = fix128_mul_kernel(scalar, dx);
        let delta_y = fix128_mul_kernel(scalar, dy);
        let delta_z = fix128_mul_kernel(scalar, dz);

        positions[a * 3u + 0u] = fix128_add_kernel(pa0, delta_x);
        positions[a * 3u + 1u] = fix128_add_kernel(pa1, delta_y);
        positions[a * 3u + 2u] = fix128_add_kernel(pa2, delta_z);
        positions[b * 3u + 0u] = fix128_sub_kernel(pb0, delta_x);
        positions[b * 3u + 1u] = fix128_sub_kernel(pb1, delta_y);
        positions[b * 3u + 2u] = fix128_sub_kernel(pb2, delta_z);
    }
}
"#;

/// WGSL compute shader source for **Fix128 AABB helpers** (v2.1.0
/// Phase 3 first primitive) — declares [`Fix128AabbGpu`] and its
/// shader-side helpers (`aabb_from_sphere`, `aabb_union`) on top of
/// the byte-exact `fix128_add / sub / lt / min / max` primitives.
///
/// This constant is a **standalone valid WGSL module**: it ships with a
/// no-op `aabb_helpers_smoke_main` compute entry so `create_shader_module`
/// succeeds without any binding. External consumers who need the
/// helpers in their own kernel typically concatenate this source with
/// their kernel body — WGSL has no `include` directive, so cat-ing
/// strings is the common pattern (mirroring the v1.4.2 rigid rod
/// shader's embedded copy of the v1.4.0 div and v1.4.1 sqrt kernels).
///
/// # Determinism contract
///
/// Every helper reduces to the byte-exact `fix128_add / sub / lt`
/// primitives; no cross-lane state, no comparison shortcuts on
/// negative-zero. Consumers that use `aabb_union` on a colour-parallel
/// dispatch must still enforce workgroup traversal order at the CPU
/// side (see [determinism-lockstep §1 経路 5][skill]).
///
/// [skill]: https://github.com/ext-sakamoro/claude-config/blob/main/claude-skills/deterministic-physics-lockstep-discipline/SKILL.md
///
/// # Layout
///
/// `Fix128AabbGpu` packs `(min.x, min.y, min.z, max.x, max.y, max.z)`
/// as six `Fix128Gpu` fields = 6 × 16 bytes = **96 bytes**. The Rust
/// mirror [`Fix128AabbGpu`] uses the same field order under `#[repr(C)]`.
///
/// # Phase 3 role
///
/// v2.1.0 ships this helpers module plus the paired
/// [`FIX128_MORTON_CODE_WGSL`] kernel. Subsequent v2.2+ kernels
/// (Morton sort, BVH build, find_pairs, narrow-phase, PGS contact)
/// consume [`Fix128AabbGpu`] as their primitive AABB representation.
pub const FIX128_AABB_HELPERS_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

struct Fix128AabbGpu {
    min_x: Fix128Gpu,
    min_y: Fix128Gpu,
    min_z: Fix128Gpu,
    max_x: Fix128Gpu,
    max_y: Fix128Gpu,
    max_z: Fix128Gpu,
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let lo_sub  = a.x - b.x;
    let borrow1 = select(0u, 1u, a.x < b.x);
    let mid_sub = a.y - b.y;
    let borrow2 = select(0u, 1u, a.y < b.y);
    let hi_sub  = mid_sub - borrow1;
    let borrow3 = select(0u, 1u, mid_sub < borrow1);
    return vec3<u32>(lo_sub, hi_sub, borrow2 + borrow3);
}

fn fix128_add(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
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

fn fix128_sub(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let lo_res         = u64_sub(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let borrow_from_lo = lo_res.z;
    let hi_res         = u64_sub(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    let hi_with_borrow = u64_sub(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(borrow_from_lo, 0u));
    var out: Fix128Gpu;
    out.hi_lo = hi_with_borrow.x;
    out.hi_hi = hi_with_borrow.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

// Signed less-than compare (mirrors `Fix128 as i128` PartialOrd on CPU).
fn fix128_lt(a: Fix128Gpu, b: Fix128Gpu) -> bool {
    let a_sign = (a.hi_hi & 0x80000000u) != 0u;
    let b_sign = (b.hi_hi & 0x80000000u) != 0u;
    if (a_sign != b_sign) {
        return a_sign;
    }
    if (a.hi_hi != b.hi_hi) { return a.hi_hi < b.hi_hi; }
    if (a.hi_lo != b.hi_lo) { return a.hi_lo < b.hi_lo; }
    if (a.lo_hi != b.lo_hi) { return a.lo_hi < b.lo_hi; }
    return a.lo_lo < b.lo_lo;
}

fn fix128_min(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    if (fix128_lt(a, b)) { return a; }
    return b;
}

fn fix128_max(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    if (fix128_lt(a, b)) { return b; }
    return a;
}

// Build an AABB centred on `pos` with half-extents equal to `radius` on
// every axis. Mirrors the CPU `AABB::from_center_half` used by
// `ALICE-Physics::solver::detect_collisions` when populating BVH input.
fn aabb_from_sphere(
    pos_x: Fix128Gpu,
    pos_y: Fix128Gpu,
    pos_z: Fix128Gpu,
    radius: Fix128Gpu,
) -> Fix128AabbGpu {
    var out: Fix128AabbGpu;
    out.min_x = fix128_sub(pos_x, radius);
    out.min_y = fix128_sub(pos_y, radius);
    out.min_z = fix128_sub(pos_z, radius);
    out.max_x = fix128_add(pos_x, radius);
    out.max_y = fix128_add(pos_y, radius);
    out.max_z = fix128_add(pos_z, radius);
    return out;
}

// Element-wise union of two AABBs (componentwise min on the mins,
// componentwise max on the maxes). Mirrors CPU `AABB::union`.
fn aabb_union(a: Fix128AabbGpu, b: Fix128AabbGpu) -> Fix128AabbGpu {
    var out: Fix128AabbGpu;
    out.min_x = fix128_min(a.min_x, b.min_x);
    out.min_y = fix128_min(a.min_y, b.min_y);
    out.min_z = fix128_min(a.min_z, b.min_z);
    out.max_x = fix128_max(a.max_x, b.max_x);
    out.max_y = fix128_max(a.max_y, b.max_y);
    out.max_z = fix128_max(a.max_z, b.max_z);
    return out;
}

// No-op smoke entry so this module is a standalone valid compute
// shader. External consumers who cat this source into their own
// kernel typically remove this entry point.
@compute @workgroup_size(1)
fn aabb_helpers_smoke_main() {
}
"#;

/// GPU-friendly axis-aligned bounding box in Fix128 space.
///
/// Mirrors `alice_physics::collider::AABB` byte-for-byte (each axis of
/// each corner is one [`Fix128Gpu`] = 16 bytes; total = 96 bytes). The
/// field order matches the WGSL [`Fix128AabbGpu`](FIX128_AABB_HELPERS_WGSL)
/// struct so a `Vec<Fix128AabbGpu>` can be uploaded as a storage
/// buffer without transformation.
///
/// # Phase 3
///
/// This is the input type for the v2.1.0 Morton code kernel and the
/// primitive AABB representation for the whole v2.2+ Phase 3 pipeline
/// (Morton sort → BVH build → find_pairs → sphere-sphere narrow-phase).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Fix128AabbGpu {
    /// Minimum X in Fix128 space.
    pub min_x: Fix128Gpu,
    /// Minimum Y in Fix128 space.
    pub min_y: Fix128Gpu,
    /// Minimum Z in Fix128 space.
    pub min_z: Fix128Gpu,
    /// Maximum X in Fix128 space.
    pub max_x: Fix128Gpu,
    /// Maximum Y in Fix128 space.
    pub max_y: Fix128Gpu,
    /// Maximum Z in Fix128 space.
    pub max_z: Fix128Gpu,
}

impl Fix128AabbGpu {
    /// Construct an AABB centred on `(pos_x, pos_y, pos_z)` with the
    /// given `radius` as half-extent on every axis. Mirrors the WGSL
    /// `aabb_from_sphere` helper byte-for-byte.
    #[must_use]
    pub fn from_sphere(
        pos_x: Fix128Gpu,
        pos_y: Fix128Gpu,
        pos_z: Fix128Gpu,
        radius: Fix128Gpu,
    ) -> Self {
        Self {
            min_x: pos_x.sub(radius),
            min_y: pos_y.sub(radius),
            min_z: pos_z.sub(radius),
            max_x: pos_x.add(radius),
            max_y: pos_y.add(radius),
            max_z: pos_z.add(radius),
        }
    }

    /// Element-wise union with `other`. Mirrors the WGSL `aabb_union`
    /// helper byte-for-byte.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min_x: Self::fix128_min(self.min_x, other.min_x),
            min_y: Self::fix128_min(self.min_y, other.min_y),
            min_z: Self::fix128_min(self.min_z, other.min_z),
            max_x: Self::fix128_max(self.max_x, other.max_x),
            max_y: Self::fix128_max(self.max_y, other.max_y),
            max_z: Self::fix128_max(self.max_z, other.max_z),
        }
    }

    /// Signed 128-bit less-than compare. Mirrors the WGSL `fix128_lt`
    /// helper. The `(hi, lo)` tuple compare works because `hi: i64`
    /// carries the sign and `lo: u64` is the low unsigned word; Rust
    /// tuple ordering does lexicographic compare with signed `hi`
    /// first and unsigned `lo` as tiebreaker.
    #[inline]
    #[must_use]
    fn lt(a: Fix128Gpu, b: Fix128Gpu) -> bool {
        (a.hi, a.lo) < (b.hi, b.lo)
    }

    #[inline]
    fn fix128_min(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
        if Self::lt(a, b) {
            a
        } else {
            b
        }
    }

    #[inline]
    fn fix128_max(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
        if Self::lt(a, b) {
            b
        } else {
            a
        }
    }
}

/// WGSL compute shader source for **Fix128 63-bit Morton code**
/// (v2.1.0 Phase 3 first primitive kernel).
///
/// Per-primitive parallel kernel that mirrors
/// `alice_physics::bvh::point_to_morton` byte-for-byte on the GPU:
///
/// 1. Compute the AABB centre as `(min + max) / 2` on each axis (Fix128
///    add + arithmetic right shift, both certified in v1.4.x).
/// 2. Compute world size `world_bounds.max - world_bounds.min` per axis.
/// 3. Normalize the centre to `[0, 1]` via Fix128 divide (v1.4.0
///    `FIX128_DIV_WGSL` embedded verbatim), then extract the upper 21
///    bits of the fractional part as the per-axis coordinate.
/// 4. Spread each 21-bit coordinate to 63 bits via the standard
///    `expand_bits` u64 bit-manipulation sequence and combine via
///    `expand(x) | (expand(y) << 1) | (expand(z) << 2)` into a full
///    63-bit Morton code.
///
/// # Bindings
///
/// - `@group(0) @binding(0) var<storage, read>       primitives:   array<Fix128AabbGpu>` — input AABBs
/// - `@group(0) @binding(1) var<uniform>             world_bounds: Fix128AabbGpu`       — world-space AABB
/// - `@group(0) @binding(2) var<storage, read_write> morton_codes: array<vec2<u32>>`    — output 63-bit codes as `(low32, high32)`
///
/// # Determinism contract
///
/// Every arithmetic step reduces to an already-certified Fix128
/// primitive (`add / sub / mul / div` from v1.4.0). The `expand_bits`
/// implementation uses only bitwise operations on `vec2<u32>` u64
/// emulation; no `atomicAdd`, no `workgroupBarrier`, no cross-lane
/// state. Per-thread output depends only on its own primitive index,
/// so parallel dispatch order across threads is irrelevant.
///
/// The 21-bit coordinate extraction mirrors the three CPU branches
/// exactly: `t < 0` → `0`, `t.hi >= 1` → `0x1FFFFF`, else
/// `(t.lo >> 43) & 0x1FFFFF`. The GPU version uses
/// `t.hi_hi != 0 || t.hi_lo >= 1` for the second branch (equivalent to
/// checking whether the integer part of the fixed-point value is
/// `>= 1`) and `t.lo_hi >> 11` for the third branch (equivalent to
/// `t.lo >> 43` because `t.lo_hi` is bits 32-63 of the low u64 half).
///
/// # Phase 3 role
///
/// This is the first kernel of the v2.2+ GPU BVH pipeline. It ships
/// alongside [`FIX128_AABB_HELPERS_WGSL`] (v2.1.0) and feeds the v2.2+
/// Morton sort → BVH build → find_pairs sequence documented in
/// [`docs/PHASE_3_DESIGN.md`](../../docs/PHASE_3_DESIGN.md).
pub const FIX128_MORTON_CODE_WGSL: &str = r#"
struct Fix128Gpu {
    hi_lo: u32,
    hi_hi: u32,
    lo_lo: u32,
    lo_hi: u32,
}

struct Fix128AabbGpu {
    min_x: Fix128Gpu,
    min_y: Fix128Gpu,
    min_z: Fix128Gpu,
    max_x: Fix128Gpu,
    max_y: Fix128Gpu,
    max_z: Fix128Gpu,
}

@group(0) @binding(0) var<storage, read>       primitives:   array<Fix128AabbGpu>;
@group(0) @binding(1) var<uniform>             world_bounds: Fix128AabbGpu;
@group(0) @binding(2) var<storage, read_write> morton_codes: array<vec2<u32>>;

// ---- 64-bit helpers ----

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

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let sum_lo = a.x + b.x;
    let carry1 = select(0u, 1u, sum_lo < a.x);
    let mid    = a.y + b.y;
    let carry2 = select(0u, 1u, mid < a.y);
    let sum_hi = mid + carry1;
    let carry3 = select(0u, 1u, sum_hi < mid);
    return vec3<u32>(sum_lo, sum_hi, carry2 + carry3);
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let lo_sub  = a.x - b.x;
    let borrow1 = select(0u, 1u, a.x < b.x);
    let mid     = a.y - b.y;
    let borrow2 = select(0u, 1u, a.y < b.y);
    let hi_sub  = mid - borrow1;
    let borrow3 = select(0u, 1u, mid < borrow1);
    return vec3<u32>(lo_sub, hi_sub, borrow2 + borrow3);
}

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

// ---- Fix128 basic ops (byte-exact copies from v1.4.x kernels) ----

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

fn fix128_sub_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    let lo_res         = u64_sub(vec2<u32>(a.lo_lo, a.lo_hi), vec2<u32>(b.lo_lo, b.lo_hi));
    let borrow_from_lo = lo_res.z;
    let hi_res         = u64_sub(vec2<u32>(a.hi_lo, a.hi_hi), vec2<u32>(b.hi_lo, b.hi_hi));
    let hi_with_borrow = u64_sub(vec2<u32>(hi_res.x, hi_res.y), vec2<u32>(borrow_from_lo, 0u));
    var out: Fix128Gpu;
    out.hi_lo = hi_with_borrow.x;
    out.hi_hi = hi_with_borrow.y;
    out.lo_lo = lo_res.x;
    out.lo_hi = lo_res.y;
    return out;
}

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

// ---- Common Fix128 sign / zero helpers ----

fn fix128_neg(x: Fix128Gpu) -> Fix128Gpu {
    let inv_lo_lo = ~x.lo_lo;
    let inv_lo_hi = ~x.lo_hi;
    let inv_hi_lo = ~x.hi_lo;
    let inv_hi_hi = ~x.hi_hi;
    let s0 = inv_lo_lo + 1u;
    let c0 = select(0u, 1u, s0 < inv_lo_lo);
    let s1 = inv_lo_hi + c0;
    let c1 = select(0u, 1u, s1 < inv_lo_hi);
    let s2 = inv_hi_lo + c1;
    let c2 = select(0u, 1u, s2 < inv_hi_lo);
    let s3 = inv_hi_hi + c2;
    var out: Fix128Gpu;
    out.lo_lo = s0;
    out.lo_hi = s1;
    out.hi_lo = s2;
    out.hi_hi = s3;
    return out;
}

fn fix128_is_zero(x: Fix128Gpu) -> bool {
    return (x.hi_lo | x.hi_hi | x.lo_lo | x.lo_hi) == 0u;
}

fn fix128_is_negative(x: Fix128Gpu) -> bool {
    return ((x.hi_hi >> 31u) & 1u) == 1u;
}

fn fix128_abs(x: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_negative(x)) { return fix128_neg(x); }
    return x;
}

fn fix128_zero() -> Fix128Gpu {
    return Fix128Gpu(0u, 0u, 0u, 0u);
}

fn fix128_half(x: Fix128Gpu) -> Fix128Gpu {
    var out: Fix128Gpu;
    out.lo_lo = (x.lo_lo >> 1u) | ((x.lo_hi & 1u) << 31u);
    out.lo_hi = (x.lo_hi >> 1u) | ((x.hi_lo & 1u) << 31u);
    out.hi_lo = (x.hi_lo >> 1u) | ((x.hi_hi & 1u) << 31u);
    out.hi_hi = bitcast<u32>(bitcast<i32>(x.hi_hi) >> 1u);
    return out;
}

// ---- u128 helpers (for div) ----

fn u128_shl1(x: vec4<u32>) -> vec4<u32> {
    let c0 = x.x >> 31u;
    let c1 = x.y >> 31u;
    let c2 = x.z >> 31u;
    return vec4<u32>(x.x << 1u, (x.y << 1u) | c0, (x.z << 1u) | c1, (x.w << 1u) | c2);
}

fn u128_ge(a: vec4<u32>, b: vec4<u32>) -> bool {
    if (a.w != b.w) { return a.w > b.w; }
    if (a.z != b.z) { return a.z > b.z; }
    if (a.y != b.y) { return a.y > b.y; }
    return a.x >= b.x;
}

fn u128_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let d0        = a.x - b.x;
    let borrow0   = select(0u, 1u, a.x < b.x);
    let m1        = a.y - b.y;
    let borrow1a  = select(0u, 1u, a.y < b.y);
    let d1        = m1 - borrow0;
    let borrow1b  = select(0u, 1u, m1 < borrow0);
    let borrow1   = borrow1a + borrow1b;
    let m2        = a.z - b.z;
    let borrow2a  = select(0u, 1u, a.z < b.z);
    let d2        = m2 - borrow1;
    let borrow2b  = select(0u, 1u, m2 < borrow1);
    let borrow2   = borrow2a + borrow2b;
    let m3        = a.w - b.w;
    let d3        = m3 - borrow2;
    return vec4<u32>(d0, d1, d2, d3);
}

fn u128_set_bit(x: vec4<u32>, bit_pos: u32) -> vec4<u32> {
    let word  = bit_pos >> 5u;
    let shift = bit_pos & 31u;
    let mask  = 1u << shift;
    var out = x;
    if      (word == 0u) { out.x = out.x | mask; }
    else if (word == 1u) { out.y = out.y | mask; }
    else if (word == 2u) { out.z = out.z | mask; }
    else                 { out.w = out.w | mask; }
    return out;
}

// ---- div (from v1.4.0, byte-exact against alice_physics::Fix128 / Fix128) ----

fn fix128_div_kernel(a: Fix128Gpu, b: Fix128Gpu) -> Fix128Gpu {
    if (fix128_is_zero(b)) { return fix128_zero(); }
    let neg_a = fix128_is_negative(a);
    let neg_b = fix128_is_negative(b);
    let result_neg = neg_a != neg_b;
    let abs_a = fix128_abs(a);
    let abs_b = fix128_abs(b);
    let a_full = vec4<u32>(abs_a.lo_lo, abs_a.lo_hi, abs_a.hi_lo, abs_a.hi_hi);
    let b_full = vec4<u32>(abs_b.lo_lo, abs_b.lo_hi, abs_b.hi_lo, abs_b.hi_hi);
    var q_int = vec4<u32>(0u);
    var r_int = vec4<u32>(0u);
    var d     = a_full;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let msb = d.w >> 31u;
        r_int   = u128_shl1(r_int);
        r_int.x = r_int.x | msb;
        d       = u128_shl1(d);
        if (u128_ge(r_int, b_full)) {
            r_int = u128_sub(r_int, b_full);
            q_int = u128_set_bit(q_int, u32(127 - i));
        }
    }
    var quot_lo_lo: u32 = 0u;
    var quot_lo_hi: u32 = 0u;
    var r = r_int;
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let overflow_bit = r.w >> 31u;
        r = u128_shl1(r);
        if (overflow_bit != 0u || u128_ge(r, b_full)) {
            r = u128_sub(r, b_full);
            if (i < 32) {
                quot_lo_lo = quot_lo_lo | (1u << u32(i));
            } else {
                quot_lo_hi = quot_lo_hi | (1u << u32(i - 32));
            }
        }
    }
    var result: Fix128Gpu;
    result.hi_lo = q_int.x;
    result.hi_hi = q_int.y;
    result.lo_lo = quot_lo_lo;
    result.lo_hi = quot_lo_hi;
    if (result_neg) { return fix128_neg(result); }
    return result;
}

// ---- u64 emulation for Morton bit-spread ----

fn u64_or(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x | b.x, a.y | b.y);
}

fn u64_and(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x & b.x, a.y & b.y);
}

fn u64_shl32(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(0u, a.x); }
fn u64_shl16(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x << 16u, (a.y << 16u) | (a.x >> 16u)); }
fn u64_shl8(a: vec2<u32>)  -> vec2<u32> { return vec2<u32>(a.x << 8u,  (a.y << 8u)  | (a.x >> 24u)); }
fn u64_shl4(a: vec2<u32>)  -> vec2<u32> { return vec2<u32>(a.x << 4u,  (a.y << 4u)  | (a.x >> 28u)); }
fn u64_shl2(a: vec2<u32>)  -> vec2<u32> { return vec2<u32>(a.x << 2u,  (a.y << 2u)  | (a.x >> 30u)); }
fn u64_shl1(a: vec2<u32>)  -> vec2<u32> { return vec2<u32>(a.x << 1u,  (a.y << 1u)  | (a.x >> 31u)); }

// Spread a 21-bit coordinate to 63 bits so each input bit `i` lands at
// output position `3 * i`. Byte-for-byte mirror of
// `alice_physics::bvh::expand_bits` on u64. Result packed as
// `vec2<u32>(low32, high32)` = full u64.
fn expand_bits_u64(v: u32) -> vec2<u32> {
    var w = vec2<u32>(v & 0x001FFFFFu, 0u);
    // Step 1: (w | (w << 32)) & 0x001F00000000FFFF
    w = u64_and(u64_or(w, u64_shl32(w)), vec2<u32>(0x0000FFFFu, 0x001F0000u));
    // Step 2: (w | (w << 16)) & 0x001F0000FF0000FF
    w = u64_and(u64_or(w, u64_shl16(w)), vec2<u32>(0xFF0000FFu, 0x001F0000u));
    // Step 3: (w | (w << 8))  & 0x100F00F00F00F00F
    w = u64_and(u64_or(w, u64_shl8(w)),  vec2<u32>(0x0F00F00Fu, 0x100F00F0u));
    // Step 4: (w | (w << 4))  & 0x10C30C30C30C30C3
    w = u64_and(u64_or(w, u64_shl4(w)),  vec2<u32>(0xC30C30C3u, 0x10C30C30u));
    // Step 5: (w | (w << 2))  & 0x1249249249249249
    w = u64_and(u64_or(w, u64_shl2(w)),  vec2<u32>(0x49249249u, 0x12492492u));
    return w;
}

// Normalize a single axis of a point relative to the world bounds and
// extract the upper 21 bits of the fractional part. Mirrors the three
// CPU branches in `alice_physics::bvh::point_to_morton`:
//   - size = 0                → coord = 0
//   - t < 0                   → coord = 0
//   - t.hi >= 1               → coord = 0x1FFFFF
//   - otherwise               → coord = (t.lo >> 43) & 0x1FFFFF
// The GPU expresses `t.lo >> 43` as `t.lo_hi >> 11` since `t.lo_hi`
// holds bits 32-63 of the low u64 half.
fn extract_u21_coord(point_axis: Fix128Gpu, bounds_min_axis: Fix128Gpu, size_axis: Fix128Gpu) -> u32 {
    if (fix128_is_zero(size_axis)) { return 0u; }
    let offset = fix128_sub_kernel(point_axis, bounds_min_axis);
    let t = fix128_div_kernel(offset, size_axis);
    if (fix128_is_negative(t)) { return 0u; }
    // `t.hi >= 1` when the integer part of the fixed-point value is at
    // least 1 — i.e. when hi_hi has any bit set or hi_lo is at least 1.
    if (t.hi_hi != 0u || t.hi_lo >= 1u) { return 0x001FFFFFu; }
    return (t.lo_hi >> 11u) & 0x001FFFFFu;
}

// ---- Main: one 63-bit Morton code per primitive ----

@compute @workgroup_size(64)
fn fix128_morton_code_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&primitives)) { return; }
    let prim = primitives[i];

    // Centre = (min + max) / 2 per axis.
    let cx_val = fix128_half(fix128_add_kernel(prim.min_x, prim.max_x));
    let cy_val = fix128_half(fix128_add_kernel(prim.min_y, prim.max_y));
    let cz_val = fix128_half(fix128_add_kernel(prim.min_z, prim.max_z));

    // World size per axis.
    let size_x = fix128_sub_kernel(world_bounds.max_x, world_bounds.min_x);
    let size_y = fix128_sub_kernel(world_bounds.max_y, world_bounds.min_y);
    let size_z = fix128_sub_kernel(world_bounds.max_z, world_bounds.min_z);

    // Normalized 21-bit coordinates.
    let cx = extract_u21_coord(cx_val, world_bounds.min_x, size_x);
    let cy = extract_u21_coord(cy_val, world_bounds.min_y, size_y);
    let cz = extract_u21_coord(cz_val, world_bounds.min_z, size_z);

    // Morton = expand(x) | (expand(y) << 1) | (expand(z) << 2).
    let ex      = expand_bits_u64(cx);
    let ey      = expand_bits_u64(cy);
    let ez      = expand_bits_u64(cz);
    let ey_shl1 = u64_shl1(ey);
    let ez_shl2 = u64_shl2(ez);
    let m       = u64_or(u64_or(ex, ey_shl1), ez_shl2);

    morton_codes[i] = m;
}
"#;

/// WGSL compute shader source for **Fix128 Morton code sort**
/// (v2.2.0 — Phase 3 §2 second primitive kernel).
///
/// Deterministic LSB-first 8-bit radix sort of 63-bit Morton codes with
/// parallel primitive indices. Byte-exact against Rust's stable
/// `sort_by_key(|(m, _)| *m)` — the sorted output is the direct input
/// for the v2.3.0 BVH build kernel.
///
/// See [`docs/PHASE_3_DESIGN.md`](../../docs/PHASE_3_DESIGN.md) §2.3
/// for the full algorithm design, determinism proof sketch, edge-case
/// list, and CPU golden strategy.
///
/// # Bindings
///
/// - `@group(0) @binding(0) var<storage, read>       codes_in:       array<vec2<u32>>` — input 63-bit Morton codes as `(low32, high32)`.
/// - `@group(0) @binding(1) var<storage, read>       indices_in:     array<u32>` — input primitive indices, parallel to `codes_in`.
/// - `@group(0) @binding(2) var<storage, read_write> codes_out:      array<vec2<u32>>` — output codes after the pass.
/// - `@group(0) @binding(3) var<storage, read_write> indices_out:    array<u32>` — output primitive indices after the pass.
/// - `@group(0) @binding(4) var<uniform>             params: SortPassParams` — `{ pass_bit_shift: u32, count: u32 }`.
/// - `@group(0) @binding(5) var<storage, read_write> histogram: array<atomic<u32>, 256>` — 256-bin histogram, zeroed by the host between passes.
/// - `@group(0) @binding(6) var<storage, read>       bucket_offsets: array<u32, 256>` — exclusive-scan of the histogram (host-side).
///
/// # Compute entries
///
/// - `fix128_morton_sort_histogram_main` (`@workgroup_size(64)`, parallel per-primitive) — builds the 256-bin histogram of the current byte via `atomicAdd`. The final counts are order-independent because addition is commutative, so parallel dispatch is safe.
/// - `fix128_morton_sort_scatter_main` (`@workgroup_size(1)`, single-thread) — sequentially scatters each `(code, index)` pair to `codes_out[bucket_offsets[b] + local_cursor[b]]`. Single-thread ensures **stability** in gid order, which is required for byte-exact CPU-GPU parity against Rust's stable Timsort.
///
/// # 8-pass control flow (host side)
///
/// For each of the 8 passes (bit shifts 0, 8, 16, 24, 32, 40, 48, 56):
/// 1. Zero the histogram buffer.
/// 2. Upload `params { pass_bit_shift, count }`.
/// 3. Dispatch `fix128_morton_sort_histogram_main` with `ceil(count / 64)` workgroups.
/// 4. Read back the histogram; compute `bucket_offsets` via CPU exclusive scan.
/// 5. Upload `bucket_offsets`.
/// 6. Dispatch `fix128_morton_sort_scatter_main` with `(1, 1, 1)` workgroup.
/// 7. Ping-pong: swap the `(codes_in, indices_in) ↔ (codes_out, indices_out)` binding roles for the next pass.
///
/// After 8 passes the final output buffer holds the sorted sequence
/// byte-for-byte identical to `sort_by_key(|(m, _)| *m)` on the CPU.
/// The v2.2.0 free function [`dispatch_fix128_morton_sort`] implements
/// this orchestration; consumers who prefer an adapter method should
/// wrap it in a `Fix128WgpuKernel::morton_sort` at a higher layer.
///
/// # Parallelisation deferred to v2.2.x
///
/// The scatter's single-thread choice is a **correctness-first**
/// decision. Parallel scatter with per-thread rank via input prefix
/// sum is compatible with stability but ~2x the code and ~2x the
/// tuning burden across the 3-platform CI matrix. It lands as an
/// additive v2.2.x kernel with the same CPU golden.
pub const FIX128_MORTON_SORT_WGSL: &str = r#"
struct SortPassParams {
    pass_bit_shift: u32,
    count:          u32,
}

@group(0) @binding(0) var<storage, read>       codes_in:       array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       indices_in:     array<u32>;
@group(0) @binding(2) var<storage, read_write> codes_out:      array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> indices_out:    array<u32>;
@group(0) @binding(4) var<uniform>             params:         SortPassParams;
@group(0) @binding(5) var<storage, read_write> histogram:      array<atomic<u32>, 256>;
@group(0) @binding(6) var<storage, read>       bucket_offsets: array<u32, 256>;

// Extract the byte at bit position `shift` (must be a multiple of 8 in
// [0, 56]) from a 63-bit Morton code packed as `(low32, high32)`.
fn extract_byte(code: vec2<u32>, shift: u32) -> u32 {
    if (shift < 32u) {
        return (code.x >> shift) & 0xFFu;
    }
    return (code.y >> (shift - 32u)) & 0xFFu;
}

// Pass 1 of 2 per radix pass: build the 256-bin histogram of the
// current byte of every Morton code.
//
// Parallel dispatch (one thread per primitive). The final histogram
// content depends only on the input codes and `pass_bit_shift`, not on
// the order in which the atomicAdd calls execute — the increment
// count per bucket is a straightforward sum of per-input matches.
//
// The host is responsible for zeroing `histogram` between passes.
@compute @workgroup_size(64)
fn fix128_morton_sort_histogram_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= params.count) { return; }
    let byte = extract_byte(codes_in[i], params.pass_bit_shift);
    atomicAdd(&histogram[byte], 1u);
}

// Pass 2 of 2 per radix pass: scatter each (code, index) pair to its
// destination in `(codes_out, indices_out)`.
//
// Single-thread sequential scatter (single workgroup, single
// invocation) for **stability**. LSB-first radix requires that
// elements with equal current-byte value preserve their previous-pass
// relative order; parallel scatter via `atomicAdd(&scatter_cursor[bucket])`
// on modern GPUs returns bucket-cursor values in atomic-order rather
// than gid-order, which breaks stability and diverges from
// `alice_physics::bvh` CPU stable sort. The v2.2.0 kernel prioritises
// byte-exact CPU-GPU parity; parallel scatter with prefix-sum-based
// rank computation is deferred to a v2.2.x optimisation.
//
// The local 256-entry cursor tracks the next write slot within each
// bucket, initialised to 0 at kernel entry. Sequential iteration in
// ascending `i` guarantees stability.
@compute @workgroup_size(1)
fn fix128_morton_sort_scatter_main() {
    var local_cursor: array<u32, 256>;
    for (var b: u32 = 0u; b < 256u; b = b + 1u) {
        local_cursor[b] = 0u;
    }
    for (var i: u32 = 0u; i < params.count; i = i + 1u) {
        let code   = codes_in[i];
        let bucket = extract_byte(code, params.pass_bit_shift);
        let pos    = bucket_offsets[bucket] + local_cursor[bucket];
        codes_out[pos]   = code;
        indices_out[pos] = indices_in[i];
        local_cursor[bucket] = local_cursor[bucket] + 1u;
    }
}
"#;

/// v2.2.0 Morton sort orchestrator — 8-pass LSB-first 8-bit radix sort
/// dispatched through the [`FIX128_MORTON_SORT_WGSL`] compute entries.
///
/// # Contract
///
/// - `codes` and `indices` must have the same length; each `codes[i]`
///   is the 63-bit Morton code for the primitive whose original index
///   is `indices[i]`, packed as `(low32, high32)`.
/// - Returns `(sorted_codes, sorted_indices)` **byte-identical** to
///   Rust's `stable_sort_by_key(|(m, _)| *m)`, where the sort key is
///   the u64 view `((high32 as u64) << 32) | (low32 as u64)`.
/// - Elements with equal Morton codes preserve their original relative
///   order (LSB-first radix + single-thread scatter = stable).
/// - `count == 0` short-circuits to empty output without dispatching
///   the kernel.
///
/// # Determinism
///
/// Every dispatch reads a fully-populated storage buffer and writes to
/// a fully-populated storage buffer. The histogram build is
/// order-independent (atomic increment sums to the same total
/// regardless of thread order); the scatter is single-thread within a
/// single workgroup so gid order is exactly the input iteration order.
/// The result is invariant across 3-platform CI (Metal / Vulkan
/// lavapipe / DX12 WARP).
///
/// # Performance note
///
/// Scatter runs single-thread for correctness; parallel scatter with
/// per-thread rank via input prefix sum is a v2.2.x optimisation. At
/// N ≤ 10000 (the current BVH broad-phase working set) the
/// single-thread scatter dominates the frame budget but still fits
/// well inside the target 60Hz step budget on M2 Metal — see the
/// v2.2.0 CHANGELOG for measurements.
///
/// # Panics
///
/// - Panics if `codes.len() != indices.len()`.
/// - Panics if the wgpu device is lost during dispatch (mirrors the
///   existing `read_buffer` behaviour).
#[cfg(feature = "physics-solver")]
#[must_use]
pub fn dispatch_fix128_morton_sort(
    device: &crate::device::GpuDevice,
    codes: &[[u32; 2]],
    indices: &[u32],
) -> (Vec<[u32; 2]>, Vec<u32>) {
    assert_eq!(
        codes.len(),
        indices.len(),
        "codes.len() must equal indices.len()"
    );
    let count = codes.len();
    if count == 0 {
        return (Vec::new(), Vec::new());
    }

    // Params uniform layout — matches the WGSL `struct SortPassParams`
    // with an explicit u64-aligned padding for portable layout across
    // the 3-platform CI matrix.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct SortPassParams {
        pass_bit_shift: u32,
        count: u32,
        _pad: [u32; 2],
    }

    // Compile the shader + both compute pipelines. An explicit shared
    // bind group layout covers all 7 bindings so a single bind group
    // can drive both the histogram (uses 3 bindings) and scatter (uses
    // 6 bindings) dispatches — using `layout: None` would derive
    // per-entry-point layouts and require distinct bind groups.
    let shader = device
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fix128_morton_sort_shader"),
            source: wgpu::ShaderSource::Wgsl(FIX128_MORTON_SORT_WGSL.into()),
        });
    let bind_group_layout =
        device
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fix128_morton_sort_bgl"),
                entries: &[
                    // 0: codes_in (read storage)
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
                    // 1: indices_in (read storage)
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
                    // 2: codes_out (read_write storage)
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
                    // 3: indices_out (read_write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 5: histogram (read_write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 6: bucket_offsets (read storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
    let pipeline_layout = device
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fix128_morton_sort_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    let histogram_pipeline =
        device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fix128_morton_sort_histogram_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("fix128_morton_sort_histogram_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
    let scatter_pipeline =
        device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fix128_morton_sort_scatter_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("fix128_morton_sort_scatter_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

    // Ping-pong storage buffers for (codes, indices).
    let codes_bytes: u64 = core::mem::size_of_val(codes) as u64;
    let indices_bytes: u64 = core::mem::size_of_val(indices) as u64;
    let buf_a_codes = device.create_buffer_init("morton_sort_a_codes", bytemuck::cast_slice(codes));
    let buf_a_indices =
        device.create_buffer_init("morton_sort_a_indices", bytemuck::cast_slice(indices));
    let buf_b_codes = device.create_buffer_empty("morton_sort_b_codes", codes_bytes);
    let buf_b_indices = device.create_buffer_empty("morton_sort_b_indices", indices_bytes);

    // Histogram + bucket_offsets storage.
    let buf_histogram = device.create_buffer_empty("morton_sort_histogram", 256 * 4);
    let buf_bucket_offsets = device.create_buffer_empty("morton_sort_bucket_offsets", 256 * 4);

    // 8-pass loop: shift 0, 8, 16, 24, 32, 40, 48, 56.
    let mut input_is_a = true;
    for pass in 0u32..8u32 {
        let shift = pass * 8;

        // Update params uniform.
        let params = SortPassParams {
            pass_bit_shift: shift,
            count: count as u32,
            _pad: [0; 2],
        };
        let buf_params =
            device.create_uniform_buffer("morton_sort_params", bytemuck::bytes_of(&params));

        // Zero the histogram between passes.
        let zeros_256 = [0u32; 256];
        device
            .queue()
            .write_buffer(&buf_histogram, 0, bytemuck::cast_slice(&zeros_256));

        // Choose input / output buffers for this pass.
        let (buf_in_codes, buf_in_indices, buf_out_codes, buf_out_indices) = if input_is_a {
            (&buf_a_codes, &buf_a_indices, &buf_b_codes, &buf_b_indices)
        } else {
            (&buf_b_codes, &buf_b_indices, &buf_a_codes, &buf_a_indices)
        };

        // Bind group. The histogram + scatter pipelines share the same
        // 7-binding layout so we build one bind group per pass and
        // reuse across both dispatches.
        let bind_group = device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("morton_sort_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_in_codes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_in_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_out_codes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf_out_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buf_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buf_histogram.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: buf_bucket_offsets.as_entire_binding(),
                    },
                ],
            });

        // Histogram dispatch (parallel).
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("morton_sort_histogram_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morton_sort_histogram_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&histogram_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (count as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        device.submit(encoder);
        device.poll_wait();

        // Read histogram; compute exclusive scan → bucket_offsets.
        let hist_raw = device.read_buffer(&buf_histogram, 256 * 4);
        let hist: &[u32] = bytemuck::cast_slice(&hist_raw);
        let mut offsets = [0u32; 256];
        let mut running = 0u32;
        for i in 0..256 {
            offsets[i] = running;
            running = running.wrapping_add(hist[i]);
        }
        device
            .queue()
            .write_buffer(&buf_bucket_offsets, 0, bytemuck::cast_slice(&offsets));

        // Scatter dispatch (single-thread for stability).
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("morton_sort_scatter_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morton_sort_scatter_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&scatter_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        device.submit(encoder);
        device.poll_wait();

        input_is_a = !input_is_a;
    }

    // After 8 passes with 8 swaps, the sorted output lives in whichever
    // buffer `input_is_a` currently points at (since the last swap
    // moved the output back to the "input" slot for the phantom 9th
    // pass that never runs).
    let (final_codes, final_indices) = if input_is_a {
        (&buf_a_codes, &buf_a_indices)
    } else {
        (&buf_b_codes, &buf_b_indices)
    };
    let codes_raw = device.read_buffer(final_codes, codes_bytes);
    let indices_raw = device.read_buffer(final_indices, indices_bytes);

    let sorted_codes: Vec<[u32; 2]> = bytemuck::cast_slice(&codes_raw).to_vec();
    let sorted_indices: Vec<u32> = bytemuck::cast_slice(&indices_raw).to_vec();
    (sorted_codes, sorted_indices)
}

// ---------------------------------------------------------------------------
// v2.3.0 — GPU LinearBvh build kernel (Phase 3 §3)
// ---------------------------------------------------------------------------

/// v2.3.0 GPU LinearBvh build kernel — single-workgroup single-thread iterative
/// port of `alice_physics::bvh::LinearBvh::build_recursive`.
///
/// Consumes the v2.2.0 sorted `(Morton codes, primitive indices)` output plus
/// a parallel array of pre-quantised i32 primitive AABBs, and produces a
/// depth-first pre-order sequence of `BvhNodeGpu` records **byte-identical**
/// to the CPU reference `LinearBvh::build(primitives).nodes`.
///
/// # Discipline — §3.1 / skill §11.4
///
/// The build MUST preserve the four correctness invariants established by
/// ALICE-Physics `dede78c` (2026-07-06 correctness fix; see
/// [`deterministic-physics-lockstep-discipline`](https://github.com/ext-sakamoro/claude-config/blob/main/claude-skills/deterministic-physics-lockstep-discipline/SKILL.md)
/// §11.4):
///
/// 1. **Position-independent placeholder** — `LEFT_ESCAPE_PLACEHOLDER = 0u`.
///    Index 0 is always the tree root; escape pointers strictly move forward;
///    no legitimate escape target is ever 0. Never use `left_idx + 1` or any
///    other position-dependent placeholder — nested recursion levels will
///    collide.
/// 2. **Subtree-size return + linear sweep** — every recursive call returns
///    the number of nodes it pushed. After the LEFT subtree returns, the
///    caller sweeps `nodes[left_idx..left_idx + left_size]` in one linear
///    pass, replacing every node whose `escape_idx() == LEFT_ESCAPE_PLACEHOLDER`
///    with the real `right_idx = left_idx + left_size`. No leftmost-spine
///    recursion; no `old_escape == root + 1` matching.
/// 3. **Debug invariant on completion** — every node's escape pointer is
///    either `ESCAPE_NONE` or strictly forward. The Rust adapter validates
///    this on readback under `#[cfg(debug_assertions)]`.
/// 4. **Traversal cycle guard** — not part of the build kernel itself, but
///    the future v2.4.0 `find_pairs` kernel MUST cap its visit counter at
///    `2 * nodes.len()` in debug builds.
///
/// # Iterative build via explicit continuation stack
///
/// WGSL has no recursion. The kernel simulates `build_recursive` with a
/// `Frame` struct (`{ start, end, escape_idx, phase, node_idx, mid,
/// left_idx, left_size }`) held in workgroup-scoped memory and a `sp`
/// (stack pointer) counter. Each frame has three phases:
///
/// - **Phase 0** — initial visit: compute the AABB union of primitives in
///   `[start, end)`, check the leaf threshold (`count <= 4`), and either
///   (a) push a leaf node and pop, or (b) push an internal node, compute
///   `mid = find_split(...)`, save state, advance to phase 1, and push
///   the LEFT child frame.
/// - **Phase 1** — after the LEFT child returns (its `subtree_size` is
///   in `return_reg`): save it as `left_size`, advance to phase 2, and
///   push the RIGHT child frame (inheriting this frame's `escape_idx`).
/// - **Phase 2** — after the RIGHT child returns: write the parent's
///   `first_child_or_prim = left_idx`, sweep the LEFT subtree to
///   backfill every `LEFT_ESCAPE_PLACEHOLDER` with `right_idx`, write
///   `return_reg = 1 + left_size + right_size`, and pop.
///
/// Return values propagate via a single `workgroup`-scoped u32 register
/// (`return_reg`) written by the popping frame and read by the parent
/// frame at the top of the next iteration.
///
/// # Bindings
///
/// - `@group(0) @binding(0)` — `sorted_codes: array<vec2<u32>>` (read).
///   The 63-bit Morton codes from v2.2.0, packed as `(low32, high32)`.
///   Used only by `find_split` to locate the highest differing bit.
/// - `@group(0) @binding(1)` — `sorted_indices: array<u32>` (read).
///   Parallel primitive indices in sorted order. Copied verbatim into
///   the `first_child_or_prim` field of every leaf via `f_start`.
/// - `@group(0) @binding(2)` — `sorted_aabbs: array<AabbI32>` (read).
///   Pre-quantised i32 AABBs in sorted order. See §CPU golden below
///   for the byte-exact equivalence with the CPU Fix128 fold.
/// - `@group(0) @binding(3)` — `params: BuildParams` (uniform).
///   `{ count: u32, leaf_max: u32, _pad0: u32, _pad1: u32 }`
///   (16-byte aligned). `leaf_max` is fixed to 4 by the Rust adapter.
/// - `@group(0) @binding(4)` — `nodes_out: array<BvhNodeGpu>` (read_write).
///   Output tree. Buffer capacity MUST be at least `2 * count + 8`.
/// - `@group(0) @binding(5)` — `node_count_out: array<atomic<u32>, 1>`
///   (read_write). Written once with the final node count at the end of
///   the dispatch.
///
/// # CPU-GPU AABB pre-quantisation equivalence
///
/// The CPU reference builds the AABB union in Fix128 space and quantises
/// to i32 only at leaf/internal creation via `aabb_to_i32_min` (floor)
/// and `aabb_to_i32_max` (ceil). The GPU pre-quantises on the host side
/// (`AabbI32Gpu::from_fix128`) and folds via i32 `min` / `max` on the
/// device. These are **byte-exact equivalent** because floor and ceil
/// are monotonic non-decreasing:
///
/// - `min(floor(a), floor(b)) = floor(min(a, b))` (monotonicity of floor)
/// - `max(ceil(a), ceil(b)) = ceil(max(a, b))` (monotonicity of ceil)
///
/// Extended to any number of terms by induction, the two folds produce
/// the same final i32 vector for every AABB in the tree.
///
/// # Determinism
///
/// Single-workgroup single-thread dispatch has no cross-thread ordering
/// hazards. Every write to `nodes_out` occurs in the same sequence and
/// with the same byte content as the CPU `Vec<BvhNode>`. The build is
/// byte-exact across the 3-platform CI matrix (Metal / Vulkan lavapipe
/// / DX12 WARP).
///
/// # Performance
///
/// Single-thread iterative build is O(N log N) for balanced Morton
/// splits; each internal node performs an O(subtree_size) linear
/// sweep for placeholder backfill. Parallel top-down BVH construction
/// (Karras-style hierarchical linear BVH with atomic index generation)
/// is 10-100x faster on large inputs but requires fresh determinism
/// analysis; deferred to v2.3.x optimisation. The v2.3.0 correctness-
/// first release lets v2.4.0 `find_pairs` integrate against a stable
/// byte-exact contract from day one.
pub const FIX128_BVH_BUILD_WGSL: &str = r#"
struct AabbI32 {
    min_x: i32,
    min_y: i32,
    min_z: i32,
    max_x: i32,
    max_y: i32,
    max_z: i32,
}

struct BvhNodeGpu {
    aabb_min_x: i32,
    aabb_min_y: i32,
    aabb_min_z: i32,
    first_child_or_prim: u32,
    aabb_max_x: i32,
    aabb_max_y: i32,
    aabb_max_z: i32,
    prim_count_escape: u32,
}

struct BuildParams {
    count:    u32,
    leaf_max: u32,
    _pad0:    u32,
    _pad1:    u32,
}

struct Frame {
    start:      u32,
    end:        u32,
    escape_idx: u32,
    phase:      u32,
    node_idx:   u32,
    mid:        u32,
    left_idx:   u32,
    left_size:  u32,
}

@group(0) @binding(0) var<storage, read>       sorted_codes:   array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read>       sorted_aabbs:   array<AabbI32>;
@group(0) @binding(3) var<uniform>             params:         BuildParams;
@group(0) @binding(4) var<storage, read_write> nodes_out:      array<BvhNodeGpu>;
@group(0) @binding(5) var<storage, read_write> node_count_out: array<atomic<u32>, 1>;

const LEFT_ESCAPE_PLACEHOLDER: u32 = 0u;
const ESCAPE_MASK_24:          u32 = 0x00FFFFFFu;
const ESCAPE_NONE_U32:         u32 = 0xFFFFFFFFu;

// Continuation stack (fixed 128 slots — safe for balanced trees up to
// ~10 million primitives; degenerate all-colocated fixtures still hit
// balanced splits via find_split's midpoint fallback).
var<workgroup> stack:      array<Frame, 128>;
// Return value of the most recently popped frame's build_recursive.
var<workgroup> return_reg: u32;
// Next node index to write in nodes_out.
var<workgroup> node_ptr:   u32;

// Compute the componentwise i32 AABB union of sorted_aabbs[start..end).
// Byte-exact equivalent to the CPU Fix128 fold followed by floor/ceil
// quantisation because floor and ceil are monotonic non-decreasing.
fn compute_aabb_union(start: u32, end: u32) -> AabbI32 {
    var aabb = sorted_aabbs[start];
    for (var i: u32 = start + 1u; i < end; i = i + 1u) {
        let p = sorted_aabbs[i];
        aabb.min_x = min(aabb.min_x, p.min_x);
        aabb.min_y = min(aabb.min_y, p.min_y);
        aabb.min_z = min(aabb.min_z, p.min_z);
        aabb.max_x = max(aabb.max_x, p.max_x);
        aabb.max_y = max(aabb.max_y, p.max_y);
        aabb.max_z = max(aabb.max_z, p.max_z);
    }
    return aabb;
}

fn u64_eq(a: vec2<u32>, b: vec2<u32>) -> bool {
    return (a.x == b.x) && (a.y == b.y);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

// Returns the highest set bit index (0..63) of a 64-bit value packed as
// (low32, high32). Returns 0 for input == 0 (unreachable in find_split
// because the caller has already checked `first_code == last_code`).
fn u64_highest_bit(v: vec2<u32>) -> u32 {
    if (v.y != 0u) {
        return 63u - countLeadingZeros(v.y);
    }
    if (v.x != 0u) {
        return 31u - countLeadingZeros(v.x);
    }
    return 0u;
}

// Returns bit `bit_idx` (0..63) of a 64-bit value packed as (low32, high32).
fn u64_bit(v: vec2<u32>, bit_idx: u32) -> u32 {
    if (bit_idx < 32u) {
        return (v.x >> bit_idx) & 1u;
    }
    return (v.y >> (bit_idx - 32u)) & 1u;
}

// Mirror of CPU `LinearBvh::find_split` byte-for-byte. Returns the split
// index m such that [start, m) goes to the LEFT subtree and [m, end)
// goes to the RIGHT.
fn find_split(start: u32, end: u32) -> u32 {
    let first_code = sorted_codes[start];
    let last_code  = sorted_codes[end - 1u];

    if (u64_eq(first_code, last_code)) {
        return (start + end) / 2u;
    }

    let diff = u64_xor(first_code, last_code);
    let highest_bit = u64_highest_bit(diff);

    var lo = start;
    var hi = end - 1u;

    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let mid_code   = sorted_codes[mid];
        let split_bit  = u64_bit(mid_code, highest_bit);
        let first_bit  = u64_bit(first_code, highest_bit);

        if (split_bit == first_bit) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    return min(max(lo, start + 1u), end - 1u);
}

fn push_leaf(aabb: AabbI32, first_prim: u32, prim_count: u32, escape_idx: u32) {
    let clamped = min(prim_count, 255u);
    let idx = node_ptr;
    nodes_out[idx].aabb_min_x = aabb.min_x;
    nodes_out[idx].aabb_min_y = aabb.min_y;
    nodes_out[idx].aabb_min_z = aabb.min_z;
    nodes_out[idx].first_child_or_prim = first_prim;
    nodes_out[idx].aabb_max_x = aabb.max_x;
    nodes_out[idx].aabb_max_y = aabb.max_y;
    nodes_out[idx].aabb_max_z = aabb.max_z;
    nodes_out[idx].prim_count_escape = ((clamped & 0xFFu) << 24u) | (escape_idx & ESCAPE_MASK_24);
    node_ptr = node_ptr + 1u;
}

fn push_internal(aabb: AabbI32, first_child: u32, escape_idx: u32) -> u32 {
    let idx = node_ptr;
    nodes_out[idx].aabb_min_x = aabb.min_x;
    nodes_out[idx].aabb_min_y = aabb.min_y;
    nodes_out[idx].aabb_min_z = aabb.min_z;
    nodes_out[idx].first_child_or_prim = first_child;
    nodes_out[idx].aabb_max_x = aabb.max_x;
    nodes_out[idx].aabb_max_y = aabb.max_y;
    nodes_out[idx].aabb_max_z = aabb.max_z;
    nodes_out[idx].prim_count_escape = escape_idx & ESCAPE_MASK_24;
    node_ptr = node_ptr + 1u;
    return idx;
}

@compute @workgroup_size(1)
fn fix128_bvh_build_main() {
    let count    = params.count;
    let leaf_max = params.leaf_max;

    node_ptr   = 0u;
    return_reg = 0u;

    if (count == 0u) {
        atomicStore(&node_count_out[0], 0u);
        return;
    }

    var sp: u32 = 0u;

    // Root frame: escape_idx = ESCAPE_NONE_U32 (the mask-24 truncation at
    // the write site produces stored value 0x00FFFFFF, matching CPU byte
    // exactly).
    stack[0].start      = 0u;
    stack[0].end        = count;
    stack[0].escape_idx = ESCAPE_NONE_U32;
    stack[0].phase      = 0u;
    sp = 1u;

    loop {
        if (sp == 0u) { break; }
        let idx      = sp - 1u;
        let phase    = stack[idx].phase;
        let f_start  = stack[idx].start;
        let f_end    = stack[idx].end;
        let f_escape = stack[idx].escape_idx;

        if (phase == 0u) {
            let count_range = f_end - f_start;
            let aabb = compute_aabb_union(f_start, f_end);

            if (count_range <= leaf_max) {
                push_leaf(aabb, f_start, count_range, f_escape);
                return_reg = 1u;
                sp = sp - 1u;
                continue;
            }

            // Internal node: push with first_child = 0 (fixed in phase 2),
            // inherit caller-supplied escape.
            let node_idx = push_internal(aabb, 0u, f_escape);
            let mid      = find_split(f_start, f_end);
            let left_idx = node_ptr;

            stack[idx].phase    = 1u;
            stack[idx].node_idx = node_idx;
            stack[idx].mid      = mid;
            stack[idx].left_idx = left_idx;

            // Push LEFT child with placeholder escape.
            stack[sp].start      = f_start;
            stack[sp].end        = mid;
            stack[sp].escape_idx = LEFT_ESCAPE_PLACEHOLDER;
            stack[sp].phase      = 0u;
            sp = sp + 1u;
            continue;
        }

        if (phase == 1u) {
            let left_size = return_reg;
            let mid       = stack[idx].mid;
            stack[idx].left_size = left_size;
            stack[idx].phase     = 2u;

            // Push RIGHT child, inheriting parent's escape.
            stack[sp].start      = mid;
            stack[sp].end        = f_end;
            stack[sp].escape_idx = f_escape;
            stack[sp].phase      = 0u;
            sp = sp + 1u;
            continue;
        }

        // phase == 2u — RIGHT subtree done. Backfill placeholders + pop.
        let right_size = return_reg;
        let node_idx   = stack[idx].node_idx;
        let left_idx   = stack[idx].left_idx;
        let left_size  = stack[idx].left_size;
        let right_idx  = left_idx + left_size;

        nodes_out[node_idx].first_child_or_prim = left_idx;

        // Linear sweep of the LEFT subtree, replacing every placeholder
        // escape with the real right_idx. Position-independent, single
        // O(subtree_size) pass — the §11.4 discipline in action.
        let end_of_left = left_idx + left_size;
        for (var slot: u32 = left_idx; slot < end_of_left; slot = slot + 1u) {
            let pce     = nodes_out[slot].prim_count_escape;
            let cur_esc = pce & ESCAPE_MASK_24;
            if (cur_esc == LEFT_ESCAPE_PLACEHOLDER) {
                let prim_count = pce >> 24u;
                nodes_out[slot].prim_count_escape = (prim_count << 24u) | (right_idx & ESCAPE_MASK_24);
            }
        }

        return_reg = 1u + left_size + right_size;
        sp = sp - 1u;
    }

    atomicStore(&node_count_out[0], node_ptr);
}
"#;

/// GPU-side 32-byte BVH node — byte-layout mirror of ALICE-Physics
/// `alice_physics::bvh::BvhNode`.
///
/// Field order and offsets are identical to the CPU counterpart:
///
/// | offset | field                 | size |
/// |--------|-----------------------|------|
/// | 0      | `aabb_min: [i32; 3]`  | 12   |
/// | 12     | `first_child_or_prim` | 4    |
/// | 16     | `aabb_max: [i32; 3]`  | 12   |
/// | 28     | `prim_count_escape`   | 4    |
///
/// The upper 8 bits of `prim_count_escape` hold the leaf primitive count
/// (0 for internal nodes); the lower 24 bits hold the escape pointer.
///
/// # Alignment
///
/// The struct uses `#[repr(C)]` without an `align` override so that
/// `bytemuck::cast_slice(&[u8]) → &[BvhNodeGpu]` works on the wgpu
/// readback buffer (which is guaranteed only 4-byte aligned). The CPU
/// `BvhNode` uses `#[repr(C, align(32))]` for cache-line friendliness
/// on the physics hot path; the byte content is identical either way.
#[cfg(feature = "physics-solver")]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNodeGpu {
    /// Componentwise `i32` floor of the AABB minimum corner (Fix128 → i32
    /// via `aabb_to_i32_min` on the CPU host).
    pub aabb_min: [i32; 3],
    /// Leaf: index into the sorted primitives array pointing to the first
    /// primitive covered by this leaf. Internal: index into the flat node
    /// array pointing to the left child.
    pub first_child_or_prim: u32,
    /// Componentwise `i32` ceil of the AABB maximum corner.
    pub aabb_max: [i32; 3],
    /// Packed: upper 8 bits = primitive count (0 for internal nodes),
    /// lower 24 bits = escape pointer (`ESCAPE_NONE = 0xFFFFFFFF` masked
    /// to `0x00FFFFFF`).
    pub prim_count_escape: u32,
}

#[cfg(feature = "physics-solver")]
impl BvhNodeGpu {
    /// Byte-exact copy from an `alice_physics::bvh::BvhNode`. Used by the
    /// v2.3.0 CPU-GPU golden test to build the expected `Vec<BvhNodeGpu>`
    /// from the CPU reference `LinearBvh::build(...).nodes` slice.
    #[must_use]
    pub fn from_physics(node: &alice_physics::bvh::BvhNode) -> Self {
        Self {
            aabb_min: node.aabb_min,
            first_child_or_prim: node.first_child_or_prim,
            aabb_max: node.aabb_max,
            prim_count_escape: node.prim_count_escape,
        }
    }

    /// Extract the escape pointer (lower 24 bits of `prim_count_escape`).
    #[must_use]
    pub const fn escape_idx(&self) -> u32 {
        self.prim_count_escape & 0x00FF_FFFF
    }

    /// Extract the primitive count (upper 8 bits of `prim_count_escape`).
    /// Returns 0 for internal nodes.
    #[must_use]
    pub const fn prim_count(&self) -> u32 {
        self.prim_count_escape >> 24
    }
}

/// GPU-side 24-byte i32 AABB — input to the v2.3.0 BVH build kernel.
///
/// Field order matches the WGSL `struct AabbI32`:
///
/// | offset | field   | size |
/// |--------|---------|------|
/// | 0      | `min_x` | 4    |
/// | 4      | `min_y` | 4    |
/// | 8      | `min_z` | 4    |
/// | 12     | `max_x` | 4    |
/// | 16     | `max_y` | 4    |
/// | 20     | `max_z` | 4    |
///
/// Constructed on the host by the caller via
/// [`AabbI32Gpu::from_physics_primitive`], which pre-quantises the
/// primitive's Fix128 AABB to i32 via floor (min) and ceil (max) — the
/// same quantisation the CPU `LinearBvh::build` applies at leaf/internal
/// creation time. See [`FIX128_BVH_BUILD_WGSL`] for the byte-exact
/// equivalence proof.
#[cfg(feature = "physics-solver")]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AabbI32Gpu {
    pub min_x: i32,
    pub min_y: i32,
    pub min_z: i32,
    pub max_x: i32,
    pub max_y: i32,
    pub max_z: i32,
}

#[cfg(feature = "physics-solver")]
impl AabbI32Gpu {
    /// Pre-quantise an `alice_physics::bvh::BvhPrimitive`'s Fix128 AABB
    /// to i32 using the same floor/ceil rule as
    /// `alice_physics::bvh::BvhNode::leaf` / `::internal`. The GPU
    /// build kernel folds these values via componentwise `min` / `max`,
    /// which is byte-exact equivalent to the CPU Fix128 fold + quantise
    /// (see monotonicity argument in [`FIX128_BVH_BUILD_WGSL`] docs).
    #[must_use]
    pub fn from_physics_primitive(prim: &alice_physics::bvh::BvhPrimitive) -> Self {
        // Reproduce `aabb_to_i32_min` / `aabb_to_i32_max` from ALICE-Physics
        // src/bvh.rs (private there, so we replicate the arithmetic).
        let a = &prim.aabb;
        Self {
            min_x: fix128_floor_i32(a.min.x),
            min_y: fix128_floor_i32(a.min.y),
            min_z: fix128_floor_i32(a.min.z),
            max_x: fix128_ceil_i32(a.max.x),
            max_y: fix128_ceil_i32(a.max.y),
            max_z: fix128_ceil_i32(a.max.z),
        }
    }
}

/// Floor of a Fix128 to i32, clamped to `[i32::MIN, i32::MAX]`. Mirrors
/// the private `fix128_floor_i32` in `alice_physics::bvh` byte-exactly:
/// the two's-complement `hi` field is already the floor for both
/// positive and negative values.
#[cfg(feature = "physics-solver")]
#[inline]
#[must_use]
fn fix128_floor_i32(v: alice_physics::math::Fix128) -> i32 {
    let hi = v.hi.max(i32::MIN as i64).min(i32::MAX as i64);
    // Explicit i32 conversion via `as` — safe because of the clamp above;
    // clippy pedantic accepts the cast here as the value is proven in
    // range.
    #[allow(clippy::cast_possible_truncation)]
    let out = hi as i32;
    out
}

/// Ceil of a Fix128 to i32, clamped to `[i32::MIN, i32::MAX]`. Mirrors
/// the private `fix128_ceil_i32` in `alice_physics::bvh` byte-exactly.
#[cfg(feature = "physics-solver")]
#[inline]
#[must_use]
fn fix128_ceil_i32(v: alice_physics::math::Fix128) -> i32 {
    let ceil = if v.lo > 0 { v.hi + 1 } else { v.hi };
    let clamped = ceil.max(i32::MIN as i64).min(i32::MAX as i64);
    #[allow(clippy::cast_possible_truncation)]
    let out = clamped as i32;
    out
}

/// v2.3.0 GPU BVH build orchestrator — dispatches [`FIX128_BVH_BUILD_WGSL`]
/// and returns the tree byte-identical to
/// `alice_physics::bvh::LinearBvh::build(primitives).nodes`.
///
/// # Contract
///
/// - `sorted_codes`, `sorted_indices`, and `sorted_aabbs` MUST have the
///   same length; each triple describes one primitive in the order
///   produced by [`dispatch_fix128_morton_sort`] (v2.2.0).
/// - Returns `Vec<BvhNodeGpu>` of length `<= 2 * count + 8`. The exact
///   length is written by the kernel to a separate atomic counter and
///   read back by the adapter.
/// - Byte-identical to the CPU reference for every fixture in the
///   v2.3.0 3-platform CI golden.
/// - `count == 0` short-circuits to an empty output without dispatching
///   the kernel.
///
/// # Debug invariant
///
/// Under `#[cfg(debug_assertions)]` the adapter runs a private
/// `debug_verify_escape_forward_impl` check on the readback, asserting
/// that every escape pointer is either `ESCAPE_NONE` (stored as
/// `0x00FFFFFF` after 24-bit truncation) or strictly forward. Any
/// backward pointer
/// indicates a bug in the port — either a placeholder was not
/// backfilled, or the sweep visited the wrong range. The check is O(N)
/// and mirrors ALICE-Physics `LinearBvh::debug_verify_escape_forward`.
///
/// # Panics
///
/// - Panics if the three input slice lengths differ.
/// - Panics if the wgpu device is lost during dispatch.
/// - In debug builds, panics if the returned tree has a backward or
///   self-referential escape pointer.
#[cfg(feature = "physics-solver")]
#[must_use]
pub fn dispatch_fix128_bvh_build(
    device: &crate::device::GpuDevice,
    sorted_codes: &[[u32; 2]],
    sorted_indices: &[u32],
    sorted_aabbs: &[AabbI32Gpu],
) -> Vec<BvhNodeGpu> {
    assert_eq!(
        sorted_codes.len(),
        sorted_indices.len(),
        "sorted_codes.len() must equal sorted_indices.len()"
    );
    assert_eq!(
        sorted_codes.len(),
        sorted_aabbs.len(),
        "sorted_codes.len() must equal sorted_aabbs.len()"
    );
    let count = sorted_codes.len();
    if count == 0 {
        return Vec::new();
    }

    // Params uniform — 16-byte aligned for cross-platform uniform buffer
    // layout (Metal / Vulkan / DX12 all require multiple-of-16 struct
    // sizes for uniform bindings).
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct BuildParams {
        count: u32,
        leaf_max: u32,
        _pad0: u32,
        _pad1: u32,
    }

    let params = BuildParams {
        count: u32::try_from(count).expect("primitive count exceeds u32::MAX"),
        leaf_max: 4,
        _pad0: 0,
        _pad1: 0,
    };

    // Node buffer capacity: a BVH over N primitives with `leaf_max = 4`
    // pushes at most `2 * ceil(N / leaf_max) - 1 <= 2 * N` nodes for
    // N >= leaf_max. Add a small safety margin for the degenerate
    // one-primitive case (which pushes exactly 1 leaf).
    let node_capacity: usize = 2 * count + 8;

    let shader = device
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fix128_bvh_build_shader"),
            source: wgpu::ShaderSource::Wgsl(FIX128_BVH_BUILD_WGSL.into()),
        });
    let bind_group_layout =
        device
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fix128_bvh_build_bgl"),
                entries: &[
                    // 0: sorted_codes (read storage)
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
                    // 1: sorted_indices (read storage)
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
                    // 2: sorted_aabbs (read storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 3: params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: nodes_out (read_write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 5: node_count_out (read_write storage, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
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
    let pipeline_layout = device
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fix128_bvh_build_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    let pipeline = device
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fix128_bvh_build_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fix128_bvh_build_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let buf_codes =
        device.create_buffer_init("bvh_build_codes", bytemuck::cast_slice(sorted_codes));
    let buf_indices =
        device.create_buffer_init("bvh_build_indices", bytemuck::cast_slice(sorted_indices));
    let buf_aabbs =
        device.create_buffer_init("bvh_build_aabbs", bytemuck::cast_slice(sorted_aabbs));
    let buf_params = device.create_uniform_buffer("bvh_build_params", bytemuck::bytes_of(&params));
    let nodes_bytes: u64 = (node_capacity * core::mem::size_of::<BvhNodeGpu>()) as u64;
    let buf_nodes = device.create_buffer_empty("bvh_build_nodes", nodes_bytes);
    let buf_node_count = device.create_buffer_empty("bvh_build_node_count", 4);

    let bind_group = device
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fix128_bvh_build_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_codes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_aabbs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_node_count.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fix128_bvh_build_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fix128_bvh_build_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    device.submit(encoder);
    device.poll_wait();

    // Read the actual node count first.
    let count_raw = device.read_buffer(&buf_node_count, 4);
    let count_slice: &[u32] = bytemuck::cast_slice(&count_raw);
    let actual_node_count = count_slice[0] as usize;
    assert!(
        actual_node_count <= node_capacity,
        "BVH build produced {} nodes, buffer capacity was {}",
        actual_node_count,
        node_capacity
    );

    let node_bytes = (actual_node_count * core::mem::size_of::<BvhNodeGpu>()) as u64;
    let nodes_raw = device.read_buffer(&buf_nodes, node_bytes);
    let nodes: Vec<BvhNodeGpu> = bytemuck::cast_slice(&nodes_raw).to_vec();

    #[cfg(debug_assertions)]
    {
        assert!(
            debug_verify_escape_forward_impl(&nodes),
            "BVH escape pointer invariant violated: backward or self-referential escape detected"
        );
    }

    nodes
}

/// Debug-only invariant check: every escape pointer in `nodes` is either
/// `ESCAPE_NONE` (stored as `0x00FF_FFFF` after 24-bit truncation) or a
/// strictly-greater node index. Backward escape pointers form cycles in
/// stackless traversal and drive `find_pairs` into unbounded push loops
/// (the 5 GB / n=50 pile SIGKILL that motivated the ALICE-Physics
/// `dede78c` correctness fix — see skill §11.4).
///
/// Mirrors ALICE-Physics `LinearBvh::debug_verify_escape_forward`
/// byte-exactly, including the vestigial `esc == ESCAPE_NONE` check
/// that never triggers because the 24-bit truncation makes the stored
/// value `0x00FF_FFFF` rather than `u32::MAX`. The second `esc <= i`
/// branch handles the ESCAPE_NONE case correctly (the stored value
/// `0x00FF_FFFF = 16_777_215` exceeds any reasonable N).
#[cfg(all(feature = "physics-solver", debug_assertions))]
fn debug_verify_escape_forward_impl(nodes: &[BvhNodeGpu]) -> bool {
    const ESCAPE_NONE: u32 = u32::MAX;
    for (i, node) in nodes.iter().enumerate() {
        let esc = node.escape_idx();
        if esc == ESCAPE_NONE {
            continue;
        }
        if (esc as usize) <= i {
            return false;
        }
    }
    true
}

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
    pipeline_div: wgpu::ComputePipeline,
    pipeline_sqrt: wgpu::ComputePipeline,
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

        let shader_div = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_div_shader"),
                source: wgpu::ShaderSource::Wgsl(FIX128_DIV_WGSL.into()),
            });

        let shader_sqrt = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_sqrt_shader"),
                source: wgpu::ShaderSource::Wgsl(FIX128_SQRT_WGSL.into()),
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
        let pipeline_div =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fix128_div_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_div,
                    entry_point: Some("fix128_div_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let pipeline_sqrt =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fix128_sqrt_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_sqrt,
                    entry_point: Some("fix128_sqrt_main"),
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
            pipeline_div,
            pipeline_sqrt,
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

    /// Dispatch the Fix128 `div` kernel (v1.4.0). `a`, `b`, and `out` must
    /// have identical lengths; the caller owns the output slice.
    ///
    /// Byte-for-byte equivalent to `Fix128Gpu::div` (the CPU reference in
    /// `physics_bridge`) on every input; the shader source
    /// [`crate::fix128::FIX128_DIV_WGSL`] entry point `fix128_div_main`
    /// implements the same 128-iter integer + 64-iter fractional long
    /// division as the CPU reference. Division by zero returns
    /// [`Fix128Gpu::ZERO`], matching `alice_physics::math::Fix128 / Fix128`.
    pub fn div(&self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        self.dispatch_binary(&self.pipeline_div, "fix128_div", a, b, out);
    }

    /// Dispatch the Fix128 `sqrt` kernel (v1.4.1). `a` and `out` must
    /// have identical lengths; the caller owns the output slice.
    ///
    /// Byte-for-byte equivalent to `Fix128Gpu::sqrt` (the CPU reference
    /// in `physics_bridge`, which delegates to
    /// `alice_physics::math::Fix128::sqrt`); the shader source
    /// [`crate::fix128::FIX128_SQRT_WGSL`] entry point `fix128_sqrt_main`
    /// runs a 64-iter Newton-Raphson loop `x = (x + a/x) / 2` on top of
    /// the v1.4.0 GPU division kernel. Sqrt of `<= 0` returns
    /// [`Fix128Gpu::ZERO`], matching the CPU sentinel.
    ///
    /// The dispatch reuses the shared 3-buffer bind group by passing
    /// `a` as the ignored `input_b`; the shader only reads `input_a`.
    pub fn sqrt(&self, a: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        self.dispatch_binary(&self.pipeline_sqrt, "fix128_sqrt", a, a, out);
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

    fn div(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        Fix128WgpuKernel::div(self, a, b, out);
    }

    fn sqrt(&mut self, a: &[Fix128Gpu], out: &mut [Fix128Gpu]) {
        Fix128WgpuKernel::sqrt(self, a, out);
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

    /// Element-wise division (Fix128 semantics, v1.4.0): `out[i] = a[i] / b[i]`.
    /// Division by zero returns `Fix128Gpu::ZERO`, matching the CPU reference
    /// `alice_physics::math::Fix128 / Fix128`.
    fn div(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut [Fix128Gpu]);

    /// Element-wise square root (Fix128 semantics, v1.4.1):
    /// `out[i] = sqrt(a[i])`. Sqrt of a non-positive input returns
    /// `Fix128Gpu::ZERO`, matching the CPU reference
    /// `alice_physics::math::Fix128::sqrt`.
    fn sqrt(&mut self, a: &[Fix128Gpu], out: &mut [Fix128Gpu]);

    /// Fix128 dot product: `out = Σ a[i] * b[i]` accumulated in
    /// ascending index order.
    fn dot(&mut self, a: &[Fix128Gpu], b: &[Fix128Gpu], out: &mut Fix128Gpu);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix128_gpu_abs_and_neg() {
        // Positive stays the same
        let pos = Fix128Gpu::from_int(5);
        assert_eq!(pos.abs().hi, 5);
        assert_eq!(pos.abs().lo, 0);
        // Zero stays zero
        assert!(Fix128Gpu::ZERO.abs().is_zero());
        // Negative flips
        let neg = Fix128Gpu::from_int(-3);
        assert_eq!(neg.abs().hi, 3);
        assert_eq!(neg.abs().lo, 0);
        // neg round-trip
        assert_eq!(pos.neg().hi, -5);
        assert_eq!(pos.neg().neg().hi, 5);
        assert!(Fix128Gpu::ZERO.neg().is_zero());
    }

    #[test]
    fn fix128_gpu_display_shows_approximate_f64() {
        // Display goes through to_f64 (non-deterministic float
        // conversion) so we assert the resulting string contains a
        // recognisable prefix rather than the exact bytes.
        let s = format!("{}", Fix128Gpu::ONE);
        assert!(s.starts_with('1'));
        let s = format!("{}", Fix128Gpu::from_int(-42));
        assert!(s.starts_with('-'));
        let s = format!("{}", Fix128Gpu::ZERO);
        assert!(s.starts_with('0'));
    }

    #[test]
    fn fix128_gpu_sign_predicates_cover_every_case() {
        // is_negative
        assert!(Fix128Gpu::from_int(-1).is_negative());
        assert!(Fix128Gpu::from_raw(-1, u64::MAX).is_negative());
        assert!(!Fix128Gpu::ZERO.is_negative());
        assert!(!Fix128Gpu::ONE.is_negative());
        assert!(!Fix128Gpu::from_raw(0, 1u64 << 63).is_negative());

        // is_zero
        assert!(Fix128Gpu::ZERO.is_zero());
        assert!(!Fix128Gpu::ONE.is_zero());
        assert!(!Fix128Gpu::from_raw(0, 1).is_zero());
        assert!(!Fix128Gpu::from_int(-1).is_zero());

        // is_positive
        assert!(Fix128Gpu::ONE.is_positive());
        assert!(Fix128Gpu::from_raw(0, 1u64 << 63).is_positive());
        assert!(Fix128Gpu::from_raw(0, 1).is_positive()); // tiny positive fraction
        assert!(!Fix128Gpu::ZERO.is_positive());
        assert!(!Fix128Gpu::from_int(-1).is_positive());
    }

    #[test]
    fn fix128_gpu_sign_predicates_are_mutually_exclusive() {
        // For every representative value, exactly one of the three
        // predicates must return true.
        let fixtures = [
            Fix128Gpu::ZERO,
            Fix128Gpu::ONE,
            Fix128Gpu::from_int(-1),
            Fix128Gpu::from_int(i64::MAX),
            Fix128Gpu::from_int(i64::MIN),
            Fix128Gpu::from_raw(0, 1),
            Fix128Gpu::from_raw(-1, u64::MAX),
        ];
        for &f in &fixtures {
            let flags = [f.is_negative(), f.is_zero(), f.is_positive()];
            let count = flags.iter().filter(|&&b| b).count();
            assert_eq!(count, 1, "exactly one sign predicate must hold for {f:?}");
        }
    }

    #[test]
    fn fix128_gpu_from_int_matches_from_raw() {
        for &n in &[-3_i64, -1, 0, 1, 2, 42, i64::MAX, i64::MIN] {
            let via_from_int = Fix128Gpu::from_int(n);
            let via_from_raw = Fix128Gpu::from_raw(n, 0);
            assert_eq!(via_from_int.hi, via_from_raw.hi);
            assert_eq!(via_from_int.lo, via_from_raw.lo);
        }
    }

    #[test]
    fn fix128_gpu_to_f64_matches_expected() {
        assert!((Fix128Gpu::ZERO.to_f64() - 0.0).abs() < 1e-12);
        assert!((Fix128Gpu::ONE.to_f64() - 1.0).abs() < 1e-12);
        assert!((Fix128Gpu::from_int(42).to_f64() - 42.0).abs() < 1e-12);
        assert!((Fix128Gpu::from_int(-5).to_f64() - -5.0).abs() < 1e-12);
        // 0.5 = raw(0, 1 << 63)
        let half = Fix128Gpu::from_raw(0, 1u64 << 63);
        assert!((half.to_f64() - 0.5).abs() < 1e-12);
    }

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

    /// The Fix128 div v1.4.0 shader ships the u128 helpers and the
    /// 128 + 64 iteration long-division entry point.
    #[test]
    fn wgsl_div_shader_helpers_present() {
        assert!(FIX128_DIV_WGSL.contains("u128_shl1"));
        assert!(FIX128_DIV_WGSL.contains("u128_ge"));
        assert!(FIX128_DIV_WGSL.contains("u128_sub"));
        assert!(FIX128_DIV_WGSL.contains("u128_set_bit"));
        assert!(FIX128_DIV_WGSL.contains("fix128_neg"));
        assert!(FIX128_DIV_WGSL.contains("fix128_abs"));
        assert!(FIX128_DIV_WGSL.contains("fix128_div_main"));
        assert!(FIX128_DIV_WGSL.contains("@compute"));
        assert!(FIX128_DIV_WGSL.contains("@workgroup_size(64)"));
        // Loop bounds must be the exact CPU-mirror constants.
        assert!(FIX128_DIV_WGSL.contains("i < 128"));
        assert!(FIX128_DIV_WGSL.contains("i >= 0"));
    }

    /// GPU dispatch bit-exact contract for `div` v1.4.0 — feeds a small
    /// panel of representative fixtures through the WGSL 128 + 64 iter
    /// long-division pipeline and checks byte-for-byte equivalence with
    /// the CPU reference [`Fix128Gpu::div`] (which delegates to
    /// `alice_physics::math::Fix128 / Fix128`).
    ///
    /// Fixtures cover the four sign-correction paths plus the divide-by-zero
    /// sentinel, mirroring the mul test structure.
    #[cfg(feature = "physics-solver")]
    #[test]
    fn wgpu_div_matches_cpu_golden() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let kernel = Fix128WgpuKernel::new(&device);

        let half_lo: u64 = 1u64 << 63;
        let a = vec![
            Fix128Gpu::from_raw(6, 0),       // 6
            Fix128Gpu::from_raw(-6, 0),      // -6
            Fix128Gpu::from_raw(6, 0),       // 6
            Fix128Gpu::from_raw(0, half_lo), // 0.5
            Fix128Gpu::from_raw(5, 0),       // 5 / 0 → zero sentinel
        ];
        let b = vec![
            Fix128Gpu::from_raw(2, 0),       // 2
            Fix128Gpu::from_raw(2, 0),       // 2
            Fix128Gpu::from_raw(-2, 0),      // -2
            Fix128Gpu::from_raw(0, half_lo), // 0.5
            Fix128Gpu::from_raw(0, 0),       // 0 → sentinel
        ];
        let mut out = vec![Fix128Gpu::ZERO; 5];

        kernel.div(&a, &b, &mut out);

        for i in 0..a.len() {
            let cpu_ref = a[i].div(b[i]);
            assert_eq!(
                out[i].hi, cpu_ref.hi,
                "div hi mismatch at i={i}: GPU {} vs CPU {}",
                out[i].hi, cpu_ref.hi
            );
            assert_eq!(
                out[i].lo, cpu_ref.lo,
                "div lo mismatch at i={i}: GPU {:#x} vs CPU {:#x}",
                out[i].lo, cpu_ref.lo
            );
        }
    }

    /// The Fix128 sqrt v1.4.1 shader ships the div helpers, the Newton
    /// loop, and the sqrt entry point.
    #[test]
    fn wgsl_sqrt_shader_helpers_present() {
        assert!(FIX128_SQRT_WGSL.contains("fix128_div_kernel"));
        assert!(FIX128_SQRT_WGSL.contains("fix128_add_kernel"));
        assert!(FIX128_SQRT_WGSL.contains("fix128_half"));
        assert!(FIX128_SQRT_WGSL.contains("countLeadingZeros"));
        assert!(FIX128_SQRT_WGSL.contains("fix128_sqrt_main"));
        assert!(FIX128_SQRT_WGSL.contains("@compute"));
        assert!(FIX128_SQRT_WGSL.contains("@workgroup_size(64)"));
        // Iteration bounds must be the exact CPU-mirror constants.
        assert!(FIX128_SQRT_WGSL.contains("i < 128")); // div: integer phase
        assert!(FIX128_SQRT_WGSL.contains("i >= 0")); // div: fractional phase
        assert!(FIX128_SQRT_WGSL.contains("i < 64")); // sqrt: Newton loop
    }

    /// GPU dispatch bit-exact contract for `sqrt` v1.4.1 — feeds a
    /// small panel through the WGSL Newton-Raphson pipeline and checks
    /// byte-for-byte equivalence with the CPU reference
    /// [`Fix128Gpu::sqrt`] (which delegates to
    /// `alice_physics::math::Fix128::sqrt`).
    ///
    /// Fixtures cover:
    ///   1. Perfect square (4)       → 2
    ///   2. Prime (2)                → irrational; exercises Newton refine
    ///   3. Fractional (0.25)        → 0.5
    ///   4. Large (10_000)           → 100
    ///   5. Negative (-1)            → ZERO sentinel
    ///   6. Zero                     → ZERO
    #[cfg(feature = "physics-solver")]
    #[test]
    fn wgpu_sqrt_matches_cpu_golden() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let kernel = Fix128WgpuKernel::new(&device);

        let quarter_lo: u64 = 1u64 << 62; // 0.25 in Fix128
        let a = vec![
            Fix128Gpu::from_raw(4, 0),          // 4
            Fix128Gpu::from_raw(2, 0),          // 2
            Fix128Gpu::from_raw(0, quarter_lo), // 0.25
            Fix128Gpu::from_raw(10_000, 0),     // 10,000
            Fix128Gpu::from_raw(-1, 0),         // -1 → 0
            Fix128Gpu::from_raw(0, 0),          // 0 → 0
        ];
        let mut out = vec![Fix128Gpu::ZERO; a.len()];

        kernel.sqrt(&a, &mut out);

        for i in 0..a.len() {
            let cpu_ref = a[i].sqrt();
            assert_eq!(
                out[i].hi, cpu_ref.hi,
                "sqrt hi mismatch at i={i}: GPU {} vs CPU {}",
                out[i].hi, cpu_ref.hi
            );
            assert_eq!(
                out[i].lo, cpu_ref.lo,
                "sqrt lo mismatch at i={i}: GPU {:#x} vs CPU {:#x}",
                out[i].lo, cpu_ref.lo
            );
        }
    }

    /// The `Fix128GpuKernel` trait bridge now routes `sqrt` to the live
    /// wgpu pipeline (v1.4.1). Verifies the trait `sqrt` matches
    /// [`Fix128WgpuKernel::sqrt`] byte-for-byte.
    #[cfg(feature = "physics-solver")]
    #[test]
    fn wgpu_trait_sqrt_matches_inherent_sqrt() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut kernel = Fix128WgpuKernel::new(&device);

        let a = vec![Fix128Gpu::from_raw(4, 0), Fix128Gpu::from_raw(9, 0)];
        let mut via_inherent = vec![Fix128Gpu::ZERO; 2];
        let mut via_trait = vec![Fix128Gpu::ZERO; 2];

        Fix128WgpuKernel::sqrt(&kernel, &a, &mut via_inherent);
        <Fix128WgpuKernel<'_> as Fix128GpuKernel>::sqrt(&mut kernel, &a, &mut via_trait);

        assert_eq!(via_inherent, via_trait);
    }

    /// The `Fix128GpuKernel` trait bridge now routes `div` to the live
    /// wgpu pipeline (v1.4.0). Verifies the trait `div` matches
    /// [`Fix128WgpuKernel::div`] byte-for-byte.
    #[cfg(feature = "physics-solver")]
    #[test]
    fn wgpu_trait_div_matches_inherent_div() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut kernel = Fix128WgpuKernel::new(&device);

        let a = vec![Fix128Gpu::from_raw(6, 0), Fix128Gpu::from_raw(-6, 0)];
        let b = vec![Fix128Gpu::from_raw(2, 0), Fix128Gpu::from_raw(3, 0)];
        let mut via_inherent = vec![Fix128Gpu::ZERO; 2];
        let mut via_trait = vec![Fix128Gpu::ZERO; 2];

        Fix128WgpuKernel::div(&kernel, &a, &b, &mut via_inherent);
        <Fix128WgpuKernel<'_> as Fix128GpuKernel>::div(&mut kernel, &a, &b, &mut via_trait);

        assert_eq!(via_inherent, via_trait);
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

    // -----------------------------------------------------------------
    // v2.1.0 Phase 3 first primitive — Fix128 AABB helpers
    // -----------------------------------------------------------------

    /// Fix128 AABB helpers WGSL ships the struct definitions and every
    /// helper the v2.2+ Phase 3 pipeline consumes: `fix128_add / sub /
    /// lt / min / max`, `aabb_from_sphere`, and `aabb_union`. The
    /// smoke `@compute` entry keeps the module standalone-compilable.
    #[test]
    fn wgsl_aabb_helpers_shader_present() {
        assert!(FIX128_AABB_HELPERS_WGSL.contains("struct Fix128Gpu"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("struct Fix128AabbGpu"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn u64_add"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn u64_sub"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn fix128_add"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn fix128_sub"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn fix128_lt"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn fix128_min"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn fix128_max"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn aabb_from_sphere"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn aabb_union"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("fn aabb_helpers_smoke_main"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("@compute"));
        assert!(FIX128_AABB_HELPERS_WGSL.contains("@workgroup_size(1)"));
    }

    /// The AABB helpers module must be a standalone valid WGSL shader.
    /// Skips when no GPU adapter is available (headless CI).
    #[test]
    fn wgsl_aabb_helpers_shader_compiles() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let _ = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_aabb_helpers_shader_compile_test"),
                source: wgpu::ShaderSource::Wgsl(FIX128_AABB_HELPERS_WGSL.into()),
            });
    }

    /// `Fix128AabbGpu` occupies the exact `6 × Fix128Gpu = 96` bytes
    /// so the type can be uploaded directly as a storage buffer.
    #[test]
    fn fix128_aabb_gpu_size_and_layout() {
        assert_eq!(core::mem::size_of::<Fix128AabbGpu>(), 96);
        assert_eq!(core::mem::size_of::<Fix128Gpu>(), 16);
    }

    /// `Fix128AabbGpu::from_sphere` matches the WGSL `aabb_from_sphere`
    /// helper byte-for-byte on the CPU reference. The GPU dispatch
    /// path is exercised by the v2.2+ Morton kernel tests.
    #[test]
    fn fix128_aabb_gpu_from_sphere() {
        let pos_x = Fix128Gpu::from_int(3);
        let pos_y = Fix128Gpu::from_int(5);
        let pos_z = Fix128Gpu::from_int(-1);
        let radius = Fix128Gpu::from_raw(0, 1u64 << 63); // 0.5
        let aabb = Fix128AabbGpu::from_sphere(pos_x, pos_y, pos_z, radius);
        assert_eq!(aabb.min_x, pos_x.sub(radius));
        assert_eq!(aabb.min_y, pos_y.sub(radius));
        assert_eq!(aabb.min_z, pos_z.sub(radius));
        assert_eq!(aabb.max_x, pos_x.add(radius));
        assert_eq!(aabb.max_y, pos_y.add(radius));
        assert_eq!(aabb.max_z, pos_z.add(radius));
    }

    /// v2.1.0 Morton code shader ships every helper it needs to normalize
    /// a Fix128 coordinate via the v1.4.0 div kernel, extract the upper
    /// 21 bits, and spread them via the u64-emulated `expand_bits`
    /// sequence into a full 63-bit Morton code. Compute entry is
    /// `fix128_morton_code_main`.
    #[test]
    fn wgsl_morton_code_shader_present() {
        assert!(FIX128_MORTON_CODE_WGSL.contains("struct Fix128Gpu"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("struct Fix128AabbGpu"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("primitives"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("world_bounds"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("morton_codes"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("fn fix128_add_kernel"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("fn fix128_sub_kernel"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("fn fix128_mul_kernel"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("fn fix128_div_kernel"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("fn fix128_half"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("fn expand_bits_u64"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("fn extract_u21_coord"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("fn fix128_morton_code_main"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("@compute"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("@workgroup_size(64)"));
        // The five Morton bit-spread mask constants must be present verbatim.
        assert!(FIX128_MORTON_CODE_WGSL.contains("0x001F0000u"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("0xFF0000FFu"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("0x100F00F0u"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("0x10C30C30u"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("0x49249249u"));
        assert!(FIX128_MORTON_CODE_WGSL.contains("0x12492492u"));
        // The 21-bit mask must be present exactly.
        assert!(FIX128_MORTON_CODE_WGSL.contains("0x001FFFFFu"));
    }

    /// Morton code shader must compile as valid WGSL end-to-end
    /// (structs + bindings + all Fix128 primitives + Morton logic +
    /// compute entry point). Skips when no GPU adapter is available
    /// (headless CI).
    #[test]
    fn wgsl_morton_code_shader_compiles() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let _ = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_morton_code_shader_compile_test"),
                source: wgpu::ShaderSource::Wgsl(FIX128_MORTON_CODE_WGSL.into()),
            });
    }

    /// v2.2.0 Morton sort kernel ships two compute entries — a parallel
    /// histogram builder and a single-thread stable scatter — plus the
    /// 7-binding contract documented in [`docs/PHASE_3_DESIGN.md`] §2.3.
    ///
    /// [`docs/PHASE_3_DESIGN.md`]: ../../docs/PHASE_3_DESIGN.md
    #[test]
    fn wgsl_morton_sort_shader_present() {
        assert!(FIX128_MORTON_SORT_WGSL.contains("struct SortPassParams"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("codes_in"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("indices_in"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("codes_out"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("indices_out"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("params"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("histogram"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("bucket_offsets"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("array<atomic<u32>, 256>"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("array<u32, 256>"));
        // Two compute entries — histogram (parallel) + scatter (single-thread).
        assert!(FIX128_MORTON_SORT_WGSL.contains("fn fix128_morton_sort_histogram_main"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("fn fix128_morton_sort_scatter_main"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("@workgroup_size(64)"));
        assert!(FIX128_MORTON_SORT_WGSL.contains("@workgroup_size(1)"));
        // Byte-extract helper that supports both 32-bit halves of the 63-bit code.
        assert!(FIX128_MORTON_SORT_WGSL.contains("fn extract_byte"));
        // Stability comment identifies the design constraint that pins
        // scatter to single-thread.
        assert!(FIX128_MORTON_SORT_WGSL.contains("stability"));
    }

    /// Byte-exact GPU-CPU golden for the v2.2.0 Morton sort kernel.
    ///
    /// Exercises seven edge cases via the [`dispatch_fix128_morton_sort`]
    /// orchestrator and compares byte-for-byte against Rust's stable
    /// `sort_by_key` on the paired `(Morton, index)` sequence:
    ///
    /// 1. **Empty** — `count == 0` short-circuits without dispatch.
    /// 2. **Single element** — trivially sorted; every pass leaves it in place.
    /// 3. **All Morton codes identical** — degenerate histogram (one bucket
    ///    holds every element); stability preserves input order.
    /// 4. **All Morton codes distinct** — general case, 64 elements.
    /// 5. **Duplicate Morton codes** — pairs of identical codes with
    ///    distinct indices; asserts that lower-input-index pairs come
    ///    first (stability).
    /// 6. **Ascending pre-sorted input** — output equals input verbatim.
    /// 7. **Descending pre-sorted input** — full reversal; every bucket
    ///    crossing is exercised.
    ///
    /// Skips when no GPU adapter is available (headless CI).
    #[cfg(feature = "physics-solver")]
    #[test]
    fn wgpu_morton_sort_matches_cpu_golden() {
        // CPU golden mirrors the target: pair each (code_u64, index)
        // then stable-sort by code. Byte layout of the returned pair is
        // then reconstructed to match the GPU output layout.
        fn cpu_sort(codes: &[[u32; 2]], indices: &[u32]) -> (Vec<[u32; 2]>, Vec<u32>) {
            let mut pairs: Vec<(u64, u32, [u32; 2])> = codes
                .iter()
                .zip(indices.iter())
                .map(|(c, i)| ((u64::from(c[0])) | ((u64::from(c[1])) << 32), *i, *c))
                .collect();
            pairs.sort_by_key(|(m, _, _)| *m);
            let sorted_codes = pairs.iter().map(|(_, _, c)| *c).collect();
            let sorted_indices = pairs.iter().map(|(_, i, _)| *i).collect();
            (sorted_codes, sorted_indices)
        }

        // Helper to build a `[u32; 2]` code from a u64 value.
        fn to_code(v: u64) -> [u32; 2] {
            [v as u32, (v >> 32) as u32]
        }

        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Headless CI, skip.
        };

        // ---- Fixture 1: empty ----
        {
            let codes: Vec<[u32; 2]> = Vec::new();
            let indices: Vec<u32> = Vec::new();
            let (gpu_c, gpu_i) = dispatch_fix128_morton_sort(&device, &codes, &indices);
            let (cpu_c, cpu_i) = cpu_sort(&codes, &indices);
            assert_eq!(gpu_c, cpu_c, "empty fixture codes mismatch");
            assert_eq!(gpu_i, cpu_i, "empty fixture indices mismatch");
        }

        // ---- Fixture 2: single element ----
        {
            let codes = vec![to_code(0x0000_1234_5678_9ABC)];
            let indices = vec![42u32];
            let (gpu_c, gpu_i) = dispatch_fix128_morton_sort(&device, &codes, &indices);
            let (cpu_c, cpu_i) = cpu_sort(&codes, &indices);
            assert_eq!(gpu_c, cpu_c, "single fixture codes mismatch");
            assert_eq!(gpu_i, cpu_i, "single fixture indices mismatch");
        }

        // ---- Fixture 3: all Morton codes identical ----
        {
            let codes = vec![to_code(0x0000_DEAD_BEEF_CAFEu64); 32];
            let indices: Vec<u32> = (0..32).collect();
            let (gpu_c, gpu_i) = dispatch_fix128_morton_sort(&device, &codes, &indices);
            let (cpu_c, cpu_i) = cpu_sort(&codes, &indices);
            assert_eq!(gpu_c, cpu_c, "identical-codes fixture codes mismatch");
            assert_eq!(gpu_i, cpu_i, "identical-codes fixture indices mismatch");
            // Stability: indices should stay in 0..32 order.
            assert_eq!(
                gpu_i,
                (0..32).collect::<Vec<_>>(),
                "identical-codes stability broken"
            );
        }

        // ---- Fixture 4: all distinct, 64 elements ----
        {
            let codes: Vec<[u32; 2]> = (0..64u64)
                .map(|i| {
                    let v = i
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(0x1234_5678_9ABC_DEF0);
                    to_code(v & 0x7FFF_FFFF_FFFF_FFFF)
                })
                .collect();
            let indices: Vec<u32> = (0..64).collect();
            let (gpu_c, gpu_i) = dispatch_fix128_morton_sort(&device, &codes, &indices);
            let (cpu_c, cpu_i) = cpu_sort(&codes, &indices);
            assert_eq!(gpu_c, cpu_c, "distinct-64 fixture codes mismatch");
            assert_eq!(gpu_i, cpu_i, "distinct-64 fixture indices mismatch");
        }

        // ---- Fixture 5: duplicate Morton codes with distinct indices ----
        {
            // Pattern: pairs of same-code elements with two indices each.
            let mut codes = Vec::new();
            let mut indices = Vec::new();
            for i in 0u32..16 {
                let m = to_code(u64::from(i) * 0x0000_0100_0000_0000);
                codes.push(m);
                indices.push(i * 2);
                codes.push(m);
                indices.push(i * 2 + 1);
            }
            let (gpu_c, gpu_i) = dispatch_fix128_morton_sort(&device, &codes, &indices);
            let (cpu_c, cpu_i) = cpu_sort(&codes, &indices);
            assert_eq!(gpu_c, cpu_c, "duplicate-codes fixture codes mismatch");
            assert_eq!(gpu_i, cpu_i, "duplicate-codes fixture indices mismatch");
            // Stability spot check: within each same-code pair, the
            // lower original index must come first.
            for chunk in gpu_i.chunks(2) {
                assert!(
                    chunk[0] < chunk[1],
                    "stability broken within same-code pair: {chunk:?}"
                );
            }
        }

        // ---- Fixture 6: ascending pre-sorted input ----
        {
            let codes: Vec<[u32; 2]> = (0..48u64).map(|v| to_code(v * 7 + 1)).collect();
            let indices: Vec<u32> = (0..48).collect();
            let (gpu_c, gpu_i) = dispatch_fix128_morton_sort(&device, &codes, &indices);
            let (cpu_c, cpu_i) = cpu_sort(&codes, &indices);
            assert_eq!(gpu_c, cpu_c, "ascending fixture codes mismatch");
            assert_eq!(gpu_i, cpu_i, "ascending fixture indices mismatch");
            // Sorted output equals input.
            assert_eq!(gpu_c, codes, "ascending pre-sorted should be a no-op");
        }

        // ---- Fixture 7: descending pre-sorted input ----
        {
            let codes: Vec<[u32; 2]> = (0..48u64).rev().map(|v| to_code(v * 7 + 1)).collect();
            let indices: Vec<u32> = (0..48).collect();
            let (gpu_c, gpu_i) = dispatch_fix128_morton_sort(&device, &codes, &indices);
            let (cpu_c, cpu_i) = cpu_sort(&codes, &indices);
            assert_eq!(gpu_c, cpu_c, "descending fixture codes mismatch");
            assert_eq!(gpu_i, cpu_i, "descending fixture indices mismatch");
        }
    }

    /// Morton sort skeleton must compile as valid WGSL end-to-end so
    /// external consumers who wire against the binding layout get
    /// a working handshake ahead of the v2.2.0 impl body landing.
    /// Skips when no GPU adapter is available (headless CI).
    #[test]
    fn wgsl_morton_sort_shader_compiles() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let _ = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_morton_sort_shader_compile_test"),
                source: wgpu::ShaderSource::Wgsl(FIX128_MORTON_SORT_WGSL.into()),
            });
    }

    /// GPU dispatch bit-exact contract for the v2.1.0 Morton kernel.
    ///
    /// Feeds 16 hand-crafted primitives + a fixed world AABB through
    /// the `fix128_morton_code_main` kernel and compares the returned
    /// `vec2<u32>` codes byte-for-byte against
    /// `alice_physics::bvh::point_to_morton` on the same primitives.
    ///
    /// The fixture exercises all three CPU coordinate branches:
    ///   - Interior primitives (t in the open unit interval)
    ///   - Boundary primitives (centre exactly on the world bounds)
    ///   - Out-of-bounds primitives on both the low (t < 0) and high
    ///     (t.hi >= 1) sides of each axis
    ///
    /// A successful golden match on all 16 fixtures confirms the GPU
    /// kernel reproduces the Fix128 divide, the 21-bit extraction, and
    /// the 5-round `expand_bits` bit-spread pipeline bit-exact against
    /// the CPU reference.
    #[cfg(feature = "physics-solver")]
    #[test]
    fn wgpu_morton_code_matches_cpu_golden() {
        use alice_physics::bvh::point_to_morton;
        use alice_physics::collider::AABB;
        use alice_physics::math::{Fix128 as PhysicsFix128, Vec3Fix};

        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // Headless CI, skip
        };

        // Byte-for-byte conversion helpers between Fix128Gpu and
        // alice_physics::math::Fix128 (both share the same
        // #[repr(C)] { hi: i64, lo: u64 } layout).
        fn to_physics(g: Fix128Gpu) -> PhysicsFix128 {
            PhysicsFix128::from_raw(g.hi, g.lo)
        }

        // Fixture: 16 primitives with mixed centres, plus a fixed world.
        //
        // The centre of each AABB is `(min + max) / 2`. Every AABB is
        // a Sphere-derived AABB with radius 0.5 so `centre == pos`.
        let world_bounds = Fix128AabbGpu::from_sphere(
            Fix128Gpu::from_int(0),
            Fix128Gpu::from_int(0),
            Fix128Gpu::from_int(0),
            Fix128Gpu::from_int(10),
        );

        let r = Fix128Gpu::from_raw(0, 1u64 << 63); // 0.5
        let mk = |x: i64, y: i64, z: i64| {
            Fix128AabbGpu::from_sphere(
                Fix128Gpu::from_int(x),
                Fix128Gpu::from_int(y),
                Fix128Gpu::from_int(z),
                r,
            )
        };
        let primitives: Vec<Fix128AabbGpu> = vec![
            mk(0, 0, 0),    // centre
            mk(1, 2, 3),    // interior
            mk(-1, -2, -3), // interior negative
            mk(5, 5, 5),    // interior positive
            mk(-5, -5, -5), // interior negative
            mk(-10, 0, 0),  // boundary low x
            mk(10, 0, 0),   // boundary high x
            mk(0, -10, 0),  // boundary low y
            mk(0, 10, 0),   // boundary high y
            mk(0, 0, -10),  // boundary low z
            mk(0, 0, 10),   // boundary high z
            mk(-15, 0, 0),  // out-of-bounds low x (t < 0 branch)
            mk(15, 0, 0),   // out-of-bounds high x (t.hi >= 1 branch)
            mk(0, -20, 0),  // out-of-bounds low y
            mk(0, 20, 0),   // out-of-bounds high y
            mk(7, -3, 4),   // interior mixed
        ];
        assert_eq!(primitives.len(), 16);

        // ---- CPU golden ----
        let cpu_codes: Vec<u64> = primitives
            .iter()
            .map(|p| {
                let centre = Vec3Fix::new(
                    (to_physics(p.min_x) + to_physics(p.max_x)).half(),
                    (to_physics(p.min_y) + to_physics(p.max_y)).half(),
                    (to_physics(p.min_z) + to_physics(p.max_z)).half(),
                );
                let wb = AABB {
                    min: Vec3Fix::new(
                        to_physics(world_bounds.min_x),
                        to_physics(world_bounds.min_y),
                        to_physics(world_bounds.min_z),
                    ),
                    max: Vec3Fix::new(
                        to_physics(world_bounds.max_x),
                        to_physics(world_bounds.max_y),
                        to_physics(world_bounds.max_z),
                    ),
                };
                point_to_morton(centre, &wb)
            })
            .collect();

        // ---- GPU dispatch ----
        let primitives_bytes: &[u8] = bytemuck::cast_slice(&primitives);
        let world_bounds_bytes: &[u8] = bytemuck::bytes_of(&world_bounds);

        let buf_primitives = device.create_buffer_init("morton_primitives", primitives_bytes);
        let buf_world = device.create_uniform_buffer("morton_world_bounds", world_bounds_bytes);
        let output_bytes: u64 = (primitives.len() * core::mem::size_of::<[u32; 2]>()) as u64;
        let buf_output = device.create_buffer_empty("morton_output", output_bytes);

        let shader = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("morton_code_shader"),
                source: wgpu::ShaderSource::Wgsl(FIX128_MORTON_CODE_WGSL.into()),
            });

        let pipeline = device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("morton_code_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("fix128_morton_code_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group = device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("morton_code_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_primitives.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_world.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_output.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("morton_code_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morton_code_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (primitives.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        device.submit(encoder);
        device.poll_wait();

        // Read back
        let raw = device.read_buffer(&buf_output, output_bytes);
        let gpu_codes: &[[u32; 2]] = bytemuck::cast_slice(&raw);

        // ---- Compare ----
        for (i, (cpu, gpu)) in cpu_codes.iter().zip(gpu_codes.iter()).enumerate() {
            let gpu_u64 = (u64::from(gpu[0])) | ((u64::from(gpu[1])) << 32);
            assert_eq!(
                *cpu, gpu_u64,
                "primitive {i}: CPU {cpu:#018x} vs GPU {gpu_u64:#018x}"
            );
        }
    }

    /// `Fix128AabbGpu::union` matches the WGSL `aabb_union` helper
    /// byte-for-byte on the CPU reference. Exercises positive, negative,
    /// and mixed-sign corners so the signed compare (`fix128_lt`)
    /// branch coverage is complete.
    #[test]
    fn fix128_aabb_gpu_union() {
        // AABB 1: [-1, -1, -1] × [1, 1, 1]
        let one = Fix128Gpu::ONE;
        let neg_one = Fix128Gpu::from_int(-1);
        let a = Fix128AabbGpu {
            min_x: neg_one,
            min_y: neg_one,
            min_z: neg_one,
            max_x: one,
            max_y: one,
            max_z: one,
        };
        // AABB 2: [0, -3, 2] × [5, 0, 4]
        let three_neg = Fix128Gpu::from_int(-3);
        let two = Fix128Gpu::from_int(2);
        let four = Fix128Gpu::from_int(4);
        let five = Fix128Gpu::from_int(5);
        let zero = Fix128Gpu::ZERO;
        let b = Fix128AabbGpu {
            min_x: zero,
            min_y: three_neg,
            min_z: two,
            max_x: five,
            max_y: zero,
            max_z: four,
        };
        let u = a.union(b);
        // Union: min = componentwise min, max = componentwise max
        assert_eq!(u.min_x, neg_one);
        assert_eq!(u.min_y, three_neg);
        assert_eq!(u.min_z, neg_one);
        assert_eq!(u.max_x, five);
        assert_eq!(u.max_y, one);
        assert_eq!(u.max_z, four);
    }

    // -----------------------------------------------------------------
    // v2.3.0 GPU BVH build kernel tests
    // -----------------------------------------------------------------

    /// v2.3.0 BVH build kernel ships the WGSL struct + binding + entry
    /// point set documented in [`docs/PHASE_3_DESIGN.md`] §2.4. This
    /// structural test runs without a GPU adapter so headless CI still
    /// catches accidental deletion of any surface-level identifier.
    ///
    /// [`docs/PHASE_3_DESIGN.md`]: ../../docs/PHASE_3_DESIGN.md
    #[test]
    fn wgsl_bvh_build_shader_present() {
        // Structs from the design doc §2.4 bindings section.
        assert!(FIX128_BVH_BUILD_WGSL.contains("struct AabbI32"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("struct BvhNodeGpu"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("struct BuildParams"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("struct Frame"));
        // Bindings (0..5 in the layout).
        assert!(FIX128_BVH_BUILD_WGSL.contains("sorted_codes"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("sorted_indices"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("sorted_aabbs"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("nodes_out"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("node_count_out"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("array<atomic<u32>, 1>"));
        // Placeholder discipline constants (§3.1 / skill §11.4).
        assert!(FIX128_BVH_BUILD_WGSL.contains("LEFT_ESCAPE_PLACEHOLDER: u32 = 0u"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("ESCAPE_MASK_24"));
        // Continuation stack + helpers.
        assert!(FIX128_BVH_BUILD_WGSL.contains("var<workgroup> stack:"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("fn compute_aabb_union"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("fn find_split"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("fn push_leaf"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("fn push_internal"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("fn u64_highest_bit"));
        // Single compute entry, dispatched at 1x1x1.
        assert!(FIX128_BVH_BUILD_WGSL.contains("fn fix128_bvh_build_main"));
        assert!(FIX128_BVH_BUILD_WGSL.contains("@workgroup_size(1)"));
        // Discipline check on the sweep loop: iterates over LEFT subtree
        // slice only, not the whole tree.
        assert!(FIX128_BVH_BUILD_WGSL.contains("LEFT_ESCAPE_PLACEHOLDER"));
    }

    /// v2.3.0 BVH build kernel must compile as valid WGSL end-to-end.
    /// Runs the naga parser via `create_shader_module` and fails loudly
    /// on any syntax / type error. Skips when no GPU adapter is
    /// available (headless CI).
    #[test]
    fn wgsl_bvh_build_shader_compiles() {
        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let _ = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fix128_bvh_build_shader_compile_test"),
                source: wgpu::ShaderSource::Wgsl(FIX128_BVH_BUILD_WGSL.into()),
            });
    }

    /// Byte-exact GPU-CPU golden for the v2.3.0 BVH build kernel.
    ///
    /// Exercises three fixtures that surface the §3.1 / skill §11.4
    /// discipline requirements:
    ///
    /// 1. **Pile (32 primitives)** — tightly-packed spheres in a 4x4x2
    ///    grid. Exercises balanced Morton splits and escape-pointer
    ///    chaining through several tree levels. Mirrors the geometry
    ///    of the `stage_breakdown_collider` bench fixture that surfaced
    ///    the ALICE-Physics `dede78c` correctness fix.
    /// 2. **Uniform grid (27 primitives)** — 3x3x3 lattice on integer
    ///    coordinates. Exercises the "highest differing bit" split path
    ///    across three cardinal axes and forces balanced tree depth.
    /// 3. **Degenerate all-colocated (16 primitives)** — all AABBs at
    ///    the origin. Exercises the `first_code == last_code` branch of
    ///    `find_split` (falls back to `(start + end) / 2`) plus the
    ///    single-AABB world bounds. This is the same degenerate
    ///    configuration that stress-tests the CPU recursion for
    ///    placeholder collision (skill §11.4).
    ///
    /// For every fixture the CPU reference `LinearBvh::build(...).nodes`
    /// is compared byte-for-byte to the GPU output, and additionally the
    /// GPU output is asserted to satisfy the §3.1 debug invariant (no
    /// backward escape pointers). Skips when no GPU adapter is available
    /// (headless CI).
    #[cfg(feature = "physics-solver")]
    #[test]
    fn wgpu_bvh_build_matches_cpu_golden() {
        use alice_physics::bvh::{point_to_morton, BvhPrimitive, LinearBvh};
        use alice_physics::collider::AABB;
        use alice_physics::math::{Fix128 as PhysicsFix128, Vec3Fix};

        let device = match crate::device::GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };

        /// Build one `BvhPrimitive` from an integer sphere centre and
        /// radius. AABB = [centre - radius, centre + radius] on each
        /// axis, matching the ALICE-Physics sphere-derived AABB shape.
        fn mk_prim(cx: i64, cy: i64, cz: i64, r: i64, idx: u32) -> BvhPrimitive {
            let fx = PhysicsFix128::from_int(cx);
            let fy = PhysicsFix128::from_int(cy);
            let fz = PhysicsFix128::from_int(cz);
            let fr = PhysicsFix128::from_int(r);
            BvhPrimitive {
                aabb: AABB::new(
                    Vec3Fix::new(fx - fr, fy - fr, fz - fr),
                    Vec3Fix::new(fx + fr, fy + fr, fz + fr),
                ),
                index: idx,
                morton: 0,
            }
        }

        /// Reproduce the world-bounds + Morton-code + stable-sort
        /// preamble from `LinearBvh::build` to obtain the three parallel
        /// arrays the v2.3.0 kernel consumes. Returns `(sorted_codes,
        /// sorted_indices, sorted_aabbs)` in the same order the CPU
        /// implementation processes them.
        fn cpu_sort(input: &[BvhPrimitive]) -> (Vec<[u32; 2]>, Vec<u32>, Vec<AabbI32Gpu>) {
            let mut prims: Vec<BvhPrimitive> = input.to_vec();
            if prims.is_empty() {
                return (Vec::new(), Vec::new(), Vec::new());
            }
            // World bounds — union of every primitive AABB.
            let mut bounds = prims[0].aabb;
            for p in &prims[1..] {
                bounds = bounds.union(&p.aabb);
            }
            // Per-primitive Morton codes (byte-exact CPU reference).
            for p in prims.iter_mut() {
                let centre = Vec3Fix::new(
                    (p.aabb.min.x + p.aabb.max.x).half(),
                    (p.aabb.min.y + p.aabb.max.y).half(),
                    (p.aabb.min.z + p.aabb.max.z).half(),
                );
                p.morton = point_to_morton(centre, &bounds);
            }
            // Stable sort (matches CPU's sort_by_key).
            prims.sort_by_key(|p| p.morton);

            let codes: Vec<[u32; 2]> = prims
                .iter()
                .map(|p| [p.morton as u32, (p.morton >> 32) as u32])
                .collect();
            let indices: Vec<u32> = prims.iter().map(|p| p.index).collect();
            let aabbs: Vec<AabbI32Gpu> = prims
                .iter()
                .map(AabbI32Gpu::from_physics_primitive)
                .collect();
            (codes, indices, aabbs)
        }

        // Compare CPU `LinearBvh` output to the GPU dispatch output on
        // the given fixture. Byte-exact assert on every node.
        let check = |label: &str, input: Vec<BvhPrimitive>| {
            let cpu_bvh = LinearBvh::build(input.clone());
            let (codes, indices, aabbs) = cpu_sort(&input);
            let gpu_nodes = dispatch_fix128_bvh_build(&device, &codes, &indices, &aabbs);

            assert_eq!(
                gpu_nodes.len(),
                cpu_bvh.nodes.len(),
                "{label}: node count mismatch (GPU={} CPU={})",
                gpu_nodes.len(),
                cpu_bvh.nodes.len()
            );
            for (i, (g, c)) in gpu_nodes.iter().zip(cpu_bvh.nodes.iter()).enumerate() {
                let c_gpu = BvhNodeGpu::from_physics(c);
                assert_eq!(
                    *g, c_gpu,
                    "{label}: node[{i}] mismatch — GPU {g:?} vs CPU {c_gpu:?}"
                );
            }
        };

        // ---- Fixture 1: pile of 32 primitives in a 4x4x2 grid ----
        {
            let mut prims: Vec<BvhPrimitive> = Vec::new();
            let mut idx = 0u32;
            for xi in 0..4i64 {
                for yi in 0..4i64 {
                    for zi in 0..2i64 {
                        // spacing 3, radius 1 → mildly overlapping neighbours
                        prims.push(mk_prim(xi * 3, yi * 3, zi * 3, 1, idx));
                        idx += 1;
                    }
                }
            }
            assert_eq!(prims.len(), 32);
            check("pile 4x4x2", prims);
        }

        // ---- Fixture 2: uniform 3x3x3 grid (27 primitives) ----
        {
            let mut prims: Vec<BvhPrimitive> = Vec::new();
            let mut idx = 0u32;
            for xi in 0..3i64 {
                for yi in 0..3i64 {
                    for zi in 0..3i64 {
                        prims.push(mk_prim(xi * 4, yi * 4, zi * 4, 1, idx));
                        idx += 1;
                    }
                }
            }
            assert_eq!(prims.len(), 27);
            check("uniform grid 3x3x3", prims);
        }

        // ---- Fixture 3: degenerate all-colocated (16 primitives) ----
        // All AABBs identical → world_bounds = a single AABB → all
        // Morton codes identical → find_split hits the `first_code ==
        // last_code` branch → midpoint fallback. This is the same
        // configuration that surfaced the CPU placeholder collision
        // in ALICE-Physics `dede78c`.
        {
            let mut prims: Vec<BvhPrimitive> = Vec::new();
            for idx in 0..16u32 {
                prims.push(mk_prim(0, 0, 0, 1, idx));
            }
            assert_eq!(prims.len(), 16);
            check("degenerate colocated", prims);
        }
    }
}
