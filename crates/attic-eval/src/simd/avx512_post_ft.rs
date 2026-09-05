//! AVX-512 (F + BW + VNNI) output-transform and layer kernels, each
//! bit-identical to its [`crate::simd::scalar_post_ft`] counterpart.
//!
//! Like [`crate::simd::avx512`], this module compiles on every `x86_64` target
//! and only its call sites are gated. `avx512vnni` is required for the
//! `dpbusd`-based affine and [`fused_fc_chain`] path; the element-wise kernels
//! need only F + BW.

use std::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_storeu_si128, _mm_unpacklo_epi64, _mm256_loadu_si256,
    _mm256_storeu_si256, _mm512_castsi512_si256, _mm512_cvtepi16_epi8, _mm512_cvtepi32_epi8,
    _mm512_cvtepi32_epi64, _mm512_cvtepi64_epi8, _mm512_dpbusd_epi32, _mm512_extracti64x4_epi64,
    _mm512_loadu_si512, _mm512_max_epi16, _mm512_max_epi32, _mm512_min_epi16, _mm512_min_epi32,
    _mm512_min_epi64, _mm512_mul_epi32, _mm512_mulhi_epi16, _mm512_reduce_add_epi32,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_slli_epi16, _mm512_srai_epi32, _mm512_srai_epi64, _mm512_zextsi256_si512,
};

use crate::simd::scalar_post_ft;
use crate::types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE, HIDDEN1_DIMS,
};

const VNNI_LANES: usize = 64;
const I32_LANES_PER_M512: usize = 16;
const I16_LANES_PER_M512: usize = 32;
const EWM_HALF: usize = HIDDEN_SIZE / 2;

// The scalar constants as `u32`: the AVX-512 shift intrinsics take an IMM8.
const WEIGHT_SCALE_BITS: u32 = 6;
const SQR_SHIFT: u32 = 19;
const EWM_CLAMP_I16: i16 = 127 * 2;

// On the `i16` domain, `mulhi(sum0 << 7, sum1) == (sum0 * sum1) >> 9`.
const EWM_PRESHIFT: u32 = 7;

/// Pairwise element-wise multiply for one perspective (see the scalar
/// [`scalar_post_ft::ewm_one_perspective`] for the exact semantics).
///
/// # Safety
/// The running CPU must support `avx512f` and `avx512bw`. `half` must be
/// [`HIDDEN_SIZE`] long and `out` must be `HIDDEN_SIZE / 2` long.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
    debug_assert_eq!(half.len(), HIDDEN_SIZE);
    debug_assert_eq!(out.len(), EWM_HALF);
    // SAFETY: `half` holds 2*EWM_HALF i16 and `out` holds EWM_HALF u8.
    unsafe { ewm_one_perspective_ptr(half.as_ptr(), out.as_mut_ptr()) };
}

/// Clipped ReLU, falling back to scalar for a lane count the kernel cannot
/// tile.
///
/// # Safety
/// The running CPU must support `avx512f` and `avx512bw`. `input.len()` must
/// equal `output.len()`.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    if input.len() < I32_LANES_PER_M512 || !input.len().is_multiple_of(I32_LANES_PER_M512) {
        scalar_post_ft::clipped_relu(input, output);
        return;
    }
    // SAFETY: the target_feature gate ensures F+BW, and the length is a nonzero
    // multiple of 16.
    unsafe { clipped_relu_kernel(input, output) }
}

/// Squared clipped ReLU with the same scalar fallback as [`clipped_relu`].
///
/// # Safety
/// The running CPU must support `avx512f` and `avx512bw`. `input.len()` must
/// equal `output.len()`.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    if input.len() < I32_LANES_PER_M512 || !input.len().is_multiple_of(I32_LANES_PER_M512) {
        scalar_post_ft::sqr_clipped_relu(input, output);
        return;
    }
    // SAFETY: target_feature gate ensures F+BW; length is a nonzero multiple of 16.
    unsafe { sqr_clipped_relu_kernel(input, output) }
}

/// The whole `fc_0 → ReLU/SqrReLU → fc_1 → ReLU → fc_2 → shortcut` chain in one
/// `target_feature` body.
///
/// # Safety
/// The running CPU must support `avx512f`, `avx512bw`, and `avx512vnni`. The
/// bias/weight slices must have the SFNN-1536 layer-stack shapes (checked by the
/// `debug_assert`s below).
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn fused_fc_chain(
    transformed: &[u8; FC_0_INPUT_DIMS],
    fc_0_biases: &[i32],
    fc_0_weights: &[i8],
    fc_1_biases: &[i32],
    fc_1_weights: &[i8],
    fc_2_biases: &[i32],
    fc_2_weights: &[i8],
) -> i32 {
    debug_assert_eq!(fc_0_biases.len(), FC_0_OUTPUT_DIMS);
    debug_assert_eq!(
        fc_0_weights.len(),
        FC_0_OUTPUT_DIMS * FC_0_PADDED_INPUT_DIMS
    );
    debug_assert_eq!(fc_1_biases.len(), FC_1_OUTPUT_DIMS);
    debug_assert_eq!(
        fc_1_weights.len(),
        FC_1_OUTPUT_DIMS * FC_1_PADDED_INPUT_DIMS
    );
    debug_assert_eq!(fc_2_biases.len(), FC_2_OUTPUT_DIMS);
    debug_assert_eq!(
        fc_2_weights.len(),
        FC_2_OUTPUT_DIMS * FC_2_PADDED_INPUT_DIMS
    );

    let mut fc_0_out = [0i32; FC_0_OUTPUT_DIMS];
    // SAFETY: the target_feature gate ensures F+BW+VNNI, and FC_0_INPUT_DIMS is
    // a multiple of 64.
    unsafe {
        affine_avx512_vnni(
            &mut fc_0_out,
            fc_0_biases,
            fc_0_weights,
            transformed,
            FC_0_INPUT_DIMS,
            FC_0_PADDED_INPUT_DIMS,
        );
    }

    // Only the first HIDDEN1_DIMS lanes feed activations; the next is the
    // shortcut.
    let mut ac_0 = [0u8; HIDDEN1_DIMS];
    let mut ac_sqr_0 = [0u8; HIDDEN1_DIMS];
    scalar_post_ft::clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_0);
    scalar_post_ft::sqr_clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_sqr_0);

    // The two-lane pad keeps the VNNI padding lanes safe.
    let mut fc_1_in = [0u8; FC_1_PADDED_INPUT_DIMS];
    fc_1_in[..HIDDEN1_DIMS].copy_from_slice(&ac_sqr_0);
    fc_1_in[HIDDEN1_DIMS..2 * HIDDEN1_DIMS].copy_from_slice(&ac_0);

    let mut fc_1_out = [0i32; FC_1_OUTPUT_DIMS];
    // SAFETY: target_feature gate ensures F+BW+VNNI.
    unsafe { affine_padded32_avx512_vnni(&mut fc_1_out, fc_1_biases, fc_1_weights, &fc_1_in) };

    let mut ac_1 = [0u8; FC_1_OUTPUT_DIMS];
    // SAFETY: the target_feature gate ensures F+BW, and FC_1_OUTPUT_DIMS is a
    // multiple of 16.
    unsafe { clipped_relu_kernel(&fc_1_out, &mut ac_1) };

    let mut fc_2_out = [0i32; FC_2_OUTPUT_DIMS];
    // SAFETY: the target_feature gate ensures F+BW+VNNI, and `ac_1.len() == 32`.
    unsafe { affine_padded32_avx512_vnni(&mut fc_2_out, fc_2_biases, fc_2_weights, &ac_1) };

    fc_2_out[0].wrapping_add(fc_0_out[HIDDEN1_DIMS])
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn ewm_one_perspective_ptr(half: *const i16, out: *mut u8) {
    let zero = _mm512_setzero_si512();
    let cap = _mm512_set1_epi16(EWM_CLAMP_I16);
    let chunks = EWM_HALF / I16_LANES_PER_M512;
    for chunk in 0..chunks {
        let off = chunk * I16_LANES_PER_M512;
        // SAFETY: `chunk*32+32 <= EWM_HALF`, so both 512-bit unaligned loads
        // stay in range.
        let (in0, in1) = unsafe {
            (
                _mm512_loadu_si512(half.add(off).cast::<__m512i>()),
                _mm512_loadu_si512(half.add(off + EWM_HALF).cast::<__m512i>()),
            )
        };
        let sum0 = _mm512_max_epi16(_mm512_min_epi16(in0, cap), zero);
        let sum1 = _mm512_max_epi16(_mm512_min_epi16(in1, cap), zero);
        let shifted = _mm512_slli_epi16::<EWM_PRESHIFT>(sum0);
        let product = _mm512_mulhi_epi16(shifted, sum1);
        let bytes = _mm512_cvtepi16_epi8(product);
        // SAFETY: `chunk*32+32 <= EWM_HALF` keeps the 32-byte store in range.
        unsafe { _mm256_storeu_si256(out.add(off).cast::<__m256i>(), bytes) };
    }
}

// `in_dims % 64 == 0` only. Each partial sum stays below `i32::MAX`, so the
// reduce's wrapping add is defensive.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn affine_avx512_vnni(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    let chunks = in_dims / VNNI_LANES;
    let input_ptr = input.as_ptr();
    for (j, out_slot) in output.iter_mut().enumerate() {
        let row_ptr = weights[j * padded_in..j * padded_in + in_dims].as_ptr();
        let mut acc = _mm512_setzero_si512();
        for k in 0..chunks {
            let offset = k * VNNI_LANES;
            // SAFETY: `chunks*VNNI_LANES == in_dims`, so both loads stay within
            // the input and the weight row.
            let (a, b) = unsafe {
                (
                    _mm512_loadu_si512(input_ptr.add(offset).cast::<__m512i>()),
                    _mm512_loadu_si512(row_ptr.add(offset).cast::<__m512i>()),
                )
            };
            acc = _mm512_dpbusd_epi32(acc, a, b);
        }
        *out_slot = biases[j].wrapping_add(_mm512_reduce_add_epi32(acc));
    }
}

// `input.len() % 16 == 0` only. `cvtepi32_epi8` is loss-free because the lanes
// are clamped to `[0,127]` first.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn clipped_relu_kernel(input: &[i32], output: &mut [u8]) {
    let chunks = input.len() / I32_LANES_PER_M512;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = _mm512_setzero_si512();
    let cap = _mm512_set1_epi32(127);
    for k in 0..chunks {
        let off = k * I32_LANES_PER_M512;
        // SAFETY: `chunks*16 == input.len() == output.len()`, so the load and
        // store stay in range.
        let x = unsafe { _mm512_loadu_si512(in_ptr.add(off).cast::<__m512i>()) };
        let shifted = _mm512_srai_epi32::<WEIGHT_SCALE_BITS>(x);
        let clamped_lo = _mm512_max_epi32(shifted, zero);
        let clamped = _mm512_min_epi32(clamped_lo, cap);
        let bytes = _mm512_cvtepi32_epi8(clamped);
        // SAFETY: as the load, on the same length invariant.
        unsafe { _mm_storeu_si128(out_ptr.add(off).cast::<__m128i>(), bytes) };
    }
}

// `input.len() % 16 == 0` only. The `i64` squaring stays bit-identical to the
// scalar form across every `i32`, `i32::MIN` included.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn sqr_clipped_relu_kernel(input: &[i32], output: &mut [u8]) {
    let chunks = input.len() / I32_LANES_PER_M512;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let cap = _mm512_set1_epi64(127);
    for k in 0..chunks {
        let off = k * I32_LANES_PER_M512;
        // SAFETY: `chunks*16 == input.len() == output.len()`, so the load and
        // store stay in range.
        let x = unsafe { _mm512_loadu_si512(in_ptr.add(off).cast::<__m512i>()) };
        let lo_i32: __m256i = _mm512_castsi512_si256(x);
        let hi_i32: __m256i = _mm512_extracti64x4_epi64::<1>(x);
        let lo_i64 = _mm512_cvtepi32_epi64(lo_i32);
        let hi_i64 = _mm512_cvtepi32_epi64(hi_i32);
        let sq_lo = _mm512_mul_epi32(lo_i64, lo_i64);
        let sq_hi = _mm512_mul_epi32(hi_i64, hi_i64);
        let shr_lo = _mm512_srai_epi64::<SQR_SHIFT>(sq_lo);
        let shr_hi = _mm512_srai_epi64::<SQR_SHIFT>(sq_hi);
        let cl_lo = _mm512_min_epi64(shr_lo, cap);
        let cl_hi = _mm512_min_epi64(shr_hi, cap);
        let bytes_lo = _mm512_cvtepi64_epi8(cl_lo);
        let bytes_hi = _mm512_cvtepi64_epi8(cl_hi);
        let combined = _mm_unpacklo_epi64(bytes_lo, bytes_hi);
        // SAFETY: see the load — same length invariant.
        unsafe { _mm_storeu_si128(out_ptr.add(off).cast::<__m128i>(), combined) };
    }
}

// The VNNI affine for fc_1/fc_2, whose `in_dims` is 32: `zextsi256_si512`
// zero-extends, so one `dpbusd` suffices.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn affine_padded32_avx512_vnni(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8; 32],
) {
    debug_assert_eq!(output.len(), biases.len());
    debug_assert_eq!(weights.len(), output.len() * 32);

    // SAFETY: `input` is a `&[u8; 32]`, so the 256-bit load reads exactly it.
    let in_full =
        unsafe { _mm512_zextsi256_si512(_mm256_loadu_si256(input.as_ptr().cast::<__m256i>())) };

    for (j, out_slot) in output.iter_mut().enumerate() {
        let row_ptr = weights[j * 32..j * 32 + 32].as_ptr();
        // SAFETY: the weight row spans 32 bytes, on the same argument.
        let row_full =
            unsafe { _mm512_zextsi256_si512(_mm256_loadu_si256(row_ptr.cast::<__m256i>())) };
        let acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), in_full, row_full);
        *out_slot = biases[j].wrapping_add(_mm512_reduce_add_epi32(acc));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligned::Aligned64;
    use crate::types::{
        FC_1_INPUT_DIMS, FC_2_INPUT_DIMS, LAYER_STACKS, NetDims, NetHeader, NetworkStack,
        NnueNetwork, NnueNetworkBuilder,
    };

    /// A single-stack network whose lone stack the caller fills through `fill`.
    /// Returned by value, so that its arena outlives the borrowed stack.
    fn stack_net(fill: impl FnOnce(&mut NnueNetworkBuilder)) -> NnueNetwork {
        let dims = NetDims {
            layer_stacks: 1,
            num_features: 1,
            ..NetDims::STANDARD
        };
        let header = NetHeader {
            version: 0,
            hash: 0,
            arch_id: String::new(),
        };
        let mut b = NnueNetworkBuilder::with_dims(header, [0u8; 32], &dims);
        fill(&mut b);
        b.build()
    }

    macro_rules! require_avx512bw {
        () => {
            if !(std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw"))
            {
                eprintln!("skipping AVX-512 parity test: avx512f/avx512bw unavailable");
                return;
            }
        };
    }

    macro_rules! require_vnni {
        () => {
            if !(std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni"))
            {
                eprintln!("skipping AVX-512+VNNI parity test: features unavailable");
                return;
            }
        };
    }

    fn seeded_i32_inputs(len: usize, seed: u32) -> Box<[i32]> {
        let mut v = vec![0i32; len].into_boxed_slice();
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = match (i + seed as usize) % 11 {
                0 => 0,
                1 => i32::MAX,
                2 => i32::MIN,
                3 => 8_127,
                4 => 8_128,
                5 => -1,
                6 => 100_000,
                7 => -100_000,
                8 => 725,
                9 => -725,
                _ => (i as i32).wrapping_mul(257).wrapping_sub(seed as i32 * 13),
            };
        }
        v
    }

    fn seeded_weights(len: usize, seed: u32) -> Box<[i8]> {
        let mut v = vec![0i8; len].into_boxed_slice();
        for (i, slot) in v.iter_mut().enumerate() {
            let raw = (i as u32).wrapping_mul(7).wrapping_add(seed) % 251;
            *slot = (raw as i32 - 125) as i8;
        }
        v
    }

    fn seeded_biases(len: usize, seed: u32) -> Box<[i32]> {
        let mut v = vec![0i32; len].into_boxed_slice();
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = (i as i32) * 17 - 9 * (seed as i32);
        }
        v
    }

    fn seeded_input(len: usize, seed: u32) -> Box<[u8]> {
        let mut v = vec![0u8; len].into_boxed_slice();
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = ((i as u32).wrapping_mul(11).wrapping_add(seed) % 256) as u8;
        }
        v
    }

    #[test]
    fn affine_matches_scalar_for_fc_0_shape() {
        require_vnni!();
        let (in_dims, padded_in, out_dims, seed) = (1536usize, 1536usize, 16usize, 41u32);
        let weights: Aligned64<i8> = seeded_weights(out_dims * padded_in, seed)
            .iter()
            .copied()
            .collect();
        let biases = seeded_biases(out_dims, seed);
        let input: Aligned64<u8> = seeded_input(padded_in, seed).iter().copied().collect();

        let mut avx = vec![0i32; out_dims].into_boxed_slice();
        // SAFETY: guarded by `require_vnni!`; in_dims is a multiple of 64.
        unsafe { affine_avx512_vnni(&mut avx, &biases, &weights, &input, in_dims, padded_in) };

        let mut sca = vec![0i32; out_dims].into_boxed_slice();
        scalar_post_ft::affine(&mut sca, &biases, &weights, &input, in_dims, padded_in);

        assert_eq!(avx, sca);
    }

    fn affine_padded32_parity(out_dims: usize, in_dims: usize, seed: u32) {
        let weights: Aligned64<i8> = seeded_weights(out_dims * 32, seed)
            .iter()
            .copied()
            .collect();
        let biases = seeded_biases(out_dims, seed);

        // The padding lanes are zero, as in the fused chain.
        let mut input = [0u8; 32];
        for (i, slot) in input.iter_mut().enumerate().take(in_dims) {
            *slot = ((i as u32).wrapping_mul(11).wrapping_add(seed) % 256) as u8;
        }
        // Zeroing the padded weight columns is what makes the scalar form, which
        // reads `in_dims`, agree with the SIMD one, which reads all 32.
        let mut weights_padded = weights;
        for j in 0..out_dims {
            for c in in_dims..32 {
                weights_padded[j * 32 + c] = 0;
            }
        }

        let mut avx = vec![0i32; out_dims].into_boxed_slice();
        // SAFETY: guarded by the caller's `require_vnni!`.
        unsafe { affine_padded32_avx512_vnni(&mut avx, &biases, &weights_padded, &input) };

        let mut sca = vec![0i32; out_dims].into_boxed_slice();
        scalar_post_ft::affine(&mut sca, &biases, &weights_padded, &input, in_dims, 32);

        assert_eq!(
            avx, sca,
            "padded32 affine mismatch at ({in_dims}->{out_dims})"
        );
    }

    #[test]
    fn affine_padded32_matches_scalar_for_fc_1_and_fc_2_shapes() {
        require_vnni!();
        affine_padded32_parity(FC_1_OUTPUT_DIMS, FC_1_INPUT_DIMS, 53);
        affine_padded32_parity(FC_2_OUTPUT_DIMS, FC_2_INPUT_DIMS, 67);
    }

    fn clipped_relu_parity(input: &[i32]) {
        let mut avx = vec![0u8; input.len()];
        let mut sca = vec![0u8; input.len()];
        // SAFETY: guarded by the caller's `require_avx512bw!`.
        unsafe { clipped_relu(input, &mut avx) };
        scalar_post_ft::clipped_relu(input, &mut sca);
        assert_eq!(avx, sca, "clipped_relu mismatch at len={}", input.len());
    }

    fn sqr_clipped_relu_parity(input: &[i32]) {
        let mut avx = vec![0u8; input.len()];
        let mut sca = vec![0u8; input.len()];
        // SAFETY: guarded by the caller's `require_avx512bw!`.
        unsafe { sqr_clipped_relu(input, &mut avx) };
        scalar_post_ft::sqr_clipped_relu(input, &mut sca);
        assert_eq!(avx, sca, "sqr_clipped_relu mismatch at len={}", input.len());
    }

    #[test]
    fn clipped_relu_matches_scalar_on_boundary_values() {
        require_avx512bw!();
        let inputs: [i32; 16] = [
            -1_000_000,
            -64,
            -1,
            0,
            1,
            63,
            64,
            8_000,
            8_127,
            8_128,
            i32::MAX,
            i32::MIN,
            2,
            -50,
            1_000,
            -1_000,
        ];
        clipped_relu_parity(&inputs);
    }

    #[test]
    fn clipped_relu_falls_back_to_scalar_for_short_input() {
        require_avx512bw!();
        let inputs = seeded_i32_inputs(15, 23);
        clipped_relu_parity(&inputs);
    }

    // `i32::MIN` and `MAX` witness the `i64`-squaring path, where
    // `mullo_epi32` would wrap.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn sqr_clipped_relu_matches_scalar_on_boundary_values() {
        require_avx512bw!();
        let inputs: [i32; 16] = [
            0,
            1,
            724,
            725,
            8_191,
            8_192,
            100_000,
            -724,
            -725,
            -100_000,
            i32::MAX,
            i32::MIN,
            -1,
            5_000_000,
            -5_000_000,
            12_345,
        ];
        sqr_clipped_relu_parity(&inputs);
    }

    #[test]
    fn sqr_clipped_relu_falls_back_to_scalar_for_short_input() {
        require_avx512bw!();
        let inputs = seeded_i32_inputs(15, 59);
        sqr_clipped_relu_parity(&inputs);
    }

    fn seeded_half(seed: u32) -> [i16; HIDDEN_SIZE] {
        let mut half = [0i16; HIDDEN_SIZE];
        for (i, slot) in half.iter_mut().enumerate() {
            *slot = match (i + seed as usize) % 13 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => 0,
                3 => -1,
                4 => 254,
                5 => 255,
                6 => 253,
                7 => -10_000,
                8 => 30_000,
                9 => 100,
                10 => 200,
                11 => 1,
                _ => (i as i32).wrapping_mul(31).wrapping_sub(seed as i32) as i16,
            };
        }
        half
    }

    #[test]
    fn ewm_one_perspective_matches_scalar() {
        require_avx512bw!();
        const HALF: usize = HIDDEN_SIZE / 2;
        for seed in [1u32, 7, 13, 100] {
            let half = seeded_half(seed);
            let mut avx = [0u8; HALF];
            let mut sca = [0u8; HALF];
            // SAFETY: guarded by `require_avx512bw!`; slice lengths match.
            unsafe { ewm_one_perspective(&half, &mut avx) };
            scalar_post_ft::ewm_one_perspective(&half, &mut sca);
            assert_eq!(avx, sca, "ewm mismatch at seed {seed}");
        }
    }

    fn seeded_net(seed: u32) -> NnueNetwork {
        stack_net(|b| {
            for (i, s) in b.fc_0_biases_mut(0).iter_mut().enumerate() {
                *s = (i as i32).wrapping_mul(11).wrapping_sub(seed as i32 * 5);
            }
            for (i, s) in b.fc_0_weights_mut(0).iter_mut().enumerate() {
                let raw = (i as u32).wrapping_mul(7).wrapping_add(seed) % 251;
                *s = (raw as i32 - 125) as i8;
            }
            for (i, s) in b.fc_1_biases_mut(0).iter_mut().enumerate() {
                *s = (i as i32).wrapping_mul(17).wrapping_sub(seed as i32 * 3);
            }
            for (i, s) in b.fc_1_weights_mut(0).iter_mut().enumerate() {
                let raw = (i as u32)
                    .wrapping_mul(13)
                    .wrapping_add(seed.wrapping_mul(31))
                    % 251;
                *s = (raw as i32 - 125) as i8;
            }
            for (i, s) in b.fc_2_biases_mut(0).iter_mut().enumerate() {
                *s = (i as i32).wrapping_mul(23).wrapping_add(seed as i32 * 7);
            }
            for (i, s) in b.fc_2_weights_mut(0).iter_mut().enumerate() {
                let raw = (i as u32)
                    .wrapping_mul(19)
                    .wrapping_add(seed.wrapping_mul(53))
                    % 251;
                *s = (raw as i32 - 125) as i8;
            }
        })
    }

    fn per_layer_reference_score(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
        let mut fc_0_out = [0i32; FC_0_OUTPUT_DIMS];
        scalar_post_ft::affine(
            &mut fc_0_out,
            &stack.fc_0_biases,
            &stack.fc_0_weights,
            transformed,
            FC_0_INPUT_DIMS,
            FC_0_PADDED_INPUT_DIMS,
        );
        let mut ac_0 = [0u8; HIDDEN1_DIMS];
        scalar_post_ft::clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_0);
        let mut ac_sqr_0 = [0u8; HIDDEN1_DIMS];
        scalar_post_ft::sqr_clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_sqr_0);

        let mut fc_1_in = [0u8; FC_1_PADDED_INPUT_DIMS];
        fc_1_in[..HIDDEN1_DIMS].copy_from_slice(&ac_sqr_0);
        fc_1_in[HIDDEN1_DIMS..2 * HIDDEN1_DIMS].copy_from_slice(&ac_0);

        let mut fc_1_out = [0i32; FC_1_OUTPUT_DIMS];
        scalar_post_ft::affine(
            &mut fc_1_out,
            &stack.fc_1_biases,
            &stack.fc_1_weights,
            &fc_1_in,
            FC_1_INPUT_DIMS,
            FC_1_PADDED_INPUT_DIMS,
        );
        let mut ac_1 = [0u8; FC_1_OUTPUT_DIMS];
        scalar_post_ft::clipped_relu(&fc_1_out, &mut ac_1);

        let mut fc_2_out = [0i32; FC_2_OUTPUT_DIMS];
        scalar_post_ft::affine(
            &mut fc_2_out,
            &stack.fc_2_biases,
            &stack.fc_2_weights,
            &ac_1,
            FC_2_INPUT_DIMS,
            FC_2_PADDED_INPUT_DIMS,
        );
        fc_2_out[0].wrapping_add(fc_0_out[HIDDEN1_DIMS])
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn fused_fc_chain_matches_per_layer_flow_for_all_buckets() {
        require_vnni!();
        for bucket in 0..LAYER_STACKS {
            let seed = 100 + bucket as u32;
            let net = seeded_net(seed);
            let stack = &net.stacks[0];
            let mut transformed = [0u8; FC_0_INPUT_DIMS];
            for (i, slot) in transformed.iter_mut().enumerate() {
                *slot = ((i as u32).wrapping_mul(11).wrapping_add(seed) % 256) as u8;
            }

            // SAFETY: guarded by `require_vnni!`.
            let fused = unsafe {
                fused_fc_chain(
                    &transformed,
                    &stack.fc_0_biases,
                    &stack.fc_0_weights,
                    &stack.fc_1_biases,
                    &stack.fc_1_weights,
                    &stack.fc_2_biases,
                    &stack.fc_2_weights,
                )
            };

            let per_layer = per_layer_reference_score(&transformed, stack);

            assert_eq!(
                fused, per_layer,
                "fused vs. per-layer mismatch at bucket {bucket} (seed {seed})"
            );
        }
    }

    #[test]
    fn fused_fc_chain_preserves_shortcut() {
        require_vnni!();
        // Only the shortcut bias is nonzero, so it alone must reach the output.
        const SHORTCUT_K: i32 = 12_345;
        let net = stack_net(|b| {
            b.fc_0_biases_mut(0)[HIDDEN1_DIMS] = SHORTCUT_K;
        });
        let stack = &net.stacks[0];
        let transformed = [42u8; FC_0_INPUT_DIMS];

        // SAFETY: guarded by `require_vnni!`.
        let fused = unsafe {
            fused_fc_chain(
                &transformed,
                &stack.fc_0_biases,
                &stack.fc_0_weights,
                &stack.fc_1_biases,
                &stack.fc_1_weights,
                &stack.fc_2_biases,
                &stack.fc_2_weights,
            )
        };
        assert_eq!(fused, SHORTCUT_K);
    }
}
