//! Scalar output-transform and layer kernels (affine / clipped-ReLU /
//! squared-clipped-ReLU / element-wise multiply).
//!
//! Ported from the read-only Rust NNUE reference implementation's
//! `scalar_post_ft.rs`. These are the always-available
//! baseline the AVX-512+VNNI kernels in [`crate::simd::avx512_post_ft`] must
//! match bit-for-bit.
//!
//! The reference exposes a whole-`Accumulator` `transformer_ewm`; because this
//! crate's `Accumulator` keeps the two perspectives in colour-indexed buffers
//! and does the side-to-move reordering itself, the output transform is factored
//! here as the per-perspective [`ewm_one_perspective`] helper the caller invokes
//! twice.

use crate::types::HIDDEN_SIZE;

/// Right-shift applied to the affine output before the clipped ReLU
/// (`kWeightScaleBits`).
const WEIGHT_SCALE_BITS: i32 = 6;

/// Right-shift applied to the squared affine output (`2 * WEIGHT_SCALE_BITS + 7`).
const SQR_SHIFT: i32 = 2 * WEIGHT_SCALE_BITS + 7;

/// FT biases and weights are scaled ×2 at load time, so the post-scale
/// accumulator bound is `[0, 254]`, not `[0, 127]`.
const EWM_CLAMP: i32 = 127 * 2;

/// Right-shift applied to the lane product in the output transform (`/ 512`).
const EWM_SHIFT: i32 = 9;

/// Integer affine transform: `output[j] = bias[j] + Σ weights[j][i] * input[i]`.
///
/// Weights are row-major `[output][padded_in]`; only the first `in_dims`
/// columns of each row are consumed (padding lanes are never read).
pub fn affine(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    debug_assert_eq!(output.len(), biases.len());
    debug_assert_eq!(weights.len(), output.len() * padded_in);
    debug_assert!(in_dims <= padded_in);
    debug_assert!(input.len() >= in_dims);

    for (j, out_slot) in output.iter_mut().enumerate() {
        let row = j * padded_in;
        let mut acc = biases[j];
        for i in 0..in_dims {
            acc = acc.wrapping_add(weights[row + i] as i32 * input[i] as i32);
        }
        *out_slot = acc;
    }
}

/// Clipped ReLU: `output = clamp(input >> WEIGHT_SCALE_BITS, 0, 127)`.
pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = (inp >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
    }
}

/// Squared clipped ReLU: `output = clamp(input² >> SQR_SHIFT, 0, 127)`. The
/// `i64` intermediate avoids overflow on the `i32` square.
pub fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let sq = (*inp as i64) * (*inp as i64);
        *out = (sq >> SQR_SHIFT).clamp(0, 127) as u8;
    }
}

/// Pairwise element-wise multiply for one perspective: lane `j` pairs with lane
/// `j + HIDDEN_SIZE/2`, each clamped to `[0, EWM_CLAMP]`, multiplied, then
/// shifted right by [`EWM_SHIFT`]. Reads `HIDDEN_SIZE` `i16`s, writes
/// `HIDDEN_SIZE/2` bytes.
pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
    const HALF: usize = HIDDEN_SIZE / 2;
    debug_assert_eq!(half.len(), HIDDEN_SIZE);
    debug_assert_eq!(out.len(), HALF);
    for j in 0..HALF {
        let s0 = (half[j] as i32).clamp(0, EWM_CLAMP);
        let s1 = (half[j + HALF] as i32).clamp(0, EWM_CLAMP);
        out[j] = ((s0 * s1) >> EWM_SHIFT) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_relu_matches_reference_on_boundary_values() {
        let inputs = [
            -1_000_000i32,
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
        ];
        let mut out = [0u8; 11];
        clipped_relu(&inputs, &mut out);
        assert_eq!(out, [0u8, 0, 0, 0, 0, 0, 1, 125, 126, 127, 127]);
    }

    #[test]
    fn sqr_clipped_relu_matches_reference_formula() {
        let inputs = [
            0i32,
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
        ];
        let mut out = [0u8; 12];
        sqr_clipped_relu(&inputs, &mut out);
        assert_eq!(out, [0u8, 0, 0, 1, 127, 127, 127, 0, 1, 127, 127, 127]);
    }

    #[test]
    fn affine_accumulates_bias_plus_dot_product() {
        let biases = [10i32, -5];
        let weights = [2i8, 3, 99, -1, 4, 88];
        let input = [5u8, 7, 0];
        let mut out = [0i32; 2];
        affine(&mut out, &biases, &weights, &input, 2, 3);
        assert_eq!(out[0], 41);
        assert_eq!(out[1], 18);
    }

    #[test]
    fn ewm_one_perspective_clamps_and_shifts() {
        const HALF: usize = HIDDEN_SIZE / 2;
        let mut half = [0i16; HIDDEN_SIZE];
        // lane 0: (200*100)>>9 = 39.
        half[0] = 200;
        half[HALF] = 100;
        // lane 1: negative s0 clamps to 0.
        half[1] = -1_000;
        half[HALF + 1] = 200;
        // lane 2: both above the ceiling clamp to 254 -> (254*254)>>9 = 126.
        half[2] = 30_000;
        half[HALF + 2] = 30_000;

        let mut out = [0u8; HALF];
        ewm_one_perspective(&half, &mut out);
        assert_eq!(out[0], 39);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 126);
        assert!(out[3..].iter().all(|&x| x == 0));
    }
}
