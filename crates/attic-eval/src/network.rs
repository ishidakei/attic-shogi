//! Layer-stack forward pass, bucket selection, and the public `evaluate` entry
//! point, ported from `sfnn-1536.h::Propagate` and `evaluate_nnue.cpp`'s
//! `ComputeScore` / `stack_index_for_nnue`. The accumulator lives in
//! [`crate::transformer`], the kernels in [`crate::simd`].
//!
//! ## Layer naming
//!
//! Prose in this crate follows the upstream naming: the **FT layer** is the
//! feature transformer, and **L1 / L2 / L3** are the dense layers after it. The
//! FT layer is never called "L1". In the identifiers used here, L1 = `fc_0`,
//! L2 = `fc_1`, L3 = `fc_2`.
//!
//! ## Forward pass
//!
//! L1 is a sparse-input affine with 16 outputs. The first 15 feed two
//! activations, whose results concatenate `[sqr(15) | relu(15)]` into L2's
//! input; L2 (→ 32) is followed by a clipped ReLU, then L3 (→ 1). The 16th L1
//! output is raw, pre-ReLU, and is added to L3's output as the shortcut term.
//! The result is divided by the live [`fv_scale`] to produce the score.

use std::sync::atomic::{AtomicI32, Ordering};

use attic_state::{Color, Position, Square};

use crate::features::king_square;
use crate::simd::post_ft_kernel;
use crate::transformer::{Accumulator, FT_OUTPUT_DIMS};
use crate::types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS,
    HIDDEN1_DIMS, LAYER_STACKS, NetworkStack, NnueNetwork,
};

/// The reference default fixed-point scale (`evaluate_nnue.cpp`), and the USI
/// option's default.
pub const FV_SCALE_DEFAULT: i32 = 16;

/// The live fixed-point scale applied to the network output, mirroring the
/// reference's mutable global `NNUE::FV_SCALE` (`evaluate_nnue.cpp`).
static FV_SCALE: AtomicI32 = AtomicI32::new(FV_SCALE_DEFAULT);

/// The current fixed-point scale (the reference live global `NNUE::FV_SCALE`).
pub fn fv_scale() -> i32 {
    FV_SCALE.load(Ordering::Relaxed)
}

/// Set the live fixed-point scale (`evaluate_nnue.cpp`). The next
/// [`evaluate`] divides by it.
pub fn set_fv_scale(scale: i32) {
    FV_SCALE.store(scale, Ordering::Relaxed);
}

/// `k*ToIndex` tables from `stack_index_for_nnue`: the own-king rank contributes
/// the coarse third `{0,3,6}`, the enemy-king rank the fine third `{0,1,2}`.
const F_RANK_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
const E_RANK_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

/// Select the layer-stack (bucket) index for `pos`.
///
/// `stack_index_for_nnue`: the own-king rank is taken in the side-to-move's
/// forward frame and the enemy-king rank in the opposite one, then combined
/// through the `k*ToIndex` tables and clamped to `[0, LAYER_STACKS)`.
///
/// # Panics
/// Panics if either king is missing.
pub fn layer_stack_index(pos: &Position) -> usize {
    let stm = pos.side_to_move();
    let f_king = king_square(pos, stm);
    let e_king = king_square(pos, stm.flip());

    // White flips ranks, so that both colours reason in an own-side-forward
    // frame.
    let flip = |sq: Square| (Square::RANKS - 1 - sq.rank()) as usize;
    let f_rank = match stm {
        Color::Black => f_king.rank() as usize,
        Color::White => flip(f_king),
    };
    let e_rank = match stm {
        Color::Black => flip(e_king),
        Color::White => e_king.rank() as usize,
    };

    (F_RANK_TO_INDEX[f_rank] + E_RANK_TO_INDEX[e_rank]).min(LAYER_STACKS - 1)
}

/// Static NNUE evaluation of `pos`, from the side-to-move perspective and
/// positive when that side is better (`ComputeScore(pos, refresh=true)`).
/// Rebuilds both accumulator halves from scratch.
///
/// The network is passed explicitly: there is no global registry, so the
/// Evaluation layer never reaches up to Protocol for it.
///
/// # Panics
/// Panics if `pos` is missing either king (via feature extraction and bucket
/// selection).
pub fn evaluate(net: &NnueNetwork, pos: &Position) -> i32 {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    evaluate_with(net, &acc, pos)
}

/// Static NNUE evaluation of `pos` from an already-computed accumulator, which
/// must be current for `pos` on both perspectives. The search's entry point,
/// threading an incrementally-updated accumulator through the tree.
///
/// # Panics
/// Panics if `pos` is missing either king (via bucket selection).
pub fn evaluate_with(net: &NnueNetwork, acc: &Accumulator, pos: &Position) -> i32 {
    let bucket = layer_stack_index(pos);
    debug_assert!(bucket < net.stacks.len());

    let mut transformed = [0u8; FT_OUTPUT_DIMS];
    acc.output_transform(pos.side_to_move(), &mut transformed);

    let score = per_layer_flow(&transformed, &net.stacks[bucket]);
    // The single FV_SCALE consumption site (`evaluate_nnue.cpp`).
    score / fv_scale()
}

/// Layer-stack forward pass over the transformed byte buffer, returning the raw
/// network output before [`fv_scale`].
///
/// A build with AVX-512 F + BW + VNNI runs the whole chain through the fused
/// kernel; otherwise the per-layer flow below runs. The two are byte-for-byte
/// equivalent.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
fn per_layer_flow(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
    // Scoped to this arm: a build without the features above compiles the arm
    // below instead, where the kernel module has no caller.
    use crate::simd::avx512_post_ft;

    // SAFETY: this arm is compiled only into a build enabling avx512f +
    // avx512bw + avx512vnni, exactly the features `fused_fc_chain` names, and
    // such a build only ever runs on a host providing them: `-C
    // target-cpu=native` means build and run happen on the same machine.
    unsafe {
        avx512_post_ft::fused_fc_chain(
            transformed,
            &stack.fc_0_biases,
            &stack.fc_0_weights,
            &stack.fc_1_biases,
            &stack.fc_1_weights,
            &stack.fc_2_biases,
            &stack.fc_2_weights,
        )
    }
}

/// Layer-stack forward pass, non-VNNI arm.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
)))]
fn per_layer_flow(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
    per_layer_flow_unfused(transformed, stack)
}

/// The unfused layer-stack forward pass. Compiled unconditionally so that the
/// equivalence tests can hold it against the fused chain, which leaves it
/// caller-less outside `cfg(test)` on a VNNI build.
#[allow(dead_code)]
fn per_layer_flow_unfused(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
    let mut fc_0_out = [0i32; FC_0_OUTPUT_DIMS];
    post_ft_kernel::affine(
        &mut fc_0_out,
        &stack.fc_0_biases,
        &stack.fc_0_weights,
        transformed,
        FC_0_INPUT_DIMS,
        FC_0_PADDED_INPUT_DIMS,
    );

    // The first HIDDEN1_DIMS outputs feed both activations; the next is reserved
    // for the post-fc_2 shortcut.
    let mut ac_0 = [0u8; HIDDEN1_DIMS];
    post_ft_kernel::clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_0);
    let mut ac_sqr_0 = [0u8; HIDDEN1_DIMS];
    post_ft_kernel::sqr_clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_sqr_0);

    // fc_1 input layout: [SqrClippedReLU(15) | ClippedReLU(15) | 0_pad(2)].
    let mut fc_1_in = [0u8; FC_1_PADDED_INPUT_DIMS];
    fc_1_in[..HIDDEN1_DIMS].copy_from_slice(&ac_sqr_0);
    fc_1_in[HIDDEN1_DIMS..2 * HIDDEN1_DIMS].copy_from_slice(&ac_0);

    let mut fc_1_out = [0i32; FC_1_OUTPUT_DIMS];
    post_ft_kernel::affine(
        &mut fc_1_out,
        &stack.fc_1_biases,
        &stack.fc_1_weights,
        &fc_1_in,
        FC_1_INPUT_DIMS,
        FC_1_PADDED_INPUT_DIMS,
    );

    let mut ac_1 = [0u8; FC_1_OUTPUT_DIMS];
    post_ft_kernel::clipped_relu(&fc_1_out, &mut ac_1);

    let mut fc_2_out = [0i32; FC_2_OUTPUT_DIMS];
    post_ft_kernel::affine(
        &mut fc_2_out,
        &stack.fc_2_biases,
        &stack.fc_2_weights,
        &ac_1,
        FC_2_INPUT_DIMS,
        FC_2_PADDED_INPUT_DIMS,
    );

    // The shortcut output is raw, pre-ReLU, and can be negative.
    fc_2_out[0].wrapping_add(fc_0_out[HIDDEN1_DIMS])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NetDims, NetHeader, NnueNetworkBuilder};
    use attic_state::parse_sfen;

    fn synthetic_header() -> NetHeader {
        NetHeader {
            version: 0,
            hash: 0,
            arch_id: "synthetic".to_string(),
        }
    }

    /// A single-stack builder with the standard FC dims but a tiny feature
    /// transformer, so that the FC forward pass runs unchanged without a 215 MiB
    /// allocation.
    fn builder_1stack_std() -> NnueNetworkBuilder {
        let dims = NetDims {
            layer_stacks: 1,
            num_features: 1,
            ..NetDims::STANDARD
        };
        NnueNetworkBuilder::with_dims(synthetic_header(), [0u8; 32], &dims)
    }

    /// A single-stack, all-zero network with the standard FC dims (tiny FT).
    fn zero_net_1stack() -> NnueNetwork {
        builder_1stack_std().build()
    }

    #[test]
    fn replicate_is_byte_identical_deep_copy() {
        // A replica must be byte-for-byte identical to its source in a freshly
        // allocated arena, so that the two evaluate any position the same while
        // living on different NUMA nodes.
        let dims = NetDims {
            layer_stacks: 1,
            num_features: 1,
            hidden_size: 5,
            ..NetDims::STANDARD
        };
        let mut builder = NnueNetworkBuilder::with_dims(synthetic_header(), [0xAB; 32], &dims);
        builder.fc_0_biases_mut(0)[0] = 123;
        builder.fc_0_weights_mut(0)[1] = -7;
        builder.fc_1_biases_mut(0)[2] = 55;
        builder.fc_2_weights_mut(0)[0] = 9;
        builder.ft_biases_mut()[0] = 4242;
        builder
            .ft_weights_mut()
            .copy_from_slice(&[1i16, 2, 3, -4, 5]);
        let net = builder.build();

        let copy = net.replicate();

        assert_eq!(&*copy.ft_biases, &*net.ft_biases);
        assert_eq!(&*copy.ft_weights, &*net.ft_weights);
        assert_eq!(copy.sha256, net.sha256);
        assert_eq!(copy.header.arch_id, net.header.arch_id);
        assert_eq!(copy.stacks.len(), net.stacks.len());
        assert_eq!(&*copy.stacks[0].fc_0_biases, &*net.stacks[0].fc_0_biases);
        assert_eq!(&*copy.stacks[0].fc_0_weights, &*net.stacks[0].fc_0_weights);
        assert_eq!(&*copy.stacks[0].fc_2_weights, &*net.stacks[0].fc_2_weights);
        assert_ne!(copy.ft_weights.as_ptr(), net.ft_weights.as_ptr());
        assert_eq!(net.allocation_disclosure().0, 1);
        assert_eq!(copy.allocation_disclosure().0, 1);
    }

    #[test]
    fn zero_network_evaluates_to_zero() {
        let net = zero_net_1stack();
        let transformed = [0u8; FC_0_INPUT_DIMS];
        assert_eq!(per_layer_flow(&transformed, &net.stacks[0]), 0);
    }

    #[test]
    fn forward_wires_shortcut_and_chain() {
        // The reference's `classic_chain_with_ewm_and_shortcut` wiring:
        // `transformed[0]=39` drives `fc_0_out[0]=50+3*39=167` and the shortcut
        // `fc_0_out[15]=100+4*39=256`; `ac_0[0]=167>>6=2` and `ac_sqr_0` is all
        // zero, so `fc_1_out[0]=30+7*2=44`, `ac_1[0]=0`, and the output is
        // `1000 + 256`.
        let mut b = builder_1stack_std();
        b.fc_0_biases_mut(0)[0] = 50;
        b.fc_0_weights_mut(0)[0] = 3;
        b.fc_0_biases_mut(0)[HIDDEN1_DIMS] = 100;
        b.fc_0_weights_mut(0)[HIDDEN1_DIMS * FC_0_PADDED_INPUT_DIMS] = 4;
        b.fc_1_biases_mut(0)[0] = 30;
        b.fc_1_weights_mut(0)[15] = 7;
        b.fc_2_biases_mut(0)[0] = 1_000;
        let net = b.build();
        let stack = &net.stacks[0];

        let mut transformed = [0u8; FC_0_INPUT_DIMS];
        transformed[0] = 39;

        assert_eq!(per_layer_flow(&transformed, stack), 1_256);
        assert_eq!(per_layer_flow(&transformed, stack) / FV_SCALE_DEFAULT, 78);
    }

    #[test]
    fn selected_forward_path_matches_the_unfused_reference() {
        // Whichever path this build compiled must agree bit-for-bit with the
        // per-layer flow. On a scalar build the two are the same code and the
        // check is a tautology, but it keeps the unfused form exercised.
        let mut b = builder_1stack_std();
        for (i, w) in b.fc_0_weights_mut(0).iter_mut().enumerate() {
            *w = ((i as i32 * 7) % 61 - 30) as i8;
        }
        for (i, bias) in b.fc_0_biases_mut(0).iter_mut().enumerate() {
            *bias = (i as i32 * 37) % 4_001 - 2_000;
        }
        for (i, w) in b.fc_1_weights_mut(0).iter_mut().enumerate() {
            *w = ((i as i32 * 11) % 53 - 26) as i8;
        }
        for (i, bias) in b.fc_1_biases_mut(0).iter_mut().enumerate() {
            *bias = (i as i32 * 101) % 3_001 - 1_500;
        }
        for (i, w) in b.fc_2_weights_mut(0).iter_mut().enumerate() {
            *w = ((i as i32 * 13) % 47 - 23) as i8;
        }
        for (i, bias) in b.fc_2_biases_mut(0).iter_mut().enumerate() {
            *bias = (i as i32 * 211) % 2_001 - 1_000;
        }
        let net = b.build();
        let stack = &net.stacks[0];

        for seed in [0u32, 1, 97] {
            let mut transformed = [0u8; FC_0_INPUT_DIMS];
            for (i, slot) in transformed.iter_mut().enumerate() {
                *slot = ((i as u32).wrapping_mul(11).wrapping_add(seed) % 256) as u8;
            }
            assert_eq!(
                per_layer_flow(&transformed, stack),
                per_layer_flow_unfused(&transformed, stack),
                "selected path disagrees with the unfused reference (seed {seed})",
            );
        }
    }

    #[test]
    fn evaluate_zero_network_is_zero_for_both_sides() {
        // Full-size zeroed FT weights, so that refresh's feature-column indexing
        // stays in bounds; the positions below are sparse enough that only a few
        // calloc pages are ever touched.
        let net = NnueNetworkBuilder::new(synthetic_header(), [0u8; 32]).build();
        for sfen in ["8K/9/9/9/9/9/9/9/8k b - 1", "8K/9/9/9/9/9/9/9/8k w - 1"] {
            let pos = parse_sfen(sfen).unwrap();
            assert_eq!(evaluate(&net, &pos), 0, "sfen `{sfen}`");
        }
    }

    /// An SFEN-parsing oracle for the bucket index, sharing no code with
    /// [`layer_stack_index`].
    fn oracle_bucket(sfen: &str) -> usize {
        let mut parts = sfen.split_whitespace();
        let board = parts.next().unwrap();
        let stm = parts.next().unwrap();

        let mut bk = None;
        let mut wk = None;
        for (rank_idx, rank_str) in board.split('/').enumerate() {
            let mut chars = rank_str.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '1'..='9' => {}
                    '+' => {
                        let _ = chars.next();
                    }
                    'K' => bk = Some(rank_idx),
                    'k' => wk = Some(rank_idx),
                    _ => {}
                }
            }
        }
        let bk = bk.unwrap();
        let wk = wk.unwrap();
        let (f_rank, e_rank) = match stm {
            "b" => (bk, 8 - wk),
            "w" => (8 - wk, bk),
            _ => unreachable!(),
        };
        (F_RANK_TO_INDEX[f_rank] + E_RANK_TO_INDEX[e_rank]).min(LAYER_STACKS - 1)
    }

    const BUCKET_FIXTURES: &[(usize, &str)] = &[
        (0, "8K/9/9/9/9/9/9/9/8k b - 1"),
        (1, "8K/9/9/9/9/8k/9/9/9 b - 1"),
        (2, "8K/9/k8/9/9/9/9/9/9 b - 1"),
        (3, "9/9/9/8K/9/9/9/9/8k b - 1"),
        (4, "9/9/9/8K/9/8k/9/9/9 b - 1"),
        (5, "9/9/k8/8K/9/9/9/9/9 b - 1"),
        (6, "9/9/9/9/9/9/8K/9/8k b - 1"),
        (7, "9/9/9/9/9/k8/8K/9/9 b - 1"),
        (8, "9/9/8k/9/9/9/8K/9/9 b - 1"),
    ];

    #[test]
    fn layer_stack_index_matches_oracle_on_each_bucket() {
        for (expected, sfen) in BUCKET_FIXTURES {
            let oracle = oracle_bucket(sfen);
            assert_eq!(oracle, *expected, "oracle disagrees on `{sfen}`");
            let pos = parse_sfen(sfen).unwrap();
            assert_eq!(
                layer_stack_index(&pos),
                *expected,
                "layer_stack_index mismatch on `{sfen}`"
            );
        }
    }

    #[test]
    fn layer_stack_index_white_to_move_mirrors_ranks() {
        let sfen = "8k/9/9/9/9/9/9/9/8K w - 1";
        assert_eq!(oracle_bucket(sfen), 8);
        assert_eq!(layer_stack_index(&parse_sfen(sfen).unwrap()), 8);
    }

    #[test]
    fn layer_stack_index_startpos_is_eight() {
        let startpos = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        assert_eq!(oracle_bucket(startpos), 8);
        assert_eq!(layer_stack_index(&parse_sfen(startpos).unwrap()), 8);
    }

    #[test]
    fn layer_stack_index_is_always_in_range() {
        for (_, sfen) in BUCKET_FIXTURES {
            let got = layer_stack_index(&parse_sfen(sfen).unwrap());
            assert!(got < LAYER_STACKS);
        }
    }
}
