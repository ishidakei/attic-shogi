//! Feature-transformer accumulator and its output transform.
//!
//! Ported from the read-only Rust NNUE reference implementation's
//! `transformer.rs` (refresh path). The per-lane kernels
//! (add/sub feature columns, the pairwise element-wise multiply) go through
//! [`crate::simd`], which picks the AVX-512 backend at compile time when the
//! build enables the features and the scalar baseline otherwise — bit-identical
//! either way.
//! The C++ ground truth is
//! `eval/nnue/nnue_feature_transformer.h` at the
//! current submodule pin.
//!
//! This module owns the accumulator type, its full-refresh path, the
//! incremental (do/undo) differential update, and the output transform that
//! packs the accumulator into the byte input buffer L1 (`fc_0`) consumes. The
//! layer-stack forward pass lives in [`crate::network`], the kernels in
//! [`crate::simd`].
//!
//! ## Incremental update ([`Accumulator::update_after_move`])
//!
//! Ported from the reference `update_on_move` / `update_on_king_move`
//! (the reference's `transformer.rs`) and the C++ `UpdateAccumulator` in
//! `nnue_feature_transformer.h`. Each perspective is derived from the pre-move
//! accumulator independently: a perspective whose own king moved is rebuilt from
//! scratch (see [`crate::features::requires_full_refresh`]); every other
//! perspective copies the previous half and applies the add/sub feature delta
//! ([`crate::features::changed_indices`]). Because the accumulator is a linear
//! sum of `i16` weight columns, `prev - removed + added` equals a from-scratch
//! refresh of the post-move position exactly (bit-for-bit under wrapping `i16`).
//!
//! ## Accumulator layout
//!
//! Two per-perspective `i16` vectors of [`HIDDEN_SIZE`], each in its own
//! 64-byte-aligned buffer (reusing [`crate::Aligned64`] from the loader child).
//! Index `[Color::Black]` always holds Black's half and `[Color::White]` always
//! holds White's half — the side-to-move reordering happens only in the output
//! transform, exactly as the reference and the C++ header do it.
//!
//! ## Output transform (pairwise element-wise multiply)
//!
//! SFNN-1536's feature transformer emits `HalfDimensions = HIDDEN_SIZE` values
//! per perspective, which the output transform folds to `HIDDEN_SIZE / 2` bytes
//! by pairing lane `j` with lane `j + HIDDEN_SIZE/2`: each is clamped to
//! `[0, 254]` (the post-×2-scale analogue of `[0, 127]`), multiplied, and
//! shifted right by 9. The two perspectives are laid out
//! `[stm-half | ~stm-half]`, so the full buffer is [`FT_OUTPUT_DIMS`] bytes.
//! This mirrors the `#else` (scalar) branch of `TransformFeatures` in the C++
//! header: `output = unsigned(clamp(s0,0,254) * clamp(s1,0,254)) / 512`.

use attic_state::{Color, Move, Position};

use crate::aligned::Aligned64;
use crate::features::{
    FeatureIndex, MoveDelta, active_features, changed_indices, requires_full_refresh,
};
use crate::finny::FinnyCache;
use crate::simd::{post_ft_kernel, transformer_kernel};
use crate::types::{FC_0_INPUT_DIMS, HIDDEN_SIZE, NnueNetwork};

/// Width of the byte buffer the output transform fills: one lane per fc_0
/// input, i.e. `HIDDEN_SIZE/2` per perspective across the two perspectives.
pub const FT_OUTPUT_DIMS: usize = FC_0_INPUT_DIMS;

/// Per-perspective feature-transformer accumulator.
///
/// Holds one `i16` vector of [`HIDDEN_SIZE`] per perspective, each in its own
/// cache-line-aligned buffer. Built by [`Accumulator::refresh`] and consumed by
/// [`Accumulator::output_transform`].
#[derive(Debug)]
pub struct Accumulator {
    /// `[Color::Black.index()]` = Black's half, `[Color::White.index()]` =
    /// White's half. Each buffer is `HIDDEN_SIZE` long and 64-byte aligned.
    perspectives: [Aligned64<i16>; Color::COUNT],
}

impl Accumulator {
    /// Allocates a zeroed accumulator (both perspectives all-zero).
    pub fn new() -> Self {
        Accumulator {
            perspectives: [
                Aligned64::<i16>::zeroed(HIDDEN_SIZE),
                Aligned64::<i16>::zeroed(HIDDEN_SIZE),
            ],
        }
    }

    /// Read-only view of one perspective's accumulator half.
    pub fn perspective(&self, color: Color) -> &[i16] {
        &self.perspectives[color.index()]
    }

    /// Full refresh: rebuild both perspectives from the network's FT biases plus
    /// the sum of the FT weight columns for each perspective's active features.
    ///
    /// Byte-for-byte equivalent to the reference `refresh`: each half starts as
    /// a copy of `net.ft_biases`, then every active feature's column is added
    /// with wrapping `i16` arithmetic (the downstream clipped output transform
    /// saturates, so overflow here is intentional and matches upstream).
    ///
    /// # Panics
    /// Panics if `pos` is missing either king (via [`active_features`]).
    pub fn refresh(&mut self, net: &NnueNetwork, pos: &Position) {
        for color in [Color::Black, Color::White] {
            let feats = active_features(pos, color);
            refresh_perspective(
                &mut self.perspectives[color.index()],
                &net.ft_biases,
                &net.ft_weights,
                &feats,
            );
        }
    }

    /// Incrementally derive the accumulator *after* `mv` from `self`, the
    /// accumulator for the position *before* the move.
    ///
    /// `pos` must be the pre-move position. The method does its own
    /// `do_move`/`undo_move` internally to read the post-move active features,
    /// so `pos` is left byte-for-byte unchanged on return — the caller applies
    /// the real move itself afterwards. Returns a fresh accumulator for the
    /// post-move position; the copy-based result composes into a do/undo stack
    /// (push the return value on `do`, pop it on `undo`).
    ///
    /// Perspectives whose own king moved are rebuilt from scratch (their whole
    /// index space shifts); the others copy `self`'s half and apply the add/sub
    /// delta. The result is bit-identical to [`Accumulator::refresh`] on the
    /// post-move position — see the module docs.
    ///
    /// # Panics
    /// Panics if `pos` (before or after `mv`) is missing either king.
    pub fn update_after_move(
        &self,
        net: &NnueNetwork,
        pos: &mut Position,
        mv: Move,
    ) -> Accumulator {
        let refresh = [
            requires_full_refresh(mv, Color::Black),
            requires_full_refresh(mv, Color::White),
        ];

        // Pre-move active features, needed only for the incrementally-updated
        // perspective(s); skip the scan for a perspective that will refresh.
        let before: [Vec<FeatureIndex>; Color::COUNT] = [
            if refresh[Color::Black.index()] {
                Vec::new()
            } else {
                active_features(pos, Color::Black)
            },
            if refresh[Color::White.index()] {
                Vec::new()
            } else {
                active_features(pos, Color::White)
            },
        ];

        let undo = pos.do_move(mv);

        let mut next = Accumulator::new();
        for color in [Color::Black, Color::White] {
            let i = color.index();
            if refresh[i] {
                refresh_perspective(
                    &mut next.perspectives[i],
                    &net.ft_biases,
                    &net.ft_weights,
                    &active_features(pos, color),
                );
            } else {
                let after = active_features(pos, color);
                let (removed, added) = changed_indices(&before[i], &after);
                next.perspectives[i].copy_from_slice(&self.perspectives[i]);
                apply_diff(&mut next.perspectives[i], &net.ft_weights, &added, &removed);
            }
        }

        pos.undo_move(mv, undo);
        next
    }

    /// Hot-path incremental update: write the post-move accumulator into `dst`,
    /// deriving each perspective from `src` (the pre-move accumulator) plus a
    /// move-derived [`MoveDelta`].
    ///
    /// `post_pos` must be the position **after** the move (the search has
    /// already applied `do_move`); it is read only to full-refresh a perspective
    /// whose own king moved (`delta.half(color) == None`). Every other
    /// perspective copies `src`'s half and applies the add/sub column delta —
    /// no `active_features` scan and no internal `do_move`/`undo_move`, unlike
    /// [`Accumulator::update_after_move`] (kept as the equivalence oracle).
    ///
    /// Writing into a caller-owned `dst` lets the search reuse a preallocated
    /// per-worker accumulator stack instead of allocating one buffer per node.
    /// The result is bit-identical to [`Accumulator::refresh`] on `post_pos`.
    ///
    /// # Panics
    /// Panics if `post_pos` is missing either king (via [`active_features`] on a
    /// refreshed perspective).
    pub fn derive_into(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &NnueNetwork,
        post_pos: &Position,
        delta: &MoveDelta,
    ) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            match delta.half(color) {
                Some(pd) => {
                    dst.perspectives[i].copy_from_slice(&src.perspectives[i]);
                    apply_diff(
                        &mut dst.perspectives[i],
                        &net.ft_weights,
                        pd.added(),
                        pd.removed(),
                    );
                }
                None => {
                    refresh_perspective(
                        &mut dst.perspectives[i],
                        &net.ft_biases,
                        &net.ft_weights,
                        &active_features(post_pos, color),
                    );
                }
            }
        }
    }

    /// [`Accumulator::derive_into`] with the king-move rebuild routed through a
    /// worker-private finny table ([`FinnyCache`]).
    ///
    /// The `delta.half(color) == Some(..)` arm — the overwhelmingly common one —
    /// is byte-for-byte the same code as [`derive_into`](Self::derive_into).
    /// Only the `None` arm differs: instead of rebuilding the half from the FT
    /// biases plus all 40 active columns, it diffs the post-move feature list
    /// against the cached refreshed half for that king square and copies the
    /// result out. Bit-identical to [`Accumulator::refresh`] on `post_pos`,
    /// exactly as `derive_into` is — the cache stores a *sum of columns*, and
    /// wrapping `i16` addition does not care how that sum is decomposed.
    ///
    /// **Deviation from the reference pin (76d58ef).** The pin has no finny
    /// tables. Upstream YaneuraOu added them in commit `72c91d8`
    /// (`nnue_feature_transformer.h`) but ships them dormant behind an
    /// undefined `USE_FINNY_TABLES`, so this is an ahead-of-pin adaptation of
    /// dormant upstream code, adopted on a measured NPS win. It is
    /// value-invariant, so it changes no search behaviour and no node count.
    ///
    /// # Panics
    /// Panics if `post_pos` is missing either king.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &NnueNetwork,
        post_pos: &Position,
        delta: &MoveDelta,
        cache: &mut FinnyCache,
    ) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            match delta.half(color) {
                Some(pd) => {
                    dst.perspectives[i].copy_from_slice(&src.perspectives[i]);
                    apply_diff(
                        &mut dst.perspectives[i],
                        &net.ft_weights,
                        pd.added(),
                        pd.removed(),
                    );
                }
                None => cache.refresh_into(net, post_pos, color, &mut dst.perspectives[i]),
            }
        }
    }

    /// Output transform: pack the accumulator into `out`, the byte input buffer
    /// for L1 (`fc_0`), for the given side to move.
    ///
    /// Layout is `[stm-half | ~stm-half]`; each half is the pairwise
    /// element-wise multiply described in the module docs. Byte-for-byte
    /// equivalent to the reference scalar `transformer_ewm`.
    pub fn output_transform(&self, stm: Color, out: &mut [u8; FT_OUTPUT_DIMS]) {
        const HALF: usize = HIDDEN_SIZE / 2;
        let stm_half = &self.perspectives[stm.index()];
        let other_half = &self.perspectives[stm.flip().index()];
        post_ft_kernel::ewm_one_perspective(stm_half, &mut out[..HALF]);
        post_ft_kernel::ewm_one_perspective(other_half, &mut out[HALF..]);
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Rebuild one perspective's half: copy the biases, then add each active
/// feature's weight column. `out` and `ft_biases` are `HIDDEN_SIZE` long;
/// `ft_weights` is a row-major `[feature][lane]` block of `HIDDEN_SIZE`-wide
/// columns.
pub(crate) fn refresh_perspective(
    out: &mut [i16],
    ft_biases: &[i16],
    ft_weights: &[i16],
    indices: &[FeatureIndex],
) {
    out.copy_from_slice(ft_biases);
    transformer_kernel::add_features(out, ft_weights, indices);
}

/// Apply a feature delta to an already-copied perspective half: subtract every
/// removed column and add every added column. Under wrapping `i16` the two
/// passes commute, so `copy(prev)` followed by `apply_diff` reproduces a
/// from-scratch refresh of the post-move feature set exactly.
///
/// The common single-add arities route to the fused
/// [`transformer_kernel::add_sub_features`] / `add_sub_sub_features` kernels
/// (one accumulator round-trip per lane); any other shape falls back to a
/// separate sub-then-add. Faithful port of the reference `apply_diff`.
pub(crate) fn apply_diff(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    match (added.len(), removed.len()) {
        (1, 1) => transformer_kernel::add_sub_features(out, weights, added, removed),
        (1, 2) => transformer_kernel::add_sub_sub_features(
            out,
            weights,
            &added[..1],
            &removed[..1],
            &removed[1..2],
        ),
        _ => {
            transformer_kernel::sub_features(out, weights, removed);
            transformer_kernel::add_features(out, weights, added);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NetHeader, NnueNetworkBuilder};
    use attic_state::{Move, PieceKind, format_usi_move, parse_sfen, parse_usi_move};

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    const HALF: usize = HIDDEN_SIZE / 2;

    // Output-transform clamp ceiling / shift, mirrored from the scalar kernel so
    // these tests can transcribe the reference formula independently.
    const EWM_CLAMP: i32 = 127 * 2;
    const EWM_SHIFT: i32 = 9;

    // --- Accumulator type --------------------------------------------------

    #[test]
    fn new_is_zeroed_and_cache_line_aligned() {
        let acc = Accumulator::new();
        for color in [Color::Black, Color::White] {
            let half = acc.perspective(color);
            assert_eq!(half.len(), HIDDEN_SIZE);
            assert!(half.iter().all(|&x| x == 0));
            assert_eq!(
                half.as_ptr() as usize % 64,
                0,
                "{color:?} half is not 64-byte aligned"
            );
        }
    }

    // --- Refresh kernel: hand-verified accumulator values ------------------

    #[test]
    fn refresh_perspective_sums_bias_and_active_columns() {
        // biases[i] = i; weights[column c][lane i] = c*100 + i. With active
        // indices {2, 5}: out[i] = i + (200 + i) + (500 + i) = 700 + 3*i.
        const N_FEATURES: usize = 8;
        let biases: Vec<i16> = (0..HIDDEN_SIZE as i16).collect();
        let mut weights = vec![0i16; N_FEATURES * HIDDEN_SIZE];
        for c in 0..N_FEATURES {
            for i in 0..HIDDEN_SIZE {
                weights[c * HIDDEN_SIZE + i] = (c as i16) * 100 + i as i16;
            }
        }

        let mut out = vec![0i16; HIDDEN_SIZE];
        refresh_perspective(&mut out, &biases, &weights, &[2, 5]);

        // Hand-checked anchors.
        assert_eq!(out[0], 700);
        assert_eq!(out[1], 703);
        assert_eq!(out[10], 730);
        assert_eq!(out[HIDDEN_SIZE - 1], 700 + 3 * (HIDDEN_SIZE as i16 - 1));
        // Full formula across every lane.
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 700 + 3 * i as i16, "lane {i}");
        }
    }

    #[test]
    fn refresh_perspective_with_no_active_features_is_the_bias_vector() {
        let biases: Vec<i16> = (0..HIDDEN_SIZE).map(|i| (i as i16) % 17 - 8).collect();
        let weights = vec![123i16; 4 * HIDDEN_SIZE];
        let mut out = vec![0i16; HIDDEN_SIZE];
        refresh_perspective(&mut out, &biases, &weights, &[]);
        assert_eq!(out, biases);
    }

    #[test]
    fn add_features_uses_wrapping_i16_arithmetic() {
        // bias = i16::MAX, one column of all-1 -> wraps to i16::MIN.
        let mut out = vec![i16::MAX; HIDDEN_SIZE];
        let weights = vec![1i16; HIDDEN_SIZE];
        transformer_kernel::add_features(&mut out, &weights, &[0]);
        assert!(out.iter().all(|&x| x == i16::MIN));
    }

    // --- Full refresh over a real Position ---------------------------------

    /// Full-size network with zeroed FT weights (lazy calloc pages keep this
    /// cheap), patterned biases, and a deterministic pattern written only into
    /// the columns the given position actually references.
    fn synthetic_net_for(pos: &Position) -> NnueNetwork {
        synthetic_net_covering(std::slice::from_ref(pos))
    }

    /// Like [`synthetic_net_for`] but seeds the weight columns referenced by
    /// *any* of `positions`. Incremental-update tests pass both the pre- and
    /// post-move positions so the features touched by the delta land on real
    /// (nonzero, index-derived) columns rather than untouched zero pages —
    /// otherwise the add-side of the diff would be a trivial no-op. The pattern
    /// depends only on `(idx, lane)`, so overlapping fills are idempotent.
    fn synthetic_net_covering(positions: &[Position]) -> NnueNetwork {
        synthetic_net_covering_salted(positions, 0)
    }

    /// [`synthetic_net_covering`] with every bias and weight shifted by `salt`,
    /// so two nets over the same positions hold demonstrably different
    /// parameters (used to prove the finny cache invalidates on a network
    /// change rather than serving a stale entry).
    fn synthetic_net_covering_salted(positions: &[Position], salt: i16) -> NnueNetwork {
        let header = NetHeader {
            version: 0,
            hash: 0,
            arch_id: "synthetic".to_string(),
        };
        // All parameters live in one arena; fill the FT arrays in place (the FC
        // arrays stay zero). Only the columns the given positions reference are
        // seeded, so the ~215 MiB weight block is barely touched.
        let mut builder = NnueNetworkBuilder::new(header, [0u8; 32]);
        for (i, slot) in builder.ft_biases_mut().iter_mut().enumerate() {
            *slot = ((i as i16) % 17 - 8).wrapping_add(salt);
        }
        {
            let ft_weights = builder.ft_weights_mut();
            for pos in positions {
                for color in [Color::Black, Color::White] {
                    for idx in active_features(pos, color) {
                        let base = idx as usize * HIDDEN_SIZE;
                        for (i, slot) in ft_weights[base..base + HIDDEN_SIZE].iter_mut().enumerate()
                        {
                            *slot = (((idx as i32).wrapping_mul(31).wrapping_add(i as i32 * 7) % 23
                                - 11) as i16)
                                .wrapping_add(salt);
                        }
                    }
                }
            }
        }
        builder.build()
    }

    /// Independent recomputation of one perspective: biases plus every active
    /// column, wrapping — deliberately not sharing code with `refresh`.
    fn expected_half(net: &NnueNetwork, pos: &Position, color: Color) -> Vec<i16> {
        let mut acc: Vec<i16> = net.ft_biases.to_vec();
        for idx in active_features(pos, color) {
            let base = idx as usize * HIDDEN_SIZE;
            let col = &net.ft_weights[base..base + HIDDEN_SIZE];
            for (a, &w) in acc.iter_mut().zip(col.iter()) {
                *a = a.wrapping_add(w);
            }
        }
        acc
    }

    #[test]
    fn refresh_matches_independent_recomputation() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let net = synthetic_net_for(&pos);
        let mut acc = Accumulator::new();
        acc.refresh(&net, &pos);
        for color in [Color::Black, Color::White] {
            assert_eq!(
                acc.perspective(color),
                expected_half(&net, &pos, color).as_slice(),
                "{color:?} perspective mismatch",
            );
        }
    }

    #[test]
    fn refresh_is_deterministic() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let net = synthetic_net_for(&pos);
        let mut a = Accumulator::new();
        let mut b = Accumulator::new();
        a.refresh(&net, &pos);
        b.refresh(&net, &pos);
        for color in [Color::Black, Color::White] {
            assert_eq!(a.perspective(color), b.perspective(color));
        }
    }

    // --- Incremental update: equals a from-scratch refresh -----------------

    /// Refresh `pos`'s accumulator, apply `mv` incrementally from it, then
    /// assert both halves are bit-identical to a from-scratch refresh of the
    /// post-move position. Also asserts `pos` is unchanged by the update call.
    fn assert_incremental_matches_refresh(sfen: &str, usi: &str) {
        let mut pos = parse_sfen(sfen).unwrap();
        let mv: Move = parse_usi_move(usi, &pos).unwrap();

        // Cover both pre- and post-move feature columns so the delta touches
        // real weights (see `synthetic_net_covering`).
        let mut after_pos = pos.clone();
        after_pos.do_move(mv);
        let net = synthetic_net_covering(&[pos.clone(), after_pos]);

        let mut prev = Accumulator::new();
        prev.refresh(&net, &pos);

        let pos_snapshot = pos.clone();
        let next = prev.update_after_move(&net, &mut pos, mv);
        assert_eq!(pos, pos_snapshot, "update_after_move mutated the position");

        let undo = pos.do_move(mv);
        let mut expected = Accumulator::new();
        expected.refresh(&net, &pos);
        pos.undo_move(mv, undo);

        for color in [Color::Black, Color::White] {
            assert_eq!(
                next.perspective(color),
                expected.perspective(color),
                "{color:?}: incremental update != refresh for `{usi}` from `{sfen}`",
            );
        }
    }

    #[test]
    fn update_after_quiet_move_matches_refresh() {
        assert_incremental_matches_refresh(STARTPOS, "7g7f");
    }

    #[test]
    fn update_after_capture_matches_refresh() {
        assert_incremental_matches_refresh("4k4/1p7/9/9/9/9/9/1R7/4K4 b - 1", "8h8b");
    }

    #[test]
    fn update_after_drop_matches_refresh() {
        assert_incremental_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e");
    }

    #[test]
    fn update_after_promotion_matches_refresh() {
        assert_incremental_matches_refresh("4k4/9/9/1P7/9/9/9/9/4K4 b - 1", "8d8c+");
    }

    #[test]
    fn update_after_black_king_move_matches_refresh() {
        // Black's own king moves -> Black refreshes, White stays incremental.
        assert_incremental_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 b - 1", "5i5h");
    }

    #[test]
    fn update_after_white_king_move_matches_refresh() {
        // White to move, White's own king moves -> White refreshes, Black
        // stays incremental.
        assert_incremental_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 w - 1", "5a5b");
    }

    #[test]
    fn update_after_move_sequence_stays_bit_exact() {
        // Thread a multi-ply line through incremental updates, do/undo style,
        // then compare against a fresh refresh at the leaf.
        const LINE: [&str; 5] = ["7g7f", "3c3d", "8h2b+", "3a2b", "5i5h"];

        // First pass: collect every position the line visits so the synthetic
        // net covers all feature columns the deltas touch.
        let mut walk = parse_sfen(STARTPOS).unwrap();
        let mut visited = vec![walk.clone()];
        for usi in LINE {
            let mv = parse_usi_move(usi, &walk).unwrap();
            walk.do_move(mv);
            visited.push(walk.clone());
        }
        let net = synthetic_net_covering(&visited);

        let mut pos = parse_sfen(STARTPOS).unwrap();
        let mut base = Accumulator::new();
        base.refresh(&net, &pos);

        let mut stack: Vec<Accumulator> = vec![base];
        for usi in LINE {
            let mv = parse_usi_move(usi, &pos).unwrap();
            let next = stack.last().unwrap().update_after_move(&net, &mut pos, mv);
            stack.push(next);
            pos.do_move(mv);

            let mut expected = Accumulator::new();
            expected.refresh(&net, &pos);
            for color in [Color::Black, Color::White] {
                assert_eq!(
                    stack.last().unwrap().perspective(color),
                    expected.perspective(color),
                    "{color:?}: drift after `{usi}`",
                );
            }
        }
    }

    // --- Hot-path dirty-piece update: derive_into == refresh == oracle -----

    use crate::features::MoveDelta;

    /// Drive [`Accumulator::derive_into`] exactly as the search does — snapshot
    /// the [`MoveDelta`] from the pre-move position, apply the move, then derive
    /// the child half from the parent — and assert the result is bit-identical
    /// to BOTH a from-scratch [`Accumulator::refresh`] and the scan-based
    /// [`Accumulator::update_after_move`] oracle.
    fn assert_derive_matches_refresh(sfen: &str, usi: &str) {
        let mut pos = parse_sfen(sfen).unwrap();
        let mv: Move = parse_usi_move(usi, &pos).unwrap();

        // Cover both pre- and post-move feature columns (see `synthetic_net_covering`).
        let mut after_pos = pos.clone();
        after_pos.do_move(mv);
        let net = synthetic_net_covering(&[pos.clone(), after_pos]);

        let mut parent = Accumulator::new();
        parent.refresh(&net, &pos);

        // Scan-based oracle (runs its own internal do/undo; leaves `pos` intact).
        let oracle = parent.update_after_move(&net, &mut pos, mv);

        // Hot-path form: delta captured pre-move, derive after do_move.
        let delta = MoveDelta::from_move(&pos, mv);
        let undo = pos.do_move(mv);
        let mut child = Accumulator::new();
        Accumulator::derive_into(&parent, &mut child, &net, &pos, &delta);

        let mut expected = Accumulator::new();
        expected.refresh(&net, &pos);
        pos.undo_move(mv, undo);

        // `derive_into` never mutates `parent` (it writes only `dst`), so a
        // search "pop" that discards `child` trivially restores the parent.
        for color in [Color::Black, Color::White] {
            assert_eq!(
                child.perspective(color),
                expected.perspective(color),
                "{color:?}: derive_into != refresh for `{usi}` from `{sfen}`",
            );
            assert_eq!(
                child.perspective(color),
                oracle.perspective(color),
                "{color:?}: derive_into != update_after_move for `{usi}` from `{sfen}`",
            );
        }
    }

    #[test]
    fn derive_after_quiet_move_matches_refresh() {
        assert_derive_matches_refresh(STARTPOS, "7g7f");
    }

    #[test]
    fn derive_after_capture_matches_refresh() {
        assert_derive_matches_refresh("4k4/1p7/9/9/9/9/9/1R7/4K4 b - 1", "8h8b");
    }

    #[test]
    fn derive_after_drop_matches_refresh() {
        assert_derive_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e");
    }

    #[test]
    fn derive_after_promotion_matches_refresh() {
        assert_derive_matches_refresh("4k4/9/9/1P7/9/9/9/9/4K4 b - 1", "8d8c+");
    }

    #[test]
    fn derive_after_capture_promotion_matches_refresh() {
        // Bishop captures and promotes to a horse; the victim enters hand as a
        // rook — the two-add / two-sub shape, with a promoted board plane.
        assert_derive_matches_refresh("4k4/2r6/9/9/9/9/9/1B7/4K4 b - 1", "8h7b+");
    }

    #[test]
    fn derive_after_capturing_a_promoted_piece_matches_refresh() {
        // The captured piece is a dragon (promoted rook); it must revert to a
        // bare rook in the capturer's hand (base kind, not the promoted plane).
        assert_derive_matches_refresh("4k4/2+r6/9/9/9/9/9/1B7/4K4 b - 1", "8h7b+");
    }

    #[test]
    fn derive_after_black_king_move_matches_refresh() {
        // Black's own king moves -> Black refreshes, White stays incremental.
        assert_derive_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 b - 1", "5i5h");
    }

    #[test]
    fn derive_after_white_king_move_matches_refresh() {
        assert_derive_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 w - 1", "5a5b");
    }

    #[test]
    fn derive_after_king_capture_matches_refresh() {
        // Black king captures an adjacent enemy piece: for White (incremental),
        // the mover is a king (shared king plane) AND a capture happens.
        assert_derive_matches_refresh("4k4/9/9/9/9/9/9/4r4/4K4 b - 1", "5i5h");
    }

    #[test]
    fn null_move_leaves_the_accumulator_unchanged() {
        // A null move changes no pieces and no king squares, so both perspective
        // halves are identical to a refresh of the post-null position — the
        // search's "reuse the parent accumulator" for a null move is exact.
        let mut pos = parse_sfen(STARTPOS).unwrap();
        let net = synthetic_net_for(&pos);
        let mut acc = Accumulator::new();
        acc.refresh(&net, &pos);

        pos.do_null_move();
        let mut expected = Accumulator::new();
        expected.refresh(&net, &pos);
        for color in [Color::Black, Color::White] {
            assert_eq!(
                acc.perspective(color),
                expected.perspective(color),
                "{color:?}: null move changed the accumulator",
            );
        }
    }

    /// Deterministic move choice for the playout below: a pure function of the
    /// ply and position so both passes (net coverage, then the accumulator
    /// thread) agree. It rotates a preference for promotions / captures so the
    /// line is guaranteed to exercise those move types, falling back to an LCG
    /// index for breadth.
    fn choose(ply: usize, pos: &Position, moves: &[Move]) -> Move {
        let idx = ((ply as u64).wrapping_mul(2_654_435_761) >> 13) as usize % moves.len();
        match ply % 4 {
            0 => moves.iter().copied().find(|m| m.is_promote()),
            1 => moves
                .iter()
                .copied()
                .find(|m| !m.is_drop() && pos.board().get(m.to_sq()).is_some()),
            _ => None,
        }
        .unwrap_or(moves[idx])
    }

    #[test]
    fn derive_into_stays_bit_exact_over_a_long_playout() {
        // A >=30-ply deterministic playout from the start position. As pieces are
        // captured, dropped, promoted, and kings shuffle, the stacked accumulator
        // (parent -> derive_into -> child, do/undo style) must equal a
        // from-scratch refresh BIT FOR BIT at every ply.
        const PLIES: usize = 48;

        // Pass 1: replay the line to collect every visited position so the
        // synthetic net seeds all feature columns the deltas touch.
        let mut walk = parse_sfen(STARTPOS).unwrap();
        let mut visited = vec![walk.clone()];
        let mut line: Vec<Move> = Vec::new();
        for ply in 0..PLIES {
            let mut moves: Vec<Move> = Vec::new();
            walk.generate_legal_all(&mut moves);
            if moves.is_empty() {
                break;
            }
            let mv = choose(ply, &walk, &moves);
            line.push(mv);
            walk.do_move(mv);
            visited.push(walk.clone());
        }
        assert!(line.len() >= 30, "playout too short: {}", line.len());
        let net = synthetic_net_covering(&visited);

        // Pass 2: thread the accumulator stack through the same line.
        let mut pos = parse_sfen(STARTPOS).unwrap();
        let mut base = Accumulator::new();
        base.refresh(&net, &pos);
        let mut stack: Vec<Accumulator> = vec![base];

        let mut saw_capture = false;
        let mut saw_promotion = false;
        let mut saw_drop = false;
        let mut saw_king_move = false;

        for (i, &mv) in line.iter().enumerate() {
            saw_capture |= !mv.is_drop() && pos.board().get(mv.to_sq()).is_some();
            saw_promotion |= mv.is_promote();
            saw_drop |= mv.is_drop();
            saw_king_move |= !mv.is_drop() && mv.moved_piece_after().kind == PieceKind::King;

            let delta = MoveDelta::from_move(&pos, mv);
            pos.do_move(mv);
            let mut child = Accumulator::new();
            Accumulator::derive_into(stack.last().unwrap(), &mut child, &net, &pos, &delta);

            let mut expected = Accumulator::new();
            expected.refresh(&net, &pos);
            for color in [Color::Black, Color::White] {
                assert_eq!(
                    child.perspective(color),
                    expected.perspective(color),
                    "{color:?}: drift at ply {i} after `{}`",
                    format_usi_move(mv),
                );
            }
            stack.push(child);
        }

        assert!(saw_capture, "playout exercised no capture");
        assert!(saw_promotion, "playout exercised no promotion");
        assert!(saw_drop, "playout exercised no drop");
        assert!(saw_king_move, "playout exercised no king move");
    }

    // --- Finny-table cached rebuild ----------------------------------------
    //
    // The gate is bit-identity against a from-scratch rebuild in every cache
    // state: cold entry, warm entry on the same position, warm entry carried
    // over from a DIFFERENT position that shares the king square (the stale-entry
    // case the diff must absorb), and across a network change (which must reset
    // the cache rather than serve a stale half).

    use crate::finny::FinnyCache;

    /// Rebuild `color`'s half through `cache` and assert it equals the
    /// independent `biases + active columns` recomputation, then re-check the
    /// cache's own invariant over every initialised entry.
    fn assert_cached_refresh_matches(
        cache: &mut FinnyCache,
        net: &NnueNetwork,
        pos: &Position,
        color: Color,
        what: &str,
    ) {
        let mut dst = vec![0i16; HIDDEN_SIZE];
        cache.refresh_into(net, pos, color, &mut dst);
        assert_eq!(
            dst,
            expected_half(net, pos, color),
            "{color:?}: cached rebuild != from-scratch refresh ({what})",
        );
        cache.assert_invariant(net);
    }

    #[test]
    fn cached_refresh_matches_a_cold_entry_then_a_warm_one() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let net = synthetic_net_for(&pos);
        let mut cache = FinnyCache::new();

        for color in [Color::Black, Color::White] {
            let king = pos.king_square(color).unwrap();
            assert!(!cache.is_warm(color, king), "entry warm before any rebuild");
            assert_cached_refresh_matches(&mut cache, &net, &pos, color, "cold entry");
            assert!(cache.is_warm(color, king), "entry not warmed by a rebuild");
            // Second pass over the same position: the diff is empty, and the
            // cached half must still be exactly right.
            assert_cached_refresh_matches(
                &mut cache,
                &net,
                &pos,
                color,
                "warm entry, same position",
            );
        }
    }

    #[test]
    fn cached_refresh_absorbs_a_stale_entry_from_a_different_position() {
        // Same black king square (5i) in both positions, wildly different piece
        // sets — so the second rebuild hits a warm entry whose cached half
        // belongs to the first position and must be diffed all the way across.
        let a = parse_sfen(STARTPOS).unwrap();
        let b = parse_sfen("4k4/9/2+R6/9/9/1n2s4/9/9/4K4 b GPl2p 1").unwrap();
        assert_eq!(a.king_square(Color::Black), b.king_square(Color::Black));

        let net = synthetic_net_covering(&[a.clone(), b.clone()]);
        let mut cache = FinnyCache::new();

        assert_cached_refresh_matches(&mut cache, &net, &a, Color::Black, "cold entry");
        assert_cached_refresh_matches(
            &mut cache,
            &net,
            &b,
            Color::Black,
            "stale entry, same bucket",
        );
        // ...and back again, so the diff runs in both directions.
        assert_cached_refresh_matches(&mut cache, &net, &a, Color::Black, "stale entry, reversed");
    }

    #[test]
    fn cached_refresh_invalidates_on_a_network_change() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let net_a = synthetic_net_covering_salted(std::slice::from_ref(&pos), 0);
        let net_b = synthetic_net_covering_salted(std::slice::from_ref(&pos), 5);
        // Both nets are live at once, so they cannot share an identity token.
        assert_ne!(
            net_a.ft_biases.to_vec(),
            net_b.ft_biases.to_vec(),
            "the two synthetic nets must actually differ",
        );

        let mut cache = FinnyCache::new();
        assert_cached_refresh_matches(&mut cache, &net_a, &pos, Color::Black, "net A, cold");

        // Rebuilding the SAME position against a different network must reset
        // the cache: reusing the entry would return net A's half.
        assert_cached_refresh_matches(&mut cache, &net_b, &pos, Color::Black, "net B after net A");
        assert!(
            !cache.is_warm(Color::White, pos.king_square(Color::White).unwrap()),
            "the reset must clear every entry, not just the one rebuilt",
        );
        // And back to net A, which must reset again rather than serve net B's half.
        assert_cached_refresh_matches(&mut cache, &net_a, &pos, Color::Black, "net A after net B");
    }

    #[test]
    fn derive_into_cached_matches_derive_into_on_a_king_move() {
        // The wiring gate: the cached hot-path entry point agrees with both the
        // uncached one and a from-scratch refresh, on a move that rebuilds.
        let mut pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let mv = parse_usi_move("5i5h", &pos).unwrap();
        let mut after_pos = pos.clone();
        after_pos.do_move(mv);
        let net = synthetic_net_covering(&[pos.clone(), after_pos]);

        let mut parent = Accumulator::new();
        parent.refresh(&net, &pos);

        let delta = MoveDelta::from_move(&pos, mv);
        assert!(
            delta.half(Color::Black).is_none(),
            "this fixture must exercise the rebuild arm",
        );
        pos.do_move(mv);

        let mut uncached = Accumulator::new();
        Accumulator::derive_into(&parent, &mut uncached, &net, &pos, &delta);

        let mut cache = FinnyCache::new();
        let mut cached = Accumulator::new();
        Accumulator::derive_into_cached(&parent, &mut cached, &net, &pos, &delta, &mut cache);

        let mut expected = Accumulator::new();
        expected.refresh(&net, &pos);

        for color in [Color::Black, Color::White] {
            assert_eq!(
                cached.perspective(color),
                expected.perspective(color),
                "{color:?}: derive_into_cached != refresh",
            );
            assert_eq!(
                cached.perspective(color),
                uncached.perspective(color),
                "{color:?}: derive_into_cached != derive_into",
            );
        }
    }

    #[test]
    fn derive_into_cached_stays_bit_exact_over_a_long_playout() {
        // The `derive_into` playout gate, re-run through one persistent finny
        // cache: every ply must still equal a from-scratch refresh bit for bit,
        // with real king moves and real warm-entry hits along the way.
        const PLIES: usize = 48;

        let mut walk = parse_sfen(STARTPOS).unwrap();
        let mut visited = vec![walk.clone()];
        let mut line: Vec<Move> = Vec::new();
        for ply in 0..PLIES {
            let mut moves: Vec<Move> = Vec::new();
            walk.generate_legal_all(&mut moves);
            if moves.is_empty() {
                break;
            }
            // Sibling king-move successors are visited by the derive loop
            // below, so their feature columns must be seeded too — otherwise
            // those rebuilds would run against untouched zero pages.
            for &m in moves
                .iter()
                .filter(|m| !m.is_drop() && m.moved_piece_after().kind == PieceKind::King)
            {
                let mut side = walk.clone();
                side.do_move(m);
                visited.push(side);
            }
            let mv = choose(ply, &walk, &moves);
            line.push(mv);
            walk.do_move(mv);
            visited.push(walk.clone());
        }
        assert!(line.len() >= 30, "playout too short: {}", line.len());
        let net = synthetic_net_covering(&visited);

        let mut cache = FinnyCache::new();
        let mut king_moves = 0usize;
        let mut warm_hits = 0usize;

        // Two passes over the same line through ONE cache. A single linear
        // playout rarely revisits a king square; a real search does constantly,
        // both by re-searching the same king move from many nodes and by trying
        // several king moves at one node. The sibling loop below models the
        // second, the repeated pass the first.
        for pass in 0..2 {
            let mut pos = parse_sfen(STARTPOS).unwrap();
            let mut base = Accumulator::new();
            base.refresh(&net, &pos);
            let mut stack: Vec<Accumulator> = vec![base];

            for (i, &mv) in line.iter().enumerate() {
                // Every legal king move at this node, derived and discarded —
                // the do/undo "try a sibling move" shape of the search.
                let mut siblings: Vec<Move> = Vec::new();
                pos.generate_legal_all(&mut siblings);
                siblings.retain(|m| {
                    *m != mv && !m.is_drop() && m.moved_piece_after().kind == PieceKind::King
                });
                for &sib in siblings.iter().chain(std::iter::once(&mv)) {
                    let delta = MoveDelta::from_move(&pos, sib);
                    let undo = pos.do_move(sib);

                    // Classify the rebuild BEFORE it happens: a rebuild arm
                    // whose entry is already initialised is exactly a warm hit.
                    for color in [Color::Black, Color::White] {
                        if delta.half(color).is_none() {
                            king_moves += 1;
                            if cache.is_warm(color, pos.king_square(color).unwrap()) {
                                warm_hits += 1;
                            }
                        }
                    }

                    let mut child = Accumulator::new();
                    Accumulator::derive_into_cached(
                        stack.last().unwrap(),
                        &mut child,
                        &net,
                        &pos,
                        &delta,
                        &mut cache,
                    );

                    let mut expected = Accumulator::new();
                    expected.refresh(&net, &pos);
                    for color in [Color::Black, Color::White] {
                        assert_eq!(
                            child.perspective(color),
                            expected.perspective(color),
                            "{color:?}: drift on pass {pass} at ply {i} after `{}`",
                            format_usi_move(sib),
                        );
                    }
                    cache.assert_invariant(&net);

                    pos.undo_move(sib, undo);
                    if sib == mv {
                        // The line move is the one that actually advances.
                        pos.do_move(mv);
                        stack.push(child);
                    }
                }
            }
        }

        assert!(king_moves > 0, "playout exercised no king move");
        assert!(warm_hits > 0, "playout never hit a warm cache entry");
    }

    // --- Output transform: clipping edges ----------------------------------

    #[test]
    fn output_transform_lays_out_stm_then_other() {
        // us(Black) lane0 pair -> (200*100)>>9 = 39; them(White) -> (250*50)>>9 = 24.
        let mut acc = Accumulator::new();
        acc.perspectives[Color::Black.index()][0] = 200;
        acc.perspectives[Color::Black.index()][HALF] = 100;
        acc.perspectives[Color::White.index()][0] = 250;
        acc.perspectives[Color::White.index()][HALF] = 50;

        let mut out = [0u8; FT_OUTPUT_DIMS];
        acc.output_transform(Color::Black, &mut out);
        assert_eq!(out[0], 39);
        assert_eq!(out[HALF], 24);
        assert!(out[1..HALF].iter().all(|&x| x == 0));
        assert!(out[HALF + 1..].iter().all(|&x| x == 0));

        // Side-to-move flips the two halves.
        let mut out_w = [0u8; FT_OUTPUT_DIMS];
        acc.output_transform(Color::White, &mut out_w);
        assert_eq!(out_w[0], 24);
        assert_eq!(out_w[HALF], 39);
    }

    #[test]
    fn output_transform_clamps_min_zero_and_max() {
        let mut acc = Accumulator::new();
        let black = &mut acc.perspectives[Color::Black.index()];
        // lane 0: negative s0 clamps to 0 -> product 0.
        black[0] = -1_000;
        black[HALF] = 200;
        // lane 1: s1 = 0 -> product 0.
        black[1] = 254;
        black[HALF + 1] = 0;
        // lane 2: both above the ceiling clamp to 254 -> (254*254)>>9 = 126.
        black[2] = 30_000;
        black[HALF + 2] = 30_000;
        // lane 3: exact ceiling values, same result.
        black[3] = EWM_CLAMP as i16;
        black[HALF + 3] = EWM_CLAMP as i16;
        // lane 4: a mid value, hand-checked: (100*50)>>9 = 9.
        black[4] = 100;
        black[HALF + 4] = 50;

        let mut out = [0u8; FT_OUTPUT_DIMS];
        acc.output_transform(Color::Black, &mut out);
        assert_eq!(out[0], 0, "negative clamps to zero");
        assert_eq!(out[1], 0, "zero factor yields zero");
        assert_eq!(out[2], 126, "saturated pair");
        assert_eq!(out[3], 126, "exact ceiling pair");
        assert_eq!(out[4], 9, "mid-range pair");
    }

    #[test]
    fn output_transform_full_row_matches_scalar_formula() {
        // Patterned accumulator across every lane, cross-checked against a
        // direct transcription of the reference scalar formula.
        let mut acc = Accumulator::new();
        for color in [Color::Black, Color::White] {
            let half = &mut acc.perspectives[color.index()];
            for i in 0..HIDDEN_SIZE {
                half[i] = ((i as i32 * 5 + color.index() as i32 * 3) % 400 - 50) as i16;
            }
        }

        for stm in [Color::Black, Color::White] {
            let mut out = [0u8; FT_OUTPUT_DIMS];
            acc.output_transform(stm, &mut out);

            let order = [stm, stm.flip()];
            for (p, persp) in order.iter().enumerate() {
                let half = acc.perspective(*persp);
                for j in 0..HALF {
                    let s0 = (half[j] as i32).clamp(0, EWM_CLAMP);
                    let s1 = (half[j + HALF] as i32).clamp(0, EWM_CLAMP);
                    let expected = ((s0 * s1) >> EWM_SHIFT) as u8;
                    assert_eq!(out[p * HALF + j], expected, "stm {stm:?} half {p} lane {j}");
                }
            }
        }
    }
}
