//! `HalfKA_hm2` active-feature index extraction, ported from
//! `eval/nnue/features/half_ka_hm2.{h,cpp}`. An `attic-state::Square` index is
//! `file * 9 + rank`, byte-identical to the reference's `Square` numbering, so
//! the index arithmetic carries over unchanged.
//!
//! A feature index is `E_KING * sq_k_code + p_adj`, where:
//! - `sq_k_code` is the perspective's own-king square, horizontally mirrored
//!   into files 1–5 (`0..45`) — the `hm` canonicalization mirrors the whole
//!   position around the king's file;
//! - `p_adj` selects a plane and a square or count within it: hand planes
//!   `[0, 90)`, board planes `[90, 1548)` interleaving friend and enemy per
//!   piece type, and one shared king plane `[1548, 1629)` both kings collapse
//!   into.
//!
//! The accumulator itself lives in [`crate::transformer`].

use attic_state::{Color, Move, Piece, PieceKind, Position, Square};

use crate::types::NUM_FEATURES;

/// A single active input-feature index into the `HalfKA_hm2` feature space.
pub type FeatureIndex = u32;

/// The feature-space origin the reference initialises empty piece slots to
/// (`EvalList::clear()`), used to pad sparse positions up to 40 slots.
const BONA_PIECE_ZERO: usize = 0;

// Hand planes. `F_*` is the friend plane base, `E_*` the enemy one. Each span
// holds one index per possible held count, with slot 0 the unused "zero" pad,
// so a pawn's plane is 19 wide.
const F_HAND_PAWN: usize = 0;
const E_HAND_PAWN: usize = F_HAND_PAWN + 19;
const F_HAND_LANCE: usize = E_HAND_PAWN + 19;
const E_HAND_LANCE: usize = F_HAND_LANCE + 5;
const F_HAND_KNIGHT: usize = E_HAND_LANCE + 5;
const E_HAND_KNIGHT: usize = F_HAND_KNIGHT + 5;
const F_HAND_SILVER: usize = E_HAND_KNIGHT + 5;
const E_HAND_SILVER: usize = F_HAND_SILVER + 5;
const F_HAND_GOLD: usize = E_HAND_SILVER + 5;
const E_HAND_GOLD: usize = F_HAND_GOLD + 5;
const F_HAND_BISHOP: usize = E_HAND_GOLD + 5;
const E_HAND_BISHOP: usize = F_HAND_BISHOP + 3;
const F_HAND_ROOK: usize = E_HAND_BISHOP + 3;
const E_HAND_ROOK: usize = F_HAND_ROOK + 3;
const FE_HAND_END: usize = E_HAND_ROOK + 3;

// Board planes: one 81-square plane per (side, effective piece type), in the
// reference's `BonaPiece` order. Promoted minors collapse onto the gold plane;
// promoted bishop and rook land on the horse and dragon planes.
const F_PAWN: usize = FE_HAND_END;
const E_PAWN: usize = F_PAWN + 81;
const F_LANCE: usize = E_PAWN + 81;
const E_LANCE: usize = F_LANCE + 81;
const F_KNIGHT: usize = E_LANCE + 81;
const E_KNIGHT: usize = F_KNIGHT + 81;
const F_SILVER: usize = E_KNIGHT + 81;
const E_SILVER: usize = F_SILVER + 81;
const F_GOLD: usize = E_SILVER + 81;
const E_GOLD: usize = F_GOLD + 81;
const F_BISHOP: usize = E_GOLD + 81;
const E_BISHOP: usize = F_BISHOP + 81;
const F_HORSE: usize = E_BISHOP + 81;
const E_HORSE: usize = F_HORSE + 81;
const F_ROOK: usize = E_HORSE + 81;
const E_ROOK: usize = F_ROOK + 81;
const F_DRAGON: usize = E_ROOK + 81;
const E_DRAGON: usize = F_DRAGON + 81;
const FE_END: usize = E_DRAGON + 81;

const SQ_NB: usize = Square::COUNT;
/// Size of one king-plane block. Both kings share the `[FE_END, E_KING)` span,
/// so a king-plane is `FE_END + 81` wide.
const E_KING: usize = FE_END + SQ_NB;

/// Number of distinct mirrored king squares (files 1–5 × 9 ranks).
const SQ_K_COUNT: usize = 5 * 9;

/// Total `HalfKA_hm2` feature dimension, kept equal to
/// [`crate::types::NUM_FEATURES`].
pub const FEATURE_DIMENSION: usize = SQ_K_COUNT * E_KING;

/// Maximum number of simultaneously active features (`PIECE_NUMBER_NB`). Every
/// legal shogi position has exactly 40 pieces across board and hands.
pub const MAX_ACTIVE_FEATURES: usize = 40;

/// The seven piece kinds that can sit in a hand, in `BonaPiece` plane order.
const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Rotate a square 180° when viewing the board from `persp`'s side, so that the
/// mirror maths downstream works in an own-side-forward frame.
#[inline]
fn from_persp(sq: Square, persp: Color) -> Square {
    match persp {
        Color::Black => sq,
        Color::White => Square::from_index((SQ_NB as u8 - 1) - sq.index())
            .expect("180-degree rotation of a valid square is valid"),
    }
}

/// The `hm` trigger: mirror when the (perspective-relative) king sits on files
/// 6–9, i.e. file index `>= 5`. Matches the C++ `sq_k >= SQ_61` test.
#[inline]
fn needs_mirror(king_sq_persp: Square) -> bool {
    king_sq_persp.file() >= 5
}

/// Horizontal file mirror: file `f -> 8 - f`, rank unchanged.
#[inline]
fn mirror_if_needed(sq: Square, mirror: bool) -> Square {
    if mirror {
        Square::new(Square::FILES - 1 - sq.file(), sq.rank())
            .expect("file mirror of a valid square is valid")
    } else {
        sq
    }
}

/// A `(friend, enemy)` board-plane base pair for a king slot — never indexed in
/// production (kings use the shared king plane), so a sentinel that would blow
/// past `E_KING` and trip [`encode_feature`]'s `debug_assert!` if ever used.
const NO_BOARD_PLANE: (usize, usize) = (usize::MAX, usize::MAX);

/// Friend/enemy board-plane bases indexed by `[PieceKind::index()][promoted]`.
/// The king row is a sentinel: kings have no board plane, the caller routing
/// them to the shared one instead.
const BOARD_PLANE: [[(usize, usize); 2]; PieceKind::COUNT] = {
    let mut table = [[NO_BOARD_PLANE; 2]; PieceKind::COUNT];
    table[PieceKind::Pawn.index()] = [(F_PAWN, E_PAWN), (F_GOLD, E_GOLD)];
    table[PieceKind::Lance.index()] = [(F_LANCE, E_LANCE), (F_GOLD, E_GOLD)];
    table[PieceKind::Knight.index()] = [(F_KNIGHT, E_KNIGHT), (F_GOLD, E_GOLD)];
    table[PieceKind::Silver.index()] = [(F_SILVER, E_SILVER), (F_GOLD, E_GOLD)];
    table[PieceKind::Gold.index()] = [(F_GOLD, E_GOLD), (F_GOLD, E_GOLD)];
    table[PieceKind::Bishop.index()] = [(F_BISHOP, E_BISHOP), (F_HORSE, E_HORSE)];
    table[PieceKind::Rook.index()] = [(F_ROOK, E_ROOK), (F_DRAGON, E_DRAGON)];
    table
};

/// Friend/enemy hand-plane bases indexed by `[PieceKind::index()]`. The king
/// row is a sentinel — kings are never held in hand.
const HAND_PLANE: [(usize, usize); PieceKind::COUNT] = {
    let mut table = [NO_BOARD_PLANE; PieceKind::COUNT];
    table[PieceKind::Pawn.index()] = (F_HAND_PAWN, E_HAND_PAWN);
    table[PieceKind::Lance.index()] = (F_HAND_LANCE, E_HAND_LANCE);
    table[PieceKind::Knight.index()] = (F_HAND_KNIGHT, E_HAND_KNIGHT);
    table[PieceKind::Silver.index()] = (F_HAND_SILVER, E_HAND_SILVER);
    table[PieceKind::Gold.index()] = (F_HAND_GOLD, E_HAND_GOLD);
    table[PieceKind::Bishop.index()] = (F_HAND_BISHOP, E_HAND_BISHOP);
    table[PieceKind::Rook.index()] = (F_HAND_ROOK, E_HAND_ROOK);
    table
};

/// Friend/enemy board-plane base for a board piece. Kings have no board plane,
/// so passing one is a caller bug.
#[inline]
fn board_plane(kind: PieceKind, promoted: bool, is_friend: bool) -> usize {
    debug_assert!(
        kind != PieceKind::King,
        "king has no board plane; caller handles it"
    );
    let (friend, enemy) = BOARD_PLANE[kind.index()][promoted as usize];
    if is_friend { friend } else { enemy }
}

/// Friend/enemy hand-plane base for a held piece kind. Kings are never held in
/// hand, so passing one is a caller bug.
#[inline]
fn hand_plane(kind: PieceKind, is_friend: bool) -> usize {
    debug_assert!(kind != PieceKind::King, "king is never held in hand");
    let (friend, enemy) = HAND_PLANE[kind.index()];
    if is_friend { friend } else { enemy }
}

/// Combine the mirrored king code with a plane-adjusted piece code.
#[inline]
fn encode_feature(sq_k_code: usize, p_adj: usize) -> FeatureIndex {
    debug_assert!(sq_k_code < SQ_K_COUNT);
    debug_assert!(p_adj < E_KING);
    let idx = E_KING * sq_k_code + p_adj;
    debug_assert!(idx < NUM_FEATURES);
    idx as FeatureIndex
}

/// Locate `color`'s king.
///
/// # Panics
/// Panics if `color` has no king: every position fed to the evaluator must have
/// both, as the reference's `pos.king_square` contract requires.
pub(crate) fn king_square(pos: &Position, color: Color) -> Square {
    pos.king_square(color)
        .unwrap_or_else(|| panic!("position has no {color:?} king"))
}

/// An 81-square mailbox scan, the oracle for [`king_square`].
#[cfg(test)]
fn king_square_scan(pos: &Position, color: Color) -> Square {
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        if let Some(piece) = pos.board().get(sq)
            && piece.kind == PieceKind::King
            && piece.color == color
        {
            return sq;
        }
    }
    panic!("position has no {color:?} king");
}

/// Active `HalfKA_hm2` feature indices for one `perspective` of `pos`: one per
/// board piece, one per held-piece instance, then `BONA_PIECE_ZERO` padding for
/// every empty piece slot. Always exactly [`MAX_ACTIVE_FEATURES`] indices, each
/// `< FEATURE_DIMENSION`, with the padding features the only duplicates.
///
/// The padding is load-bearing. The reference iterates a *fixed* 40 piece-number
/// slots and initialises every unused one to `BONA_PIECE_ZERO`, so each absent
/// piece contributes the feature `E_KING * sq_k_code + 0`. The accumulator sums
/// feature columns, so those repeated zero-features shift the result: omitting
/// them diverges from ground truth on any sparse position.
///
/// # Panics
/// Panics if `pos` is missing the `perspective` side's king.
pub fn active_features(pos: &Position, perspective: Color) -> Vec<FeatureIndex> {
    let mut list = Vec::with_capacity(MAX_ACTIVE_FEATURES);
    active_features_into(pos, perspective, &mut list);
    list
}

/// [`active_features`] writing into a caller-owned buffer, so that the
/// finny-table cache's king-move arm does not allocate per node.
///
/// # Panics
/// Panics if `pos` is missing the `perspective` side's king.
pub(crate) fn active_features_into(
    pos: &Position,
    perspective: Color,
    list: &mut Vec<FeatureIndex>,
) {
    list.clear();
    list.reserve(MAX_ACTIVE_FEATURES);

    let own_king_persp = from_persp(king_square(pos, perspective), perspective);
    let mirror = needs_mirror(own_king_persp);
    let sq_k_code = mirror_if_needed(own_king_persp, mirror).index() as usize;

    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        let Some(piece) = pos.board().get(sq) else {
            continue;
        };
        let is_friend = piece.color == perspective;
        let sq_persp = mirror_if_needed(from_persp(sq, perspective), mirror);
        let sq_code = sq_persp.index() as usize;

        let p_adj = if piece.kind == PieceKind::King {
            FE_END + sq_code
        } else {
            board_plane(piece.kind, piece.promoted, is_friend) + sq_code
        };
        list.push(encode_feature(sq_k_code, p_adj));
    }

    // The k-th held piece of a kind lands at `base + k`.
    for hand_color in [Color::Black, Color::White] {
        let is_friend = hand_color == perspective;
        let hand = pos.hand(hand_color);
        for kind in HAND_KINDS {
            let count = hand.count(kind);
            let base = hand_plane(kind, is_friend);
            for i in 1..=count as usize {
                list.push(encode_feature(sq_k_code, base + i));
            }
        }
    }

    // A legal 40-piece position is already full and skips this loop.
    debug_assert!(list.len() <= MAX_ACTIVE_FEATURES);
    while list.len() < MAX_ACTIVE_FEATURES {
        list.push(encode_feature(sq_k_code, BONA_PIECE_ZERO));
    }
}

/// Active features for both perspectives, indexed by [`Color::index`].
pub fn active_features_both(pos: &Position) -> [Vec<FeatureIndex>; Color::COUNT] {
    [
        active_features(pos, Color::Black),
        active_features(pos, Color::White),
    ]
}

/// Whether `perspective`'s accumulator half must be fully refreshed after `mv`
/// rather than updated incrementally (`requires_full_refresh`,
/// `half_ka_hm2.cpp`).
///
/// A `HalfKA_hm2` index embeds the perspective's own-king `sq_k_code`, so when
/// that king moves *every* index shifts — and the mirror flag may flip with it —
/// and no add/sub delta can express the change. The reference's separate
/// mirror-boundary case is subsumed here: only a king move can change the
/// perspective's own-king file.
pub fn requires_full_refresh(mv: Move, perspective: Color) -> bool {
    if mv.is_drop() {
        return false;
    }
    // A king never promotes, so `moved_piece_after` reports it directly.
    let piece = mv.moved_piece_after();
    piece.kind == PieceKind::King && piece.color == perspective
}

/// Multiset difference of two active-feature lists: `(removed, added)` such
/// that applying them to the `before` multiset yields `after`.
///
/// The inputs carry their `BONA_PIECE_ZERO` padding, but a legal move conserves
/// total piece count, so for any perspective updated incrementally the padding
/// multiplicity is equal on both sides and cancels.
pub(crate) fn changed_indices(
    before: &[FeatureIndex],
    after: &[FeatureIndex],
) -> (Vec<FeatureIndex>, Vec<FeatureIndex>) {
    let mut scratch = DiffScratch::default();
    changed_indices_into(before, after, &mut scratch);
    (scratch.removed, scratch.added)
}

/// Reusable buffers for [`changed_indices_into`], owned by the finny-table cache
/// so that a cached rebuild allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct DiffScratch {
    sorted_before: Vec<FeatureIndex>,
    sorted_after: Vec<FeatureIndex>,
    /// Features present in `before` but not in `after` (multiset semantics).
    pub(crate) removed: Vec<FeatureIndex>,
    /// Features present in `after` but not in `before` (multiset semantics).
    pub(crate) added: Vec<FeatureIndex>,
}

/// [`changed_indices`] writing into caller-owned buffers. This is the single
/// implementation; `changed_indices` is a thin allocating wrapper.
pub(crate) fn changed_indices_into(
    before: &[FeatureIndex],
    after: &[FeatureIndex],
    scratch: &mut DiffScratch,
) {
    let DiffScratch {
        sorted_before,
        sorted_after,
        removed,
        added,
    } = scratch;

    sorted_before.clear();
    sorted_before.extend_from_slice(before);
    sorted_after.clear();
    sorted_after.extend_from_slice(after);
    sorted_before.sort_unstable();
    sorted_after.sort_unstable();
    removed.clear();
    added.clear();

    let (mut i, mut j) = (0usize, 0usize);
    while i < sorted_before.len() && j < sorted_after.len() {
        match sorted_before[i].cmp(&sorted_after[j]) {
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                removed.push(sorted_before[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                added.push(sorted_after[j]);
                j += 1;
            }
        }
    }
    removed.extend_from_slice(&sorted_before[i..]);
    added.extend_from_slice(&sorted_after[j..]);
}

// The hot search path cannot afford [`active_features`]'s full 40-slot scan and
// merge at every `do_move`, so the reference instead tracks the handful of
// "dirty" pieces a move touches and rewrites only their feature columns.
// [`MoveDelta::from_move`] reads those straight off the move and the pre-move
// position.
//
// The result is bit-exact, not approximate: the only `active_features` entries
// that differ between the pre- and post-move positions are exactly these dirty
// pieces, every legal move conserving the total piece count so that the
// `BONA_PIECE_ZERO` padding multiplicity cancels.

/// One board or hand slot a move touches, in a perspective-independent form.
#[derive(Clone, Copy)]
enum Dirty {
    /// A piece sitting on `sq` (a king is encoded via the shared king plane).
    Board { sq: Square, piece: Piece },
    /// The `count`-th (1-based) held piece of `kind` in `color`'s hand.
    Hand {
        color: Color,
        kind: PieceKind,
        count: usize,
    },
}

impl Dirty {
    /// Encode this slot into the feature index for `persp`, whose own-king code
    /// is `(sq_k_code, mirror)`.
    #[inline]
    fn encode(self, persp: Color, sq_k_code: usize, mirror: bool) -> FeatureIndex {
        match self {
            Dirty::Board { sq, piece } => {
                let sq_code = mirror_if_needed(from_persp(sq, persp), mirror).index() as usize;
                let p_adj = if piece.kind == PieceKind::King {
                    FE_END + sq_code
                } else {
                    board_plane(piece.kind, piece.promoted, piece.color == persp) + sq_code
                };
                encode_feature(sq_k_code, p_adj)
            }
            Dirty::Hand { color, kind, count } => {
                encode_feature(sq_k_code, hand_plane(kind, color == persp) + count)
            }
        }
    }
}

/// The add/sub feature delta for one perspective across a move. At most two
/// columns change per side, so the lists are fixed two-slot arrays.
#[derive(Clone, Copy, Default)]
pub struct PerspectiveDelta {
    removed: [FeatureIndex; 2],
    n_removed: usize,
    added: [FeatureIndex; 2],
    n_added: usize,
}

impl PerspectiveDelta {
    /// Feature columns to subtract from the pre-move accumulator half.
    #[inline]
    pub fn removed(&self) -> &[FeatureIndex] {
        &self.removed[..self.n_removed]
    }

    /// Feature columns to add to the pre-move accumulator half.
    #[inline]
    pub fn added(&self) -> &[FeatureIndex] {
        &self.added[..self.n_added]
    }
}

/// The per-perspective feature delta a move induces, computed straight from the
/// pre-move position without the [`active_features`] scan.
pub struct MoveDelta {
    halves: [Option<PerspectiveDelta>; Color::COUNT],
}

impl MoveDelta {
    /// This move's delta for `perspective`, or `None` if that perspective's own
    /// king moved and its half must be refreshed from scratch.
    #[inline]
    pub fn half(&self, perspective: Color) -> Option<&PerspectiveDelta> {
        self.halves[perspective.index()].as_ref()
    }

    /// Compute the delta for `mv` from the **pre-move** `pos`, which is not
    /// mutated.
    ///
    /// # Panics
    /// Panics if a non-drop move has no piece on its origin square, or if an
    /// incrementally-updated perspective is missing its king.
    pub fn from_move(pos: &Position, mv: Move) -> MoveDelta {
        let mover = pos.side_to_move();
        let to = mv.to_sq();

        let mut removed: [Option<Dirty>; 2] = [None, None];
        let mut added: [Option<Dirty>; 2] = [None, None];

        if mv.is_drop() {
            // The top held slot leaves the hand and an unpromoted board piece
            // appears at `to`.
            let kind = mv.dropped_piece_kind();
            let count = pos.hand(mover).count(kind) as usize;
            removed[0] = Some(Dirty::Hand {
                color: mover,
                kind,
                count,
            });
            added[0] = Some(Dirty::Board {
                sq: to,
                piece: Piece::new(kind, mover),
            });
        } else {
            let from = mv.from_sq();
            // The `Move` encoding already carries the moving piece, so there is
            // no need to re-read the board. `before` undoes any promotion.
            let after = mv.moved_piece_after();
            let before = if mv.is_promote() {
                Piece {
                    promoted: false,
                    ..after
                }
            } else {
                after
            };
            debug_assert_eq!(
                pos.board().get(from),
                Some(before),
                "Move encoding disagrees with the board on the mover at {from:?}",
            );
            removed[0] = Some(Dirty::Board {
                sq: from,
                piece: before,
            });
            added[0] = Some(Dirty::Board {
                sq: to,
                piece: after,
            });

            // The victim leaves the board and enters the mover's hand as its
            // unpromoted kind, at the next count slot.
            if let Some(captured) = pos.board().get(to) {
                removed[1] = Some(Dirty::Board {
                    sq: to,
                    piece: captured,
                });
                let base = captured.kind;
                let count = pos.hand(mover).count(base) as usize + 1;
                added[1] = Some(Dirty::Hand {
                    color: mover,
                    kind: base,
                    count,
                });
            }
        }

        let encode_half = |persp: Color| -> Option<PerspectiveDelta> {
            if requires_full_refresh(mv, persp) {
                return None;
            }
            // The perspective's own king did not move, so its `(sq_k_code,
            // mirror)` is the same pre- and post-move.
            let king_persp = from_persp(king_square(pos, persp), persp);
            let mirror = needs_mirror(king_persp);
            let sq_k_code = mirror_if_needed(king_persp, mirror).index() as usize;

            let mut pd = PerspectiveDelta::default();
            for d in removed.iter().flatten() {
                pd.removed[pd.n_removed] = d.encode(persp, sq_k_code, mirror);
                pd.n_removed += 1;
            }
            for d in added.iter().flatten() {
                pd.added[pd.n_added] = d.encode(persp, sq_k_code, mirror);
                pd.n_added += 1;
            }
            Some(pd)
        };

        MoveDelta {
            halves: [encode_half(Color::Black), encode_half(Color::White)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attic_state::{parse_sfen, parse_usi_move};
    use std::collections::HashSet;

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    // Sparse and hand-heavy: 6 board pieces and 6 held ones.
    const DROP_HEAVY: &str = "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1";

    fn sorted(list: &[FeatureIndex]) -> Vec<FeatureIndex> {
        let mut v = list.to_vec();
        v.sort_unstable();
        v
    }

    fn assert_no_duplicates(list: &[FeatureIndex]) {
        let set: HashSet<_> = list.iter().copied().collect();
        assert_eq!(set.len(), list.len(), "feature list has duplicate indices");
    }

    fn assert_in_bounds(list: &[FeatureIndex]) {
        for &idx in list {
            assert!(
                (idx as usize) < FEATURE_DIMENSION,
                "feature index {idx} out of bounds",
            );
        }
    }

    /// Horizontal mirror of `pos`: every board piece file `f -> 8 - f`, with
    /// ranks and hands unchanged.
    fn mirror_position(pos: &Position) -> Position {
        let mut mirrored = Position::empty();
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if let Some(piece) = pos.board().get(sq) {
                let msq = Square::new(Square::FILES - 1 - sq.file(), sq.rank()).unwrap();
                mirrored.board_mut().set(msq, Some(piece));
            }
        }
        for color in [Color::Black, Color::White] {
            for kind in HAND_KINDS {
                for _ in 0..pos.hand(color).count(kind) {
                    mirrored.hand_mut(color).increment(kind);
                }
            }
        }
        mirrored.set_side_to_move(pos.side_to_move());
        mirrored
    }

    #[test]
    fn king_square_matches_scan_oracle() {
        for sfen in [STARTPOS, DROP_HEAVY] {
            let pos = parse_sfen(sfen).unwrap();
            for color in [Color::Black, Color::White] {
                assert_eq!(
                    king_square(&pos, color),
                    king_square_scan(&pos, color),
                    "sfen `{sfen}` color {color:?}",
                );
            }
        }
    }

    #[test]
    fn plane_constants_match_reference() {
        assert_eq!(FE_HAND_END, 90);
        assert_eq!(FE_END, 1_548);
        assert_eq!(E_KING, 1_629);
        assert_eq!(FEATURE_DIMENSION, NUM_FEATURES);
        assert_eq!(FEATURE_DIMENSION, SQ_K_COUNT * E_KING);
    }

    #[test]
    fn startpos_has_forty_features_per_perspective() {
        let pos = parse_sfen(STARTPOS).unwrap();
        for persp in [Color::Black, Color::White] {
            let list = active_features(&pos, persp);
            assert_eq!(
                list.len(),
                MAX_ACTIVE_FEATURES,
                "perspective {persp:?}: expected 40 active features",
            );
            assert_no_duplicates(&list);
            assert_in_bounds(&list);
        }
    }

    #[test]
    fn startpos_both_kings_share_one_plane() {
        let pos = parse_sfen(STARTPOS).unwrap();
        for persp in [Color::Black, Color::White] {
            let own_king_persp = from_persp(king_square(&pos, persp), persp);
            let mirror = needs_mirror(own_king_persp);
            let sq_k_code = mirror_if_needed(own_king_persp, mirror).index() as usize;
            let plane_lo = (E_KING * sq_k_code + FE_END) as FeatureIndex;
            let plane_hi = (E_KING * sq_k_code + E_KING) as FeatureIndex;

            let list = active_features(&pos, persp);
            let kings = list
                .iter()
                .filter(|&&f| f >= plane_lo && f < plane_hi)
                .count();
            assert_eq!(kings, 2, "both kings should land in the shared king plane");
        }
    }

    #[test]
    fn active_features_is_deterministic() {
        let pos = parse_sfen(STARTPOS).unwrap();
        assert_eq!(
            active_features(&pos, Color::Black),
            active_features(&pos, Color::Black),
        );
    }

    #[test]
    fn both_perspectives_helper_matches_single() {
        let pos = parse_sfen(DROP_HEAVY).unwrap();
        let both = active_features_both(&pos);
        assert_eq!(
            both[Color::Black.index()],
            active_features(&pos, Color::Black)
        );
        assert_eq!(
            both[Color::White.index()],
            active_features(&pos, Color::White)
        );
    }

    #[test]
    fn drop_heavy_extracts_cleanly_and_indexes_hands() {
        let pos = parse_sfen(DROP_HEAVY).unwrap();
        for persp in [Color::Black, Color::White] {
            let list = active_features(&pos, persp);
            assert_eq!(list.len(), MAX_ACTIVE_FEATURES, "perspective {persp:?}");
            assert_in_bounds(&list);

            // The real features are distinct; the padding slots all repeat the
            // single BONA_PIECE_ZERO feature for this perspective's king.
            let own_king_persp = from_persp(king_square(&pos, persp), persp);
            let mirror = needs_mirror(own_king_persp);
            let sq_k_code = mirror_if_needed(own_king_persp, mirror).index() as usize;
            let pad = (E_KING * sq_k_code) as FeatureIndex;
            assert_eq!(
                list.iter().filter(|&&f| f == pad).count(),
                28,
                "perspective {persp:?}: expected 28 padding features",
            );
            let distinct: HashSet<_> = list.iter().copied().collect();
            assert_eq!(
                distinct.len(),
                13,
                "perspective {persp:?}: 12 distinct real + 1 padding value",
            );
        }
    }

    #[test]
    fn every_position_yields_exactly_forty_features() {
        // A full and a sparse position must both produce exactly
        // PIECE_NUMBER_NB features per perspective.
        for sfen in [STARTPOS, DROP_HEAVY] {
            let pos = parse_sfen(sfen).unwrap();
            for persp in [Color::Black, Color::White] {
                assert_eq!(
                    active_features(&pos, persp).len(),
                    MAX_ACTIVE_FEATURES,
                    "sfen `{sfen}` perspective {persp:?}",
                );
            }
        }
    }

    #[test]
    fn hand_piece_indices_span_consecutive_slots() {
        // Each side holds two pawns and a gold, so from Black's perspective the
        // friend pawns occupy `F_HAND_PAWN+{1,2}` and the enemy ones
        // `E_HAND_PAWN+{1,2}`, all offset by the same king block.
        let pos = parse_sfen(DROP_HEAVY).unwrap();
        let own_king_persp = from_persp(king_square(&pos, Color::Black), Color::Black);
        let mirror = needs_mirror(own_king_persp);
        let sq_k_code = mirror_if_needed(own_king_persp, mirror).index() as usize;
        let base = (E_KING * sq_k_code) as FeatureIndex;

        let list: HashSet<_> = active_features(&pos, Color::Black).into_iter().collect();
        for expected in [
            base + (F_HAND_PAWN + 1) as FeatureIndex,
            base + (F_HAND_PAWN + 2) as FeatureIndex,
            base + (F_HAND_GOLD + 1) as FeatureIndex,
            base + (E_HAND_PAWN + 1) as FeatureIndex,
            base + (E_HAND_PAWN + 2) as FeatureIndex,
            base + (E_HAND_GOLD + 1) as FeatureIndex,
        ] {
            assert!(list.contains(&expected), "missing hand feature {expected}");
        }
    }

    #[test]
    fn mirror_property_preserves_feature_set() {
        // Both kings sit off the centre file, so the mirror flag flips between
        // a position and its mirror and the two share one feature set per
        // perspective. A centre-file king is a genuinely distinct position.
        let sfen = "2k6/9/4+R4/8P/9/1n7/9/9/6K2 b B2Pgp 1";
        let pos = parse_sfen(sfen).unwrap();
        let mirrored = mirror_position(&pos);
        for persp in [Color::Black, Color::White] {
            assert_eq!(
                sorted(&active_features(&pos, persp)),
                sorted(&active_features(&mirrored, persp)),
                "perspective {persp:?}: mirror changed the feature set",
            );
        }
    }

    /// Reconstruct the `after` multiset from `before + added - removed`, an
    /// independent oracle for [`changed_indices`].
    fn apply_multiset(
        before: &[FeatureIndex],
        removed: &[FeatureIndex],
        added: &[FeatureIndex],
    ) -> Vec<FeatureIndex> {
        let mut counts: std::collections::BTreeMap<FeatureIndex, i32> =
            std::collections::BTreeMap::new();
        for &x in before {
            *counts.entry(x).or_insert(0) += 1;
        }
        for &x in removed {
            *counts.entry(x).or_insert(0) -= 1;
        }
        for &x in added {
            *counts.entry(x).or_insert(0) += 1;
        }
        let mut out = Vec::new();
        for (&k, &c) in &counts {
            assert!(c >= 0, "negative multiplicity for {k}");
            for _ in 0..c {
                out.push(k);
            }
        }
        out
    }

    /// Diff the pre- and post-move feature lists for a perspective, as the
    /// accumulator does, and confirm the delta reconstructs the post-move
    /// multiset.
    fn check_diff_invariant(sfen: &str, usi: &str) {
        let mut pos = parse_sfen(sfen).unwrap();
        let mv = parse_usi_move(usi, &pos).unwrap();
        for persp in [Color::Black, Color::White] {
            assert!(
                !requires_full_refresh(mv, persp),
                "test move {usi} must be incremental for {persp:?}"
            );
            let before = active_features(&pos, persp);
            let undo = pos.do_move(mv);
            let after = active_features(&pos, persp);
            pos.undo_move(mv, undo);

            let (removed, added) = changed_indices(&before, &after);
            assert_eq!(
                apply_multiset(&before, &removed, &added),
                sorted(&after),
                "{persp:?}: diff does not reconstruct post-move features for {usi}",
            );
        }
    }

    #[test]
    fn requires_full_refresh_flags_only_own_king_moves() {
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let king_move = parse_usi_move("5i5h", &pos).unwrap();
        assert!(requires_full_refresh(king_move, Color::Black));
        assert!(!requires_full_refresh(king_move, Color::White));

        let start = parse_sfen(STARTPOS).unwrap();
        let pawn = parse_usi_move("7g7f", &start).unwrap();
        assert!(!requires_full_refresh(pawn, Color::Black));
        assert!(!requires_full_refresh(pawn, Color::White));
    }

    #[test]
    fn requires_full_refresh_drops_are_never_refresh() {
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
        let drop = parse_usi_move("P*5f", &pos).unwrap();
        assert!(!requires_full_refresh(drop, Color::Black));
        assert!(!requires_full_refresh(drop, Color::White));
    }

    #[test]
    fn changed_indices_reconstructs_after_for_all_move_types() {
        check_diff_invariant(STARTPOS, "7g7f");
        check_diff_invariant("4k4/1p7/9/9/9/9/9/1R7/4K4 b - 1", "8h8b");
        check_diff_invariant("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e");
        check_diff_invariant("4k4/9/9/1P7/9/9/9/9/4K4 b - 1", "8d8c+");
        check_diff_invariant("4k4/2p6/9/9/9/9/9/1B7/4K4 b - 1", "8h7b+");
    }

    #[test]
    fn changed_indices_of_identical_lists_is_empty() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let feats = active_features(&pos, Color::Black);
        let (removed, added) = changed_indices(&feats, &feats);
        assert!(removed.is_empty() && added.is_empty());
    }

    #[test]
    fn mirror_property_needs_mirror_flag_actually_flips() {
        // Exactly one of the two positions must trigger the mirror path for each
        // perspective, so that the test above is not a trivial no-op.
        let sfen = "2k6/9/4+R4/8P/9/1n7/9/9/6K2 b B2Pgp 1";
        let pos = parse_sfen(sfen).unwrap();
        let mirrored = mirror_position(&pos);
        for persp in [Color::Black, Color::White] {
            let m_pos = needs_mirror(from_persp(king_square(&pos, persp), persp));
            let m_mir = needs_mirror(from_persp(king_square(&mirrored, persp), persp));
            assert_ne!(
                m_pos, m_mir,
                "perspective {persp:?}: mirror flag did not flip"
            );
        }
    }
}
