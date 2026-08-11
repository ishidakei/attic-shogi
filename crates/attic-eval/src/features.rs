//! `HalfKA_hm2` active-feature index extraction over `attic-state::Position`.
//!
//! SFNN-1536 uses `Features::FeatureSet<Features::HalfKA_hm2<Side::kFriend>>`
//! (see `eval/nnue/architectures/sfnn-1536.h`). This
//! module ports the index math of that feature set — full-refresh path only —
//! onto this workspace's `Position` API.
//!
//! The C++ ground truth lives at
//! `eval/nnue/features/half_ka_hm2.{h,cpp}`; the
//! index arithmetic is a faithful port of the read-only Rust NNUE reference
//! implementation's `features.rs`, adapted from that reference's Position
//! API to `attic-state`. The two coordinate systems already agree — an
//! `attic-state::Square` index is `file * 9 + rank` with file `0` = shogi file
//! `1` and rank `0` = shogi rank `a`, byte-identical to YaneuraOu's `Square`
//! numbering — so the arithmetic carries over unchanged.
//!
//! ## Feature layout (identical to the reference / C++ `HalfKA_hm2`)
//!
//! A feature index is `E_KING * sq_k_code + p_adj`, where:
//! - `sq_k_code` is the perspective's own-king square, horizontally mirrored
//!   into files 1–5 (`0..45`). Mirroring the whole position around the king's
//!   file is the `hm` ("horizontal mirror") canonicalization.
//! - `p_adj` selects a plane and a square/count within it: hand planes
//!   `[0, 90)`, board planes `[90, 1548)` (friend/enemy interleaved per piece
//!   type), and a single shared king plane `[1548, 1629)` that both kings
//!   collapse into.
//!
//! The accumulator itself, and the application of a changed-index delta to it,
//! live in [`crate::transformer`].

use attic_state::{Color, Move, Piece, PieceKind, Position, Square};

use crate::types::NUM_FEATURES;

/// A single active input-feature index into the `HalfKA_hm2` feature space.
pub type FeatureIndex = u32;

/// `BONA_PIECE_ZERO`: the feature-space origin the reference initialises empty
/// piece slots to (`EvalList::clear()`). Used to pad sparse positions up to the
/// fixed 40 piece slots the reference iterates.
const BONA_PIECE_ZERO: usize = 0;

// --- Hand planes ---------------------------------------------------------
// `F_*` is the friend (own-side) plane base, `E_*` the enemy plane base. The
// per-piece span leaves room for one index per possible held count (a pawn can
// be held up to 18 times, so its plane spans 19 slots — slot 0 is the unused
// "zero" pad, counts land at `base + 1 ..= base + count`).
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

// --- Board planes --------------------------------------------------------
// One 81-square plane per (side, effective piece type). Promoted minor pieces
// collapse onto the gold plane; promoted bishop/rook land on the horse/dragon
// planes. Ordering matches YaneuraOu's `BonaPiece` board layout.
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

/// Total `HalfKA_hm2` feature dimension: `SQ_K_COUNT * E_KING`. Kept equal to
/// [`crate::types::NUM_FEATURES`] (asserted in tests).
pub const FEATURE_DIMENSION: usize = SQ_K_COUNT * E_KING;

/// Maximum number of simultaneously-active features (`PIECE_NUMBER_NB`): every
/// legal shogi position has exactly 40 pieces split across board and hands, so
/// a full position yields exactly this many active indices.
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

/// Rotate a square 180° when viewing the board from `persp`'s side. Black keeps
/// the square as-is; White flips it (the enemy king / mirror math then operates
/// in a canonical own-side-forward frame).
#[inline]
fn from_persp(sq: Square, persp: Color) -> Square {
    match persp {
        Color::Black => sq,
        // 180° rotation == SQ_NB - 1 - index.
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
///
/// Promoted pawn/lance/knight/silver collapse to the gold plane; promoted
/// bishop/rook to horse/dragon; gold never promotes (both slots stay gold). The
/// king row is a sentinel — kings have no board plane and the caller routes
/// them to the shared king plane. Branch-free index-table form of the
/// reference's piece-to-`BonaPiece`-base mapping.
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

/// Friend/enemy board-plane base for a board piece. Promoted pawn/lance/knight/
/// silver collapse to the gold plane; promoted bishop/rook to horse/dragon.
///
/// Kings have no board plane — the caller routes them to the shared king plane,
/// so passing a king is a caller bug, guarded by `debug_assert!`.
#[inline]
fn board_plane(kind: PieceKind, promoted: bool, is_friend: bool) -> usize {
    debug_assert!(
        kind != PieceKind::King,
        "king has no board plane; caller handles it"
    );
    let (friend, enemy) = BOARD_PLANE[kind.index()][promoted as usize];
    if is_friend { friend } else { enemy }
}

/// Friend/enemy hand-plane base for a held piece kind.
///
/// Kings are never held in hand, so passing a king is a caller bug, guarded by
/// `debug_assert!`.
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

/// Locate `color`'s king on the board via the O(1) piece-set accessor
/// [`Position::king_square`].
///
/// # Panics
/// Panics if `color` has no king — every position fed to the evaluator must
/// have both kings, matching the reference's `pos.king_square` contract.
pub(crate) fn king_square(pos: &Position, color: Color) -> Square {
    pos.king_square(color)
        .unwrap_or_else(|| panic!("position has no {color:?} king"))
}

/// An 81-square mailbox scan, the `#[cfg(test)]` equivalence oracle for
/// [`king_square`] (mirrors `attic-state`'s `try_find_king_scan`).
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

/// Active `HalfKA_hm2` feature indices for one `perspective` of `pos`.
///
/// Returns exactly [`MAX_ACTIVE_FEATURES`] indices: one per board piece, one per
/// held-piece instance, then `BONA_PIECE_ZERO` padding (see below) for every
/// empty piece slot. A legal position has all 40 slots filled and needs no
/// padding; a sparse (piece-dropped) position is padded up to 40 with repeated
/// `E_KING * sq_k_code + 0` features. Every index is `< FEATURE_DIMENSION`. The
/// list may contain duplicates (the padding features, and only those).
///
/// ## Why padding
///
/// The reference builds each accumulator half by iterating a *fixed* 40
/// piece-number slots (`AppendActiveIndices` over `PIECE_NUMBER_NB`, with the
/// two kings at slots 38/39). `EvalList::clear()` initialises every unused slot
/// to `BONA_PIECE_ZERO` (`0`), so each absent piece contributes the feature
/// `MakeIndex(sq_k, 0) = E_KING * sq_k_code + 0`. Because the accumulator *sums*
/// feature columns, these repeated zero-features shift the result and must be
/// reproduced exactly, or a sparse-position evaluation diverges from ground
/// truth.
///
/// # Panics
/// Panics if `pos` is missing the `perspective` side's king.
pub fn active_features(pos: &Position, perspective: Color) -> Vec<FeatureIndex> {
    let mut list = Vec::with_capacity(MAX_ACTIVE_FEATURES);

    let own_king_persp = from_persp(king_square(pos, perspective), perspective);
    let mirror = needs_mirror(own_king_persp);
    let sq_k_code = mirror_if_needed(own_king_persp, mirror).index() as usize;

    // Board pieces.
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        let Some(piece) = pos.board().get(sq) else {
            continue;
        };
        let is_friend = piece.color == perspective;
        let sq_persp = mirror_if_needed(from_persp(sq, perspective), mirror);
        let sq_code = sq_persp.index() as usize;

        let p_adj = if piece.kind == PieceKind::King {
            // Both kings collapse into the shared `[FE_END, E_KING)` plane.
            FE_END + sq_code
        } else {
            board_plane(piece.kind, piece.promoted, is_friend) + sq_code
        };
        list.push(encode_feature(sq_k_code, p_adj));
    }

    // Hand pieces: the k-th held piece of a kind lands at `base + k`.
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

    // Pad empty piece slots with the `BONA_PIECE_ZERO` feature, matching the
    // reference's fixed 40-slot iteration (see the doc comment above). A legal
    // 40-piece position is already full and skips this loop.
    debug_assert!(list.len() <= MAX_ACTIVE_FEATURES);
    while list.len() < MAX_ACTIVE_FEATURES {
        list.push(encode_feature(sq_k_code, BONA_PIECE_ZERO));
    }

    list
}

/// Active features for both perspectives, indexed by [`Color::index`]
/// (`[0]` = Black, `[1]` = White). Convenience for consumers that refresh both
/// accumulators from a single position.
pub fn active_features_both(pos: &Position) -> [Vec<FeatureIndex>; Color::COUNT] {
    [
        active_features(pos, Color::Black),
        active_features(pos, Color::White),
    ]
}

/// Whether `perspective`'s accumulator half must be fully refreshed after `mv`
/// (rather than updated incrementally).
///
/// A `HalfKA_hm2` feature index is `own_king_relative` (it embeds the
/// perspective's own-king `sq_k_code`). When that own king moves, *every* index
/// for the perspective shifts — and the horizontal-mirror flag may flip too —
/// so no add/sub delta can express the change and the whole half is rebuilt.
/// Drops never move the king, and a non-drop move only forces a refresh for the
/// side whose king is the moved piece. Faithful port of the reference
/// `requires_full_refresh` (`half_ka_hm2.cpp`'s king-move / mirror-boundary
/// rule); the mirror-boundary case is subsumed here because only a king move
/// can change the perspective's own-king file.
pub fn requires_full_refresh(mv: Move, perspective: Color) -> bool {
    if mv.is_drop() {
        return false;
    }
    // A king never promotes, so `moved_piece_after` reports the moving king
    // directly for a king move.
    let piece = mv.moved_piece_after();
    piece.kind == PieceKind::King && piece.color == perspective
}

/// Sorted-merge multiset difference of two active-feature lists: returns
/// `(removed, added)` such that applying `removed` then `added` to the `before`
/// multiset yields the `after` multiset.
///
/// Both inputs are the fixed-length ([`MAX_ACTIVE_FEATURES`]) lists produced by
/// [`active_features`], including their `BONA_PIECE_ZERO` padding. Because a
/// legal move conserves total piece count, the padding multiplicity is equal on
/// both sides (for any perspective updated incrementally, i.e. whose own king
/// did not move), so the padding features cancel and never appear in either
/// output. Faithful port of the reference `feature_diff`'s merge loop.
pub(crate) fn changed_indices(
    before: &[FeatureIndex],
    after: &[FeatureIndex],
) -> (Vec<FeatureIndex>, Vec<FeatureIndex>) {
    let mut sorted_before = before.to_vec();
    let mut sorted_after = after.to_vec();
    sorted_before.sort_unstable();
    sorted_after.sort_unstable();

    let mut removed = Vec::new();
    let mut added = Vec::new();
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
    (removed, added)
}

// --- Move-derived (dirty-piece) accumulator delta ------------------------
//
// The hot search path cannot afford [`active_features`]'s full 40-slot scan
// (plus sort + merge) at every `do_move`. YaneuraOu instead tracks the handful
// of "dirty" pieces a move touches and rewrites only their feature columns.
// [`MoveDelta::from_move`] is that dirty-piece form: it reads the pre-move
// board / hands directly from the move (mover from/to with promotion, the
// captured piece's board removal and the capturer's hand gain, or a drop's
// hand removal + board addition) and, for every perspective whose own king did
// NOT move, encodes the exact add/sub feature indices `active_features` would
// have differed by. A perspective whose own king moved is flagged for a full
// refresh (its whole own-king-relative index space shifts).
//
// Bit-exactness: for a perspective updated incrementally, `refresh(before) +
// added_columns - removed_columns == refresh(after)` under wrapping `i16`,
// because the only `active_features` entries that changed between the pre- and
// post-move positions are exactly these dirty pieces (the total board+hand
// piece count is conserved by every legal move, so the `BONA_PIECE_ZERO`
// padding multiplicity is unchanged and cancels). This is verified BIT-FOR-BIT
// against both [`Accumulator::refresh`] and the scan-based
// [`Accumulator::update_after_move`] in the transformer's oracle tests.

/// One board or hand slot a move touches, in a perspective-independent form.
/// Encoded per perspective into a `HalfKA_hm2` feature index by [`Dirty::encode`].
#[derive(Clone, Copy)]
enum Dirty {
    /// A piece sitting on `sq` (a king is encoded via the shared king plane).
    Board { sq: Square, piece: Piece },
    /// The `count`-th (1-based) held piece of `kind` in `color`'s hand — the
    /// slot `active_features` emits as `hand_plane(kind, ..) + count`.
    Hand {
        color: Color,
        kind: PieceKind,
        count: usize,
    },
}

impl Dirty {
    /// Encode this slot into the feature index for `persp`, whose own-king code
    /// is `(sq_k_code, mirror)`. Byte-for-byte the same arithmetic
    /// [`active_features`] applies to the same slot.
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
/// columns change per side (mover + captured, or a drop's hand + board), so the
/// lists are fixed two-slot arrays — no heap allocation on the hot path.
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
///
/// `half(color)` is `None` when that perspective must be fully refreshed (its
/// own king moved) and `Some(delta)` otherwise. Consumed by
/// [`crate::Accumulator::derive_into`], which the search threads through its
/// per-worker accumulator stack.
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

    /// Compute the delta for `mv` from the **pre-move** `pos`.
    ///
    /// Reads only the mover's origin square, the captured square, and the
    /// relevant hand counts — a handful of accesses, versus the 81-square +
    /// hand scan [`active_features`] performs. `pos` is not mutated.
    ///
    /// # Panics
    /// Panics if a non-drop move has no piece on its origin square, or if an
    /// incrementally-updated perspective is missing its king.
    pub fn from_move(pos: &Position, mv: Move) -> MoveDelta {
        let mover = pos.side_to_move();
        let to = mv.to_sq();

        // Perspective-independent dirty slots: at most two removed, two added.
        let mut removed: [Option<Dirty>; 2] = [None, None];
        let mut added: [Option<Dirty>; 2] = [None, None];

        if mv.is_drop() {
            // Drop: the top held slot leaves the hand; a fresh (unpromoted) board
            // piece appears at `to`.
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
            // The `Move` encoding already carries the moving piece (bits 16..21),
            // so take it from there instead of re-reading the board. `before` is
            // the post-move piece with any promotion undone; a non-promoting move
            // leaves it unchanged.
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

            // Capture: the victim leaves the board and enters the mover's hand as
            // its base (unpromoted) kind, at the next count slot.
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
            // mirror)` is the same pre- and post-move; read it from `pos`.
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
    // Sparse, hand-heavy: 6 board pieces + 6 held pieces = 12 total.
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

    /// Horizontal mirror of `pos`: every board piece file `f -> 8 - f`, ranks
    /// and hands unchanged (a file mirror leaves held pieces untouched).
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
        // The O(1) piece-set lookup must agree square-for-square with an
        // 81-square scan on every position the evaluator sees.
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
            // 6 board pieces + 6 held pieces = 12 real, padded up to 40.
            assert_eq!(list.len(), MAX_ACTIVE_FEATURES, "perspective {persp:?}");
            assert_in_bounds(&list);

            // The 12 real features are distinct; the 28 padding slots all repeat
            // the single BONA_PIECE_ZERO feature for this perspective's king.
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
        // Both a full (startpos) and a sparse (drop-heavy) position must produce
        // exactly PIECE_NUMBER_NB features per perspective — the reference's
        // fixed-slot invariant.
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
        // Black holds 2 pawns + 1 gold; White holds 2 pawns + 1 gold. From
        // Black's perspective the friend pawns occupy F_HAND_PAWN+{1,2}, the
        // friend gold F_HAND_GOLD+1, the enemy pawns E_HAND_PAWN+{1,2}, the
        // enemy gold E_HAND_GOLD+1 — all offset by the same king block.
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
        // Both kings sit off the centre file, so the `hm` canonicalization
        // makes a position and its horizontal mirror share one feature set for
        // each perspective (the mirror flag flips between the two, cancelling
        // out). A centre-file king would break this — that case is a genuinely
        // distinct position, not a mirror-equivalent one.
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

    // --- Incremental-update support: refresh classification + diff ---------

    /// Reconstruct the `after` multiset from `before + added - removed`, sorted,
    /// as an independent oracle for [`changed_indices`].
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

    /// Drive `changed_indices` the way the accumulator does: diff the pre- and
    /// post-move feature lists for a perspective and confirm the delta
    /// reconstructs the post-move multiset. Asserts the move is incremental for
    /// the perspective (no own-king move).
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
            // Padding features cancel: neither list may contain the pad index.
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
        // Quiet, capture, drop, promotion, capture-promotion — each incremental
        // for both perspectives.
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
        // Guard the test above: confirm exactly one of (pos, mirror) triggers
        // the horizontal-mirror path for each perspective, so the equality is
        // exercising the mirror math rather than a trivial no-op.
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
