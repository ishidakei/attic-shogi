//! Attack detection, king lookup, and the drop-legality (`uchifuzume`)
//! predicate — the board-query helpers shared by the search-side move
//! generators and the mate / SEE oracles.
//!
//! Legal-move generation itself lives in [`crate::search_movegen`]: the single
//! entry point is [`Position::generate_legal_all`] (the `generate<LEGAL_ALL>`
//! port), and every consumer routes through it — there is no second,
//! repetition-aware legal-move pipeline.
//!
//! What this module holds is the query surface those generators (and the
//! `#[cfg(test)]` oracles) consume: [`is_attacked_by`], [`find_king`] /
//! [`try_find_king`], `attackers_bb` / `attackers_bb_occ`, and the in-place
//! uchifuzume port [`drop_is_uchifuzume`] / [`Position::legal_drop`]. The
//! reference enforces `uchifuzume` (打ち歩詰め) and `nifu` (二歩) at
//! drop-generation time inside `GenerateDropMoves` (mirrored in
//! `search_movegen`), so no legal-move list is filtered for them after the
//! fact.
//!
//! The attack-detection design and the in-place `uchifuzume` probe shape
//! adopted here are documented at their definition sites below.

use crate::board::Board;
use crate::color::Color;
use crate::move_::Move;
use crate::piece::PieceKind;
// `Piece` is consumed only by the `#[cfg(test)]` scanning oracles / unit tests;
// the production query surface works in `PieceKind`s and bitboard patterns.
#[cfg(test)]
use crate::piece::Piece;
use crate::position::Position;
use crate::square::Square;

pub(crate) const PAWN_STEPS: &[(i8, i8)] = &[(0, -1)];
pub(crate) const KNIGHT_STEPS: &[(i8, i8)] = &[(1, -2), (-1, -2)];
pub(crate) const SILVER_STEPS: &[(i8, i8)] = &[(0, -1), (1, -1), (-1, -1), (1, 1), (-1, 1)];
pub(crate) const GOLD_STEPS: &[(i8, i8)] = &[(0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1)];
pub(crate) const KING_STEPS: &[(i8, i8)] = &[
    (0, -1),
    (1, -1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (1, 1),
    (-1, 1),
];
// The direction tables below feed only [`movement`], which is consumed solely
// by the `#[cfg(test)]` scanning oracles (the production path reads the
// precomputed bitboard attack tables). They are therefore
// `#[cfg(test)]` too. The step tables (`PAWN_STEPS` … `KING_STEPS`) stay
// ungated because `crate::bitboard` builds its production attack tables from
// them.
#[cfg(test)]
const KING_ORTH_STEPS: &[(i8, i8)] = &[(0, -1), (1, 0), (-1, 0), (0, 1)];
#[cfg(test)]
const KING_DIAG_STEPS: &[(i8, i8)] = &[(1, -1), (-1, -1), (1, 1), (-1, 1)];
#[cfg(test)]
const LANCE_DIRS: &[(i8, i8)] = &[(0, -1)];
#[cfg(test)]
const BISHOP_DIRS: &[(i8, i8)] = &[(1, -1), (-1, -1), (1, 1), (-1, 1)];
#[cfg(test)]
const ROOK_DIRS: &[(i8, i8)] = &[(0, -1), (0, 1), (1, 0), (-1, 0)];
#[cfg(test)]
const NO_DIRS: &[(i8, i8)] = &[];

pub(crate) fn step_signed(sq: Square, df: i8, dr: i8) -> Option<Square> {
    let f = sq.file() as i8 + df;
    let r = sq.rank() as i8 + dr;
    if (0..Square::FILES as i8).contains(&f) && (0..Square::RANKS as i8).contains(&r) {
        Square::new(f as u8, r as u8)
    } else {
        None
    }
}

pub(crate) fn is_in_promotion_zone(sq: Square, color: Color) -> bool {
    match color {
        Color::Black => sq.rank() < 3,
        Color::White => sq.rank() >= 6,
    }
}

#[cfg(test)]
type StepTable = &'static [(i8, i8)];

#[cfg(test)]
pub(crate) fn movement(piece: Piece) -> (StepTable, StepTable) {
    match (piece.kind, piece.promoted) {
        (PieceKind::Pawn, false) => (PAWN_STEPS, NO_DIRS),
        (PieceKind::Lance, false) => (NO_DIRS, LANCE_DIRS),
        (PieceKind::Knight, false) => (KNIGHT_STEPS, NO_DIRS),
        (PieceKind::Silver, false) => (SILVER_STEPS, NO_DIRS),
        (PieceKind::Gold, _)
        | (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => {
            (GOLD_STEPS, NO_DIRS)
        }
        (PieceKind::Bishop, false) => (NO_DIRS, BISHOP_DIRS),
        (PieceKind::Bishop, true) => (KING_ORTH_STEPS, BISHOP_DIRS),
        (PieceKind::Rook, false) => (NO_DIRS, ROOK_DIRS),
        (PieceKind::Rook, true) => (KING_DIAG_STEPS, ROOK_DIRS),
        (PieceKind::King, _) => (KING_STEPS, NO_DIRS),
    }
}

#[cfg(test)]
pub(crate) fn dr_sign_for(color: Color) -> i8 {
    match color {
        Color::Black => 1,
        Color::White => -1,
    }
}

pub(crate) fn find_king(board: &Board, color: Color) -> Square {
    try_find_king(board, color).unwrap_or_else(|| panic!("position has no {color:?} king"))
}

/// Like `find_king`, but returns `None` instead of panicking when no king of
/// `color` is on the board. Used by callers that may legitimately observe a
/// transient king-less scratch state (e.g. `Position::do_move` recomputing
/// `gives_check` for the opposite side after a pseudo-legal probe move that
/// happened to capture the king).
pub(crate) fn try_find_king(board: &Board, color: Color) -> Option<Square> {
    // The KING piece set is kept in sync by the single `Board::set` /
    // `toggle_sets` mutation funnel, mirroring the reference's
    // `update_kingSquare()` deriving the square from `pieces(c, KING)`; the
    // derived square is not cached in a field because the set read is already
    // O(1).
    //
    // Equivalence with the 81-square scan oracle: the two-lane bitboard
    // iterates squares in ascending index order (lane 0 covering 0..=62, then
    // lane 1 with a +63 bias — see `BitboardIter`), so its lowest set square is
    // the scan's first match. A king-less board gives an empty set — this
    // covers the transient king-captured scratch states above, because
    // `Board::set` removes the captured king from the piece sets at the same
    // moment it clears the square.
    board
        .pieces_pattern(color, crate::board::pat::KING)
        .squares()
        .next()
}

/// An 81-square scan, the `#[cfg(test)]` equivalence oracle for
/// [`try_find_king`] (same pattern as [`is_attacked_by_scan`]).
#[cfg(test)]
pub(crate) fn try_find_king_scan(board: &Board, color: Color) -> Option<Square> {
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).unwrap();
        if let Some(p) = board.get(sq)
            && p.color == color
            && p.kind == PieceKind::King
        {
            return Some(sq);
        }
    }
    None
}

// The three helpers below serve only the `#[cfg(test)]` scanning oracle
// [`is_attacked_by_scan`]; the production path reads the piece sets instead.
#[cfg(test)]
fn is_gold_like(p: Piece) -> bool {
    if p.kind == PieceKind::Gold {
        return true;
    }
    if !p.promoted {
        return false;
    }
    matches!(
        p.kind,
        PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver
    )
}

#[cfg(test)]
fn matches_unpromoted_kind(board: &Board, sq: Square, color: Color, kind: PieceKind) -> bool {
    match board.get(sq) {
        Some(p) => p.color == color && p.kind == kind && !p.promoted,
        None => false,
    }
}

#[cfg(test)]
fn scan_slider(
    board: &Board,
    sq: Square,
    df: i8,
    dr: i8,
    color: Color,
    kind: PieceKind,
    require_unpromoted: bool,
) -> bool {
    let mut cur = sq;
    loop {
        cur = match step_signed(cur, df, dr) {
            Some(s) => s,
            None => return false,
        };
        match board.get(cur) {
            None => continue,
            Some(p) => {
                if p.color != color || p.kind != kind {
                    return false;
                }
                if require_unpromoted && p.promoted {
                    return false;
                }
                return true;
            }
        }
    }
}

/// True iff `sq` is attacked by any piece of `attacker` on `board`.
///
/// Bitboard form (the reference `attackers_to` reverse-symmetry): a `attacker`
/// piece on `s` attacks `sq` iff, imagining a piece of that pattern on `sq` with
/// the attacker's perspective flipped for the asymmetric steppers / lance, it
/// reaches `s`. So each pattern's attackers are `<reverse attack of sq> &
/// board.pieces_pattern(attacker, pattern)`, read straight off the incrementally
/// maintained piece sets — no 81-square scan. Horse / dragon fold the king ring
/// into the slider rays (`HORSE = BISHOP | ring`, `DRAGON = ROOK | ring`). The
/// former scanning implementation is retained as the `#[cfg(test)]` oracle
/// [`is_attacked_by_scan`].
pub(crate) fn is_attacked_by(board: &Board, sq: Square, attacker: Color) -> bool {
    use crate::bitboard::{
        bishop_attacks, gold_attacks, king_attacks, knight_attacks, lance_attacks, pawn_attacks,
        rook_attacks, silver_attacks,
    };
    use crate::board::pat;

    let occ = board.occupied();
    let opp = attacker.flip();

    // Steppers: reverse-attack symmetry flips the attacker's colour (a Black
    // pawn attacks `sq` from where a White pawn on `sq` would attack).
    if !(pawn_attacks(opp, sq) & board.pieces_pattern(attacker, pat::PAWN)).is_empty() {
        return true;
    }
    // Lance: a forward-only slider; the reverse ray flips colour, occ-limited.
    if !(lance_attacks(opp, sq, occ) & board.pieces_pattern(attacker, pat::LANCE)).is_empty() {
        return true;
    }
    if !(knight_attacks(opp, sq) & board.pieces_pattern(attacker, pat::KNIGHT)).is_empty() {
        return true;
    }
    if !(silver_attacks(opp, sq) & board.pieces_pattern(attacker, pat::SILVER)).is_empty() {
        return true;
    }
    // Gold-like (gold + the four promoted minors, one pattern slot).
    if !(gold_attacks(opp, sq) & board.pieces_pattern(attacker, pat::GOLD)).is_empty() {
        return true;
    }

    // Sliders (symmetric, so no colour flip). Horse / dragon add the king ring.
    let king_ring = king_attacks(attacker, sq);
    let bishop = bishop_attacks(sq, occ);
    if !(bishop & board.pieces_pattern(attacker, pat::BISHOP)).is_empty() {
        return true;
    }
    if !((bishop | king_ring) & board.pieces_pattern(attacker, pat::HORSE)).is_empty() {
        return true;
    }
    let rook = rook_attacks(sq, occ);
    if !(rook & board.pieces_pattern(attacker, pat::ROOK)).is_empty() {
        return true;
    }
    if !((rook | king_ring) & board.pieces_pattern(attacker, pat::DRAGON)).is_empty() {
        return true;
    }
    // King (ring is colour-symmetric).
    if !(king_ring & board.pieces_pattern(attacker, pat::KING)).is_empty() {
        return true;
    }
    false
}

/// The set of `attacker` pieces on `board` that attack `sq` — the reverse
/// lookup [`is_attacked_by`] performs, returning every attacker instead of
/// short-circuiting. Fills the checkers set from the own-king square (one lookup
/// replacing an 81-square scan). Uses the board's current occupancy; see
/// [`attackers_bb_occ`] for the occupancy-parameterized form the SEE / mate
/// substrate consumes.
pub(crate) fn attackers_bb(
    board: &Board,
    sq: Square,
    attacker: Color,
) -> crate::bitboard::Bitboard {
    attackers_bb_occ(board, sq, attacker, board.occupied())
}

/// The set of `attacker` pieces on `board` that attack `sq`, evaluating slider
/// rays against the supplied `occ` rather than `board.occupied()`. The
/// occupancy-parameterized reverse lookup the reference's `attackers_to(sq, occ)`
/// performs (per colour): step attackers are occupancy-independent; sliders
/// (lance / bishop / rook, plus the slider component of horse / dragon) are cut
/// by `occ`. The horse / dragon king-ring component is a step effect and stays
/// occupancy-independent.
///
/// Note the returned squares are read from `board.pieces_pattern(...)` (the true
/// board pieces), *not* masked by `occ` — callers that pass an `occ` with pieces
/// removed (SEE's consumed attackers, mate's moved mover) intersect the result
/// with `occ` themselves when they need the "still present under `occ`" set.
pub(crate) fn attackers_bb_occ(
    board: &Board,
    sq: Square,
    attacker: Color,
    occ: crate::bitboard::Bitboard,
) -> crate::bitboard::Bitboard {
    use crate::bitboard::{
        bishop_attacks, gold_attacks, king_attacks, knight_attacks, lance_attacks, pawn_attacks,
        rook_attacks, silver_attacks,
    };
    use crate::board::pat;

    let opp = attacker.flip();
    let king_ring = king_attacks(attacker, sq);
    let bishop = bishop_attacks(sq, occ);
    let rook = rook_attacks(sq, occ);
    (pawn_attacks(opp, sq) & board.pieces_pattern(attacker, pat::PAWN))
        | (lance_attacks(opp, sq, occ) & board.pieces_pattern(attacker, pat::LANCE))
        | (knight_attacks(opp, sq) & board.pieces_pattern(attacker, pat::KNIGHT))
        | (silver_attacks(opp, sq) & board.pieces_pattern(attacker, pat::SILVER))
        | (gold_attacks(opp, sq) & board.pieces_pattern(attacker, pat::GOLD))
        | (bishop & board.pieces_pattern(attacker, pat::BISHOP))
        | ((bishop | king_ring) & board.pieces_pattern(attacker, pat::HORSE))
        | (rook & board.pieces_pattern(attacker, pat::ROOK))
        | ((rook | king_ring) & board.pieces_pattern(attacker, pat::DRAGON))
        | (king_ring & board.pieces_pattern(attacker, pat::KING))
}

/// Both colours' attackers of `sq` under occupancy `occ`, collected in a single
/// pass — the fused form of
/// `attackers_bb_occ(.., Black, occ) | attackers_bb_occ(.., White, occ)`.
///
/// Mirrors the reference `Position::attackers_to(sq, occ)` (position.cpp):
/// the two occupancy-limited slider walks (bishop, rook) are computed ONCE for
/// both colours instead of once per colour, and each side's lances are folded
/// into the shared rook ray by pre-masking with the occupancy-free forward-file
/// step effect ([`crate::bitboard::lance_step_effect`]) — avoiding the two
/// per-colour lance walks entirely. A `color` lance attacks `sq` iff it lies on
/// the reverse ray, i.e. the `opp`-direction step effect from `sq`, and
/// `rook & that ray` recovers exactly the occupancy-cut file segment. The step
/// effects (pawn / knight / silver / gold, plus the horse / dragon king-ring
/// component) stay per colour. Bit-identical to the per-colour OR; the SEE
/// collection site is the sole caller, and [`attackers_bb_occ`] remains the
/// equivalence oracle.
pub(crate) fn attackers_to_both(
    board: &Board,
    sq: Square,
    occ: crate::bitboard::Bitboard,
) -> crate::bitboard::Bitboard {
    use crate::bitboard::{
        Bitboard, bishop_attacks, gold_attacks, king_attacks, knight_attacks, lance_step_effect,
        pawn_attacks, rook_attacks, silver_attacks,
    };
    use crate::board::pat;

    // Sliders computed once for both colours; the king ring is colour-symmetric.
    let bishop = bishop_attacks(sq, occ);
    let rook = rook_attacks(sq, occ);
    let king_ring = king_attacks(Color::Black, sq);

    let mut out = Bitboard::EMPTY;
    for color in [Color::Black, Color::White] {
        let opp = color.flip();
        let lance_ray = lance_step_effect(opp, sq);
        out = out
            | (pawn_attacks(opp, sq) & board.pieces_pattern(color, pat::PAWN))
            | ((rook & lance_ray) & board.pieces_pattern(color, pat::LANCE))
            | (knight_attacks(opp, sq) & board.pieces_pattern(color, pat::KNIGHT))
            | (silver_attacks(opp, sq) & board.pieces_pattern(color, pat::SILVER))
            | (gold_attacks(opp, sq) & board.pieces_pattern(color, pat::GOLD))
            | (bishop & board.pieces_pattern(color, pat::BISHOP))
            | ((bishop | king_ring) & board.pieces_pattern(color, pat::HORSE))
            | (rook & board.pieces_pattern(color, pat::ROOK))
            | ((rook | king_ring) & board.pieces_pattern(color, pat::DRAGON))
            | (king_ring & board.pieces_pattern(color, pat::KING));
    }
    out
}

/// An 81-square scanning implementation of [`is_attacked_by`], derived
/// independently of the piece sets so it can serve as their equivalence oracle.
#[cfg(test)]
pub(crate) fn is_attacked_by_scan(board: &Board, sq: Square, attacker: Color) -> bool {
    let dr_sign = dr_sign_for(attacker);

    // Pawn (non-promoted only — promoted pawn moves like Gold and is handled below).
    if let Some(src) = step_signed(sq, 0, dr_sign)
        && matches_unpromoted_kind(board, src, attacker, PieceKind::Pawn)
    {
        return true;
    }

    // Knight (non-promoted only).
    for &df in &[-1i8, 1] {
        if let Some(src) = step_signed(sq, df, 2 * dr_sign)
            && matches_unpromoted_kind(board, src, attacker, PieceKind::Knight)
        {
            return true;
        }
    }

    // Silver (non-promoted only).
    for &(df, dr) in SILVER_STEPS {
        if let Some(src) = step_signed(sq, -df, -dr * dr_sign)
            && matches_unpromoted_kind(board, src, attacker, PieceKind::Silver)
        {
            return true;
        }
    }

    // Gold-like: Gold OR promoted {Pawn, Lance, Knight, Silver}.
    for &(df, dr) in GOLD_STEPS {
        if let Some(src) = step_signed(sq, -df, -dr * dr_sign)
            && let Some(p) = board.get(src)
            && p.color == attacker
            && is_gold_like(p)
        {
            return true;
        }
    }

    // 8-neighbour: King; +Bishop (Horse) for orthogonal step; +Rook (Dragon) for diagonal step.
    for &(df, dr) in KING_STEPS {
        let Some(src) = step_signed(sq, df, dr) else {
            continue;
        };
        let Some(p) = board.get(src) else { continue };
        if p.color != attacker {
            continue;
        }
        if p.kind == PieceKind::King {
            return true;
        }
        let is_orth = df == 0 || dr == 0;
        let is_diag = df != 0 && dr != 0;
        if is_orth && p.kind == PieceKind::Bishop && p.promoted {
            return true;
        }
        if is_diag && p.kind == PieceKind::Rook && p.promoted {
            return true;
        }
    }

    // Lance (slider, forward-only; non-promoted — promoted lance moves like Gold).
    if scan_slider(board, sq, 0, dr_sign, attacker, PieceKind::Lance, true) {
        return true;
    }

    // Bishop / Horse (diagonal slider; both promoted states share the slider component).
    for &(df, dr) in BISHOP_DIRS {
        if scan_slider(board, sq, df, dr, attacker, PieceKind::Bishop, false) {
            return true;
        }
    }

    // Rook / Dragon (orthogonal slider).
    for &(df, dr) in ROOK_DIRS {
        if scan_slider(board, sq, df, dr, attacker, PieceKind::Rook, false) {
            return true;
        }
    }

    false
}

/// Production uchifuzume (打ち歩詰め) predicate: true iff dropping the pawn
/// described by `m` — a drop by the side to move in `pre`, the **pre-drop**
/// position — would be uchifuzume (unanswerable pawn-drop mate) and is therefore
/// illegal. In-place port of the reference `Position::legal_drop`
/// (`position.cpp`): no clone, no move generation, no allocation.
///
/// Returns `false` for any non-pawn-drop, and for a pawn drop that does not
/// check the enemy king (the only drops that can be uchifuzume). The heavy
/// lifting is delegated to [`Position::legal_drop`], which assumes that
/// checking precondition; this wrapper establishes it. Equivalent to
/// `!pre.legal_drop(to)` under that precondition.
pub(crate) fn drop_is_uchifuzume(pre: &Position, m: Move) -> bool {
    if !m.is_drop() || m.dropped_piece_kind() != PieceKind::Pawn {
        return false;
    }
    let us = pre.side_to_move();
    let to = m.to_sq();
    // A dropped pawn checks exactly the single square directly ahead of it; it
    // can be uchifuzume only when the enemy king sits on that square. (The
    // reference relies on `GenerateDropMoves` calling `legal_drop` solely for
    // checking drops; this guard reconstructs that precondition — see the
    // `pawnEffect(us, to) == king` assert inside `legal_drop`.)
    let Some(king) = try_find_king(pre.board(), us.flip()) else {
        return false;
    };
    if !crate::bitboard::pawn_attacks(us, to).test(king) {
        return false;
    }
    !pre.legal_drop(to)
}

impl Position {
    /// Port of the reference `Position::legal_drop(to)` (`position.cpp`,
    /// the live `#if !defined(LONG_EFFECT_LIBRARY)` branch): returns `true` iff
    /// dropping a side-to-move pawn on `to` is **legal**, i.e. *not* uchifuzume.
    ///
    /// Precondition (asserted, as the reference does): the dropped pawn on `to`
    /// checks the enemy king — `to` is the single square our pawn attacks, and
    /// the enemy king stands on it. Callers ([`drop_is_uchifuzume`],
    /// [`Position::pawn_drop_is_uchifuzume`]) establish this before calling.
    ///
    /// Answers the "is this uchifuzume?" question with effect queries only — no
    /// board clone, no `do_move`, no move generation — mirroring the reference:
    ///   1. If no own piece defends `to`, the king simply captures the pawn.
    ///   2. Else if some enemy attacker of `to` (via the specialised
    ///      [`attackers_to_pawn`]: king excluded, lance impossible by
    ///      construction) is either not pinned to its king, or shares `to`'s file
    ///      (the reference's same-file exception, its worked example 3), the pawn
    ///      can be captured.
    ///   3. Else scan the enemy king's escape ring (own-piece-free, excluding
    ///      `to`) under occupancy `pieces() ^ to` — the dropped pawn now blocks
    ///      rays; any square we do not attack is an escape.
    ///
    /// If none of 1–3 holds, the drop is uchifuzume: return `false`.
    pub(crate) fn legal_drop(&self, to: Square) -> bool {
        use crate::bitboard::{file_mask, king_attacks, pawn_attacks};

        let board = self.board();
        let us = self.side_to_move();
        let them = us.flip();
        let king = find_king(board, them);

        // 打とうとする歩の利きに相手玉がいることは前提条件。
        debug_assert!(
            pawn_attacks(us, to).test(king),
            "legal_drop precondition: the dropped pawn on `to` must check the enemy king",
        );

        // この歩に利いている自駒がなければ玉が取れるので合法(打ち歩詰めではない)。
        if !is_attacked_by(board, to, us) {
            return true;
        }

        // `to` に利いている敵駒(玉・香・歩を除く)を列挙。取れるなら打ち歩詰めではない。
        let b = attackers_to_pawn(board, them, to);

        // 敵玉に対してpinされている駒(自駒も含むが、b は敵駒なので問題ない)。
        // 参照実装が `st->blockersForKing[~us]` を読むのと同じく、per-state な
        // check-info キャッシュから読む。
        let pinned = self.check_info().blockers(them);

        // pinされていない駒が1つでもあれば取れる。玉頭方向(同じ筋)への移動は
        // pin方向と一致しないので、同じ筋の攻撃駒は pin されていない扱いにする(例3対策)。
        if !(b & (!pinned | file_mask(to.file()))).is_empty() {
            return true;
        }

        // 玉の退路を探す。to には歩が立つので占有に加える(= pieces() ^ to)。
        let occ = board.occupied() ^ crate::bitboard::Bitboard::from_square(to);
        let mut escape = king_attacks(them, king) & !board.pieces_color(them);
        escape ^= crate::bitboard::Bitboard::from_square(to);
        for king_to in escape.squares() {
            if attackers_bb_occ(board, king_to, us, occ).is_empty() {
                return true; // 退路が見つかったので打ち歩詰めではない。
            }
        }

        // すべての検査を抜けたので打ち歩詰め。
        false
    }
}

/// `c`'s pieces attacking the pawn-drop square `pawn_sq`, port of the reference
/// `Position::attackers_to_pawn` (`position.cpp`). By construction of
/// the uchifuzume test the enemy king is excluded (already handled) and a lance
/// can never attack `pawn_sq` (the king stands directly between), so this is a
/// specialised — cheaper — variant of the general attacker query: only knight,
/// silver, gold(+promoted minors), bishop/horse and rook/dragon contribute.
fn attackers_to_pawn(board: &Board, c: Color, pawn_sq: Square) -> crate::bitboard::Bitboard {
    use crate::bitboard::{
        bishop_attacks, gold_attacks, knight_attacks, rook_attacks, silver_attacks,
    };
    use crate::board::pat;

    let them = c.flip();
    let occ = board.occupied();

    // 馬・龍は銀と金の両方の利きに寄与する。
    let bb_hd = board.pieces_pattern(c, pat::HORSE) | board.pieces_pattern(c, pat::DRAGON);

    let knight = board.pieces_pattern(c, pat::KNIGHT);
    let silver = board.pieces_pattern(c, pat::SILVER);
    let golds = board.pieces_pattern(c, pat::GOLD);
    let bishop_horse = board.pieces_pattern(c, pat::BISHOP) | board.pieces_pattern(c, pat::HORSE);
    let rook_dragon = board.pieces_pattern(c, pat::ROOK) | board.pieces_pattern(c, pat::DRAGON);

    (knight_attacks(them, pawn_sq) & knight)
        | (silver_attacks(them, pawn_sq) & (silver | bb_hd))
        | (gold_attacks(them, pawn_sq) & (golds | bb_hd))
        | (bishop_attacks(pawn_sq, occ) & bishop_horse)
        | (rook_attacks(pawn_sq, occ) & rook_dragon)
}

/// True iff `m` is a Pawn drop by `mover` and the resulting position (already
/// applied to `post`) is checkmate against the opponent — i.e., uchifuzume.
/// `probe_buf` is reused across calls to keep the probe allocation-free.
///
/// `#[cfg(test)]`: the equivalence oracle for the in-place
/// [`drop_is_uchifuzume`] / [`Position::legal_drop`] port (the same discipline
/// as the `see_ge` reference twin). Production never clones and probes.
///
/// The inner mate probe enumerates the opponent's legal replies with the
/// search-side [`Position::generate_evasions`] (`all == true`) filtered by
/// [`Position::is_legal`] — the post-drop opponent is in check by construction
/// (checked above), so evasions are the correct generator. Uchifuzume is
/// enforced at drop-generation time inside the evasion generator, so this probe
/// does not recurse into `is_uchifuzume_after_drop`, matching the reference's
/// non-recursive `legal_drop`.
#[cfg(test)]
pub(crate) fn is_uchifuzume_after_drop(
    post: &Position,
    m: Move,
    mover: Color,
    probe_buf: &mut Vec<Move>,
) -> bool {
    use crate::search_movegen::ExtMove;
    if !m.is_drop() || m.dropped_piece_kind() != PieceKind::Pawn {
        return false;
    }
    let opp_king_sq = find_king(post.board(), mover.flip());
    if !is_attacked_by(post.board(), opp_king_sq, mover) {
        return false;
    }
    probe_buf.clear();
    let mut pseudo: Vec<ExtMove> = Vec::new();
    post.generate_evasions(true, &mut pseudo);
    for em in pseudo {
        if post.is_legal(em.mv) {
            probe_buf.push(em.mv);
        }
    }
    probe_buf.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfen::parse_sfen;

    fn legal_count(sfen: &str) -> usize {
        let pos = parse_sfen(sfen).unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        moves.len()
    }

    fn legal_moves(sfen: &str) -> Vec<Move> {
        let pos = parse_sfen(sfen).unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        moves
    }

    #[test]
    fn startpos_legal_move_count_is_30() {
        assert_eq!(legal_count(crate::sfen::STARTPOS_SFEN), 30);
    }

    #[test]
    fn startpos_contains_7g7f_pawn_push() {
        let pos = Position::startpos();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let from = Square::new(6, 6).unwrap();
        let to = Square::new(6, 5).unwrap();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let expected = Move::make(from, to, pawn);
        assert!(
            moves.contains(&expected),
            "7g7f not in legal-move list: {moves:?}"
        );
    }

    #[test]
    fn pawn_one_rank_from_last_generates_only_promote() {
        // Black pawn at 5b (file=4, rank=1) pushing to 5a (file=4, rank=0).
        // Both kings present so legality filter has somewhere to find them.
        let sfen = "4k4/4P4/9/9/9/9/9/9/4K4 b - 1";
        let moves = legal_moves(sfen);
        let pawn_pushes: Vec<&Move> = moves
            .iter()
            .filter(|m| !m.is_drop() && m.from_sq() == Square::new(4, 1).unwrap())
            .collect();
        // Pawn at (4,1) → (4,0): forced promote, only promote variant.
        assert_eq!(
            pawn_pushes.len(),
            1,
            "expected exactly one pawn move (forced promote), got {pawn_pushes:?}"
        );
        assert!(
            pawn_pushes[0].is_promote(),
            "expected forced-promote variant"
        );
    }

    #[test]
    fn pawn_into_zone_but_not_last_rank_generates_both_variants() {
        // Black pawn at 5d (file=4, rank=3) pushing to 5c (file=4, rank=2).
        // Rank 2 is in the promotion zone but not the last rank.
        let sfen = "4k4/9/9/4P4/9/9/9/9/4K4 b - 1";
        let moves = legal_moves(sfen);
        let from = Square::new(4, 3).unwrap();
        let to = Square::new(4, 2).unwrap();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        assert!(moves.contains(&Move::make(from, to, pawn)));
        assert!(moves.contains(&Move::make_promote(from, to, pawn)));
    }

    #[test]
    fn pinned_silver_cannot_move_off_pin_ray() {
        // Black king at 5a (file=4, rank=0). Black silver at 5b (file=4, rank=1).
        // White rook at 5i (file=4, rank=8) pinning the silver to the king down
        // the 5-file. Silver may step to (3,0) or (5,0) (sideways pin not on
        // the file), but moving off the file (e.g. to (3,2) or (5,2)) leaves
        // the king in check.
        // Use Silver's pseudo-legal steps from (4,1) to assert filtering:
        // pseudo steps are (0,0), (3,0), (5,0), (3,2), (5,2). Pin filters out
        // (3,2) and (5,2). King moves are not relevant here (he must stay on
        // the file or accept check). For brevity, set up no other pieces near
        // the king and check that no silver move leaves the file.
        let sfen = "4k4/4S4/9/9/9/9/9/9/4r2K1 b - 1";
        let moves = legal_moves(sfen);
        let silver_from = Square::new(4, 1).unwrap();
        for m in &moves {
            if m.is_drop() {
                continue;
            }
            if m.from_sq() == silver_from {
                let to = m.to_sq();
                assert_eq!(to.file(), 4, "pinned silver moved off the 5-file: {to:?}",);
            }
        }
    }

    #[test]
    fn king_cannot_step_into_attacked_square() {
        // Black king at 5e (file=4, rank=4). White rook at 4e (file=5, rank=4)
        // — that is, immediately east of the king, giving check. The rook's
        // attack along rank 4 sweeps west: square (3, 4) is on the ray and is
        // attacked the moment the king vacates (4, 4). So the king's
        // pseudo-legal step to (3, 4) is illegal. The king CAN capture the
        // rook by moving to (5, 4).
        let sfen = "9/9/9/9/3rK4/9/9/9/4k4 b - 1";
        let pos = parse_sfen(sfen).unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let king_from = Square::new(4, 4).unwrap();
        let king = Piece::new(PieceKind::King, Color::Black);
        let illegal_into_ray = Move::make(king_from, Square::new(3, 4).unwrap(), king);
        let legal_capture = Move::make(king_from, Square::new(5, 4).unwrap(), king);
        assert!(
            !moves.contains(&illegal_into_ray),
            "king moved into rook ray: legal moves were {moves:?}",
        );
        assert!(
            moves.contains(&legal_capture),
            "king should be able to capture the checking rook: {moves:?}",
        );
    }

    #[test]
    fn is_attacked_by_pawn_in_front() {
        // Black pawn at (4,4); attacks (4,3).
        let mut board = Board::empty();
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Pawn, Color::Black)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(4, 5).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(3, 3).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_lance_through_empty_squares() {
        // Black lance at (4,8); attacks (4,0) through empty path.
        let mut board = Board::empty();
        board.set(
            Square::new(4, 8).unwrap(),
            Some(Piece::new(PieceKind::Lance, Color::Black)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 0).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_lance_blocked_returns_false() {
        // Black lance at (4,8); blocker at (4,4). Squares above the blocker
        // are NOT attacked.
        let mut board = Board::empty();
        board.set(
            Square::new(4, 8).unwrap(),
            Some(Piece::new(PieceKind::Lance, Color::Black)),
        );
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Pawn, Color::White)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 4).unwrap(),
            Color::Black
        )); // first blocker IS attacked
        assert!(!is_attacked_by(
            &board,
            Square::new(4, 0).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_knight_at_jump_offsets() {
        // Black knight at (4,4) attacks (3,2) and (5,2). Nothing else.
        let mut board = Board::empty();
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Knight, Color::Black)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(3, 2).unwrap(),
            Color::Black
        ));
        assert!(is_attacked_by(
            &board,
            Square::new(5, 2).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(3, 6).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_bishop_along_diagonal() {
        let mut board = Board::empty();
        board.set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::Bishop, Color::Black)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(8, 8).unwrap(),
            Color::Black
        ));
        // Block it.
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Pawn, Color::White)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 4).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(8, 8).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_horse_picks_up_orthogonal_step() {
        // Promoted bishop (Horse) at (4,4) attacks (4,5) (orthogonal 1-step,
        // which a plain bishop does NOT cover).
        let mut board = Board::empty();
        let horse = Piece {
            kind: PieceKind::Bishop,
            color: Color::Black,
            promoted: true,
        };
        board.set(Square::new(4, 4).unwrap(), Some(horse));
        assert!(is_attacked_by(
            &board,
            Square::new(4, 5).unwrap(),
            Color::Black
        ));
        assert!(is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::Black
        ));
        assert!(is_attacked_by(
            &board,
            Square::new(3, 4).unwrap(),
            Color::Black
        ));
        // Plain bishop does NOT attack (4,5).
        let mut board2 = Board::empty();
        board2.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Bishop, Color::Black)),
        );
        assert!(!is_attacked_by(
            &board2,
            Square::new(4, 5).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_dragon_picks_up_diagonal_step() {
        // Promoted rook (Dragon) at (4,4) attacks (5,5) — diagonal 1-step.
        let mut board = Board::empty();
        let dragon = Piece {
            kind: PieceKind::Rook,
            color: Color::Black,
            promoted: true,
        };
        board.set(Square::new(4, 4).unwrap(), Some(dragon));
        assert!(is_attacked_by(
            &board,
            Square::new(5, 5).unwrap(),
            Color::Black
        ));
        // Plain rook does NOT.
        let mut board2 = Board::empty();
        board2.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        assert!(!is_attacked_by(
            &board2,
            Square::new(5, 5).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_promoted_pawn_uses_gold_pattern() {
        // +Pawn (Tokin) at (4,4): attacks like Gold. (4,3) — yes (forward).
        // (3,3) — yes (forward-diagonal). (3,5) — no (backward-diagonal isn't
        // a gold pattern).
        let mut board = Board::empty();
        let tokin = Piece {
            kind: PieceKind::Pawn,
            color: Color::Black,
            promoted: true,
        };
        board.set(Square::new(4, 4).unwrap(), Some(tokin));
        assert!(is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::Black
        ));
        assert!(is_attacked_by(
            &board,
            Square::new(3, 3).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(3, 5).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_white_attacker_flips_direction() {
        // White pawn at (4,4) attacks (4,5) (white forward = rank increasing).
        let mut board = Board::empty();
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Pawn, Color::White)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 5).unwrap(),
            Color::White
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::White
        ));
    }

    #[test]
    fn empty_hand_emits_no_drops() {
        // Two kings only; both hands empty. Each side has board moves; no drops.
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(moves.iter().all(|m| !m.is_drop()), "no drops expected");
    }

    #[test]
    fn bishop_in_hand_can_drop_on_every_empty_square() {
        // Two kings only; Black has 1 bishop in hand. 81 - 2 = 79 empty squares.
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b B 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let drops: Vec<&Move> = moves.iter().filter(|m| m.is_drop()).collect();
        assert_eq!(drops.len(), 79, "expected 79 bishop drops, got {drops:?}");
        assert!(
            drops
                .iter()
                .all(|m| m.dropped_piece_kind() == PieceKind::Bishop),
        );
    }

    #[test]
    fn pawn_drop_is_blocked_by_own_pawn_on_same_file() {
        // Black has 1 pawn in hand; one own pawn on file 4 (5e). Pawn drops on
        // file 4 are forbidden (`nifu`); other files only restricted by the
        // last-rank rule (Black can't drop on rank 0).
        let pos = parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn)
            .collect();
        // Empty squares = 81 - 3 = 78. Forbidden: every empty square on file 4
        // (9 - 3 occupied = 6), plus every empty rank-0 square (9 - 1 occupied
        // = 8). The two sets do not overlap in *empty* squares — their only
        // intersection (4,0) holds the white king. So 6 + 8 = 14 forbidden.
        // Legal pawn drops: 78 - 14 = 64.
        assert_eq!(pawn_drops.len(), 64, "got {pawn_drops:?}");
        assert!(
            pawn_drops.iter().all(|m| m.to_sq().file() != 4),
            "nifu violated"
        );
        assert!(
            pawn_drops.iter().all(|m| m.to_sq().rank() != 0),
            "pawn dropped on Black's last rank",
        );
    }

    #[test]
    fn nifu_does_not_trigger_on_tokin() {
        // Black holds 1 pawn; Black's promoted pawn (Tokin, `+P`) at 5e
        // (file=4, rank=4). Tokin is not an unpromoted pawn so nifu must NOT
        // fire on file 4 — pawn drops on file 4's empty squares are legal.
        let pos = parse_sfen("4k4/9/9/9/4+P4/9/9/9/4K4 b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops_on_file_4: Vec<&Move> = moves
            .iter()
            .filter(|m| {
                m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn && m.to_sq().file() == 4
            })
            .collect();
        // Empty squares on file 4 (excluding kings at (4,0)/(4,8) and the
        // Tokin at (4,4)): (4,1), (4,2), (4,3), (4,5), (4,6), (4,7) — six
        // squares. None is on Black's rank-0, so all six are legal drops.
        assert_eq!(
            pawn_drops_on_file_4.len(),
            6,
            "expected 6 pawn drops on file 4 (Tokin does not trigger nifu), got {pawn_drops_on_file_4:?}",
        );
    }

    #[test]
    fn nifu_on_file_zero_does_not_leak_to_file_one() {
        // Black holds 1 pawn; own unpromoted pawn at (file=0, rank=4)
        // (encoded as `8P`). Off-by-one guard: file 0 must be filtered, file
        // 1 must NOT be filtered.
        let pos = parse_sfen("4k4/9/9/9/8P/9/9/9/4K4 b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn)
            .collect();
        assert!(
            pawn_drops.iter().all(|m| m.to_sq().file() != 0),
            "nifu failed to filter file 0: {pawn_drops:?}",
        );
        assert!(
            pawn_drops.iter().any(|m| m.to_sq().file() == 1),
            "file 1 was filtered too — off-by-one in nifu file index: {pawn_drops:?}",
        );
    }

    #[test]
    fn nifu_applies_to_white_pawns_too() {
        // White to move; White holds 1 pawn; own White unpromoted pawn at
        // (file=4, rank=4). Pawn drops on file 4 must be filtered. White's
        // own last rank is rank 8, so the rank-exclusion does not collide
        // with the nifu test on file 4.
        let pos = parse_sfen("4k4/9/9/9/4p4/9/9/9/4K4 w p 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn)
            .collect();
        assert!(
            !pawn_drops.is_empty(),
            "white had no legal pawn drops at all — setup wrong",
        );
        assert!(
            pawn_drops.iter().all(|m| m.to_sq().file() != 4),
            "nifu (white side) failed to filter file 4: {pawn_drops:?}",
        );
    }

    #[test]
    fn nifu_does_not_filter_lance_drops() {
        // Same shape as the pawn-on-file-4 nifu test, but Black holds a
        // lance instead. Lance is not subject to nifu — drops on file 4 are
        // legal (only the lance last-rank rule applies).
        let pos = parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b L 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let lance_drops_on_file_4: Vec<&Move> = moves
            .iter()
            .filter(|m| {
                m.is_drop() && m.dropped_piece_kind() == PieceKind::Lance && m.to_sq().file() == 4
            })
            .collect();
        // Empty squares on file 4: (4,1)..(4,3), (4,5)..(4,7) — six. None
        // is on Black's rank 0, so all six are legal lance drops.
        assert_eq!(
            lance_drops_on_file_4.len(),
            6,
            "lance drop on shared-file with own pawn was wrongly filtered: {lance_drops_on_file_4:?}",
        );
    }

    #[test]
    fn knight_drop_excluded_from_last_two_ranks() {
        // Black has 1 knight in hand; only kings on the board.
        // Empty squares = 79. Forbidden ranks for Black knight: 0 and 1
        // (9 + 9 = 18 squares; minus the white king at rank 0 = 17 forbidden).
        // Legal: 79 - 17 = 62.
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b N 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let knight_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Knight)
            .collect();
        assert_eq!(knight_drops.len(), 62);
        assert!(
            knight_drops.iter().all(|m| m.to_sq().rank() >= 2),
            "knight dropped on rank 0 or 1"
        );
    }

    #[test]
    fn drop_blocking_check_is_legal_capture_is_not_required() {
        // Black king at 5i (file=4, rank=8). White rook at 5a (file=4, rank=0)
        // gives check along the 5-file with the path clear. Black has 1 gold
        // in hand. Any gold drop on the 5-file between king and rook
        // interposes; any other drop leaves the king in check (illegal).
        let pos = parse_sfen("4r4/9/9/9/9/9/9/9/4K4 b G 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let gold_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Gold)
            .collect();
        for m in &gold_drops {
            let to = m.to_sq();
            assert_eq!(to.file(), 4, "non-blocking gold drop survived: {to:?}");
            assert!((1..=7).contains(&to.rank()), "gold drop off interpose ray");
        }
        // Exactly the seven interpose squares are legal.
        assert_eq!(gold_drops.len(), 7);
    }

    // -- Uchifuzume (打ち歩詰め) ----------------------------------------------
    //
    // Mating-net shared by the next four tests:
    //   White king at 9a (file=8, rank=0).
    //   Black gold at 9c (file=8, rank=2)  — covers (8,1) and (7,1).
    //   Black knight at 7c (file=6, rank=2) — covers (5,0) and (7,0).
    //   Black king at 1i (file=0, rank=8)   — distant, irrelevant.
    // A Black pawn drop at (file=8, rank=1) attacks the white king, the
    // dropped pawn is defended by the gold, and every white-king escape is
    // covered. Replacing one piece in this shape gives the matrix below.

    #[test]
    fn uchifuzume_filters_pawn_drop_mate() {
        // Pawn drop in front of the white king is checkmate — uchifuzume
        // must reject it.
        let pos = parse_sfen("k8/9/G1N6/9/9/9/9/9/8K b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let mating_drop =
            Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            !moves.contains(&mating_drop),
            "uchifuzume failed to filter pawn-drop mate: {moves:?}",
        );
    }

    #[test]
    fn pawn_drop_check_is_legal_when_attacker_can_be_captured() {
        // Same mating shape but with a White silver at 8c (file=7, rank=2).
        // The silver can move to (8,1) and capture the dropped pawn — so
        // White has a legal reply, the position is NOT mate, and the pawn
        // drop must remain legal.
        let pos = parse_sfen("k8/9/GsN6/9/9/9/9/9/8K b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let checking_drop =
            Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            moves.contains(&checking_drop),
            "pawn drop wrongly filtered when capture is available: {moves:?}",
        );
    }

    #[test]
    fn gold_drop_mate_is_legal_uchifuzume_is_pawn_only() {
        // Same mating shape but Black drops a Gold instead of a Pawn. The
        // rule applies only to pawn drops — Gold-drop mate is a legal move.
        let pos = parse_sfen("k8/9/G1N6/9/9/9/9/9/8K b G 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let mating_gold_drop =
            Move::make_drop(PieceKind::Gold, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            moves.contains(&mating_gold_drop),
            "gold-drop mate was filtered (uchifuzume must be pawn-only): {moves:?}",
        );
    }

    #[test]
    fn pawn_drop_check_is_legal_when_king_can_step_out() {
        // Two kings, no other pieces. A Black pawn drop at (8,1) gives check
        // but the white king has free escape squares (no Black piece covers
        // them) — pawn drop must remain legal.
        let pos = parse_sfen("k8/9/9/9/9/9/9/9/8K b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let checking_drop =
            Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            moves.contains(&checking_drop),
            "pawn drop wrongly filtered when king has an escape: {moves:?}",
        );
    }

    #[test]
    fn nifu_and_uchifuzume_compose_without_panic() {
        // Mating shape PLUS Black's own unpromoted pawn on file 8 at
        // (file=8, rank=4). Nifu rejects every Black pawn-drop on file 8 at
        // the pseudo-legal layer; the uchifuzume probe must therefore never
        // see this drop and the two filters must compose without panicking.
        let pos = parse_sfen("k8/9/G1N6/9/P8/9/9/9/8K b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops_on_file_8: Vec<&Move> = moves
            .iter()
            .filter(|m| {
                m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn && m.to_sq().file() == 8
            })
            .collect();
        assert!(
            pawn_drops_on_file_8.is_empty(),
            "nifu should reject every pawn drop on file 8: {pawn_drops_on_file_8:?}",
        );
    }

    // ---- in-place `legal_drop` equivalence oracle ------------------------
    //
    // The production uchifuzume filter is the in-place `drop_is_uchifuzume` /
    // `Position::legal_drop` port (no board clone, no inner move generation).
    // The `#[cfg(test)]` clone-and-probe (`is_uchifuzume_after_drop`) is its
    // equivalence oracle: the two must agree on every candidate square the
    // generators can probe.

    /// A tiny xorshift PRNG (independent copy — test modules don't share scope).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn pick(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// For the single geometric pawn-drop-mate candidate at `p` (the square a
    /// side-to-move pawn would occupy to check the enemy king), assert the
    /// production in-place predicate [`drop_is_uchifuzume`] agrees with the
    /// retained clone-and-probe oracle [`is_uchifuzume_after_drop`]. Requires a
    /// pawn in the mover's hand (the oracle plays the drop). Returns the number
    /// of candidates actually compared (0 or 1) so callers can gauge coverage.
    fn assert_drop_uchifuzume_agrees(p: &Position, ctx: &str) -> usize {
        let us = p.side_to_move();
        if p.hand(us).count(PieceKind::Pawn) == 0 {
            return 0;
        }
        let Some(king) = try_find_king(p.board(), us.flip()) else {
            return 0;
        };
        let mut compared = 0;
        for idx in 0..Square::COUNT as u8 {
            let to = Square::from_index(idx).unwrap();
            if p.board().get(to).is_some() {
                continue;
            }
            // Only the square from which our pawn would check the enemy king can
            // be uchifuzume — exactly the candidate the generators probe.
            if !crate::bitboard::pawn_attacks(us, to).test(king) {
                continue;
            }
            let m = Move::make_drop(PieceKind::Pawn, us, to);
            let new = drop_is_uchifuzume(p, m);
            let mut post = p.clone();
            post.do_move(m);
            let mut buf = Vec::new();
            let oracle = is_uchifuzume_after_drop(&post, m, us, &mut buf);
            assert_eq!(
                new, oracle,
                "{ctx}: drop_is_uchifuzume={new} disagrees with clone-probe oracle={oracle} \
                 for pawn drop to {to:?}",
            );
            compared += 1;
        }
        compared
    }

    /// Drop-mate-rich seeds: pawn in hand, exposed kings, and (for the pinned
    /// shapes) defenders around the enemy king, plus a few parity-fixture SFENs.
    const UCHI_ORACLE_SFENS: &[&str] = &[
        // Parity fixtures with hands / drops in play.
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        // Drop-mate seeds (the four existing uchifuzume-test shapes).
        "k8/9/G1N6/9/9/9/9/9/8K b P 1",  // uchifuzume mate
        "k8/9/GsN6/9/9/9/9/9/8K b P 1",  // capturable attacker → legal
        "k8/9/9/9/9/9/9/9/8K b P 1",     // king can step out → legal
        "k8/9/G1N6/9/P8/9/9/9/8K b P 1", // nifu + mate shape
    ];

    /// Deterministic random playouts from each seed; at every ply the in-place
    /// predicate must agree with the clone-and-probe oracle on the pawn-drop-mate
    /// candidate. Gate (a)/(b): parity-fixture playouts + drop-mate-rich seeds.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn legal_drop_matches_clone_probe_oracle_over_playouts() {
        const MIN_PLIES: usize = 40;
        let mut total_compared = 0usize;
        for (fi, sfen) in UCHI_ORACLE_SFENS.iter().enumerate() {
            let mut p = parse_sfen(sfen).expect("valid SFEN");
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, crate::position::Undo)> = Vec::new();
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                total_compared +=
                    assert_drop_uchifuzume_agrees(&p, &format!("fixture {fi} ply {plies}"));
                let legal = {
                    let mut v = Vec::new();
                    p.generate_legal_all(&mut v);
                    v
                };
                if legal.is_empty() {
                    // Terminal — unwind and continue the walk from the root.
                    while let Some((m, u)) = stack.pop() {
                        p.undo_move(m, u);
                    }
                    if stack.is_empty() {
                        break;
                    }
                    continue;
                }
                let m = legal[rng.pick(legal.len())];
                let u = p.do_move(m);
                stack.push((m, u));
                plies += 1;
            }
        }
        assert!(
            total_compared > 0,
            "no pawn-drop-mate candidate was ever probed — coverage is vacuous",
        );
    }

    /// Gate (c): hand-crafted positions for the reference's three commented
    /// example shapes, including the same-file exception (例3) both firing and
    /// not firing. Each asserts the exact uchifuzume verdict AND agreement with
    /// the clone-and-probe oracle. `to` is `(file 0, rank 1)` throughout — the
    /// square directly in front of a White king in the `1a` corner `(0, 0)`.
    #[test]
    fn legal_drop_pin_example_shapes() {
        let to = Square::new(0, 1).unwrap();

        // 例3 FIRING — the sole attacker is a file-pinned White rook that CAN
        // recapture along its pin line (same file as `to`): LEGAL. Without the
        // same-file exception this would be misjudged uchifuzume.
        //   White K@1a, White rook@1c (pinned up file 1 by Black lance@1d),
        //   Black golds@2b,3b cover the king's escapes and defend `to`.
        let p = parse_sfen("8k/6G2/7Gr/8L/9/9/9/9/K8 b P 1").unwrap();
        assert!(
            p.legal_drop(to),
            "例3 firing: same-file pinned rook can recapture — must be legal",
        );
        assert_eq!(assert_drop_uchifuzume_agrees(&p, "例3 firing"), 1);

        // 例1 — a rank-pinned White bishop attacks `to` diagonally but cannot
        // recapture (leaving the rank exposes the king to the Black rook), and it
        // is NOT on `to`'s file: uchifuzume.
        //   White K@1a, White bishop@2a (pinned along rank a by Black rook@4a),
        //   Black gold@2c defends `to` and covers 2b.
        let p = parse_sfen("5R1bk/9/7G1/9/9/9/9/9/K8 b P 1").unwrap();
        assert!(
            !p.legal_drop(to),
            "例1: rank-pinned bishop cannot recapture — must be uchifuzume",
        );
        assert_eq!(assert_drop_uchifuzume_agrees(&p, "例1"), 1);

        // 例2 — a diagonally-pinned White rook attacks `to` along the rank but
        // cannot recapture (leaving the diagonal exposes the king to the Black
        // bishop): uchifuzume. Same-file exception must NOT fire (rook on file 1).
        //   White K@1a, White rook@2b (pinned along the a1–i9 diagonal by Black
        //   bishop@3c), Black golds@2b… defend `to` and cover the escape.
        let p = parse_sfen("8k/6Gr1/6BG1/9/9/9/9/9/K8 b P 1").unwrap();
        assert!(
            !p.legal_drop(to),
            "例2: diagonally-pinned rook cannot recapture — must be uchifuzume",
        );
        assert_eq!(assert_drop_uchifuzume_agrees(&p, "例2"), 1);
    }

    // ---- 4-fold-repetition filter ----------------------------------------
    //
    // Upstream YaneuraOu's perft and its LEGAL_ALL movegen are both
    // repetition-blind, so plain 4-fold is not filtered here either. The tests
    // below pin that semantics, so a regression toward filter-the-move would be
    // caught here.

    /// Two-king board with no other pieces. Independent copy of
    /// `position::tests::setup_king_shuffle_pos` — test modules don't share
    /// scope, and the helper is small enough that re-stating it here is
    /// clearer than threading visibility.
    fn setup_king_shuffle_pos() -> Position {
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p
    }

    /// Four moves that, applied in order from `setup_king_shuffle_pos`,
    /// return the position to its starting state.
    fn shuffle_cycle() -> [Move; 4] {
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        [
            Move::make(Square::new(0, 8).unwrap(), Square::new(1, 8).unwrap(), bk),
            Move::make(Square::new(0, 0).unwrap(), Square::new(1, 0).unwrap(), wk),
            Move::make(Square::new(1, 8).unwrap(), Square::new(0, 8).unwrap(), bk),
            Move::make(Square::new(1, 0).unwrap(), Square::new(0, 0).unwrap(), wk),
        ]
    }

    #[test]
    fn move_completing_three_fold_is_allowed() {
        // After two full king-shuffle cycles, the start state appears in
        // history twice. The same BK side-step would push a state whose
        // `position_occurrences` then reaches 3 — three-fold, not 4-fold.
        // The move must still be in the legal list. Pins the
        // threshold-not-met case as a regression guard against any future
        // filter that fires too eagerly.
        let mut pos = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        for _ in 0..2 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        assert_eq!(
            pos.position_occurrences(),
            2,
            "two cycles should leave the start state matched twice in history",
        );
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK side-step should be legal at the post-2-cycle state \
             (its post-state would be the 3rd occurrence — below the 4-fold threshold)",
        );
    }

    #[test]
    fn move_completing_four_fold_is_allowed() {
        // After three full king-shuffle cycles, the start state appears in
        // history three times. The same BK side-step would push a state
        // whose `position_occurrences` then reaches 4. Under the pinned
        // "no-perft-change" semantics (design.md § "perft semantics under
        // repetition"), the move is NOT filtered — it stays in the legal
        // list. Pins the no-perft-change branch against future regression
        // toward filter-the-move.
        let mut pos = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        assert_eq!(
            pos.position_occurrences(),
            3,
            "three cycles should leave the start state matched 3× in history",
        );
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK side-step should remain legal even when its post-state would be the \
             4th occurrence — no-perft-change semantics, mirrors upstream's \
             repetition-blind LEGAL_ALL/perft",
        );
    }

    #[test]
    fn fresh_history_no_false_positive() {
        // Empty history; one move; the just-pushed entry's
        // `position_occurrences` is 1 (it matches itself per the inclusive
        // counting convention pinned in design.md § Encapsulation). The
        // move must be in the legal list of the pre-move position. Pins
        // that the (absent) 4-fold filter doesn't accidentally fire on a
        // fresh state.
        let pos = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK side-step should be legal from the fresh start state (empty history)",
        );

        let mut after = pos.clone();
        after.do_move(cycle[0]);
        assert_eq!(
            after.position_occurrences(),
            1,
            "after one move from empty history, the just-pushed entry matches itself once",
        );
    }

    // ---- 連続王手の千日手 (perpetual check): reference (repetition-blind) ----
    //
    // The reference `generate<LEGAL_ALL>` (`movegen.cpp`) carries no
    // repetition term of any kind: a move that completes a perpetual-check
    // 4-fold is still generated. This port carries no movegen-time
    // perpetual-check filter either; the consequences of perpetual check are
    // handled by the repetition scoring in the search and, in real games, by
    // server adjudication.
    //
    // These tests keep the same geometry as regression coverage but
    // assert the reference outcome: the perpetual-completing move IS in the
    // `generate_legal_all` list, exactly like every other legal move.

    /// Geometry for `perpetual_check_completing_move_is_generated`:
    /// White king at 5a (file=4, rank=0) in check from Black rook at 5b
    /// (file=4, rank=1). Black king at 1i (file=0, rank=8) far away,
    /// irrelevant. White to move (in check). Black is the perpetual
    /// checker; the 4-move cycle returns to this state with Black giving
    /// check on every Black move.
    fn setup_perpetual_check_pos() -> Position {
        let mut p = Position::empty();
        p.set_side_to_move(Color::White);
        p.board_mut().set(
            Square::new(4, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p.board_mut().set(
            Square::new(4, 1).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p
    }

    /// Cycle for `setup_perpetual_check_pos`: WK escape, BR check, WK
    /// escape back, BR check (returns to start).
    fn perpetual_check_cycle() -> [Move; 4] {
        let wk = Piece::new(PieceKind::King, Color::White);
        let br = Piece::new(PieceKind::Rook, Color::Black);
        [
            Move::make(Square::new(4, 0).unwrap(), Square::new(3, 0).unwrap(), wk),
            Move::make(Square::new(4, 1).unwrap(), Square::new(3, 1).unwrap(), br),
            Move::make(Square::new(3, 0).unwrap(), Square::new(4, 0).unwrap(), wk),
            Move::make(Square::new(3, 1).unwrap(), Square::new(4, 1).unwrap(), br),
        ]
    }

    #[test]
    fn perpetual_check_completing_move_is_generated() {
        // 3 full perpetual-check cycles + the first 3 moves of cycle 4 →
        // Black to move at the post-m3 state. The candidate cycle[3] (BR
        // 4b→5b) would push the start state for the 4th time
        // (`position_occurrences == 4`), and every Black move in the
        // most-recent cycle was a check. Under reference (repetition-blind)
        // semantics, the movegen does NOT filter it — the perpetual-check
        // consequence is scored by the search, not removed here.
        let mut pos = setup_perpetual_check_pos();
        let cycle = perpetual_check_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        pos.do_move(cycle[0]);
        pos.do_move(cycle[1]);
        pos.do_move(cycle[2]);
        assert_eq!(
            pos.side_to_move(),
            Color::Black,
            "after 3 cycles + m1+m2+m3 of cycle 4, Black is to move",
        );
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[3]),
            "BR 4b→5b is generated (reference is repetition-blind): {moves:?}",
        );
    }

    /// Geometry for `four_fold_with_check_on_most_but_not_all_cycle_steps`:
    /// same king/rook layout as the perpetual-check fixture, but the cycle
    /// detours the rook off-file before returning. m2 (BR 5b→9b, away from
    /// WK) is *not* a check; m4 (BR 9b→5b, returns) IS a check.
    fn setup_mixed_check_pos() -> Position {
        // Identical to `setup_perpetual_check_pos` — only the cycle differs.
        setup_perpetual_check_pos()
    }

    /// Cycle for `setup_mixed_check_pos`: WK escape, BR detour
    /// (non-check), WK return, BR back to original (check, returns to
    /// start state). Mover (Black) plays m2 and m4; only m4 gives check.
    fn mixed_check_cycle() -> [Move; 4] {
        let wk = Piece::new(PieceKind::King, Color::White);
        let br = Piece::new(PieceKind::Rook, Color::Black);
        [
            // m1 (W): WK 5a → 4a (escape from start-state check).
            Move::make(Square::new(4, 0).unwrap(), Square::new(3, 0).unwrap(), wk),
            // m2 (B): BR 5b → 9b — move off file 4, no check on WK at 4a.
            Move::make(Square::new(4, 1).unwrap(), Square::new(8, 1).unwrap(), br),
            // m3 (W): WK 4a → 5a — back; safe since no rook on file 4.
            Move::make(Square::new(3, 0).unwrap(), Square::new(4, 0).unwrap(), wk),
            // m4 (B): BR 9b → 5b — back; gives check (rook on WK's file).
            Move::make(Square::new(8, 1).unwrap(), Square::new(4, 1).unwrap(), br),
        ]
    }

    #[test]
    fn four_fold_with_check_on_most_but_not_all_cycle_steps_is_generated() {
        // Build the same 4-fold-completion shape as test 1, but with the
        // mixed-check cycle: Black's m2 doesn't give check; m4 does. The
        // candidate cycle[3] completes the 4-fold. Under reference
        // (repetition-blind) semantics the candidate is generated regardless
        // of the check pattern; the `position_occurrences == 4` sanity check
        // documents that this really is a 4-fold completion.
        let mut pos = setup_mixed_check_pos();
        let cycle = mixed_check_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        pos.do_move(cycle[0]);
        pos.do_move(cycle[1]);
        pos.do_move(cycle[2]);
        assert_eq!(pos.side_to_move(), Color::Black);

        let mut scratch = pos.clone();
        scratch.do_move(cycle[3]);
        assert_eq!(
            scratch.position_occurrences(),
            4,
            "candidate cycle[3] should complete the 4-fold: history = {:?}",
            scratch,
        );

        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[3]),
            "BR 9b→5b is generated (reference is repetition-blind): {moves:?}",
        );
    }

    /// Mirror of `setup_perpetual_check_pos` with colors flipped: BK at
    /// 5i (file=4, rank=8) in check from White rook at 5h (file=4,
    /// rank=7); WK at 1a (file=0, rank=0) far away. Black to move (in
    /// check). White is the perpetual checker.
    fn setup_white_perpetual_check_pos() -> Position {
        let mut p = Position::empty();
        // side_to_move defaults to Black in `Position::empty()`, but be
        // explicit in case the default changes.
        p.set_side_to_move(Color::Black);
        p.board_mut().set(
            Square::new(4, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 7).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::White)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p
    }

    /// Cycle for `setup_white_perpetual_check_pos`: BK escape, WR check,
    /// BK escape back, WR check (returns to start). Symmetric to
    /// `perpetual_check_cycle` with colors flipped.
    fn white_perpetual_check_cycle() -> [Move; 4] {
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wr = Piece::new(PieceKind::Rook, Color::White);
        [
            Move::make(Square::new(4, 8).unwrap(), Square::new(3, 8).unwrap(), bk),
            Move::make(Square::new(4, 7).unwrap(), Square::new(3, 7).unwrap(), wr),
            Move::make(Square::new(3, 8).unwrap(), Square::new(4, 8).unwrap(), bk),
            Move::make(Square::new(3, 7).unwrap(), Square::new(4, 7).unwrap(), wr),
        ]
    }

    #[test]
    fn four_fold_where_opponent_was_the_checker_is_legal_for_mover() {
        // Symmetric to test 1 with colors flipped. After 3 full cycles
        // we're back at the start state (Black to move, Black in check).
        // Candidate cycle[0] (BK escape) would complete a 4-fold for the
        // post-state X1, but White (the perpetual checker) is the side
        // that's been giving check on every cycle step. The
        // perpetual-check rule rejects only the *checker*'s move; the
        // chased side's escape moves are unaffected — cycle[0] must stay
        // in the legal list.
        let mut pos = setup_white_perpetual_check_pos();
        let cycle = white_perpetual_check_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        assert_eq!(pos.side_to_move(), Color::Black);
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK escape should remain legal — White is the perpetual checker, not Black: {moves:?}",
        );
    }

    #[test]
    fn non_repeating_check_move_is_unaffected() {
        // Empty history, no prior occurrences. A Black rook move that
        // gives check (rook to 5h, attacking WK on file 4) is *not* a
        // 4-fold completion, so the perpetual-check filter cannot fire.
        // Pins against a "I check ⇒ filtered" misimplementation.
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(8, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        let mut moves = Vec::new();
        p.generate_legal_all(&mut moves);
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let checking_move =
            Move::make(Square::new(0, 8).unwrap(), Square::new(4, 8).unwrap(), rook);
        assert!(
            moves.contains(&checking_move),
            "rook check (no prior history) should remain legal: {moves:?}",
        );
    }

    #[test]
    fn four_fold_without_any_check_is_generated() {
        // Reuse Phase 2's king-shuffle setup (no checks anywhere in the
        // cycle). After 3 cycles, the start state appears 3× in history.
        // The candidate cycle[0] (BK side-step) would push a state whose
        // post-`position_occurrences` reaches 4. A plain 4-fold makes the
        // game drawn, not the move illegal, so the move is generated.
        let mut pos = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        // Sanity: the candidate really does complete a 4-fold.
        let mut scratch = pos.clone();
        scratch.do_move(cycle[0]);
        assert_eq!(scratch.position_occurrences(), 4);

        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK side-step is generated — plain 4-fold is drawn, not illegal: {moves:?}",
        );
    }
}

// The piece-set `try_find_king` must agree square-for-square with the
// 81-square scan (`try_find_king_scan`) at every reachable position, and both
// must return `None` on a king-less board.
#[cfg(test)]
mod gate_318 {
    use super::{try_find_king, try_find_king_scan};
    use crate::color::Color;
    use crate::move_::Move;
    use crate::piece::{Piece, PieceKind};
    use crate::sfen::parse_sfen;
    use crate::square::Square;

    const SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn pick(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn piece_set_matches_scan_over_playouts() {
        const MIN_PLIES: usize = 60;
        for (fi, sfen) in SFENS.iter().enumerate() {
            let mut p = parse_sfen(sfen).expect("valid SFEN");
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (fi as u64).wrapping_add(1));
            let mut plies = 0usize;
            loop {
                for c in [Color::Black, Color::White] {
                    assert_eq!(
                        try_find_king(p.board(), c),
                        try_find_king_scan(p.board(), c),
                        "fixture {fi} ply {plies}: {c:?} king lookup diverges",
                    );
                }
                if plies >= MIN_PLIES {
                    break;
                }
                let mut legal = Vec::new();
                p.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    break;
                }
                let m: Move = legal[rng.pick(legal.len())];
                p.do_move(m);
                plies += 1;
            }
        }
    }

    #[test]
    fn king_less_board_returns_none() {
        let p = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").expect("valid SFEN");
        let mut board = *p.board();
        let bk = try_find_king(&board, Color::Black).expect("black king present");
        let wk = try_find_king(&board, Color::White).expect("white king present");

        // Remove the black king only: black side is now king-less, white unchanged.
        board.set(bk, None);
        assert_eq!(try_find_king(&board, Color::Black), None);
        assert_eq!(try_find_king_scan(&board, Color::Black), None);
        assert_eq!(try_find_king(&board, Color::White), Some(wk));
        assert_eq!(try_find_king_scan(&board, Color::White), Some(wk));

        // Remove the white king too: both sides king-less.
        board.set(wk, None);
        assert_eq!(try_find_king(&board, Color::Black), None);
        assert_eq!(try_find_king(&board, Color::White), None);

        // Put a lone king back at an arbitrary square and confirm it is found.
        let sq = Square::new(3, 5).unwrap();
        board.set(sq, Some(Piece::new(PieceKind::King, Color::Black)));
        assert_eq!(try_find_king(&board, Color::Black), Some(sq));
        assert_eq!(try_find_king_scan(&board, Color::Black), Some(sq));
    }
}
