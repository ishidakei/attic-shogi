//! One-ply mate detector (`Mate::mate_1ply`), ported from upstream YaneuraOu's
//! `source/mate/mate1ply_without_effect.cpp` at the current submodule pin.
//!
//! # Which reference implementation is active
//!
//! `source/config.h` defines `USE_MATE_1PLY` for the `YANEURAOU_ENGINE` family
//! (the NNUE / KPPT / KPP_KKPT / MATERIAL builds) and leaves `LONG_EFFECT_LIBRARY`
//! **undefined** for the standard (non-`MATERIAL_LEVEL>=2`) build. The mate
//! source guards select the implementation with
//! `#if defined(USE_MATE_1PLY) && !defined(LONG_EFFECT_LIBRARY)`, so the active
//! translation unit is `mate1ply_without_effect.cpp` (the Bonanza6-style,
//! table-driven detector) — **not** `mate1ply_with_effect.cpp`. This module ports
//! that file.
//!
//! # What is (and is not) ported — faithful to the reference's *misses*
//!
//! The reference detector is deliberately incomplete: it does not find every
//! legal 1-ply mate. In `mate1ply_without_effect.cpp` large blocks are compiled
//! out with `#if 0`:
//!
//! * distant dropped-rook / dropped-bishop "spear" mates (the `離し角・飛車`
//!   block, ~lines 1157-1452),
//! * double-check (`両王手`) mates and the 24-neighbour discovered-attack search
//!   (~lines 1630-2159),
//! * the (illegal) `打ち歩詰め` pawn-drop probe (~lines 885-904).
//!
//! Only the live code path is ported here, and parity means returning a mate
//! move / `None` in exactly the same positions as the reference — the misses are
//! preserved, not "fixed".
//!
//! ## Candidate tables
//!
//! The reference filters non-slider move origins through `CHECK_CAND_BB`
//! (`init_check_bb`): per piece-kind and enemy-king square, the superset of
//! origins from which that kind could deliver the relevant check. Those tables
//! are ported verbatim as compile-time consts in [`check_cand`] and each
//! non-slider move-mate loop pre-filters its iterated origins with the matching
//! mask before the inner verification tests. Because the table is a superset
//! filter — it only removes origins the inner tests would reject anyway — the
//! filtered origin set is iterated in the same ascending square order (matching
//! `bb.pop()`) and the first mate found (hence the returned move) is identical
//! in every position. The slider kinds (`dragon`/`rook`/`horse`/`bishop`) have
//! no candidate table (their reference cases are `#if 0`), so those loops stay
//! unfiltered. The retained unfiltered path — [`check_cand::Cands::unfiltered`],
//! all-ones masks — is the `#[cfg(test)]` equivalence oracle.
//!
//! The two pawn blocks additionally rely on the table to prove the origin holds
//! an (unpromoted) pawn and that the promotion destination lies in the enemy
//! field; both implicit facts remain reintroduced here as explicit guards (see
//! `mate_pawn_no_promote` / `mate_pawn_promote`). `CHECK_AROUND_BB` and
//! `NextSquare` are used only inside the `#if 0` blocks and are not needed.
//!
//! # Square indexing
//!
//! The engine's [`Square`] index is `(file-1)*9 + (rank-1)`, bit-for-bit
//! identical to YaneuraOu's `SQ`. Ascending `Square::index()` iteration therefore
//! reproduces the reference's `Bitboard::pop()` (least-significant-square-first)
//! order exactly, which is what makes the *returned move* match — the reference
//! returns the first mate it finds in that order.

use crate::bitboard::{self, Bitboard};
use crate::board::{Board, pat};
use crate::color::Color;
use crate::move_::Move;
use crate::movegen::{attackers_bb_occ, is_in_promotion_zone, step_signed, try_find_king};
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;

mod check_cand;
use check_cand::Cands;

// --------------------------------------------------------------------------
//  81-bit square set
// --------------------------------------------------------------------------

/// A set of board squares, one bit per [`Square::index`]. A thin newtype over
/// the shared [`Bitboard`] substrate that keeps this module's reference-shaped
/// method names (`andnot` matches YaneuraOu's `Bitboard::andnot`, `~self &
/// other`; `sub` is the more natural `self & ~other`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Bb(Bitboard);

impl Bb {
    const EMPTY: Bb = Bb(Bitboard::EMPTY);

    fn of(sq: Square) -> Bb {
        Bb(Bitboard::from_square(sq))
    }

    fn contains(self, sq: Square) -> bool {
        self.0.test(sq)
    }

    fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    /// `true` iff more than one bit is set (the reference's `more_than_one`).
    fn more_than_one(self) -> bool {
        self.0.popcount() > 1
    }

    fn with(self, sq: Square) -> Bb {
        Bb(self.0 | Bitboard::from_square(sq))
    }

    fn without(self, sq: Square) -> Bb {
        Bb(self.0 & !Bitboard::from_square(sq))
    }

    fn and(self, o: Bb) -> Bb {
        Bb(self.0 & o.0)
    }

    fn or(self, o: Bb) -> Bb {
        Bb(self.0 | o.0)
    }

    /// `self` with the bits of `o` removed (`self & ~o`).
    fn sub(self, o: Bb) -> Bb {
        Bb(self.0 & !o.0)
    }

    /// YaneuraOu's `Bitboard::andnot`: `~self & o`.
    fn andnot(self, o: Bb) -> Bb {
        Bb(!self.0 & o.0)
    }

    /// Ascending-square-index iteration (least-significant square first),
    /// matching `Bitboard::pop()`.
    fn iter(self) -> bitboard::BitboardIter {
        self.0.squares()
    }
}

// --------------------------------------------------------------------------
//  Effect (attack-pattern) helpers
// --------------------------------------------------------------------------

/// Forward rank delta for a side's own advance (Black toward rank 0, White
/// toward rank 8). Distinct from `movegen::dr_sign_for`, which multiplies
/// black-orientation step tables.
fn forward_dr(color: Color) -> i8 {
    match color {
        Color::Black => -1,
        Color::White => 1,
    }
}

// The effect primitives below are table lookups / Qugiy sliders on the shared
// [`bitboard`] substrate, so no attack set is constructed per call on the hot
// path. Scalar forms that step square-by-square over delta lists (`step_effect`
// / `ray_effect` / `slide_effect` and their wrappers) live in [`scan_oracle`] as
// `#[cfg(test)]` equivalence oracles; the effect-equivalence gate asserts each
// primitive below matches its scalar oracle bit-for-bit over every square,
// colour, and (for the occupancy-aware sliders) sampled occupancy.

fn pawn_effect(c: Color, sq: Square) -> Bb {
    Bb(bitboard::pawn_attacks(c, sq))
}

fn knight_effect(c: Color, sq: Square) -> Bb {
    Bb(bitboard::knight_attacks(c, sq))
}

fn silver_effect(c: Color, sq: Square) -> Bb {
    Bb(bitboard::silver_attacks(c, sq))
}

fn gold_effect(c: Color, sq: Square) -> Bb {
    Bb(bitboard::gold_attacks(c, sq))
}

/// The king ring — colour-symmetric, so the shared table's Black slice serves.
fn king_effect(sq: Square) -> Bb {
    Bb(bitboard::king_attacks(Color::Black, sq))
}

fn cross45_step_effect(sq: Square) -> Bb {
    Bb(bitboard::cross45_step_effect(sq))
}

fn rook_step_effect(sq: Square) -> Bb {
    Bb(bitboard::rook_step_effect(sq))
}

fn bishop_step_effect(sq: Square) -> Bb {
    Bb(bitboard::bishop_step_effect(sq))
}

fn lance_step_effect(c: Color, sq: Square) -> Bb {
    Bb(bitboard::lance_step_effect(c, sq))
}

fn rook_effect(sq: Square, occ: Bb) -> Bb {
    Bb(bitboard::rook_attacks(sq, occ.0))
}

fn bishop_effect(sq: Square, occ: Bb) -> Bb {
    Bb(bitboard::bishop_attacks(sq, occ.0))
}

fn lance_effect(c: Color, sq: Square, occ: Bb) -> Bb {
    Bb(bitboard::lance_attacks(c, sq, occ.0))
}

fn horse_effect(sq: Square, occ: Bb) -> Bb {
    Bb(bitboard::horse_attacks(sq, occ.0))
}

fn dragon_effect(sq: Square, occ: Bb) -> Bb {
    Bb(bitboard::dragon_attacks(sq, occ.0))
}

// --------------------------------------------------------------------------
//  Board-wide square-set snapshots
// --------------------------------------------------------------------------

/// Every occupied square (bitboard substrate).
fn occupied(board: &Board) -> Bb {
    Bb(board.occupied())
}

/// The occupied squares of `color` (bitboard substrate).
fn pieces_of_color(board: &Board, color: Color) -> Bb {
    Bb(board.pieces_color(color))
}

/// Both kings, from the per-`(colour, pattern)` piece sets.
fn both_kings(board: &Board) -> Bb {
    Bb(board.pieces_pattern(Color::Black, pat::KING)
        | board.pieces_pattern(Color::White, pat::KING))
}

/// `pieces(color, pattern)` — the `color` pieces in a single [`pat`] attack
/// bucket, from the board's piece sets. Each move-mate
/// loop's `is_<kind>` predicate maps 1:1 to a pattern slot (e.g. `is_dragon` ⇔
/// [`pat::DRAGON`], `is_golds` ⇔ [`pat::GOLD`], `is_rook_unpromoted` ⇔
/// [`pat::ROOK`]). Iterated ascending to match `bb.pop()` order.
fn pieces_bucket(board: &Board, color: Color, pattern: usize) -> Bb {
    Bb(board.pieces_pattern(color, pattern))
}

/// An un-promoted pawn (`pieces(PAWN)` membership) — the one `is_<kind>`
/// predicate still consulted as a point test (the candidate-table guard in
/// [`mate_pawn_no_promote`], which reads a single square rather than a set).
fn is_pawn_unpromoted(p: Piece) -> bool {
    p.kind == PieceKind::Pawn && !p.promoted
}

/// `color`'s pieces (of any kind) attacking `sq` under occupancy `occ`. A
/// square is considered only if it is occupied *in `occ`* — clearing the mover's
/// `from` square therefore drops that piece from the set, which is exactly the
/// reference's `attackers_to<C>(sq, slide) ^ from` net (the moved piece is
/// removed) while still revealing any x-ray slider behind it.
///
/// Bitboard form: the reverse-attack lookup
/// ([`crate::movegen::attackers_bb_occ`], evaluated against `occ`) intersected
/// with `occ` itself — the intersection reproduces the scalar scan's "only
/// squares present in `occ`" restriction (the reverse lookup sources attacker
/// squares from the board's piece sets, which still include the mover's `from`).
fn attackers_of_color(board: &Board, sq: Square, occ: Bb, color: Color) -> Bb {
    let occ_bb = occ.0;
    Bb(attackers_bb_occ(board, sq, color, occ_bb) & occ_bb)
}

// --------------------------------------------------------------------------
//  Geometry: aligned / between
// --------------------------------------------------------------------------

/// `aligned(s1, s2, s3)` with `s3` the king: `s1` and `s2` lie in the **same**
/// direction from `s3`. Mirrors the reference `directions_of(s1,s3) &&
/// directions_of(s1,s3) == directions_of(s2,s3)` — a straight line *through* the
/// king (opposite sides) is not aligned. Backed by the shared [`bitboard::ray_dir`]
/// table (the reference `Effect8::directions_of`).
fn aligned(s1: Square, s2: Square, s3: Square) -> bool {
    match (bitboard::ray_dir(s1, s3), bitboard::ray_dir(s2, s3)) {
        (Some(d1), Some(d2)) => d1 == d2,
        _ => false,
    }
}

/// Squares strictly between `a` and `b` when they share a queen line, else the
/// empty set (the reference's `between_bb`). The shared [`bitboard::between`]
/// table lookup.
fn between_bb(a: Square, b: Square) -> Bb {
    Bb(bitboard::between(a, b))
}

// --------------------------------------------------------------------------
//  Pins / blockers
// --------------------------------------------------------------------------

/// Enemy sliders aimed at `ksq` (king of `king_color`) along their step lines,
/// optionally excluding `avoid`. Mirrors the sniper set of both
/// `update_slider_blockers` and `pinned_pieces(avoid)`.
///
/// Bitboard form: the enemy rook/dragon on the king's orthogonal step-effect,
/// enemy bishop/horse on its diagonal step-effect, and enemy (unpromoted) lance
/// on `king_color`'s forward lance-line — each computed as
/// `pattern_set & <line>`, then `avoid` cleared. The `*_attacks(.., EMPTY)`
/// queries give the full ray to the edge (the reference's `...StepEffect`).
fn snipers_to_king(board: &Board, ksq: Square, king_color: Color, avoid: Option<Square>) -> Bb {
    let enemy = king_color.flip();
    let rook_line = bitboard::rook_attacks(ksq, Bitboard::EMPTY);
    let bishop_line = bitboard::bishop_attacks(ksq, Bitboard::EMPTY);
    let lance_line = bitboard::lance_attacks(king_color, ksq, Bitboard::EMPTY);
    let rook_dragon =
        board.pieces_pattern(enemy, pat::ROOK) | board.pieces_pattern(enemy, pat::DRAGON);
    let bishop_horse =
        board.pieces_pattern(enemy, pat::BISHOP) | board.pieces_pattern(enemy, pat::HORSE);
    let lance = board.pieces_pattern(enemy, pat::LANCE);
    let mut snipers =
        (rook_line & rook_dragon) | (bishop_line & bishop_horse) | (lance_line & lance);
    if let Some(a) = avoid {
        snipers.clear(a);
    }
    Bb(snipers)
}

/// `st->blockersForKing[c]` via `update_slider_blockers`: pieces that are the
/// sole occupant between `c`'s king and an enemy sniper, with all snipers
/// removed from the occupancy first. Delegates to the shared bitboard
/// [`crate::see::slider_blockers`] (whose `.0` is exactly `blockersForKing[c]`).
///
/// Production `mate_ctx` does not call this — it reads the identical blocker
/// sets from the per-position check-info cache. It is kept under
/// `#[cfg(test)]` as the reference the scan-oracle equivalence gate checks the
/// bitboard `slider_blockers` form against.
#[cfg(test)]
fn blockers_for_king(board: &Board, c: Color) -> Bb {
    Bb(crate::see::slider_blockers(board, c).0)
}

/// `Position::pinned_pieces<C>(avoid)`: `C`'s own pieces pinned to `C`'s king,
/// computed with `avoid` removed from both the sniper set and the between-count.
/// Unlike [`blockers_for_king`] this does **not** strip other snipers from the
/// occupancy (it counts against the full `pieces()`), matching the reference's
/// distinct formula used for `new_pin`.
fn pinned_pieces_avoid(board: &Board, c: Color, avoid: Option<Square>) -> Bb {
    let Some(ksq) = try_find_king(board, c) else {
        return Bb::EMPTY;
    };
    let snipers = snipers_to_king(board, ksq, c, avoid);
    let occ = occupied(board);
    let own = pieces_of_color(board, c);
    let mut result = Bb::EMPTY;
    for pinner in snipers.iter() {
        let mut b = between_bb(ksq, pinner).and(occ);
        if let Some(a) = avoid {
            b = b.without(a);
        }
        if !b.more_than_one() {
            result = result.or(b.and(own));
        }
    }
    result
}

// --------------------------------------------------------------------------
//  Mate helper predicates
// --------------------------------------------------------------------------

/// Context threaded through the whole detector, computed once per query.
struct Ctx<'a> {
    board: &'a Board,
    us: Color,
    them: Color,
    sq_king: Square,
    our_king: Square,
    occ_all: Bb,
    occ_us: Bb,
    bb_drop: Bb,
    bb_move: Bb,
    /// Them's pieces pinned to Them's king.
    pinned: Bb,
    /// Us pieces that give discovered check to Them's king when moved.
    dc_candidates: Bb,
    /// Us pieces pinned to Us's own king.
    our_pinned: Bb,
}

/// `discovered(from, to, ourKing, ourPinned)` — moving `from`→`to` exposes Us's
/// own king (an illegal move).
fn discovered(ctx: &Ctx, from: Square, to: Square) -> bool {
    !ctx.our_pinned.is_empty() && ctx.our_pinned.contains(from) && !aligned(from, to, ctx.our_king)
}

/// Can the `king_color` king reach a square other than `to` / `bb_avoid`?
///
/// `from` distinguishes the two reference overloads: `Some(_)` is the move
/// variant (the mover left `from`; the king is removed from the occupancy so an
/// x-ray behind it is seen), `None` is the drop variant (the piece appears on
/// `to`; the king is *not* removed — the reference's deliberate omission). In
/// both, `base_occ` is the caller's `slide_` (for the move variant it already
/// has `from` cleared).
fn can_king_escape(
    board: &Board,
    king_color: Color,
    from: Option<Square>,
    to: Square,
    bb_avoid: Bb,
    base_occ: Bb,
    own: Bb,
) -> bool {
    let attacker = king_color.flip();
    let king_sq = match try_find_king(board, king_color) {
        Some(s) => s,
        None => return false,
    };
    let mut occ = base_occ.with(to);
    if from.is_some() {
        occ = occ.without(king_sq);
    }
    // king 8-neighbourhood, minus the just-attacked squares, `to`, and own pieces.
    let blocked = bb_avoid.or(Bb::of(to)).or(own);
    let escapes = king_effect(king_sq).sub(blocked);
    for esc in escapes.iter() {
        // The mover's `from` is already excluded from `attackers_of_color`
        // (it is cleared in `occ` for the move variant), matching the
        // reference's `Bitboard(from).andnot(attackers_to(...))`.
        if attackers_of_color(board, esc, occ, attacker).is_empty() {
            return true;
        }
    }
    false
}

/// Can a `king_color` piece *other than the king* capture on `to` without being
/// pinned off its line? (`can_piece_capture`; `pinned` is `king_color`'s pin
/// set, `slide` the occupancy.)
fn can_piece_capture(board: &Board, king_color: Color, to: Square, pinned: Bb, slide: Bb) -> bool {
    let king_sq = match try_find_king(board, king_color) {
        Some(s) => s,
        None => return false,
    };
    let attackers = attackers_of_color(board, to, slide, king_color).sub(both_kings(board));
    for from in attackers.iter() {
        if pinned.is_empty() || !pinned.contains(from) || aligned(from, to, king_sq) {
            return true;
        }
    }
    false
}

// --------------------------------------------------------------------------
//  Drop mates
// --------------------------------------------------------------------------

/// Returned-move constructor for a board move whose piece stands on `from`.
fn make_move(board: &Board, from: Square, to: Square) -> Move {
    Move::make(from, to, board.get(from).expect("mover on `from`"))
}

/// Returned-move constructor for a promoting board move (the piece on `from` is
/// unpromoted).
fn make_move_promote(board: &Board, from: Square, to: Square) -> Move {
    Move::make_promote(from, to, board.get(from).expect("mover on `from`"))
}

fn drop_mates(ctx: &Ctx, hand: &crate::hand::Hand) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let sq_king = ctx.sq_king;
    let them_pieces = ctx.occ_them_pieces();

    let has = |k: PieceKind| hand.count(k) > 0;

    // -- Rook, dropped adjacent (orthogonally) to the king.
    if has(PieceKind::Rook) {
        let bb = rook_step_effect(sq_king)
            .and(king_effect(sq_king))
            .and(ctx.bb_drop);
        for to in bb.iter() {
            if attackers_of_color(board, to, ctx.occ_all, us).is_empty() {
                continue;
            }
            let bb_attacks = rook_step_effect(to);
            if can_king_escape(board, them, None, to, bb_attacks, ctx.occ_all, them_pieces) {
                continue;
            }
            if can_piece_capture(board, them, to, ctx.pinned, ctx.occ_all) {
                continue;
            }
            return Some(Move::make_drop(PieceKind::Rook, us, to));
        }
    }

    // -- Lance, dropped on the single square directly in front of the king.
    if has(PieceKind::Lance) {
        let bb = pawn_effect(them, sq_king).and(ctx.bb_drop);
        if let Some(to) = bb.iter().next()
            && !attackers_of_color(board, to, ctx.occ_all, us).is_empty()
        {
            let bb_attacks = lance_step_effect(us, to);
            if !can_king_escape(board, them, None, to, bb_attacks, ctx.occ_all, them_pieces)
                && !can_piece_capture(board, them, to, ctx.pinned, ctx.occ_all)
            {
                return Some(Move::make_drop(PieceKind::Lance, us, to));
            }
        }
    }

    // -- Bishop, dropped diagonally adjacent to the king.
    if has(PieceKind::Bishop) {
        let bb = cross45_step_effect(sq_king).and(ctx.bb_drop);
        for to in bb.iter() {
            if attackers_of_color(board, to, ctx.occ_all, us).is_empty() {
                continue;
            }
            let bb_attacks = bishop_step_effect(to);
            if can_king_escape(board, them, None, to, bb_attacks, ctx.occ_all, them_pieces) {
                continue;
            }
            if can_piece_capture(board, them, to, ctx.pinned, ctx.occ_all) {
                continue;
            }
            return Some(Move::make_drop(PieceKind::Bishop, us, to));
        }
    }

    // -- Gold.
    if has(PieceKind::Gold) {
        let mut bb = gold_effect(them, sq_king).and(ctx.bb_drop);
        // With a rook in hand the square directly in front of the king was
        // already tried (a rook there mates if a gold would), so drop it.
        if has(PieceKind::Rook) {
            bb = bb.sub(pawn_effect(us, sq_king));
        }
        for to in bb.iter() {
            if attackers_of_color(board, to, ctx.occ_all, us).is_empty() {
                continue;
            }
            let bb_attacks = gold_effect(us, to);
            if can_king_escape(board, them, None, to, bb_attacks, ctx.occ_all, them_pieces) {
                continue;
            }
            if can_piece_capture(board, them, to, ctx.pinned, ctx.occ_all) {
                continue;
            }
            return Some(Move::make_drop(PieceKind::Gold, us, to));
        }
    }

    // -- Silver.
    if has(PieceKind::Silver) {
        // Gold-drop already covered the gold-attack squares; a bishop in hand
        // covers the rest, so silver can add nothing when both are held.
        let skip = has(PieceKind::Gold) && has(PieceKind::Bishop);
        if !skip {
            let bb = if has(PieceKind::Gold) {
                silver_effect(them, sq_king).and(gold_effect(them, sq_king).andnot(ctx.bb_drop))
            } else {
                silver_effect(them, sq_king).and(ctx.bb_drop)
            };
            for to in bb.iter() {
                if attackers_of_color(board, to, ctx.occ_all, us).is_empty() {
                    continue;
                }
                let bb_attacks = silver_effect(us, to);
                if can_king_escape(board, them, None, to, bb_attacks, ctx.occ_all, them_pieces) {
                    continue;
                }
                if can_piece_capture(board, them, to, ctx.pinned, ctx.occ_all) {
                    continue;
                }
                return Some(Move::make_drop(PieceKind::Silver, us, to));
            }
        }
    }

    // -- Knight (no support test: the king cannot reach a knight's square).
    if has(PieceKind::Knight) {
        let bb = knight_effect(them, sq_king).and(ctx.bb_drop);
        for to in bb.iter() {
            if can_king_escape(board, them, None, to, Bb::EMPTY, ctx.occ_all, them_pieces) {
                continue;
            }
            if can_piece_capture(board, them, to, ctx.pinned, ctx.occ_all) {
                continue;
            }
            return Some(Move::make_drop(PieceKind::Knight, us, to));
        }
    }

    None
}

// --------------------------------------------------------------------------
//  Move mates
// --------------------------------------------------------------------------

impl Ctx<'_> {
    fn occ_them_pieces(&self) -> Bb {
        self.occ_all.sub(self.occ_us)
    }
}

fn mate_dragon(ctx: &Ctx) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    for from in pieces_bucket(board, us, pat::DRAGON).iter() {
        let slide = ctx.occ_all.without(from);
        let bb_check = dragon_effect(from, slide)
            .and(ctx.bb_move)
            .and(king_effect(ctx.sq_king));
        let new_pin = pinned_pieces_avoid(board, them, Some(from));
        for to in bb_check.iter() {
            if attackers_of_color(board, to, slide, us).is_empty() {
                continue;
            }
            if discovered(ctx, from, to) {
                continue;
            }
            let bb_attacks = if cross45_step_effect(ctx.sq_king).contains(to) {
                dragon_effect(to, slide)
            } else {
                rook_step_effect(to).or(king_effect(to))
            };
            if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
                continue;
            }
            if can_piece_capture(board, them, to, new_pin, slide) {
                continue;
            }
            return Some(make_move(board, from, to));
        }
    }
    None
}

fn mate_rook(ctx: &Ctx) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    for from in pieces_bucket(board, us, pat::ROOK).iter() {
        let slide = ctx.occ_all.without(from);
        let bb_check = rook_effect(from, slide)
            .and(ctx.bb_move)
            .and(king_effect(ctx.sq_king));
        let new_pin = pinned_pieces_avoid(board, them, Some(from));
        for to in bb_check.iter() {
            if attackers_of_color(board, to, slide, us).is_empty() {
                continue;
            }
            let promote = can_promote(us, from, to);
            let bb_attacks = if promote {
                if cross45_step_effect(ctx.sq_king).contains(to) {
                    dragon_effect(to, slide)
                } else {
                    rook_step_effect(to).or(king_effect(to))
                }
            } else {
                rook_step_effect(to)
            };
            if !bb_attacks.contains(ctx.sq_king) {
                continue;
            }
            if discovered(ctx, from, to) {
                continue;
            }
            if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
                continue;
            }
            if !ctx.dc_candidates.contains(from)
                && can_piece_capture(board, them, to, new_pin, slide)
            {
                continue;
            }
            return Some(if promote {
                make_move_promote(board, from, to)
            } else {
                make_move(board, from, to)
            });
        }
    }
    None
}

fn mate_horse(ctx: &Ctx) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    for from in pieces_bucket(board, us, pat::HORSE).iter() {
        let slide = ctx.occ_all.without(from);
        let bb_check = horse_effect(from, slide)
            .and(ctx.bb_move)
            .and(king_effect(ctx.sq_king));
        let new_pin = pinned_pieces_avoid(board, them, Some(from));
        for to in bb_check.iter() {
            if attackers_of_color(board, to, slide, us).is_empty() {
                continue;
            }
            if discovered(ctx, from, to) {
                continue;
            }
            let bb_attacks = bishop_step_effect(to).or(king_effect(to));
            if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
                continue;
            }
            let two_check = ctx.dc_candidates.contains(from) && !aligned(from, to, ctx.sq_king);
            if !two_check && can_piece_capture(board, them, to, new_pin, slide) {
                continue;
            }
            return Some(make_move(board, from, to));
        }
    }
    None
}

fn mate_bishop(ctx: &Ctx) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    for from in pieces_bucket(board, us, pat::BISHOP).iter() {
        let slide = ctx.occ_all.without(from);
        let bb_check = bishop_effect(from, slide)
            .and(ctx.bb_move)
            .and(king_effect(ctx.sq_king));
        let new_pin = pinned_pieces_avoid(board, them, Some(from));
        for to in bb_check.iter() {
            if attackers_of_color(board, to, slide, us).is_empty() {
                continue;
            }
            let promote = can_promote(us, from, to);
            let bb_attacks = if promote {
                bishop_step_effect(to).or(king_effect(to))
            } else {
                bishop_step_effect(to)
            };
            if !bb_attacks.contains(ctx.sq_king) {
                continue;
            }
            if discovered(ctx, from, to) {
                continue;
            }
            if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
                continue;
            }
            if !ctx.dc_candidates.contains(from)
                && can_piece_capture(board, them, to, new_pin, slide)
            {
                continue;
            }
            return Some(if promote {
                make_move_promote(board, from, to)
            } else {
                make_move(board, from, to)
            });
        }
    }
    None
}

fn mate_lance(ctx: &Ctx, cand: Bb) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    // Enemy first-rank-from-third (RANK_3 for Black, RANK_7 for White): where an
    // un-promoted lance can skewer.
    let skewer_rank: u8 = if us == Color::Black { 2 } else { 6 };
    for from in pieces_bucket(board, us, pat::LANCE).and(cand).iter() {
        let slide = ctx.occ_all.without(from);
        let bb_check = lance_effect(us, from, slide)
            .and(ctx.bb_move)
            .and(gold_effect(them, ctx.sq_king));
        for to in bb_check.iter() {
            // First: promotion-to-gold check (or a straight non-promoting check
            // when `to` is not in the promotion zone).
            let bb_attacks = if can_promote_to(us, to) {
                gold_effect(us, to)
            } else {
                lance_step_effect(us, to)
            };
            let promote_branch_ok = bb_attacks.contains(ctx.sq_king)
                && !attackers_of_color(board, to, slide, us).is_empty()
                && !discovered(ctx, from, to)
                && !can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces)
                && (ctx.dc_candidates.contains(from)
                    || !can_piece_capture(board, them, to, ctx.pinned, slide));
            if promote_branch_ok {
                return Some(if can_promote_to(us, to) {
                    make_move_promote(board, from, to)
                } else {
                    make_move(board, from, to)
                });
            }
            // Otherwise: an un-promoted skewer from the enemy third rank.
            if to.rank() == skewer_rank {
                let bb_attacks = lance_step_effect(us, to);
                if bb_attacks.contains(ctx.sq_king)
                    && !attackers_of_color(board, to, slide, us).is_empty()
                    && !discovered(ctx, from, to)
                    && !can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces)
                    && !can_piece_capture(board, them, to, ctx.pinned, slide)
                {
                    return Some(make_move(board, from, to));
                }
            }
        }
    }
    None
}

fn mate_gold(ctx: &Ctx, cand: Bb) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    for from in pieces_bucket(board, us, pat::GOLD).and(cand).iter() {
        let bb_check = gold_effect(us, from)
            .and(ctx.bb_move)
            .and(gold_effect(them, ctx.sq_king));
        if bb_check.is_empty() {
            continue;
        }
        let slide = ctx.occ_all.without(from);
        let new_pin = pinned_pieces_avoid(board, them, Some(from));
        for to in bb_check.iter() {
            if attackers_of_color(board, to, slide, us).is_empty() {
                continue;
            }
            if discovered(ctx, from, to) {
                continue;
            }
            let bb_attacks = gold_effect(us, to);
            if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
                continue;
            }
            let two_check = ctx.dc_candidates.contains(from) && !aligned(from, to, ctx.sq_king);
            if !two_check && can_piece_capture(board, them, to, new_pin, slide) {
                continue;
            }
            return Some(make_move(board, from, to));
        }
    }
    None
}

fn mate_silver(ctx: &Ctx, cand: Bb) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    for from in pieces_bucket(board, us, pat::SILVER).and(cand).iter() {
        let bb_check = silver_effect(us, from)
            .and(ctx.bb_move)
            .and(king_effect(ctx.sq_king));
        if bb_check.is_empty() {
            continue;
        }
        let slide = ctx.occ_all.without(from);
        let new_pin = pinned_pieces_avoid(board, them, Some(from));
        for to in bb_check.iter() {
            // Non-promoting silver check.
            let bb_attacks = silver_effect(us, to);
            let two_check = ctx.dc_candidates.contains(from) && !aligned(from, to, ctx.sq_king);
            let plain_ok = bb_attacks.contains(ctx.sq_king)
                && !attackers_of_color(board, to, slide, us).is_empty()
                && !discovered(ctx, from, to)
                && !can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces)
                && (two_check || !can_piece_capture(board, them, to, new_pin, slide));
            if plain_ok {
                return Some(make_move(board, from, to));
            }
            // Promoting silver (to gold) check.
            if !can_promote(us, from, to) {
                continue;
            }
            let bb_attacks = gold_effect(us, to);
            if !bb_attacks.contains(ctx.sq_king) {
                continue;
            }
            if attackers_of_color(board, to, slide, us).is_empty() {
                continue;
            }
            if discovered(ctx, from, to) {
                continue;
            }
            if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
                continue;
            }
            if !two_check && can_piece_capture(board, them, to, new_pin, slide) {
                continue;
            }
            return Some(make_move_promote(board, from, to));
        }
    }
    None
}

fn mate_knight(ctx: &Ctx, cand: Bb) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    for from in pieces_bucket(board, us, pat::KNIGHT).and(cand).iter() {
        let bb_check = knight_effect(us, from).and(ctx.bb_move);
        if bb_check.is_empty() {
            continue;
        }
        let slide = ctx.occ_all.without(from);
        let new_pin = pinned_pieces_avoid(board, them, Some(from));
        for to in bb_check.iter() {
            let bb_attacks = knight_effect(us, to);
            if bb_attacks.contains(ctx.sq_king) {
                // Non-promoting knight check (needs no support test — the king
                // cannot reach a knight square). A knight cannot both promote-
                // and non-promote-check the same square, so a failure here does
                // not fall through to the promotion branch.
                if discovered(ctx, from, to) {
                    continue;
                }
                if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
                    continue;
                }
                if !ctx.dc_candidates.contains(from)
                    && can_piece_capture(board, them, to, new_pin, slide)
                {
                    continue;
                }
                return Some(make_move(board, from, to));
            }
            // Promoting knight (to gold) check.
            if !can_promote(us, from, to) {
                continue;
            }
            let bb_attacks = gold_effect(us, to);
            if !bb_attacks.contains(ctx.sq_king) {
                continue;
            }
            if attackers_of_color(board, to, slide, us).is_empty() {
                continue;
            }
            if discovered(ctx, from, to) {
                continue;
            }
            if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
                continue;
            }
            if !ctx.dc_candidates.contains(from)
                && can_piece_capture(board, them, to, new_pin, slide)
            {
                continue;
            }
            return Some(make_move_promote(board, from, to));
        }
    }
    None
}

/// Non-promoting pawn-push mate (`PIECE_TYPE_CHECK_PAWN_WITH_NO_PRO`). The
/// reference computes the unique `to` (directly in front of the king) and
/// `from`; the candidate-table guard (dropped here) implied that `from` holds an
/// unpromoted Us pawn — reintroduced as an explicit check.
fn mate_pawn_no_promote(ctx: &Ctx, cand: Bb) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    // Candidate pre-filter: a Us pawn must stand on the single candidate origin.
    if cand.and(pieces_bucket(board, us, pat::PAWN)).is_empty() {
        return None;
    }
    let push = (0i8, -forward_dr(us)); // `SQ_D` for Black, `SQ_U` for White.
    let to = step_signed(ctx.sq_king, push.0, push.1)?;
    // `to` must be empty or an enemy piece (never blocked by our own).
    if let Some(p) = board.get(to)
        && p.color == us
    {
        return None;
    }
    let from = step_signed(to, push.0, push.1)?;
    // Candidate-table guard: the pushing pawn must actually stand on `from`.
    match board.get(from) {
        Some(p) if p.color == us && is_pawn_unpromoted(p) => {}
        _ => return None,
    }
    // A push into the promotion zone is handled by the promoting-pawn block.
    if can_promote_to(us, to) {
        return None;
    }
    let slide = ctx.occ_all.without(from);
    if attackers_of_color(board, to, slide, us).is_empty() {
        return None;
    }
    if discovered(ctx, from, to) {
        return None;
    }
    if can_king_escape(board, them, Some(from), to, Bb::EMPTY, slide, them_pieces) {
        return None;
    }
    if can_piece_capture(board, them, to, ctx.pinned, slide) {
        return None;
    }
    Some(make_move(board, from, to))
}

/// Promoting pawn-push mate (`PIECE_TYPE_CHECK_PAWN_WITH_PRO`). The dropped table
/// guaranteed the promotion destination sits in the enemy field; reintroduced
/// here as `is_in_promotion_zone(to)`.
fn mate_pawn_promote(ctx: &Ctx, cand: Bb) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    let push = (0i8, forward_dr(us));
    for from in pieces_bucket(board, us, pat::PAWN).and(cand).iter() {
        let Some(to) = step_signed(from, push.0, push.1) else {
            continue;
        };
        if let Some(p) = board.get(to)
            && p.color == us
        {
            continue;
        }
        // The promotion (Tokin) check pattern is a gold.
        let bb_attacks = gold_effect(us, to);
        if !bb_attacks.contains(ctx.sq_king) {
            continue;
        }
        // Candidate-table guarantee: `to` must be in the promotion zone.
        if !is_in_promotion_zone(to, us) {
            continue;
        }
        let slide = ctx.occ_all.without(from);
        if attackers_of_color(board, to, slide, us).is_empty() {
            continue;
        }
        if discovered(ctx, from, to) {
            continue;
        }
        if can_king_escape(board, them, Some(from), to, bb_attacks, slide, them_pieces) {
            continue;
        }
        if can_piece_capture(board, them, to, ctx.pinned, slide) {
            continue;
        }
        return Some(make_move_promote(board, from, to));
    }
    None
}

/// The non-slider move-mate group (gold, silver, knight, pawn-push,
/// pawn-promote), guarded by the reference's `PIECE_TYPE_CHECK_NON_SLIDER`
/// early-out: if no Us gold / silver / knight / pawn stands on a non-slider
/// candidate square, none of the blocks can fire, so skip them all.
fn mate_non_slider(ctx: &Ctx, cands: &Cands) -> Option<Move> {
    let board = ctx.board;
    let us = ctx.us;
    let group = pieces_bucket(board, us, pat::GOLD)
        .or(pieces_bucket(board, us, pat::SILVER))
        .or(pieces_bucket(board, us, pat::KNIGHT))
        .or(pieces_bucket(board, us, pat::PAWN));
    if cands.non_slider.and(group).is_empty() {
        return None;
    }
    mate_gold(ctx, cands.gold)
        .or_else(|| mate_silver(ctx, cands.silver))
        .or_else(|| mate_knight(ctx, cands.knight))
        .or_else(|| mate_pawn_no_promote(ctx, cands.pawn_no_pro))
        .or_else(|| mate_pawn_promote(ctx, cands.pawn_pro))
}

/// The move-mate chain: unfiltered sliders (dragon, rook, horse, bishop), then
/// the candidate-filtered lance and non-slider group. `cands` supplies the
/// per-kind `CHECK_CAND_BB` masks (or all-ones in the unfiltered twin).
fn mate_moves(ctx: &Ctx, cands: &Cands) -> Option<Move> {
    mate_dragon(ctx)
        .or_else(|| mate_rook(ctx))
        .or_else(|| mate_horse(ctx))
        .or_else(|| mate_bishop(ctx))
        .or_else(|| mate_lance(ctx, cands.lance))
        .or_else(|| mate_non_slider(ctx, cands))
}

/// `canPromote(c, from, to)` — either endpoint in the enemy field.
fn can_promote(c: Color, from: Square, to: Square) -> bool {
    is_in_promotion_zone(from, c) || is_in_promotion_zone(to, c)
}

/// `canPromote(c, to)` — the destination endpoint only.
fn can_promote_to(c: Color, to: Square) -> bool {
    is_in_promotion_zone(to, c)
}

impl Position {
    /// One-ply mate detector: if the side to move can deliver mate in a single
    /// move, return that move; otherwise `None`.
    ///
    /// A faithful port of upstream YaneuraOu's `Mate::mate_1ply`
    /// (`mate1ply_without_effect.cpp`) — including its deliberate misses (see the
    /// module docs). Its precondition, like the reference's
    /// `ASSERT_LV3(!pos.checkers())`, is that the side to move is **not** in
    /// check; the search only calls it at such nodes.
    pub fn mate_1ply(&self) -> Option<Move> {
        let ctx = mate_ctx(self)?;
        let cands = Cands::for_king(ctx.us, ctx.sq_king);
        mate_dispatch(self, &ctx, &cands)
    }

    /// The retained pre-filter (twin) form of [`Position::mate_1ply`]: identical
    /// logic driven by all-ones candidate masks, so every move-mate loop
    /// iterates its full piece bucket. The candidate-filter equivalence gate
    /// asserts this returns the identical move on every probed position.
    #[cfg(test)]
    pub fn mate_1ply_unfiltered(&self) -> Option<Move> {
        let ctx = mate_ctx(self)?;
        mate_dispatch(self, &ctx, &Cands::unfiltered())
    }
}

/// Build the per-query [`Ctx`], or `None` when either king is missing.
fn mate_ctx(pos: &Position) -> Option<Ctx<'_>> {
    let board = pos.board();
    let us = pos.side_to_move();
    let them = us.flip();

    // Source both kings and both `blockersForKing` sets from the per-position
    // check-info cache — `do_move` already computed them (for `is_legal` /
    // `gives_check`), so recomputing them here (two `slider_blockers` passes and
    // two king scans on every non-check node) is pure duplication. The cache is
    // computed for the side to move, so `enemy_king` is Them's king, `own_king`
    // is Us's king, and `blockers` is colour-indexed — matching the identities
    // `try_find_king(board, them)`, `try_find_king(board, us)`, and
    // `blockers_for_king(board, c)` bit-for-bit. Copy the plain
    // `Square`/[`Bitboard`] values out of the `Ref` guard so it does not
    // outlive this borrow.
    let (sq_king, our_king, blockers_them, blockers_us) = {
        let ci = pos.check_info();
        (
            ci.enemy_king()?,
            ci.own_king()?,
            Bb(ci.blockers(them)),
            Bb(ci.blockers(us)),
        )
    };

    let occ_all = occupied(board);
    let occ_us = pieces_of_color(board, us);

    Some(Ctx {
        board,
        us,
        them,
        sq_king,
        our_king,
        occ_all,
        occ_us,
        bb_drop: occ_all.andnot(full_board()),
        bb_move: occ_us.andnot(full_board()),
        pinned: blockers_them.and(pieces_of_color(board, them)),
        dc_candidates: blockers_them.and(occ_us),
        our_pinned: blockers_us.and(occ_us),
    })
}

/// Drop mates then move mates. The drop section takes no candidate table (the
/// reference does not filter drops); `cands` gates only the move-mate loops.
fn mate_dispatch(pos: &Position, ctx: &Ctx, cands: &Cands) -> Option<Move> {
    // -- Drop mates (rook, lance, bishop, gold, silver, knight).
    if let Some(m) = drop_mates(ctx, pos.hand(ctx.us)) {
        return Some(m);
    }
    // -- Move mates (dragon, rook, horse, bishop, lance, then gold, silver,
    //    knight, pawn-push, pawn-promote).
    mate_moves(ctx, cands)
}

/// The full 81-square set (`~pos.pieces()` uses this as the universe).
fn full_board() -> Bb {
    Bb(Bitboard::FULL)
}

#[cfg(test)]
mod scan_oracle;
