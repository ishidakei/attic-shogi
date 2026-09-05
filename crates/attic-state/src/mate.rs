//! One-ply mate detector, ported from `Mate::mate_1ply`
//! (`mate/mate1ply_without_effect.cpp` — the table-driven detector the standard
//! build selects, not `mate1ply_with_effect.cpp`).
//!
//! **The detector is deliberately incomplete**: it does not find every legal
//! 1-ply mate. The reference compiles several blocks out with `#if 0` —
//! distant dropped-rook / dropped-bishop "spear" mates, double-check (両王手)
//! mates with their 24-neighbour discovered-attack search, and the illegal
//! 打ち歩詰め pawn-drop probe. Only the live path is ported, and parity means
//! returning a mate move or `None` in exactly the same positions: the misses
//! are preserved, not fixed.
//!
//! Non-slider move origins are pre-filtered through the `CHECK_CAND_BB` tables
//! ported in [`check_cand`]. Being a superset filter, the table only removes
//! origins the inner tests would reject anyway, so the surviving origins are
//! iterated in the same ascending square order and the first mate found — hence
//! the returned move — is unchanged. The slider kinds have no such table, their
//! reference cases being `#if 0`.
//!
//! The two pawn blocks lean on the table for two facts it happens to guarantee:
//! that the origin holds an unpromoted pawn, and that the promotion destination
//! lies in the enemy field. Both are reintroduced here as explicit guards.

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

/// A set of board squares, wrapping [`Bitboard`] to keep this module's
/// reference-shaped method names: `andnot` is the reference's `~self & other`,
/// while `sub` is the more natural `self & ~other`.
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

/// Forward rank delta for a side's own advance. Distinct from
/// `movegen::dr_sign_for`, which multiplies black-orientation step tables.
fn forward_dr(color: Color) -> i8 {
    match color {
        Color::Black => -1,
        Color::White => 1,
    }
}

// The effect primitives below are table lookups on the shared [`bitboard`]
// substrate. Scalar forms that step square-by-square live in [`scan_oracle`] as
// their oracles.

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

/// Every occupied square.
fn occupied(board: &Board) -> Bb {
    Bb(board.occupied())
}

/// The occupied squares of `color`.
fn pieces_of_color(board: &Board, color: Color) -> Bb {
    Bb(board.pieces_color(color))
}

/// Both kings, from the per-`(colour, pattern)` piece sets.
fn both_kings(board: &Board) -> Bb {
    Bb(board.pieces_pattern(Color::Black, pat::KING)
        | board.pieces_pattern(Color::White, pat::KING))
}

/// `pieces(color, pattern)` — the `color` pieces in a single [`pat`] attack
/// bucket. Each move-mate loop's `is_<kind>` predicate maps 1:1 to one slot.
fn pieces_bucket(board: &Board, color: Color, pattern: usize) -> Bb {
    Bb(board.pieces_pattern(color, pattern))
}

/// An un-promoted pawn — the one `is_<kind>` predicate consulted as a point
/// test rather than a set.
fn is_pawn_unpromoted(p: Piece) -> bool {
    p.kind == PieceKind::Pawn && !p.promoted
}

/// `color`'s pieces attacking `sq` under occupancy `occ`, counting only squares
/// occupied *in `occ`*. Clearing the mover's `from` therefore drops that piece
/// while still revealing any x-ray slider behind it — the reference's
/// `attackers_to<C>(sq, slide) ^ from`.
///
/// The result is intersected with `occ` because the reverse-attack lookup
/// sources attacker squares from the board's piece sets, which still hold the
/// mover's `from`.
fn attackers_of_color(board: &Board, sq: Square, occ: Bb, color: Color) -> Bb {
    let occ_bb = occ.0;
    Bb(attackers_bb_occ(board, sq, color, occ_bb) & occ_bb)
}

// --------------------------------------------------------------------------
//  Geometry: aligned / between
// --------------------------------------------------------------------------

/// `aligned(s1, s2, s3)` with `s3` the king: `s1` and `s2` lie in the **same**
/// direction from `s3`. A straight line *through* the king, with the two on
/// opposite sides, is not aligned.
fn aligned(s1: Square, s2: Square, s3: Square) -> bool {
    match (bitboard::ray_dir(s1, s3), bitboard::ray_dir(s2, s3)) {
        (Some(d1), Some(d2)) => d1 == d2,
        _ => false,
    }
}

/// Squares strictly between `a` and `b` when they share a queen line, else the
/// empty set (`between_bb`).
fn between_bb(a: Square, b: Square) -> Bb {
    Bb(bitboard::between(a, b))
}

// --------------------------------------------------------------------------
//  Pins / blockers
// --------------------------------------------------------------------------

/// Enemy sliders aimed at `ksq` along their step lines, optionally excluding
/// `avoid` — the sniper set of both `update_slider_blockers` and
/// `pinned_pieces(avoid)`. The `*_attacks(.., EMPTY)` queries give the full ray
/// to the edge.
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

/// `st->blockersForKing[c]`: pieces that are the sole occupant between `c`'s
/// king and an enemy sniper, with all snipers removed from the occupancy first.
#[cfg(test)]
fn blockers_for_king(board: &Board, c: Color) -> Bb {
    Bb(crate::see::slider_blockers(board, c).0)
}

/// `Position::pinned_pieces<C>(avoid)`: `C`'s own pieces pinned to `C`'s king,
/// computed with `avoid` removed from both the sniper set and the between-count.
/// Unlike [`blockers_for_king`] this does **not** strip other snipers from the
/// occupancy, matching the reference's distinct `new_pin` formula.
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
/// `from` distinguishes the two reference overloads. `Some(_)` is the move
/// variant, where the king is removed from the occupancy so that an x-ray behind
/// it is seen; `None` is the drop variant, where the reference deliberately
/// leaves the king in.
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
    let blocked = bb_avoid.or(Bb::of(to)).or(own);
    let escapes = king_effect(king_sq).sub(blocked);
    for esc in escapes.iter() {
        // The mover's `from` is already cleared from `occ` in the move variant,
        // matching the reference's `Bitboard(from).andnot(attackers_to(...))`.
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

    // A knight needs no support test: the king cannot reach its square.
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
    // The enemy third rank, where an un-promoted lance can skewer.
    let skewer_rank: u8 = if us == Color::Black { 2 } else { 6 };
    for from in pieces_bucket(board, us, pat::LANCE).and(cand).iter() {
        let slide = ctx.occ_all.without(from);
        let bb_check = lance_effect(us, from, slide)
            .and(ctx.bb_move)
            .and(gold_effect(them, ctx.sq_king));
        for to in bb_check.iter() {
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
                // A knight cannot both promote- and non-promote-check the same
                // square, so a failure here does not fall through below.
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
/// reference's candidate-table guard implied that `from` holds an unpromoted Us
/// pawn; that is checked explicitly here.
fn mate_pawn_no_promote(ctx: &Ctx, cand: Bb) -> Option<Move> {
    let board = ctx.board;
    let (us, them) = (ctx.us, ctx.them);
    let them_pieces = ctx.occ_them_pieces();
    if cand.and(pieces_bucket(board, us, pat::PAWN)).is_empty() {
        return None;
    }
    let push = (0i8, -forward_dr(us)); // `SQ_D` for Black, `SQ_U` for White.
    let to = step_signed(ctx.sq_king, push.0, push.1)?;
    if let Some(p) = board.get(to)
        && p.color == us
    {
        return None;
    }
    let from = step_signed(to, push.0, push.1)?;
    match board.get(from) {
        Some(p) if p.color == us && is_pawn_unpromoted(p) => {}
        _ => return None,
    }
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

/// Promoting pawn-push mate (`PIECE_TYPE_CHECK_PAWN_WITH_PRO`). The reference's
/// candidate table guaranteed the destination sits in the enemy field; that is
/// checked explicitly here.
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
        let bb_attacks = gold_effect(us, to);
        if !bb_attacks.contains(ctx.sq_king) {
            continue;
        }
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

/// The non-slider move-mate group, behind the reference's
/// `PIECE_TYPE_CHECK_NON_SLIDER` early-out: with no Us piece on any non-slider
/// candidate square, none of the blocks can fire.
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

/// The move-mate chain: unfiltered sliders, then the candidate-filtered lance
/// and non-slider group. `cands` supplies the per-kind masks, or all-ones in
/// the unfiltered twin.
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
    /// Includes the reference's deliberate misses; see the module head. Its
    /// precondition, like the reference's `ASSERT_LV3(!pos.checkers())`, is that
    /// the side to move is **not** in check.
    pub fn mate_1ply(&self) -> Option<Move> {
        let ctx = mate_ctx(self)?;
        let cands = Cands::for_king(ctx.us, ctx.sq_king);
        mate_dispatch(self, &ctx, &cands)
    }

    /// [`Position::mate_1ply`] driven by all-ones candidate masks, so that every
    /// move-mate loop iterates its full piece bucket. It must return the
    /// identical move on every position.
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

    // Both kings and both `blockersForKing` sets come from the per-position
    // check-info cache, which `do_move` already filled. Its frame is the side to
    // move, so `enemy_king` is Them's king and `own_king` is Us's. The values
    // are copied out of the `Ref` guard so it does not outlive this borrow.
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

/// Drop mates then move mates. The reference does not filter drops, so `cands`
/// gates only the move-mate loops.
fn mate_dispatch(pos: &Position, ctx: &Ctx, cands: &Cands) -> Option<Move> {
    if let Some(m) = drop_mates(ctx, pos.hand(ctx.us)) {
        return Some(m);
    }
    mate_moves(ctx, cands)
}

/// The full 81-square set (`~pos.pieces()` uses this as the universe).
fn full_board() -> Bb {
    Bb(Bitboard::FULL)
}

#[cfg(test)]
mod scan_oracle;
