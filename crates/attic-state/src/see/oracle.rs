//! `#[cfg(test)]` equivalence oracles for the bitboard SEE in [`super`].
//!
//! Two scalar oracles are kept, each derived independently of the bitboard
//! substrate:
//!
//! * [`Position::see_ge_reference`] — the full-board-rescan form: it re-derives
//!   `attackers_to` against the mutated occupancy every step.
//! * [`Position::see_ge_incremental`] — collect once + scalar x-ray maintenance
//!   (`attackers_to_bits` + `reveal_xray`).
//!
//! Both share the scalar attacker machinery below and the scalar
//! [`slider_blockers_scalar`] (an `update_slider_blockers` walk, the oracle for
//! the bitboard blocker sets). The bitboard `super::see_ge` must agree with both
//! on every move / threshold; `super::equivalence` enforces it over fixture
//! playouts.

use super::{
    BISHOP_VALUE, DRAGON_VALUE, GOLD_VALUE, HORSE_VALUE, KING_VALUE, KNIGHT_VALUE, LANCE_VALUE,
    PAWN_VALUE, ROOK_VALUE, SILVER_VALUE, piece_value,
};
use crate::bitboard::Bitboard;
use crate::color::Color;
use crate::move_::Move;
use crate::movegen::{dr_sign_for, movement, step_signed, try_find_king};
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;

/// Materialize a raw attacker/blocker `u128` mask (this oracle's internal
/// representation) as a [`Bitboard`], so comparisons against the production
/// `super::slider_blockers` go through the typed value.
fn bits_to_bb(mut bits: u128) -> Bitboard {
    let mut bb = Bitboard::empty();
    while bits != 0 {
        let idx = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        bb |= Bitboard::from_square(Square::from_index(idx).unwrap());
    }
    bb
}

/// The reference's `pieces(<TYPE>)` least-valuable-attacker buckets, including
/// the terminal KING (which the production [`super::Bucket`] omits because it is
/// handled by the loop's king branch rather than a `swap` update).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bucket {
    Pawn,
    Lance,
    Knight,
    Silver,
    Golds,
    Bishop,
    Rook,
    Horse,
    Dragon,
    King,
}

/// The non-king buckets in the reference's else-if try order.
const BUCKET_ORDER: [Bucket; 9] = [
    Bucket::Pawn,
    Bucket::Lance,
    Bucket::Knight,
    Bucket::Silver,
    Bucket::Golds,
    Bucket::Bishop,
    Bucket::Rook,
    Bucket::Horse,
    Bucket::Dragon,
];

fn bucket_of(p: Piece) -> Bucket {
    match (p.kind, p.promoted) {
        (PieceKind::Pawn, false) => Bucket::Pawn,
        (PieceKind::Lance, false) => Bucket::Lance,
        (PieceKind::Knight, false) => Bucket::Knight,
        (PieceKind::Silver, false) => Bucket::Silver,
        (PieceKind::Gold, _)
        | (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => {
            Bucket::Golds
        }
        (PieceKind::Bishop, false) => Bucket::Bishop,
        (PieceKind::Bishop, true) => Bucket::Horse,
        (PieceKind::Rook, false) => Bucket::Rook,
        (PieceKind::Rook, true) => Bucket::Dragon,
        (PieceKind::King, _) => Bucket::King,
    }
}

fn bucket_value(b: Bucket) -> i32 {
    match b {
        Bucket::Pawn => PAWN_VALUE,
        Bucket::Lance => LANCE_VALUE,
        Bucket::Knight => KNIGHT_VALUE,
        Bucket::Silver => SILVER_VALUE,
        Bucket::Golds => GOLD_VALUE,
        Bucket::Bishop => BISHOP_VALUE,
        Bucket::Rook => ROOK_VALUE,
        Bucket::Horse => HORSE_VALUE,
        Bucket::Dragon => DRAGON_VALUE,
        Bucket::King => KING_VALUE,
    }
}

/// An 81-bit occupancy set over board squares (`bit i` ⇔ square index `i`).
#[derive(Clone, Copy)]
struct Occupancy(u128);

impl Occupancy {
    fn contains(self, sq: Square) -> bool {
        (self.0 >> sq.index()) & 1 != 0
    }

    fn clear(&mut self, sq: Square) {
        self.0 &= !(1u128 << sq.index());
    }
}

/// Forward rank delta for a color's own pieces (Black advances toward rank 0,
/// White toward rank 8).
fn forward_dr(color: Color) -> i8 {
    match color {
        Color::Black => -1,
        Color::White => 1,
    }
}

/// Does the piece on `from` attack `to`, given `occupied` as the blocking set?
fn piece_attacks(piece: Piece, from: Square, to: Square, occupied: Occupancy) -> bool {
    let df = to.file() as i8 - from.file() as i8;
    let dr = to.rank() as i8 - from.rank() as i8;
    let sign = dr_sign_for(piece.color);
    let (steps, slides) = movement(piece);
    for &(sdf, sdr) in steps {
        if df == sdf && dr == sdr * sign {
            return true;
        }
    }
    for &(sdf, sdr) in slides {
        let (ddf, ddr) = (sdf, sdr * sign);
        let mut cur = from;
        loop {
            cur = match step_signed(cur, ddf, ddr) {
                Some(s) => s,
                None => break,
            };
            if cur == to {
                return true;
            }
            if occupied.contains(cur) {
                break;
            }
        }
    }
    false
}

/// Bitboard of every piece (both colors) that attacks `to` under `occupied`.
fn attackers_to_bits(board: &crate::board::Board, to: Square, occupied: Occupancy) -> u128 {
    let mut bits: u128 = 0;
    for index in 0..Square::COUNT as u8 {
        let from = Square::from_index(index).unwrap();
        if !occupied.contains(from) {
            continue;
        }
        if let Some(piece) = board.get(from)
            && piece_attacks(piece, from, to, occupied)
        {
            bits |= 1u128 << from.index();
        }
    }
    bits
}

/// Squares of every piece (both colors) that attacks `to` under `occupied`.
fn attackers_to(board: &crate::board::Board, to: Square, occupied: Occupancy) -> Vec<Square> {
    let mut out = Vec::new();
    for index in 0..Square::COUNT as u8 {
        let from = Square::from_index(index).unwrap();
        if !occupied.contains(from) {
            continue;
        }
        if let Some(piece) = board.get(from)
            && piece_attacks(piece, from, to, occupied)
        {
            out.push(from);
        }
    }
    out
}

/// Reveal the single x-ray attacker uncovered when the piece on `removed_sq` is
/// consumed (scalar walk along the opened ray).
fn reveal_xray(
    board: &crate::board::Board,
    to: Square,
    removed_sq: Square,
    occupied: Occupancy,
) -> u128 {
    let df = (removed_sq.file() as i8 - to.file() as i8).signum();
    let dr = (removed_sq.rank() as i8 - to.rank() as i8).signum();
    let mut cur = to;
    loop {
        cur = match step_signed(cur, df, dr) {
            Some(s) => s,
            None => return 0,
        };
        if occupied.contains(cur) {
            if let Some(piece) = board.get(cur)
                && piece_attacks(piece, cur, to, occupied)
            {
                return 1u128 << cur.index();
            }
            return 0;
        }
    }
}

/// Split an attacker bitboard into `(stm_attackers, opponent_attackers)`.
fn split_by_color(board: &crate::board::Board, attackers: u128, stm: Color) -> (u128, u128) {
    let mut stm_bits: u128 = 0;
    let mut opp_bits: u128 = 0;
    let mut bits = attackers;
    while bits != 0 {
        let idx = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let sq = Square::from_index(idx).unwrap();
        if let Some(piece) = board.get(sq) {
            if piece.color == stm {
                stm_bits |= 1u128 << idx;
            } else {
                opp_bits |= 1u128 << idx;
            }
        }
    }
    (stm_bits, opp_bits)
}

/// Is `piece` (an enemy slider) a sniper aimed at the king along ray `(df,dr)`?
fn is_sniper_type(piece: Piece, df: i8, dr: i8, king_color: Color) -> bool {
    let is_diag = df != 0 && dr != 0;
    if is_diag {
        piece.kind == PieceKind::Bishop
    } else if piece.kind == PieceKind::Rook {
        true
    } else {
        piece.kind == PieceKind::Lance && !piece.promoted && df == 0 && dr == forward_dr(king_color)
    }
}

/// A scalar `update_slider_blockers(c)`: returns
/// `(blockersForKing[c], pinners[~c])`. The equivalence oracle for the bitboard
/// [`super::slider_blockers`].
pub(crate) fn slider_blockers_scalar(board: &crate::board::Board, c: Color) -> (u128, u128) {
    let ksq = match try_find_king(board, c) {
        Some(s) => s,
        None => return (0, 0),
    };
    let enemy = c.flip();
    let mut blockers: u128 = 0;
    let mut pinners: u128 = 0;

    const RAY_DIRS: [(i8, i8); 8] = [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (1, -1),
        (-1, -1),
        (1, 1),
        (-1, 1),
    ];

    for &(df, dr) in &RAY_DIRS {
        let mut count = 0usize;
        let mut only_sq = ksq;
        let mut only_color = c;
        let mut cur = ksq;
        loop {
            cur = match step_signed(cur, df, dr) {
                Some(s) => s,
                None => break,
            };
            let Some(p) = board.get(cur) else { continue };
            if p.color == enemy && is_sniper_type(p, df, dr, c) {
                if count == 1 {
                    blockers |= 1u128 << only_sq.index();
                    if only_color == c {
                        pinners |= 1u128 << cur.index();
                    }
                }
            } else {
                count += 1;
                only_sq = cur;
                only_color = p.color;
            }
        }
    }

    (blockers, pinners)
}

/// [`slider_blockers_scalar`] with both halves lifted into [`Bitboard`], so it
/// can be compared directly against the production `super::slider_blockers`
/// (which returns `Bitboard`s).
pub(crate) fn slider_blockers_scalar_bb(
    board: &crate::board::Board,
    c: Color,
) -> (Bitboard, Bitboard) {
    let (blockers, pinners) = slider_blockers_scalar(board, c);
    (bits_to_bb(blockers), bits_to_bb(pinners))
}

impl Position {
    /// Full-board-rescan SEE oracle — re-derives `attackers_to` against the
    /// mutated occupancy every step.
    pub(crate) fn see_ge_reference(&self, m: Move, threshold: i32) -> bool {
        let board = self.board();
        let drop = m.is_drop();
        let to = m.to_sq();

        let victim_value = match board.get(to) {
            Some(p) => piece_value(p),
            None => 0,
        };
        let mut swap = victim_value - threshold;
        if swap < 0 {
            return false;
        }

        let mover = self.side_to_move();
        let from_value = if drop {
            piece_value(Piece::new(m.dropped_piece_kind(), mover))
        } else {
            piece_value(
                board
                    .get(m.from_sq())
                    .expect("see_ge_reference: board move has no piece on `from`"),
            )
        };
        swap = from_value - swap;
        if swap <= 0 {
            return true;
        }

        let mut occ_bits: u128 = 0;
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if board.get(sq).is_some() {
                occ_bits |= 1u128 << index;
            }
        }
        let mut occupied = Occupancy(occ_bits);
        occupied.clear(to);
        if !drop {
            occupied.clear(m.from_sq());
        }

        let (blockers_black, pinners_white) = slider_blockers_scalar(board, Color::Black);
        let (blockers_white, pinners_black) = slider_blockers_scalar(board, Color::White);
        let blockers_for_king = |c: Color| match c {
            Color::Black => blockers_black,
            Color::White => blockers_white,
        };
        let pinners_against = |stm: Color| match stm {
            Color::Black => pinners_white,
            Color::White => pinners_black,
        };

        let mut stm = mover;
        let mut res: i32 = 1;

        loop {
            stm = stm.flip();
            let attackers = attackers_to(board, to, occupied);
            let mut stm_attackers: Vec<Square> = attackers
                .iter()
                .copied()
                .filter(|&sq| board.get(sq).is_some_and(|p| p.color == stm))
                .collect();

            if stm_attackers.is_empty() {
                break;
            }

            if pinners_against(stm) & occupied.0 != 0 {
                let pinned = blockers_for_king(stm);
                stm_attackers.retain(|&sq| pinned & (1u128 << sq.index()) == 0);
                if stm_attackers.is_empty() {
                    break;
                }
            }

            res ^= 1;

            let mut chosen: Option<(Bucket, Square)> = None;
            'buckets: for &b in &BUCKET_ORDER {
                let mut best: Option<Square> = None;
                for &sq in &stm_attackers {
                    let piece = board.get(sq).expect("attacker square is occupied");
                    if bucket_of(piece) == b {
                        best = Some(match best {
                            Some(cur) if cur.index() <= sq.index() => cur,
                            _ => sq,
                        });
                    }
                }
                if let Some(sq) = best {
                    chosen = Some((b, sq));
                    break 'buckets;
                }
            }

            let (bucket, lva_sq) = match chosen {
                Some(c) => c,
                None => {
                    let opp_attacks = attackers
                        .iter()
                        .any(|&sq| board.get(sq).is_some_and(|p| p.color != stm));
                    let final_res = if opp_attacks { res ^ 1 } else { res };
                    return final_res != 0;
                }
            };

            swap = bucket_value(bucket) - swap;
            if swap < res {
                break;
            }

            occupied.clear(lva_sq);
        }

        res != 0
    }

    /// Incremental SEE oracle — collect-once + scalar x-ray reveal.
    pub(crate) fn see_ge_incremental(&self, m: Move, threshold: i32) -> bool {
        let board = self.board();
        let drop = m.is_drop();
        let to = m.to_sq();

        let victim_value = match board.get(to) {
            Some(p) => piece_value(p),
            None => 0,
        };
        let mut swap = victim_value - threshold;
        if swap < 0 {
            return false;
        }

        let mover = self.side_to_move();
        let from_value = if drop {
            piece_value(Piece::new(m.dropped_piece_kind(), mover))
        } else {
            piece_value(
                board
                    .get(m.from_sq())
                    .expect("see_ge_incremental: board move has no piece on `from`"),
            )
        };
        swap = from_value - swap;
        if swap <= 0 {
            return true;
        }

        let mut occ_bits: u128 = 0;
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if board.get(sq).is_some() {
                occ_bits |= 1u128 << index;
            }
        }
        let mut occupied = Occupancy(occ_bits);
        occupied.clear(to);
        if !drop {
            occupied.clear(m.from_sq());
        }

        let (blockers_black, pinners_white) = slider_blockers_scalar(board, Color::Black);
        let (blockers_white, pinners_black) = slider_blockers_scalar(board, Color::White);
        let blockers_for_king = |c: Color| match c {
            Color::Black => blockers_black,
            Color::White => blockers_white,
        };
        let pinners_against = |stm: Color| match stm {
            Color::Black => pinners_white,
            Color::White => pinners_black,
        };

        let mut attackers: u128 = attackers_to_bits(board, to, occupied);

        let mut stm = mover;
        let mut res: i32 = 1;

        loop {
            stm = stm.flip();
            attackers &= occupied.0;
            let (mut stm_attackers, opp_attackers) = split_by_color(board, attackers, stm);

            if stm_attackers == 0 {
                break;
            }

            if pinners_against(stm) & occupied.0 != 0 {
                stm_attackers &= !blockers_for_king(stm);
                if stm_attackers == 0 {
                    break;
                }
            }

            res ^= 1;

            let (bucket, lva_sq) = match least_valuable_attacker_scan(board, stm_attackers) {
                Some(c) => c,
                None => {
                    let final_res = if opp_attackers != 0 { res ^ 1 } else { res };
                    return final_res != 0;
                }
            };

            swap = bucket_value(bucket) - swap;
            if swap < res {
                break;
            }

            occupied.clear(lva_sq);
            if bucket != Bucket::Knight {
                attackers |= reveal_xray(board, to, lva_sq, occupied);
            }
        }

        res != 0
    }
}

/// Scalar least-valuable-attacker scan classifying each set square by
/// [`bucket_of`]. Returns `None` when only a KING
/// remains.
fn least_valuable_attacker_scan(
    board: &crate::board::Board,
    stm_attackers: u128,
) -> Option<(Bucket, Square)> {
    fn bucket_rank(b: Bucket) -> u8 {
        match b {
            Bucket::Pawn => 0,
            Bucket::Lance => 1,
            Bucket::Knight => 2,
            Bucket::Silver => 3,
            Bucket::Golds => 4,
            Bucket::Bishop => 5,
            Bucket::Rook => 6,
            Bucket::Horse => 7,
            Bucket::Dragon => 8,
            Bucket::King => 9,
        }
    }
    let mut best: Option<(u8, Bucket, Square)> = None;
    let mut bits = stm_attackers;
    while bits != 0 {
        let idx = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let sq = Square::from_index(idx).unwrap();
        let bucket = bucket_of(board.get(sq).expect("attacker square is occupied"));
        let rank = bucket_rank(bucket);
        if best.is_none_or(|(best_rank, _, _)| rank < best_rank) {
            best = Some((rank, bucket, sq));
        }
    }
    match best {
        Some((_, Bucket::King, _)) => None,
        Some((_, bucket, sq)) => Some((bucket, sq)),
        None => None,
    }
}
