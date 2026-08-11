//! Compile-time `CHECK_CAND_BB` candidate-origin tables, ported verbatim from
//! upstream YaneuraOu's `init_check_bb` (`mate1ply_without_effect.cpp`).
//!
//! For each enemy-king square, mover colour, and piece-kind, the entry is the
//! **superset** of origin squares from which a mover piece of that kind could
//! possibly deliver the relevant (possibly-promoting) check. The move-mate loops
//! in [`super`] intersect the iterated origin set with the matching entry before
//! their inner verification tests — most board pieces are excluded before any
//! per-piece work, and because the table is a superset the *first* mate found
//! (hence the returned move) is unchanged.
//!
//! Only the kinds the live code path consumes are built (the reference's `#if 0`
//! slider blocks — `PIECE_TYPE_CHECK_BISHOP` / `ROOK` / `PRO_*` — are left
//! unported, exactly as the reference leaves their table entries empty). The
//! `PIECE_TYPE_CHECK_NON_SLIDER` entry is the union of the gold / knight / silver
//! / pawn kinds, used for the early-out that skips the whole non-slider group
//! when no such piece stands on a candidate square.
//!
//! The tables are `const`-built the same way as the `RAY` / `BETWEEN` tables in
//! [`crate::bitboard`]; a `#[cfg(test)]` runtime reference (mirroring
//! `init_check_bb` through the already-trusted runtime effect helpers) pins them
//! entry-for-entry in [`super::scan_oracle`].

use super::Bb;
use crate::bitboard::Bitboard;
use crate::color::Color;
use crate::square::Square;

// PieceTypeCheck indices — the live subset of the reference enum.
pub(super) const PAWN_NO_PRO: usize = 0;
pub(super) const PAWN_PRO: usize = 1;
pub(super) const LANCE: usize = 2;
pub(super) const KNIGHT: usize = 3;
pub(super) const SILVER: usize = 4;
pub(super) const GOLD: usize = 5;
pub(super) const NON_SLIDER: usize = 6;
pub(super) const KIND_NB: usize = 7;

const N: usize = 81;

// --------------------------------------------------------------------------
//  const geometry / effect helpers ([`Bitboard`] masks over Square indices)
// --------------------------------------------------------------------------

const fn c_file(idx: usize) -> i32 {
    (idx / 9) as i32
}
const fn c_rank(idx: usize) -> i32 {
    (idx % 9) as i32
}
const fn c_on(f: i32, r: i32) -> bool {
    f >= 0 && f < 9 && r >= 0 && r < 9
}
const fn c_bit_fr(f: i32, r: i32) -> Bitboard {
    Bitboard::single((f as usize) * 9 + (r as usize))
}
const fn c_bit(idx: usize) -> Bitboard {
    Bitboard::single(idx)
}
/// Rank-axis sign for a colour (`dr_sign_for`): Black `+1`, White `-1`.
const fn c_sign(color: usize) -> i32 {
    if color == 0 { 1 } else { -1 }
}
/// `enemy_field(mc) & idx`: is `idx` in the mover's promotion zone.
const fn c_is_promo(idx: usize, mc: usize) -> bool {
    let r = c_rank(idx);
    if mc == 0 { r < 3 } else { r >= 6 }
}

/// Step-effect mask: each `(df, dr)` applied as `(df, dr * sign)`, mirroring
/// [`crate::movegen::step_signed`].
const fn c_step(idx: usize, deltas: &[(i32, i32)], sign: i32) -> Bitboard {
    let f0 = c_file(idx);
    let r0 = c_rank(idx);
    let mut m = Bitboard::EMPTY;
    let mut i = 0;
    while i < deltas.len() {
        let (df, dr) = deltas[i];
        let f = f0 + df;
        let r = r0 + dr * sign;
        if c_on(f, r) {
            m = m.or(c_bit_fr(f, r));
        }
        i += 1;
    }
    m
}

/// Full ray to the board edge along `(df, dr)`, excluding `idx` (a `...StepEffect`).
const fn c_ray(idx: usize, df: i32, dr: i32) -> Bitboard {
    let mut f = c_file(idx);
    let mut r = c_rank(idx);
    let mut m = Bitboard::EMPTY;
    loop {
        f += df;
        r += dr;
        if !c_on(f, r) {
            break;
        }
        m = m.or(c_bit_fr(f, r));
    }
    m
}

// Black-orientation step tables (rank axis multiplied by `c_sign`), matching
// the runtime `*_D` tables in [`super`].
const PAWN_D: [(i32, i32); 1] = [(0, -1)];
const KNIGHT_D: [(i32, i32); 2] = [(-1, -2), (1, -2)];
const SILVER_D: [(i32, i32); 5] = [(0, -1), (-1, -1), (1, -1), (-1, 1), (1, 1)];
const GOLD_D: [(i32, i32); 6] = [(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1)];
const CROSS45_D: [(i32, i32); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];

const fn c_pawn(idx: usize, color: usize) -> Bitboard {
    c_step(idx, &PAWN_D, c_sign(color))
}
const fn c_knight(idx: usize, color: usize) -> Bitboard {
    c_step(idx, &KNIGHT_D, c_sign(color))
}
const fn c_silver(idx: usize, color: usize) -> Bitboard {
    c_step(idx, &SILVER_D, c_sign(color))
}
const fn c_gold(idx: usize, color: usize) -> Bitboard {
    c_step(idx, &GOLD_D, c_sign(color))
}
const fn c_cross45(idx: usize) -> Bitboard {
    c_step(idx, &CROSS45_D, 1)
}
const fn c_lance_ray(idx: usize, color: usize) -> Bitboard {
    // lanceStepEffect(color, idx): ray in `forward_dr(color)` (Black up = -1).
    let dr = if color == 0 { -1 } else { 1 };
    c_ray(idx, 0, dr)
}

// --------------------------------------------------------------------------
//  per-kind candidate builders (verbatim `init_check_bb` cases)
// --------------------------------------------------------------------------

// In every builder `ksq` = enemy-king square, `mc` = mover colour, and
// `kc = ~mc` = king colour; the reference's effects use `~c`.

const fn b_pawn_no_pro(ksq: usize, mc: usize) -> Bitboard {
    let kc = 1 - mc;
    // to = pawnEffect(~c, sq) & ~enemy_field(c): the unique square in front of
    // the king, unless that push would land in the promotion zone.
    let pe = c_pawn(ksq, kc);
    if pe.is_empty() {
        return Bitboard::EMPTY;
    }
    let to = pe.lowest_index() as usize;
    if c_is_promo(to, mc) {
        return Bitboard::EMPTY;
    }
    // bb = pawnEffect(~c, to): the pushing pawn's origin.
    c_pawn(to, kc).without_index(ksq)
}

const fn b_pawn_pro(ksq: usize, mc: usize) -> Bitboard {
    let kc = 1 - mc;
    // bb = pawnBbEffect(~c, goldEffect(~c, sq) & enemy_field(c)).
    let gold = c_gold(ksq, kc);
    let mut out = Bitboard::EMPTY;
    let mut to = 0;
    while to < N {
        if gold.contains_index(to) && c_is_promo(to, mc) {
            out = out.or(c_pawn(to, kc));
        }
        to += 1;
    }
    out.without_index(ksq)
}

const fn b_lance(ksq: usize, mc: usize) -> Bitboard {
    let kc = 1 - mc;
    let mut bb = c_lance_ray(ksq, kc);
    if c_is_promo(ksq, mc) {
        // In the enemy field, promoting lances one file either side also check.
        let f = c_file(ksq);
        let r = c_rank(ksq);
        if f != 0 {
            bb = bb.or(c_lance_ray(((f - 1) as usize) * 9 + r as usize, kc));
        }
        if f != 8 {
            bb = bb.or(c_lance_ray(((f + 1) as usize) * 9 + r as usize, kc));
        }
    }
    bb.without_index(ksq)
}

const fn b_knight(ksq: usize, mc: usize) -> Bitboard {
    let kc = 1 - mc;
    let mut bb = Bitboard::EMPTY;
    // Knight-of-knight (non-promoting checks).
    let n = c_knight(ksq, kc);
    let mut to = 0;
    while to < N {
        if n.contains_index(to) {
            bb = bb.or(c_knight(to, kc));
        }
        to += 1;
    }
    // Promote-to-gold checks: goldEffect(~c, sq) & enemy_field(c), then reverse
    // by the knight hop.
    let g = c_gold(ksq, kc);
    let mut to2 = 0;
    while to2 < N {
        if g.contains_index(to2) && c_is_promo(to2, mc) {
            bb = bb.or(c_knight(to2, kc));
        }
        to2 += 1;
    }
    bb.without_index(ksq)
}

const fn b_gold(ksq: usize, mc: usize) -> Bitboard {
    let kc = 1 - mc;
    let g = c_gold(ksq, kc);
    let mut bb = Bitboard::EMPTY;
    let mut to = 0;
    while to < N {
        if g.contains_index(to) {
            bb = bb.or(c_gold(to, kc));
        }
        to += 1;
    }
    bb.without_index(ksq)
}

const fn b_silver(ksq: usize, mc: usize) -> Bitboard {
    let kc = 1 - mc;
    let mut bb = Bitboard::EMPTY;
    // Silver-of-silver (non-promoting checks).
    let s = c_silver(ksq, kc);
    let mut to = 0;
    while to < N {
        if s.contains_index(to) {
            bb = bb.or(c_silver(to, kc));
        }
        to += 1;
    }
    // Promote-to-gold: goldEffect(~c, sq) & enemy_field(c), reversed by a silver
    // step.
    let g = c_gold(ksq, kc);
    let mut to2 = 0;
    while to2 < N {
        if g.contains_index(to2) && c_is_promo(to2, mc) {
            bb = bb.or(c_silver(to2, kc));
        }
        to2 += 1;
    }
    // A 4th-rank king reached by a 3rd-rank silver promoting: the square below
    // the king, its 45-degree neighbours, and the two-file-distant squares.
    let kf = c_file(ksq);
    let kr = c_rank(ksq);
    let r4 = if mc == 0 { 3 } else { 5 };
    if kr == r4 {
        let r3 = if mc == 0 { 2 } else { 6 };
        let toi = (kf as usize) * 9 + (r3 as usize);
        bb = bb.or(c_bit(toi));
        bb = bb.or(c_cross45(toi));
        if kf >= 2 {
            bb = bb.or(c_bit_fr(kf - 2, r3));
        }
        if kf <= 6 {
            bb = bb.or(c_bit_fr(kf + 2, r3));
        }
    }
    // A 5th-rank king: promoting "back-attack" from a knight square.
    if kr == 4 {
        bb = bb.or(c_knight(ksq, mc));
    }
    bb.without_index(ksq)
}

const fn build() -> [[[Bitboard; 2]; KIND_NB]; N] {
    let mut t = [[[Bitboard::EMPTY; 2]; KIND_NB]; N];
    let mut sq = 0;
    while sq < N {
        let mut mc = 0;
        while mc < 2 {
            let p = b_pawn_no_pro(sq, mc);
            let pp = b_pawn_pro(sq, mc);
            let l = b_lance(sq, mc);
            let k = b_knight(sq, mc);
            let s = b_silver(sq, mc);
            let g = b_gold(sq, mc);
            t[sq][PAWN_NO_PRO][mc] = p;
            t[sq][PAWN_PRO][mc] = pp;
            t[sq][LANCE][mc] = l;
            t[sq][KNIGHT][mc] = k;
            t[sq][SILVER][mc] = s;
            t[sq][GOLD][mc] = g;
            // NON_SLIDER = union of the non-slider kinds.
            t[sq][NON_SLIDER][mc] = g.or(k).or(s).or(p).or(pp).without_index(sq);
            mc += 1;
        }
        sq += 1;
    }
    t
}

/// `CHECK_CAND_BB[sq_king][kind][color]`. A `static` (not `const`) so the
/// ~9 KB table lives once in `.rodata`.
static TABLE: [[[Bitboard; 2]; KIND_NB]; N] = build();

/// `check_cand_bb(us, kind, sq_king)` — the candidate-origin set.
pub(super) fn entry(sq_king: Square, kind: usize, us: Color) -> Bb {
    Bb(TABLE[sq_king.index() as usize][kind][us.index()])
}

/// The per-`(king, mover)` candidate masks threaded through the move-mate loops.
#[derive(Clone, Copy)]
pub(super) struct Cands {
    pub(super) lance: Bb,
    pub(super) gold: Bb,
    pub(super) silver: Bb,
    pub(super) knight: Bb,
    pub(super) pawn_no_pro: Bb,
    pub(super) pawn_pro: Bb,
    pub(super) non_slider: Bb,
}

impl Cands {
    pub(super) fn for_king(us: Color, sq_king: Square) -> Cands {
        Cands {
            lance: entry(sq_king, LANCE, us),
            gold: entry(sq_king, GOLD, us),
            silver: entry(sq_king, SILVER, us),
            knight: entry(sq_king, KNIGHT, us),
            pawn_no_pro: entry(sq_king, PAWN_NO_PRO, us),
            pawn_pro: entry(sq_king, PAWN_PRO, us),
            non_slider: entry(sq_king, NON_SLIDER, us),
        }
    }

    /// All-ones masks: the filter degenerates to the identity, reproducing the
    /// pre-filter (twin) behaviour byte-for-byte. The oracle in the mate gate
    /// compares [`Cands::for_king`] against this.
    #[cfg(test)]
    pub(super) fn unfiltered() -> Cands {
        let full = super::full_board();
        Cands {
            lance: full,
            gold: full,
            silver: full,
            knight: full,
            pawn_no_pro: full,
            pawn_pro: full,
            non_slider: full,
        }
    }
}

/// Fidelity gate: the `const`-built [`TABLE`] must equal a runtime reference
/// that mirrors `init_check_bb` through the already-trusted runtime effect
/// helpers in [`super`], entry-for-entry over all 81 king squares × 2 colours ×
/// every ported kind. Independent of position sampling, so it pins the tables
/// even where the mate-search playouts never reach.
#[cfg(test)]
mod tests {
    use super::super::{
        cross45_step_effect, gold_effect, knight_effect, lance_step_effect, pawn_effect,
        silver_effect,
    };
    use super::*;
    use crate::movegen::is_in_promotion_zone;

    fn sq(i: usize) -> Square {
        Square::from_index(i as u8).unwrap()
    }

    /// `enemy_field(mc)` — the mover's promotion zone as a set.
    fn ef(mc: Color) -> Bb {
        let mut bb = Bb::EMPTY;
        for i in 0..N {
            let s = sq(i);
            if is_in_promotion_zone(s, mc) {
                bb = bb.with(s);
            }
        }
        bb
    }

    fn ref_pawn_no_pro(ksq: Square, mc: Color) -> Bb {
        let kc = mc.flip();
        let masked = pawn_effect(kc, ksq).sub(ef(mc));
        let Some(to) = masked.iter().next() else {
            return Bb::EMPTY;
        };
        pawn_effect(kc, to).without(ksq)
    }

    fn ref_pawn_pro(ksq: Square, mc: Color) -> Bb {
        let kc = mc.flip();
        let mut out = Bb::EMPTY;
        for to in gold_effect(kc, ksq).and(ef(mc)).iter() {
            for from in pawn_effect(kc, to).iter() {
                out = out.with(from);
            }
        }
        out.without(ksq)
    }

    fn ref_lance(ksq: Square, mc: Color) -> Bb {
        let kc = mc.flip();
        let mut bb = lance_step_effect(kc, ksq);
        if is_in_promotion_zone(ksq, mc) {
            let f = ksq.file();
            if f != 0 {
                bb = bb.or(lance_step_effect(
                    kc,
                    Square::new(f - 1, ksq.rank()).unwrap(),
                ));
            }
            if f != 8 {
                bb = bb.or(lance_step_effect(
                    kc,
                    Square::new(f + 1, ksq.rank()).unwrap(),
                ));
            }
        }
        bb.without(ksq)
    }

    fn ref_knight(ksq: Square, mc: Color) -> Bb {
        let kc = mc.flip();
        let mut bb = Bb::EMPTY;
        for to in knight_effect(kc, ksq).iter() {
            bb = bb.or(knight_effect(kc, to));
        }
        for to in gold_effect(kc, ksq).and(ef(mc)).iter() {
            bb = bb.or(knight_effect(kc, to));
        }
        bb.without(ksq)
    }

    fn ref_gold(ksq: Square, mc: Color) -> Bb {
        let kc = mc.flip();
        let mut bb = Bb::EMPTY;
        for to in gold_effect(kc, ksq).iter() {
            bb = bb.or(gold_effect(kc, to));
        }
        bb.without(ksq)
    }

    fn ref_silver(ksq: Square, mc: Color) -> Bb {
        let kc = mc.flip();
        let mut bb = Bb::EMPTY;
        for to in silver_effect(kc, ksq).iter() {
            bb = bb.or(silver_effect(kc, to));
        }
        for to in gold_effect(kc, ksq).and(ef(mc)).iter() {
            bb = bb.or(silver_effect(kc, to));
        }
        let r4 = if mc == Color::Black { 3u8 } else { 5 };
        if ksq.rank() == r4 {
            let r3 = if mc == Color::Black { 2u8 } else { 6 };
            let to = Square::new(ksq.file(), r3).unwrap();
            bb = bb.with(to);
            bb = bb.or(cross45_step_effect(to));
            let f = to.file();
            if f >= 2 {
                bb = bb.with(Square::new(f - 2, r3).unwrap());
            }
            if f <= 6 {
                bb = bb.with(Square::new(f + 2, r3).unwrap());
            }
        }
        if ksq.rank() == 4 {
            bb = bb.or(knight_effect(mc, ksq));
        }
        bb.without(ksq)
    }

    #[test]
    fn const_tables_match_runtime_reference() {
        for i in 0..N {
            let ksq = sq(i);
            for mc in [Color::Black, Color::White] {
                let p = ref_pawn_no_pro(ksq, mc);
                let pp = ref_pawn_pro(ksq, mc);
                let l = ref_lance(ksq, mc);
                let k = ref_knight(ksq, mc);
                let s = ref_silver(ksq, mc);
                let g = ref_gold(ksq, mc);
                let ns = g.or(k).or(s).or(p).or(pp).without(ksq);

                let cases = [
                    (PAWN_NO_PRO, p, "pawn_no_pro"),
                    (PAWN_PRO, pp, "pawn_pro"),
                    (LANCE, l, "lance"),
                    (KNIGHT, k, "knight"),
                    (SILVER, s, "silver"),
                    (GOLD, g, "gold"),
                    (NON_SLIDER, ns, "non_slider"),
                ];
                for (kind, want, name) in cases {
                    assert_eq!(
                        entry(ksq, kind, mc),
                        want,
                        "CHECK_CAND_BB[{ksq:?}][{name}][{mc:?}] mismatch",
                    );
                }
            }
        }
    }
}
