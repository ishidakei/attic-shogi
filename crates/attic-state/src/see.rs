//! Static Exchange Evaluation "greater-or-equal" test (`see_ge`), ported from
//! upstream YaneuraOu's `Position::see_ge`
//! (`source/position.cpp`, the `#if !STOCKFISH` shogi path)
//! at the current submodule pin.
//!
//! `see_ge(m, threshold)` returns `true` iff the static exchange evaluation of
//! move `m` — the material swing of the optimal capture/recapture sequence on
//! `m`'s destination square — is greater than or equal to `threshold`. The
//! algorithm is a null-window swap over the destination square: it alternately
//! resolves the least-valuable attacker of each side, updating a running
//! `swap` bound, and early-exits the moment the threshold decision is settled.
//!
//! Faithful-port notes (all mirroring the reference exactly):
//!
//! * The moving piece's own promotion is **not** credited — `see_ge` uses the
//!   *unpromoted* value of the piece standing on `from` (`type_of(piece_on
//!   (from))`), and the reference comment states the final promotion gain is
//!   deliberately ignored so the early cutoff stays valid. A piece that is
//!   *already* promoted on the board contributes its promoted value, both as
//!   the victim on `to` (`PieceValue[piece_on(to)]`) and as an attacker (a
//!   Tokin/`+P` is a GOLDS-bucket attacker worth `GoldValue`, a Horse is worth
//!   `HorseValue`, a Dragon `DragonValue`).
//! * Drops restore the moved piece's value from the dropped kind (there is no
//!   piece on `from`); a drop onto an empty square has victim value `0`.
//! * The least-valuable-attacker else-if order is preserved verbatim: PAWN,
//!   LANCE, KNIGHT, SILVER, GOLDS, BISHOP, ROOK, HORSE, DRAGON, KING — note
//!   ROOK is tried before HORSE even though `HorseValue < RookValue`; this is
//!   the reference ordering and node-count parity depends on it.
//! * The attacker set of both sides is collected **once** on the bitboard
//!   substrate ([`crate::movegen::attackers_bb_occ`], the table-driven
//!   `attackers_to(to, occupied)`) and then maintained incrementally, mirroring
//!   the reference's `attackers |= rayEffect<…>(to, occupied) & …` bookkeeping:
//!   each iteration drops consumed attackers (`attackers &= occupied`) and, when
//!   a non-knight attacker is removed, reveals the x-ray sliders standing behind
//!   it on the opened ray by recomputing the occupancy-limited slider query
//!   ([`reveal_sliders`]). Knights reveal nothing (they jump). This is
//!   behaviourally identical to re-deriving `attackers_to` against the mutated
//!   occupancy every step, since removing a piece can only open the single ray
//!   that passes through its square.
//! * The pinned-piece guard (`pinners(~stm) & occupied` ⇒ drop
//!   `blockers_for_king(stm)` from the attacker set) is ported from
//!   `update_slider_blockers`, computed once over the pre-move full board.

use crate::bitboard::{Bitboard, between, bishop_attacks, lance_attacks, ray_dir, rook_attacks};
use crate::board::pat;
use crate::color::Color;
use crate::move_::Move;
use crate::movegen::{attackers_to_both, try_find_king};
use crate::piece::Piece;
use crate::position::Position;
use crate::square::Square;

// --- Apery material values -------------------------------------------------
//
// Ported verbatim from the reference's `Eval::` enum in
// `evaluate.h` (the `USE_PIECE_VALUE` block). These
// are the constants `Eval::PieceValue[]` (`eval/evaluate_bona_piece.cpp`) is
// built from and that `see_ge` consults; the four promoted minor pieces all
// collapse to `GOLD_VALUE` in that table.
const PAWN_VALUE: i32 = 90;
const LANCE_VALUE: i32 = 315;
const KNIGHT_VALUE: i32 = 405;
const SILVER_VALUE: i32 = 495;
const GOLD_VALUE: i32 = 540;
const BISHOP_VALUE: i32 = 855;
const ROOK_VALUE: i32 = 990;
const PRO_PAWN_VALUE: i32 = 540;
const PRO_LANCE_VALUE: i32 = 540;
const PRO_KNIGHT_VALUE: i32 = 540;
const PRO_SILVER_VALUE: i32 = 540;
const HORSE_VALUE: i32 = 945;
const DRAGON_VALUE: i32 = 1395;
const KING_VALUE: i32 = 15000;

/// `Eval::PieceValue[piece]` — the value of a concrete board piece, with
/// promoted pieces returning their promoted value. Used for the victim on `to`
/// and for restoring the moving piece's value from `from`.
///
/// Exposed crate-wide (and re-exported from the crate root) because move
/// ordering in the Search layer scores captures with the same
/// `Eval::PieceValue[]` table the reference `MovePicker` consults
/// (`captureHistory[...] + 7 * PieceValue[captured]`, evasion capture bias
/// `PieceValue[victim] + (1 << 28)`).
pub fn piece_value(p: Piece) -> i32 {
    use crate::piece::PieceKind;
    match (p.kind, p.promoted) {
        (PieceKind::Pawn, false) => PAWN_VALUE,
        (PieceKind::Pawn, true) => PRO_PAWN_VALUE,
        (PieceKind::Lance, false) => LANCE_VALUE,
        (PieceKind::Lance, true) => PRO_LANCE_VALUE,
        (PieceKind::Knight, false) => KNIGHT_VALUE,
        (PieceKind::Knight, true) => PRO_KNIGHT_VALUE,
        (PieceKind::Silver, false) => SILVER_VALUE,
        (PieceKind::Silver, true) => PRO_SILVER_VALUE,
        (PieceKind::Gold, _) => GOLD_VALUE,
        (PieceKind::Bishop, false) => BISHOP_VALUE,
        (PieceKind::Bishop, true) => HORSE_VALUE,
        (PieceKind::Rook, false) => ROOK_VALUE,
        (PieceKind::Rook, true) => DRAGON_VALUE,
        (PieceKind::King, _) => KING_VALUE,
    }
}

/// The reference's `pieces(<TYPE>)` least-valuable-attacker buckets, in the
/// exact else-if order of the `see_ge` loop. `Golds` covers plain gold and any
/// promoted `{Pawn, Lance, Knight, Silver}`; `Horse`/`Dragon` are the promoted
/// bishop / rook. Bucket ⇔ [`pat`] pattern-slot is a bijection over the nine
/// capturing buckets ([`BUCKET_ORDER`]).
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
}

/// The nine capturing buckets paired with their [`pat`] pattern slot, in the
/// reference's verbatim else-if try order (KING is the terminal `else`, handled
/// separately). `Golds` maps to [`pat::GOLD`] (plain gold + promoted minors).
const BUCKET_ORDER: [(Bucket, usize); 9] = [
    (Bucket::Pawn, pat::PAWN),
    (Bucket::Lance, pat::LANCE),
    (Bucket::Knight, pat::KNIGHT),
    (Bucket::Silver, pat::SILVER),
    (Bucket::Golds, pat::GOLD),
    (Bucket::Bishop, pat::BISHOP),
    (Bucket::Rook, pat::ROOK),
    (Bucket::Horse, pat::HORSE),
    (Bucket::Dragon, pat::DRAGON),
];

/// The `PawnValue` / `LanceValue` / … constant the loop subtracts for a
/// least-valuable attacker of the given bucket.
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
    }
}

/// Locate the least-valuable attacker within `stm_attackers`, honoring the
/// reference's verbatim bucket try-order and, within a bucket, the lowest square
/// index (`bb.pop()`). Returns `None` when only a KING remains — the caller has
/// already guaranteed `stm_attackers` is non-empty, so `None` means the terminal
/// king branch.
///
/// Bitboard form: each bucket is an intersection with the board's per-`(colour,
/// pattern)` piece set ([`Board::pieces_pattern`]), exactly the reference's
/// `stmAttackers & pieces(<TYPE>)`. `trailing_zeros` picks the lowest square
/// index within the winning bucket — the reference's `bb.pop()` tie-break.
#[inline]
fn least_valuable_attacker(
    board: &crate::board::Board,
    stm_attackers: Bitboard,
    stm: Color,
) -> Option<(Bucket, Square)> {
    for (bucket, pattern) in BUCKET_ORDER {
        let set = stm_attackers & board.pieces_pattern(stm, pattern);
        if let Some(sq) = set.squares().next() {
            return Some((bucket, sq));
        }
    }
    None
}

/// Reveal the x-ray sliders uncovered when the piece on `removed` (already
/// cleared from `occ`) is consumed. `removed` is always aligned with `to` on one
/// of the eight ray directions — the caller only invokes `reveal_sliders` for
/// non-knight buckets, whose attackers slide onto or step adjacent to `to`.
///
/// The reference switches on `directions_of(to, sq)` and recomputes a single
/// `rayEffect<DIR>(to, occupied)` intersected with the sliders that can travel
/// that ray. Here the equivalent occupancy-limited slider query is recomputed:
/// removing a piece can only open the one ray through its square, so a diagonal
/// removal can only reveal a bishop/horse (`bishop_attacks`), an orthogonal
/// removal a rook/dragon (`rook_attacks`), and a *vertical* removal additionally
/// a lance (`lance_attacks`) — matching the reference's `pieces(BISHOP_HORSE)` /
/// `pieces(ROOK_DRAGON)` / `pieces(…, LANCE)` masks per direction. OR-ing the
/// full slider query is idempotent for attackers already present and, since only
/// the removed piece's ray opened, adds exactly the newly revealed sliders.
#[inline]
fn reveal_sliders(
    board: &crate::board::Board,
    to: Square,
    removed: Square,
    occ: Bitboard,
) -> Bitboard {
    let (df, dr) = ray_dir(to, removed).expect("non-knight SEE attacker is aligned with `to`");
    if df != 0 && dr != 0 {
        // Diagonal: bishop or horse behind the removed piece.
        let bishop_horse = board.pieces_pattern(Color::Black, pat::BISHOP)
            | board.pieces_pattern(Color::White, pat::BISHOP)
            | board.pieces_pattern(Color::Black, pat::HORSE)
            | board.pieces_pattern(Color::White, pat::HORSE);
        bishop_attacks(to, occ) & bishop_horse
    } else {
        // Orthogonal: rook or dragon behind the removed piece …
        let rook_dragon = board.pieces_pattern(Color::Black, pat::ROOK)
            | board.pieces_pattern(Color::White, pat::ROOK)
            | board.pieces_pattern(Color::Black, pat::DRAGON)
            | board.pieces_pattern(Color::White, pat::DRAGON);
        let mut revealed = rook_attacks(to, occ) & rook_dragon;
        if df == 0 {
            // … and, on a vertical ray, a lance too. A Black lance attacks `to`
            // from below (reverse ray = White lance direction from `to`); a
            // White lance from above.
            revealed |= lance_attacks(Color::White, to, occ)
                & board.pieces_pattern(Color::Black, pat::LANCE);
            revealed |= lance_attacks(Color::Black, to, occ)
                & board.pieces_pattern(Color::White, pat::LANCE);
        }
        revealed
    }
}

/// Port of `Position::update_slider_blockers(c)`: returns
/// `(blockersForKing[c], pinners[~c])` as square sets. `blockersForKing[c]` is
/// the set of single blockers between `c`'s king and an enemy sniper;
/// `pinners[~c]` is the set of enemy snipers whose single blocker is one of
/// `c`'s own pieces. Both are computed over the full pre-move board.
///
/// Bitboard form (the reference's own shape): the sniper set is the enemy
/// rook/dragon on `c`'s king's orthogonal step-effect, plus enemy bishop/horse
/// on its diagonal step-effect, plus enemy lance on `c`'s forward lance-line;
/// all snipers are removed from the occupancy, and for each sniper the single
/// occupant between it and the king (via the [`between`](crate::bitboard::between)
/// table) is the blocker (a pinner when that occupant is `c`'s own piece).
///
/// Exposed crate-wide so the per-state check-info cache
/// ([`crate::search_movegen`]) can reuse the same `blockersForKing` computation
/// its `is_legal` / `gives_check` predicates need, rather than duplicating it.
pub(crate) fn slider_blockers(board: &crate::board::Board, c: Color) -> (Bitboard, Bitboard) {
    let ksq = match try_find_king(board, c) {
        Some(s) => s,
        None => return (Bitboard::EMPTY, Bitboard::EMPTY),
    };
    let enemy = c.flip();

    // Step-effect (occupancy-independent) lines from the king, and the enemy
    // sliders sitting on each. `*_attacks(.., EMPTY)` = the full ray to the edge
    // (the reference's `rookStepEffect` / `bishopStepEffect` / `lanceStepEffect`).
    let rook_line = rook_attacks(ksq, Bitboard::EMPTY);
    let bishop_line = bishop_attacks(ksq, Bitboard::EMPTY);
    let lance_line = lance_attacks(c, ksq, Bitboard::EMPTY);
    let rook_dragon =
        board.pieces_pattern(enemy, pat::ROOK) | board.pieces_pattern(enemy, pat::DRAGON);
    let bishop_horse =
        board.pieces_pattern(enemy, pat::BISHOP) | board.pieces_pattern(enemy, pat::HORSE);
    let lance = board.pieces_pattern(enemy, pat::LANCE);
    let snipers = (rook_line & rook_dragon) | (bishop_line & bishop_horse) | (lance_line & lance);

    // Occupancy with the snipers removed (`pieces() ^ snipers`), so a slider
    // standing in front of another sniper is not itself counted as a blocker.
    let occupancy = board.occupied() ^ snipers;
    let own = board.pieces_color(c);

    let mut blockers = Bitboard::EMPTY;
    let mut pinners = Bitboard::EMPTY;
    for sniper_sq in snipers.squares() {
        let b = between(ksq, sniper_sq) & occupancy;
        // Exactly one piece between the sniper and the king.
        if b.popcount() == 1 {
            blockers |= b;
            if !(b & own).is_empty() {
                pinners |= Bitboard::from_square(sniper_sq);
            }
        }
    }

    (blockers, pinners)
}

impl Position {
    /// Static Exchange Evaluation "greater-or-equal" test.
    ///
    /// Returns `true` iff the SEE value of move `m` is `>= threshold`. Ported
    /// faithfully from upstream YaneuraOu's `Position::see_ge`; see the module
    /// docs for the precise semantics that are (and are not) modelled.
    ///
    /// `m` is interpreted against the current side to move: for a board move
    /// the piece is read off `from`; for a drop the value is restored from the
    /// dropped kind. The move is not required to be a capture, but SEE is only
    /// meaningful for captures — a quiet move onto an empty square has victim
    /// value `0`.
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        let board = self.board();
        let drop = m.is_drop();
        let to = m.to_sq();

        // swap = PieceValue[piece_on(to)] - threshold. If the victim alone
        // cannot reach the threshold even before any recapture, fail fast.
        let victim_value = match board.get(to) {
            Some(p) => piece_value(p),
            None => 0,
        };
        let mut swap = victim_value - threshold;
        if swap < 0 {
            return false;
        }

        // swap = PieceValue[from_pt] - swap. `from_pt` is the *unpromoted-as-it-
        // -stands* piece: for a board move the piece currently on `from` (a
        // promotion move leaves that piece unpromoted until it reaches `to`),
        // for a drop the dropped kind. If giving the moving piece back still
        // clears the threshold, succeed immediately.
        let mover = self.side_to_move();
        let from_value = if drop {
            piece_value(Piece::new(m.dropped_piece_kind(), mover))
        } else {
            piece_value(
                board
                    .get(m.from_sq())
                    .expect("see_ge: board move has no piece on `from`"),
            )
        };
        swap = from_value - swap;
        if swap <= 0 {
            return true;
        }

        // occupied = pieces() ^ from ^ to. Clearing `from` reveals x-ray
        // attackers behind the moving piece; `to` is cleared as well (the
        // reference xors it — it never blocks attacks to itself, so this is
        // immaterial to the attacker computation but kept for fidelity).
        let mut occupied = board.occupied();
        occupied.clear(to);
        if !drop {
            occupied.clear(m.from_sq());
        }

        // Pin state, read from the per-state check-info cache — the reference
        // reads `st->blockersForKing` / `st->pinners`, filled once per do_move
        // in `set_check_info`, rather than recomputing inside see_ge. The cache
        // stores exactly what `slider_blockers` computes:
        // `blockers(c)` == `slider_blockers(c).0` and `pinners(~c)` ==
        // `slider_blockers(c).1`. Copy the plain [`Bitboard`] values out of the
        // `Ref` guard so it does not outlive this borrow.
        let (blockers_black, blockers_white, pinners_black, pinners_white) = {
            let ci = self.check_info();
            (
                ci.blockers(Color::Black),
                ci.blockers(Color::White),
                ci.pinners(Color::Black),
                ci.pinners(Color::White),
            )
        };
        let blockers_for_king = |c: Color| match c {
            Color::Black => blockers_black,
            Color::White => blockers_white,
        };
        // pinners(~stm): the ~stm snipers pinning stm's pieces to stm's king,
        // i.e. `pinners[~stm]`. `slider_blockers(c)` yields `pinners[~c]`, so
        // `pinners[~stm]` is the value returned alongside `blockersForKing[stm]`.
        let pinners_against = |stm: Color| match stm {
            Color::Black => pinners_white,
            Color::White => pinners_black,
        };

        // Attacker set of both sides, collected once from the piece sets under
        // the post-move occupancy (`attackers_to(to, occupied)`). From here it
        // is maintained incrementally: `attackers &= occupied` drops consumed
        // pieces at the top of each iteration and `reveal_sliders` adds the
        // sliders uncovered behind a consumed non-knight attacker.
        let mut attackers = attackers_to_both(board, to, occupied);

        let mut stm = mover;
        let mut res: i32 = 1;

        loop {
            stm = stm.flip();
            attackers &= occupied;
            let mut stm_attackers = attackers & board.pieces_color(stm);

            // If stm has no attacker left, it cannot continue the exchange and
            // loses the null-window decision.
            if stm_attackers.is_empty() {
                break;
            }

            // Don't allow pinned pieces to attack while a pinner is still on
            // the board.
            if !(pinners_against(stm) & occupied).is_empty() {
                stm_attackers &= !blockers_for_king(stm);
                if stm_attackers.is_empty() {
                    break;
                }
            }

            res ^= 1;

            let (bucket, lva_sq) = match least_valuable_attacker(board, stm_attackers, stm) {
                Some(c) => c,
                None => {
                    // Only a KING remains as an attacker. Capturing with the
                    // king loses it if the opponent still attacks `to`.
                    let opp_attackers = attackers & board.pieces_color(stm.flip());
                    let final_res = if !opp_attackers.is_empty() {
                        res ^ 1
                    } else {
                        res
                    };
                    return final_res != 0;
                }
            };

            swap = bucket_value(bucket) - swap;
            if swap < res {
                break;
            }

            // Remove the consumed attacker, then reveal the x-ray sliders (if
            // any) behind it. Knights jump, so they uncover nothing.
            occupied.clear(lva_sq);
            if bucket != Bucket::Knight {
                attackers |= reveal_sliders(board, to, lva_sq, occupied);
            }
        }

        res != 0
    }
}

#[cfg(test)]
mod oracle;

#[cfg(test)]
mod equivalence {
    use super::*;
    use crate::move_::Move;
    use crate::piece::PieceKind;
    use crate::sfen::parse_sfen;

    /// The six perft-fixture SFENs plus the SEE unit-test seed positions. The
    /// playout below drives one deterministic game from each and compares the
    /// optimized `see_ge` against both the `see_ge_reference` (full-rescan) and
    /// `see_ge_incremental` (scalar) oracles on *every* legal move
    /// (not only captures) across the full threshold sweep.
    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    /// The fixed threshold sweep the gate mandates.
    const THRESHOLDS: &[i32] = &[-2000, -990, -500, -90, -1, 0, 1, 90, 500, 990, 2000];

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

    /// The move's own "naive exchange value" — the victim on `to` restored,
    /// minus the mover's unpromoted value — added to the swept thresholds so the
    /// boundary at the move's break-even point is probed directly.
    fn naive_exchange_value(pos: &Position, m: Move) -> i32 {
        let board = pos.board();
        let victim = board.get(m.to_sq()).map_or(0, piece_value);
        let mover = if m.is_drop() {
            piece_value(Piece::new(m.dropped_piece_kind(), pos.side_to_move()))
        } else {
            board.get(m.from_sq()).map_or(0, piece_value)
        };
        victim - mover
    }

    fn check_move(pos: &Position, m: Move) {
        let mut thresholds: Vec<i32> = THRESHOLDS.to_vec();
        let naive = naive_exchange_value(pos, m);
        thresholds.push(naive);
        thresholds.push(naive + 1);
        thresholds.push(naive - 1);
        for &th in &thresholds {
            let got = pos.see_ge(m, th);
            assert_eq!(
                got,
                pos.see_ge_reference(m, th),
                "see_ge disagrees with full-rescan reference on move {m:?} at threshold {th}",
            );
            assert_eq!(
                got,
                pos.see_ge_incremental(m, th),
                "see_ge disagrees with pre-change incremental oracle on move {m:?} at threshold {th}",
            );
        }
    }

    /// `slider_blockers` equivalence: the bitboard form must equal the scalar
    /// walk for both colours at the given position.
    fn check_slider_blockers(pos: &Position) {
        let board = pos.board();
        for c in [Color::Black, Color::White] {
            assert_eq!(
                slider_blockers(board, c),
                oracle::slider_blockers_scalar_bb(board, c),
                "slider_blockers disagrees with scalar oracle for {c:?}",
            );
        }
    }

    /// Fused-vs-per-colour equivalence: the single-pass
    /// [`attackers_to_both`] must equal the per-colour OR of [`attackers_bb_occ`]
    /// on *every* square, under the full board occupancy AND under SEE-style
    /// reduced occupancies (each occupied square removed from `occ` in turn,
    /// modelling consumed attackers / x-ray reveals). Any mismatch is a hard stop.
    fn check_attackers_to_both(pos: &Position) {
        use crate::bitboard::Bitboard;
        use crate::movegen::attackers_bb_occ;

        let board = pos.board();
        let full = board.occupied();

        for idx in 0..Square::COUNT as u8 {
            let sq = Square::from_index(idx).unwrap();

            // Full occupancy, plus one variant per occupied square removed.
            let mut occs: Vec<Bitboard> = vec![full];
            for removed in full.squares() {
                occs.push(full & !Bitboard::from_square(removed));
            }

            for &occ in &occs {
                let fused = attackers_to_both(board, sq, occ);
                let per_color = attackers_bb_occ(board, sq, Color::Black, occ)
                    | attackers_bb_occ(board, sq, Color::White, occ);
                assert_eq!(
                    fused, per_color,
                    "attackers_to_both disagrees with per-colour OR at {sq:?} under occ {occ:?}",
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn optimized_see_ge_matches_reference_on_fixture_playouts() {
        const MIN_PLIES: usize = 60;

        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
            let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D ^ (fi as u64).wrapping_add(1));
            let mut legal: Vec<Move> = Vec::new();

            let mut plies = 0usize;
            while plies < MIN_PLIES {
                check_slider_blockers(&pos);
                check_attackers_to_both(&pos);

                legal.clear();
                pos.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    break;
                }

                for &m in &legal {
                    check_move(&pos, m);
                }

                let m = legal[rng.pick(legal.len())];
                pos.do_move(m);
                plies += 1;
            }
        }
    }

    /// A promoted pawn (Tokin) attacks `to` as a GOLDS bucket, and a Horse /
    /// Dragon standing already-promoted contributes their promoted values — a
    /// direct pin of the bucket/value mapping the loop depends on.
    #[test]
    fn promoted_piece_values() {
        assert_eq!(piece_value(Piece::new(PieceKind::Pawn, Color::Black)), 90);
        assert_eq!(
            piece_value(Piece::promoted(PieceKind::Pawn, Color::Black).unwrap()),
            540
        );
        assert_eq!(
            piece_value(Piece::promoted(PieceKind::Bishop, Color::Black).unwrap()),
            945
        );
        assert_eq!(
            piece_value(Piece::promoted(PieceKind::Rook, Color::Black).unwrap()),
            1395
        );
    }
}
