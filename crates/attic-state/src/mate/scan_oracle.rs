//! `#[cfg(test)]` scan oracles for the bitboard attacker plumbing in [`super`],
//! plus the squarewise-equivalence test over them.
//!
//! The production mate detector reads the board **only** through a handful of
//! helpers backed by the bitboard substrate
//! ([`crate::movegen::attackers_bb_occ`], the slider queries, and
//! [`crate::see::slider_blockers`]) — the mate-search logic itself touches no
//! squares directly, so those helpers are the only surface on which the detector
//! could drift from a square-by-square derivation. This module holds
//! 81-square-scan forms of them, and the test below asserts, over fixture
//! playouts, that the production helpers agree with the scan oracles squarewise
//! on every argument the search can pass — attacker sets, sniper sets,
//! pinned/blocker sets, and the piece-set snapshots. Total helper equivalence
//! over the reachable argument space ⇒ an identical `mate_1ply` result.

use super::{Bb, between_bb, is_pawn_unpromoted};
use crate::board::Board;
use crate::color::Color;
use crate::movegen::{dr_sign_for, movement, step_signed, try_find_king};
use crate::piece::{Piece, PieceKind};
use crate::square::Square;

/// A raw board-index occupancy (`u128`, this oracle's sampling representation)
/// materialized as a [`Bb`] for the production helpers under test.
fn occ_bb(bits: u128) -> Bb {
    let mut bb = Bb::EMPTY;
    let mut b = bits;
    while b != 0 {
        let i = b.trailing_zeros() as u8;
        b &= b - 1;
        bb = bb.with(Square::from_index(i).unwrap());
    }
    bb
}

// -- board-wide snapshots (81-square scans) ---------------------------------

fn occupied_scan(board: &Board) -> Bb {
    let mut bb = Bb::EMPTY;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).unwrap();
        if board.get(sq).is_some() {
            bb = bb.with(sq);
        }
    }
    bb
}

fn pieces_of_color_scan(board: &Board, color: Color) -> Bb {
    let mut bb = Bb::EMPTY;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).unwrap();
        if let Some(p) = board.get(sq)
            && p.color == color
        {
            bb = bb.with(sq);
        }
    }
    bb
}

fn both_kings_scan(board: &Board) -> Bb {
    let mut bb = Bb::EMPTY;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).unwrap();
        if let Some(p) = board.get(sq)
            && p.kind == PieceKind::King
        {
            bb = bb.with(sq);
        }
    }
    bb
}

/// Squares of `color`'s pieces that satisfy `pred`, as a set.
fn color_pieces_where(board: &Board, color: Color, pred: impl Fn(Piece) -> bool) -> Bb {
    let mut bb = Bb::EMPTY;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).unwrap();
        if let Some(p) = board.get(sq)
            && p.color == color
            && pred(p)
        {
            bb = bb.with(sq);
        }
    }
    bb
}

// The reference `pieces(Us, PT)` buckets used by the move loops.
fn is_dragon(p: Piece) -> bool {
    p.kind == PieceKind::Rook && p.promoted
}
fn is_rook_unpromoted(p: Piece) -> bool {
    p.kind == PieceKind::Rook && !p.promoted
}
fn is_horse(p: Piece) -> bool {
    p.kind == PieceKind::Bishop && p.promoted
}
fn is_bishop_unpromoted(p: Piece) -> bool {
    p.kind == PieceKind::Bishop && !p.promoted
}
fn is_lance_unpromoted(p: Piece) -> bool {
    p.kind == PieceKind::Lance && !p.promoted
}
fn is_silver_unpromoted(p: Piece) -> bool {
    p.kind == PieceKind::Silver && !p.promoted
}
fn is_knight_unpromoted(p: Piece) -> bool {
    p.kind == PieceKind::Knight && !p.promoted
}
/// `pieces(Us, GOLDS)` — plain gold and every promoted {pawn, lance, knight,
/// silver}.
fn is_golds(p: Piece) -> bool {
    if p.kind == PieceKind::Gold {
        return true;
    }
    p.promoted
        && matches!(
            p.kind,
            PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver
        )
}

// -- attacker / sniper / pin scans ------------------------------------------

/// Squares attacked by `piece` sitting on `from`, under occupancy `occ`.
fn attacks_from(piece: Piece, from: Square, occ: Bb) -> Bb {
    let sign = dr_sign_for(piece.color);
    let (steps, slides) = movement(piece);
    let mut bb = Bb::EMPTY;
    for &(df, dr) in steps {
        if let Some(to) = step_signed(from, df, dr * sign) {
            bb = bb.with(to);
        }
    }
    for &(df, dr) in slides {
        let (ddf, ddr) = (df, dr * sign);
        let mut cur = from;
        while let Some(next) = step_signed(cur, ddf, ddr) {
            bb = bb.with(next);
            if occ.contains(next) {
                break;
            }
            cur = next;
        }
    }
    bb
}

/// `color`'s pieces (of any kind) attacking `sq` under occupancy `occ`, only
/// considering squares occupied in `occ`.
fn attackers_of_color_scan(board: &Board, sq: Square, occ: Bb, color: Color) -> Bb {
    let mut bb = Bb::EMPTY;
    for from in occ.iter() {
        if let Some(p) = board.get(from)
            && p.color == color
            && attacks_from(p, from, occ).contains(sq)
        {
            bb = bb.with(from);
        }
    }
    bb
}

fn forward_dr(color: Color) -> i8 {
    match color {
        Color::Black => -1,
        Color::White => 1,
    }
}

const ORTH_DIRS: &[(i8, i8)] = &[(0, -1), (0, 1), (-1, 0), (1, 0)];
const DIAG_DIRS: &[(i8, i8)] = &[(-1, -1), (1, -1), (-1, 1), (1, 1)];

fn ray_effect(sq: Square, dirs: &[(i8, i8)]) -> Bb {
    let mut bb = Bb::EMPTY;
    for &(df, dr) in dirs {
        let mut cur = sq;
        while let Some(next) = step_signed(cur, df, dr) {
            bb = bb.with(next);
            cur = next;
        }
    }
    bb
}

fn rook_step_effect(sq: Square) -> Bb {
    ray_effect(sq, ORTH_DIRS)
}

fn bishop_step_effect(sq: Square) -> Bb {
    ray_effect(sq, DIAG_DIRS)
}

fn lance_step_effect(c: Color, sq: Square) -> Bb {
    ray_effect(sq, &[(0, forward_dr(c))])
}

// -- scalar effect oracles (square-stepping forms) --------------------------
//
// Scalar counterparts of the table-lookup / Qugiy-slider effect functions in
// [`super`], sharing none of their data. The
// `effect_primitives_match_scalar_oracle` gate below asserts each production
// primitive equals its scalar oracle bit-for-bit.

// Black-orientation step tables (rank axis multiplied by `dr_sign_for`).
const PAWN_D: &[(i8, i8)] = &[(0, -1)];
const KNIGHT_D: &[(i8, i8)] = &[(-1, -2), (1, -2)];
const SILVER_D: &[(i8, i8)] = &[(0, -1), (-1, -1), (1, -1), (-1, 1), (1, 1)];
const GOLD_D: &[(i8, i8)] = &[(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1)];
const KING_D: &[(i8, i8)] = &[
    (0, -1),
    (1, -1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (1, 1),
    (-1, 1),
];
const CROSS45_D: &[(i8, i8)] = &[(-1, -1), (1, -1), (-1, 1), (1, 1)];

fn step_effect(sq: Square, color: Color, deltas: &[(i8, i8)]) -> Bb {
    let sign = dr_sign_for(color);
    let mut bb = Bb::EMPTY;
    for &(df, dr) in deltas {
        if let Some(to) = step_signed(sq, df, dr * sign) {
            bb = bb.with(to);
        }
    }
    bb
}

/// Slider ray in each direction, stopping at (and including) the first square
/// occupied under `occ`.
fn slide_effect(sq: Square, dirs: &[(i8, i8)], occ: Bb) -> Bb {
    let mut bb = Bb::EMPTY;
    for &(df, dr) in dirs {
        let mut cur = sq;
        while let Some(next) = step_signed(cur, df, dr) {
            bb = bb.with(next);
            if occ.contains(next) {
                break;
            }
            cur = next;
        }
    }
    bb
}

fn pawn_effect_scalar(c: Color, sq: Square) -> Bb {
    step_effect(sq, c, PAWN_D)
}
fn knight_effect_scalar(c: Color, sq: Square) -> Bb {
    step_effect(sq, c, KNIGHT_D)
}
fn silver_effect_scalar(c: Color, sq: Square) -> Bb {
    step_effect(sq, c, SILVER_D)
}
fn gold_effect_scalar(c: Color, sq: Square) -> Bb {
    step_effect(sq, c, GOLD_D)
}
fn king_effect_scalar(sq: Square) -> Bb {
    step_effect(sq, Color::Black, KING_D)
}
fn cross45_step_effect_scalar(sq: Square) -> Bb {
    step_effect(sq, Color::Black, CROSS45_D)
}
fn rook_effect_scalar(sq: Square, occ: Bb) -> Bb {
    slide_effect(sq, ORTH_DIRS, occ)
}
fn bishop_effect_scalar(sq: Square, occ: Bb) -> Bb {
    slide_effect(sq, DIAG_DIRS, occ)
}
fn lance_effect_scalar(c: Color, sq: Square, occ: Bb) -> Bb {
    slide_effect(sq, &[(0, forward_dr(c))], occ)
}
fn horse_effect_scalar(sq: Square, occ: Bb) -> Bb {
    bishop_effect_scalar(sq, occ).or(king_effect_scalar(sq))
}
fn dragon_effect_scalar(sq: Square, occ: Bb) -> Bb {
    rook_effect_scalar(sq, occ).or(king_effect_scalar(sq))
}

/// Enemy sliders aimed at `ksq` (king of `king_color`) along their step lines,
/// optionally excluding `avoid`.
fn snipers_to_king_scan(
    board: &Board,
    ksq: Square,
    king_color: Color,
    avoid: Option<Square>,
) -> Bb {
    let enemy = king_color.flip();
    let rook_line = rook_step_effect(ksq);
    let bishop_line = bishop_step_effect(ksq);
    let lance_line = lance_step_effect(king_color, ksq);
    let mut bb = Bb::EMPTY;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).unwrap();
        if Some(sq) == avoid {
            continue;
        }
        let Some(p) = board.get(sq) else { continue };
        if p.color != enemy {
            continue;
        }
        let is_sniper = match p.kind {
            PieceKind::Rook => rook_line.contains(sq),
            PieceKind::Bishop => bishop_line.contains(sq),
            PieceKind::Lance => !p.promoted && lance_line.contains(sq),
            _ => false,
        };
        if is_sniper {
            bb = bb.with(sq);
        }
    }
    bb
}

/// `st->blockersForKing[c]` via `update_slider_blockers`.
fn blockers_for_king_scan(board: &Board, c: Color) -> Bb {
    let Some(ksq) = try_find_king(board, c) else {
        return Bb::EMPTY;
    };
    let snipers = snipers_to_king_scan(board, ksq, c, None);
    let occupancy = occupied_scan(board).sub(snipers);
    let mut blockers = Bb::EMPTY;
    for sniper in snipers.iter() {
        let b = between_bb(ksq, sniper).and(occupancy);
        if !b.is_empty() && !b.more_than_one() {
            blockers = blockers.or(b);
        }
    }
    blockers
}

/// `Position::pinned_pieces<C>(avoid)`.
fn pinned_pieces_avoid_scan(board: &Board, c: Color, avoid: Option<Square>) -> Bb {
    let Some(ksq) = try_find_king(board, c) else {
        return Bb::EMPTY;
    };
    let snipers = snipers_to_king_scan(board, ksq, c, avoid);
    let occ = occupied_scan(board);
    let own = pieces_of_color_scan(board, c);
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

// -- production helpers == scan oracles --------------------------------------

use crate::position::Position;
use crate::sfen::parse_sfen;

const FIXTURE_SFENS: &[&str] = &[
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

fn all_squares() -> impl Iterator<Item = Square> {
    (0..Square::COUNT as u8).map(|i| Square::from_index(i).unwrap())
}

/// The occupancies the mate search can pass to `attackers_of_color`: the full
/// board occupancy, every single-square removal (the mover's `from` cleared),
/// and deterministic random board-masked patterns (covering the multi-removal +
/// `to`-added occupancies of `can_king_escape`, since `attackers_of_color` is a
/// pure function of `occ`).
fn attacker_occupancies(board: &Board, rng: &mut Rng) -> Vec<u128> {
    const BOARD_MASK: u128 = (1u128 << Square::COUNT) - 1;
    let occ_all = board.occupied().raw();
    let mut out = vec![occ_all];
    // Every single-square removal (the mover's `from` cleared).
    for sq in all_squares() {
        if occ_all & (1u128 << sq.index()) != 0 {
            out.push(occ_all & !(1u128 << sq.index()));
        }
    }
    // Deterministic random board-masked patterns, plus `occ_all` perturbed by a
    // random mask (bits toggled) — covering multi-removal + added-square occs.
    let rand_u128 =
        |rng: &mut Rng| ((rng.next() as u128) ^ ((rng.next() as u128) << 64)) & BOARD_MASK;
    for _ in 0..16 {
        out.push(rand_u128(rng));
        out.push(occ_all ^ rand_u128(rng));
    }
    out
}

fn check_helpers(pos: &Position, rng: &mut Rng) {
    let board = pos.board();

    // Board-wide snapshots.
    assert_eq!(super::occupied(board), occupied_scan(board), "occupied");
    assert_eq!(
        super::both_kings(board),
        both_kings_scan(board),
        "both_kings"
    );
    for c in [Color::Black, Color::White] {
        assert_eq!(
            super::pieces_of_color(board, c),
            pieces_of_color_scan(board, c),
            "pieces_of_color {c:?}",
        );
    }

    // Pattern buckets vs the `is_<kind>` predicate scan.
    use crate::board::pat;
    type PredCase = (usize, fn(Piece) -> bool);
    let cases: [PredCase; 9] = [
        (pat::PAWN, is_pawn_unpromoted),
        (pat::LANCE, is_lance_unpromoted),
        (pat::KNIGHT, is_knight_unpromoted),
        (pat::SILVER, is_silver_unpromoted),
        (pat::GOLD, is_golds),
        (pat::BISHOP, is_bishop_unpromoted),
        (pat::ROOK, is_rook_unpromoted),
        (pat::HORSE, is_horse),
        (pat::DRAGON, is_dragon),
    ];
    for c in [Color::Black, Color::White] {
        for (pattern, pred) in cases {
            assert_eq!(
                super::pieces_bucket(board, c, pattern),
                color_pieces_where(board, c, pred),
                "pieces_bucket {c:?} pattern {pattern}",
            );
        }
    }

    // Sniper / blocker / pinned sets, both colours, every `avoid`.
    for c in [Color::Black, Color::White] {
        assert_eq!(
            super::blockers_for_king(board, c),
            blockers_for_king_scan(board, c),
            "blockers_for_king {c:?}",
        );
        let Some(ksq) = try_find_king(board, c) else {
            continue;
        };
        let avoids = std::iter::once(None).chain(all_squares().map(Some));
        for avoid in avoids {
            assert_eq!(
                super::snipers_to_king(board, ksq, c, avoid),
                snipers_to_king_scan(board, ksq, c, avoid),
                "snipers_to_king {c:?} avoid {avoid:?}",
            );
            assert_eq!(
                super::pinned_pieces_avoid(board, c, avoid),
                pinned_pieces_avoid_scan(board, c, avoid),
                "pinned_pieces_avoid {c:?} avoid {avoid:?}",
            );
        }
    }

    // Attacker sets: every target square × colour × occupancy variant.
    let occs = attacker_occupancies(board, rng);
    for sq in all_squares() {
        for color in [Color::Black, Color::White] {
            for &occ in &occs {
                assert_eq!(
                    super::attackers_of_color(board, sq, occ_bb(occ), color),
                    attackers_of_color_scan(board, sq, occ_bb(occ), color),
                    "attackers_of_color sq {sq:?} color {color:?} occ {occ:#x}",
                );
            }
        }
    }
}

/// Mate-rich, king-exposed seeds (droppable hands, promoted pieces near the
/// king) layered on top of the parity fixtures, used by the candidate-filter
/// equivalence gate. Includes positions where the reference-faithful *misses*
/// occur (distant-drop / double-check mates the detector never finds) so both
/// implementations must miss identically.
const FILTER_SEEDS: &[&str] = &[
    // The three head-mate fixtures from `tests/mate_1ply.rs`.
    "k8/9/G1N6/9/9/9/9/9/8K b G 1",
    "k8/1G7/9/9/9/9/9/9/8K b R 1",
    "k8/9/G8/9/9/9/9/9/L7K b - 1",
    // Both kings exposed with large hands of every droppable kind.
    "9/4k4/9/9/9/9/9/4K4/9 b RBGSNLP2rb2g2s2n2l2p 1",
    "4k4/9/9/9/9/9/9/9/4K4 b 2R2B2G2S2N2L9P 1",
    // Promoted minors clustered in front of an exposed enemy king.
    "4k4/3+P+P+P3/9/9/9/9/9/4K4/9 b GSNLRBrb 1",
    "4k4/3+S+N+L3/9/9/9/9/9/4K4/9 b GSNLPrbg 1",
    // Silver / knight / lance origins near a 3rd-4th rank king (the SILVER
    // rank-4/rank-5 promotion candidates and lance skewer band).
    "9/9/4k4/2SSS4/2NNN4/2LLL4/9/4K4/9 b GP2g2s 1",
];

/// Every occupancy-free effect primitive equals its scalar oracle on every
/// square (and both colours where colour-dependent), and every occupancy-aware
/// slider equals its scalar oracle over occupancies sampled from fixture
/// playouts plus deterministic random board masks.
#[test]
#[cfg_attr(miri, ignore)]
fn effect_primitives_match_scalar_oracle() {
    // -- Occupancy-free step / ray effects: square × colour, exhaustive.
    for sq in all_squares() {
        assert_eq!(
            super::king_effect(sq),
            king_effect_scalar(sq),
            "king {sq:?}"
        );
        assert_eq!(
            super::cross45_step_effect(sq),
            cross45_step_effect_scalar(sq),
            "cross45 {sq:?}"
        );
        assert_eq!(
            super::rook_step_effect(sq),
            rook_step_effect(sq),
            "rook_step {sq:?}"
        );
        assert_eq!(
            super::bishop_step_effect(sq),
            bishop_step_effect(sq),
            "bishop_step {sq:?}"
        );
        for c in [Color::Black, Color::White] {
            assert_eq!(
                super::pawn_effect(c, sq),
                pawn_effect_scalar(c, sq),
                "pawn {c:?} {sq:?}"
            );
            assert_eq!(
                super::knight_effect(c, sq),
                knight_effect_scalar(c, sq),
                "knight {c:?} {sq:?}"
            );
            assert_eq!(
                super::silver_effect(c, sq),
                silver_effect_scalar(c, sq),
                "silver {c:?} {sq:?}"
            );
            assert_eq!(
                super::gold_effect(c, sq),
                gold_effect_scalar(c, sq),
                "gold {c:?} {sq:?}"
            );
            assert_eq!(
                super::lance_step_effect(c, sq),
                lance_step_effect(c, sq),
                "lance_step {c:?} {sq:?}"
            );
        }
    }

    // -- Occupancy-aware sliders: sample occupancies and compare per square.
    let check_sliders = |occ: Bb| {
        for sq in all_squares() {
            assert_eq!(
                super::rook_effect(sq, occ),
                rook_effect_scalar(sq, occ),
                "rook {sq:?}"
            );
            assert_eq!(
                super::bishop_effect(sq, occ),
                bishop_effect_scalar(sq, occ),
                "bishop {sq:?}"
            );
            assert_eq!(
                super::horse_effect(sq, occ),
                horse_effect_scalar(sq, occ),
                "horse {sq:?}"
            );
            assert_eq!(
                super::dragon_effect(sq, occ),
                dragon_effect_scalar(sq, occ),
                "dragon {sq:?}"
            );
            for c in [Color::Black, Color::White] {
                assert_eq!(
                    super::lance_effect(c, sq, occ),
                    lance_effect_scalar(c, sq, occ),
                    "lance {c:?} {sq:?}"
                );
            }
        }
    };

    // Real board occupancies from fixture playouts.
    let mut rng = Rng(0xD1CE_F00D_BEEF_1234);
    for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        let mut legal: Vec<Move> = Vec::new();
        for _ in 0..40 {
            check_sliders(super::occupied(pos.board()));
            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                break;
            }
            let m = legal[rng.pick(legal.len())];
            pos.do_move(m);
        }
        let _ = fi;
    }
    // Deterministic random board-masked occupancies (edge cases the playouts
    // may not reach: crowded / sparse boards).
    const BOARD_MASK: u128 = (1u128 << Square::COUNT) - 1;
    for _ in 0..24 {
        let bits = ((rng.next() as u128) ^ ((rng.next() as u128) << 64)) & BOARD_MASK;
        check_sliders(occ_bb(bits));
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn filtered_mate_matches_unfiltered_twin_over_playouts() {
    const MIN_PLIES: usize = 120;

    let seeds = FIXTURE_SFENS.iter().chain(FILTER_SEEDS.iter());
    for (fi, sfen) in seeds.enumerate() {
        // Two independent playout streams per seed widen the sampled reach.
        for stream in 0..2u64 {
            let mut p = parse_sfen(sfen).expect("seed sfen parses");
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ ((fi as u64) << 8) ^ stream);
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                if !p.in_check() {
                    assert_eq!(
                        p.mate_1ply(),
                        p.mate_1ply_unfiltered(),
                        "seed {fi} stream {stream} ply {plies}: filtered != unfiltered\n{}",
                        crate::sfen::format_sfen(&p),
                    );
                }
                let mut legal: Vec<Move> = Vec::new();
                p.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    break;
                }
                let m = legal[rng.pick(legal.len())];
                p.do_move(m);
                plies += 1;
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn bitboard_helpers_match_scan_oracle_on_fixture_playouts() {
    const MIN_PLIES: usize = 60;

    for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        let mut rng = Rng(0x0BAD_F00D_1357_9BDF ^ (fi as u64).wrapping_add(1));
        let mut legal: Vec<Move> = Vec::new();

        let mut plies = 0usize;
        while plies < MIN_PLIES {
            check_helpers(&pos, &mut rng);

            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                break;
            }

            let m = legal[rng.pick(legal.len())];
            pos.do_move(m);
            plies += 1;
        }
    }
}

use crate::move_::Move;
