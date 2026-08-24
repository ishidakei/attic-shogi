//! Static Exchange Evaluation (`Position::see_ge`) gate.
//!
//! Hand-built exchange scenarios with the expected SEE value worked out in the
//! comments, threshold-boundary probing, a reference-anchored startpos case,
//! and a seeded-playout determinism / no-panic sweep across the six perft
//! fixture SFENs. See `crates/attic-state/src/see.rs` for the port itself and
//! the faithful-behaviour notes it mirrors.

use attic_state::board::Board;
use attic_state::color::Color;
use attic_state::move_::{Move, parse_usi_move};
use attic_state::piece::{Piece, PieceKind};
use attic_state::position::Position;
use attic_state::sfen::parse_sfen;
use attic_state::square::Square;

/// Apery material values (mirrors `see.rs`), so the expected SEE
/// values below can be written in named terms.
const PAWN: i32 = 90;
const LANCE: i32 = 315;
const SILVER: i32 = 495;
const GOLD: i32 = 540;
const ROOK: i32 = 990;
const DRAGON: i32 = 1395;

fn sq(file: u8, rank: u8) -> Square {
    Square::new(file, rank).unwrap()
}

fn set(pos: &mut Position, file: u8, rank: u8, kind: PieceKind, color: Color) {
    pos.board_mut()
        .set(sq(file, rank), Some(Piece::new(kind, color)));
}

fn set_promoted(pos: &mut Position, file: u8, rank: u8, kind: PieceKind, color: Color) {
    pos.board_mut()
        .set(sq(file, rank), Some(Piece::promoted(kind, color).unwrap()));
}

/// A position with only the two kings tucked in opposite corners, `mover` to
/// move. The corners keep the kings clear of the exchange geometry on file 4 /
/// rank 4 used by the scenarios below.
fn two_king_board(mover: Color) -> Position {
    let mut pos = Position::empty();
    set(&mut pos, 0, 8, PieceKind::King, Color::Black);
    set(&mut pos, 0, 0, PieceKind::King, Color::White);
    pos.set_side_to_move(mover);
    pos
}

/// Assert the SEE value of `m` is exactly `value`: `see_ge` is true for every
/// threshold `<= value` (probing `value` and `value - 1`) and false for every
/// threshold `> value` (probing `value + 1`). This is the reference's own
/// `see_ge_th` boundary check.
fn assert_see_value(pos: &Position, m: Move, value: i32) {
    assert!(
        pos.see_ge(m, value),
        "see_ge(m, {value}) should be true (threshold == SEE value)",
    );
    assert!(
        pos.see_ge(m, value - 1),
        "see_ge(m, {}) should be true (threshold below SEE value)",
        value - 1,
    );
    assert!(
        !pos.see_ge(m, value + 1),
        "see_ge(m, {}) should be false (threshold above SEE value)",
        value + 1,
    );
}

#[test]
fn undefended_capture_wins_the_victim() {
    // Black rook on 5i (4,8) slides up the empty 5-file and captures an
    // undefended White pawn on 5e (4,4). Nothing recaptures, so SEE = +Pawn.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    assert_see_value(&pos, m, PAWN);
}

#[test]
fn defended_pawn_capture_by_rook_is_a_loss() {
    // Black rook captures a White pawn on 5e (4,4) that a White gold on 6f
    // (3,3) defends. White recaptures the rook: SEE = Pawn - Rook = -900.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
    set(&mut pos, 3, 3, PieceKind::Gold, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    assert_see_value(&pos, m, PAWN - ROOK);
    // The verdict at threshold 0 is a loss.
    assert!(!pos.see_ge(m, 0));
}

#[test]
fn three_deep_recapture_chain_nets_a_pawn() {
    // File-4 stack over target 5e (4,4) = White pawn:
    //   Black pawn 5d (4,5)   — front attacker (LVA)
    //   Black lance 5g (4,6)  — x-ray behind the pawn, revealed once it moves
    //   White pawn 5f (4,3)   — defender
    // Sequence: bP x wP (+90), wP x bP (-90), bL x wP (+90). The black lance
    // ends safe, so SEE = +Pawn. (White declining to recapture also leaves
    // black +Pawn.)
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
    set(&mut pos, 4, 5, PieceKind::Pawn, Color::Black);
    set(&mut pos, 4, 6, PieceKind::Lance, Color::Black);
    set(&mut pos, 4, 3, PieceKind::Pawn, Color::White);
    let m = Move::make(
        sq(4, 5),
        sq(4, 4),
        Piece::new(PieceKind::Pawn, Color::Black),
    );
    assert_see_value(&pos, m, PAWN);
}

#[test]
fn xray_rook_behind_rook_flips_loss_to_win() {
    // Doubled Black rooks on the 5-file over target 5e (4,4) = White pawn,
    // with one White rook defending along rank 4 from 7e (2,4):
    //   Black rook 5d (4,5)  — front
    //   Black rook 5g (4,6)  — x-ray behind the front rook
    //   White rook 7e (2,4)  — defender
    // Without the rear rook: bR x wP (+90), wR x bR (-990) -> SEE = -900 (loss).
    // With it: + bR x wR (+990) -> SEE = +90 (win). The rear rook flips the
    // sign, so probe both.
    let base = |with_rear: bool| {
        let mut pos = two_king_board(Color::Black);
        set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
        set(&mut pos, 4, 5, PieceKind::Rook, Color::Black);
        set(&mut pos, 2, 4, PieceKind::Rook, Color::White);
        if with_rear {
            set(&mut pos, 4, 6, PieceKind::Rook, Color::Black);
        }
        pos
    };
    let m = Move::make(
        sq(4, 5),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );

    let without = base(false);
    assert_see_value(&without, m, PAWN - ROOK);
    assert!(!without.see_ge(m, 0), "without the x-ray rook it is a loss");

    let with = base(true);
    assert_see_value(&with, m, PAWN - ROOK + ROOK);
    assert!(with.see_ge(m, 0), "the x-ray rook flips it to a win");
}

#[test]
fn xray_lance_behind_lance_changes_the_verdict() {
    // Doubled Black lances on the 5-file over target 5e (4,4) = White pawn,
    // with one White rook defending from 7e (2,4):
    //   Black lance 5d (4,5) — front
    //   Black lance 5g (4,6) — x-ray behind it
    //   White rook 7e (2,4)  — defender
    // Without the rear lance: bL x wP (+90), wR x bL (-315) -> SEE = -225
    // (loss). With it, White declines to recapture (recapturing would let the
    // rear lance take the rook), so SEE = +90 (win).
    let base = |with_rear: bool| {
        let mut pos = two_king_board(Color::Black);
        set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
        set(&mut pos, 4, 5, PieceKind::Lance, Color::Black);
        set(&mut pos, 2, 4, PieceKind::Rook, Color::White);
        if with_rear {
            set(&mut pos, 4, 6, PieceKind::Lance, Color::Black);
        }
        pos
    };
    let m = Move::make(
        sq(4, 5),
        sq(4, 4),
        Piece::new(PieceKind::Lance, Color::Black),
    );

    let without = base(false);
    assert_see_value(&without, m, PAWN - LANCE);
    assert!(
        !without.see_ge(m, 0),
        "without the x-ray lance it is a loss"
    );

    let with = base(true);
    assert_see_value(&with, m, PAWN);
    assert!(with.see_ge(m, 0), "the x-ray lance flips it to a win");
}

#[test]
fn promoted_victim_contributes_its_promoted_value() {
    // Capturing an already-promoted piece uses its promoted value: an
    // undefended White dragon on 5e (4,4) is worth DragonValue, not RookValue.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set_promoted(&mut pos, 4, 4, PieceKind::Rook, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    assert_see_value(&pos, m, DRAGON);
}

#[test]
fn move_promotion_is_not_credited() {
    // The reference's see_ge ignores the promotion of the move itself: it uses
    // the *unpromoted* value of the moving piece and adds no promotion bonus.
    // A Black pawn on 5d (4,3) captures a White pawn on 5c (4,2) — in the
    // promotion zone but not a forced promotion — defended by a White gold on
    // 6b (3,1). SEE = 0 whether or not the move promotes.
    //
    //   bP x wP (+90), wG x (mover) (-90 using the *pawn's* value) -> SEE = 0.
    // If promotion were credited the recaptured piece would be a Tokin worth
    // GoldValue and the value would differ; asserting the promote and
    // non-promote variants are identical pins the faithful behaviour.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 3, PieceKind::Pawn, Color::Black);
    set(&mut pos, 4, 2, PieceKind::Pawn, Color::White);
    set(&mut pos, 3, 1, PieceKind::Gold, Color::White);
    let pawn = Piece::new(PieceKind::Pawn, Color::Black);
    let quiet = Move::make(sq(4, 3), sq(4, 2), pawn);
    let promote = Move::make_promote(sq(4, 3), sq(4, 2), pawn);

    assert_see_value(&pos, quiet, 0);
    // Promotion of the move does not change the SEE value at any threshold.
    for th in [-GOLD, -PAWN, -1, 0, 1, PAWN, GOLD] {
        assert_eq!(
            pos.see_ge(quiet, th),
            pos.see_ge(promote, th),
            "promotion flag must not change see_ge at threshold {th}",
        );
    }
}

#[test]
fn pinned_defender_cannot_recapture() {
    // A pinned recapturer is dropped from the attacker set (the reference's
    // `pinners(~stm) & occupied` guard). Black rook on 5i (4,8) captures a
    // White pawn on 5e (4,4). White's only defender is a gold on 4f (5,3),
    // but it is pinned to the White king on 4i (5,0) by a Black lance on 4a
    // (5,8) down the 4-file. Pinned, the gold cannot recapture, so the capture
    // is safe: SEE = +Pawn. Remove the pin and the gold recaptures the rook:
    // SEE = Pawn - Rook.
    let base = |with_pin: bool| {
        let mut pos = two_king_board(Color::Black);
        // Reposition the White king onto the 4-file so the lance can pin.
        pos.board_mut().set(sq(0, 0), None);
        set(&mut pos, 5, 0, PieceKind::King, Color::White);
        set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
        set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
        set(&mut pos, 5, 3, PieceKind::Gold, Color::White);
        if with_pin {
            set(&mut pos, 5, 8, PieceKind::Lance, Color::Black);
        }
        pos
    };
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );

    let pinned = base(true);
    assert_see_value(&pinned, m, PAWN);
    assert!(
        pinned.see_ge(m, 0),
        "pinned defender makes the capture safe"
    );

    let free = base(false);
    assert_see_value(&free, m, PAWN - ROOK);
    assert!(!free.see_ge(m, 0), "an unpinned gold recaptures — a loss");
}

#[test]
fn threshold_semantics_sweep_around_a_known_exchange() {
    // The defended-rook-capture loss (SEE = -900): probe a spread of
    // thresholds either side and require the exact `v <= SEE` boundary.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
    set(&mut pos, 3, 3, PieceKind::Gold, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    let see = PAWN - ROOK; // -900
    for th in [-2000, -1000, see - 1, see, see + 1, 0, 90, 500, 2000] {
        assert_eq!(
            pos.see_ge(m, th),
            th <= see,
            "see_ge(m, {th}) should equal ({th} <= {see})",
        );
    }
}

#[test]
fn reference_pos1move_bishop_capture_promote_is_see_zero() {
    // The reference's own `see_ge` "pos1move" test: from startpos play 7g7f,
    // 3c3d, then evaluate 8h2b+ (bishop takes bishop with promotion). It is
    // recaptured by the silver, so the material is even and SEE = 0. This
    // anchors the port against a reference-blessed value.
    let mut pos = Position::startpos();
    for usi in ["7g7f", "3c3d"] {
        let m = parse_usi_move(usi, &pos).expect("startpos move parses");
        pos.do_move(m);
    }
    let m = parse_usi_move("8h2b+", &pos).expect("bishop capture-promote parses");
    assert_see_value(&pos, m, 0);
}

/// Build the reference `see_ge` test's position P2: from startpos play 7g7f,
/// 3c3d, 8h2b+ (bishop-capture-promote to a horse on 2b), then White's 8c8d
/// pawn push (declining to recapture the horse). Black is to move.
fn reference_pos2() -> Position {
    let mut pos = Position::startpos();
    for usi in ["7g7f", "3c3d", "8h2b+", "8c8d"] {
        let m = parse_usi_move(usi, &pos).expect("reference-sequence move parses");
        pos.do_move(m);
    }
    pos
}

#[test]
fn reference_pos2move_horse_to_31_is_horse_for_silver() {
    // Reference "pos2move": from P2, 2b3a (horse into 3a) is met by the gold,
    // trading the horse for the silver it captured. SEE = -Horse + Silver.
    let pos = reference_pos2();
    let m = parse_usi_move("2b3a", &pos).expect("horse move parses");
    assert_see_value(&pos, m, -945 + SILVER);
}

#[test]
fn reference_pos2drop_bishop_drop_is_bishop_for_knight() {
    // Reference "pos2drop": from P2, B*3c is answered by the knight and then
    // recaptured by the horse. SEE = -Bishop + Knight.
    let pos = reference_pos2();
    let m = parse_usi_move("B*3c", &pos).expect("bishop drop parses");
    assert_see_value(&pos, m, -855 + 405);
}

#[test]
fn reference_pos2move_horse_to_33_is_a_free_loss() {
    // Reference "pos2move" (second): from P2, 2b3c hangs the horse to the
    // knight for nothing. SEE = -Horse.
    let pos = reference_pos2();
    let m = parse_usi_move("2b3c", &pos).expect("horse move parses");
    assert_see_value(&pos, m, -945);
}

#[test]
fn silver_recapture_is_the_least_valuable_attacker() {
    // Undefended-looking capture where the recapture choice matters: Black
    // rook captures a White silver on 5e (4,4) defended by both a White silver
    // on 6f (3,3) and a White gold on 4f (5,3). SEE picks the cheaper silver
    // to recapture: bR x wS (+495), wS x bR (-990) -> SEE = Silver - Rook.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set(&mut pos, 4, 4, PieceKind::Silver, Color::White);
    set(&mut pos, 3, 3, PieceKind::Silver, Color::White);
    set(&mut pos, 5, 3, PieceKind::Gold, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    assert_see_value(&pos, m, SILVER - ROOK);
}

// -- Determinism / no-panic sweep over the perft fixtures --------------------

/// The six perft-fixture SFENs (matching `tests/fixtures/perft/*.json` and the
/// list mirrored in `position.rs`). The seeded playout below drives one
/// deterministic game from each and exercises `see_ge` on every legal capture.
const FIXTURE_SFENS: &[&str] = &[
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
    "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
    "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",                // sennichite
];

/// Small deterministic xorshift64* (mirrors the driver in `position.rs`).
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

/// A capture is a non-drop move landing on an occupied square.
fn is_capture(board: &Board, m: Move) -> bool {
    !m.is_drop() && board.get(m.to_sq()).is_some()
}

/// Thresholds swept per capture: two calls each (determinism) and a
/// monotonicity check (`see_ge(v)` implies `see_ge(v')` for all `v' <= v`).
const THRESHOLDS: [i32; 9] = [-3000, -900, -90, -1, 0, 1, 90, 900, 3000];

#[test]
#[cfg_attr(miri, ignore)]
fn see_ge_is_deterministic_and_panic_free_on_fixture_playouts() {
    const MIN_PLIES: usize = 40;

    for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ (fi as u64).wrapping_add(1));
        let mut legal: Vec<Move> = Vec::new();

        let mut plies = 0usize;
        while plies < MIN_PLIES {
            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                break;
            }

            // Probe see_ge on every legal capture at this node.
            for &m in &legal {
                if !is_capture(pos.board(), m) {
                    continue;
                }
                // THRESHOLDS is ascending; see_ge is monotone non-increasing
                // in the threshold, so a `true` at a higher threshold implies
                // `true` at every lower one already visited.
                let mut prev: Option<bool> = None;
                for &th in &THRESHOLDS {
                    let a = pos.see_ge(m, th);
                    let b = pos.see_ge(m, th);
                    assert_eq!(a, b, "see_ge not deterministic (fixture {fi}, th {th})");
                    if let Some(pv) = prev {
                        assert!(
                            !a || pv,
                            "see_ge true at higher threshold but false at a lower one (fixture {fi})",
                        );
                    }
                    prev = Some(a);
                }
            }

            let m = legal[rng.pick(legal.len())];
            pos.do_move(m);
            plies += 1;
        }
    }
}
