//! Static Exchange Evaluation tests: hand-built exchange scenarios with their
//! expected value worked out below, threshold-boundary probing, a
//! reference-anchored startpos case, and a seeded-playout sweep.

use attic_state::board::Board;
use attic_state::color::Color;
use attic_state::move_::{Move, parse_usi_move};
use attic_state::piece::{Piece, PieceKind};
use attic_state::position::Position;
use attic_state::sfen::parse_sfen;
use attic_state::square::Square;

/// Apery material values, so that the expected values below can be written in
/// named terms.
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

/// Only the two kings, tucked into opposite corners so that they stay clear of
/// the exchange geometry the scenarios below build on file 4 / rank 4.
fn two_king_board(mover: Color) -> Position {
    let mut pos = Position::empty();
    set(&mut pos, 0, 8, PieceKind::King, Color::Black);
    set(&mut pos, 0, 0, PieceKind::King, Color::White);
    pos.set_side_to_move(mover);
    pos
}

/// Assert the SEE value of `m` is exactly `value`, by probing the threshold
/// boundary either side — the reference's own `see_ge_th` check.
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
    // Nothing recaptures the undefended pawn, so SEE = +Pawn.
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
    // A gold defends the pawn and recaptures the rook: SEE = Pawn - Rook.
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
    assert!(!pos.see_ge(m, 0));
}

#[test]
fn three_deep_recapture_chain_nets_a_pawn() {
    // A black lance x-rays behind the capturing pawn, so after bP x wP (+90)
    // and wP x bP (-90) the lance recaptures (+90) and ends safe: SEE = +Pawn.
    // White declining to recapture leaves black +Pawn too.
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
    // Doubled black rooks against one white defender. Without the rear rook:
    // bR x wP (+90), wR x bR (-990) -> SEE = -900. With it, the rear rook takes
    // the defender (+990) -> SEE = +90, so the sign flips and both are probed.
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
    // Doubled black lances against a white rook. Without the rear lance:
    // bL x wP (+90), wR x bL (-315) -> SEE = -225. With it, White declines to
    // recapture — that would hang the rook to the rear lance — so SEE = +90.
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
    // undefended dragon is worth DragonValue, not RookValue.
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
    // A defended pawn capture inside the promotion zone, but not forced:
    // bP x wP (+90), then the gold recaptures at the *pawn's* value (-90), so
    // SEE = 0 either way. Were the promotion credited, the recaptured piece
    // would be a Tokin worth GoldValue and the two variants would differ.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 3, PieceKind::Pawn, Color::Black);
    set(&mut pos, 4, 2, PieceKind::Pawn, Color::White);
    set(&mut pos, 3, 1, PieceKind::Gold, Color::White);
    let pawn = Piece::new(PieceKind::Pawn, Color::Black);
    let quiet = Move::make(sq(4, 3), sq(4, 2), pawn);
    let promote = Move::make_promote(sq(4, 3), sq(4, 2), pawn);

    assert_see_value(&pos, quiet, 0);
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
    // A pinned recapturer is dropped from the attacker set, so the capture is
    // safe at SEE = +Pawn. Removing the pin lets the gold recapture the rook,
    // for SEE = Pawn - Rook.
    let base = |with_pin: bool| {
        let mut pos = two_king_board(Color::Black);
        // The white king must share the lance's file to be pinnable.
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
    // A spread of thresholds either side of a known SEE = -900.
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
    // The reference's own `see_ge` "pos1move" case: a bishop-takes-bishop with
    // promotion, recaptured by the silver, so the material is even.
    let mut pos = Position::startpos();
    for usi in ["7g7f", "3c3d"] {
        let m = parse_usi_move(usi, &pos).expect("startpos move parses");
        pos.do_move(m);
    }
    let m = parse_usi_move("8h2b+", &pos).expect("bishop capture-promote parses");
    assert_see_value(&pos, m, 0);
}

/// The reference `see_ge` test's position P2, where White has declined to
/// recapture the horse. Black to move.
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
    // The horse steps into the gold's reach, trading itself for the silver it
    // captured.
    let pos = reference_pos2();
    let m = parse_usi_move("2b3a", &pos).expect("horse move parses");
    assert_see_value(&pos, m, -945 + SILVER);
}

#[test]
fn reference_pos2drop_bishop_drop_is_bishop_for_knight() {
    // The dropped bishop is answered by the knight and recaptured by the horse.
    let pos = reference_pos2();
    let m = parse_usi_move("B*3c", &pos).expect("bishop drop parses");
    assert_see_value(&pos, m, -855 + 405);
}

#[test]
fn reference_pos2move_horse_to_33_is_a_free_loss() {
    // The horse hangs itself to the knight for nothing.
    let pos = reference_pos2();
    let m = parse_usi_move("2b3c", &pos).expect("horse move parses");
    assert_see_value(&pos, m, -945);
}

#[test]
fn silver_recapture_is_the_least_valuable_attacker() {
    // Two defenders of different value: SEE must recapture with the cheaper
    // silver, for bR x wS (+495) then wS x bR (-990).
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

/// The six perft-fixture SFENs; the playout below drives one deterministic game
/// from each.
const FIXTURE_SFENS: &[&str] = &[
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
    "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
    "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",                // sennichite
];

/// A small deterministic xorshift64*.
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

/// Thresholds swept per capture, two calls each for determinism plus a
/// monotonicity check.
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

            for &m in &legal {
                if !is_capture(pos.board(), m) {
                    continue;
                }
                // `see_ge` is monotone non-increasing in the threshold and
                // THRESHOLDS is ascending, so a `true` here implies `true` at
                // every threshold already visited.
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
