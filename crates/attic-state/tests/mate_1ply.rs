//! Soundness and determinism tests for the one-ply mate detector.
//!
//! The detector reproduces the reference's deliberate misses, so this suite does
//! **not** assert completeness. What it pins is that a returned move really is
//! a legal mate in one, that repeated calls agree, and — against a vacuously
//! sound "always `None`" — that the detector fires on known head mates.
//!
//! Exact position-to-move agreement with the reference is arbitrated elsewhere,
//! by the search node-count parity tests that run with this detector wired into
//! qsearch: the reference exposes no drivable entry point for `Mate::mate_1ply`
//! to compare against directly.

use attic_state::move_::{Move, format_usi_move};
use attic_state::position::{Position, Undo};
use attic_state::sfen::parse_sfen;

/// The six perft-fixture SFENs.
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

fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut v = Vec::new();
    pos.generate_legal_all(&mut v);
    v
}

/// Assert a returned move really is a legal mate in one.
fn assert_is_mate(pos: &Position, m: Move, ctx: &str) {
    let legal = legal_moves(pos);
    assert!(
        legal.contains(&m),
        "{ctx}: mate_1ply returned {} which is not legal (legal: {:?})",
        format_usi_move(m),
        legal
            .iter()
            .map(|&x| format_usi_move(x))
            .collect::<Vec<_>>(),
    );
    assert!(
        pos.gives_check(m),
        "{ctx}: mate_1ply move {} does not give check",
        format_usi_move(m),
    );
    let mut after = pos.clone();
    let undo: Undo = after.do_move(m);
    let replies = legal_moves(&after);
    assert!(
        replies.is_empty(),
        "{ctx}: mate_1ply move {} leaves {} legal replies (e.g. {})",
        format_usi_move(m),
        replies.len(),
        replies
            .first()
            .map(|&r| format_usi_move(r))
            .unwrap_or_default(),
    );
    after.undo_move(m, undo);
}

#[test]
#[cfg_attr(miri, ignore)]
fn mate_1ply_is_sound_and_deterministic_over_fixture_playouts() {
    const MIN_PLIES: usize = 60;
    let mut fired = 0usize;

    for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        let mut rng = Rng(0x51ED_2701_A17C_0FE3 ^ (fi as u64).wrapping_add(1));
        let mut stack: Vec<(Move, Undo)> = Vec::new();

        let mut plies = 0usize;
        while plies < MIN_PLIES {
            if !pos.in_check() {
                let r1 = pos.mate_1ply();
                let r2 = pos.mate_1ply();
                assert_eq!(
                    r1, r2,
                    "fixture {fi} ply {plies}: mate_1ply is not deterministic",
                );
                if let Some(m) = r1 {
                    assert_is_mate(&pos, m, &format!("fixture {fi} ply {plies}"));
                    fired += 1;
                }
            }

            let legal = legal_moves(&pos);
            if legal.is_empty() {
                while let Some((m, u)) = stack.pop() {
                    pos.undo_move(m, u);
                }
                continue;
            }
            let m = legal[rng.pick(legal.len())];
            let u = pos.do_move(m);
            stack.push((m, u));
            plies += 1;
        }
    }

    // Without a positive branch somewhere in the sweep, "sound" is vacuous.
    assert!(
        fired > 0,
        "mate_1ply never returned a move across the fixture playouts",
    );
}

#[test]
fn startpos_has_no_one_ply_mate() {
    assert_eq!(Position::startpos().mate_1ply(), None);
}

/// A gold drop beside a cornered king, supported and with the escapes covered.
/// The exact square is pinned too: it is the lowest-index candidate, which the
/// reference's `bb.pop()` order returns first.
#[test]
fn finds_gold_drop_head_mate() {
    let pos = parse_sfen("k8/9/G1N6/9/9/9/9/9/8K b G 1").unwrap();
    let m = pos
        .mate_1ply()
        .expect("gold-drop head mate should be found");
    assert_is_mate(&pos, m, "gold-drop head mate");
    assert_eq!(format_usi_move(m), "G*8a");
}

/// A rook dropped adjacent to a cornered king, with a gold guarding both the
/// escapes and the drop square.
#[test]
fn finds_rook_drop_mate() {
    let pos = parse_sfen("k8/1G7/9/9/9/9/9/9/8K b R 1").unwrap();
    let m = pos.mate_1ply().expect("rook-drop mate should be found");
    assert_is_mate(&pos, m, "rook-drop mate");
    assert_eq!(format_usi_move(m), "R*8a");
}

/// A board-move mate, exercising the move-mate branch rather than a drop: a
/// gold advances beside a cornered king, supported by a lance the advance itself
/// unblocks. The nearer diagonal push is *unsupported*, so the reference skips
/// it — pinned here as a regression guard.
#[test]
fn finds_a_move_mate() {
    let pos = parse_sfen("k8/9/G8/9/9/9/9/9/L7K b - 1").unwrap();
    let m = pos.mate_1ply().expect("gold-push mate should be found");
    assert_is_mate(&pos, m, "move mate");
    assert_eq!(format_usi_move(m), "9c9b");
    assert!(!m.is_drop(), "expected a board move, got a drop");
}
