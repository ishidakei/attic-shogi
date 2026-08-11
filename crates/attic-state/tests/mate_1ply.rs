//! Gate for the one-ply mate detector (`Position::mate_1ply`).
//!
//! The detector is a faithful port of upstream YaneuraOu's `Mate::mate_1ply`
//! (`source/mate/mate1ply_without_effect.cpp`), including its
//! deliberate misses — so this gate does **not** assert completeness (it is a
//! non-goal to find every 1-ply mate the reference misses). What it pins is:
//!
//! 1. **Soundness (hermetic):** over seeded random playouts from the six perft
//!    fixtures, at every visited position where the side to move is not in check
//!    (the reference's `ASSERT_LV3(!checkers())` precondition), whenever
//!    `mate_1ply` returns a move it is (a) legal, (b) gives check, and (c) leaves
//!    the opponent with zero legal replies — i.e. it really is mate.
//! 2. **Determinism:** the same position yields the same result across repeated
//!    calls.
//! 3. A handful of hand-built positions with a known head-mate confirm the
//!    detector actually fires (guards against a vacuously-sound "always None").
//!
//! # Why there is no direct reference-parity test here
//!
//! No drivable reference entry point for `Mate::mate_1ply` exists in the pinned
//! submodule: it is exercised only from inside `yaneuraou-search.cpp` (the 一手
//! 詰め block), and none of the test commands (`source/testcmd/`, incl.
//! `mate_test_cmd.cpp` / `unit_test.cpp`) nor any USI extension exposes it
//! directly (`grep -rn mate_1ply source` finds only search-internal call sites).
//! Adding one would require patching the read-only submodule, which project
//! policy forbids. Exact (position → move / none) agreement is therefore
//! arbitrated indirectly, by the search node-count parity gates that run with
//! this detector wired into qsearch; this file supplies the hermetic soundness
//! and determinism gates.

use attic_state::move_::{Move, format_usi_move};
use attic_state::position::{Position, Undo};
use attic_state::sfen::parse_sfen;

/// The six perft-fixture SFENs (matching `tests/fixtures/perft/*.json`).
const FIXTURE_SFENS: &[&str] = &[
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
    "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
    "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",                // sennichite
];

/// Small deterministic xorshift64* (mirrors the drivers elsewhere in this
/// crate); `Math.random`-style nondeterminism is banned in this workspace.
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

/// Assert a move returned by `mate_1ply` for `pos` (side to move not in check)
/// really is a legal mate-in-one: it is in the legal move list, it gives check,
/// and the opponent has no legal reply.
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
fn mate_1ply_is_sound_and_deterministic_over_fixture_playouts() {
    const MIN_PLIES: usize = 60;
    let mut fired = 0usize;

    for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        let mut rng = Rng(0x51ED_2701_A17C_0FE3 ^ (fi as u64).wrapping_add(1));
        let mut stack: Vec<(Move, Undo)> = Vec::new();

        let mut plies = 0usize;
        while plies < MIN_PLIES {
            // The reference precondition is `!checkers()`; only probe there.
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
                // Terminal: unwind fully and restart from the root.
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

    // The drop-heavy fixture in particular carries mate-in-1 leaves; the sweep
    // should exercise the detector's positive branch at least once (otherwise
    // "sound" would be vacuous).
    assert!(
        fired > 0,
        "mate_1ply never returned a move across the fixture playouts",
    );
}

#[test]
fn startpos_has_no_one_ply_mate() {
    assert_eq!(Position::startpos().mate_1ply(), None);
}

/// Gold-drop head mate: White king 9a; Black gold 9c and knight 7c cover the
/// escape squares; Black king tucked away. A supported gold drop beside the king
/// is mate (same net as the movegen crate's
/// `gold_drop_mate_is_legal_uchifuzume_is_pawn_only` fixture). The exact square
/// (`G*8a`, the lowest-index candidate the reference's `bb.pop()` order returns
/// first) is pinned as a determinism/regression guard.
#[test]
fn finds_gold_drop_head_mate() {
    let pos = parse_sfen("k8/9/G1N6/9/9/9/9/9/8K b G 1").unwrap();
    let m = pos
        .mate_1ply()
        .expect("gold-drop head mate should be found");
    assert_is_mate(&pos, m, "gold-drop head mate");
    assert_eq!(format_usi_move(m), "G*8a");
}

/// Rook-drop head mate: White king cornered at 9a with a Black gold on 8b
/// guarding the escape squares and the drop square. Dropping the rook adjacent
/// to the king delivers a supported, escape-proof check.
#[test]
fn finds_rook_drop_mate() {
    let pos = parse_sfen("k8/1G7/9/9/9/9/9/9/8K b R 1").unwrap();
    let m = pos.mate_1ply().expect("rook-drop mate should be found");
    assert_is_mate(&pos, m, "rook-drop mate");
    assert_eq!(format_usi_move(m), "R*8a");
}

/// A board-move mate (exercises the move-mate branch, not a drop). White king
/// cornered at 9a=(8,0); Black gold on 9c=(8,2) and a Black lance behind it on
/// 9i=(8,8) up the 9-file. The gold advances to 9b=(8,1): it is a check the king
/// cannot capture (the lance, unblocked once the gold vacates 9c, supports 9b)
/// and cannot flee (the gold covers both 8a/8b escapes). The nearer diagonal
/// push to 8b=(7,1) is *unsupported*, so the reference skips it and returns the
/// supported 9c9b — pinned here as a determinism/regression guard.
#[test]
fn finds_a_move_mate() {
    let pos = parse_sfen("k8/9/G8/9/9/9/9/9/L7K b - 1").unwrap();
    let m = pos.mate_1ply().expect("gold-push mate should be found");
    assert_is_mate(&pos, m, "move mate");
    assert_eq!(format_usi_move(m), "9c9b");
    assert!(!m.is_drop(), "expected a board move, got a drop");
}
