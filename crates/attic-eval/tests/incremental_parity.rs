//! Gate: the incremental accumulator update must equal a from-scratch refresh
//! **exactly** (i16 equality) after every do and every undo, and both eval
//! paths must agree.
//!
//! Two drivers exercise [`attic_eval::Accumulator::update_after_move`]:
//!
//! 1. **Fixture lines** — every `tests/fixtures/eval/*.json` with a `moves`
//!    array is replayed move-by-move, then fully unwound.
//! 2. **Randomized playouts** — from each of the six fixture SFENs, a
//!    deterministic pseudo-random legal game of ≥ 30 plies (via the real
//!    movegen), with terminal positions unwound and restarted from the root.
//!
//! At each ply the incremental accumulator (threaded through a do/undo stack) is
//! compared bit-for-bit against a fresh [`Accumulator::refresh`], and
//! [`evaluate_with`] over the incremental accumulator is checked against the
//! full-refresh [`evaluate`] — at the child (after the do) and again at the
//! parent (after the undo).
//!
//! The network file is staged locally at
//! `eval/nn.bin` and is never committed. When it is
//! absent the test prints a notice and passes, so the default `cargo test` run
//! stays green everywhere.

use std::path::PathBuf;

use attic_eval::{Accumulator, NnueNetwork, evaluate, evaluate_with, load_network};
use attic_state::{Color, Move, Position, Undo, format_usi_move, parse_sfen, parse_usi_move};

/// The six eval-fixture SFENs the playout driver seeds from.
const FIXTURE_SFENS: &[&str] = &[
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
    "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
    "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",                // sennichite
];

fn workspace_relative(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn nn_bin_path() -> PathBuf {
    workspace_relative("eval/nn.bin")
}

/// Refresh a fresh accumulator from `pos`.
fn refreshed(net: &NnueNetwork, pos: &Position) -> Accumulator {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    acc
}

/// Assert both halves of `acc` are bit-identical to a from-scratch refresh of
/// `pos`, and that `evaluate_with(acc)` matches the full-refresh `evaluate`.
fn assert_matches_refresh(net: &NnueNetwork, acc: &Accumulator, pos: &Position, ctx: &str) {
    let fresh = refreshed(net, pos);
    for color in [Color::Black, Color::White] {
        assert_eq!(
            acc.perspective(color),
            fresh.perspective(color),
            "{ctx}: {color:?} half diverged from refresh",
        );
    }
    assert_eq!(
        evaluate_with(net, acc, pos),
        evaluate(net, pos),
        "{ctx}: evaluate paths disagree",
    );
}

/// A do/undo frame: the move applied plus its `Undo` token and the incremental
/// accumulator for the resulting child position.
struct Frame {
    mv: Move,
    undo: Undo,
    acc: Accumulator,
}

/// Advance `pos` by `mv`: build the incremental child accumulator from `parent`,
/// apply the move for real, and verify the child accumulator matches a fresh
/// refresh of the post-move position. Returns the frame to push.
fn advance(
    net: &NnueNetwork,
    pos: &mut Position,
    parent: &Accumulator,
    mv: Move,
    ctx: &str,
) -> Frame {
    let acc = parent.update_after_move(net, pos, mv);
    let undo = pos.do_move(mv);
    assert_matches_refresh(net, &acc, pos, &format!("{ctx} [after do]"));
    Frame { mv, undo, acc }
}

/// Pop `frame` off `pos` and confirm the now-current parent accumulator is
/// still valid for the restored position.
fn retreat(net: &NnueNetwork, pos: &mut Position, frame: Frame, parent: &Accumulator, ctx: &str) {
    pos.undo_move(frame.mv, frame.undo);
    assert_matches_refresh(net, parent, pos, &format!("{ctx} [after undo]"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn incremental_accumulator_matches_refresh_on_fixture_lines() {
    let nn_bin = nn_bin_path();
    if !nn_bin.exists() {
        eprintln!(
            "skipping incremental_accumulator_matches_refresh_on_fixture_lines: {} is not present (staged only on the dev VM)",
            nn_bin.display()
        );
        return;
    }
    let net = load_network(&nn_bin).expect("real nn.bin should load and validate");

    let dir = workspace_relative("tests/fixtures/eval");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read fixtures dir")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();

    let mut any_moves = false;
    for path in paths {
        let text = std::fs::read_to_string(&path).expect("read fixture");
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse fixture");
        let sfen = json["sfen"].as_str().expect("fixture sfen");
        let moves: Vec<String> = match json.get("moves") {
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .map(|m| m.as_str().expect("move string").to_string())
                .collect(),
            _ => Vec::new(),
        };
        if moves.is_empty() {
            continue;
        }
        any_moves = true;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();

        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        let root = refreshed(&net, &pos);
        let mut frames: Vec<Frame> = Vec::new();

        for (ply, usi) in moves.iter().enumerate() {
            let mv = parse_usi_move(usi, &pos).expect("fixture move parses");
            let ctx = format!("{name} ply {ply} `{usi}`");
            let parent = frames.last().map_or(&root, |f| &f.acc);
            let frame = advance(&net, &mut pos, parent, mv, &ctx);
            frames.push(frame);
        }

        // Unwind the whole line, checking parity after each undo.
        while let Some(frame) = frames.pop() {
            let usi = format_usi_move(frame.mv);
            let parent = frames.last().map_or(&root, |f| &f.acc);
            retreat(
                &net,
                &mut pos,
                frame,
                parent,
                &format!("{name} unwind `{usi}`"),
            );
        }
    }

    assert!(any_moves, "no fixture had a `moves` array to exercise");
}

/// Small deterministic xorshift64* — no external RNG crate, and
/// `Math.random`-style nondeterminism is banned in this workspace.
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

#[test]
#[cfg_attr(miri, ignore)]
fn incremental_accumulator_matches_refresh_on_random_playouts() {
    let nn_bin = nn_bin_path();
    if !nn_bin.exists() {
        eprintln!(
            "skipping incremental_accumulator_matches_refresh_on_random_playouts: {} is not present (staged only on the dev VM)",
            nn_bin.display()
        );
        return;
    }
    let net = load_network(&nn_bin).expect("real nn.bin should load and validate");

    const MIN_PLIES: usize = 30;

    for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        let root = refreshed(&net, &pos);
        // Seed derived from the fixture index so each game is distinct yet
        // fully reproducible run-to-run.
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (fi as u64).wrapping_add(1));

        let mut frames: Vec<Frame> = Vec::new();
        let mut legal: Vec<Move> = Vec::new();

        let mut plies = 0usize;
        while plies < MIN_PLIES {
            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                // Terminal (mate/stalemate): unwind fully and restart from root,
                // exercising the undo path along the way.
                while let Some(frame) = frames.pop() {
                    let usi = format_usi_move(frame.mv);
                    let parent = frames.last().map_or(&root, |f| &f.acc);
                    retreat(
                        &net,
                        &mut pos,
                        frame,
                        parent,
                        &format!("fixture {fi} restart `{usi}`"),
                    );
                }
                continue;
            }
            let mv = legal[rng.pick(legal.len())];
            let ctx = format!("fixture {fi} ply {plies} `{}`", format_usi_move(mv));
            let parent = frames.last().map_or(&root, |f| &f.acc);
            let frame = advance(&net, &mut pos, parent, mv, &ctx);
            frames.push(frame);
            plies += 1;
        }

        // Unwind the completed game, verifying every undo.
        while let Some(frame) = frames.pop() {
            let usi = format_usi_move(frame.mv);
            let parent = frames.last().map_or(&root, |f| &f.acc);
            retreat(
                &net,
                &mut pos,
                frame,
                parent,
                &format!("fixture {fi} unwind `{usi}`"),
            );
        }
    }
}
