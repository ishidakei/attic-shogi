//! Depth-1 search parity gate (blocking).
//!
//! Runs the `go depth 1` root search ([`QSearch::run_root`]) against the six
//! reference-captured fixtures under `tests/fixtures/search-depth1/` and asserts
//! that **bestmove, score, and nodes** all match exactly. These three are one
//! inseparable set: the `(nodes & 14)` root tie-break means a single-node drift
//! can cascade into a different score and a flipped bestmove, so a mismatch on
//! any of them signals a divergence in the search itself.
//!
//! The fixtures were captured with Threads=1, no book, `usinewgame` before each
//! position, `go depth 1`, USI_Hash default 1024 MiB, FV_SCALE=16 — reproduced
//! here by resizing the transposition table to 1024 MiB, clearing it per
//! fixture (the `usinewgame` equivalent), and letting `run_root` bump the
//! generation (the `go` equivalent).
//!
//! Like the other real-network tests, this is skipped with a notice when
//! `nn.bin` is absent (a checkout without it staged), so the default
//! `cargo test` run stays green everywhere.

use std::path::PathBuf;

use attic_search::{QSearch, RootKind};
use attic_state::{Move, Position, format_usi_move, parse_sfen, parse_usi_move};
use attic_storage::TranspositionTable;
use serde::Deserialize;

/// `VALUE_MATE` (`types.h`).
const VALUE_MATE: i32 = 32000;
/// `VALUE_TB_WIN_IN_MAX_PLY` (`types.h`): the `is_decisive` threshold.
const VALUE_TB_WIN_IN_MAX_PLY: i32 = VALUE_MATE - 246;
/// `Eval::PawnValue` (`NormalizeToPawnValue`, `usi.cpp`).
const PAWN_VALUE: i32 = 90;
/// Engine default `USI_Hash` in MiB (`tests/fixtures/search-depth1/README.md`).
const HASH_MB: usize = 1024;

const FIXTURES: &[&str] = &[
    "startpos.json",
    "drop-heavy.json",
    "mid-game-tactical.json",
    "check-evasion.json",
    "promotion-zone-edges.json",
    "sennichite.json",
];

#[derive(Debug, Deserialize)]
struct Fixture {
    sfen: String,
    /// Optional USI moves applied after the SFEN (USI `position ... moves ...`).
    #[serde(default)]
    moves: Vec<String>,
    depth: i32,
    bestmove: String,
    score: ScoreJson,
    nodes: u64,
    /// The principal variation (desirable but not gated).
    #[serde(default)]
    pv: Vec<String>,
}

/// Fixture score: exactly one of `cp` or `mate` is present.
#[derive(Debug, Deserialize)]
struct ScoreJson {
    #[serde(default)]
    cp: Option<i32>,
    #[serde(default)]
    mate: Option<i32>,
}

fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

fn load_fixture(name: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/search-depth1")
        .join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"))
}

/// Parse the SFEN and apply the optional `moves` prefix, mirroring USI
/// `position sfen <SFEN> moves <m1> <m2> ...`.
fn setup(fixture: &Fixture) -> Position {
    let mut pos = parse_sfen(&fixture.sfen).expect("valid fixture SFEN");
    for usi in &fixture.moves {
        let m = parse_usi_move(usi, &pos).unwrap_or_else(|e| panic!("bad move {usi}: {e:?}"));
        pos.do_move(m);
    }
    pos
}

/// The USI-string form of the outcome's bestmove (fixtures use ordinary moves;
/// the resign / win sentinels never occur for the six fixtures).
fn bestmove_usi(best_move: Move, kind: RootKind) -> String {
    match kind {
        RootKind::Resign => "resign".to_string(),
        RootKind::DeclarationWin => "win".to_string(),
        RootKind::Normal => format_usi_move(best_move),
    }
}

/// `is_decisive` (`types.h`).
fn is_decisive(v: i32) -> bool {
    v.abs() >= VALUE_TB_WIN_IN_MAX_PLY
}

/// Format a search value the way the reference USI layer does (`score.cpp` /
/// `usi.cpp` `format_score`): a mate distance for decisive scores, else
/// `100 * v / PawnValue` centipawns (C++ truncating division).
fn format_score(v: i32) -> ScoreJson {
    if is_decisive(v) {
        let distance = VALUE_MATE - v.abs();
        ScoreJson {
            cp: None,
            mate: Some(if v > 0 { distance } else { -distance }),
        }
    } else {
        ScoreJson {
            cp: Some(100 * v / PAWN_VALUE),
            mate: None,
        }
    }
}

fn assert_fixture(name: &str, net: &attic_eval::NnueNetwork, tt: &mut TranspositionTable) {
    let fixture = load_fixture(name);
    assert_eq!(fixture.depth, 1, "{name}: depth-1 fixtures only");

    // usinewgame: clear the table (also resets the generation to 0).
    tt.clear();
    let pos = setup(&fixture);

    let outcome = {
        let mut search = QSearch::new(net, tt);
        search.run_root(&pos, fixture.depth)
    };

    // bestmove.
    let got_best = bestmove_usi(outcome.best_move, outcome.kind);
    assert_eq!(
        got_best, fixture.bestmove,
        "{name}: bestmove mismatch (got {got_best}, want {})",
        fixture.bestmove
    );

    // score (cp or mate).
    let got_score = format_score(outcome.score);
    assert_eq!(
        got_score.cp, fixture.score.cp,
        "{name}: score cp mismatch (raw value {})",
        outcome.score
    );
    assert_eq!(
        got_score.mate, fixture.score.mate,
        "{name}: score mate mismatch (raw value {})",
        outcome.score
    );

    // nodes.
    assert_eq!(
        outcome.nodes, fixture.nodes,
        "{name}: node count mismatch (got {}, want {})",
        outcome.nodes, fixture.nodes
    );

    // pv is desirable but not gated; surface a divergence as a notice only.
    let got_pv: Vec<String> = outcome.pv.iter().map(|&m| format_usi_move(m)).collect();
    if got_pv != fixture.pv {
        eprintln!(
            "{name}: pv differs (got {got_pv:?}, want {:?}) — not gated",
            fixture.pv
        );
    }
}

#[test]
fn depth1_search_matches_reference_fixtures() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping depth1_search_matches_reference_fixtures: {} is not present (staged only on the dev VM)",
            path.display()
        );
        return;
    }

    let net = attic_eval::load_network(&path).expect("real nn.bin should load and validate");

    // One 1024 MiB table, cleared per fixture (the usinewgame equivalent).
    let mut tt = TranspositionTable::new();
    tt.resize(HASH_MB);

    for name in FIXTURES {
        assert_fixture(name, &net, &mut tt);
    }
}
