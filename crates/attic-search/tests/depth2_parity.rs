//! Depth-2 search parity gate — the pawn-history-aliasing regression.
//!
//! Runs the `go depth 2` root search ([`QSearch::run_root`]) against a single
//! reference-captured fixture — `position startpos moves 7g7f` — and asserts
//! **bestmove, score, and nodes** exactly.
//!
//! This is the minimal position at which Zobrist-table aliasing is observable:
//! a one-node divergence (1752 against the reference's 1753) that the
//! `(nodes & 14)` root tie-break amplifies into a flipped bestmove by depth 8.
//! The pawn history (`pawnHistory`, 8192 planes) and the correction histories
//! are hash tables indexed by `pawn_key & (size - 1)`, so their collision
//! structure — and therefore the quiet move ordering they drive — depends on the
//! concrete key values, not just on the key *structure*. A privately seeded
//! table aliases differently from the reference and flips a quiet's ordering on
//! the first colliding pawn structure, which cascades through PVS re-search
//! bounds into the node count. `crates/attic-state/src/key.rs` reproduces the
//! reference's Zobrist bit-for-bit so the aliasing matches; this fixture is what
//! catches a regression in that.
//!
//! Captured with Threads=1, no book, `usinewgame`, `go depth 2`, USI_Hash 1024
//! MiB, FV_SCALE 16 — reproduced here by resizing the transposition table to
//! 1024 MiB and clearing it (the `usinewgame` equivalent). Skipped with a notice
//! when `nn.bin` is absent, like the other real-network tests.

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
/// Engine default `USI_Hash` in MiB.
const HASH_MB: usize = 1024;

#[derive(Debug, Deserialize)]
struct FixtureJson {
    sfen: String,
    #[serde(default)]
    moves: Vec<String>,
    depth: i32,
    bestmove: String,
    score: ScoreJson,
    nodes: u64,
    #[serde(default)]
    pv: Vec<String>,
}

/// Fixture score: exactly one of `cp` or `mate` is present.
#[derive(Debug, Deserialize, PartialEq)]
struct ScoreJson {
    #[serde(default)]
    cp: Option<i32>,
    #[serde(default)]
    mate: Option<i32>,
}

fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

fn load_fixture(name: &str) -> FixtureJson {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/search-depth2")
        .join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"))
}

/// Parse the SFEN and apply the optional `moves` prefix, mirroring USI
/// `position sfen <SFEN> moves <m1> <m2> ...`.
fn setup(fixture: &FixtureJson) -> Position {
    let mut pos = parse_sfen(&fixture.sfen).expect("valid fixture SFEN");
    for usi in &fixture.moves {
        let m = parse_usi_move(usi, &pos).unwrap_or_else(|e| panic!("bad move {usi}: {e:?}"));
        pos.do_move(m);
    }
    pos
}

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

/// Format a search value the way the reference USI layer does.
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

/// `position startpos moves 7g7f`, `go depth 2`: bestmove / score / nodes exact.
#[test]
#[cfg_attr(miri, ignore)]
fn depth2_startpos_7g7f_matches_reference() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping depth2_startpos_7g7f_matches_reference: {} is not present (staged only on the dev VM)",
            path.display()
        );
        return;
    }

    let net = attic_eval::load_network(&path).expect("real nn.bin should load and validate");
    let mut tt = TranspositionTable::new();
    tt.resize(HASH_MB);
    tt.clear();

    let json = load_fixture("startpos-7g7f.json");
    assert_eq!(json.depth, 2, "depth-2 fixture only");
    let pos = setup(&json);

    let outcome = {
        let mut search = QSearch::new(&net, &tt);
        search.run_root(&pos, json.depth)
    };

    let got_best = bestmove_usi(outcome.best_move, outcome.kind);
    assert_eq!(
        got_best, json.bestmove,
        "bestmove mismatch (got {got_best}, want {})",
        json.bestmove
    );

    let got_score = format_score(outcome.score);
    assert_eq!(
        got_score, json.score,
        "score mismatch (raw value {})",
        outcome.score
    );

    assert_eq!(
        outcome.nodes, json.nodes,
        "node count mismatch (got {}, want {})",
        outcome.nodes, json.nodes
    );

    let got_pv: Vec<String> = outcome.pv.iter().map(|&m| format_usi_move(m)).collect();
    if got_pv != json.pv {
        eprintln!(
            "pv differs (got {got_pv:?}, want {:?}) — not gated",
            json.pv
        );
    }
}
