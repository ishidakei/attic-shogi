//! Depth-16 search parity test — the null-move **verification search**.
//!
//! Step 9's verification search (`yaneuraou-search.cpp`) is the only
//! regime the shallower tiers cannot reach: its guard is
//! `nmpMinPly == 0 && depth >= 16`, so below that a null-move fail-high returns
//! `nullValue` outright and the whole block is dead. From depth 16 up, a
//! fail-high instead re-searches the **same node** at a reduced depth with
//! null-move pruning disabled for a while, and only returns `nullValue` when
//! that verification also fails high.
//!
//! That re-entry is what makes the tier worth gating rather than merely running.
//! Re-entering on this node's own stack cell rewrites `ss->staticEval` and can
//! flip `ss->ttPv`, and every reference read of those two after Step 9 is a live
//! one — so a port that cached them across Step 9 passes every shallower tier
//! and diverges only here.
//!
//! One position keeps the tier affordable: at depth 16 `startpos` is roughly 20
//! times the depth-8 fixture's node count.

use std::path::PathBuf;

use attic_search::{QSearch, RootKind};
use attic_state::{Move, Position, format_usi_move, parse_sfen, parse_usi_move};
use attic_storage::TranspositionTable;
use serde::Deserialize;

/// `VALUE_MATE` (`types.h`).
const VALUE_MATE: i32 = 32000;
/// The `is_decisive` threshold (`types.h`).
const VALUE_TB_WIN_IN_MAX_PLY: i32 = VALUE_MATE - 246;
/// `Eval::PawnValue` (`usi.cpp`).
const PAWN_VALUE: i32 = 90;
/// Engine default `USI_Hash` in MiB (`tests/fixtures/search-depth16/README.md`).
const HASH_MB: usize = 1024;

#[derive(Debug, Deserialize)]
struct FixtureJson {
    sfen: String,
    /// Optional USI moves applied after the SFEN.
    #[serde(default)]
    moves: Vec<String>,
    depth: i32,
    bestmove: String,
    score: ScoreJson,
    nodes: u64,
    /// The principal variation (desirable but not gated).
    #[serde(default)]
    #[allow(dead_code)]
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
        .join("../../tests/fixtures/search-depth16")
        .join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"))
}

/// Parse the SFEN and apply the optional `moves` prefix.
fn setup(fixture: &FixtureJson) -> Position {
    let mut pos = parse_sfen(&fixture.sfen).expect("valid fixture SFEN");
    for usi in &fixture.moves {
        let m = parse_usi_move(usi, &pos).unwrap_or_else(|e| panic!("bad move {usi}: {e:?}"));
        pos.do_move(m);
    }
    pos
}

/// The USI-string form of the outcome's bestmove.
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

/// Format a search value as the reference USI layer does (`format_score`,
/// `usi.cpp`): a mate distance for a decisive score, else centipawns.
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

#[test]
#[cfg_attr(miri, ignore)]
fn depth16_search_matches_reference_fixture() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping: {} not present (obtained out-of-band)",
            path.display()
        );
        return;
    }
    let net = attic_eval::load_network(&path).expect("real nn.bin should load and validate");

    let mut tt = TranspositionTable::new();
    tt.resize(HASH_MB);

    let name = "startpos.json";
    let json = load_fixture(name);
    assert_eq!(json.depth, 16, "{name}: depth-16 fixtures only");

    // The `usinewgame` equivalent, which also resets the generation.
    tt.clear();
    let pos = setup(&json);

    let outcome = {
        let mut qs = QSearch::new(&net, &tt);
        qs.run_root(&pos, json.depth)
    };

    let got_best = bestmove_usi(outcome.best_move, outcome.kind);
    assert_eq!(
        got_best, json.bestmove,
        "{name}: bestmove mismatch (got {got_best}, want {})",
        json.bestmove
    );

    let got_score = format_score(outcome.score);
    assert_eq!(
        got_score, json.score,
        "{name}: score mismatch (raw value {})",
        outcome.score
    );

    assert_eq!(
        outcome.nodes, json.nodes,
        "{name}: node count mismatch (got {}, want {})",
        outcome.nodes, json.nodes
    );
}
