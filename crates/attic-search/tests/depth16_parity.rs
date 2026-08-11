//! Depth-16 search parity gate — the null-move **verification search**.
//!
//! Runs the `go depth 16` root search ([`QSearch::run_root`]) against the
//! reference-captured fixture under `tests/fixtures/search-depth16/` and asserts
//! **bestmove, score, and nodes** exactly, as an inseparable triple (the
//! `(nodes & 14)` root tie-break means a single node of drift can flip the
//! bestmove).
//!
//! ## Why depth 16 specifically
//!
//! Step 9's verification search (`yaneuraou-search.cpp` at the pin
//! `76d58ef`) is the only regime the depth-1/2/3/5/8 tiers cannot reach: its
//! guard is `nmpMinPly == 0 && depth >= 16`, so a null-move fail-high below
//! depth 16 returns `nullValue` outright and the whole block is dead. From
//! depth 16 up, a fail-high instead re-searches the **same node** (same `ss`, no
//! `do_move`) at `depth - R` with null-move pruning disabled until `ss->ply`
//! climbs past `nmpMinPly = ss->ply + 3 * (depth - R) / 4`, and only returns
//! `nullValue` when that verification also fails high — otherwise the node falls
//! through to its ordinary moves loop.
//!
//! That re-entry is also what makes the tier worth gating rather than merely
//! running: because it re-enters on this node's own stack cell, it rewrites
//! `ss->staticEval` (Step 6 runs again with a correction value the null-move
//! subtree may have moved) and can flip `ss->ttPv` (a fail-low re-entry applies
//! `ss->ttPv |= (ss-1)->ttPv`). Every reference read of those two fields after
//! Step 9 is a live `ss->` read, so `qsearch.rs` re-syncs its locals right after
//! the block. A port that cached them across Step 9 passes every shallower tier
//! and diverges only here.
//!
//! One position keeps the tier affordable: `startpos` at depth 16 is ~230k
//! cumulative nodes, roughly 20x the depth-8 fixture. The six-position sweep
//! stays at depth 8.
//!
//! The fixture was captured with Threads=1, no book, `usinewgame` before the
//! position, `go depth 16`, USI_Hash default 1024 MiB — reproduced here by
//! resizing the transposition table to 1024 MiB, clearing it (the `usinewgame`
//! equivalent), and letting `run_root` bump the generation.
//!
//! Like the other real-network tests, this is skipped with a notice when
//! `nn.bin` is absent (a checkout without it staged), so the default
//! `cargo test` run stays green everywhere `nn.bin` is not staged.

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
/// Engine default `USI_Hash` in MiB (`tests/fixtures/search-depth16/README.md`).
const HASH_MB: usize = 1024;

#[derive(Debug, Deserialize)]
struct FixtureJson {
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

/// The USI-string form of the outcome's bestmove (the fixture is an ordinary
/// move; the resign / win sentinels never occur for it).
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

#[test]
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

    // usinewgame: clear the table (also resets the generation to 0).
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
