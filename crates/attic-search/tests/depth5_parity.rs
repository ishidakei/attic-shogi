//! Depth-5 search parity gate.
//!
//! Runs the `go depth 5` root search ([`QSearch::run_root`]) — real iterative
//! deepening (iterations 1..5), real aspiration windows, and the root move loop
//! recursing into the ported interior `search<PV/NonPV>` — against the six
//! reference-captured fixtures under `tests/fixtures/search-depth5/` and asserts
//! **bestmove, score, and nodes** exactly for all six, as one inseparable set.
//!
//! `nodes` is cumulative over the whole `go` (every iteration 1..5 and every
//! aspiration re-search), and the `(nodes & 14)` root tie-break means a single
//! node of drift can flip the bestmove — so all three are asserted together.
//! The cumulative gate transitively pins the depth-1..4 iterations too.
//!
//! ## The singular-extension family is live at depth 5
//!
//! The singular guard `!rootNode && depth >= 6 + ss->ttPv` looks unreachable
//! from a depth-5 root, but an LMR re-search deepening chain does reach it:
//! `startpos` at interior `ply 2, depth 6, ttPv=false` and `sennichite` at
//! `ply 1, depth 6/7` and `ply 2, depth 6`. A port without the singular logic
//! therefore diverges here, not only at the depth-8 tier.
//!
//! Depth 5 also activates the ProbCut reduced search (`prob_cut_depth == 1`) and
//! the deeper null-move / futility / LMR regimes; `sennichite` additionally
//! exercises the game-history repetition detection this port carries (the
//! `pliesFromNull` / incremental `Position::is_repetition` chain that spans the
//! `moves` prefix before the search root).
//!
//! The fixtures were captured with Threads=1, no book, `usinewgame` before each
//! position, `go depth 5`, USI_Hash default 1024 MiB, FV_SCALE 16 — reproduced
//! here by resizing the transposition table to 1024 MiB, clearing it per fixture
//! (the `usinewgame` equivalent), and letting `run_root` bump the generation.
//!
//! Like the other real-network tests, this is skipped with a notice when
//! `nn.bin` is absent, so the default `cargo test` run stays green everywhere
//! `nn.bin` is not staged.

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
/// Engine default `USI_Hash` in MiB (`tests/fixtures/search-depth5/README.md`).
const HASH_MB: usize = 1024;

/// The six fixtures. Every one gates `bestmove` / `score` / `nodes` hard.
const FIXTURES: &[&str] = &[
    "startpos.json",
    "drop-heavy.json",
    "mid-game-tactical.json",
    "check-evasion.json",
    "promotion-zone-edges.json",
    "sennichite.json",
];

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
        .join("../../tests/fixtures/search-depth5")
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
    let json = load_fixture(name);
    assert_eq!(json.depth, 5, "{name}: depth-5 fixtures only");

    // usinewgame: clear the table (also resets the generation to 0).
    tt.clear();
    let pos = setup(&json);

    let outcome = {
        let mut search = QSearch::new(net, tt);
        search.run_root(&pos, json.depth)
    };

    // bestmove — gated for every fixture.
    let got_best = bestmove_usi(outcome.best_move, outcome.kind);
    assert_eq!(
        got_best, json.bestmove,
        "{name}: bestmove mismatch (got {got_best}, want {})",
        json.bestmove
    );

    // score (cp or mate) — gated for every fixture.
    let got_score = format_score(outcome.score);
    assert_eq!(
        got_score, json.score,
        "{name}: score mismatch (raw value {})",
        outcome.score
    );

    // nodes — gated hard for all six (cumulative over the whole `go`).
    assert_eq!(
        outcome.nodes, json.nodes,
        "{name}: node count mismatch (got {}, want {})",
        outcome.nodes, json.nodes
    );

    // pv is desirable but not gated; surface a divergence as a notice only.
    let got_pv: Vec<String> = outcome.pv.iter().map(|&m| format_usi_move(m)).collect();
    if got_pv != json.pv {
        eprintln!(
            "{name}: pv differs (got {got_pv:?}, want {:?}) — not gated",
            json.pv
        );
    }
}

/// All six fixtures gate `bestmove` / `score` / `nodes` hard, as one
/// inseparable set. Skipped with a notice when `nn.bin` is absent.
#[test]
fn depth5_search_matches_reference_fixtures() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping depth5_search_matches_reference_fixtures: {} is not present (staged only on the dev VM)",
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
