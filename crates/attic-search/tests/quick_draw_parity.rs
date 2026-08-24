//! QUICK_DRAW repetition-semantics parity gate (blocking, `quick-draw` only).
//!
//! The other `depth*_parity` suites gate positions where the two repetition
//! configurations happen to agree. This one gates the position where they do
//! **not**: bench position 3, whose search reaches a line that repeats the root
//! board with the side to move's hand strictly poorer.
//!
//! ```text
//! 6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1
//! ```
//!
//! Along `S*5c K5bx5c G*5b K5cx5b` the position at ply 4 repeats the root board
//! with Black's hand strictly poorer. Under `ENABLE_QUICK_DRAW`
//! (`position.cpp`, which the reference's `FOR_TOURNAMENT` build
//! compiles) that adjudicates `REPETITION_INFERIOR` immediately: there is no
//! `st->repetition < ply` root gate, so a recurrence landing exactly *on* the
//! search root still counts. The non-QUICK_DRAW gate evaluates `4 < 4` and
//! searches on, which is worth 11 extra nodes at depth 3 (3,924 vs the
//! reference's 3,913) and cascades through the transposition table from there.
//!
//! Both fixtures were captured from the **tournament** reference build
//! (`cargo xtask build-reference`) with Threads=1, no book, `usinewgame` before
//! the position, and — unlike the other search fixtures — an explicit
//! `USI_Hash 256`, recorded in the fixture as `hash_mb`. Depth 3 is the minimal
//! depth that reaches the diverging line; depth 8 pins that the divergence stays
//! closed once the transposition table and the history tables are warm.
//!
//! This suite exists only in the `quick-draw` configuration. In the
//! non-QUICK_DRAW one the engine is deliberately a different engine here, and
//! the numbers below would (correctly) not match.
//!
//! Like the other real-network tests, it is skipped with a notice when `nn.bin`
//! is absent, so the default `cargo test` run stays green everywhere.

#![cfg(feature = "quick-draw")]

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

/// Both fixtures gate `bestmove` / `score` / `nodes` hard.
const FIXTURES: &[&str] = &["bench-pos3-depth3.json", "bench-pos3-depth8.json"];

#[derive(Debug, Deserialize)]
struct FixtureJson {
    sfen: String,
    /// Optional USI moves applied after the SFEN (USI `position ... moves ...`).
    #[serde(default)]
    moves: Vec<String>,
    depth: i32,
    /// `USI_Hash` in MiB the fixture was captured with. Required here: the whole
    /// point of the suite is an exact `nodes` comparison, and the table size
    /// changes it.
    hash_mb: usize,
    /// `FV_SCALE` the reference searched with, recorded by `capture-search` from
    /// the engine's own `info string engine option override` line. Absent means
    /// the engine default. The reference reads this from the local, uncommitted
    /// `eval/eval_options.txt`, so it is not implied
    /// by the submodule pin and has to travel with the fixture.
    #[serde(default)]
    fv_scale: Option<i32>,
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
        .join("../../tests/fixtures/search-quick-draw")
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

fn assert_fixture(name: &str, net: &attic_eval::NnueNetwork) {
    let json = load_fixture(name);

    // Match the divisor the capture ran with. This is process-global state, but
    // each integration test file is its own binary and this one holds a single
    // test, so there is nothing to race with.
    attic_eval::set_fv_scale(json.fv_scale.unwrap_or(attic_eval::FV_SCALE_DEFAULT));

    // A fresh table per fixture at the captured size — the fixture's fresh
    // process plus `usinewgame`.
    let mut tt = TranspositionTable::new();
    tt.resize(json.hash_mb);

    let pos = setup(&json);
    let outcome = {
        let mut search = QSearch::new(net, &tt);
        search.run_root(&pos, json.depth)
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

    // pv is desirable but not gated; surface a divergence as a notice only.
    let got_pv: Vec<String> = outcome.pv.iter().map(|&m| format_usi_move(m)).collect();
    if got_pv != json.pv {
        eprintln!(
            "{name}: pv differs (got {got_pv:?}, want {:?}) — not gated",
            json.pv
        );
    }
}

/// Both fixtures gate `bestmove` / `score` / `nodes` hard, each an inseparable
/// triple. Skipped with a notice when `nn.bin` is absent.
#[test]
#[cfg_attr(miri, ignore)]
fn quick_draw_search_matches_tournament_reference_fixtures() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping quick_draw_search_matches_tournament_reference_fixtures: {} is not present (staged only on the dev VM)",
            path.display()
        );
        return;
    }

    let net = attic_eval::load_network(&path).expect("real nn.bin should load and validate");

    for name in FIXTURES {
        assert_fixture(name, &net);
    }
}
