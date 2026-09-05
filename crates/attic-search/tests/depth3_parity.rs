//! Depth-3 search parity test. `nodes` is cumulative over the whole `go`.
//!
//! ## `sennichite`: game-history repetition
//!
//! That fixture's 12-ply `moves` prefix walks both kings back and forth, so the
//! search root has already occurred earlier in the **game history**. Detecting
//! a forced fourfold whose earlier occurrences lie before the search root needs
//! the `pliesFromNull` and incremental-repetition machinery this port carries;
//! a ply-limited check would show up as exactly a three-node surplus.
//!
//! Under the default `quick-draw` configuration those three nodes are pruned
//! for a different reason — that walk adjudicates at the *second* occurrence and
//! likewise ignores the search ply. All six fixtures here match in both
//! configurations; `tests/quick_draw_parity.rs` is where the two disagree.

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
/// Engine default `USI_Hash` in MiB (`tests/fixtures/search/README.md`).
const HASH_MB: usize = 1024;

/// One fixture, plus whether its cumulative node count is gated hard. All six
/// currently are; the flag lets a future fixture opt into a soft check.
struct Fixture {
    name: &'static str,
    gate_nodes: bool,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "startpos.json",
        gate_nodes: true,
    },
    Fixture {
        name: "drop-heavy.json",
        gate_nodes: true,
    },
    Fixture {
        name: "mid-game-tactical.json",
        gate_nodes: true,
    },
    Fixture {
        name: "check-evasion.json",
        gate_nodes: true,
    },
    Fixture {
        name: "promotion-zone-edges.json",
        gate_nodes: true,
    },
    Fixture {
        name: "sennichite.json",
        gate_nodes: true,
    },
];

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
        .join("../../tests/fixtures/search")
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

fn assert_fixture(fixture: &Fixture, net: &attic_eval::NnueNetwork, tt: &mut TranspositionTable) {
    let name = fixture.name;
    let json = load_fixture(name);
    assert_eq!(json.depth, 3, "{name}: depth-3 fixtures only");

    // The `usinewgame` equivalent, which also resets the generation.
    tt.clear();
    let pos = setup(&json);

    let outcome = {
        let mut search = QSearch::new(net, tt);
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

    if fixture.gate_nodes {
        assert_eq!(
            outcome.nodes, json.nodes,
            "{name}: node count mismatch (got {}, want {})",
            outcome.nodes, json.nodes
        );
    } else if outcome.nodes != json.nodes {
        eprintln!(
            "{name}: node count {} != reference {} — expected, pending the deferred \
             pliesFromNull / game-history repetition rework (bestmove and score match)",
            outcome.nodes, json.nodes
        );
    }

    // The PV is desirable but not gated.
    let got_pv: Vec<String> = outcome.pv.iter().map(|&m| format_usi_move(m)).collect();
    if got_pv != json.pv {
        eprintln!(
            "{name}: pv differs (got {got_pv:?}, want {:?}) — not gated",
            json.pv
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn depth3_search_matches_reference_fixtures() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping depth3_search_matches_reference_fixtures: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }

    let net = attic_eval::load_network(&path).expect("real nn.bin should load and validate");

    // One table, cleared per fixture.
    let mut tt = TranspositionTable::new();
    tt.resize(HASH_MB);

    for fixture in FIXTURES {
        assert_fixture(fixture, &net, &mut tt);
    }
}
