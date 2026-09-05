//! Depth-8 search parity test. `nodes` is cumulative over the whole `go`, so it
//! transitively pins the depth-1..7 iterations too.
//!
//! Depth 8 is the first depth at which the singular guard
//! `!rootNode && depth >= 6 + ss->ttPv` fires directly at interior nodes rather
//! than only through LMR re-search deepening, so it exercises the singular
//! family — multi-cut pruning, the margin extensions, the negative extensions —
//! end to end. It also newly reaches internal iterative reduction and the
//! `depth > 5` disjunct of the non-PV early TT cutoff.
//!
//! ## What `startpos` at depth 8 is sensitive to
//!
//! Two whole-engine invariants show up here and nowhere shallower:
//!
//! 1. **Zobrist aliasing**, observable from depth 2, where the minimal repro is
//!    gated. The hash-indexed pawn and correction histories alias by concrete
//!    key value, so a privately seeded Zobrist table flips a quiet's ordering on
//!    the first colliding pawn structure — a single node there, which the root
//!    tie-break amplifies into a flipped bestmove here.
//! 2. **Continuation planes in qsearch**, observable from depth 6. The reference
//!    sets them inside `do_move` for **every** move, qsearch moves included, and
//!    a deeper node reads the continuation-correction plane at `(ss-2)` /
//!    `(ss-4)` — so leaving them unset at a qsearch ply shifts a descendant's
//!    corrected eval by ~1 and flips a shallow prune once the tables warm up.

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
/// Engine default `USI_Hash` in MiB (`tests/fixtures/search-depth8/README.md`).
const HASH_MB: usize = 1024;

/// The six fixtures, each gating `bestmove` / `score` / `nodes` hard.
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
        .join("../../tests/fixtures/search-depth8")
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

fn assert_fixture(name: &str, net: &attic_eval::NnueNetwork, tt: &mut TranspositionTable) {
    let json = load_fixture(name);
    assert_eq!(json.depth, 8, "{name}: depth-8 fixtures only");

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

    // Cumulative over the whole `go`.
    assert_eq!(
        outcome.nodes, json.nodes,
        "{name}: node count mismatch (got {}, want {})",
        outcome.nodes, json.nodes
    );

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
fn depth8_search_matches_reference_fixtures() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping depth8_search_matches_reference_fixtures: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }

    let net = attic_eval::load_network(&path).expect("real nn.bin should load and validate");

    // One table, cleared per fixture.
    let mut tt = TranspositionTable::new();
    tt.resize(HASH_MB);

    for name in FIXTURES {
        assert_fixture(name, &net, &mut tt);
    }
}
