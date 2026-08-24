//! Parity test for perft: assert our `perft` is byte-equal to the reference's
//! captured node counts at every depth in each fixture under
//! `tests/fixtures/perft/`.
//!
//! Default `cargo test` runs every (fixture, depth) pair where
//! `depth <= DEFAULT_DEPTH_LIMIT`. Depths above the limit (currently startpos
//! 4 and 5) are gated behind `#[ignore]` because debug-profile perft over
//! them is too slow to belong on the default gate; CI invokes them via
//! `cargo test --release -- --include-ignored`.

use std::path::PathBuf;

use attic::perft::perft;
use attic_state::{parse_sfen, parse_usi_move};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    sfen: String,
    /// Optional USI moves to apply after the SFEN before running perft. Used
    /// by fixtures whose perft tree depends on position history (e.g.
    /// `sennichite.json`, where the prefix sets up an N-fold repetition
    /// state). Defaults to empty, so a fixture without a prefix parses.
    #[serde(default)]
    moves: Vec<String>,
    results: Vec<DepthResult>,
}

#[derive(Debug, Deserialize)]
struct DepthResult {
    depth: u32,
    expected_nodes: u64,
}

const FIXTURES: &[&str] = &[
    "startpos.json",
    "drop-heavy.json",
    "mid-game-tactical.json",
    "check-evasion.json",
    "promotion-zone-edges.json",
    "sennichite.json",
];

const DEFAULT_DEPTH_LIMIT: u32 = 3;

fn load_fixture(name: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/perft")
        .join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"))
}

fn assert_fixture_depth(name: &str, depth: u32) {
    let fixture = load_fixture(name);
    let entry = fixture
        .results
        .iter()
        .find(|r| r.depth == depth)
        .unwrap_or_else(|| panic!("fixture {name} has no entry for depth {depth}"));
    let mut pos = parse_sfen(&fixture.sfen).expect("fixture sfen parses");
    for m in &fixture.moves {
        let parsed = parse_usi_move(m, &pos)
            .unwrap_or_else(|e| panic!("fixture {name} prefix move {m:?}: {e}"));
        pos.do_move(parsed);
    }
    let actual = perft(&mut pos, depth);
    assert_eq!(
        actual, entry.expected_nodes,
        "perft({name}, {depth}) mismatch: ours = {actual}, reference = {}",
        entry.expected_nodes
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn perft_parity_default_depths() {
    for name in FIXTURES {
        let fixture = load_fixture(name);
        for r in &fixture.results {
            if r.depth > DEFAULT_DEPTH_LIMIT {
                continue;
            }
            assert_fixture_depth(name, r.depth);
        }
    }
}

#[test]
#[ignore = "slow under debug; run with `cargo test --release -- --include-ignored`"]
fn perft_startpos_depth_4() {
    assert_fixture_depth("startpos.json", 4);
}

#[test]
#[ignore = "slow under debug; run with `cargo test --release -- --include-ignored`"]
fn perft_startpos_depth_5() {
    assert_fixture_depth("startpos.json", 5);
}
