//! Session-level depth-1 parity gate (blocking) against the real network.
//!
//! Drives a full USI session in-process — `usi` → `setoption EvalDir` →
//! `isready` → `position startpos` → `go depth 1` — and asserts the emitted
//! `bestmove`, `score`, and `nodes` exactly match the reference-captured
//! `tests/fixtures/search-depth1/startpos.json`. This proves the driver drives
//! the ported depth-1 root search ([`attic_search::QSearch::run_root`]) with
//! the reference TT sizing (1024 MiB, allocated on the first successful
//! `isready`) and generation semantics (`run_root` bumps it per `go`).
//!
//! The three fields are one inseparable set: the `(nodes & 14)` root tie-break
//! means a single-node drift can cascade into a different score and a flipped
//! bestmove, so a mismatch on any of them signals a search divergence.
//!
//! The SFNN-1536 network is staged locally at
//! `eval/nn.bin` and is never committed. When absent
//! (a checkout without it staged) the test prints a notice and passes, so the
//! default `cargo test` run stays green everywhere.

use std::path::PathBuf;

use attic_protocol::UsiDriver;
use serde::Deserialize;

fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/search-depth1/startpos.json")
}

/// The gated subset of a depth-1 fixture (see `depth1_parity.rs`).
#[derive(Debug, Deserialize)]
struct Fixture {
    bestmove: String,
    score: ScoreJson,
    nodes: u64,
}

/// Fixture score: exactly one of `cp` or `mate` is present.
#[derive(Debug, Deserialize)]
struct ScoreJson {
    #[serde(default)]
    cp: Option<i32>,
    #[serde(default)]
    mate: Option<i32>,
}

fn drive(input: &str) -> String {
    let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), std::sync::Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// Extract the value following `key` in a whitespace-tokenised `info` line, e.g.
/// `field_after("... nodes 30 pv ...", "nodes") == Some("30")`.
fn field_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mut it = line.split_whitespace();
    while let Some(tok) = it.next() {
        if tok == key {
            return it.next();
        }
    }
    None
}

#[test]
fn depth1_session_matches_reference_startpos_fixture() {
    let dir = eval_dir();
    if !dir.join("nn.bin").exists() {
        eprintln!(
            "skipping depth1_session_matches_reference_startpos_fixture: {} is not present (staged only on the dev VM)",
            dir.join("nn.bin").display()
        );
        return;
    }

    let raw = std::fs::read_to_string(fixture_path()).expect("read startpos fixture");
    let fixture: Fixture = serde_json::from_str(&raw).expect("parse startpos fixture");

    let eval_dir_arg = dir.to_str().expect("utf-8 eval dir");
    // `Threads value 1` pins the single-worker search: the default is 4, and once
    // helpers really search they pollute the shared TT, so any
    // fixture-node assertion must run on one worker to stay deterministic.
    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {eval_dir_arg}\n\
         isready\n\
         position startpos\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive(&session);

    assert!(
        out.contains("readyok\n"),
        "real network must load (readyok), got:\n{out}"
    );

    // bestmove (no ponder expected for the single-move startpos PV, but tolerate
    // one by comparing only the move token).
    let bestmove_line = out
        .lines()
        .find(|l| l.starts_with("bestmove "))
        .unwrap_or_else(|| panic!("missing bestmove in:\n{out}"));
    let got_best = bestmove_line
        .strip_prefix("bestmove ")
        .unwrap()
        .split_whitespace()
        .next()
        .expect("bestmove token");
    assert_eq!(got_best, fixture.bestmove, "bestmove mismatch in:\n{out}");

    // The single depth-1 info line carries the score and node count.
    let info_line = out
        .lines()
        .find(|l| l.starts_with("info depth 1 "))
        .unwrap_or_else(|| panic!("missing depth-1 info line in:\n{out}"));

    let got_nodes: u64 = field_after(info_line, "nodes")
        .expect("nodes field")
        .parse()
        .expect("nodes integer");
    assert_eq!(got_nodes, fixture.nodes, "node count mismatch in:\n{out}");

    // score is `cp <X>` or `mate <Y>`; compare against whichever the fixture has.
    let score_kind = field_after(info_line, "score").expect("score kind");
    let score_val: i32 = field_after(info_line, score_kind)
        .expect("score value")
        .parse()
        .expect("score integer");
    match (fixture.score.cp, fixture.score.mate) {
        (Some(cp), None) => {
            assert_eq!(score_kind, "cp", "expected a cp score in:\n{out}");
            assert_eq!(score_val, cp, "score cp mismatch in:\n{out}");
        }
        (None, Some(mate)) => {
            assert_eq!(score_kind, "mate", "expected a mate score in:\n{out}");
            assert_eq!(score_val, mate, "score mate mismatch in:\n{out}");
        }
        other => panic!("fixture score must be exactly one of cp/mate, got {other:?}"),
    }
}
