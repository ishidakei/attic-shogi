//! Driver-level session tests for the MultiPV + PV-output group: the real
//! MultiPV loop, the `PvInterval` throttle, `ConsiderationMode`, and
//! the voting-off-under-MultiPV path.
//!
//! Each test drives a full `usi → setoption → isready → position → go` session
//! in-process against a synthetic (all-zero) network staged in a temp dir, so
//! they are hermetic. They use [`StreamHarness`] and wait for the `bestmove`
//! before quitting: a MultiPV search runs many root searches per iteration, so
//! quitting early would abort it mid-iteration (`quit` sets the stop flag). All
//! PV-content assertions that depend on per-iteration output set `PvInterval
//! value 0` in the preamble so the pin's default 300 ms throttle does not
//! suppress the intermediate lines.

mod common;

use attic_state::parse_usi_move;
use common::{StreamHarness, TempDir, bestmove_lines, legal, parse, write_synthetic_nn_bin};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
/// White to move with exactly one legal move: the king on 5a must capture the
/// checking gold on 5b (every other escape square is covered by that gold).
const ONE_LEGAL_MOVE: &str = "4k4/4G4/9/9/9/9/9/9/4K4 w - 1";

/// Drive a full session on a stream harness with the single-threaded synthetic
/// preamble plus `extra` option lines, then `position` + `go`, waiting for the
/// `bestmove` so the search fully completes before `quit`. Returns the transcript.
fn run_session(evaldir: &str, threads: u32, extra: &[&str], position: &str, go: &str) -> String {
    let h = StreamHarness::start();
    h.send("usi");
    h.send(&format!("setoption name Threads value {threads}"));
    h.send(&format!("setoption name EvalDir value {evaldir}"));
    for line in extra {
        h.send(line);
    }
    h.send("isready");
    assert!(
        h.wait_until(30_000, |o| o.contains("readyok")),
        "network must load and ack readyok"
    );
    h.send(position);
    h.send(go);
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "search `{go}` must finish (one bestmove):\n{}",
        h.output()
    );
    h.quit_join()
}

/// The `multipv` index of an `info` line, if present.
fn multipv_of(line: &str) -> Option<usize> {
    field_after(line, "multipv").and_then(|t| t.parse().ok())
}

/// The value following `key` in a whitespace-tokenised `info` line.
fn field_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mut it = line.split_whitespace();
    while let Some(tok) = it.next() {
        if tok == key {
            return it.next();
        }
    }
    None
}

/// A sortable score key for a `score cp X` / `score mate Y` line: a mate for the
/// side to move outranks any cp, a mate against ranks below any cp.
fn score_key(line: &str) -> i64 {
    match field_after(line, "score") {
        Some("cp") => field_after(line, "cp")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        Some("mate") => {
            let m: i64 = field_after(line, "mate")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if m >= 0 {
                1_000_000 - m
            } else {
                -1_000_000 - m
            }
        }
        _ => 0,
    }
}

/// The first PV move token of an `info … pv m1 m2 …` line, if any.
fn first_pv_move(line: &str) -> Option<&str> {
    field_after(line, "pv")
}

/// Every `info … multipv <i>` line emitted for a single completed iteration —
/// the last contiguous run of `multipv 1..N` lines before the `bestmove`. This is
/// robust to the pin's `d = max(1, depth - 1)` relabel of an un-searched line: it
/// groups by the emitted block, not by the `depth` field.
fn last_multipv_block(out: &str) -> Vec<&str> {
    let lines: Vec<&str> = out.lines().collect();
    // Find the last `multipv 1` line, then take the contiguous multipv run from it.
    let start = lines
        .iter()
        .rposition(|l| multipv_of(l) == Some(1))
        .expect("at least one multipv 1 line");
    let mut block = Vec::new();
    for l in &lines[start..] {
        if multipv_of(l).is_some() {
            block.push(*l);
        } else {
            break;
        }
    }
    block
}

#[test]
fn multipv_three_emits_three_ranked_lines_per_iteration() {
    let dir = TempDir::new("multipv3");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let out = run_session(
        e,
        1,
        &[
            "setoption name MultiPV value 3",
            "setoption name PvInterval value 0",
        ],
        "position startpos",
        "go depth 3",
    );

    // The last completed iteration emits exactly three ranked lines.
    let block = last_multipv_block(&out);
    assert_eq!(
        block.len(),
        3,
        "a completed MultiPV=3 iteration emits exactly 3 lines in:\n{out}"
    );
    let idxs: Vec<usize> = block.iter().filter_map(|l| multipv_of(l)).collect();
    assert_eq!(
        idxs,
        vec![1, 2, 3],
        "multipv indices must be 1..3 in:\n{out}"
    );

    // Scores are non-increasing by index.
    let scores: Vec<i64> = block.iter().map(|l| score_key(l)).collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "scores must be non-increasing by multipv index, got {scores:?} in:\n{out}"
    );

    // Three distinct first moves.
    let firsts: Vec<&str> = block.iter().filter_map(|l| first_pv_move(l)).collect();
    assert_eq!(firsts.len(), 3, "each line has a pv in:\n{out}");
    let distinct: std::collections::HashSet<&str> = firsts.iter().copied().collect();
    assert_eq!(distinct.len(), 3, "first moves must be distinct in:\n{out}");

    // bestmove equals the multipv-1 line's first move.
    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 1, "one bestmove in:\n{out}");
    let best = bms[0].split_whitespace().next().unwrap();
    assert_eq!(
        best, firsts[0],
        "bestmove must equal the multipv 1 first move in:\n{out}"
    );
}

#[test]
fn multipv_clamps_to_legal_move_count() {
    let dir = TempDir::new("multipv-clamp");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let legal_count = legal(&parse(STARTPOS)).len();
    assert!(legal_count > 1, "startpos has many legal moves");

    // MultiPV 600 >> legal-move count ⇒ clamps to the legal-move count.
    let out = run_session(
        e,
        1,
        &[
            "setoption name MultiPV value 600",
            "setoption name PvInterval value 0",
        ],
        "position startpos",
        "go depth 2",
    );
    let block = last_multipv_block(&out);
    assert_eq!(
        block.len(),
        legal_count,
        "MultiPV must clamp to the {legal_count} legal moves in:\n{out}"
    );
    let idxs: Vec<usize> = block.iter().filter_map(|l| multipv_of(l)).collect();
    assert_eq!(
        idxs,
        (1..=legal_count).collect::<Vec<_>>(),
        "multipv indices must run 1..={legal_count} in:\n{out}"
    );
}

#[test]
fn multipv_single_legal_move_works() {
    let dir = TempDir::new("multipv-single");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let pos = parse(ONE_LEGAL_MOVE);
    let moves = legal(&pos);
    assert_eq!(moves.len(), 1, "fixture must have exactly one legal move");

    let out = run_session(
        e,
        1,
        &[
            "setoption name MultiPV value 5",
            "setoption name PvInterval value 0",
        ],
        &format!("position sfen {ONE_LEGAL_MOVE}"),
        "go depth 2",
    );
    // Only one PV line can exist; the bestmove is the forced move.
    let block = last_multipv_block(&out);
    assert_eq!(
        block.len(),
        1,
        "a single-move position emits one line in:\n{out}"
    );
    assert_eq!(multipv_of(block[0]), Some(1));
    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 1, "one bestmove in:\n{out}");
    let tok = bms[0].split_whitespace().next().unwrap();
    let mv = parse_usi_move(tok, &pos).expect("well-formed USI move");
    assert!(moves.contains(&mv), "{tok} is not the forced legal move");
}

#[test]
fn threads2_multipv2_completes_with_both_lines_and_a_legal_bestmove() {
    // Voting is off under MultiPV > 1 (the reference `MultiPV == 1` guard). The
    // search still completes, emits both PV lines, and returns a legal bestmove.
    let dir = TempDir::new("threads2-multipv2");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let out = run_session(
        e,
        2,
        &[
            "setoption name MultiPV value 2",
            "setoption name PvInterval value 0",
        ],
        "position startpos",
        "go depth 3",
    );

    assert!(
        out.lines().any(|l| multipv_of(l) == Some(1)),
        "must emit a multipv 1 line in:\n{out}"
    );
    assert!(
        out.lines().any(|l| multipv_of(l) == Some(2)),
        "must emit a multipv 2 line in:\n{out}"
    );
    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 1, "one bestmove in:\n{out}");
    let start = parse(STARTPOS);
    let tok = bms[0].split_whitespace().next().unwrap();
    let mv = parse_usi_move(tok, &start).expect("well-formed USI move");
    assert!(legal(&start).contains(&mv), "{tok} is not a legal move");
}

#[test]
fn pv_interval_zero_prints_every_iteration() {
    let dir = TempDir::new("pvinterval0");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let out = run_session(
        e,
        1,
        &["setoption name PvInterval value 0"],
        "position startpos",
        "go depth 3",
    );
    for d in 1..=3 {
        assert!(
            out.lines()
                .any(|l| l.starts_with(&format!("info depth {d} "))),
            "PvInterval 0 must emit a depth-{d} info line in:\n{out}"
        );
    }
}

#[test]
fn pv_interval_default_still_emits_a_final_pv_before_bestmove() {
    // With the default PvInterval 300 a fast fixed-depth search may suppress the
    // intermediate lines, but the final PV always precedes `bestmove`.
    let dir = TempDir::new("pvinterval-default");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    // No PvInterval override ⇒ the 300 ms default.
    let out = run_session(e, 1, &[], "position startpos", "go depth 3");

    let bm_pos = out
        .find("\nbestmove ")
        .map(|p| p + 1)
        .or_else(|| out.starts_with("bestmove ").then_some(0))
        .unwrap_or_else(|| panic!("missing bestmove in:\n{out}"));

    // At least one `info … pv` line, before the bestmove.
    let pv_line = out[..bm_pos]
        .lines()
        .rev()
        .find(|l| l.starts_with("info depth") && l.contains(" pv "))
        .unwrap_or_else(|| panic!("no info pv line before bestmove in:\n{out}"));

    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 1, "one bestmove in:\n{out}");
    let best = bms[0].split_whitespace().next().unwrap();
    assert_eq!(
        first_pv_move(pv_line),
        Some(best),
        "final PV's first move must agree with bestmove in:\n{out}"
    );
}

#[test]
fn consideration_mode_pv_replays_as_a_legal_sequence() {
    let dir = TempDir::new("consideration");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    // ConsiderationMode forces the interval to 0 internally, so per-iteration PVs
    // are emitted; the PV is collected from the transposition table.
    let out = run_session(
        e,
        1,
        &["setoption name ConsiderationMode value true"],
        "position startpos",
        "go depth 4",
    );

    let bms = bestmove_lines(&out);
    assert_eq!(
        bms.len(),
        1,
        "engine must exit cleanly with one bestmove in:\n{out}"
    );

    // The last info line's PV replays legally from the root.
    let pv_line = out
        .lines()
        .rfind(|l| l.starts_with("info depth") && l.contains(" pv "))
        .unwrap_or_else(|| panic!("no info pv line in:\n{out}"));
    let pv_str = pv_line.split(" pv ").nth(1).unwrap();

    let mut pos = parse(STARTPOS);
    let mut count = 0;
    for tok in pv_str.split_whitespace() {
        let mv = match parse_usi_move(tok, &pos) {
            Ok(m) => m,
            Err(_) => break, // a terminal marker or unparseable tail token ends the PV
        };
        assert!(
            legal(&pos).contains(&mv),
            "PV move {tok} (#{count}) is illegal from its position in:\n{out}"
        );
        pos.do_move(mv);
        count += 1;
    }
    assert!(
        count >= 1,
        "the consideration PV must have at least one move in:\n{out}"
    );
}
