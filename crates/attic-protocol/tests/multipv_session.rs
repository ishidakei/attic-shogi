//! Session tests for the MultiPV loop, the `PvInterval` throttle,
//! `ConsiderationMode`, and the voting-off-under-MultiPV path.
//!
//! Each test waits for the `bestmove` before quitting: `quit` sets the stop
//! flag, which would abort a MultiPV search mid-iteration. Assertions that
//! depend on per-iteration output set `PvInterval value 0` so the default
//! 300 ms throttle does not suppress the intermediate lines.

mod common;

use attic_state::parse_usi_move;
use common::{StreamHarness, TempDir, bestmove_lines, legal, parse, write_synthetic_nn_bin};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
/// White to move with exactly one legal move: the king on 5a must capture the
/// checking gold on 5b, every other escape being covered by it.
const ONE_LEGAL_MOVE: &str = "4k4/4G4/9/9/9/9/9/9/4K4 w - 1";

/// Drive a full session with the single-threaded synthetic preamble plus `extra`
/// option lines, waiting for the `bestmove` so the search completes before
/// `quit`.
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

fn multipv_of(line: &str) -> Option<usize> {
    field_after(line, "multipv").and_then(|t| t.parse().ok())
}

fn field_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mut it = line.split_whitespace();
    while let Some(tok) = it.next() {
        if tok == key {
            return it.next();
        }
    }
    None
}

/// A sortable score key: a mate for the side to move outranks any cp, a mate
/// against ranks below any cp.
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

fn first_pv_move(line: &str) -> Option<&str> {
    field_after(line, "pv")
}

/// Every `info … multipv <i>` line of a single completed iteration: the last
/// contiguous run of `multipv 1..N` lines before the `bestmove`. Grouping by the
/// emitted block rather than the `depth` field survives the reference's
/// `d = max(1, depth - 1)` relabel of an un-searched line.
fn last_multipv_block(out: &str) -> Vec<&str> {
    let lines: Vec<&str> = out.lines().collect();
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
#[cfg_attr(miri, ignore)]
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

    let scores: Vec<i64> = block.iter().map(|l| score_key(l)).collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "scores must be non-increasing by multipv index, got {scores:?} in:\n{out}"
    );

    let firsts: Vec<&str> = block.iter().filter_map(|l| first_pv_move(l)).collect();
    assert_eq!(firsts.len(), 3, "each line has a pv in:\n{out}");
    let distinct: std::collections::HashSet<&str> = firsts.iter().copied().collect();
    assert_eq!(distinct.len(), 3, "first moves must be distinct in:\n{out}");

    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 1, "one bestmove in:\n{out}");
    let best = bms[0].split_whitespace().next().unwrap();
    assert_eq!(
        best, firsts[0],
        "bestmove must equal the multipv 1 first move in:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn multipv_clamps_to_legal_move_count() {
    let dir = TempDir::new("multipv-clamp");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let legal_count = legal(&parse(STARTPOS)).len();
    assert!(legal_count > 1, "startpos has many legal moves");

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
#[cfg_attr(miri, ignore)]
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
#[cfg_attr(miri, ignore)]
fn threads2_multipv2_completes_with_both_lines_and_a_legal_bestmove() {
    // Voting is off under MultiPV > 1 (the reference `MultiPV == 1` guard).
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
#[cfg_attr(miri, ignore)]
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
#[cfg_attr(miri, ignore)]
fn pv_interval_default_still_emits_a_final_pv_before_bestmove() {
    let dir = TempDir::new("pvinterval-default");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let out = run_session(e, 1, &[], "position startpos", "go depth 3");

    let bm_pos = out
        .find("\nbestmove ")
        .map(|p| p + 1)
        .or_else(|| out.starts_with("bestmove ").then_some(0))
        .unwrap_or_else(|| panic!("missing bestmove in:\n{out}"));

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
#[cfg_attr(miri, ignore)]
fn consideration_mode_pv_replays_as_a_legal_sequence() {
    let dir = TempDir::new("consideration");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    // ConsiderationMode forces the interval to 0, so per-iteration PVs appear.
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
