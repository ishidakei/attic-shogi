//! Session tests for the `USI_Hash` resize, `DepthLimit` / `NodesLimit`,
//! `MaxMovesToDraw`, and the `gameover` command.
//!
//! Each test drives a full session against a synthetic all-zero network staged
//! in a temp dir, waiting for the `bestmove` before quitting so a fixed result
//! never races the `quit`-driven join.

mod common;

use attic_state::parse_usi_move;
use common::{StreamHarness, TempDir, bestmove_lines, legal, parse, write_synthetic_nn_bin};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// A gold-drop head mate for Black at a high game ply: `G*8a` mates the White
/// king on 9a, which is not itself in check at the root.
const MATE_AT_PLY_100: &str = "k8/9/G1N6/9/9/9/9/9/8K b G 100";

/// Send the standard single-threaded synthetic-network preamble and block until
/// `readyok`. `extra` carries any option lines to insert before `isready`.
fn start_ready(evaldir: &str, extra: &[&str]) -> StreamHarness {
    let h = StreamHarness::start();
    h.send("usi");
    h.send("setoption name Threads value 1");
    h.send(&format!("setoption name EvalDir value {evaldir}"));
    for line in extra {
        h.send(line);
    }
    h.send("isready");
    assert!(
        h.wait_until(30_000, |o| o.contains("readyok")),
        "network must load and ack readyok"
    );
    h
}

/// Preamble option for every test that compares one search's transcript against
/// another's: it makes the `info` output a pure function of the search.
///
/// Under the default `PvInterval 300` the per-iteration PV is gated on the wall
/// clock, and the coordinator emits its end-of-search fallback PV only when the
/// last iteration's own line was throttled away. Those two lines report
/// different depths — the aborted iteration's versus the last completed one — so
/// two searches visiting the identical node sequence can print different final
/// lines purely by timing. `PvInterval 0` removes the gate.
const DETERMINISTIC_PV: &str = "setoption name PvInterval value 0";

/// One summary per completed search in `out`: its last `info depth …` line
/// joined with its `bestmove …` line. Loading the synthetic network dominates a
/// session's cost, so tests comparing two searches run both in one session and
/// split the transcript here.
///
/// Which `info` line ends up last is only meaningful under [`DETERMINISTIC_PV`],
/// so every session whose summaries are compared must send it. The `nps` and
/// `time` fields are dropped, being wall-clock derived: two searches over the
/// identical node sequence report different values for them.
fn go_summaries(out: &str) -> Vec<String> {
    let mut res = Vec::new();
    let mut cur_info = String::new();
    for line in out.lines() {
        if line.starts_with("info depth") {
            cur_info = without_timing(line);
        } else if line.starts_with("bestmove") {
            res.push(format!("{cur_info}\n{line}"));
            cur_info = String::new();
        }
    }
    res
}

/// `line` with the ` nps <n>` and ` time <n>` fields removed.
fn without_timing(line: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut toks = line.split(' ');
    while let Some(tok) = toks.next() {
        if tok == "nps" || tok == "time" {
            toks.next();
        } else {
            out.push(tok);
        }
    }
    out.join(" ")
}

/// Send `position` then `go`, and block until the transcript holds
/// `expect_bestmoves` total `bestmove` lines.
fn go_and_wait(h: &StreamHarness, position: &str, go: &str, expect_bestmoves: usize) {
    h.send(position);
    h.send(go);
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == expect_bestmoves),
        "search `{go}` must finish (expected {expect_bestmoves} bestmoves):\n{}",
        h.output()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn usi_hash_small_matches_default_fixed_depth() {
    // A small `USI_Hash` changes speed, not a fixed-depth result. Both searches
    // run in one session so the network loads once.
    let dir = TempDir::new("hash-fixed");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go depth 3", 1);
    // `usinewgame` makes the second search independent of the first, leaving the
    // hash size as the only difference.
    h.send("usinewgame");
    h.send("setoption name USI_Hash value 8");
    h.send("isready");
    go_and_wait(&h, "position startpos", "go depth 3", 2);
    let out = h.quit_join();

    let s = go_summaries(&out);
    assert_eq!(s.len(), 2, "two searches in:\n{out}");
    assert_eq!(
        s[0], s[1],
        "USI_Hash 8 must not change the depth-3 result:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn usi_hash_mid_session_resize_between_gos() {
    // The `isready` that applies the new size joins the first worker before
    // resizing, so both `go`s emit a bestmove and the engine exits cleanly.
    let dir = TempDir::new("hash-resize");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[]);
    h.send("position startpos");
    h.send("go depth 2");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "first go must complete"
    );
    h.send("setoption name USI_Hash value 16");
    h.send("isready");
    h.send("position startpos");
    h.send("go depth 2");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 2),
        "second go after a mid-session resize must complete"
    );
    let out = h.quit_join();
    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 2, "exactly two bestmoves in:\n{out}");
    let start = parse(STARTPOS);
    for bm in &bms {
        let tok = bm.split_whitespace().next().unwrap();
        let mv = parse_usi_move(tok, &start).expect("well-formed USI move");
        assert!(
            legal(&start).contains(&mv),
            "{tok} is not a legal startpos move"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn depth_limit_caps_search_and_matches_plain_go_depth() {
    // With `DepthLimit 2` and a generous movetime the search must stop where a
    // plain `go depth 2` does; a later explicit `go depth 4` must then overwrite
    // the option-seeded limit.
    let dir = TempDir::new("depthlimit");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go depth 2", 1);
    let plain_out = h.output();

    // `usinewgame` isolates each search so the two are comparable.
    h.send("usinewgame");
    h.send("setoption name DepthLimit value 2");
    go_and_wait(&h, "position startpos", "go movetime 5000", 2);
    let capped_out = h.output();
    let capped_block = capped_out[plain_out.len()..].to_string();
    assert!(
        !capped_block.lines().any(|l| l.starts_with("info depth 3")),
        "DepthLimit 2 must stop at depth 2 (no info depth 3) in:\n{capped_block}"
    );

    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go depth 4", 3);
    let out = h.quit_join();
    let override_block = out[capped_out.len()..].to_string();
    assert!(
        override_block
            .lines()
            .any(|l| l.starts_with("info depth 4")),
        "explicit go depth 4 must reach depth 4 despite DepthLimit 2 in:\n{override_block}"
    );

    let s = go_summaries(&out);
    assert_eq!(s.len(), 3, "three searches in:\n{out}");
    assert_eq!(
        s[0], s[1],
        "DepthLimit 2 must match plain go depth 2:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn nodes_limit_matches_go_nodes() {
    // With a `NodesLimit` below the position's uncapped node count, an
    // option-seeded bare `go` must match an explicit `go nodes N`.
    let dir = TempDir::new("nodeslimit");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go nodes 3000", 1);
    h.send("usinewgame");
    h.send("setoption name NodesLimit value 3000");
    go_and_wait(&h, "position startpos", "go", 2);
    let out = h.quit_join();

    let s = go_summaries(&out);
    assert_eq!(s.len(), 2, "two searches in:\n{out}");
    assert_eq!(
        s[0], s[1],
        "NodesLimit 3000 must behave exactly as go nodes 3000:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn explicit_zero_depth_and_nodes_override_the_options_as_no_limit() {
    // A zero `depth` or `nodes` on the `go` line sets no limit, and it still
    // wins over a nonzero `DepthLimit` / `NodesLimit`.
    let dir = TempDir::new("zero-override");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go nodes 3000", 1);
    h.send("usinewgame");
    h.send("setoption name DepthLimit value 2");
    go_and_wait(&h, "position startpos", "go depth 0 nodes 3000", 2);
    h.send("usinewgame");
    h.send("setoption name DepthLimit value 0");
    go_and_wait(&h, "position startpos", "go depth 3", 3);
    h.send("usinewgame");
    h.send("setoption name NodesLimit value 50");
    go_and_wait(&h, "position startpos", "go nodes 0 depth 3", 4);
    let out = h.quit_join();

    let s = go_summaries(&out);
    assert_eq!(s.len(), 4, "four searches in:\n{out}");
    assert!(
        !s[0].starts_with("info depth 1 ") && !s[0].starts_with("info depth 2 "),
        "go nodes 3000 must search past depth 2:\n{out}"
    );
    assert_eq!(
        s[0], s[1],
        "go depth 0 must lift DepthLimit 2 and match go nodes 3000:\n{out}"
    );
    assert_eq!(
        s[2], s[3],
        "go nodes 0 must lift NodesLimit 50 and match go depth 3:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn zero_movetime_and_mate_set_no_limit() {
    // A zero `movetime` or `mate` budget sets no limit, so the search runs to
    // the given depth instead of stopping at once.
    let dir = TempDir::new("zero-movetime");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go depth 3", 1);
    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go movetime 0 depth 3", 2);
    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go mate 0 depth 3", 3);
    let out = h.quit_join();

    let s = go_summaries(&out);
    assert_eq!(s.len(), 3, "three searches in:\n{out}");
    assert!(
        s[0].starts_with("info depth 3 "),
        "go depth 3 must complete depth 3:\n{out}"
    );
    assert_eq!(s[0], s[1], "go movetime 0 must set no limit:\n{out}");
    assert_eq!(s[0], s[2], "go mate 0 must set no limit:\n{out}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn negative_depth_runs_no_iteration() {
    // The iterative-deepening loop is never entered, so the reply carries the
    // first root move unsearched, as the reference's does.
    let dir = TempDir::new("negative-depth");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go depth -3", 1);
    let out = h.quit_join();

    let s = go_summaries(&out);
    assert_eq!(s.len(), 1, "one search in:\n{out}");
    assert_eq!(
        s[0], "info depth 1 multipv 1 score cp 0 nodes 0 hashfull 0 pv 1g1f\nbestmove 1g1f",
        "go depth -3 must not search:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn go_with_a_skipped_word_searches() {
    let dir = TempDir::new("skipped-word");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go depth 2", 1);
    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go depth 2abc", 2);
    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go movestogo 40 depth 2", 3);
    let out = h.quit_join();

    assert!(
        !out.contains("unknown command"),
        "a skipped word must not make the line unknown in:\n{out}"
    );
    let s = go_summaries(&out);
    assert_eq!(s.len(), 3, "three searches in:\n{out}");
    assert_eq!(s[0], s[1], "go depth 2abc must be go depth 2:\n{out}");
    assert_eq!(
        s[0], s[2],
        "go movestogo 40 depth 2 must be go depth 2:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn max_moves_to_draw_scores_a_last_move_mate_and_draws_past_the_limit() {
    // With `MaxMovesToDraw` equal to the root's game ply, the mating drop is
    // the last allowed move and still reports `mate 1`. From a root with no
    // mate in one, every child lies past the limit with legal moves, so the
    // score collapses to a draw-band `cp`.
    let dir = TempDir::new("mmtd");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &["setoption name MaxMovesToDraw value 100"]);

    go_and_wait(
        &h,
        &format!("position sfen {MATE_AT_PLY_100}"),
        "go depth 2",
        1,
    );
    let mate = h.output();
    assert!(
        mate.lines().any(|l| l.contains("score mate 1 ")),
        "a mate on the last allowed move must score as a mate in:\n{mate}"
    );
    assert_eq!(
        bestmove_lines(&mate)[0].split_whitespace().next().unwrap(),
        "G*8a",
        "the search plays the mating drop in:\n{mate}"
    );

    h.send("usinewgame");
    let startpos_at_100 = STARTPOS.replace(" b - 1", " b - 100");
    go_and_wait(
        &h,
        &format!("position sfen {startpos_at_100}"),
        "go depth 2",
        2,
    );
    let out = h.quit_join();
    let capped = &out[mate.len()..];
    assert!(
        !capped.lines().any(|l| l.contains("score mate")),
        "no mate past the limit in:\n{capped}"
    );
    let scores: Vec<i32> = capped
        .lines()
        .filter_map(|l| {
            let mut t = l.split_whitespace().skip_while(|&w| w != "cp");
            t.next()?;
            t.next()?.parse().ok()
        })
        .collect();
    assert!(
        !scores.is_empty() && scores.iter().all(|cp| cp.abs() <= 3),
        "every score must be in the draw band in:\n{capped}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn gameover_releases_infinite_search_and_a_fresh_go_works() {
    let dir = TempDir::new("gameover-inf");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[]);
    h.send("position startpos");
    h.send("go infinite");
    // An infinite search emits no bestmove until stopped, so let the worker run
    // briefly and confirm nothing has appeared.
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(
        !h.output().contains("bestmove"),
        "no bestmove until gameover:\n{}",
        h.output()
    );

    h.send("gameover lose");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "gameover must release the bestmove:\n{}",
        h.output()
    );

    h.send("usinewgame");
    h.send("position startpos");
    h.send("go depth 2");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 2),
        "a fresh go after gameover must complete:\n{}",
        h.output()
    );
    let out = h.quit_join();
    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 2, "two bestmoves total in:\n{out}");
    let start = parse(STARTPOS);
    let tok = bms[1].split_whitespace().next().unwrap();
    let mv = parse_usi_move(tok, &start).expect("well-formed USI move");
    assert!(
        legal(&start).contains(&mv),
        "{tok} is not a legal startpos move"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn gameover_result_token_is_optional_and_ignored() {
    let dir = TempDir::new("gameover-bare");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[]);
    h.send("position startpos");
    h.send("go infinite");
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(
        !h.output().contains("bestmove"),
        "no bestmove until gameover:\n{}",
        h.output()
    );
    h.send("gameover");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "bare gameover must release the bestmove:\n{}",
        h.output()
    );
    let out = h.quit_join();
    assert_eq!(bestmove_lines(&out).len(), 1, "one bestmove in:\n{out}");
    assert!(
        !out.contains("unknown command"),
        "gameover must not be an unknown command in:\n{out}"
    );
}
