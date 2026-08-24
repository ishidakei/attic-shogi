//! Driver-level session tests for the drive / limit group:
//! `USI_Hash` resize, `DepthLimit` / `NodesLimit`, `MaxMovesToDraw`, and the
//! `gameover` command.
//!
//! Each test drives a full `usi → setoption → isready → position → go` session
//! in-process against a synthetic (all-zero) network staged in a temp dir, so
//! they are hermetic. Multi-depth / node-capped / infinite searches run on a
//! worker thread, so they use [`StreamHarness`] and wait for the `bestmove`
//! before quitting — a fixed result never races the `quit`-driven join.

mod common;

use attic_state::parse_usi_move;
use common::{StreamHarness, TempDir, bestmove_lines, legal, parse, write_synthetic_nn_bin};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// A gold-drop head mate for Black at a high game ply (`G*8a` mates the White
/// king on 9a, which is not itself in check at the root). Used to show the
/// `MaxMovesToDraw` horizon suppressing a mate.
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
/// Under the default `PvInterval 300` the per-iteration PV is gated on wall clock
/// (`attic-search/src/qsearch.rs:2016`, `pv_interval_elapsed`), and the
/// coordinator only emits its end-of-search fallback PV when the last iteration's
/// own line was throttled away (`attic-protocol/src/driver.rs:2821`,
/// `!uci_pv_sent`). The two lines report different depths — the aborted
/// iteration's `root_depth` versus the last *completed* depth — so whether a
/// search's last `info` line reads `depth 11` or `depth 10` depends only on how
/// long it happened to take. Two searches that visit the identical node sequence
/// can therefore print different final lines, which is flaky, not a divergence.
/// `PvInterval 0` removes the gate: every iteration prints unconditionally.
const DETERMINISTIC_PV: &str = "setoption name PvInterval value 0";

/// One summary per completed search in `out`: each search's last `info depth …`
/// line joined with its `bestmove …` line, in order. Loading the 107 MiB
/// synthetic network dominates a session's cost, so tests that compare two
/// searches run both in ONE session (one load) and split the transcript here.
///
/// Which `info` line ends up last is only meaningful under [`DETERMINISTIC_PV`],
/// so every session whose summaries are compared must send it.
fn go_summaries(out: &str) -> Vec<String> {
    let mut res = Vec::new();
    let mut cur_info = "";
    for line in out.lines() {
        if line.starts_with("info depth") {
            cur_info = line;
        } else if line.starts_with("bestmove") {
            res.push(format!("{cur_info}\n{line}"));
            cur_info = "";
        }
    }
    res
}

/// Send `position` then `go`, and block until the transcript holds
/// `expect_bestmoves` total `bestmove` lines (i.e. this search has finished).
fn go_and_wait(h: &StreamHarness, position: &str, go: &str, expect_bestmoves: usize) {
    h.send(position);
    h.send(go);
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == expect_bestmoves),
        "search `{go}` must finish (expected {expect_bestmoves} bestmoves):\n{}",
        h.output()
    );
}

// -------------------------------------------------------------------------
// USI_Hash.
// -------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn usi_hash_small_matches_default_fixed_depth() {
    // A small USI_Hash only changes speed, not a fixed-depth result: the depth-3
    // startpos search under USI_Hash 8 must produce the same info line (nodes /
    // score) and bestmove as the default 1024. Both run in one session (default
    // first, then a mid-session resize to 8) so the network loads once.
    let dir = TempDir::new("hash-fixed");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go depth 3", 1);
    // `usinewgame` resets the game-scoped histories and clears the table, so the
    // second search is independent of the first — the only difference is the
    // hash size under test.
    h.send("usinewgame");
    h.send("setoption name USI_Hash value 8");
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
    // A resize between two go's works and the second search completes. The
    // driver joins the first (async) worker before resizing, so both go's emit a
    // bestmove and the engine exits cleanly.
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

// -------------------------------------------------------------------------
// DepthLimit / NodesLimit.
// -------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn depth_limit_caps_search_and_matches_plain_go_depth() {
    // With DepthLimit 2 and a generous movetime the search stops at depth 2 (no
    // `info depth 3`), and its bestmove / nodes equal the plain `go depth 2` run.
    // Both run in one session (plain first, then DepthLimit 2) so the net loads
    // once. An explicit `go depth 4` in the same session then reaches depth 4,
    // proving an explicit token overwrites the option-seeded limit.
    let dir = TempDir::new("depthlimit");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go depth 2", 1);
    let plain_out = h.output();

    // `usinewgame` isolates each search (fresh histories + cleared table) so the
    // DepthLimit-capped run is comparable to the plain `go depth 2`.
    h.send("usinewgame");
    h.send("setoption name DepthLimit value 2");
    go_and_wait(&h, "position startpos", "go movetime 5000", 2);
    let capped_out = h.output();
    let capped_block = capped_out[plain_out.len()..].to_string();
    assert!(
        !capped_block.lines().any(|l| l.starts_with("info depth 3")),
        "DepthLimit 2 must stop at depth 2 (no info depth 3) in:\n{capped_block}"
    );

    // An explicit go depth 4 overwrites the DepthLimit-seeded 2.
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
    // With NodesLimit N (below the position's uncapped node count) the final
    // aggregated node count respects the cap exactly as `go nodes N` does: the
    // explicit `go nodes N` and the option-seeded bare `go` produce the same
    // final info line (nodes) and bestmove. Both run in one session, under
    // `PvInterval 0` so neither transcript depends on the wall clock.
    let dir = TempDir::new("nodeslimit");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[DETERMINISTIC_PV]);
    go_and_wait(&h, "position startpos", "go nodes 3000", 1);
    // `usinewgame` isolates the two node-capped searches so they are comparable.
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

// -------------------------------------------------------------------------
// MaxMovesToDraw.
// -------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn max_moves_to_draw_adjudicates_a_draw_past_the_horizon() {
    // A mate-in-1 position at a high game ply. Unlimited (MaxMovesToDraw 0 =
    // default) the search finds the mate; a small MaxMovesToDraw makes every
    // interior / qsearch node adjudicate a draw before the mate is seen, so the
    // reported score collapses from `mate` to a draw-band `cp` value and the
    // search still terminates with a bestmove.
    let dir = TempDir::new("mmtd");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();
    let position = format!("position sfen {MATE_AT_PLY_100}");

    let h = start_ready(e, &[]);

    // Value 0 (unlimited, default): searches normally and finds the mate.
    go_and_wait(&h, &position, "go depth 2", 1);
    let unlimited = h.output();
    assert!(
        unlimited.lines().any(|l| l.contains("score mate")),
        "unlimited MaxMovesToDraw must find the mate in:\n{unlimited}"
    );
    assert_eq!(
        bestmove_lines(&unlimited)[0]
            .split_whitespace()
            .next()
            .unwrap(),
        "G*8a",
        "unlimited search plays the mating drop in:\n{unlimited}"
    );

    // Small horizon (50 < game ply 100): the mate is suppressed for a draw.
    h.send("usinewgame");
    h.send("setoption name MaxMovesToDraw value 50");
    go_and_wait(&h, &position, "go depth 2", 2);
    let out = h.quit_join();
    let capped = out[unlimited.len()..].to_string();
    assert!(
        !capped.lines().any(|l| l.contains("score mate")),
        "MaxMovesToDraw must suppress the mate in:\n{capped}"
    );
    assert!(
        capped.lines().any(|l| l.contains("score cp")),
        "the capped search must report a draw-adjudicated cp score in:\n{capped}"
    );
}

// -------------------------------------------------------------------------
// Gameover (handled exactly like stop).
// -------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn gameover_releases_infinite_search_and_a_fresh_go_works() {
    // `go infinite` then `gameover lose` releases the bestmove exactly as `stop`
    // would; afterwards `usinewgame` + a fresh `go` works normally.
    let dir = TempDir::new("gameover-inf");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[]);
    h.send("position startpos");
    h.send("go infinite");
    // An infinite search emits no bestmove until stopped (this engine reports a
    // single final `info`/`bestmove` on abort, no per-iteration progress lines),
    // so let the worker run briefly and confirm nothing has been emitted yet.
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

    // A fresh game after gameover works.
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
    // A bare `gameover` (no win/lose/draw token) is accepted and releases the
    // held reply just like `gameover lose` / `stop`.
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
    // The token itself is never echoed as an unknown command.
    assert!(
        !out.contains("unknown command"),
        "gameover must not be an unknown command in:\n{out}"
    );
}
