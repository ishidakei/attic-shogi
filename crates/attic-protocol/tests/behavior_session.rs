//! Session tests for `DrawValueBlack` / `DrawValueWhite`, `ResignValue`, and
//! the `go mate` limit.
//!
//! Each test drives a full session against a synthetic all-zero network staged
//! in a temp dir. Searches run on a worker thread, so the harness waits for the
//! `bestmove` before comparing or quitting.

mod common;

use common::{StreamHarness, TempDir, bestmove_lines, write_synthetic_nn_bin};

/// Startpos at a high game ply, so a small `MaxMovesToDraw` adjudicates an
/// immediate draw at every child node and the reported root score collapses to
/// the draw contempt for the root side.
const STARTPOS_PLY_100_BLACK: &str =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 100";
const STARTPOS_PLY_100_WHITE: &str =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 100";

/// Black to move with a mate-in-1: `G*8a` mates the White king on 9a.
const MATE_IN_1_BLACK: &str = "k8/9/G1N6/9/9/9/9/9/8K b G 1";

/// Black to move, forced mated-in-2 by White: the king on 9i is boxed in and
/// Black's only legal move is the 5e pawn push, after which White drops `g*8i`.
/// A real search returns a decisive mated score, so `ResignValue` can fire.
const MATED_IN_2_BLACK: &str = "8k/9/9/9/4P4/9/g1n6/9/K8 b g 1";

/// Send the standard single-threaded synthetic-network preamble and block until
/// `readyok`. `extra` carries any option lines to insert before `isready`.
fn start_ready(evaldir: &str, threads: u32, extra: &[String]) -> StreamHarness {
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
    h
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

/// The `cp` value of the last `info ... score cp N ...` line in `text`, if any.
fn last_score_cp(text: &str) -> Option<i64> {
    let mut found = None;
    for line in text.lines() {
        if let Some(rest) = line.split(" score cp ").nth(1)
            && let Some(tok) = rest.split_whitespace().next()
            && let Ok(v) = tok.parse::<i64>()
        {
            found = Some(v);
        }
    }
    found
}

#[test]
#[cfg_attr(miri, ignore)]
fn draw_value_black_shifts_the_root_side_adjudicated_score() {
    // With MaxMovesToDraw 50 every child adjudicates a draw, so the root reports
    // the Black-side draw contempt: `DrawValueBlack * Pawn / 100`.
    let dir = TempDir::new("dv-black");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();
    let position = format!("position sfen {STARTPOS_PLY_100_BLACK}");

    let h = start_ready(
        e,
        1,
        &["setoption name MaxMovesToDraw value 50".to_string()],
    );

    go_and_wait(&h, &position, "go depth 2", 1);
    let default_out = h.output();
    let default_cp = last_score_cp(&default_out).expect("a cp score under default draw value");
    assert!(
        default_cp <= 0,
        "default DrawValueBlack must report a non-positive cp, got {default_cp} in:\n{default_out}"
    );

    h.send("usinewgame");
    h.send("setoption name DrawValueBlack value 500");
    go_and_wait(&h, &position, "go depth 2", 2);
    let after = h.output();
    let shifted = after[default_out.len()..].to_string();
    let shifted_cp = last_score_cp(&shifted).expect("a cp score under DrawValueBlack 500");
    assert!(
        shifted_cp >= 400,
        "DrawValueBlack 500 must lift the root score to ~ +500 cp, got {shifted_cp} in:\n{shifted}"
    );

    // DrawValueWhite must not affect a Black-to-move root.
    h.send("usinewgame");
    h.send("setoption name DrawValueBlack value -2");
    h.send("setoption name DrawValueWhite value 500");
    go_and_wait(&h, &position, "go depth 2", 3);
    let end = h.quit_join();
    let white_leg = end[after.len()..].to_string();
    let white_cp = last_score_cp(&white_leg).expect("a cp score under DrawValueWhite 500");
    assert!(
        white_cp <= 0,
        "DrawValueWhite must not shift a Black-to-move root, got {white_cp} in:\n{white_leg}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn draw_value_white_shifts_a_white_to_move_root() {
    let dir = TempDir::new("dv-white");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();
    let position = format!("position sfen {STARTPOS_PLY_100_WHITE}");

    let h = start_ready(
        e,
        1,
        &["setoption name MaxMovesToDraw value 50".to_string()],
    );

    go_and_wait(&h, &position, "go depth 2", 1);
    let default_out = h.output();
    let default_cp = last_score_cp(&default_out).expect("a cp score under default draw value");
    assert!(
        default_cp <= 0,
        "default DrawValueWhite must report a non-positive cp, got {default_cp} in:\n{default_out}"
    );

    h.send("usinewgame");
    h.send("setoption name DrawValueWhite value 500");
    go_and_wait(&h, &position, "go depth 2", 2);
    let end = h.quit_join();
    let shifted = end[default_out.len()..].to_string();
    let shifted_cp = last_score_cp(&shifted).expect("a cp score under DrawValueWhite 500");
    assert!(
        shifted_cp >= 400,
        "DrawValueWhite 500 must lift the white-to-move root score, got {shifted_cp} in:\n{shifted}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn resign_value_resigns_a_lost_position_but_default_plays() {
    let dir = TempDir::new("resign");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();
    let position = format!("position sfen {MATED_IN_2_BLACK}");

    let h = start_ready(e, 1, &[]);

    // The default threshold is unreachable, so it plays its one legal move.
    go_and_wait(&h, &position, "go depth 3", 1);
    let default_out = h.output();
    let default_bm = bestmove_lines(&default_out)[0]
        .split_whitespace()
        .next()
        .unwrap()
        .to_string();
    assert_ne!(
        default_bm, "resign",
        "default ResignValue must not resign a searchable position:\n{default_out}"
    );
    assert!(
        default_bm.starts_with("5e5d"),
        "default must play the only legal move 5e5d, got {default_bm} in:\n{default_out}"
    );

    h.send("usinewgame");
    h.send("setoption name ResignValue value 100");
    go_and_wait(&h, &position, "go depth 3", 2);
    let end = h.quit_join();
    let resign_leg = end[default_out.len()..].to_string();
    let resign_bm = bestmove_lines(&resign_leg)[0]
        .split_whitespace()
        .next()
        .unwrap()
        .to_string();
    assert_eq!(
        resign_bm, "resign",
        "ResignValue 100 must resign a lost position:\n{resign_leg}"
    );

    // The final PV must precede `bestmove resign` (`yaneuraou-search.cpp`)
    // so the GUI sees the score the resignation was decided on. A run whose last
    // iteration was already emitted would otherwise resign with no `info score`
    // line at all.
    let before_bestmove = resign_leg
        .split_once("bestmove")
        .map(|(head, _)| head)
        .unwrap_or(&resign_leg);
    assert!(
        before_bestmove
            .lines()
            .any(|l| l.starts_with("info ") && l.contains(" score ")),
        "resigning by value must print the deciding PV first:\n{resign_leg}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn go_mate_finds_the_mate_and_terminates_on_quiet() {
    let dir = TempDir::new("go-mate");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, 1, &[]);

    go_and_wait(
        &h,
        &format!("position sfen {MATE_IN_1_BLACK}"),
        "go mate 5000",
        1,
    );
    let mate_out = h.output();
    assert_eq!(
        bestmove_lines(&mate_out)[0].split_whitespace().next(),
        Some("G*8a"),
        "go mate must play the mating drop:\n{mate_out}"
    );

    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go mate 2000", 2);
    let end = h.quit_join();
    let quiet_leg = end[mate_out.len()..].to_string();
    let quiet_bm = bestmove_lines(&quiet_leg)[0]
        .split_whitespace()
        .next()
        .unwrap()
        .to_string();
    assert_ne!(quiet_bm, "resign", "quiet go mate must not resign");
    assert_ne!(quiet_bm, "win", "quiet go mate must not declare a win");
}

#[test]
#[cfg_attr(miri, ignore)]
fn go_mate_infinite_releases_on_stop() {
    // `go mate infinite` has no time bound, so nothing is emitted until `stop`.
    let dir = TempDir::new("go-mate-inf");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, 1, &[]);
    h.send("position startpos");
    h.send("go mate infinite");
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(
        !h.output().contains("bestmove"),
        "an unbounded go mate must not reply before stop:\n{}",
        h.output()
    );
    h.send("stop");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "stop must release the go mate reply:\n{}",
        h.output()
    );
    h.quit_join();
}

#[test]
#[cfg_attr(miri, ignore)]
fn go_mate_threads2_smoke_completes() {
    // The vote is off under `limits.mate`, so the main worker reports.
    let dir = TempDir::new("go-mate-t2");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, 2, &[]);
    go_and_wait(&h, "position startpos", "go mate 2000", 1);
    let out = h.quit_join();
    assert_eq!(
        bestmove_lines(&out).len(),
        1,
        "threads=2 go mate must complete with one bestmove:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn time_managed_threads2_smoke_completes() {
    // A real clock engages time management, exercising the Lazy-SMP
    // best-move-change aggregation (`yaneuraou-search.cpp`).
    let dir = TempDir::new("time-t2");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, 2, &[]);
    go_and_wait(&h, "position startpos", "go btime 300 wtime 300", 1);
    let out = h.quit_join();
    assert_eq!(
        bestmove_lines(&out).len(),
        1,
        "threads=2 time-managed search must complete with one bestmove:\n{out}"
    );
}
