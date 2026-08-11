//! Driver-level session tests for real ponder: `go ponder`
//! holds its reply, `ponderhit` continues under time management (including the
//! `stopOnPonderhit` prompt-stop path), `Stochastic_Ponder` rewinds and
//! re-issues, and `gameover` during pondering still terminates.
//!
//! Each test drives a full `usi → setoption → isready → position → go ponder`
//! session in-process against a synthetic (all-zero) network staged in a temp
//! dir, so they are hermetic. Ponder searches never self-terminate, so they use
//! [`StreamHarness`] and drive `stop` / `ponderhit` / `gameover` explicitly. The
//! wall bounds are deliberately loose (debug builds poll the clock only at
//! ~512-node `check_time` checkpoints), like the fischer / byoyomi timing tests
//! — they prove exactly one `bestmove` arrives at the right moment, not a precise
//! deadline.

mod common;

use std::time::{Duration, Instant};

use attic_state::parse_usi_move;
use common::{StreamHarness, TempDir, bestmove_lines, legal, parse, write_synthetic_nn_bin};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// Send the standard single-threaded synthetic-network preamble and block until
/// `readyok`. `extra` carries any option lines to insert before `isready`.
fn start_ready(evaldir: &str, extra: &[String]) -> StreamHarness {
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

/// The USI move token of the single `bestmove` line (dropping any ` ponder …`
/// suffix). Panics unless exactly one `bestmove` has been emitted.
fn sole_bestmove(out: &str) -> String {
    let bms = bestmove_lines(out);
    assert_eq!(bms.len(), 1, "expected exactly one bestmove in:\n{out}");
    bms[0]
        .split_whitespace()
        .next()
        .expect("bestmove token")
        .to_string()
}

/// Assert `tok` is a legal move in the position reached from startpos by `moves`.
fn assert_legal_after(moves: &[&str], tok: &str) {
    let mut pos = parse(STARTPOS);
    for m in moves {
        let mv = parse_usi_move(m, &pos).expect("legal setup move");
        pos.do_move(mv);
    }
    let mv = parse_usi_move(tok, &pos)
        .unwrap_or_else(|_| panic!("bestmove {tok:?} is not a well-formed USI move"));
    assert!(legal(&pos).contains(&mv), "bestmove {tok:?} is not legal");
}

// -------------------------------------------------------------------------
// 1. `go ponder` holds; `stop` releases exactly one bestmove.
// -------------------------------------------------------------------------

#[test]
fn go_ponder_holds_until_stop() {
    let dir = TempDir::new("ponder-hold-stop");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[]);
    h.send("position startpos");
    h.send("go ponder btime 60000 wtime 60000");

    // A pondering search never self-terminates: no bestmove for a comfortable
    // interval, whatever the clock says.
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        !h.output().contains("bestmove"),
        "no bestmove while pondering:\n{}",
        h.output()
    );

    // `stop` ends it with exactly one bestmove.
    h.send("stop");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "stop must release exactly one bestmove:\n{}",
        h.output()
    );
    let out = h.quit_join();
    assert_legal_after(&[], &sole_bestmove(&out));
}

// -------------------------------------------------------------------------
// 2. `go ponder` then `ponderhit`: one bestmove, emitted after the ponderhit.
// -------------------------------------------------------------------------

#[test]
fn ponderhit_continues_and_emits_one_bestmove() {
    let dir = TempDir::new("ponder-hit");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[]);
    h.send("position startpos");
    // A comfortable clock: the budget is not exhausted during the short ponder,
    // so the ponderhit resumes normal time management rather than the prompt
    // `stopOnPonderhit` path.
    h.send("go ponder btime 800 wtime 800");

    std::thread::sleep(Duration::from_millis(60));
    assert!(
        !h.output().contains("bestmove"),
        "no bestmove before the ponderhit:\n{}",
        h.output()
    );

    let t = Instant::now();
    h.send("ponderhit");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "ponderhit must yield exactly one bestmove:\n{}",
        h.output()
    );
    // Bounded by the clock counted from the ponderhit (loose for a debug build).
    assert!(
        t.elapsed() < Duration::from_secs(10),
        "bestmove after ponderhit took too long: {:?}",
        t.elapsed()
    );
    let out = h.quit_join();
    assert_legal_after(&[], &sole_bestmove(&out));
}

// -------------------------------------------------------------------------
// 3. `stopOnPonderhit`: tiny clock, ponderhit well after the budget is spent.
// -------------------------------------------------------------------------

#[test]
fn stop_on_ponderhit_stops_promptly_after_a_late_ponderhit() {
    let dir = TempDir::new("ponder-stoponhit");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[]);
    h.send("position startpos");
    // A tiny clock: the soft budget is exhausted almost immediately, arming
    // `stopOnPonderhit` while pondering.
    h.send("go ponder btime 10 wtime 10");

    // Well past the budget: still no bestmove, because pondering holds regardless
    // of the exhausted clock.
    std::thread::sleep(Duration::from_millis(400));
    assert!(
        !h.output().contains("bestmove"),
        "pondering must hold even past an exhausted budget:\n{}",
        h.output()
    );

    // The ponderhit makes the first checkpoint stop the search promptly.
    let t = Instant::now();
    h.send("ponderhit");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "a late ponderhit must release exactly one bestmove:\n{}",
        h.output()
    );
    assert!(
        t.elapsed() < Duration::from_secs(10),
        "bestmove after a stopOnPonderhit ponderhit took too long: {:?}",
        t.elapsed()
    );
    let out = h.quit_join();
    assert_legal_after(&[], &sole_bestmove(&out));
}

// -------------------------------------------------------------------------
// 4. `Stochastic_Ponder`: rewound ponder, ponderhit re-issues on the current
//    position, exactly one bestmove (legal in the current position).
// -------------------------------------------------------------------------

#[test]
fn stochastic_ponder_reissues_on_the_current_position() {
    let dir = TempDir::new("ponder-stochastic");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(
        e,
        &["setoption name Stochastic_Ponder value true".to_string()],
    );
    // The current position is one move deep; `go ponder` internally rewinds it to
    // startpos and ponders there.
    h.send("position startpos moves 7g7f");
    h.send("go ponder btime 800 wtime 800");

    std::thread::sleep(Duration::from_millis(60));
    assert!(
        !h.output().contains("bestmove"),
        "no bestmove is emitted before the ponderhit under Stochastic_Ponder:\n{}",
        h.output()
    );

    h.send("ponderhit");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "the re-issued search must emit exactly one bestmove:\n{}",
        h.output()
    );
    let out = h.quit_join();
    // Exactly one bestmove, legal in the CURRENT position (White to move after
    // 7g7f), not the rewound startpos.
    assert_legal_after(&["7g7f"], &sole_bestmove(&out));
}

// -------------------------------------------------------------------------
// 5. `gameover` during `go ponder` still terminates the search.
// -------------------------------------------------------------------------

#[test]
fn gameover_terminates_a_pondering_search() {
    let dir = TempDir::new("ponder-gameover");
    write_synthetic_nn_bin(dir.path());
    let e = dir.path().to_str().unwrap();

    let h = start_ready(e, &[]);
    h.send("position startpos");
    h.send("go ponder btime 60000 wtime 60000");

    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !h.output().contains("bestmove"),
        "no bestmove while pondering:\n{}",
        h.output()
    );

    h.send("gameover lose");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "gameover must terminate a pondering search:\n{}",
        h.output()
    );
    let out = h.quit_join();
    assert_eq!(bestmove_lines(&out).len(), 1, "one bestmove in:\n{out}");
}
