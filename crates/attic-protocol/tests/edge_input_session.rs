//! Session tests for inputs at the edge of what the engine accepts: a `go` line
//! carrying `searchmoves`, a search shallow enough that the ponder move comes
//! from a bare transposition-table entry, and `spin` values that are not
//! integers or do not fit one.
//!
//! Each test pins today's behaviour and its name says whether that behaviour
//! is the intended one.

mod common;

use common::{StreamHarness, TempDir, bestmove_lines, drive, legal, parse, write_synthetic_nn_bin};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// Start a single-threaded session on the synthetic network, `extra` option
/// lines included, and wait for `readyok`.
fn start_ready(evaldir: &str, extra: &[&str]) -> StreamHarness {
    let h = StreamHarness::start();
    h.send("usi");
    h.send("setoption name Threads value 1");
    h.send("setoption name USI_Hash value 16");
    h.send(&format!("setoption name EvalDir value {evaldir}"));
    for line in extra {
        h.send(line);
    }
    h.send("isready");
    assert!(
        h.wait_until(30_000, |o| o.contains("readyok")),
        "network must load and ack readyok:\n{}",
        h.output()
    );
    h
}

fn multipv_of(line: &str) -> Option<usize> {
    let mut it = line.split_whitespace();
    while let Some(tok) = it.next() {
        if tok == "multipv" {
            return it.next().and_then(|t| t.parse().ok());
        }
    }
    None
}

/// `searchmoves` is not supported: the whole `go` line is reported as unknown
/// and no search starts, so no `bestmove` answers it. The reference instead
/// searches only the listed moves (`8h2b`, a legal non-promotion here), so this
/// is a deliberate difference from it, not the intended end state of the
/// command.
#[test]
#[cfg_attr(miri, ignore)]
fn unsupported_go_searchmoves_is_refused_and_starts_no_search() {
    let dir = TempDir::new("edge-searchmoves");
    write_synthetic_nn_bin(dir.path());
    let h = start_ready(dir.path().to_str().unwrap(), &[]);

    h.send("position startpos moves 7g7f 3c3d");
    h.send("go depth 3 searchmoves 8h2b");
    // A plain search afterwards shows the refused line left nothing running.
    h.send("go depth 1");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "the plain search must answer:\n{}",
        h.output()
    );
    let out = h.quit_join();

    assert!(
        out.lines()
            .any(|l| l == "info string unknown command: go depth 3 searchmoves 8h2b"),
        "the searchmoves line is reported as unknown:\n{out}"
    );
    assert_eq!(
        bestmove_lines(&out).len(),
        1,
        "only the plain search answers:\n{out}"
    );
}

/// A fresh depth-1 search from the start position: the final PV is the bare
/// best move, so the ponder move is looked up in the transposition table, where
/// the child's entry may carry a score and no move. The ponder lookup skips such
/// an entry, and the answer is a legal best move with, if any, a legal ponder
/// move. Intended behaviour; this test build runs with debug assertions on.
#[test]
#[cfg_attr(miri, ignore)]
fn fresh_depth1_search_answers_a_legal_bestmove_and_ponder() {
    let dir = TempDir::new("edge-depth1");
    write_synthetic_nn_bin(dir.path());
    let h = start_ready(dir.path().to_str().unwrap(), &[]);

    h.send("position startpos");
    h.send("go depth 1");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "the search must answer:\n{}",
        h.output()
    );
    let out = h.quit_join();

    let line = bestmove_lines(&out)[0];
    let mut toks = line.split_whitespace();
    let best = toks.next().expect("a bestmove token");
    let mut pos = parse(STARTPOS);
    let mv = legal(&pos)
        .into_iter()
        .find(|&m| attic_state::format_usi_move(m) == best)
        .unwrap_or_else(|| panic!("bestmove {best} is not legal at the start:\n{out}"));
    let _ = pos.do_move(mv);
    if toks.next() == Some("ponder") {
        let ponder = toks.next().expect("a ponder move after `ponder`");
        assert!(
            legal(&pos)
                .into_iter()
                .any(|m| attic_state::format_usi_move(m) == ponder),
            "ponder {ponder} is not legal after {best}:\n{out}"
        );
    }
}

/// The same ponder lookup, at the end of each of the default `bench`
/// positions' depth-1 searches. Intended behaviour.
#[test]
#[cfg_attr(miri, ignore)]
fn depth1_bench_of_the_default_positions_completes() {
    let dir = TempDir::new("edge-bench-depth1");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().unwrap();
    let out = drive(&format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         bench 64 1 1 default depth\n\
         quit\n"
    ));
    assert!(
        out.lines().any(|l| l.contains("bench: positions=4")),
        "the bench summary covers the four default positions:\n{out}"
    );
    assert_eq!(
        bestmove_lines(&out).len(),
        4,
        "one answer per position:\n{out}"
    );
}

/// A `spin` value that is not an integer, or does not fit a 64-bit one, is
/// rejected silently and the option keeps its previous value; the session goes
/// on. Intended behaviour, and the reference's own: it prints nothing for a
/// rejected spin either.
#[test]
#[cfg_attr(miri, ignore)]
fn invalid_or_oversized_spin_values_are_ignored_and_the_session_goes_on() {
    let dir = TempDir::new("edge-spin");
    write_synthetic_nn_bin(dir.path());
    let h = start_ready(
        dir.path().to_str().unwrap(),
        &[
            "setoption name Threads value abc",
            "setoption name Threads value 999999999999999999999999999999",
            "setoption name MultiPV value 2",
            "setoption name MultiPV value abc",
            "setoption name MultiPV value 999999999999999999999999999999",
            "setoption name PvInterval value 0",
        ],
    );

    h.send("position startpos");
    h.send("go depth 2");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "the search must answer:\n{}",
        h.output()
    );
    let out = h.quit_join();

    assert!(
        !out.lines()
            .any(|l| l.contains("abc") || l.contains("999999999999999999999999999999")),
        "a rejected spin prints nothing:\n{out}"
    );
    let last_two: Vec<usize> = out.lines().rev().filter_map(multipv_of).take(2).collect();
    assert_eq!(
        last_two,
        vec![2, 1],
        "MultiPV kept the value 2 it held before the rejected ones:\n{out}"
    );
}
