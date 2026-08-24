//! Driver-level session tests for the `bench` command: a
//! reproducible NPS benchmark ported from the pinned reference's `bench`
//! (`source/benchmark.cpp` + `usi.cpp`).
//!
//! The hermetic syntax tests (bare `bench`, garbage argument, `current` source)
//! run WITHOUT a network: each position resigns instantly, so they exercise the
//! argument parse and summary plumbing without a real search. The determinism
//! and multi-thread gates stage a synthetic (all-zero) network in a temp dir and
//! run a small fixed-depth bench so they stay fast.

mod common;

use common::{TempDir, drive, write_synthetic_nn_bin};

/// Extract the `nodes=` field from the single `bench:` summary line in `out`.
fn bench_summary_nodes(out: &str) -> u64 {
    let line = out
        .lines()
        .find(|l| l.contains("bench: positions="))
        .unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    let field = line
        .split_whitespace()
        .find_map(|t| t.strip_prefix("nodes="))
        .unwrap_or_else(|| panic!("no nodes= field in bench summary: {line:?}"));
    field.parse().expect("nodes= is an integer")
}

/// The `positions=` field from the `bench:` summary line.
fn bench_summary_positions(out: &str) -> u64 {
    let line = out
        .lines()
        .find(|l| l.contains("bench: positions="))
        .unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    line.split_whitespace()
        .find_map(|t| t.strip_prefix("positions="))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no positions= field in: {line:?}"))
}

// -------------------------------------------------------------------------
// Syntax variants (hermetic — no network, so each position resigns).
// -------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn bare_bench_runs_the_four_default_positions() {
    // No network loaded → each of the four default positions resigns instantly,
    // and the summary reports positions=4, nodes=0. This proves the default
    // position list and the summary plumbing without a 60-second real search.
    let out = drive("bench\nquit\n");
    assert_eq!(
        bench_summary_positions(&out),
        4,
        "bare bench runs the four default positions:\n{out}"
    );
    assert_eq!(
        bench_summary_nodes(&out),
        0,
        "no network → zero nodes:\n{out}"
    );
    let bestmoves = common::bestmove_lines(&out);
    assert_eq!(
        bestmoves,
        vec!["resign"; 4],
        "one resign per position:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn garbage_argument_fails_loudly_without_panicking() {
    // A non-integer TT size is a loud parse error, not a panic and not a search.
    let out = drive("bench notanumber\nquit\n");
    assert!(
        out.contains("info string bench: invalid ttSizeMB"),
        "expected a loud parse error:\n{out}"
    );
    assert!(
        !out.contains("bench: positions="),
        "a parse error runs no positions:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn unsupported_limit_type_fails_loudly() {
    // `perft` / `eval` are out of NPS-bench scope; they are reported, not run.
    let out = drive("bench 16 1 5 default perft\nquit\n");
    assert!(
        out.contains("info string bench: unsupported limit type `perft`"),
        "expected the scope-divergence notice:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn current_source_benches_the_set_position() {
    // `current` benches exactly one position (the session position). With no
    // network it resigns, but the summary must report positions=1.
    let session = "position startpos moves 7g7f\n\
                   bench 16 1 4 current depth\n\
                   quit\n";
    let out = drive(session);
    assert_eq!(
        bench_summary_positions(&out),
        1,
        "current source = one position:\n{out}"
    );
}

// -------------------------------------------------------------------------
// Determinism — identical total nodes across runs (threads=1).
// -------------------------------------------------------------------------

/// The standard bench session against a staged synthetic network: a small
/// fixed-depth default bench so CI stays fast. Returns the full transcript.
fn bench_session(evaldir: &str, bench_line: &str) -> String {
    let input = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name USI_Hash value 16\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         {bench_line}\n\
         quit\n"
    );
    drive(&input)
}

#[test]
#[cfg_attr(miri, ignore)]
fn two_runs_in_one_process_report_identical_nodes() {
    let dir = TempDir::new("bench-determinism-1proc");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 path");

    // Two bench runs in ONE process (one network load). Each resets the TT and
    // histories at its start, so run 2 sees the same clean state as run 1.
    let input = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name USI_Hash value 16\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         bench 16 1 3 default depth\n\
         bench 16 1 3 default depth\n\
         quit\n"
    );
    let out = drive(&input);
    let summaries: Vec<&str> = out
        .lines()
        .filter(|l| l.contains("bench: positions="))
        .collect();
    assert_eq!(summaries.len(), 2, "two bench summaries:\n{out}");

    let nodes0 = bench_summary_nodes(summaries[0]);
    let nodes1 = bench_summary_nodes(summaries[1]);
    assert!(
        nodes0 > 0,
        "the synthetic-network bench searched real nodes:\n{out}"
    );
    assert_eq!(
        nodes0, nodes1,
        "two in-process bench runs must report identical total nodes:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn two_process_launches_report_identical_nodes() {
    let dir = TempDir::new("bench-determinism-2proc");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 path");

    // Two independent driver runs (separate "process launches") with identical
    // input must report the same total nodes — determinism does not depend on
    // in-process carry-over.
    let a = bench_session(evaldir, "bench 16 1 3 default depth");
    let b = bench_session(evaldir, "bench 16 1 3 default depth");
    let na = bench_summary_nodes(&a);
    let nb = bench_summary_nodes(&b);
    assert!(na > 0, "real search:\n{a}");
    assert_eq!(
        na, nb,
        "two process launches must agree on total nodes\nA:\n{a}\nB:\n{b}"
    );
}

// -------------------------------------------------------------------------
// A threads=2 bench completes and reports a summary (no determinism).
// -------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn threads_two_bench_completes_and_reports() {
    let dir = TempDir::new("bench-threads2");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 path");

    let out = bench_session(evaldir, "bench 16 2 3 default depth");
    assert_eq!(
        bench_summary_positions(&out),
        4,
        "a threads=2 bench still runs all four default positions:\n{out}"
    );
    // No node-count assertion: multi-thread node totals vary run to run.
}
