//! Session tests for the `bench` command (`benchmark.cpp`, `usi.cpp`).
//!
//! The syntax tests run without a network: each position resigns instantly, so
//! they exercise the argument parse and summary plumbing without a real search.

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

#[test]
#[cfg_attr(miri, ignore)]
fn bare_bench_runs_the_four_default_positions() {
    // With no network each of the four default positions resigns instantly.
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
    let out = drive("bench 16 1 5 default perft\nquit\n");
    assert!(
        out.contains("info string bench: unsupported limit type `perft`"),
        "expected the scope-divergence notice:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn current_source_benches_the_set_position() {
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

/// A small fixed-depth default bench against a staged synthetic network.
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

    // Two runs in one process. Each resets the TT and histories at its start, so
    // the second sees the same clean state as the first.
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

    // Determinism must not depend on in-process carry-over.
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
}
