//! `bench` node counts against the reference engine (YaneuraOu @ 52f12e4f,
//! built with `cargo xtask build-reference` and run as
//! `YaneuraOu-by-gcc bench 1024 1 <limit> default <depth|nodes>` against the
//! same `nn.bin`). Every expected figure below is the reference's output, not
//! this engine's.
//!
//! The node-limited forms pin where a search stops: the reference tests the
//! node limit only every `min(512, limit / 1024)` interior search nodes, on a
//! counter that `usinewgame` resets and every later `go` resumes, so each
//! position's count also depends on where the positions before it stopped.
//!
//! The network file is staged locally and never committed, so when it is absent
//! each test prints a notice and passes.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

/// One `bench` run's figures: the searched nodes of each position, read from
/// the last `info … nodes N` line before its `bestmove`, and the summary total.
struct BenchRun {
    per_position: Vec<u64>,
    total: u64,
}

/// Run `bench_line` in a fresh engine process, or `None` when the network file
/// is absent.
fn run_bench(test: &str, bench_line: &str) -> Option<BenchRun> {
    let dir = eval_dir();
    if !dir.join("nn.bin").exists() {
        eprintln!(
            "skipping {test}: {} is not present (obtained out-of-band)",
            dir.join("nn.bin").display()
        );
        return None;
    }
    let eval_dir_arg = dir.to_str().expect("utf-8 eval dir");

    let mut child = Command::new(env!("CARGO_BIN_EXE_attic"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn engine");
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(
            format!("setoption name EvalDir value {eval_dir_arg}\nisready\n{bench_line}\nquit\n")
                .as_bytes(),
        )
        .expect("write engine stdin");
    let out = child.wait_with_output().expect("engine output");
    assert!(out.status.success(), "engine exited with {}", out.status);
    let out = String::from_utf8(out.stdout).expect("utf-8 engine output");

    let mut per_position = Vec::new();
    let mut last_nodes = None;
    let mut total = None;
    for line in out.lines() {
        if line.starts_with("info depth ") {
            let mut tokens = line.split_whitespace();
            if tokens.any(|t| t == "nodes") {
                last_nodes = tokens.next().and_then(|n| n.parse::<u64>().ok());
            }
        } else if line.starts_with("bestmove ") {
            per_position.push(
                last_nodes
                    .take()
                    .unwrap_or_else(|| panic!("no info nodes line before {line:?}")),
            );
        } else if let Some(rest) = line.strip_prefix("info string bench: ") {
            total = rest
                .split_whitespace()
                .find_map(|kv| kv.strip_prefix("nodes="))
                .and_then(|n| n.parse::<u64>().ok());
        }
    }
    let total = total.unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    Some(BenchRun {
        per_position,
        total,
    })
}

fn assert_bench(test: &str, bench_line: &str, per_position: &[u64], total: u64) {
    let Some(run) = run_bench(test, bench_line) else {
        return;
    };
    assert_eq!(
        run.per_position, per_position,
        "`{bench_line}` per-position nodes differ from the reference"
    );
    assert_eq!(
        run.total, total,
        "`{bench_line}` total differs from the reference"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn nodes_1m_bench_stops_where_the_reference_does() {
    assert_bench(
        "nodes_1m_bench_stops_where_the_reference_does",
        "bench 1024 1 1000000 default nodes",
        &[1_000_164, 1_000_774, 1_000_276, 1_000_080],
        4_001_294,
    );
}

/// Under a 100k limit the check interval shrinks to `100000 / 1024 = 97`.
#[test]
#[cfg_attr(miri, ignore)]
fn nodes_100k_bench_stops_where_the_reference_does() {
    assert_bench(
        "nodes_100k_bench_stops_where_the_reference_does",
        "bench 1024 1 100000 default nodes",
        &[100_052, 100_097, 100_105, 100_124],
        400_378,
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn depth_8_bench_matches_the_reference() {
    let Some(run) = run_bench(
        "depth_8_bench_matches_the_reference",
        "bench 1024 1 8 default depth",
    ) else {
        return;
    };
    assert_eq!(run.total, 23_580);
}

#[test]
#[cfg_attr(miri, ignore)]
fn depth_10_bench_matches_the_reference() {
    let Some(run) = run_bench(
        "depth_10_bench_matches_the_reference",
        "bench 1024 1 10 default depth",
    ) else {
        return;
    };
    assert_eq!(run.total, 61_507);
}
