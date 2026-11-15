//! Session tests for the `NumaPolicy` option, the two NUMA information lines,
//! and when `Threads` / `NumaPolicy` take effect.
//!
//! `setoption` only stores `Threads`, `USI_Hash` and `NumaPolicy`; the next
//! successful `isready` applies them. The allocation line `NumaPolicy` prints
//! reports the applied pool — zero threads before the first `isready` — which
//! makes the deferral observable.
//!
//! `NumaPolicy none` is used wherever an allocation line is asserted, so it
//! never carries a binding suffix and the assertions stay deterministic on any
//! machine.

mod common;

use common::{TempDir, drive, write_synthetic_nn_bin};

fn allocation_lines(out: &str) -> Vec<&str> {
    out.lines()
        .filter(|l| l.starts_with("info string Using "))
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn numa_policy_none_emits_both_lines_threads_emits_nothing() {
    // No worker exists before the first `isready`.
    let out = drive(
        "usi\n\
         setoption name NumaPolicy value none\n\
         setoption name Threads value 2\n\
         quit\n",
    );

    let processor_lines = out
        .lines()
        .filter(|l| l.starts_with("info string Available processors:"))
        .count();
    assert_eq!(
        processor_lines, 1,
        "NumaPolicy emits the config line: {out:?}"
    );
    assert_eq!(
        allocation_lines(&out),
        ["info string Using 0 thread"],
        "NumaPolicy emits the allocation line; Threads emits nothing: {out:?}"
    );
    assert!(
        !out.contains("with NUMA node thread binding"),
        "none policy must not bind: {out:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn setoption_threads_alone_prints_nothing() {
    assert_eq!(
        drive(
            "setoption name Threads value 1\n\
             setoption name Threads value 3\n\
             setoption name USI_Hash value 16\n\
             quit\n"
        ),
        ""
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn threads_and_numa_policy_take_effect_at_isready() {
    let dir = TempDir::new("numa-deferred");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    // Each `NumaPolicy none` probes the live pool size. Before the first
    // `isready` no worker exists yet; `Threads 3` set after it waits for the
    // next `isready`, through a `usinewgame` in between.
    let out = drive(&format!(
        "setoption name EvalDir value {evaldir}\n\
         setoption name USI_Hash value 16\n\
         setoption name NumaPolicy value none\n\
         setoption name Threads value 2\n\
         isready\n\
         setoption name NumaPolicy value none\n\
         setoption name Threads value 3\n\
         setoption name NumaPolicy value none\n\
         usinewgame\n\
         setoption name NumaPolicy value none\n\
         isready\n\
         setoption name NumaPolicy value none\n\
         isready\n\
         setoption name NumaPolicy value none\n\
         quit\n"
    ));

    assert_eq!(out.matches("readyok\n").count(), 3, "{out}");
    assert_eq!(
        allocation_lines(&out),
        [
            "info string Using 0 thread",
            "info string Using 2 threads",
            "info string Using 2 threads",
            "info string Using 2 threads",
            "info string Using 3 threads",
            "info string Using 3 threads",
        ],
        "Threads applies at isready, not at setoption or usinewgame: {out}"
    );
    assert!(!out.contains("with NUMA node thread binding"), "{out}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn invalid_numa_policy_is_reported_and_keeps_the_previous_layout() {
    let dir = TempDir::new("numa-invalid");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    // The config line printed by a successful `NumaPolicy` reports the live
    // layout, so the closing `none` reads back what the rejected value left.
    let out = drive(&format!(
        "setoption name EvalDir value {evaldir}\n\
         setoption name USI_Hash value 16\n\
         setoption name Threads value 1\n\
         setoption name NumaPolicy value 0\n\
         isready\n\
         setoption name NumaPolicy value 0,0\n\
         setoption name NumaPolicy value 0-3x:\n\
         isready\n\
         setoption name NumaPolicy value none\n\
         quit\n"
    ));

    assert_eq!(out.matches("readyok\n").count(), 2, "{out}");
    let rejected: Vec<&str> = out
        .lines()
        .filter(|l| l.starts_with("info string NumaPolicy:"))
        .collect();
    assert_eq!(
        rejected,
        [
            "info string NumaPolicy: invalid value '0,0', keeping previous config.",
            "info string NumaPolicy: invalid value '0-3x:', keeping previous config.",
        ],
        "{out}"
    );
    let processor_lines: Vec<&str> = out
        .lines()
        .filter(|l| l.starts_with("info string Available processors:"))
        .collect();
    assert_eq!(
        processor_lines.len(),
        2,
        "a rejected value prints no layout: {out}"
    );
    assert_eq!(
        processor_lines[1], "info string Available processors: 0",
        "{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn numa_policy_config_line_matches_available_processors_format() {
    // The exact CPU set is machine-specific, so only the prefix is asserted.
    let out = drive("setoption name NumaPolicy value none\nquit\n");
    let line = out
        .lines()
        .find(|l| l.starts_with("info string Available processors:"))
        .unwrap_or_else(|| panic!("no Available processors line: {out:?}"));
    let list = line
        .strip_prefix("info string Available processors: ")
        .expect("prefix");
    assert!(
        !list.is_empty(),
        "processor list must be non-empty: {line:?}"
    );
}
