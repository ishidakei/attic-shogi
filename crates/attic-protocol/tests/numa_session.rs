//! Session tests for the `NumaPolicy` option and the two NUMA information lines
//! (`engine.cpp`, `usi.cpp`).
//!
//! `NumaPolicy none` is used throughout so the allocation line never carries a
//! binding suffix and the assertions stay deterministic on any machine.

use std::sync::{Arc, Mutex};

use attic_protocol::UsiDriver;

fn drive(input: &str) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

#[test]
#[cfg_attr(miri, ignore)]
fn numa_policy_none_emits_both_lines_threads_emits_allocation_only() {
    // At the NumaPolicy step the thread count is still the default 4.
    let out = drive(
        "usi\n\
         setoption name NumaPolicy value none\n\
         setoption name Threads value 2\n\
         quit\n",
    );

    // Exactly once: `Threads` does not repeat the config line.
    let processor_lines = out
        .lines()
        .filter(|l| l.starts_with("info string Available processors:"))
        .count();
    assert_eq!(
        processor_lines, 1,
        "NumaPolicy emits the config line once; Threads does not: {out:?}"
    );

    assert!(
        out.contains("info string Using 4 threads\n"),
        "NumaPolicy emits the allocation line (4 threads): {out:?}"
    );
    assert!(
        out.contains("info string Using 2 threads\n"),
        "Threads emits the allocation line (2 threads): {out:?}"
    );

    assert!(
        !out.contains("with NUMA node thread binding"),
        "none policy must not bind: {out:?}"
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
