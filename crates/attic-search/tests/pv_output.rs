//! Unit tests for the fail-high/low PV-output gate (`fail_lh_pv_gate`), the pure
//! predicate behind the reference `main_manager()->pv()` fail-LH call site
//! (`yaneuraou-search.cpp`). Ported per the pin, which gates on
//! `mainThread && multiPV == 1 && fail-bound && nodes > 10M && (rootDepth < 3 ||
//! interval elapsed) && OutputFailLHPV`.

use attic_search::fail_lh_pv_gate;

/// The all-true baseline: a main-thread, single-PV, over-beta fail high past the
/// node floor with the interval elapsed and the option on.
fn baseline() -> bool {
    fail_lh_pv_gate(
        true,       // main_thread
        1,          // multi_pv
        200,        // best_value (>= beta ⇒ fail high)
        -100,       // alpha
        150,        // beta
        20_000_000, // nodes (> 10M)
        5,          // root_depth (>= 3)
        true,       // interval_elapsed
        true,       // output_fail_lh_pv
    )
}

#[test]
fn all_conditions_met_prints() {
    assert!(baseline());
}

#[test]
fn requires_main_thread() {
    assert!(!fail_lh_pv_gate(
        false, 1, 200, -100, 150, 20_000_000, 5, true, true
    ));
}

#[test]
fn requires_single_pv() {
    assert!(!fail_lh_pv_gate(
        true, 2, 200, -100, 150, 20_000_000, 5, true, true
    ));
}

#[test]
fn requires_a_fail_bound() {
    // A value strictly inside (alpha, beta) is neither a fail high nor a fail low.
    assert!(!fail_lh_pv_gate(
        true, 1, 0, -100, 150, 20_000_000, 5, true, true
    ));
    // Fail low (best_value <= alpha) also prints.
    assert!(fail_lh_pv_gate(
        true, 1, -100, -100, 150, 20_000_000, 5, true, true
    ));
}

#[test]
fn requires_the_node_floor() {
    assert!(!fail_lh_pv_gate(
        true, 1, 200, -100, 150, 10_000_000, 5, true, true
    ));
    assert!(fail_lh_pv_gate(
        true, 1, 200, -100, 150, 10_000_001, 5, true, true
    ));
}

#[test]
fn shallow_depth_bypasses_the_interval_gate() {
    // rootDepth < 3 prints even when the interval has NOT elapsed.
    assert!(fail_lh_pv_gate(
        true, 1, 200, -100, 150, 20_000_000, 2, false, true
    ));
    // At depth >= 3 the interval must have elapsed.
    assert!(!fail_lh_pv_gate(
        true, 1, 200, -100, 150, 20_000_000, 3, false, true
    ));
}

#[test]
fn requires_output_fail_lh_pv() {
    assert!(!fail_lh_pv_gate(
        true, 1, 200, -100, 150, 20_000_000, 5, true, false
    ));
}
