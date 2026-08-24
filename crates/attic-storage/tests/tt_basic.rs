//! Gate tests for the transposition-table port
//! (`crates/attic-storage/src/tt.rs`), checked against the semantics of
//! `source/tt.cpp`.
//!
//! # Addressing model used throughout
//!
//! With `resize(1)` the table holds `1 · 1024 · 1024 / 32 = 32768 = 2¹⁵`
//! clusters, so
//!
//! ```text
//! cluster_index_pre_side = mul_hi64(key, 32768) = (key · 2¹⁵) >> 64 = key >> 49
//! in_cluster_key_frag    = key & 0xffff
//! ```
//!
//! Bits `49..64` (the top 15) select the cluster and bits `0..16` are the
//! stored key fragment — disjoint ranges, so [`key`] can set each independently
//! (middle bits kept zero). The side-to-move is OR-ed into cluster-index bit 0,
//! so fixing `hi` **and** `side` keeps a family of keys in one cluster while the
//! fragment varies.

use attic_storage::{Bound, DEPTH_NONE, TTData, TranspositionTable};

/// Build a key that lands in cluster `hi` (before the side fold) with
/// in-cluster fragment `frag`. Requires `hi < 2¹⁵`.
fn key(hi: u64, frag: u16) -> u64 {
    assert!(hi < (1 << 15));
    (hi << 49) | frag as u64
}

/// Probe `k` and, on the returned writer, store `(depth, mv, ...)`. Uses the
/// table's current generation, as a real caller would.
#[allow(clippy::too_many_arguments)]
fn store(
    tt: &mut TranspositionTable,
    k: u64,
    side: u8,
    value: i32,
    pv: bool,
    bound: Bound,
    depth: i32,
    mv: u16,
    eval: i32,
) {
    let generation = tt.generation();
    let (_, _, w) = tt.probe(k, side);
    w.write(k, value, pv, bound, depth, mv, eval, generation);
}

#[test]
#[cfg_attr(miri, ignore)]
fn store_probe_round_trip_every_field() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);

    let k = key(100, 0x1234);
    let side = 0;

    // Miss on an empty table.
    let (found, data, w) = tt.probe(k, side);
    assert!(!found);
    assert_eq!(data, miss_sentinel());
    w.write(
        k,
        -321,
        true,
        Bound::Lower,
        17,
        0x0abc,
        -654,
        tt_generation_zero(),
    );

    // Hit: every field survives the round trip.
    let (found, data, _) = tt.probe(k, side);
    assert!(found);
    assert_eq!(data.value, -321);
    assert_eq!(data.eval, -654);
    assert_eq!(data.depth, 17);
    assert_eq!(data.bound, Bound::Lower);
    assert!(data.is_pv);
    assert_eq!(data.move16, 0x0abc);
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_bound_and_pv_combination_round_trips() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 1;

    for (i, bound) in [Bound::None, Bound::Upper, Bound::Lower, Bound::Exact]
        .into_iter()
        .enumerate()
    {
        for pv in [false, true] {
            // Distinct fragment per case so they don't overwrite each other.
            let frag = 0x100 + (i as u16) * 2 + pv as u16;
            let k = key(7, frag);
            store(&mut tt, k, side, 10 + i as i32, pv, bound, 5, frag, -10);
            let (found, data, _) = tt.probe(k, side);
            assert!(found, "bound={bound:?} pv={pv} should be found");
            assert_eq!(data.bound, bound);
            assert_eq!(data.is_pv, pv);
            assert_eq!(data.value, 10 + i as i32);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn miss_on_wrong_key() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;

    let stored = key(42, 0xBEEF);
    store(
        &mut tt,
        stored,
        side,
        100,
        false,
        Bound::Exact,
        8,
        0x0111,
        100,
    );

    // Same cluster, different fragment → miss.
    let (found, data, _) = tt.probe(key(42, 0xBEE0), side);
    assert!(!found);
    assert_eq!(data, miss_sentinel());

    // Different cluster entirely → miss.
    let (found, _, _) = tt.probe(key(99, 0xBEEF), side);
    assert!(!found);

    // Same key but opposite side lands in a different cluster → miss.
    let (found, _, _) = tt.probe(stored, 1);
    assert!(!found);
}

#[test]
#[cfg_attr(miri, ignore)]
fn replacement_evicts_lowest_priority_entry() {
    // Fill one three-entry cluster, all written at generation 0 so every
    // relative_age is 0 and replace_priority == depth8 == depth − DEPTH_NONE.
    //
    //   slot 0: frag 1, depth 10 → depth8 13, priority 13
    //   slot 1: frag 2, depth  5 → depth8  8, priority  8   ← lowest
    //   slot 2: frag 3, depth 20 → depth8 23, priority 23
    //
    // A miss then replaces the lowest-priority entry, i.e. slot 1 (frag 2).
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let hi = 100;

    store(&mut tt, key(hi, 1), side, 0, false, Bound::Lower, 10, 1, 0);
    store(&mut tt, key(hi, 2), side, 0, false, Bound::Lower, 5, 2, 0);
    store(&mut tt, key(hi, 3), side, 0, false, Bound::Lower, 20, 3, 0);

    // All three present before the eviction.
    assert!(tt.probe(key(hi, 1), side).0);
    assert!(tt.probe(key(hi, 2), side).0);
    assert!(tt.probe(key(hi, 3), side).0);

    // Miss on frag 4 → writer targets the evicted slot; write frag 4 there.
    store(&mut tt, key(hi, 4), side, 0, false, Bound::Lower, 1, 4, 0);

    // frag 2 (depth 5) was the least valuable and is gone; frags 1, 3, 4 remain.
    assert!(
        !tt.probe(key(hi, 2), side).0,
        "frag 2 should have been evicted"
    );
    assert!(tt.probe(key(hi, 1), side).0, "frag 1 should survive");
    assert!(tt.probe(key(hi, 3), side).0, "frag 3 should survive");
    assert!(tt.probe(key(hi, 4), side).0, "frag 4 should now be present");
}

#[test]
#[cfg_attr(miri, ignore)]
fn generation_aging_lowers_replacement_priority() {
    // A deep-but-old entry loses to a shallow-but-fresh one once enough
    // generations pass, because replace_priority = depth8 − 8·relative_age.
    //
    // After three new_search() bumps the table is at generation 3:
    //   P: frag 1, depth 20, gen 0 → depth8 23, age 3, priority 23 − 24 = −1  ← lowest
    //   Q: frag 2, depth  3, gen 3 → depth8  6, age 0, priority  6
    //   R: frag 3, depth  8, gen 3 → depth8 11, age 0, priority 11
    //
    // Without aging P's priority would be 23 (highest, never evicted); aging
    // flips it to the lowest, so the miss evicts P.
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let hi = 200;

    // P written at generation 0.
    assert_eq!(tt.generation(), 0);
    store(&mut tt, key(hi, 1), side, 0, false, Bound::Lower, 20, 1, 0);

    // Advance to generation 3, then write Q and R.
    tt.new_search();
    tt.new_search();
    tt.new_search();
    assert_eq!(tt.generation(), 3);
    store(&mut tt, key(hi, 2), side, 0, false, Bound::Lower, 3, 2, 0);
    store(&mut tt, key(hi, 3), side, 0, false, Bound::Lower, 8, 3, 0);

    // Sanity: all three occupy the cluster.
    assert!(tt.probe(key(hi, 1), side).0);
    assert!(tt.probe(key(hi, 2), side).0);
    assert!(tt.probe(key(hi, 3), side).0);

    // Miss → evicts the aged, deep entry P (frag 1).
    store(&mut tt, key(hi, 4), side, 0, false, Bound::Lower, 1, 4, 3);
    assert!(
        !tt.probe(key(hi, 1), side).0,
        "aged deep entry P should be evicted"
    );
    assert!(
        tt.probe(key(hi, 2), side).0,
        "fresh shallow entry Q should survive"
    );
    assert!(tt.probe(key(hi, 3), side).0, "fresh entry R should survive");
}

#[test]
#[cfg_attr(miri, ignore)]
fn save_preserves_move_when_new_move_absent() {
    // The reference keeps the old move if the incoming move is none (0) and the
    // key still matches. A second write to the same key with mv = 0 must keep
    // the earlier move but refresh the value.
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let k = key(300, 0x55);

    store(&mut tt, k, side, 1, false, Bound::Lower, 10, 0x0777, 1);
    // Same key, mv = 0, deeper: refreshes value, keeps move.
    store(&mut tt, k, side, 2, false, Bound::Lower, 12, 0, 2);

    let (found, data, _) = tt.probe(k, side);
    assert!(found);
    assert_eq!(
        data.move16, 0x0777,
        "old move retained when new move is absent"
    );
    assert_eq!(data.value, 2, "value refreshed");
    assert_eq!(data.depth, 12, "depth refreshed");
}

#[test]
#[cfg_attr(miri, ignore)]
fn resize_sizes_by_formula_and_clears() {
    let mut tt = TranspositionTable::new();
    assert_eq!(tt.cluster_count(), 0, "fresh table is empty");

    // clusterCount = mb · 1024 · 1024 / sizeof(Cluster=32) = mb · 32768.
    tt.resize(1);
    assert_eq!(tt.cluster_count(), 32_768);
    tt.resize(4);
    assert_eq!(tt.cluster_count(), 4 * 32_768);

    // Store, then a *different* size reallocates and clears.
    let k = key(1, 0x1);
    store(&mut tt, k, 0, 5, false, Bound::Exact, 9, 0x1, 5);
    assert!(tt.probe(k, 0).0);

    tt.resize(2);
    assert_eq!(tt.cluster_count(), 2 * 32_768);
    assert!(!tt.probe(k, 0).0, "resize to a new size clears the table");

    // Re-store works after resize.
    store(&mut tt, k, 0, 7, false, Bound::Exact, 9, 0x1, 7);
    assert_eq!(tt.probe(k, 0).1.value, 7);
}

#[test]
#[cfg_attr(miri, ignore)]
fn resize_to_same_size_is_a_no_op() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let k = key(5, 0x9);
    store(&mut tt, k, 0, 5, false, Bound::Exact, 9, 0x9, 5);
    let before = tt.checksum();

    // Requesting the same MiB yields the same cluster count → no reallocation,
    // no clear (faithful to the reference's early return).
    tt.resize(1);
    assert_eq!(tt.cluster_count(), 32_768);
    assert_eq!(
        tt.checksum(),
        before,
        "same-size resize leaves the table intact"
    );
    assert!(tt.probe(k, 0).0);
}

#[test]
#[cfg_attr(miri, ignore)]
fn clear_zeroes_entries_and_generation() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    tt.new_search();
    let k = key(11, 0x22);
    store(&mut tt, k, 0, 5, false, Bound::Exact, 9, 0x22, 5);
    assert!(tt.probe(k, 0).0);

    tt.clear();
    assert_eq!(tt.generation(), 0, "clear resets generation");
    assert!(!tt.probe(k, 0).0, "clear empties every entry");
}

#[test]
fn new_search_wraps_within_five_bits() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    // 32 bumps wrap 0 → 0 (generation is 5 bits: 0..=31).
    for _ in 0..31 {
        tt.new_search();
    }
    assert_eq!(tt.generation(), 31);
    tt.new_search();
    assert_eq!(tt.generation(), 0, "generation wraps at 2^5");
}

#[test]
#[cfg_attr(miri, ignore)]
fn determinism_identical_sequences_yield_identical_tables() {
    // Two tables driven by byte-identical operation sequences must end in the
    // same state, verified by a checksum over all entries.
    fn run() -> TranspositionTable {
        let mut tt = TranspositionTable::new();
        tt.resize(1);
        for round in 0..4 {
            tt.new_search();
            for f in 1..=6u16 {
                let k = key((f as u64) * 3, f.wrapping_mul(37).wrapping_add(1));
                store(
                    &mut tt,
                    k,
                    (f & 1) as u8,
                    (round * 100 + f as i32) - 250,
                    f % 2 == 0,
                    Bound::Lower,
                    (f as i32) + round,
                    f,
                    round * 10,
                );
            }
        }
        tt
    }

    assert_eq!(run().checksum(), run().checksum());
}

/// Alignment the huge-page-backed TT allocation uses on this target: a 2 MiB
/// huge-page boundary on Linux, a 4 KiB page boundary elsewhere. Mirrors the
/// private `TT_ALLOC_ALIGN` in `src/tt.rs`.
const EXPECTED_TT_ALIGN: usize = if cfg!(target_os = "linux") {
    2 * 1024 * 1024
} else {
    4096
};

#[test]
#[cfg_attr(miri, ignore)]
fn resized_table_base_pointer_is_page_aligned() {
    let mut tt = TranspositionTable::new();
    assert_eq!(tt.backing_ptr_addr(), 0, "unsized table reports no address");

    for &mb in &[1usize, 2, 8, 64] {
        tt.resize(mb);
        let addr = tt.backing_ptr_addr();
        assert_ne!(addr, 0);
        assert_eq!(
            addr % EXPECTED_TT_ALIGN,
            0,
            "TT base for {mb} MiB not {EXPECTED_TT_ALIGN}-aligned",
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn fresh_resize_reads_back_all_misses() {
    // The huge-page allocation is `alloc_zeroed`, so a freshly resized table is
    // fully unoccupied — every probe across the sampled clusters is a miss with
    // the sentinel payload, exactly like the reference's post-resize clear.
    let mut tt = TranspositionTable::new();
    tt.resize(2);
    for hi in 0..2048u64 {
        for side in 0..2u8 {
            // A nonzero key fragment cannot match the zeroed entries' `key == 0`,
            // so probe takes the true miss path and returns the sentinel.
            let k = key(hi & 0x7fff, (hi as u16).wrapping_mul(7) | 1);
            let (found, data, _) = tt.probe(k, side);
            assert!(!found, "fresh table entry occupied at hi={hi}");
            assert_eq!(data, miss_sentinel());
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn resize_grow_shrink_same_cycles_preserve_semantics() {
    // Walk grow → shrink → same across several sizes; after each *change* the
    // table is a valid, cleared allocation, and a same-size request is a
    // no-op that leaves stored data intact.
    let mut tt = TranspositionTable::new();
    let k = key(3, 0xabcd);

    for &mb in &[1usize, 4, 2, 8, 8, 1, 1, 16] {
        let prev_count = tt.cluster_count();
        let prev_sum = tt.checksum();
        tt.resize(mb);
        assert_eq!(tt.cluster_count(), mb * 32_768);

        if tt.cluster_count() == prev_count {
            // Same size → untouched (no realloc, no clear).
            assert_eq!(tt.checksum(), prev_sum, "same-size resize must be a no-op");
        } else {
            // Changed size → aligned, cleared allocation.
            assert_eq!(tt.backing_ptr_addr() % EXPECTED_TT_ALIGN, 0);
            assert!(!tt.probe(k, 0).0, "grow/shrink clears the table");
            // Store survives until the next size change.
            store(&mut tt, k, 0, 42, false, Bound::Exact, 9, 0xabcd, 42);
            assert_eq!(tt.probe(k, 0).1.value, 42);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn many_resizes_do_not_leak_or_corrupt() {
    // Repeatedly reallocate under the new aligned layout; each drop frees with
    // the matching layout. A double free / bad layout would trip the allocator
    // here, and the final store/probe proves the last allocation is sound.
    let mut tt = TranspositionTable::new();
    for i in 0..40u64 {
        let mb = 1 + (i % 4) as usize; // cycles 1,2,3,4 → forces real reallocs
        tt.resize(mb);
        let k = key(i & 0x7fff, i as u16);
        store(&mut tt, k, 0, 3, false, Bound::Exact, 5, i as u16, 3);
        assert!(tt.probe(k, 0).0);
    }
}

/// Linux-only, best-effort THP-uptake diagnostic (not a hard gate): after a
/// ≥ 64 MiB resize, scan `/proc/self/smaps` for the mapping that contains the
/// TT base address and report its `AnonHugePages` figure, so transparent-
/// huge-page adoption is visible. Never fails the suite — when THP is
/// disabled (`AnonHugePages: 0 kB`) it simply reports zero.
#[test]
#[cfg(target_os = "linux")]
#[cfg_attr(miri, ignore)]
fn thp_uptake_diagnostic_over_64mib() {
    use std::fs;

    let mut tt = TranspositionTable::new();
    tt.resize(64); // ≥ 64 MiB as required by the gate
    let base = tt.backing_ptr_addr();
    assert_ne!(base, 0);

    let smaps = match fs::read_to_string("/proc/self/smaps") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("THP diagnostic: /proc/self/smaps unreadable ({e}); skipping");
            return;
        }
    };

    // smaps is a sequence of blocks, each headed by `start-end perms ...`. Find
    // the block whose address range contains `base`, then read its
    // `AnonHugePages:` line.
    // Parse the `start-end` in a smaps block header, returning whether `base`
    // falls inside the range.
    fn header_contains(line: &str, base: u64) -> Option<bool> {
        let range = line
            .split_once(' ')
            .map(|(r, _)| r)
            .filter(|r| r.contains('-'))?;
        let (start, end) = range.split_once('-')?;
        let s = u64::from_str_radix(start, 16).ok()?;
        let e = u64::from_str_radix(end, 16).ok()?;
        Some((s..e).contains(&base))
    }

    let mut in_region = false;
    let mut anon_huge_kb: Option<u64> = None;
    for line in smaps.lines() {
        if let Some(contains) = header_contains(line, base as u64) {
            in_region = contains;
        } else if in_region && let Some(rest) = line.strip_prefix("AnonHugePages:") {
            anon_huge_kb = rest
                .trim()
                .strip_suffix(" kB")
                .and_then(|n| n.trim().parse::<u64>().ok());
            break;
        }
    }

    match anon_huge_kb {
        Some(kb) => eprintln!(
            "THP diagnostic: TT region at {base:#x} backed by {kb} kB AnonHugePages \
             (0 means THP disabled or not yet faulted in)"
        ),
        None => eprintln!(
            "THP diagnostic: no AnonHugePages line found for TT region at {base:#x} \
             (kernel without THP accounting); skipping"
        ),
    }
    // Intentionally no assertion: THP availability is environmental.
}

/// The `TTData` a miss returns (mirrors the reference's miss sentinel).
fn miss_sentinel() -> TTData {
    TTData {
        move16: 0,
        value: attic_storage::VALUE_NONE,
        eval: attic_storage::VALUE_NONE,
        depth: DEPTH_NONE,
        bound: Bound::None,
        is_pv: false,
    }
}

/// Generation of a freshly-resized table is 0; a tiny helper to make the
/// round-trip test read clearly.
fn tt_generation_zero() -> u8 {
    0
}
