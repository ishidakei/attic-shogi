//! Fixture-backed tests for the sysfs-driven detection path, plus a best-effort
//! smoke test against the live `/sys` tree on Linux.
//!
//! Each fixture is a committed miniature sysfs tree under `tests/fixtures/`; the
//! injectable [`SysfsOptions::root`] points at one, so no test touches the real
//! machine topology (except the explicitly Linux-gated smoke test at the end).

use std::collections::BTreeSet;
use std::path::PathBuf;

use attic_numa::{CpuIndex, NumaAutoPolicy, NumaConfig, SysfsOptions};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn all_cpus(n: CpuIndex) -> BTreeSet<CpuIndex> {
    (0..n).collect()
}

fn set(cpus: &[CpuIndex]) -> BTreeSet<CpuIndex> {
    cpus.iter().copied().collect()
}

fn opts(name: &str, allowed: BTreeSet<CpuIndex>, system_threads: CpuIndex) -> SysfsOptions {
    SysfsOptions {
        root: fixture(name),
        allowed_cpus: allowed,
        system_threads,
    }
}

// -- system-NUMA parse from a fixture sysfs tree --------------------------

#[test]
fn system_numa_two_nodes() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 2);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    assert_eq!(cfg.nodes()[1], set(&[4, 5, 6, 7]));
    // Two nodes require replication even without a custom string.
    assert!(cfg.requires_memory_replication());
}

// -- missing-file fallbacks ----------------------------------------------

#[test]
fn missing_cpulist_falls_back_to_single_node() {
    // node1/cpulist is absent, so detection discards the partial config and
    // falls back to a single node with all allowed CPUs `0..system_threads`.
    let o = opts("missing_cpulist", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 1);
    assert_eq!(cfg.nodes()[0], all_cpus(8));
}

#[test]
fn missing_online_falls_back_to_single_node() {
    let o = opts("no_online", all_cpus(4), 4);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 1);
    assert_eq!(cfg.nodes()[0], all_cpus(4));
}

// -- affinity filtering: respect vs hardware ------------------------------

#[test]
fn respect_affinity_filters_disallowed_cpus() {
    // Only CPUs 0..3 are allowed; node1 (CPUs 4..7) becomes empty and is
    // removed.
    let o = opts("two_node_l3", set(&[0, 1, 2, 3]), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 1);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    assert!(!cfg.is_custom_affinity());
}

#[test]
fn hardware_policy_ignores_affinity_but_marks_custom() {
    // respect_affinity = false: the allowed set is ignored, so both nodes
    // survive; the result is flagged custom.
    let o = opts("two_node_l3", set(&[0, 1, 2, 3]), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, false, &o);
    assert_eq!(cfg.num_numa_nodes(), 2);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    assert_eq!(cfg.nodes()[1], set(&[4, 5, 6, 7]));
    assert!(cfg.is_custom_affinity());
}

// -- L3-aware detection + bundling from a fixture -------------------------

#[test]
fn l3_domains_policy_one_node_per_domain() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::L3Domains, true, &o);
    // Four L3 domains: {0,1}, {2,3}, {4,5}, {6,7}; no bundling.
    assert_eq!(cfg.num_numa_nodes(), 4);
    assert_eq!(cfg.nodes()[0], set(&[0, 1]));
    assert_eq!(cfg.nodes()[1], set(&[2, 3]));
    assert_eq!(cfg.nodes()[2], set(&[4, 5]));
    assert_eq!(cfg.nodes()[3], set(&[6, 7]));
}

#[test]
fn bundled_l3_merges_within_system_node() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 4 }, true, &o);
    // Within each system node the two 2-CPU domains merge (2 + 2 <= 4).
    assert_eq!(cfg.num_numa_nodes(), 2);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    assert_eq!(cfg.nodes()[1], set(&[4, 5, 6, 7]));
}

#[test]
fn bundled_l3_below_boundary_does_not_merge() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 2 }, true, &o);
    // 2 + 2 = 4 > 2, so no domains merge.
    assert_eq!(cfg.num_numa_nodes(), 4);
}

#[test]
fn bundled_l3_respects_affinity_filter() {
    // Only node0's CPUs are allowed; the L3 domains on node1 vanish.
    let o = opts("two_node_l3", set(&[0, 1, 2, 3]), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 32 }, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 1);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
}

// -- logical -> system NUMA node mapping (replication granularity) --------

#[test]
fn bundled_l3_logical_nodes_map_to_their_system_node() {
    // BundledL3{4} over two_node_l3 bundles each system node's two L3 domains
    // back into one logical node, so logical == system here: logical 0 -> system
    // 0, logical 1 -> system 1.
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 4 }, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 2);
    assert_eq!(cfg.system_node_of_logical(0, &o), 0);
    assert_eq!(cfg.system_node_of_logical(1, &o), 1);
}

#[test]
fn l3_bundled_logical_nodes_in_one_system_node_share_discriminator() {
    // BundledL3{2} keeps all four L3 domains as distinct logical nodes, but the
    // first two ({0,1},{2,3}) live on system node 0 and the last two on system
    // node 1. Two logical nodes sharing a system node therefore map to the same
    // discriminator — the signal to share one network copy between them.
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 2 }, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 4);
    assert_eq!(cfg.system_node_of_logical(0, &o), 0);
    assert_eq!(cfg.system_node_of_logical(1, &o), 0);
    assert_eq!(cfg.system_node_of_logical(2, &o), 1);
    assert_eq!(cfg.system_node_of_logical(3, &o), 1);
}

#[test]
fn system_nodes_for_binding_resolves_a_whole_assignment() {
    // A four-logical-node config (BundledL3{2}); a binding assignment that puts
    // workers on logical nodes [0, 1, 2, 3, 0] resolves to system nodes
    // [0, 0, 1, 1, 0].
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 2 }, true, &o);
    let bound = [0usize, 1, 2, 3, 0];
    assert_eq!(
        cfg.system_nodes_for_binding(&bound, &o),
        vec![0, 0, 1, 1, 0]
    );
}

#[test]
fn unassigned_cpu_falls_back_to_system_node_zero() {
    // A custom logical config whose sole CPU (100) does not appear in the
    // fixture's system topology; the lookup falls back to system node 0.
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_string("100").expect("valid custom config");
    assert_eq!(cfg.system_node_of_logical(0, &o), 0);
}

// -- best-effort Linux smoke ----------------------------------------------

#[cfg(target_os = "linux")]
#[test]
fn smoke_from_system_real_sys() {
    // Structure-only assertions: the real /sys parses without error and yields
    // at least one non-empty node. We do NOT assert any values, since those are
    // machine-specific.
    let cfg = NumaConfig::from_system(&NumaAutoPolicy::BundledL3 { bundle_size: 32 }, true);
    assert!(cfg.num_numa_nodes() >= 1);
    for n in 0..cfg.num_numa_nodes() {
        assert!(cfg.num_cpus_in_numa_node(n) >= 1);
    }
    assert!(cfg.num_cpus() >= 1);
}
