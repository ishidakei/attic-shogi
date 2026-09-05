//! Fixture-backed tests for the sysfs-driven detection path, plus a best-effort
//! smoke test against the live `/sys` tree on Linux.
//!
//! Each fixture is a committed miniature sysfs tree under `tests/fixtures/`, so
//! no test touches the real machine topology except the Linux-gated smoke test.
//!
//! Every test here is ignored under miri: reading a fixture opens a file, which
//! miri's isolation mode does not support. The detection logic is pure parsing
//! over the fixture bytes, and the crate's unit tests cover it without a
//! filesystem.

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

#[test]
#[cfg_attr(miri, ignore)]
fn system_numa_two_nodes() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 2);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    assert_eq!(cfg.nodes()[1], set(&[4, 5, 6, 7]));
    assert!(cfg.requires_memory_replication());
}

#[test]
#[cfg_attr(miri, ignore)]
fn missing_cpulist_falls_back_to_single_node() {
    // A missing per-node `cpulist` discards the partial config entirely.
    let o = opts("missing_cpulist", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 1);
    assert_eq!(cfg.nodes()[0], all_cpus(8));
}

#[test]
#[cfg_attr(miri, ignore)]
fn missing_online_falls_back_to_single_node() {
    let o = opts("no_online", all_cpus(4), 4);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 1);
    assert_eq!(cfg.nodes()[0], all_cpus(4));
}

#[test]
#[cfg_attr(miri, ignore)]
fn respect_affinity_filters_disallowed_cpus() {
    // node1's CPUs are all disallowed, so it becomes empty and is removed.
    let o = opts("two_node_l3", set(&[0, 1, 2, 3]), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 1);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    assert!(!cfg.is_custom_affinity());
}

#[test]
#[cfg_attr(miri, ignore)]
fn hardware_policy_ignores_affinity_but_marks_custom() {
    let o = opts("two_node_l3", set(&[0, 1, 2, 3]), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, false, &o);
    assert_eq!(cfg.num_numa_nodes(), 2);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    assert_eq!(cfg.nodes()[1], set(&[4, 5, 6, 7]));
    assert!(cfg.is_custom_affinity());
}

#[test]
#[cfg_attr(miri, ignore)]
fn l3_domains_policy_one_node_per_domain() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::L3Domains, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 4);
    assert_eq!(cfg.nodes()[0], set(&[0, 1]));
    assert_eq!(cfg.nodes()[1], set(&[2, 3]));
    assert_eq!(cfg.nodes()[2], set(&[4, 5]));
    assert_eq!(cfg.nodes()[3], set(&[6, 7]));
}

#[test]
#[cfg_attr(miri, ignore)]
fn bundled_l3_merges_within_system_node() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 4 }, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 2);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    assert_eq!(cfg.nodes()[1], set(&[4, 5, 6, 7]));
}

#[test]
#[cfg_attr(miri, ignore)]
fn bundled_l3_below_boundary_does_not_merge() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 2 }, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 4);
}

#[test]
#[cfg_attr(miri, ignore)]
fn bundled_l3_respects_affinity_filter() {
    let o = opts("two_node_l3", set(&[0, 1, 2, 3]), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 32 }, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 1);
    assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
}

#[test]
#[cfg_attr(miri, ignore)]
fn bundled_l3_logical_nodes_map_to_their_system_node() {
    // This bundle size merges each system node's two L3 domains back into one
    // logical node, so the logical and system indices coincide.
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 4 }, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 2);
    assert_eq!(cfg.system_node_of_logical(0, &o), 0);
    assert_eq!(cfg.system_node_of_logical(1, &o), 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn l3_bundled_logical_nodes_in_one_system_node_share_discriminator() {
    // This bundle size keeps all four L3 domains as distinct logical nodes, two
    // per system node — so two logical nodes map to the same discriminator, the
    // signal to share one network copy between them.
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 2 }, true, &o);
    assert_eq!(cfg.num_numa_nodes(), 4);
    assert_eq!(cfg.system_node_of_logical(0, &o), 0);
    assert_eq!(cfg.system_node_of_logical(1, &o), 0);
    assert_eq!(cfg.system_node_of_logical(2, &o), 1);
    assert_eq!(cfg.system_node_of_logical(3, &o), 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn system_nodes_for_binding_resolves_a_whole_assignment() {
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_sysfs(&NumaAutoPolicy::BundledL3 { bundle_size: 2 }, true, &o);
    let bound = [0usize, 1, 2, 3, 0];
    assert_eq!(
        cfg.system_nodes_for_binding(&bound, &o),
        vec![0, 0, 1, 1, 0]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn unassigned_cpu_falls_back_to_system_node_zero() {
    // The sole CPU does not appear in the fixture's system topology, so the
    // lookup falls back to system node 0.
    let o = opts("two_node_l3", all_cpus(8), 8);
    let cfg = NumaConfig::from_string("100").expect("valid custom config");
    assert_eq!(cfg.system_node_of_logical(0, &o), 0);
}

#[cfg(target_os = "linux")]
#[test]
#[cfg_attr(miri, ignore)]
fn smoke_from_system_real_sys() {
    // Structure only: the values are machine-specific.
    let cfg = NumaConfig::from_system(&NumaAutoPolicy::BundledL3 { bundle_size: 32 }, true);
    assert!(cfg.num_numa_nodes() >= 1);
    for n in 0..cfg.num_numa_nodes() {
        assert!(cfg.num_cpus_in_numa_node(n) >= 1);
    }
    assert!(cfg.num_cpus() >= 1);
}
