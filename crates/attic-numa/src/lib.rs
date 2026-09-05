//! NUMA topology discovery — a Linux-only port of the reference engine's
//! `NumaConfig` machinery (`numa.h`).
//!
//! The crate describes the machine's NUMA layout and, on Linux, can bind the
//! calling thread to a node; memory replication is not modelled. On a
//! single-node machine no thread ever binds, since `auto` never suggests it.
//!
//! The reference's `_WIN64` paths are not ported. The pure parsing and topology
//! code runs everywhere; only [`startup_affinity`] and the default `/sys` root
//! of [`NumaConfig::from_system`] are meaningful on Linux, and both degrade to
//! an "all system threads" fallback elsewhere so the crate stays buildable and
//! testable off Linux.
//!
//! All sysfs readers take an injectable root path via [`SysfsOptions`], so tests
//! run against fixture directories rather than the live `/sys` tree.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A processor index, always as the operating system numbers it.
pub type CpuIndex = usize;

/// A logical NUMA-node index within a [`NumaConfig`]. These do *not* necessarily
/// match the system's own numbering: L3-aware subdivision, empty-node removal,
/// and custom configurations all renumber nodes.
pub type NumaIndex = usize;

/// How [`NumaConfig::from_system`] maps the machine to logical NUMA nodes
/// (`numa.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaAutoPolicy {
    /// Use the system's own NUMA nodes verbatim.
    SystemNuma,
    /// Use system-reported L3 cache domains, one logical node per domain.
    L3Domains,
    /// Group system-reported L3 domains (within each system NUMA node) until
    /// each bundle reaches `bundle_size` CPUs.
    BundledL3 { bundle_size: usize },
}

/// The engine's default policy: bundle L3 domains up to 32 CPUs
/// (`engine.cpp`).
pub const DEFAULT_POLICY: NumaAutoPolicy = NumaAutoPolicy::BundledL3 { bundle_size: 32 };

/// The fail-loud paths the reference resolves with `std::exit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumaError {
    /// A CPU index was assigned to a node while already owned by another. The
    /// reference `from_string` calls `std::exit(EXIT_FAILURE)` here.
    DuplicateCpu(CpuIndex),
}

impl fmt::Display for NumaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumaError::DuplicateCpu(c) => {
                write!(f, "CPU {c} is assigned to more than one NUMA node")
            }
        }
    }
}

impl std::error::Error for NumaError {}

/// Injectable inputs for the sysfs-driven detection path, so tests can
/// substitute a fixture directory and a synthetic affinity set.
#[derive(Debug, Clone)]
pub struct SysfsOptions {
    /// Root under which the `devices/system/...` hierarchy lives.
    pub root: PathBuf,
    /// The CPUs the process may run on, consulted only when a detection call
    /// passes `respect_affinity = true`.
    pub allowed_cpus: BTreeSet<CpuIndex>,
    /// The hardware-thread count to assume when falling back to a single node.
    pub system_threads: CpuIndex,
}

/// The CPUs sharing one L3, tagged with the *system* NUMA node they belong to.
#[derive(Debug, Default, Clone)]
struct L3Domain {
    system_numa_index: NumaIndex,
    cpus: BTreeSet<CpuIndex>,
}

/// An immutable description of the machine's NUMA layout (`numa.h`). Every
/// node it exposes is non-empty.
#[derive(Debug, Clone)]
pub struct NumaConfig {
    nodes: Vec<BTreeSet<CpuIndex>>,
    node_by_cpu: BTreeMap<CpuIndex, NumaIndex>,
    highest_cpu_index: CpuIndex,
    /// Set when the configuration may not match the current process affinity: a
    /// custom string, or `respect_affinity = false`.
    custom_affinity: bool,
}

impl Default for NumaConfig {
    /// A single node containing CPUs `0..system_threads()` (`numa.h`).
    fn default() -> Self {
        Self::new()
    }
}

impl NumaConfig {
    /// One node holding every hardware thread (`numa.h`).
    pub fn new() -> Self {
        let mut cfg = Self::empty();
        let num_cpus = system_threads();
        cfg.add_cpu_range_to_node(0, 0, num_cpus - 1);
        cfg
    }

    /// An empty configuration with no nodes (the reference's `empty()`).
    fn empty() -> Self {
        NumaConfig {
            nodes: Vec::new(),
            node_by_cpu: BTreeMap::new(),
            highest_cpu_index: 0,
            custom_affinity: false,
        }
    }

    /// Parse the reference's custom node syntax (`numa.h`), e.g.
    /// `"0-15,32-47:16-31,48-63"`: `':'` separates nodes, `','` separates
    /// entries, `"a-b"` is an inclusive range, and empty node groups are
    /// skipped. A repeated CPU is a [`NumaError::DuplicateCpu`].
    pub fn from_string(s: &str) -> Result<Self, NumaError> {
        let mut cfg = Self::empty();

        let mut n: NumaIndex = 0;
        for node_str in s.split(':') {
            let indices = indices_from_shortened_string(node_str);
            if !indices.is_empty() {
                for idx in indices {
                    if !cfg.add_cpu_to_node(n, idx) {
                        return Err(NumaError::DuplicateCpu(idx));
                    }
                }
                n += 1;
            }
        }

        cfg.custom_affinity = true;
        Ok(cfg)
    }

    /// Autodetect the NUMA layout from the live `/sys` tree, the real startup
    /// affinity snapshot, and the real hardware-thread count.
    pub fn from_system(policy: &NumaAutoPolicy, respect_affinity: bool) -> Self {
        let opts = SysfsOptions {
            root: PathBuf::from("/sys"),
            allowed_cpus: startup_affinity().clone(),
            system_threads: system_threads(),
        };
        Self::from_sysfs(policy, respect_affinity, &opts)
    }

    /// Autodetect the NUMA layout from an injectable sysfs root, mirroring the
    /// reference `from_system` Linux branch (`numa.h`).
    pub fn from_sysfs(
        policy: &NumaAutoPolicy,
        respect_affinity: bool,
        opts: &SysfsOptions,
    ) -> Self {
        let mut cfg = Self::empty();
        let mut l3_success = false;

        if !matches!(policy, NumaAutoPolicy::SystemNuma) {
            let bundle_size = match policy {
                NumaAutoPolicy::BundledL3 { bundle_size } => *bundle_size,
                _ => 0,
            };
            if let Some(l3_cfg) = try_get_l3_aware_config(opts, respect_affinity, bundle_size) {
                cfg = l3_cfg;
                l3_success = true;
            }
        }

        if !l3_success {
            cfg = from_system_numa(opts, respect_affinity);
        }

        cfg.remove_empty_numa_nodes();

        if !respect_affinity {
            cfg.custom_affinity = true;
        }

        cfg
    }

    /// Whether CPU `c` is assigned to some node (`numa.h`).
    pub fn is_cpu_assigned(&self, c: CpuIndex) -> bool {
        self.node_by_cpu.contains_key(&c)
    }

    /// The number of NUMA nodes (`numa.h`).
    pub fn num_numa_nodes(&self) -> NumaIndex {
        self.nodes.len()
    }

    /// The number of CPUs in node `n` (`numa.h`).
    ///
    /// # Panics
    /// If `n` is out of range, mirroring the reference `assert`.
    pub fn num_cpus_in_numa_node(&self, n: NumaIndex) -> CpuIndex {
        assert!(n < self.nodes.len());
        self.nodes[n].len()
    }

    /// The total number of assigned CPUs (`numa.h`).
    pub fn num_cpus(&self) -> CpuIndex {
        self.node_by_cpu.len()
    }

    /// Whether NUMA-replicated memory is required: a custom affinity, or more
    /// than one node (`numa.h`).
    pub fn requires_memory_replication(&self) -> bool {
        self.custom_affinity || self.nodes.len() > 1
    }

    /// The per-node CPU sets, in node order.
    pub fn nodes(&self) -> &[BTreeSet<CpuIndex>] {
        &self.nodes
    }

    /// The node owning CPU `c`, if any.
    pub fn node_of_cpu(&self, c: CpuIndex) -> Option<NumaIndex> {
        self.node_by_cpu.get(&c).copied()
    }

    /// The *system* NUMA node a logical node belongs to (`get_discriminator`,
    /// `numa.h`) — the granularity at which memory is replicated, so
    /// two logical nodes sharing one system node can share a single copy.
    ///
    /// The reference keys its shared-memory segment on the system topology as
    /// text prefixed to this index; with no shared-memory layer and one topology
    /// per process, the index alone is the discriminator here.
    ///
    /// # Panics
    /// If `idx` is out of range, mirroring the reference's `nodes[idx]`.
    pub fn system_node_of_logical(&self, idx: NumaIndex, opts: &SysfsOptions) -> NumaIndex {
        let cfg_sys = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, false, opts);
        self.system_node_of_logical_in(idx, &cfg_sys)
    }

    /// The system node of every worker's logical node, in `bound` order — a
    /// batch [`Self::system_node_of_logical`] that reads sysfs once.
    pub fn system_nodes_for_binding(
        &self,
        bound: &[NumaIndex],
        opts: &SysfsOptions,
    ) -> Vec<NumaIndex> {
        let cfg_sys = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, false, opts);
        bound
            .iter()
            .map(|&logical| self.system_node_of_logical_in(logical, &cfg_sys))
            .collect()
    }

    fn system_node_of_logical_in(&self, idx: NumaIndex, cfg_sys: &NumaConfig) -> NumaIndex {
        let cpu = *self.nodes[idx]
            .iter()
            .next()
            .expect("every exposed NUMA node is non-empty");
        cfg_sys.node_of_cpu(cpu).unwrap_or(0)
    }

    /// Whether the configuration is flagged custom: a custom string, or built
    /// without respecting process affinity.
    pub fn is_custom_affinity(&self) -> bool {
        self.custom_affinity
    }

    /// Whether the engine should distribute and bind its worker threads across
    /// NUMA nodes for the requested thread count (`numa.h`).
    ///
    /// A custom affinity always suggests binding, since the OS affinity may not
    /// match what the user asked for, and a single thread never binds. Otherwise
    /// binding is suggested when the threads cannot reasonably be contained by
    /// the largest node, or when there are enough of them to spread across the
    /// non-small nodes with minimal disparity.
    pub fn suggests_binding_threads(&self, num_threads: CpuIndex) -> bool {
        if self.custom_affinity {
            return true;
        }

        if num_threads <= 1 {
            return false;
        }

        let largest_node_size = self.nodes.iter().map(|cpus| cpus.len()).max().unwrap_or(0);

        // The reference's `SmallNodeThreshold`. An empty node is always small.
        const SMALL_NODE_THRESHOLD: f64 = 0.6;
        let is_node_small = |node: &BTreeSet<CpuIndex>| {
            node.len() as f64 / largest_node_size as f64 <= SMALL_NODE_THRESHOLD
        };

        let num_not_small_nodes = self
            .nodes
            .iter()
            .filter(|cpus| !is_node_small(cpus))
            .count();

        (num_threads > largest_node_size / 2 || num_threads >= num_not_small_nodes * 4)
            && self.nodes.len() > 1
    }

    /// Assign each of `num_threads` worker threads to a NUMA node
    /// (`numa.h`), greedily filling the node that minimises
    /// `(occupation + 1) / node_size`, ties going to the lowest index. No node is
    /// favoured, so multiple engine instances do not all crowd node 0.
    pub fn distribute_threads_among_numa_nodes(&self, num_threads: CpuIndex) -> Vec<NumaIndex> {
        let mut ns: Vec<NumaIndex> = Vec::new();

        if self.nodes.len() == 1 {
            ns.resize(num_threads, 0);
            return ns;
        }

        let mut occupation = vec![0usize; self.nodes.len()];
        for _ in 0..num_threads {
            let mut best_node: NumaIndex = 0;
            let mut best_fill = f32::MAX;
            for (n, node) in self.nodes.iter().enumerate() {
                let fill = (occupation[n] + 1) as f32 / node.len() as f32;
                if fill < best_fill {
                    best_node = n;
                    best_fill = fill;
                }
            }
            ns.push(best_node);
            occupation[best_node] += 1;
        }

        ns
    }

    /// Bind the *current* thread to NUMA node `n`, restricting its CPU affinity
    /// to that node's CPUs (`numa.h`, Linux branch). A no-op off
    /// Linux.
    ///
    /// # Panics
    /// Where the reference reaches `std::exit(EXIT_FAILURE)`:
    /// * if `n` is out of range or the node is empty;
    /// * if `highest_cpu_index >= 1024` — a fixed 1024-CPU `cpu_set_t` is used
    ///   rather than the reference's dynamic `CPU_ALLOC`, so a CPU index that
    ///   would not fit is rejected rather than silently truncated;
    /// * if `sched_setaffinity` fails.
    pub fn bind_current_thread_to_numa_node(&self, n: NumaIndex) {
        if n >= self.nodes.len() || self.nodes[n].is_empty() {
            panic!(
                "bind_current_thread_to_numa_node: node {n} is out of range or empty \
                 (config has {} node(s))",
                self.nodes.len()
            );
        }
        bind_current_thread_to_cpus(self.highest_cpu_index, &self.nodes[n]);
    }

    /// Run `f` on a temporary thread bound to NUMA node `n`, then join it
    /// (`numa.h`), so a region the closure allocates and fills has its
    /// pages first-touched on `n`.
    ///
    /// Off Linux the bind is a no-op but the closure still runs on the temporary
    /// thread, so the control flow is identical across platforms.
    pub fn execute_on_numa_node<F>(&self, n: NumaIndex, f: F)
    where
        F: FnOnce() + Send,
    {
        std::thread::scope(|scope| {
            scope.spawn(|| {
                self.bind_current_thread_to_numa_node(n);
                f();
            });
        });
    }

    /// Drop any empty nodes, preserving the order of the rest
    /// (`numa.h`).
    fn remove_empty_numa_nodes(&mut self) {
        self.nodes.retain(|cpus| !cpus.is_empty());
        // The reference leaves `node_by_cpu` mapping CPUs to *pre-removal*
        // indices; rebuilding it here keeps the reverse map self-consistent, and
        // callers rebuild configs rather than mutate them, so nothing observes
        // the difference.
        self.node_by_cpu.clear();
        for (n, cpus) in self.nodes.iter().enumerate() {
            for &c in cpus {
                self.node_by_cpu.insert(c, n);
            }
        }
    }

    /// Assign CPU `c` to node `n`, returning `false` and leaving the structure
    /// unmodified if `c` is already assigned (`numa.h`).
    fn add_cpu_to_node(&mut self, n: NumaIndex, c: CpuIndex) -> bool {
        if self.is_cpu_assigned(c) {
            return false;
        }

        while self.nodes.len() <= n {
            self.nodes.push(BTreeSet::new());
        }

        self.nodes[n].insert(c);
        self.node_by_cpu.insert(c, n);

        if c > self.highest_cpu_index {
            self.highest_cpu_index = c;
        }

        true
    }

    /// Assign the inclusive CPU range `cfirst..=clast` to node `n`.
    /// All-or-nothing: `false` and unmodified if any CPU is already assigned
    /// (`numa.h`).
    fn add_cpu_range_to_node(&mut self, n: NumaIndex, cfirst: CpuIndex, clast: CpuIndex) -> bool {
        for c in cfirst..=clast {
            if self.is_cpu_assigned(c) {
                return false;
            }
        }

        while self.nodes.len() <= n {
            self.nodes.push(BTreeSet::new());
        }

        for c in cfirst..=clast {
            self.nodes[n].insert(c);
            self.node_by_cpu.insert(c, n);
        }

        if clast > self.highest_cpu_index {
            self.highest_cpu_index = clast;
        }

        true
    }
}

impl fmt::Display for NumaConfig {
    /// The canonical shortened form, re-compressing consecutive CPUs into
    /// `"a-b"` ranges (`numa.h`). `from_string(x.to_string())` reproduces
    /// `x`'s node structure.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut is_first_node = true;
        for cpus in &self.nodes {
            if !is_first_node {
                write!(f, ":")?;
            }

            let v: Vec<CpuIndex> = cpus.iter().copied().collect();
            let mut is_first_set = true;
            let mut range_start = 0usize; // index into `v`
            let mut i = 0usize;
            while i < v.len() {
                let at_range_end = i + 1 == v.len() || v[i + 1] != v[i] + 1;
                if at_range_end {
                    if !is_first_set {
                        write!(f, ",")?;
                    }
                    let last = v[i];
                    if i != range_start {
                        write!(f, "{}-{}", v[range_start], last)?;
                    } else {
                        write!(f, "{last}")?;
                    }
                    range_start = i + 1;
                    is_first_set = false;
                }
                i += 1;
            }

            is_first_node = false;
        }

        Ok(())
    }
}

/// Read a sysfs file under `root`. `None` when it cannot be opened, `Some("")`
/// for an empty file — the reference `read_file_to_string`
/// (`misc.cpp`) draws the same distinction.
fn read_sysfs(root: &Path, rel: &str) -> Option<String> {
    std::fs::read_to_string(root.join(rel)).ok()
}

/// Remove all ASCII whitespace from `s` (`misc.cpp`).
fn remove_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_ascii_whitespace()).collect()
}

/// Parse a single decimal index, tolerating surrounding whitespace and trailing
/// non-digits as the reference `str_to_size_t`'s `stoull` does
/// (`misc.cpp`).
fn parse_size_t(s: &str) -> Option<CpuIndex> {
    let t = s.trim_start();
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<CpuIndex>().ok()
}

/// Expand the reference's shortened index-list syntax into a flat list
/// (`numa.h`): `','` separates entries, each a single index or an
/// inclusive `"a-b"` range, with empty entries skipped.
fn indices_from_shortened_string(s: &str) -> Vec<CpuIndex> {
    let mut indices = Vec::new();

    if s.is_empty() {
        return indices;
    }

    for ss in s.split(',') {
        if ss.is_empty() {
            continue;
        }

        let parts: Vec<&str> = ss.split('-').collect();
        match parts.as_slice() {
            [single] => {
                if let Some(c) = parse_size_t(single) {
                    indices.push(c);
                }
            }
            [first, last] => {
                if let (Some(cfirst), Some(clast)) = (parse_size_t(first), parse_size_t(last)) {
                    for c in cfirst..=clast {
                        indices.push(c);
                    }
                }
            }
            // The reference handles only the 1- and 2-part cases.
            _ => {}
        }
    }

    indices
}

fn is_cpu_allowed(opts: &SysfsOptions, respect_affinity: bool, c: CpuIndex) -> bool {
    !respect_affinity || opts.allowed_cpus.contains(&c)
}

/// The system-NUMA sysfs config path (`numa.h`). A missing `online`
/// file, or a missing per-node `cpulist`, falls back to a single node holding
/// every allowed CPU.
fn from_system_numa(opts: &SysfsOptions, respect_affinity: bool) -> NumaConfig {
    let mut cfg = NumaConfig::empty();
    let mut use_fallback = false;

    match read_sysfs(&opts.root, "devices/system/node/online") {
        Some(node_ids) if !node_ids.is_empty() => {
            let node_ids = remove_whitespace(&node_ids);
            for n in indices_from_shortened_string(&node_ids) {
                let path = format!("devices/system/node/node{n}/cpulist");
                match read_sysfs(&opts.root, &path) {
                    // An empty node still has a whitespace-only file, and empty
                    // nodes are fine — only a missing file is a fallback.
                    None => {
                        use_fallback = true;
                        break;
                    }
                    Some(cpu_ids) => {
                        let cpu_ids = remove_whitespace(&cpu_ids);
                        for c in indices_from_shortened_string(&cpu_ids) {
                            if is_cpu_allowed(opts, respect_affinity, c) {
                                cfg.add_cpu_to_node(n, c);
                            }
                        }
                    }
                }
            }
        }
        _ => {
            use_fallback = true;
        }
    }

    if use_fallback {
        // The reference's `fallback()` likewise resets `cfg` to empty.
        cfg = NumaConfig::empty();
        for c in 0..opts.system_threads {
            if is_cpu_allowed(opts, respect_affinity, c) {
                cfg.add_cpu_to_node(0, c);
            }
        }
    }

    cfg
}

/// Attempt the L3-aware config path (`numa.h`): walk CPUs by "next
/// unseen CPU", reading each one's `cache/index3/shared_cpu_list` and tagging
/// its domain with the owning system NUMA node. `None` if no domains were found.
fn try_get_l3_aware_config(
    opts: &SysfsOptions,
    respect_affinity: bool,
    bundle_size: usize,
) -> Option<NumaConfig> {
    let system_config = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, respect_affinity, opts);

    let mut l3_domains: Vec<L3Domain> = Vec::new();
    let mut seen: BTreeSet<CpuIndex> = BTreeSet::new();

    // The scan really terminates on the first missing or empty sysfs file; this
    // bound only guards against a malformed tree that never grows `seen`.
    const MAX_CPU_SCAN: CpuIndex = 1 << 20;

    loop {
        let next = {
            let mut candidate = 0;
            while candidate < MAX_CPU_SCAN && seen.contains(&candidate) {
                candidate += 1;
            }
            candidate
        };
        if next >= MAX_CPU_SCAN {
            break;
        }

        let path = format!("devices/system/cpu/cpu{next}/cache/index3/shared_cpu_list");
        let siblings = match read_sysfs(&opts.root, &path) {
            Some(s) if !s.is_empty() => s,
            _ => break,
        };

        let mut domain = L3Domain::default();
        for c in indices_from_shortened_string(&siblings) {
            if is_cpu_allowed(opts, respect_affinity, c) {
                // The reference's `.at(c)`, a fail-loud lookup: on a consistent
                // system every allowed CPU is in the system-NUMA config.
                let sys_idx = *system_config
                    .node_by_cpu
                    .get(&c)
                    .expect("L3 CPU missing from system NUMA config");
                domain.system_numa_index = sys_idx;
                domain.cpus.insert(c);
            }
            seen.insert(c);
        }

        if !domain.cpus.is_empty() {
            l3_domains.push(domain);
        }
    }

    if !l3_domains.is_empty() {
        Some(from_l3_info(l3_domains, bundle_size))
    } else {
        None
    }
}

/// Bundle L3 domains into logical NUMA nodes (`numa.h`): group by
/// system NUMA node, then repeatedly merge adjacent pairs within each group
/// while `|a| + |b| <= bundle_size`.
fn from_l3_info(domains: Vec<L3Domain>, bundle_size: usize) -> NumaConfig {
    debug_assert!(!domains.is_empty());

    // A `BTreeMap` iterates keys in ascending order, like the reference's
    // `std::map`.
    let mut list: BTreeMap<NumaIndex, Vec<L3Domain>> = BTreeMap::new();
    for d in domains {
        list.entry(d.system_numa_index).or_default().push(d);
    }

    let mut cfg = NumaConfig::empty();
    let mut n: NumaIndex = 0;
    for (_, mut ds) in list {
        loop {
            let mut changed = false;
            let mut j = 0;
            while j + 1 < ds.len() {
                if ds[j].cpus.len() + ds[j + 1].cpus.len() <= bundle_size {
                    changed = true;
                    let mut next = ds.remove(j + 1);
                    ds[j].cpus.append(&mut next.cpus);
                }
                // `j` advances every iteration, as the reference for-loop does:
                // a just-merged node is not re-checked within the same pass.
                j += 1;
            }
            if !changed {
                break;
            }
        }

        for d in &ds {
            let dn = n;
            n += 1;
            for &cpu in &d.cpus {
                cfg.add_cpu_to_node(dn, cpu);
            }
        }
    }

    cfg
}

/// The number of usable hardware threads, at least 1 (`numa.h`).
pub fn system_threads() -> CpuIndex {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// The set of CPUs the process was allowed to run on *at startup*, captured
/// once (`STARTUP_PROCESSOR_AFFINITY`, `numa.h`) so detection does not
/// change behaviour as the live affinity changes over time. Off Linux it
/// degrades to all system threads.
pub fn startup_affinity() -> &'static BTreeSet<CpuIndex> {
    static STARTUP: OnceLock<BTreeSet<CpuIndex>> = OnceLock::new();
    STARTUP.get_or_init(capture_process_affinity)
}

#[cfg(target_os = "linux")]
fn capture_process_affinity() -> BTreeSet<CpuIndex> {
    // A fixed 1024-CPU `cpu_set_t`, narrower than the reference's
    // `CPU_ALLOC(1024 * 64)` (`numa.h`). A machine exceeding it fails
    // loud at bind time rather than silently mis-binding.
    let mut cpus = BTreeSet::new();
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let size = std::mem::size_of::<libc::cpu_set_t>();
        let status = libc::sched_getaffinity(0, size, &mut set as *mut libc::cpu_set_t);
        if status != 0 {
            // Assume all system threads rather than aborting from a library.
            return (0..system_threads()).collect();
        }
        for c in 0..(size * 8) {
            if libc::CPU_ISSET(c, &set) {
                cpus.insert(c);
            }
        }
    }
    cpus
}

#[cfg(not(target_os = "linux"))]
fn capture_process_affinity() -> BTreeSet<CpuIndex> {
    (0..system_threads()).collect()
}

/// The Linux affinity-setting core of
/// [`NumaConfig::bind_current_thread_to_numa_node`]. Fail-loud on every error
/// path, mirroring the reference `std::exit(EXIT_FAILURE)`.
#[cfg(target_os = "linux")]
fn bind_current_thread_to_cpus(highest_cpu_index: CpuIndex, cpus: &BTreeSet<CpuIndex>) {
    // A fixed 1024-CPU `cpu_set_t` instead of the reference's dynamic
    // `CPU_ALLOC(highestCpuIndex + 1)`, so an index that would not fit is a
    // fail-loud error rather than a silent out-of-bounds `CPU_SET`.
    assert!(
        highest_cpu_index < 1024,
        "bind_current_thread_to_numa_node: highest CPU index {highest_cpu_index} \
         exceeds this port's fixed 1024-CPU cpu_set_t capacity"
    );
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            libc::CPU_SET(c, &mut set);
        }
        let size = std::mem::size_of::<libc::cpu_set_t>();
        let status = libc::sched_setaffinity(0, size, &set as *const libc::cpu_set_t);
        if status != 0 {
            panic!(
                "bind_current_thread_to_numa_node: sched_setaffinity failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // The reference's defensive re-schedule, so the thread lands on the
        // newly-allowed CPUs promptly.
        libc::sched_yield();
    }
}

/// Non-Linux no-op counterpart of [`bind_current_thread_to_cpus`].
#[cfg(not(target_os = "linux"))]
fn bind_current_thread_to_cpus(_highest_cpu_index: CpuIndex, _cpus: &BTreeSet<CpuIndex>) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(cpus: &[CpuIndex]) -> BTreeSet<CpuIndex> {
        cpus.iter().copied().collect()
    }

    #[test]
    fn parse_simple_list_and_range() {
        assert_eq!(indices_from_shortened_string("0-3,8"), vec![0, 1, 2, 3, 8]);
        assert_eq!(indices_from_shortened_string("5"), vec![5]);
        assert_eq!(indices_from_shortened_string("2-2"), vec![2]);
    }

    #[test]
    fn parse_empty_and_empty_entries() {
        assert_eq!(indices_from_shortened_string(""), Vec::<CpuIndex>::new());
        assert_eq!(indices_from_shortened_string("0,,3"), vec![0, 3]);
    }

    #[test]
    fn parse_tolerates_whitespace() {
        assert_eq!(
            indices_from_shortened_string(&remove_whitespace(" 0-3 , 8 \n")),
            vec![0, 1, 2, 3, 8]
        );
        // Tolerated directly, as the reference relies on `stoull` doing.
        assert_eq!(indices_from_shortened_string("0-3\n"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn descending_range_is_empty() {
        assert_eq!(indices_from_shortened_string("5-3"), Vec::<CpuIndex>::new());
    }

    #[test]
    fn from_string_valid_two_nodes() {
        let cfg = NumaConfig::from_string("0-15,32-47:16-31,48-63").unwrap();
        assert_eq!(cfg.num_numa_nodes(), 2);
        assert_eq!(cfg.num_cpus_in_numa_node(0), 32);
        assert_eq!(cfg.num_cpus_in_numa_node(1), 32);
        assert!(cfg.is_cpu_assigned(0));
        assert!(cfg.is_cpu_assigned(63));
        assert!(!cfg.is_cpu_assigned(64));
        assert_eq!(cfg.node_of_cpu(32), Some(0));
        assert_eq!(cfg.node_of_cpu(16), Some(1));
        assert!(cfg.is_custom_affinity());
        assert!(cfg.requires_memory_replication());
    }

    #[test]
    fn from_string_empty_groups_are_skipped() {
        let cfg = NumaConfig::from_string("0-3::4-7").unwrap();
        assert_eq!(cfg.num_numa_nodes(), 2);
        assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
        assert_eq!(cfg.nodes()[1], set(&[4, 5, 6, 7]));
    }

    #[test]
    fn from_string_duplicate_cpu_within_node_fails() {
        assert!(matches!(
            NumaConfig::from_string("0,0"),
            Err(NumaError::DuplicateCpu(0))
        ));
    }

    #[test]
    fn from_string_duplicate_cpu_across_nodes_fails() {
        assert!(matches!(
            NumaConfig::from_string("0-3:3-5"),
            Err(NumaError::DuplicateCpu(3))
        ));
    }

    #[test]
    fn from_string_empty_is_empty_custom_config() {
        let cfg = NumaConfig::from_string("").unwrap();
        assert_eq!(cfg.num_numa_nodes(), 0);
        assert!(cfg.is_custom_affinity());
        assert!(cfg.requires_memory_replication());
    }

    #[test]
    fn to_string_canonical_range_compression() {
        let cfg = NumaConfig::from_string("0,1,2,3,8:16-31").unwrap();
        assert_eq!(cfg.to_string(), "0-3,8:16-31");
    }

    #[test]
    fn to_string_single_cpu_nodes() {
        let cfg = NumaConfig::from_string("0:5:9").unwrap();
        assert_eq!(cfg.to_string(), "0:5:9");
    }

    #[test]
    fn to_string_round_trip() {
        for s in ["0-3,8:16-31", "0:1:2", "0-63", "0,2,4,6"] {
            let cfg = NumaConfig::from_string(s).unwrap();
            let round = NumaConfig::from_string(&cfg.to_string()).unwrap();
            assert_eq!(cfg.to_string(), round.to_string());
            assert_eq!(cfg.nodes(), round.nodes());
        }
    }

    fn domain(sys: NumaIndex, cpus: &[CpuIndex]) -> L3Domain {
        L3Domain {
            system_numa_index: sys,
            cpus: set(cpus),
        }
    }

    #[test]
    fn l3_bundle_merges_within_budget() {
        let domains = vec![
            domain(0, &[0, 1]),
            domain(0, &[2, 3]),
            domain(1, &[4, 5]),
            domain(1, &[6, 7]),
        ];
        let cfg = from_l3_info(domains, 4);
        assert_eq!(cfg.num_numa_nodes(), 2);
        assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
        assert_eq!(cfg.nodes()[1], set(&[4, 5, 6, 7]));
    }

    #[test]
    fn l3_bundle_no_merge_below_boundary() {
        let domains = vec![
            domain(0, &[0, 1]),
            domain(0, &[2, 3]),
            domain(1, &[4, 5]),
            domain(1, &[6, 7]),
        ];
        let cfg = from_l3_info(domains, 3);
        assert_eq!(cfg.num_numa_nodes(), 4);
        assert_eq!(cfg.nodes()[0], set(&[0, 1]));
        assert_eq!(cfg.nodes()[1], set(&[2, 3]));
        assert_eq!(cfg.nodes()[2], set(&[4, 5]));
        assert_eq!(cfg.nodes()[3], set(&[6, 7]));
    }

    #[test]
    fn l3_bundle_size_zero_never_merges() {
        let domains = vec![domain(0, &[0, 1]), domain(0, &[2, 3])];
        let cfg = from_l3_info(domains, 0);
        assert_eq!(cfg.num_numa_nodes(), 2);
    }

    #[test]
    fn l3_bundle_boundary_exact_merges() {
        let domains = vec![domain(0, &[0, 1]), domain(0, &[2, 3])];
        let cfg = from_l3_info(domains, 4);
        assert_eq!(cfg.num_numa_nodes(), 1);
        assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    }

    #[test]
    fn l3_bundle_pass_semantics_leaves_odd_tail() {
        // The first pass merges (0,1)+(2,3) and advances past the result, so
        // (4,5) is untouched; the second pass finds 4+2 > 4 and stops.
        let domains = vec![domain(0, &[0, 1]), domain(0, &[2, 3]), domain(0, &[4, 5])];
        let cfg = from_l3_info(domains, 4);
        assert_eq!(cfg.num_numa_nodes(), 2);
        assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
        assert_eq!(cfg.nodes()[1], set(&[4, 5]));
    }

    #[test]
    fn default_config_single_node() {
        let cfg = NumaConfig::new();
        assert_eq!(cfg.num_numa_nodes(), 1);
        assert_eq!(cfg.num_cpus(), system_threads());
        assert!(!cfg.is_custom_affinity());
        assert!(!cfg.requires_memory_replication());
    }

    /// Build a non-custom config from explicit per-node CPU lists.
    /// `from_string` cannot be used: it forces `custom_affinity`.
    fn config_from_nodes(node_cpus: &[&[CpuIndex]]) -> NumaConfig {
        let mut cfg = NumaConfig::empty();
        for (n, cpus) in node_cpus.iter().enumerate() {
            for &c in *cpus {
                assert!(cfg.add_cpu_to_node(n, c));
            }
        }
        cfg
    }

    #[test]
    fn suggests_binding_custom_affinity_always_true() {
        // `custom_affinity` short-circuits before every other check, even for a
        // single thread.
        let cfg = NumaConfig::from_string("0-3:4-7").unwrap();
        assert!(cfg.is_custom_affinity());
        assert!(cfg.suggests_binding_threads(1));
        assert!(cfg.suggests_binding_threads(8));
    }

    #[test]
    fn suggests_binding_single_thread_or_single_node_false() {
        let two = config_from_nodes(&[&[0, 1, 2, 3], &[4, 5, 6, 7]]);
        assert!(!two.suggests_binding_threads(1));
        assert!(!two.suggests_binding_threads(0));
        let one = config_from_nodes(&[&[0, 1, 2, 3, 4, 5, 6, 7]]);
        assert!(!one.suggests_binding_threads(8));
    }

    #[test]
    fn suggests_binding_largest_over_two_branch() {
        // Two equal 4-CPU nodes: largest/2 = 2 and num_not_small = 2.
        let cfg = config_from_nodes(&[&[0, 1, 2, 3], &[4, 5, 6, 7]]);
        assert!(!cfg.suggests_binding_threads(2));
        // The `largest / 2` branch.
        assert!(cfg.suggests_binding_threads(3));
    }

    #[test]
    fn suggests_binding_four_times_not_small_branch() {
        // One big node of 20 plus a small node of 4, so num_not_small = 1.
        let big: Vec<CpuIndex> = (0..20).collect();
        let small: Vec<CpuIndex> = (20..24).collect();
        let cfg = config_from_nodes(&[&big, &small]);
        assert!(!cfg.suggests_binding_threads(3));
        // The `4 * num_not_small` branch, with `largest / 2` false here.
        assert!(cfg.suggests_binding_threads(4));
    }

    #[test]
    fn suggests_binding_small_node_threshold_is_inclusive_0_6() {
        // Four threads keeps the `largest / 2` branch false, so only the
        // small-node classification decides.
        let big: Vec<CpuIndex> = (0..20).collect();

        // 12/20 = 0.6 is at the threshold, so this node counts as small.
        let at_boundary: Vec<CpuIndex> = (20..32).collect();
        let small_cfg = config_from_nodes(&[&big, &at_boundary]);
        assert!(small_cfg.suggests_binding_threads(4));

        // 13/20 = 0.65 is above it, so this node is not small.
        let above_boundary: Vec<CpuIndex> = (20..33).collect();
        let big_cfg = config_from_nodes(&[&big, &above_boundary]);
        assert!(!big_cfg.suggests_binding_threads(4));
    }

    #[test]
    fn distribute_single_node_all_zero() {
        let cfg = config_from_nodes(&[&[0, 1, 2, 3]]);
        assert_eq!(cfg.distribute_threads_among_numa_nodes(3), vec![0, 0, 0]);
    }

    #[test]
    fn distribute_two_equal_nodes_alternates() {
        let cfg = config_from_nodes(&[&[0, 1], &[2, 3]]);
        assert_eq!(cfg.distribute_threads_among_numa_nodes(4), vec![0, 1, 0, 1]);
    }

    #[test]
    fn distribute_ties_go_to_lowest_index() {
        // The first pick is a tie, which must land on node 0.
        let cfg = config_from_nodes(&[&[0, 1], &[2, 3]]);
        assert_eq!(cfg.distribute_threads_among_numa_nodes(1), vec![0]);
        assert_eq!(cfg.distribute_threads_among_numa_nodes(2), vec![0, 1]);
    }

    #[cfg(target_os = "linux")]
    fn current_thread_affinity() -> BTreeSet<CpuIndex> {
        let mut cpus = BTreeSet::new();
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            let size = std::mem::size_of::<libc::cpu_set_t>();
            let status = libc::sched_getaffinity(0, size, &mut set as *mut libc::cpu_set_t);
            assert_eq!(status, 0, "sched_getaffinity failed in test");
            for c in 0..(size * 8) {
                if libc::CPU_ISSET(c, &set) {
                    cpus.insert(c);
                }
            }
        }
        cpus
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bind_sets_exactly_the_node_cpus() {
        // Spawned so the test runner's own affinity is never perturbed, and
        // built from the currently allowed CPUs so the bind targets a valid set.
        let handle = std::thread::spawn(|| {
            let allowed: Vec<CpuIndex> = current_thread_affinity().into_iter().collect();
            assert!(!allowed.is_empty(), "the test thread must have >= 1 CPU");
            let cfg = config_from_nodes(&[&allowed]);
            cfg.bind_current_thread_to_numa_node(0);
            let after = current_thread_affinity();
            let expected: BTreeSet<CpuIndex> = allowed.into_iter().collect();
            assert_eq!(after, expected);
        });
        handle.join().expect("bind test thread must not panic");
    }

    #[test]
    #[should_panic(expected = "out of range or empty")]
    fn bind_out_of_range_node_panics() {
        let cfg = config_from_nodes(&[&[0, 1]]);
        cfg.bind_current_thread_to_numa_node(5);
    }

    #[test]
    fn execute_on_numa_node_runs_closure_bound() {
        // Built from the currently allowed CPUs so the bind inside
        // `execute_on_numa_node` targets a valid set.
        let allowed: Vec<CpuIndex> = current_thread_affinity().into_iter().collect();
        assert!(!allowed.is_empty(), "the test thread must have >= 1 CPU");
        let cfg = config_from_nodes(&[&allowed]);
        let expected: BTreeSet<CpuIndex> = allowed.into_iter().collect();

        let mut ran = false;
        let mut observed: BTreeSet<CpuIndex> = BTreeSet::new();
        cfg.execute_on_numa_node(0, || {
            ran = true;
            observed = current_thread_affinity();
        });
        assert!(ran, "the closure must run to completion");
        assert_eq!(
            observed, expected,
            "the closure ran on a thread bound to node 0"
        );
    }
}
