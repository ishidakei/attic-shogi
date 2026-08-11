//! NUMA topology discovery — a Linux-only port of the reference engine's
//! `NumaConfig` machinery (`source/numa.h`, pinned submodule
//! @ `76d58ef`).
//!
//! This crate describes the machine's NUMA layout and, on Linux, can bind the
//! calling thread to a node; memory replication is not modelled. The protocol
//! layer consumes it for the `NumaPolicy` option and
//! thread-to-node binding; on a single-node machine no thread ever
//! binds (`auto` never suggests it), so parity fixtures and both bench forms
//! stay byte-identical there by construction.
//!
//! # What is ported
//!
//! * [`NumaConfig`] — an immutable description of NUMA nodes: for each logical
//!   node, the ordered set of CPUs it owns, plus a CPU → node reverse map, the
//!   highest CPU index seen, and a `custom_affinity` flag. Its invariants mirror
//!   the reference: the exposed nodes are never empty, and assigning a CPU that
//!   is already owned by some node is a fail-loud error.
//! * [`NumaConfig::from_string`] / [`Display`] — the reference's custom
//!   `':'`-separated / `','`-separated / `"a-b"`-range syntax, round-tripping
//!   through a canonical shortened form.
//! * [`NumaConfig::from_system`] — sysfs-driven autodetection under the three
//!   [`NumaAutoPolicy`] variants, including the L3-aware bundling that groups
//!   L3 cache domains within a system NUMA node up to a bundle size.
//! * [`NumaConfig::suggests_binding_threads`] /
//!   [`NumaConfig::distribute_threads_among_numa_nodes`] /
//!   [`NumaConfig::bind_current_thread_to_numa_node`] — the binding-decision,
//!   thread-distribution, and (Linux) `sched_setaffinity` pieces the engine uses
//!   to pin workers to nodes.
//! * [`startup_affinity`] — a *once-at-startup* snapshot of the process's CPU
//!   affinity (`sched_getaffinity`), matching the reference's use of startup
//!   affinities "so as not to modify its own behaviour in time".
//!
//! Linux-only by design (a deliberate decision for this port): the reference's
//! `_WIN64` paths are **not** ported. The pure parsing and topology code
//! compiles and runs everywhere; only the real-syscall pieces
//! ([`startup_affinity`] and the default `/sys` root of [`NumaConfig::from_system`])
//! are meaningful on Linux, and they degrade to a safe "all system threads"
//! fallback elsewhere so the crate stays buildable and testable on non-Linux
//! CI.
//!
//! # Testability
//!
//! All sysfs readers take an injectable root path via [`SysfsOptions`], so unit
//! and integration tests run against fixture directories rather than the live
//! `/sys` tree. [`NumaConfig::from_system`] is the thin production wrapper that
//! plugs in the real `/sys` root, the real [`startup_affinity`] snapshot, and
//! the real [`system_threads`] count.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A processor (CPU) index, as numbered by the operating system.
///
/// These always correspond to the actual numbering the system uses (mirroring
/// the reference `CpuIndex = size_t`).
pub type CpuIndex = usize;

/// A logical NUMA-node index within a [`NumaConfig`].
///
/// These do **not** necessarily correspond to the system's own NUMA-node
/// numbering: L3-aware subdivision, empty-node removal, and custom
/// configurations can all renumber nodes (mirroring the reference
/// `NumaIndex = size_t`).
pub type NumaIndex = usize;

/// Policy for how [`NumaConfig::from_system`] maps the machine to logical NUMA
/// nodes.
///
/// Mirrors the reference `NumaAutoPolicy` variant set (`numa.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaAutoPolicy {
    /// Use the system's own NUMA nodes verbatim.
    SystemNuma,
    /// Use system-reported L3 cache domains, one logical node per domain.
    L3Domains,
    /// Group system-reported L3 domains (within each system NUMA node) until
    /// each bundle reaches `bundle_size` CPUs.
    BundledL3 {
        /// Target maximum CPU count per bundled node.
        bundle_size: usize,
    },
}

/// The engine's default policy: bundle L3 domains up to 32 CPUs
/// (`source/engine.cpp`).
pub const DEFAULT_POLICY: NumaAutoPolicy = NumaAutoPolicy::BundledL3 { bundle_size: 32 };

/// Error type for the fail-loud paths that the reference resolves with
/// `std::exit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumaError {
    /// A CPU index was assigned to a node while already owned by some node.
    ///
    /// The reference `from_string` calls `std::exit(EXIT_FAILURE)` here; we
    /// surface it as a recoverable error instead.
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

/// Injectable inputs for the sysfs-driven detection path.
///
/// Production code fills this with the real `/sys` root, the real
/// [`startup_affinity`] snapshot, and the real [`system_threads`] count (see
/// [`NumaConfig::from_system`]); tests substitute a fixture directory and a
/// synthetic affinity set / thread count.
#[derive(Debug, Clone)]
pub struct SysfsOptions {
    /// Root under which the `devices/system/...` sysfs hierarchy lives
    /// (`/sys` in production).
    pub root: PathBuf,
    /// The set of CPUs the process is allowed to run on. Consulted only when a
    /// detection call passes `respect_affinity = true`.
    pub allowed_cpus: BTreeSet<CpuIndex>,
    /// The number of hardware threads to assume when a fallback to a single
    /// all-CPU node is required.
    pub system_threads: CpuIndex,
}

/// A system L3 cache domain: the CPUs sharing one L3, tagged with the *system*
/// NUMA node they belong to. Mirrors the reference `struct L3Domain`.
#[derive(Debug, Default, Clone)]
struct L3Domain {
    system_numa_index: NumaIndex,
    cpus: BTreeSet<CpuIndex>,
}

/// An immutable description of the machine's NUMA layout.
///
/// The CPU numbers always match the system's own numbering; the NUMA-node
/// numbers may not (see [`NumaIndex`]). Every node exposed by a `NumaConfig` is
/// guaranteed non-empty. Mirrors the reference `class NumaConfig`
/// (`numa.h`).
#[derive(Debug, Clone)]
pub struct NumaConfig {
    /// Per-node ordered CPU sets, indexed by [`NumaIndex`].
    nodes: Vec<BTreeSet<CpuIndex>>,
    /// Reverse map: CPU → owning node.
    node_by_cpu: BTreeMap<CpuIndex, NumaIndex>,
    /// The largest CPU index ever assigned.
    highest_cpu_index: CpuIndex,
    /// Set when the configuration was produced in a way that may not match the
    /// current process affinity (custom string, or `respect_affinity = false`).
    custom_affinity: bool,
}

impl Default for NumaConfig {
    /// A single node containing CPUs `0..system_threads()`
    /// (`numa.h`).
    fn default() -> Self {
        Self::new()
    }
}

impl NumaConfig {
    /// The default configuration: one node holding every hardware thread
    /// (`numa.h`).
    pub fn new() -> Self {
        let mut cfg = Self::empty();
        let num_cpus = system_threads();
        // `system_threads()` is always >= 1, so `num_cpus - 1` cannot underflow.
        cfg.add_cpu_range_to_node(0, 0, num_cpus - 1);
        cfg
    }

    /// An empty configuration with no nodes (the reference's private
    /// `empty()`).
    fn empty() -> Self {
        NumaConfig {
            nodes: Vec::new(),
            node_by_cpu: BTreeMap::new(),
            highest_cpu_index: 0,
            custom_affinity: false,
        }
    }

    /// Parses the reference's custom node syntax (`numa.h`).
    ///
    /// `':'` separates nodes, `','` separates entries within a node, and
    /// `"a-b"` denotes an inclusive CPU range. Empty node groups are skipped.
    /// A CPU that appears more than once is a fail-loud
    /// [`NumaError::DuplicateCpu`]. The result has `custom_affinity` set.
    ///
    /// Example: `"0-15,32-47:16-31,48-63"`.
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

    /// Autodetects the NUMA layout from the live `/sys` tree, the real startup
    /// affinity snapshot, and the real hardware-thread count.
    ///
    /// This is the thin production wrapper over [`NumaConfig::from_sysfs`]. On
    /// non-Linux targets the affinity snapshot degrades to "all system
    /// threads" (see [`startup_affinity`]).
    pub fn from_system(policy: &NumaAutoPolicy, respect_affinity: bool) -> Self {
        let opts = SysfsOptions {
            root: PathBuf::from("/sys"),
            allowed_cpus: startup_affinity().clone(),
            system_threads: system_threads(),
        };
        Self::from_sysfs(policy, respect_affinity, &opts)
    }

    /// Autodetects the NUMA layout from an injectable sysfs root.
    ///
    /// Mirrors the reference `from_system` Linux branch (`numa.h`):
    /// unless the policy is [`NumaAutoPolicy::SystemNuma`], first try the
    /// L3-aware config; fall back to the system-NUMA sysfs config otherwise.
    /// Empty nodes are removed at the end, and `respect_affinity = false`
    /// marks the result custom.
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
    /// Panics if `n` is out of range, mirroring the reference `assert`.
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

    /// Read-only access to the per-node CPU sets, in node order.
    pub fn nodes(&self) -> &[BTreeSet<CpuIndex>] {
        &self.nodes
    }

    /// The node owning CPU `c`, if any.
    pub fn node_of_cpu(&self, c: CpuIndex) -> Option<NumaIndex> {
        self.node_by_cpu.get(&c).copied()
    }

    /// The *system* NUMA node a logical node belongs to (`get_discriminator`,
    /// `numa.h`).
    ///
    /// The reference uses this to decide the replication granularity of
    /// `LazyNumaReplicatedSystemWide`: the copy granularity is the hardware /
    /// system NUMA domain, not the (possibly L3-bundled) logical node. It takes
    /// the logical node's first CPU, resolves it against a *system* config
    /// (`NumaConfig::from_sysfs(SystemNuma, respect_affinity = false, opts)`), and
    /// falls back to system node 0 for an unassigned CPU. Two logical nodes that
    /// share one system node therefore return the same value — the signal the
    /// port uses to share a single network copy between them.
    ///
    /// The discriminator's textual system-topology prefix (`cfg_sys.to_string() +
    /// "$" + sys_idx`) keys the reference's shared-memory segment; this port has
    /// no shared-memory layer (a declared scope reduction) and lives in one
    /// process with one topology, so the system-node index alone is the
    /// discriminator.
    ///
    /// # Panics
    /// Panics if `idx` is out of range (mirroring the reference's `nodes[idx]`
    /// dereference). Every exposed node is non-empty, so the first-CPU lookup
    /// always succeeds.
    pub fn system_node_of_logical(&self, idx: NumaIndex, opts: &SysfsOptions) -> NumaIndex {
        let cfg_sys = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, false, opts);
        self.system_node_of_logical_in(idx, &cfg_sys)
    }

    /// The system node of every worker's logical node, in `bound` order.
    ///
    /// A batch [`Self::system_node_of_logical`] that builds the system config once
    /// (it reads sysfs), for resolving a whole binding assignment at pool-rebuild
    /// time. Entry `i` is the system node the reference would replicate worker
    /// `i`'s network onto.
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

    /// The shared body of the two mappers above, given a prebuilt system config.
    fn system_node_of_logical_in(&self, idx: NumaIndex, cfg_sys: &NumaConfig) -> NumaIndex {
        let cpu = *self.nodes[idx]
            .iter()
            .next()
            .expect("every exposed NUMA node is non-empty");
        cfg_sys.node_of_cpu(cpu).unwrap_or(0)
    }

    /// Whether the configuration is flagged custom (custom string, or built
    /// without respecting process affinity).
    pub fn is_custom_affinity(&self) -> bool {
        self.custom_affinity
    }

    /// Whether the engine should distribute and bind its worker threads across
    /// NUMA nodes for the requested thread count (`numa.h`).
    ///
    /// A custom affinity always suggests binding (the OS affinity may not match
    /// what the user asked for). A single thread never binds. Otherwise: let
    /// `largest` be the biggest node's CPU count; a node is "small" when its
    /// size is at most `SmallNodeThreshold = 0.6` of `largest`; let
    /// `num_not_small` be the count of non-small nodes. Binding is suggested
    /// when `num_threads` cannot reasonably be contained by the first node
    /// (`num_threads > largest / 2`) or there are enough threads to spread
    /// across the non-small nodes with minimal disparity
    /// (`num_threads >= num_not_small * 4`) — and there is more than one node.
    pub fn suggests_binding_threads(&self, num_threads: CpuIndex) -> bool {
        if self.custom_affinity {
            return true;
        }

        // A single thread cannot be distributed, so it is never bound.
        if num_threads <= 1 {
            return false;
        }

        let largest_node_size = self.nodes.iter().map(|cpus| cpus.len()).max().unwrap_or(0);

        // `SmallNodeThreshold = 0.6` — a node is small (in particular, an empty
        // node) when its share of the largest node is at or below this.
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
    /// (`numa.h`).
    ///
    /// A single-node config puts every thread on node 0. Otherwise the
    /// assignment greedily fills the node that minimises
    /// `(occupation + 1) / node_size` (strict `<`, so ties go to the lowest node
    /// index), incrementing that node's occupation after each pick. No node is
    /// favoured, so multiple engine instances do not all crowd node 0.
    pub fn distribute_threads_among_numa_nodes(&self, num_threads: CpuIndex) -> Vec<NumaIndex> {
        let mut ns: Vec<NumaIndex> = Vec::new();

        if self.nodes.len() == 1 {
            // Special case for when there are no real NUMA nodes: keep the
            // default path simple by putting everything on node 0.
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
    /// to that node's CPUs (`numa.h`, Linux branch).
    ///
    /// # Panics
    /// Fail-loud, mirroring the reference's `std::exit(EXIT_FAILURE)` (the
    /// accepted port form for these paths — see the crate PR):
    /// * if `n` is out of range or the node is empty;
    /// * (Linux) if `highest_cpu_index >= 1024` — this port uses a fixed
    ///   1024-CPU `cpu_set_t` rather than the reference's dynamic
    ///   `CPU_ALLOC(highestCpuIndex + 1)`, so a CPU index that would not fit the
    ///   fixed mask is rejected rather than silently truncated;
    /// * (Linux) if `sched_setaffinity` fails.
    ///
    /// On non-Linux targets this is a no-op (the reference's real binding is
    /// Linux/Win64-only and Win64 is out of scope for this port).
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
    /// (`numa.h`).
    ///
    /// The reference uses this so an on-node allocation's pages are first-touched
    /// on that node: the thread binds, the closure allocates (and, in this port's
    /// case, fills) the region, and the kernel's first-touch policy places the
    /// pages on `n`. The closure runs to completion before this returns; `f`'s
    /// captures may be borrowed for the duration (a scoped thread), so it can
    /// write its result back into a caller-owned slot.
    ///
    /// On non-Linux targets the bind is a no-op (see
    /// [`Self::bind_current_thread_to_numa_node`]); the closure still runs on the
    /// temporary thread, so the control flow is identical across platforms.
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

    /// Drops any empty nodes, preserving the order of the rest
    /// (`numa.h`).
    fn remove_empty_numa_nodes(&mut self) {
        self.nodes.retain(|cpus| !cpus.is_empty());
        // `node_by_cpu` is untouched: it maps CPUs to *pre-removal* node
        // indices. Matching the reference, callers rebuild configs rather than
        // mutate them, so no code observes stale reverse-map indices after a
        // removal. To keep this port's reverse map internally consistent we
        // rebuild it from the surviving nodes.
        self.node_by_cpu.clear();
        for (n, cpus) in self.nodes.iter().enumerate() {
            for &c in cpus {
                self.node_by_cpu.insert(c, n);
            }
        }
    }

    /// Assigns CPU `c` to node `n`.
    ///
    /// Returns `false` (leaving the structure unmodified) if `c` is already
    /// assigned; `true` on success (`numa.h`).
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

    /// Assigns the inclusive CPU range `cfirst..=clast` to node `n`.
    ///
    /// All-or-nothing: returns `false` (unmodified) if any CPU in the range is
    /// already assigned (`numa.h`).
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
    /// Emits the canonical shortened form, re-compressing consecutive CPUs into
    /// `"a-b"` ranges (`numa.h`). The round-trip
    /// `from_string(x.to_string())` reproduces `x`'s node structure.
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

/// Reads a sysfs file under `root`, returning its contents.
///
/// Returns `None` if the file cannot be read (does not exist); `Some("")` for
/// an empty file. Mirrors the reference `read_file_to_string`
/// (`misc.cpp`), which returns `nullopt` only when the file cannot be
/// opened.
fn read_sysfs(root: &Path, rel: &str) -> Option<String> {
    std::fs::read_to_string(root.join(rel)).ok()
}

/// Removes all ASCII whitespace from `s` (`misc.cpp`).
fn remove_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_ascii_whitespace()).collect()
}

/// Parses a single decimal index, tolerating surrounding whitespace and
/// trailing non-digits like the reference `str_to_size_t`'s `stoull`
/// (`misc.cpp`). Returns `None` when no leading digits are present.
fn parse_size_t(s: &str) -> Option<CpuIndex> {
    let t = s.trim_start();
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<CpuIndex>().ok()
}

/// Expands the reference's "shortened index-list" syntax into a flat list of
/// indices (`numa.h`).
///
/// `','` separates entries; each entry is either a single index or an inclusive
/// `"a-b"` range. Empty entries are skipped, and an empty input yields no
/// indices.
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
            // Entries with 0 or 3+ dash-separated parts are ignored, matching
            // the reference (which only handles the 1- and 2-part cases).
            _ => {}
        }
    }

    indices
}

/// Whether CPU `c` is allowed given a detection call's affinity setting.
fn is_cpu_allowed(opts: &SysfsOptions, respect_affinity: bool, c: CpuIndex) -> bool {
    !respect_affinity || opts.allowed_cpus.contains(&c)
}

/// The system-NUMA sysfs config path (`numa.h`).
///
/// Reads `devices/system/node/online`, then each node's `cpulist`. A missing
/// `online` file (or a missing per-node `cpulist`) falls back to a single node
/// containing all allowed CPUs `0..system_threads`.
fn from_system_numa(opts: &SysfsOptions, respect_affinity: bool) -> NumaConfig {
    let mut cfg = NumaConfig::empty();
    let mut use_fallback = false;

    match read_sysfs(&opts.root, "devices/system/node/online") {
        Some(node_ids) if !node_ids.is_empty() => {
            let node_ids = remove_whitespace(&node_ids);
            for n in indices_from_shortened_string(&node_ids) {
                let path = format!("devices/system/node/node{n}/cpulist");
                match read_sysfs(&opts.root, &path) {
                    // Only bail if the file does not exist. An empty node still
                    // has a (whitespace-only) file, and empty nodes are fine.
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
        // Discard any partial config, exactly as the reference's `fallback()`
        // resets `cfg` to empty.
        cfg = NumaConfig::empty();
        for c in 0..opts.system_threads {
            if is_cpu_allowed(opts, respect_affinity, c) {
                cfg.add_cpu_to_node(0, c);
            }
        }
    }

    cfg
}

/// Attempts the L3-aware config path (`numa.h`).
///
/// Walks CPUs via "next unseen CPU", reading each one's
/// `cache/index3/shared_cpu_list`; each L3 domain keeps its owning *system*
/// NUMA node (looked up from a system-NUMA config). Stops at the first
/// missing/empty file. Returns `None` if no L3 domains were found.
fn try_get_l3_aware_config(
    opts: &SysfsOptions,
    respect_affinity: bool,
    bundle_size: usize,
) -> Option<NumaConfig> {
    // Get the normal system config so we know which NUMA node each L3 domain
    // belongs to.
    let system_config = NumaConfig::from_sysfs(&NumaAutoPolicy::SystemNuma, respect_affinity, opts);

    let mut l3_domains: Vec<L3Domain> = Vec::new();
    let mut seen: BTreeSet<CpuIndex> = BTreeSet::new();

    // Defensive upper bound on the "next unseen CPU" scan. The loop really
    // terminates on the first missing/empty sysfs file; this only guards
    // against a malformed fixture that never grows `seen`.
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
            // Missing or empty file: we have read all available CPUs.
            _ => break,
        };

        let mut domain = L3Domain::default();
        for c in indices_from_shortened_string(&siblings) {
            if is_cpu_allowed(opts, respect_affinity, c) {
                // `.at(c)` in the reference — a fail-loud lookup. On a
                // consistent system every allowed CPU is present in the
                // system-NUMA config.
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

/// Bundles L3 domains into logical NUMA nodes (`numa.h`).
///
/// Domains are grouped by their system NUMA node; within each group, adjacent
/// pairs are repeatedly merged while `|a| + |b| <= bundle_size`; the surviving
/// domains are numbered sequentially.
fn from_l3_info(domains: Vec<L3Domain>, bundle_size: usize) -> NumaConfig {
    debug_assert!(!domains.is_empty());

    // Group by system NUMA index. A `BTreeMap` iterates keys in ascending
    // order, matching the reference `std::map`.
    let mut list: BTreeMap<NumaIndex, Vec<L3Domain>> = BTreeMap::new();
    for d in domains {
        list.entry(d.system_numa_index).or_default().push(d);
    }

    let mut cfg = NumaConfig::empty();
    let mut n: NumaIndex = 0;
    for (_, mut ds) in list {
        // Scan through pairs and merge them. With roughly equal L3 sizes this
        // gives a decent distribution.
        loop {
            let mut changed = false;
            let mut j = 0;
            while j + 1 < ds.len() {
                if ds[j].cpus.len() + ds[j + 1].cpus.len() <= bundle_size {
                    changed = true;
                    let mut next = ds.remove(j + 1);
                    ds[j].cpus.append(&mut next.cpus);
                }
                // `j` advances every iteration, exactly as the reference
                // for-loop does: a just-merged node is not re-checked against
                // its new neighbour within the same pass.
                j += 1;
            }
            // `ds.len()` strictly decreases whenever `changed`, so this
            // terminates.
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

/// The number of usable hardware threads, at least 1
/// (`numa.h` / `SYSTEM_THREADS_NB`).
pub fn system_threads() -> CpuIndex {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// The set of CPUs the process was allowed to run on **at startup**, captured
/// once.
///
/// Mirrors the reference `STARTUP_PROCESSOR_AFFINITY` (`numa.h`): a
/// deliberate startup snapshot so detection does not change behaviour as the
/// live affinity changes over time. On Linux this is the result of
/// `sched_getaffinity`; on other targets it degrades to all system threads.
pub fn startup_affinity() -> &'static BTreeSet<CpuIndex> {
    static STARTUP: OnceLock<BTreeSet<CpuIndex>> = OnceLock::new();
    STARTUP.get_or_init(capture_process_affinity)
}

#[cfg(target_os = "linux")]
fn capture_process_affinity() -> BTreeSet<CpuIndex> {
    // This port deliberately uses a fixed 1024-CPU `cpu_set_t` here. This is a
    // narrower cap than the reference's startup snapshot, which allocates
    // `CPU_ALLOC(1024 * 64)` (`numa.h`) precisely because the platform
    // default of 1024 may be too small on very large machines. The fixed cap is
    // acceptable for this project's target hardware (well under 1024 logical
    // CPUs); a machine that exceeds it would fail loud at bind time via
    // [`NumaConfig::bind_current_thread_to_numa_node`] rather than silently
    // mis-binding.
    let mut cpus = BTreeSet::new();
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let size = std::mem::size_of::<libc::cpu_set_t>();
        let status = libc::sched_getaffinity(0, size, &mut set as *mut libc::cpu_set_t);
        if status != 0 {
            // Soft error: assume all system threads are available rather than
            // aborting the process from within a library.
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

/// The Linux affinity-setting core of [`NumaConfig::bind_current_thread_to_numa_node`].
///
/// Builds a CPU mask of `cpus`, applies it to the current thread via
/// `sched_setaffinity(0, ...)`, then `sched_yield`s (the reference's defensive
/// re-schedule so the thread lands on the newly-allowed CPUs promptly). Fail-loud
/// on every error path, mirroring the reference `std::exit(EXIT_FAILURE)`.
#[cfg(target_os = "linux")]
fn bind_current_thread_to_cpus(highest_cpu_index: CpuIndex, cpus: &BTreeSet<CpuIndex>) {
    // This port sizes the mask with a fixed `cpu_set_t` (1024-CPU capacity)
    // instead of the reference's dynamic `CPU_ALLOC(highestCpuIndex + 1)`. A CPU
    // index that would not fit the fixed mask is a fail-loud error rather than a
    // silent out-of-bounds `CPU_SET`.
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
        // Defensive re-schedule; allowed because this is not performance critical.
        libc::sched_yield();
    }
}

/// Non-Linux no-op counterpart of [`bind_current_thread_to_cpus`]: this port does
/// not bind threads off Linux (the reference's other real path is Win64, out of
/// scope).
#[cfg(not(target_os = "linux"))]
fn bind_current_thread_to_cpus(_highest_cpu_index: CpuIndex, _cpus: &BTreeSet<CpuIndex>) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(cpus: &[CpuIndex]) -> BTreeSet<CpuIndex> {
        cpus.iter().copied().collect()
    }

    // -- shortened-list parsing -------------------------------------------

    #[test]
    fn parse_simple_list_and_range() {
        assert_eq!(indices_from_shortened_string("0-3,8"), vec![0, 1, 2, 3, 8]);
        assert_eq!(indices_from_shortened_string("5"), vec![5]);
        assert_eq!(indices_from_shortened_string("2-2"), vec![2]);
    }

    #[test]
    fn parse_empty_and_empty_entries() {
        assert_eq!(indices_from_shortened_string(""), Vec::<CpuIndex>::new());
        // Empty entries between commas are skipped.
        assert_eq!(indices_from_shortened_string("0,,3"), vec![0, 3]);
    }

    #[test]
    fn parse_tolerates_whitespace() {
        // sysfs content is passed through `remove_whitespace` first.
        assert_eq!(
            indices_from_shortened_string(&remove_whitespace(" 0-3 , 8 \n")),
            vec![0, 1, 2, 3, 8]
        );
        // A trailing newline within a token is tolerated directly (as the
        // reference relies on `stoull` doing).
        assert_eq!(indices_from_shortened_string("0-3\n"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn descending_range_is_empty() {
        assert_eq!(indices_from_shortened_string("5-3"), Vec::<CpuIndex>::new());
    }

    // -- from_string ------------------------------------------------------

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
        // custom_affinity alone forces replication.
        assert!(cfg.requires_memory_replication());
    }

    // -- to_string round-trip ---------------------------------------------

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

    // -- L3 pair-merge bundling on synthetic domains ----------------------

    fn domain(sys: NumaIndex, cpus: &[CpuIndex]) -> L3Domain {
        L3Domain {
            system_numa_index: sys,
            cpus: set(cpus),
        }
    }

    #[test]
    fn l3_bundle_merges_within_budget() {
        // Two system nodes, two L3 domains of 2 CPUs each.
        let domains = vec![
            domain(0, &[0, 1]),
            domain(0, &[2, 3]),
            domain(1, &[4, 5]),
            domain(1, &[6, 7]),
        ];
        // bundle_size = 4: each pair (2+2 <= 4) merges into one node per system
        // node.
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
        // bundle_size = 3: 2+2 = 4 > 3, so nothing merges; four nodes.
        let cfg = from_l3_info(domains, 3);
        assert_eq!(cfg.num_numa_nodes(), 4);
        assert_eq!(cfg.nodes()[0], set(&[0, 1]));
        assert_eq!(cfg.nodes()[1], set(&[2, 3]));
        assert_eq!(cfg.nodes()[2], set(&[4, 5]));
        assert_eq!(cfg.nodes()[3], set(&[6, 7]));
    }

    #[test]
    fn l3_bundle_size_zero_never_merges() {
        // L3DomainsPolicy is modelled as bundle_size = 0.
        let domains = vec![domain(0, &[0, 1]), domain(0, &[2, 3])];
        let cfg = from_l3_info(domains, 0);
        assert_eq!(cfg.num_numa_nodes(), 2);
    }

    #[test]
    fn l3_bundle_boundary_exact_merges() {
        // |a| + |b| == bundle_size must merge (`<=`).
        let domains = vec![domain(0, &[0, 1]), domain(0, &[2, 3])];
        let cfg = from_l3_info(domains, 4);
        assert_eq!(cfg.num_numa_nodes(), 1);
        assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
    }

    #[test]
    fn l3_bundle_pass_semantics_leaves_odd_tail() {
        // Three domains of 2 within one system node, bundle_size = 4.
        // Pass 1: merge (0,1)+(2,3) -> {0,1,2,3}; j advances past the merged
        // node, so (4,5) is not merged in this pass. After the pass `changed`
        // is true, so a second pass runs: now [{0,1,2,3}, {4,5}], 4+2 = 6 > 4,
        // no merge. Result: two nodes.
        let domains = vec![domain(0, &[0, 1]), domain(0, &[2, 3]), domain(0, &[4, 5])];
        let cfg = from_l3_info(domains, 4);
        assert_eq!(cfg.num_numa_nodes(), 2);
        assert_eq!(cfg.nodes()[0], set(&[0, 1, 2, 3]));
        assert_eq!(cfg.nodes()[1], set(&[4, 5]));
    }

    // -- default construction ---------------------------------------------

    #[test]
    fn default_config_single_node() {
        let cfg = NumaConfig::new();
        assert_eq!(cfg.num_numa_nodes(), 1);
        assert_eq!(cfg.num_cpus(), system_threads());
        assert!(!cfg.is_custom_affinity());
        // A single node without custom affinity needs no replication.
        assert!(!cfg.requires_memory_replication());
    }

    // -- suggests_binding_threads -----------------------------------------

    /// Build a NON-custom config from explicit per-node CPU lists (the tests
    /// module can reach the private `empty` / `add_cpu_to_node`). `from_string`
    /// cannot be used because it forces `custom_affinity`.
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
        // `custom_affinity` short-circuits to true before every other check —
        // even for a single thread.
        let cfg = NumaConfig::from_string("0-3:4-7").unwrap();
        assert!(cfg.is_custom_affinity());
        assert!(cfg.suggests_binding_threads(1));
        assert!(cfg.suggests_binding_threads(8));
    }

    #[test]
    fn suggests_binding_single_thread_or_single_node_false() {
        // A single thread never binds (non-custom).
        let two = config_from_nodes(&[&[0, 1, 2, 3], &[4, 5, 6, 7]]);
        assert!(!two.suggests_binding_threads(1));
        assert!(!two.suggests_binding_threads(0));
        // A single node never binds regardless of thread count.
        let one = config_from_nodes(&[&[0, 1, 2, 3, 4, 5, 6, 7]]);
        assert!(!one.suggests_binding_threads(8));
    }

    #[test]
    fn suggests_binding_largest_over_two_branch() {
        // Two equal 4-CPU nodes: largest = 4, largest/2 = 2, num_not_small = 2.
        // (num_threads > 2 || num_threads >= 8) && nodes > 1.
        let cfg = config_from_nodes(&[&[0, 1, 2, 3], &[4, 5, 6, 7]]);
        // 2 > 2 is false and 2 >= 8 is false → no binding.
        assert!(!cfg.suggests_binding_threads(2));
        // 3 > 2 is true → binding (the `largest / 2` branch).
        assert!(cfg.suggests_binding_threads(3));
    }

    #[test]
    fn suggests_binding_four_times_not_small_branch() {
        // One big node (20) plus a small node (4, ratio 0.2 ≤ 0.6). largest = 20,
        // largest/2 = 10, num_not_small = 1.
        let big: Vec<CpuIndex> = (0..20).collect();
        let small: Vec<CpuIndex> = (20..24).collect();
        let cfg = config_from_nodes(&[&big, &small]);
        // 3 > 10 false, 3 >= 4 false → no binding.
        assert!(!cfg.suggests_binding_threads(3));
        // 4 > 10 false, but 4 >= 4*1 true → binding (the `4 * num_not_small`
        // branch, distinct from the `largest / 2` branch which is false here).
        assert!(cfg.suggests_binding_threads(4));
    }

    #[test]
    fn suggests_binding_small_node_threshold_is_inclusive_0_6() {
        // largest = 20, largest/2 = 10. Pick num_threads = 4 so the `largest/2`
        // branch (4 > 10) is false; only `num_threads >= 4 * num_not_small`
        // decides, isolating the small-node classification.
        let big: Vec<CpuIndex> = (0..20).collect();

        // Second node of 12 CPUs: 12/20 = 0.6, which is `<= 0.6` → SMALL, so
        // num_not_small = 1 and 4 >= 4*1 → binding.
        let at_boundary: Vec<CpuIndex> = (20..32).collect();
        let small_cfg = config_from_nodes(&[&big, &at_boundary]);
        assert!(small_cfg.suggests_binding_threads(4));

        // Second node of 13 CPUs: 13/20 = 0.65 > 0.6 → NOT small, so
        // num_not_small = 2 and 4 >= 4*2 is false → no binding.
        let above_boundary: Vec<CpuIndex> = (20..33).collect();
        let big_cfg = config_from_nodes(&[&big, &above_boundary]);
        assert!(!big_cfg.suggests_binding_threads(4));
    }

    // -- distribute_threads_among_numa_nodes ------------------------------

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
        // The first pick is a tie (both fills 1/2); it must land on node 0.
        let cfg = config_from_nodes(&[&[0, 1], &[2, 3]]);
        assert_eq!(cfg.distribute_threads_among_numa_nodes(1), vec![0]);
        assert_eq!(cfg.distribute_threads_among_numa_nodes(2), vec![0, 1]);
    }

    // -- bind_current_thread_to_numa_node ---------------------------------

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
        // Run in a spawned thread so we never perturb the test runner's own
        // affinity. Build a single-node config from the CPUs currently allowed,
        // bind to it, then confirm `sched_getaffinity` reports exactly that set.
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
        // Node 5 does not exist — fail-loud.
        cfg.bind_current_thread_to_numa_node(5);
    }

    #[test]
    fn execute_on_numa_node_runs_closure_bound() {
        // Build a single-node config from the CPUs currently allowed, so the bind
        // inside `execute_on_numa_node` targets a valid set (mirrors
        // `bind_sets_exactly_the_node_cpus`). The closure must run to completion
        // and observe the bound affinity.
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
