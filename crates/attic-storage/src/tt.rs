//! Transposition table — a faithful port of the YaneuraOu reference
//! (`source/tt.h` / `tt.cpp` at the pinned submodule
//! commit), specialised to the engine's default build configuration.
//!
//! # Configuration
//!
//! The reference is heavily `#define`-parameterised; we mirror the **default**
//! build (`source/config.h`):
//!
//! - `HASH_KEY_BITS == 64` → the position key is a plain 64-bit `Key64`, and
//!   the in-cluster key fragment stored per entry is the key's low 16 bits.
//! - `TT_CLUSTER_SIZE == 3` → each cluster holds three [`TTEntry`]s, so the
//!   per-entry key fragment type (`TTE_KEY_TYPE`) is `uint16_t`, an entry is
//!   10 bytes, and a cluster is exactly 32 bytes (`3 × 10 + 2` padding).
//!
//! These choices are load-bearing for search-node parity with the reference:
//! the cluster size and the `clusterCount = mb·2²⁰ / sizeof(Cluster)` sizing
//! arithmetic together decide which positions collide, and the in-cluster
//! replacement policy decides which entry survives a collision — both feed
//! directly into how many nodes qsearch visits.
//!
//! # Layering
//!
//! The Storage layer must not depend on the State
//! layer, so this module speaks in primitives: the position key is a `u64`,
//! the side-to-move is a `u8` (0 = Black, 1 = White, matching the reference's
//! `Color` being folded into cluster-index bit 0), and the best move is stored
//! as its low-16-bit fragment (`u16`, the reference's `Move16`). The Search
//! layer — which owns both this table and `attic-state` — is responsible for
//! widening the 16-bit move back to a full move and for validating it against
//! the actual position; that reconstruction is out of scope here.
//!
//! # Threading
//!
//! The table is shared across search threads as `Arc<TranspositionTable>`: the
//! read/write path ([`TranspositionTable::probe`] / [`TranspositionTable::locate`]
//! / [`TranspositionTable::write_at`]) takes `&self`, so several workers can hit
//! one instance concurrently. Each entry field is an atomic accessed with
//! [`Ordering::Relaxed`] loads/stores — the Rust-sound equivalent of the
//! reference's racy, lock-free `TTEntry` (`source/tt.h`),
//! which reads and writes plain members across threads with no synchronisation.
//! On x86-64 a relaxed atomic load/store lowers to a plain `MOV`, so the
//! single-thread codegen is identical to the previous plain-field layout and the
//! data layout is unchanged (a flat array of fixed-size 32-byte clusters).
//!
//! ## Torn-entry tolerance
//!
//! Relaxed atomics make each *field* access indivisible, but an entry is six
//! fields with no cross-field atomicity: a concurrent write on another thread
//! can land between this thread's field reads, so [`TTEntry::read`] may return a
//! payload that mixes an old key fragment with a new value/move (a *torn*
//! entry). The reference tolerates exactly this. Correctness does not depend on
//! entries being coherent: a stale or mismatched key fragment simply reads back
//! as a miss or a wrong-position hit, and the Search layer validates every TT
//! move against the actual position before trusting it (see the module docs
//! above) — the shared-atomic layout here makes no coherence promise beyond
//! per-field atomicity.
//!
//! ## Single-thread decision-identity
//!
//! With one searcher the atomics are never contended, so every field read
//! observes the value the same thread last wrote: `probe` / `locate` /
//! `write_at` make the exact same cluster-geometry, entry-selection,
//! replacement-scoring and generation decisions as the previous plain-field
//! implementation, and return the same data. The search-node parity gates are
//! the proof.
//!
//! ## Lifecycle exclusivity
//!
//! [`TranspositionTable::resize`] and [`TranspositionTable::clear`] take
//! `&mut self`: they are driver-side lifecycle operations run only while no
//! search holds the table. With the table behind an `Arc`, the driver reaches
//! them via [`Arc::get_mut`], which returns `Some` only when the refcount is 1
//! — i.e. after every search worker has been joined and dropped its clone. The
//! type system therefore enforces that a resize/clear never races a probe.

use std::alloc::Layout;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicI16, AtomicU8, AtomicU16, Ordering};

use crate::large_page;

/// The single memory ordering used for every entry field access, matching the
/// reference's unsynchronised (racy) reads and writes. On x86-64 this lowers to
/// a plain `MOV`.
const REL: Ordering = Ordering::Relaxed;

/// Search value. The reference's `Value` is a plain `int`; we mirror that with
/// `i32`. Stored in an entry truncated to `i16` (all real values fit).
pub type Value = i32;

/// Search depth. The reference's `Depth` is a plain `int`; we mirror that with
/// `i32`. Stored in an entry offset by [`DEPTH_NONE`] and truncated to `u8`.
pub type Depth = i32;

/// `DEPTH_NONE` (`source/types.h`). Entries are stored with
/// `depth8 = depth − DEPTH_NONE`, so an all-zero entry reads back as
/// `DEPTH_NONE` and is treated as unoccupied.
pub const DEPTH_NONE: Depth = -3;

/// `VALUE_NONE` (`source/types.h`). The sentinel returned
/// for a miss.
pub const VALUE_NONE: Value = 32002;

/// The reference's default `USI_Hash` in MiB (`yaneuraou-search.cpp`), matched
/// so that gate conditions equal the fixture-capture conditions. This is only
/// the default the driver passes to `resize`: a fresh table is empty
/// (`clusterCount == 0`) exactly like the reference's default constructor.
pub const DEFAULT_HASH_MB: usize = 1024;

// genBound8 bit layout (`source/tt.cpp`):
// `generation (5) | bound (2) << 5 | pv (1) << 7`.
const GENERATION_BITS: u8 = 5;
const GENERATION_MASK: u8 = (1 << GENERATION_BITS) - 1;
const BOUND_SHIFT: u8 = GENERATION_BITS;
const BOUND_MASK: u8 = 0b11 << BOUND_SHIFT;
const PV_SHIFT: u8 = BOUND_SHIFT + 2;
const PV_MASK: u8 = 1 << PV_SHIFT;

/// Number of entries per cluster (`TT_CLUSTER_SIZE == 3`).
const CLUSTER_SIZE: usize = 3;

/// Bound type of a stored value (`source/types.h`). The
/// discriminants matter: `Exact == Upper | Lower`, and the value is packed
/// into `genBound8`.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    /// No bound — used when only a move / static eval is stored.
    None = 0,
    /// Upper bound: the true value is `≤` the stored value (fail-low).
    Upper = 1,
    /// Lower bound: the true value is `≥` the stored value (fail-high).
    Lower = 2,
    /// Exact value (PV node, non-mate).
    Exact = 3,
}

impl Bound {
    /// Recover a `Bound` from its 2-bit encoding. Every value `0..=3` is valid.
    #[inline]
    fn from_u8(v: u8) -> Bound {
        match v & 0b11 {
            0 => Bound::None,
            1 => Bound::Upper,
            2 => Bound::Lower,
            _ => Bound::Exact,
        }
    }
}

/// A decoded copy of an entry's payload — the reference's `TTData`. Reads are
/// by value; nothing here borrows the table.
///
/// `move16` is the low-16-bit move fragment as stored; widening and validating
/// it against a position is the Search layer's job (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TTData {
    /// Best move for this position, as a 16-bit fragment (`0` = none).
    pub move16: u16,
    /// Search value returned at this node.
    pub value: Value,
    /// Static / qsearch evaluation at this node.
    pub eval: Value,
    /// Depth the value was searched to.
    pub depth: Depth,
    /// Bound type of `value`.
    pub bound: Bound,
    /// Whether this was a PV node.
    pub is_pv: bool,
}

impl TTData {
    /// The miss sentinel returned by a failed [`TranspositionTable::probe`],
    /// mirroring the reference's
    /// `TTData{Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_NONE, BOUND_NONE, false}`.
    #[inline]
    fn none() -> TTData {
        TTData {
            move16: 0,
            value: VALUE_NONE,
            eval: VALUE_NONE,
            depth: DEPTH_NONE,
            bound: Bound::None,
            is_pv: false,
        }
    }
}

/// This entry's age relative to `curr_generation` (the reference's
/// `TTEntry::relative_age`), computed from a raw `gen_bound8` byte.
///
/// Generations are counted like clock hours: `0 − 1 == 31`. Unsigned
/// (wrapping) subtraction then masking to 5 bits gives the required borrow
/// regardless of the upper pv / bound bits packed alongside. Taking the byte by
/// value lets callers reuse a single relaxed load instead of re-reading the
/// atomic.
#[inline]
fn relative_age(gen_bound8: u8, curr_generation: u8) -> u8 {
    curr_generation.wrapping_sub(gen_bound8) & GENERATION_MASK
}

/// A single transposition-table entry — 10 bytes, laid out field-for-field
/// like the reference's `TTEntry`, with each field an atomic so the entry can
/// be shared `&self` across threads (see the module docs' Threading section).
/// `#[repr(C)]` pins the layout so a [`Cluster`] is exactly 32 bytes and the
/// `clusterCount` sizing arithmetic matches the reference. Every atomic has the
/// same size and alignment as its plain counterpart, so the layout is
/// byte-for-byte the plain-field version's.
#[repr(C)]
#[derive(Default)]
struct TTEntry {
    /// Low 16 bits of the position key (`TTE_KEY_TYPE`).
    key: AtomicU16,
    /// `depth − DEPTH_NONE`; `0` means unoccupied.
    depth8: AtomicU8,
    /// Packed `generation | bound << 5 | pv << 7`.
    gen_bound8: AtomicU8,
    /// Best move fragment (`Move16`).
    move16: AtomicU16,
    /// Search value.
    value16: AtomicI16,
    /// Static / qsearch eval.
    eval16: AtomicI16,
}

impl TTEntry {
    /// Decode the packed bitfields into external types (`TTEntry::read`).
    ///
    /// Each field is loaded independently ([`Ordering::Relaxed`]); under
    /// contention the six loads can straddle a concurrent write and yield a
    /// torn payload (see the module docs). Single-threaded, every load observes
    /// this thread's last write, so the decode is exact.
    #[inline]
    fn read(&self) -> TTData {
        let gen_bound8 = self.gen_bound8.load(REL);
        TTData {
            move16: self.move16.load(REL),
            value: self.value16.load(REL) as Value,
            eval: self.eval16.load(REL) as Value,
            depth: DEPTH_NONE + self.depth8.load(REL) as Depth,
            bound: Bound::from_u8((gen_bound8 & BOUND_MASK) >> BOUND_SHIFT),
            is_pv: (gen_bound8 & PV_MASK) != 0,
        }
    }

    /// Cheap occupancy check (`TTEntry::is_occupied`): `depth8 != 0`, i.e. the
    /// external depth is not `DEPTH_NONE`.
    #[inline]
    fn is_occupied(&self) -> bool {
        self.depth8.load(REL) != 0
    }

    /// Replacement priority used by `probe`: `depth8 − 8 · relative_age`.
    /// Lower is more replaceable. Computed in `i32` to match the reference's
    /// `int` promotion (the subtraction can go negative).
    #[inline]
    fn replace_priority(&self, curr_generation: u8) -> i32 {
        let depth8 = self.depth8.load(REL) as i32;
        let gen_bound8 = self.gen_bound8.load(REL);
        depth8 - 8 * relative_age(gen_bound8, curr_generation) as i32
    }

    /// Zero every field (the per-entry half of [`TranspositionTable::clear`]).
    /// Uses `&mut self` (`get_mut`) so it runs as plain stores under the
    /// lifecycle-exclusivity contract, no atomic ops.
    #[inline]
    fn reset(&mut self) {
        *self.key.get_mut() = 0;
        *self.depth8.get_mut() = 0;
        *self.gen_bound8.get_mut() = 0;
        *self.move16.get_mut() = 0;
        *self.value16.get_mut() = 0;
        *self.eval16.get_mut() = 0;
    }

    /// Store a new node's data, possibly overwriting an older position
    /// (`TTEntry::save`). `k` is the low-16-bit key fragment; `curr_generation`
    /// is the caller-supplied age (the reference lets learners pass a per-thread
    /// generation, so it is an argument rather than read from the table).
    ///
    /// Takes `&self`: the fields are mutated through relaxed atomic stores,
    /// exactly like the reference's racy in-place writes. The old `key` /
    /// `depth8` / `gen_bound8` are each read once up front (before any store),
    /// so the replacement decision sees the pre-save entry state — bit-identical
    /// to the previous plain-field logic in single-threaded execution.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn save(
        &self,
        k: u16,
        v: Value,
        pv: bool,
        b: Bound,
        d: Depth,
        m: u16,
        ev: Value,
        curr_generation: u8,
    ) {
        let old_key = self.key.load(REL);
        let old_depth8 = self.depth8.load(REL);
        let old_gen_bound8 = self.gen_bound8.load(REL);

        // Preserve the old move if we don't have a new one for this position.
        if m != 0 || k != old_key {
            self.move16.store(m, REL);
        }

        // Overwrite less valuable entries (cheapest checks first). The depth
        // comparison is done in i32 to match the reference's int promotion of
        // `depth8` (`depth8 - 4` may be negative).
        if b == Bound::Exact
            || k != old_key
            || d - DEPTH_NONE + 2 * pv as Depth > old_depth8 as Depth - 4
            || relative_age(old_gen_bound8, curr_generation) != 0
        {
            debug_assert!(d > DEPTH_NONE);
            debug_assert!(d - DEPTH_NONE < 256);
            debug_assert!(curr_generation <= GENERATION_MASK);

            self.key.store(k, REL);
            self.depth8.store((d - DEPTH_NONE) as u8, REL);
            self.gen_bound8.store(
                curr_generation | (b as u8) << BOUND_SHIFT | (pv as u8) << PV_SHIFT,
                REL,
            );
            self.value16.store(v as i16, REL);
            self.eval16.store(ev as i16, REL);
        }
    }
}

/// A cluster of [`CLUSTER_SIZE`] entries plus padding to a round 32 bytes.
/// Entries in one cluster share a hash slot; a collision spills into the
/// following entries and is resolved by the replacement policy in `probe`.
#[repr(C)]
#[derive(Default)]
struct Cluster {
    entry: [TTEntry; CLUSTER_SIZE],
    _padding: [u8; 2],
}

// The `clusterCount` sizing arithmetic and cache-line reasoning both assume a
// 32-byte cluster; pin it so a layout regression is a compile error.
const _: () = assert!(size_of::<Cluster>() == 32);

/// Base alignment of the cluster allocation, mirroring the reference's
/// `aligned_large_pages_alloc` (`source/memory.cpp`): a
/// 2 MiB huge-page boundary on Linux so a `MADV_HUGEPAGE` hint can back the
/// region with transparent huge pages, and a 4 KiB page boundary elsewhere
/// (macOS dev machines / CI), where the huge-page path does not apply.
///
/// The same value the shared allocator uses ([`crate::large_page::LARGE_PAGE_ALIGN`]);
/// the TT allocates through that shared helper, so this alias just documents the
/// TT's own dependence on the constant.
const TT_ALLOC_ALIGN: usize = crate::large_page::LARGE_PAGE_ALIGN;

/// The transposition table's owned backing store: a raw, [`TT_ALLOC_ALIGN`]-
/// aligned, zero-initialised block of [`Cluster`]s, mirroring the reference's
/// `aligned_large_pages_alloc` / `aligned_large_pages_free` pair
/// (`source/memory.cpp`, `tt.cpp`).
///
/// Unlike the previous `Box<[Cluster]>` backing store, the allocation size is
/// rounded **up** to a whole multiple of [`TT_ALLOC_ALIGN`] (the tail beyond
/// `len` clusters stays unused), the base pointer is aligned to the huge-page
/// boundary, and — on Linux — the region is `madvise(MADV_HUGEPAGE)`-hinted at
/// allocation time. The exposed slice still covers exactly `len` clusters, so
/// every consumer ([`Deref`]/[`DerefMut`] to `[Cluster]`) sees the same view as
/// before.
struct ClusterArray {
    /// Base of the aligned allocation. Dangling (never dereferenced) when
    /// `len == 0`; always [`TT_ALLOC_ALIGN`]-aligned otherwise.
    ptr: NonNull<Cluster>,
    /// Number of live clusters (`clusterCount`). The slice exposes exactly this
    /// many; the rounded-up allocation tail is not part of it.
    len: usize,
    /// The exact [`Layout`] the block was allocated with (rounded-up size +
    /// [`TT_ALLOC_ALIGN`] alignment), replayed verbatim to [`dealloc`] on drop.
    layout: Layout,
}

// SAFETY: `ClusterArray` owns a heap block of `Cluster`, which is itself
// `Send`/`Sync` (every field is an atomic). The raw `NonNull` backing store
// suppresses the automatic derivation, so it is restored explicitly. The block
// is uniquely owned (freed once, in `Drop`), so moving it across threads and
// sharing `&` to it are both sound.
unsafe impl Send for ClusterArray {}
unsafe impl Sync for ClusterArray {}

impl ClusterArray {
    /// An unsized backing store (`clusterCount == 0`) — allocates nothing,
    /// matching the reference's default-constructed table.
    fn empty() -> Self {
        ClusterArray {
            ptr: NonNull::dangling(),
            len: 0,
            // Zero-sized, correctly aligned; never handed to `dealloc`.
            layout: Layout::from_size_align(0, TT_ALLOC_ALIGN)
                .expect("TT_ALLOC_ALIGN is a valid power-of-two alignment"),
        }
    }

    /// Allocate `cluster_count` zeroed clusters on a [`TT_ALLOC_ALIGN`] boundary,
    /// rounding the byte size up to a whole multiple of the alignment, then (on
    /// Linux) issuing a best-effort `MADV_HUGEPAGE` hint over the whole rounded
    /// region. Delegates to the shared [`crate::large_page::alloc_zeroed_large`]
    /// helper, a faithful port of `aligned_large_pages_alloc`
    /// (`source/memory.cpp`): the Linux path is
    /// silent on both success and failure, and the `madvise` return value is
    /// ignored exactly as the reference ignores it.
    fn alloc(cluster_count: usize) -> Self {
        if cluster_count == 0 {
            return Self::empty();
        }

        // A `Cluster` is 32 bytes, so the product cannot overflow for any table
        // size the driver can request. The shared helper rounds the byte size up
        // to a whole multiple of the alignment (`size = ((allocSize + alignment -
        // 1) / alignment) * alignment`) and issues the `MADV_HUGEPAGE` hint.
        let bytes = cluster_count * size_of::<Cluster>();
        let (raw, layout) = large_page::alloc_zeroed_large(bytes);

        // The returned block is `TT_ALLOC_ALIGN`-aligned and fully zeroed; an
        // all-zero bit pattern is a valid `Cluster` (every field is an atomic
        // integer whose zero value is its `Default`, and the padding is inert),
        // so the block is a valid, unoccupied cluster array — the same
        // post-`clear()` state the reference relies on.
        ClusterArray {
            ptr: raw.cast(),
            len: cluster_count,
            layout,
        }
    }
}

impl Drop for ClusterArray {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: `ptr` came from `large_page::alloc_zeroed_large` with
            // exactly `layout` (an unsized store keeps `len == 0` and is
            // skipped), and it is freed exactly once because `ClusterArray`
            // uniquely owns the block.
            unsafe {
                large_page::free_large(self.ptr.cast(), self.layout);
            }
        }
    }
}

impl Deref for ClusterArray {
    type Target = [Cluster];

    #[inline]
    fn deref(&self) -> &[Cluster] {
        // SAFETY: for `len > 0`, `ptr` addresses `len` contiguous, initialised
        // (`alloc_zeroed`) `Cluster`s within one allocation. For `len == 0` this
        // yields an empty slice, for which `from_raw_parts` accepts any aligned
        // non-null pointer (`NonNull::dangling` is suitably aligned). The borrow
        // is tied to `&self`, so no `&mut` alias can coexist.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for ClusterArray {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Cluster] {
        // SAFETY: as `deref`, but the `&mut self` borrow guarantees exclusivity,
        // so handing out `&mut [Cluster]` introduces no aliasing.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

/// High 64 bits of the 128-bit product `a · b` (`mul_hi64`,
/// `source/misc.h`). Used to map a 64-bit key onto
/// `0..clusterCount` without requiring a power-of-two table size.
#[inline]
fn mul_hi64(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// The engine's single global transposition table — a contiguous array of
/// [`Cluster`]. A faithful port of the reference's `TranspositionTable`.
pub struct TranspositionTable {
    /// Allocated clusters. Empty (length 0) until [`Self::resize`]. Shared
    /// `&self` across search threads; every field inside is atomic. Backed by a
    /// huge-page-aligned, `MADV_HUGEPAGE`-hinted allocation (see
    /// [`ClusterArray`]).
    table: ClusterArray,
    /// Generation counter, bumped once per [`Self::new_search`]. Only the low
    /// [`GENERATION_BITS`] bits are significant. Atomic so the (main) searcher
    /// can bump it through the shared `&self`.
    generation8: AtomicU8,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TranspositionTable {
    /// A fresh, empty table (`clusterCount == 0`), matching the reference's
    /// default constructor. Call [`Self::resize`] before use.
    pub fn new() -> Self {
        TranspositionTable {
            table: ClusterArray::empty(),
            generation8: AtomicU8::new(0),
        }
    }

    /// Number of allocated clusters (the reference's `clusterCount`).
    #[inline]
    pub fn cluster_count(&self) -> usize {
        self.table.len()
    }

    /// Base virtual address of the cluster allocation, for Linux transparent-
    /// huge-page diagnostics: the caller can match this against a
    /// `/proc/self/smaps` region to read its `AnonHugePages` figure and see how
    /// much of the TT the kernel actually backed with huge pages. Returns `0`
    /// on an unsized table (`clusterCount == 0`). Carries no other semantics.
    #[inline]
    pub fn backing_ptr_addr(&self) -> usize {
        if self.table.is_empty() {
            0
        } else {
            self.table.as_ptr() as usize
        }
    }

    /// Resize the table to `mb_size` MiB and clear it
    /// (`TranspositionTable::resize`).
    ///
    /// `clusterCount = mb_size · 1024 · 1024 / sizeof(Cluster)`, which is always
    /// even (`sizeof(Cluster) == 32` divides `1 MiB`), as required so that
    /// folding the side-to-move into cluster-index bit 0 stays in range. If the
    /// requested size yields the current cluster count the table is left
    /// untouched (no reallocation, no clear), exactly like the reference.
    pub fn resize(&mut self, mb_size: usize) {
        let new_cluster_count = mb_size * 1024 * 1024 / size_of::<Cluster>();
        debug_assert!(new_cluster_count & 1 == 0);

        if new_cluster_count == self.table.len() {
            return;
        }

        // A freshly allocated region is zeroed (`alloc_zeroed`), i.e. every
        // entry is unoccupied (`depth8 == 0`); this stands in for the
        // reference's post-resize `clear()`. Dropping the previous
        // `ClusterArray` frees the old allocation with its matching layout —
        // mirroring the reference's `aligned_large_pages_free` before the new
        // `aligned_large_pages_alloc`.
        self.table = ClusterArray::alloc(new_cluster_count);
    }

    /// Zero every entry and reset the generation (`TranspositionTable::clear`).
    /// A `&mut self` lifecycle operation (see the module docs' exclusivity
    /// contract); the field writes are plain stores via `get_mut`, not atomic
    /// ops.
    pub fn clear(&mut self) {
        *self.generation8.get_mut() = 0;
        for cluster in self.table.iter_mut() {
            for entry in cluster.entry.iter_mut() {
                entry.reset();
            }
        }
    }

    /// Bump the generation at the start of a root search
    /// (`TranspositionTable::new_search`). Wraps within
    /// [`GENERATION_BITS`] so it never spills into the bound / pv bits of
    /// `genBound8`. Takes `&self`: the (main) searcher advances the generation
    /// through the shared table, matching the reference where `new_search` runs
    /// on the shared `TranspositionTable` before the workers start.
    pub fn new_search(&self) {
        let next = self.generation8.load(REL).wrapping_add(1) & GENERATION_MASK;
        self.generation8.store(next, REL);
    }

    /// The current generation used when writing new data
    /// (`TranspositionTable::generation`).
    #[inline]
    pub fn generation(&self) -> u8 {
        self.generation8.load(REL)
    }

    /// Approximate table occupancy in permille, counting only entries younger
    /// than `max_age` (`TranspositionTable::hashfull`). Samples the first 1000
    /// clusters; the table must hold at least that many.
    pub fn hashfull(&self, max_age: u8) -> u32 {
        let generation = self.generation8.load(REL);
        let mut cnt = 0u32;
        for cluster in self.table.iter().take(1000) {
            for entry in &cluster.entry {
                if entry.is_occupied()
                    && relative_age(entry.gen_bound8.load(REL), generation) <= max_age
                {
                    cnt += 1;
                }
            }
        }
        cnt / CLUSTER_SIZE as u32
    }

    /// Cluster index for `key` with `side_to_move` (0 or 1) folded into bit 0
    /// (`TranspositionTable::first_entry`).
    ///
    /// `mul_hi64(key, clusterCount)` lands in `0..clusterCount`; clearing bit 0
    /// and OR-ing in the side keeps it in range (because `clusterCount` is
    /// even) while guaranteeing the two sides never share a cluster.
    #[inline]
    fn cluster_index(&self, key: u64, side_to_move: u8) -> usize {
        let index = mul_hi64(key, self.table.len() as u64) as usize;
        (index & !1) | (side_to_move as usize & 1)
    }

    /// Software-prefetch the cluster that [`Self::probe`] / [`Self::locate`]
    /// would select for `(key, side_to_move)`, issuing an L1/L2 prefetch hint
    /// (`_MM_HINT_T0`) for the cluster's cache line. Uses the *exact* same index
    /// math as [`Self::cluster_index`], so the line brought in is the one the
    /// subsequent probe will read.
    ///
    /// The reference (`Position::do_move`,
    /// `source/position.cpp` ~1837-1842 / ~1996-2001) issues
    /// this prefetch mid-`do_move`, the instant the post-move hash key is known,
    /// because its `Position` holds a TT pointer. This port's `Position`
    /// (attic-state) must not depend on the TT (attic-storage) under the
    /// layering rules, so the hint is issued from the Search layer just
    /// after `do_move` returns — a few nanoseconds later than the reference, but
    /// still well ahead of the child's TT probe. An accepted, output-preserving
    /// placement difference: a prefetch has no architectural semantics.
    ///
    /// A no-op on an unsized table (`clusterCount == 0`) and on non-x86-64
    /// targets.
    #[inline]
    pub fn prefetch(&self, key: u64, side_to_move: u8) {
        #[cfg(target_arch = "x86_64")]
        {
            // Guard the unsized table: `cluster_index` would form an
            // out-of-bounds pointer (bit 0 can be set with zero clusters).
            if self.table.is_empty() {
                return;
            }
            let ci = self.cluster_index(key, side_to_move);
            // SAFETY: `_mm_prefetch` is a pure hardware hint — it neither reads
            // nor writes the pointed-to memory in any observable way and cannot
            // fault, so it has no memory-safety effect. `ci` is in
            // `0..table.len()` (guaranteed by `cluster_index`, whose result is
            // `mul_hi64(key, len)` masked into range), so the pointer is a live,
            // in-bounds address of an allocated `Cluster`. This is the sole
            // `unsafe` block in the module; the prefetch is the whole reason for
            // it. (`_mm_prefetch` itself carries no preconditions.)
            unsafe {
                use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                let ptr = self.table.as_ptr().add(ci) as *const i8;
                _mm_prefetch::<_MM_HINT_T0>(ptr);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (key, side_to_move);
        }
    }

    /// Look up `key` (`TranspositionTable::probe`).
    ///
    /// Returns `(found, data, writer)`:
    /// - `found` is `true` when a matching, occupied entry exists (possibly a
    ///   16-bit key collision);
    /// - `data` is a copy of that entry's payload, or [`TTData::none`] on a
    ///   miss;
    /// - `writer` targets the matching entry on a hit, or the least-valuable
    ///   entry to replace on a miss.
    ///
    /// Panics if the table has not been sized (`clusterCount == 0`).
    pub fn probe(&self, key: u64, side_to_move: u8) -> (bool, TTData, TTWriter<'_>) {
        assert!(
            !self.table.is_empty(),
            "TranspositionTable::probe called before resize"
        );

        let ci = self.cluster_index(key, side_to_move);
        let key16 = key as u16;
        let generation = self.generation8.load(REL);
        let cluster = &self.table[ci];

        // The low 16 bits are the in-cluster key. On a match, return that
        // entry's data and a writer to it.
        if let Some(i) = (0..CLUSTER_SIZE).find(|&i| cluster.entry[i].key.load(REL) == key16) {
            let found = cluster.entry[i].is_occupied();
            let data = cluster.entry[i].read();
            return (found, data, TTWriter::new(&cluster.entry[i]));
        }

        // Miss: pick the entry to replace. Value = depth − 8·relative_age;
        // lower is more replaceable.
        let mut replace = 0;
        for i in 1..CLUSTER_SIZE {
            if cluster.entry[replace].replace_priority(generation)
                > cluster.entry[i].replace_priority(generation)
            {
                replace = i;
            }
        }

        (
            false,
            TTData::none(),
            TTWriter::new(&cluster.entry[replace]),
        )
    }

    /// Like [`Self::probe`] but returns the chosen entry's *location*
    /// ([`TtSlot`]) instead of a borrowing [`TTWriter`]. The location is a pair
    /// of plain indices, so the caller can hold it across the recursive search
    /// calls that also mutate the table and later write to the **same** physical
    /// entry via [`Self::write_at`] — reproducing the reference, which obtains
    /// one `TTWriter` at the node's Step-4 probe and writes through it at every
    /// write site (Step 5 / Step 6 / ProbCut / the post-move-loop tail). A
    /// re-probe at write time would re-run the replacement selection against a
    /// cluster the children have since churned and could land on a different
    /// slot, drifting the stored TT state and hence later probes.
    ///
    /// Panics if the table has not been sized (`clusterCount == 0`).
    pub fn locate(&self, key: u64, side_to_move: u8) -> (bool, TTData, TtSlot) {
        assert!(
            !self.table.is_empty(),
            "TranspositionTable::locate called before resize"
        );

        let ci = self.cluster_index(key, side_to_move);
        let key16 = key as u16;
        let generation = self.generation8.load(REL);
        let cluster = &self.table[ci];

        if let Some(i) = (0..CLUSTER_SIZE).find(|&i| cluster.entry[i].key.load(REL) == key16) {
            let found = cluster.entry[i].is_occupied();
            let data = cluster.entry[i].read();
            return (
                found,
                data,
                TtSlot {
                    cluster: ci,
                    entry: i,
                },
            );
        }

        let mut replace = 0;
        for i in 1..CLUSTER_SIZE {
            if cluster.entry[replace].replace_priority(generation)
                > cluster.entry[i].replace_priority(generation)
            {
                replace = i;
            }
        }

        (
            false,
            TTData::none(),
            TtSlot {
                cluster: ci,
                entry: replace,
            },
        )
    }

    /// Store into the exact entry captured by [`Self::locate`], running the same
    /// replacement policy as [`TTWriter::write`]. Because the entry is addressed
    /// by index (not re-selected), a write here targets the reference's held-
    /// writer slot even when a child has since overwritten that entry.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn write_at(
        &self,
        slot: TtSlot,
        key: u64,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: Depth,
        mv: u16,
        eval: Value,
        generation: u8,
    ) {
        self.table[slot.cluster].entry[slot.entry]
            .save(key as u16, value, pv, bound, depth, mv, eval, generation);
    }

    /// A stable checksum over the whole table's raw bytes, for determinism
    /// tests (two identical operation sequences must yield identical tables).
    pub fn checksum(&self) -> u64 {
        // FNV-1a over every entry field, in declaration order.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| {
            h ^= x;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        for cluster in self.table.iter() {
            for e in &cluster.entry {
                mix(e.key.load(REL) as u64);
                mix(e.depth8.load(REL) as u64);
                mix(e.gen_bound8.load(REL) as u64);
                mix(e.move16.load(REL) as u64);
                mix(e.value16.load(REL) as u16 as u64);
                mix(e.eval16.load(REL) as u16 as u64);
            }
        }
        mix(self.generation8.load(REL) as u64);
        h
    }
}

/// The location of a resolved TT entry (cluster + in-cluster index), captured
/// at [`TranspositionTable::locate`] time. Holds no borrow, so it survives the
/// recursive search calls between a node's probe and its writes.
#[derive(Clone, Copy, Debug)]
pub struct TtSlot {
    cluster: usize,
    entry: usize,
}

/// A thin, single-use handle for writing one entry (`TTWriter`). Obtained from
/// [`TranspositionTable::probe`]; it borrows the chosen entry `&self` and
/// writes through the entry's atomics (racy in-place, like the reference).
pub struct TTWriter<'a> {
    entry: &'a TTEntry,
}

impl<'a> TTWriter<'a> {
    #[inline]
    fn new(entry: &'a TTEntry) -> Self {
        TTWriter { entry }
    }

    /// Store data into the targeted entry, subject to the replacement policy
    /// (`TTWriter::write`). `key` is the full 64-bit position key (its low 16
    /// bits become the stored fragment); `mv` is the 16-bit move fragment;
    /// `generation` is the current [`TranspositionTable::generation`].
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        self,
        key: u64,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: Depth,
        mv: u16,
        eval: Value,
        generation: u8,
    ) {
        self.entry
            .save(key as u16, value, pv, bound, depth, mv, eval, generation);
    }
}

#[cfg(test)]
mod alloc_tests {
    //! Low-level checks on the huge-page-aligned backing store
    //! ([`ClusterArray`]), which the public API cannot observe directly:
    //! base-pointer alignment, size round-up, zeroed initial contents, and the
    //! unsized (`clusterCount == 0`) shape. The public resize/probe behaviour is
    //! covered by the integration gates in `tests/tt_basic.rs`.

    use super::*;

    /// Alignment the current target uses for the TT allocation — 2 MiB on
    /// Linux (huge-page boundary), 4 KiB elsewhere. Mirrors [`TT_ALLOC_ALIGN`].
    const EXPECTED_ALIGN: usize = if cfg!(target_os = "linux") {
        2 * 1024 * 1024
    } else {
        4096
    };

    #[test]
    fn align_constant_matches_target() {
        assert_eq!(TT_ALLOC_ALIGN, EXPECTED_ALIGN);
    }

    #[test]
    fn base_pointer_is_aligned() {
        // A range of sizes, including one whose byte size is far below the
        // alignment and one spanning many huge pages.
        for &cc in &[2usize, 32768, 3 * 32768, 1024 * 1024 / size_of::<Cluster>()] {
            let a = ClusterArray::alloc(cc);
            assert_eq!(
                a.ptr.as_ptr() as usize % TT_ALLOC_ALIGN,
                0,
                "base pointer for {cc} clusters not {TT_ALLOC_ALIGN}-aligned",
            );
            assert_eq!(a.len, cc);
        }
    }

    #[test]
    fn allocation_size_rounds_up_to_alignment() {
        // Tiny request: byte size (64 B) is far below the alignment, so it must
        // round up to exactly one alignment unit.
        let a = ClusterArray::alloc(2);
        assert_eq!(a.layout.size(), TT_ALLOC_ALIGN);
        assert_eq!(a.layout.align(), TT_ALLOC_ALIGN);

        // Request whose exact byte size is not a whole multiple of the
        // alignment must round strictly up, and always cover the live clusters.
        let cc = 32768 + 1; // 32769 * 32 B = 1 MiB + 32 B
        let a = ClusterArray::alloc(cc);
        let bytes = cc * size_of::<Cluster>();
        assert_eq!(a.layout.size() % TT_ALLOC_ALIGN, 0);
        assert!(a.layout.size() >= bytes);
        assert!(a.layout.size() - bytes < TT_ALLOC_ALIGN);
    }

    #[test]
    fn fresh_allocation_is_zeroed() {
        let a = ClusterArray::alloc(4096);
        for cluster in a.iter() {
            for e in &cluster.entry {
                assert!(!e.is_occupied());
                assert_eq!(e.key.load(REL), 0);
                assert_eq!(e.gen_bound8.load(REL), 0);
                assert_eq!(e.move16.load(REL), 0);
                assert_eq!(e.value16.load(REL), 0);
                assert_eq!(e.eval16.load(REL), 0);
            }
            assert_eq!(cluster._padding, [0u8; 2]);
        }
    }

    #[test]
    fn empty_backing_store_allocates_nothing() {
        let a = ClusterArray::empty();
        assert_eq!(a.len, 0);
        assert!(a.is_empty());
        // `alloc(0)` takes the same unsized path.
        let z = ClusterArray::alloc(0);
        assert_eq!(z.len, 0);
        assert!(z.is_empty());
    }

    #[test]
    fn repeated_alloc_and_drop_reuses_cleanly() {
        // Exercise the alloc/Drop pair many times; under the matching-layout
        // free, this neither leaks nor corrupts the allocator.
        for _ in 0..64 {
            let a = ClusterArray::alloc(32768);
            assert_eq!(a.len, 32768);
            drop(a);
        }
    }
}
