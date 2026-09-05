//! Transposition table, a port of the reference `tt.h` / `tt.cpp` specialised
//! to the engine's default build configuration: a 64-bit position key whose low
//! 16 bits are the in-cluster fragment, and three 10-byte entries per 32-byte
//! cluster (`TT_CLUSTER_SIZE == 3`).
//!
//! Those choices are load-bearing for search-node parity. The cluster size and
//! the `clusterCount = mb·2²⁰ / sizeof(Cluster)` arithmetic together decide
//! which positions collide, and the in-cluster replacement policy decides which
//! entry survives a collision.
//!
//! The Storage layer must not depend on the State layer, so this module speaks
//! in primitives: a `u64` key, a `u8` side-to-move, and the best move as its
//! low-16-bit fragment. Widening that fragment and validating it against a
//! position is the Search layer's job.
//!
//! # Threading
//!
//! The table is shared as `Arc<TranspositionTable>`, and every entry field is
//! an atomic accessed with [`Ordering::Relaxed`] — the Rust-sound equivalent of
//! the reference's racy, lock-free `TTEntry`.
//!
//! Relaxed atomics make each *field* access indivisible, but an entry is six
//! fields with no cross-field atomicity, so a concurrent write can land between
//! this thread's reads and [`TTEntry::read`] can return a payload mixing an old
//! key fragment with a new value. **This is tolerated, as the reference
//! tolerates it**: a mismatched fragment reads back as a miss or a
//! wrong-position hit, and the Search layer validates every TT move against the
//! actual position before trusting it.
//!
//! [`TranspositionTable::resize`] and [`TranspositionTable::clear`] take
//! `&mut self`, so the driver must reach them through [`Arc::get_mut`], which
//! returns `Some` only once every worker has dropped its clone. The type system
//! therefore enforces that a resize or clear never races a probe.

use std::alloc::Layout;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicI16, AtomicU8, AtomicU16, Ordering};

use crate::large_page;

/// The single memory ordering used for every entry field access, matching the
/// reference's unsynchronised reads and writes.
const REL: Ordering = Ordering::Relaxed;

/// Search value, mirroring the reference's `int`-typed `Value`. Stored in an
/// entry truncated to `i16`, which every real value fits.
pub type Value = i32;

/// Search depth, mirroring the reference's `int`-typed `Depth`. Stored in an
/// entry offset by [`DEPTH_NONE`] and truncated to `u8`.
pub type Depth = i32;

/// `DEPTH_NONE` (`types.h`). Entries store `depth8 = depth − DEPTH_NONE`, so an
/// all-zero entry reads back as `DEPTH_NONE` and counts as unoccupied.
pub const DEPTH_NONE: Depth = -3;

/// `VALUE_NONE` (`types.h`), the sentinel returned for a miss.
pub const VALUE_NONE: Value = 32002;

/// The reference's default `USI_Hash` in MiB. Only the default the driver
/// passes to `resize`: a fresh table is empty, like the reference's
/// default-constructed one.
pub const DEFAULT_HASH_MB: usize = 1024;

// The `genBound8` bit layout (`tt.cpp`):
// `generation (5) | bound (2) << 5 | pv (1) << 7`.
const GENERATION_BITS: u8 = 5;
const GENERATION_MASK: u8 = (1 << GENERATION_BITS) - 1;
const BOUND_SHIFT: u8 = GENERATION_BITS;
const BOUND_MASK: u8 = 0b11 << BOUND_SHIFT;
const PV_SHIFT: u8 = BOUND_SHIFT + 2;
const PV_MASK: u8 = 1 << PV_SHIFT;

/// Number of entries per cluster (`TT_CLUSTER_SIZE == 3`).
const CLUSTER_SIZE: usize = 3;

/// Bound type of a stored value (`types.h`). The discriminants matter:
/// `Exact == Upper | Lower`, and the value is packed into `genBound8`.
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

/// A decoded copy of an entry's payload (`TTData`). Nothing here borrows the
/// table.
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
    /// The miss sentinel a failed [`TranspositionTable::probe`] returns.
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

/// This entry's age relative to `curr_generation` (`TTEntry::relative_age`),
/// computed from a raw `gen_bound8` byte.
///
/// Generations are counted like clock hours: `0 − 1 == 31`. Wrapping
/// subtraction then masking to 5 bits gives the required borrow whatever the
/// pv / bound bits packed alongside hold.
#[inline]
fn relative_age(gen_bound8: u8, curr_generation: u8) -> u8 {
    curr_generation.wrapping_sub(gen_bound8) & GENERATION_MASK
}

/// A single transposition-table entry — 10 bytes, laid out field-for-field like
/// the reference's `TTEntry`, with each field an atomic. `#[repr(C)]` pins the
/// layout so that a [`Cluster`] is exactly 32 bytes and the `clusterCount`
/// arithmetic matches the reference.
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
    /// Under contention the six independent loads can straddle a concurrent
    /// write and yield a torn payload; see the module head.
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

    /// `TTEntry::is_occupied`: the external depth is not `DEPTH_NONE`.
    #[inline]
    fn is_occupied(&self) -> bool {
        self.depth8.load(REL) != 0
    }

    /// Replacement priority, lower being more replaceable. Computed in `i32` to
    /// match the reference's `int` promotion: the subtraction can go negative.
    #[inline]
    fn replace_priority(&self, curr_generation: u8) -> i32 {
        let depth8 = self.depth8.load(REL) as i32;
        let gen_bound8 = self.gen_bound8.load(REL);
        depth8 - 8 * relative_age(gen_bound8, curr_generation) as i32
    }

    /// Zero every field. Takes `&mut self`, so the writes are plain stores
    /// under the exclusivity contract rather than atomic ops.
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
    /// (`TTEntry::save`). `curr_generation` is an argument rather than read from
    /// the table because the reference lets learners pass a per-thread one.
    ///
    /// The old `key` / `depth8` / `gen_bound8` are each read once before any
    /// store, so the replacement decision sees the pre-save entry state.
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

        if m != 0 || k != old_key {
            self.move16.store(m, REL);
        }

        // The depth comparison is in `i32` to match the reference's `int`
        // promotion: `depth8 - 4` may be negative.
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
/// Entries in one cluster share a hash slot.
#[repr(C)]
#[derive(Default)]
struct Cluster {
    entry: [TTEntry; CLUSTER_SIZE],
    _padding: [u8; 2],
}

// The `clusterCount` arithmetic assumes a 32-byte cluster, so a layout
// regression must be a compile error.
const _: () = assert!(size_of::<Cluster>() == 32);

/// Base alignment of the cluster allocation, an alias for the shared
/// [`crate::large_page::LARGE_PAGE_ALIGN`] the TT allocates through.
const TT_ALLOC_ALIGN: usize = crate::large_page::LARGE_PAGE_ALIGN;

/// The transposition table's owned backing store: a raw, [`TT_ALLOC_ALIGN`]-
/// aligned, zero-initialised block of [`Cluster`]s. The exposed slice covers
/// exactly `len` clusters; the round-up tail of the allocation is not part of
/// it.
struct ClusterArray {
    /// Base of the aligned allocation. Dangling (never dereferenced) when
    /// `len == 0`; always [`TT_ALLOC_ALIGN`]-aligned otherwise.
    ptr: NonNull<Cluster>,
    /// Number of live clusters (`clusterCount`).
    len: usize,
    /// The exact [`Layout`] the block was allocated with, replayed verbatim to
    /// [`dealloc`] on drop.
    layout: Layout,
}

// SAFETY: `ClusterArray` uniquely owns a heap block of `Cluster`, freed once in
// `Drop`; `Cluster` is itself `Send`/`Sync`, every field being an atomic. The
// raw `NonNull` suppresses the automatic derivation, so it is restored here.
unsafe impl Send for ClusterArray {}
unsafe impl Sync for ClusterArray {}

impl ClusterArray {
    /// An unsized backing store (`clusterCount == 0`), allocating nothing.
    fn empty() -> Self {
        ClusterArray {
            ptr: NonNull::dangling(),
            len: 0,
            layout: Layout::from_size_align(0, TT_ALLOC_ALIGN)
                .expect("TT_ALLOC_ALIGN is a valid power-of-two alignment"),
        }
    }

    /// Allocate `cluster_count` zeroed clusters through the shared
    /// [`crate::large_page::alloc_zeroed_large`].
    fn alloc(cluster_count: usize) -> Self {
        if cluster_count == 0 {
            return Self::empty();
        }

        // A `Cluster` is 32 bytes, so the product cannot overflow for any table
        // size the driver can request.
        let bytes = cluster_count * size_of::<Cluster>();
        let (raw, layout) = large_page::alloc_zeroed_large(bytes);

        // An all-zero bit pattern is a valid, unoccupied `Cluster`, so the
        // zeroed block stands in for the reference's post-resize `clear()`.
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

/// High 64 bits of the 128-bit product `a · b` (`mul_hi64`, `misc.h`), which
/// maps a key onto `0..clusterCount` without a power-of-two table size.
#[inline]
fn mul_hi64(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// The engine's single global transposition table, a contiguous array of
/// [`Cluster`].
pub struct TranspositionTable {
    /// Allocated clusters, empty until [`Self::resize`].
    table: ClusterArray,
    /// Generation counter, bumped once per [`Self::new_search`]. Only the low
    /// [`GENERATION_BITS`] bits are significant.
    generation8: AtomicU8,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TranspositionTable {
    /// A fresh, empty table. Call [`Self::resize`] before use.
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

    /// Base virtual address of the cluster allocation, so that a caller can
    /// match it against a `/proc/self/smaps` region and read how much of the TT
    /// the kernel backed with huge pages. `0` on an unsized table.
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
    /// `clusterCount = mb_size · 1024 · 1024 / sizeof(Cluster)` is always even,
    /// which is what keeps the side-to-move fold into cluster-index bit 0 in
    /// range. A request yielding the current cluster count leaves the table
    /// untouched, as the reference does.
    pub fn resize(&mut self, mb_size: usize) {
        let new_cluster_count = mb_size * 1024 * 1024 / size_of::<Cluster>();
        debug_assert!(new_cluster_count & 1 == 0);

        if new_cluster_count == self.table.len() {
            return;
        }

        // A freshly allocated region is zeroed, so every entry is unoccupied;
        // this stands in for the reference's post-resize `clear()`.
        self.table = ClusterArray::alloc(new_cluster_count);
    }

    /// Zero every entry and reset the generation
    /// (`TranspositionTable::clear`).
    pub fn clear(&mut self) {
        *self.generation8.get_mut() = 0;
        for cluster in self.table.iter_mut() {
            for entry in cluster.entry.iter_mut() {
                entry.reset();
            }
        }
    }

    /// Bump the generation at the start of a root search
    /// (`TranspositionTable::new_search`), wrapping within [`GENERATION_BITS`]
    /// so that it never spills into the bound / pv bits of `genBound8`.
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
    /// clusters, so the table must hold at least that many.
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
    /// and OR-ing in the side keeps it in range, `clusterCount` being even,
    /// while guaranteeing the two sides never share a cluster.
    #[inline]
    fn cluster_index(&self, key: u64, side_to_move: u8) -> usize {
        let index = mul_hi64(key, self.table.len() as u64) as usize;
        (index & !1) | (side_to_move as usize & 1)
    }

    /// Software-prefetch the cluster [`Self::probe`] would select for
    /// `(key, side_to_move)`. A no-op on an unsized table and off x86-64.
    ///
    /// The reference issues this mid-`do_move` (`position.cpp`), the
    /// instant the post-move key is known, because its `Position` holds a TT
    /// pointer. The layering rules forbid that here, so the hint comes from the
    /// Search layer just after `do_move` returns — a few nanoseconds later, and
    /// output-preserving, a prefetch having no architectural semantics.
    #[inline]
    pub fn prefetch(&self, key: u64, side_to_move: u8) {
        #[cfg(target_arch = "x86_64")]
        {
            // On an unsized table `cluster_index` would form an out-of-bounds
            // pointer: bit 0 can be set with zero clusters.
            if self.table.is_empty() {
                return;
            }
            let ci = self.cluster_index(key, side_to_move);
            // SAFETY: `_mm_prefetch` is a pure hardware hint — it neither reads
            // nor writes the pointed-to memory in any observable way and cannot
            // fault. `ci` is in `0..table.len()`, guaranteed by
            // `cluster_index`, so the pointer is a live, in-bounds address of an
            // allocated `Cluster`.
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

        if let Some(i) = (0..CLUSTER_SIZE).find(|&i| cluster.entry[i].key.load(REL) == key16) {
            let found = cluster.entry[i].is_occupied();
            let data = cluster.entry[i].read();
            return (found, data, TTWriter::new(&cluster.entry[i]));
        }

        // On a miss, replace the least valuable entry.
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

    /// Like [`Self::probe`] but returning the chosen entry's *location* rather
    /// than a borrowing [`TTWriter`], so that the caller can hold it across the
    /// recursive search calls that also mutate the table. The reference obtains
    /// one `TTWriter` at a node's probe and writes through it at every write
    /// site; a re-probe at write time would re-run the replacement selection
    /// against a cluster the children have since churned, and could land on a
    /// different slot.
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

    /// Store into the exact entry [`Self::locate`] captured, under the same
    /// replacement policy as [`TTWriter::write`].
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

    /// A stable checksum over the whole table, for determinism tests.
    pub fn checksum(&self) -> u64 {
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

/// The location of a resolved TT entry, captured at
/// [`TranspositionTable::locate`] time. Holds no borrow, so it survives the
/// recursive search calls between a node's probe and its writes.
#[derive(Clone, Copy, Debug)]
pub struct TtSlot {
    cluster: usize,
    entry: usize,
}

/// A single-use handle for writing one entry (`TTWriter`), obtained from
/// [`TranspositionTable::probe`].
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
    //! Checks on the backing store that the public API cannot observe:
    //! base-pointer alignment, size round-up, zeroed initial contents, and the
    //! unsized shape.

    use super::*;

    /// The alignment the current target uses, mirroring [`TT_ALLOC_ALIGN`].
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
        // A byte size far below the alignment must round up to exactly one
        // alignment unit.
        let a = ClusterArray::alloc(2);
        assert_eq!(a.layout.size(), TT_ALLOC_ALIGN);
        assert_eq!(a.layout.align(), TT_ALLOC_ALIGN);

        // A byte size that is not a whole multiple of the alignment must round
        // strictly up, and still cover the live clusters.
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
        let z = ClusterArray::alloc(0);
        assert_eq!(z.len, 0);
        assert!(z.is_empty());
    }

    #[test]
    fn repeated_alloc_and_drop_reuses_cleanly() {
        // Under a matching-layout free, repetition neither leaks nor corrupts
        // the allocator.
        for _ in 0..64 {
            let a = ClusterArray::alloc(32768);
            assert_eq!(a.len, 32768);
            drop(a);
        }
    }
}
