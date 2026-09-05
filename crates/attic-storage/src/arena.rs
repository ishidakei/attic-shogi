//! One large-page allocation carved into many 64-byte-aligned typed sub-arrays,
//! the shape the reference gets from `make_unique_large_page<NnueNetworks>`
//! (`evaluate_nnue.cpp`) and `make_unique_large_page<Search::Worker>`
//! (`thread.cpp`).
//!
//! # The self-referential shape
//!
//! An [`ArenaSlice`] is a raw-pointer view into the arena's heap buffer; it does
//! **not** own or borrow-check against the arena. The intended use is that one
//! struct owns *both* the [`LargePageArena`] and the [`ArenaSlice`]s carved from
//! it. That is sound because the backing bytes live on the heap behind a
//! `NonNull`: moving the owning struct copies the pointer, never the bytes, so
//! every view stays valid, and [`ArenaSlice`] has no `Drop` glue, so field-drop
//! order cannot produce a use-after-free.
//!
//! The owner must keep the arena alive as long as the views, and never hand out
//! a `&mut` to the arena while a view aliases it. Both fall out of "fill through
//! `slice_mut`, then only read through the views".

use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

use crate::large_page::{LargePageArray, Zeroable, rounded_size};

/// Alignment every carved sub-array starts on — the 64-byte cache line the
/// AVX-512 NNUE kernels assume.
pub const ARENA_SUB_ALIGN: usize = 64;

/// The location of one carved sub-array within an arena. Pure `Copy` metadata,
/// so a layout can be recorded once and replayed against any arena built from
/// it — a NUMA replica, say.
pub struct Section<T> {
    offset: usize,
    len: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Section<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Section<T> {}

impl<T> fmt::Debug for Section<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Section")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

impl<T> Section<T> {
    /// The element count this section covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the section covers zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The byte offset of the section's first element from the arena base.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The exclusive end byte offset (`offset + len * size_of::<T>()`).
    pub fn end(&self) -> usize {
        self.offset + self.len * size_of::<T>()
    }
}

/// Accumulates a packed, [`ARENA_SUB_ALIGN`]-aligned layout of typed sub-arrays.
/// Its cursor only ever moves forward, so the reserved sections are provably
/// non-overlapping and all fall inside `[0, total_bytes)`.
#[derive(Clone, Debug, Default)]
pub struct ArenaLayout {
    cursor: usize,
}

impl ArenaLayout {
    /// An empty layout (cursor at 0).
    pub fn new() -> Self {
        Self { cursor: 0 }
    }

    /// Reserve space for `count` elements of `T`, 64-byte aligned, and return the
    /// [`Section`] describing it.
    ///
    /// # Panics
    /// Panics if `align_of::<T>()` exceeds [`ARENA_SUB_ALIGN`] (so the 64-byte
    /// start is a valid `T` alignment) or if the byte maths overflow `usize`.
    pub fn reserve<T>(&mut self, count: usize) -> Section<T> {
        assert!(
            align_of::<T>() <= ARENA_SUB_ALIGN,
            "arena sub-array alignment {} exceeds {ARENA_SUB_ALIGN}",
            align_of::<T>(),
        );
        let offset = self.cursor.next_multiple_of(ARENA_SUB_ALIGN);
        let bytes = count
            .checked_mul(size_of::<T>())
            .expect("arena section byte size overflows usize");
        self.cursor = offset
            .checked_add(bytes)
            .expect("arena cursor overflows usize");
        Section {
            offset,
            len: count,
            _marker: PhantomData,
        }
    }

    /// The total byte size reserved so far — the size the backing arena covers.
    pub fn total_bytes(&self) -> usize {
        self.cursor
    }
}

/// A single large-page allocation carved into typed sub-arrays, reached either
/// by [`slice_mut`](Self::slice_mut) for filling or [`view`](Self::view) for
/// storing alongside the arena.
pub struct LargePageArena {
    /// The single backing allocation. Its length is `total_bytes.max(1)`, so
    /// that a zero-byte layout still has a real, non-dangling base.
    backing: LargePageArray<u8>,
    /// The exact byte size the layout requested (`<= backing.len()`).
    total_bytes: usize,
}

impl fmt::Debug for LargePageArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LargePageArena")
            .field("total_bytes", &self.total_bytes)
            .field("reserved_bytes", &self.reserved_bytes())
            .finish()
    }
}

impl LargePageArena {
    /// Allocate a zeroed arena covering `layout`.
    pub fn new(layout: &ArenaLayout) -> Self {
        Self::with_capacity(layout.total_bytes())
    }

    /// Allocate a zeroed arena of exactly `total_bytes`, for when the layout
    /// total was recorded separately.
    pub fn with_capacity(total_bytes: usize) -> Self {
        // At least one byte, so that even a degenerate empty layout gets a
        // real allocation with a non-null, aligned base.
        let backing = LargePageArray::<u8>::zeroed(total_bytes.max(1));
        Self {
            backing,
            total_bytes,
        }
    }

    /// The exact byte size the layout requested.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// The byte size the allocator actually committed: the request rounded up
    /// to a whole [`LARGE_PAGE_ALIGN`](crate::LARGE_PAGE_ALIGN) multiple.
    pub fn reserved_bytes(&self) -> usize {
        rounded_size(self.total_bytes.max(1))
    }

    /// A mutable view of `section` for in-place filling. Each call borrows the
    /// arena exclusively, so two of these never alias.
    ///
    /// # Panics
    /// Panics if `section` is not fully inside this arena (only possible if it
    /// came from a different, larger layout).
    pub fn slice_mut<T: Zeroable>(&mut self, section: Section<T>) -> &mut [T] {
        assert!(
            section.end() <= self.total_bytes,
            "arena section {section:?} out of bounds (arena has {} bytes)",
            self.total_bytes,
        );
        if section.len == 0 {
            return &mut [];
        }
        // SAFETY: `offset` is 64-aligned (hence aligned for `T`) and the section
        // lies inside the backing `[u8]` (asserted above). The bytes were zeroed
        // at allocation and `T: Zeroable`, so they form `section.len` valid `T`.
        // `&mut self` guarantees no other view of these bytes is live. The base
        // is taken from `base_nonnull()`, so the pointer carries the
        // allocation's raw, write-capable provenance and never routes through a
        // `&[u8]`/`&mut [u8]` reference reborrow.
        unsafe {
            let p = self.backing.base_nonnull().as_ptr().add(section.offset) as *mut T;
            slice::from_raw_parts_mut(p, section.len)
        }
    }

    /// A detached [`ArenaSlice`] view of `section`, valid as long as this
    /// arena's backing allocation lives.
    ///
    /// # Contract
    /// At most one live view may exist per section, and no
    /// [`slice_mut`](Self::slice_mut) call may overlap a live view: an
    /// `ArenaSlice` can hand out `&mut [T]`, so either would alias it. The
    /// owning structs uphold this by carving each section exactly once at
    /// construction.
    ///
    /// # Panics
    /// Panics if `section` is not fully inside this arena.
    pub fn view<T: Zeroable>(&self, section: Section<T>) -> ArenaSlice<T> {
        assert!(
            section.end() <= self.total_bytes,
            "arena section {section:?} out of bounds (arena has {} bytes)",
            self.total_bytes,
        );
        // SAFETY: `offset` is in-bounds and 64-aligned; the base is non-null, so
        // `base + offset` is a non-null, `T`-aligned pointer into the arena. The
        // base is taken from `base_nonnull()`, so the pointer carries the
        // allocation's raw, write-capable provenance and never routes through a
        // `&[u8]` reference reborrow — the view may later write through it.
        let p = unsafe { self.backing.base_nonnull().as_ptr().add(section.offset) as *mut T };
        ArenaSlice {
            ptr: NonNull::new(p).expect("arena base pointer is non-null"),
            len: section.len,
            _marker: PhantomData,
        }
    }

    /// The requested bytes of the backing allocation (for byte-equal cloning).
    pub fn as_bytes(&self) -> &[u8] {
        &self.backing[..self.total_bytes]
    }

    /// A fresh arena of the same size with byte-identical contents in a
    /// distinct allocation.
    pub fn clone_backing(&self) -> Self {
        let mut out = Self::with_capacity(self.total_bytes);
        if self.total_bytes != 0 {
            out.backing[..self.total_bytes].copy_from_slice(&self.backing[..self.total_bytes]);
        }
        out
    }
}

/// A raw-pointer view into a [`LargePageArena`] sub-array, exposing it as a
/// `[T]`. It does not own the backing bytes: the arena it was carved from must
/// outlive it. See the module head.
pub struct ArenaSlice<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: an `ArenaSlice<T>` is a view of `[T]` whose backing is kept alive by
// the same owner; it behaves like `&[T]`/`&mut [T]`, so it is `Send`/`Sync`
// exactly when `T` is. The owning struct enforces that no aliasing `&mut`
// coexists (each table gets a disjoint section).
unsafe impl<T: Send> Send for ArenaSlice<T> {}
unsafe impl<T: Sync> Sync for ArenaSlice<T> {}

impl<T> Deref for ArenaSlice<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` addresses `len` contiguous, initialised (zeroed) `T`
        // inside the still-live arena; the borrow is tied to `&self`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for ArenaSlice<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as `deref`, but `&mut self` guarantees exclusive access to this
        // view; disjoint sections never overlap, so no aliasing `&mut` exists.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: fmt::Debug> fmt::Debug for ArenaSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_are_aligned_disjoint_and_in_bounds() {
        let mut layout = ArenaLayout::new();
        let a = layout.reserve::<i32>(3); // 12 bytes -> padded to 64
        let b = layout.reserve::<i16>(5); // 10 bytes -> padded to 64
        let c = layout.reserve::<i8>(1);
        assert_eq!(a.offset(), 0);
        assert_eq!(b.offset(), 64);
        assert_eq!(c.offset(), 128);
        assert!(b.offset() >= a.end());
        assert!(c.offset() >= b.end());
        assert!(c.end() <= layout.total_bytes());
    }

    #[test]
    fn views_are_64_aligned_zeroed_and_correct_length() {
        let mut layout = ArenaLayout::new();
        let s0 = layout.reserve::<i16>(1000);
        let s1 = layout.reserve::<i32>(7);
        let s2 = layout.reserve::<i8>(40);
        let arena = LargePageArena::new(&layout);

        for (ptr, len) in [
            (arena.view(s0).as_ptr() as usize, 1000),
            (arena.view(s1).as_ptr() as usize, 7),
            (arena.view(s2).as_ptr() as usize, 40),
        ] {
            assert_eq!(ptr % ARENA_SUB_ALIGN, 0, "sub-array base not 64-aligned");
            let _ = len;
        }
        assert_eq!(arena.view(s0).len(), 1000);
        assert!(arena.view(s0).iter().all(|&x| x == 0));
        assert!(arena.view(s1).iter().all(|&x| x == 0));
        assert!(arena.view(s2).iter().all(|&x| x == 0));
    }

    #[test]
    fn slice_mut_fills_in_place_and_view_reads_it_back() {
        let mut layout = ArenaLayout::new();
        let s = layout.reserve::<i32>(64);
        let mut arena = LargePageArena::new(&layout);
        for (i, slot) in arena.slice_mut(s).iter_mut().enumerate() {
            *slot = i as i32 - 20;
        }
        let view = arena.view(s);
        assert_eq!(view[0], -20);
        assert_eq!(view[63], 43);
    }

    #[test]
    fn clone_backing_is_byte_equal_and_distinct() {
        let mut layout = ArenaLayout::new();
        let s = layout.reserve::<i16>(100);
        let mut arena = LargePageArena::new(&layout);
        for (i, slot) in arena.slice_mut(s).iter_mut().enumerate() {
            *slot = (i as i16) - 50;
        }
        let copy = arena.clone_backing();
        assert_eq!(arena.as_bytes(), copy.as_bytes());
        assert_ne!(
            arena.view(s).as_ptr(),
            copy.view(s).as_ptr(),
            "replica must be a distinct allocation",
        );
    }

    #[test]
    fn reserved_bytes_rounds_up_to_large_page() {
        let mut layout = ArenaLayout::new();
        layout.reserve::<u8>(10);
        let arena = LargePageArena::new(&layout);
        assert_eq!(arena.total_bytes(), 10);
        assert_eq!(arena.reserved_bytes(), crate::LARGE_PAGE_ALIGN);
    }
}
