//! Shared huge-page-backed allocator, a port of the reference's
//! `aligned_large_pages_alloc` / `make_unique_large_page`
//! (`source/memory.h`, `memory.cpp`).
//!
//! The reference routes its biggest allocations — the transposition table, the
//! dynamically sized history tables, and the loaded NNUE parameters — through a
//! single large-page allocator so a `madvise(MADV_HUGEPAGE)` hint can back them
//! with transparent huge pages. This module is the port's one copy of that
//! policy, reused by the TT ([`crate::tt`]), the search-layer history tables,
//! and the eval-layer network parameters instead of the logic being duplicated
//! at each site.
//!
//! The policy, matching `aligned_large_pages_alloc`
//! (`memory.cpp`) exactly:
//!
//! * base alignment [`LARGE_PAGE_ALIGN`] — a 2 MiB huge-page boundary on Linux
//!   (so the `MADV_HUGEPAGE` hint can take effect), a 4 KiB page boundary
//!   elsewhere (macOS dev machines / CI, where the huge-page path does not
//!   apply);
//! * the byte size is rounded **up** to a whole multiple of [`LARGE_PAGE_ALIGN`]
//!   (`size = ((allocSize + alignment - 1) / alignment) * alignment`);
//! * the region is zero-initialised (`alloc_zeroed`);
//! * on Linux the whole rounded region is `madvise(MADV_HUGEPAGE)`-hinted, the
//!   call being silent and its return value ignored exactly as the reference
//!   ignores it (the hint is best-effort).
//!
//! Two safe containers sit on top of that raw helper:
//!
//! * [`LargePageArray<T>`] — a large-page-backed `[T]`, the analogue of
//!   `Box<[T]>` (and of `make_unique_large_page<T[]>`);
//! * [`LargePageBox<T>`] — a large-page-backed single `T`, the analogue of
//!   `Box<T>` (and of `make_unique_large_page<T>`), used to keep a fixed-shape
//!   multidimensional array's compile-time dimensions.
//!
//! Both construct their storage zeroed, so the element type must have a valid
//! all-zero bit pattern; that requirement is captured by the [`Zeroable`]
//! marker.

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::fmt;
use std::mem::{align_of, size_of};
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::slice;

/// Base alignment of a large-page allocation, mirroring the reference's
/// `aligned_large_pages_alloc`: a 2 MiB huge-page boundary on Linux so a
/// `MADV_HUGEPAGE` hint can back the region with transparent huge pages, and a
/// 4 KiB page boundary elsewhere, where the huge-page path does not apply.
#[cfg(target_os = "linux")]
pub const LARGE_PAGE_ALIGN: usize = 2 * 1024 * 1024;
/// See the Linux definition; 4 KiB elsewhere.
#[cfg(not(target_os = "linux"))]
pub const LARGE_PAGE_ALIGN: usize = 4096;

/// Types whose all-zero bit pattern is a valid, fully initialised value, so a
/// zeroed allocation is a valid `Self`. Implemented for the fixed-width integers
/// the history / NNUE tables store, for the atomic integer the thread-shared
/// history tables store, and, transitively, for arrays of them.
///
/// # Safety
///
/// An all-zero byte pattern of `size_of::<Self>()` bytes must be a valid value
/// of `Self`, and `Self` must need no drop glue: the containers below free the
/// backing bytes directly and never run element destructors. The plain-integer
/// impls satisfy the latter because they are `Copy`; [`AtomicI16`] satisfies it
/// because it has no `Drop` (an atomic integer is trivially destructible).
pub unsafe trait Zeroable {}

// SAFETY: `0` is a valid value of each of these integer types, and all are
// trivially destructible.
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for u8 {}

// SAFETY: `AtomicI16` has the same layout as `i16`, its all-zero pattern is the
// integer `0` (a valid initialised atomic), and it has no `Drop` glue. It is the
// entry type of the thread-shared correction / pawn history tables, which need
// interior mutability under a shared `&SharedHistories`.
unsafe impl Zeroable for core::sync::atomic::AtomicI16 {}

// SAFETY: an array is valid when every element is, and an all-zero array is an
// array of all-zero (valid) elements; an array needs no drop glue when its
// elements need none.
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}

/// Round `min_bytes` up to a whole multiple of [`LARGE_PAGE_ALIGN`]
/// (`size = ((allocSize + alignment - 1) / alignment) * alignment`).
pub(crate) const fn rounded_size(min_bytes: usize) -> usize {
    min_bytes.div_ceil(LARGE_PAGE_ALIGN) * LARGE_PAGE_ALIGN
}

/// Allocate a zeroed block of at least `min_bytes`, sized up to a whole multiple
/// of [`LARGE_PAGE_ALIGN`] and aligned to [`LARGE_PAGE_ALIGN`], then (on Linux)
/// `madvise(MADV_HUGEPAGE)`-hint the whole rounded region. `min_bytes` must be
/// non-zero. Returns the base pointer and the exact [`Layout`] used — replay
/// that layout to [`free_large`] to free.
///
/// The raw core of [`LargePageArray`], [`LargePageBox`] and the TT's cluster
/// store, so the huge-page policy lives in exactly one place.
pub(crate) fn alloc_zeroed_large(min_bytes: usize) -> (NonNull<u8>, Layout) {
    debug_assert!(min_bytes > 0, "large-page allocation must be non-empty");

    let size = rounded_size(min_bytes);
    let layout = Layout::from_size_align(size, LARGE_PAGE_ALIGN)
        .expect("large-page allocation size/alignment is valid");

    // SAFETY: `layout` has non-zero size (`min_bytes >= 1` ⇒ `size >=
    // LARGE_PAGE_ALIGN`). `alloc_zeroed` returns a `LARGE_PAGE_ALIGN`-aligned,
    // fully zeroed block.
    let raw = unsafe { alloc_zeroed(layout) };
    let ptr = match NonNull::new(raw) {
        Some(p) => p,
        None => handle_alloc_error(layout),
    };

    #[cfg(target_os = "linux")]
    {
        // SAFETY: `ptr`/`size` describe the live allocation just returned.
        // `madvise` only adjusts kernel paging policy for that range; it neither
        // reads nor writes the memory and cannot invalidate it. The return value
        // is ignored on purpose — the hint is best-effort, and the reference
        // discards it identically.
        unsafe {
            libc::madvise(ptr.as_ptr() as *mut libc::c_void, size, libc::MADV_HUGEPAGE);
        }
    }

    (ptr, layout)
}

/// Free a block returned by [`alloc_zeroed_large`].
///
/// # Safety
///
/// `ptr` must have come from [`alloc_zeroed_large`] with exactly `layout`, and
/// must not be freed more than once.
pub(crate) unsafe fn free_large(ptr: NonNull<u8>, layout: Layout) {
    // SAFETY: contract delegated to the caller (see the doc comment).
    unsafe { dealloc(ptr.as_ptr(), layout) }
}

/// A large-page-backed, zero-initialised `[T]` — the analogue of `Box<[T]>`
/// (and of the reference's `make_unique_large_page<T[]>`). The base pointer is
/// [`LARGE_PAGE_ALIGN`]-aligned and, on Linux, the backing region is
/// `MADV_HUGEPAGE`-hinted. The exposed slice covers exactly `len` elements; the
/// round-up tail of the allocation is not part of it.
pub struct LargePageArray<T: Zeroable> {
    /// Base of the aligned allocation. Dangling (never dereferenced) when
    /// `len == 0`; always [`LARGE_PAGE_ALIGN`]-aligned otherwise.
    ptr: NonNull<T>,
    /// Number of live elements the slice exposes.
    len: usize,
    /// Exact [`Layout`] the block was allocated with, replayed to [`free_large`]
    /// on drop. Meaningless (zero-sized) when `len == 0`.
    layout: Layout,
}

// SAFETY: `LargePageArray<T>` uniquely owns its heap block of `T`, freed once in
// `Drop`, just like `Box<[T]>`; so it is `Send`/`Sync` exactly when `T` is.
unsafe impl<T: Zeroable + Send> Send for LargePageArray<T> {}
unsafe impl<T: Zeroable + Sync> Sync for LargePageArray<T> {}

impl<T: Zeroable> LargePageArray<T> {
    /// A zeroed array of `len` elements, its base pointer on a
    /// [`LARGE_PAGE_ALIGN`] boundary. `len == 0` allocates nothing (a dangling,
    /// aligned sentinel).
    pub fn zeroed(len: usize) -> Self {
        debug_assert!(
            size_of::<T>() > 0,
            "LargePageArray element must not be a ZST"
        );
        debug_assert!(align_of::<T>() <= LARGE_PAGE_ALIGN);

        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                // Zero-sized, correctly aligned; never handed to `free_large`.
                layout: Layout::from_size_align(0, LARGE_PAGE_ALIGN)
                    .expect("LARGE_PAGE_ALIGN is a valid power-of-two alignment"),
            };
        }

        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("LargePageArray byte size overflows usize");
        let (raw, layout) = alloc_zeroed_large(bytes);
        // `raw` is `LARGE_PAGE_ALIGN`-aligned, hence aligned for `T`.
        Self {
            ptr: raw.cast(),
            len,
            layout,
        }
    }

    /// The stored base pointer, carrying the allocation's original raw
    /// provenance (write-capable), **not** a `&[T]`/`&mut [T]` reborrow.
    ///
    /// Exists for arena views that carve sub-array pointers out of the backing
    /// allocation and must not route through a reference reborrow: going via
    /// `as_ptr` / `as_mut_ptr` (which resolve through `Deref` to the slice
    /// methods) would stamp the pointer with the provenance of a temporary
    /// shared/exclusive reference, making later writes through it undefined
    /// behaviour under Stacked/Tree Borrows. This returns the raw base instead.
    pub(crate) fn base_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    /// Copy `src` into a fresh large-page-backed array of the same length.
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Copy,
    {
        let buf = Self::zeroed(src.len());
        if !src.is_empty() {
            // SAFETY: `buf` holds `src.len()` elements in one allocation,
            // disjoint from `src`; `T: Copy` makes a bytewise copy valid.
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), buf.ptr.as_ptr(), src.len());
            }
        }
        buf
    }
}

impl<T: Zeroable> Deref for LargePageArray<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: for `len > 0`, `ptr` addresses `len` contiguous, initialised
        // (`alloc_zeroed`) `T` within one allocation. For `len == 0` this yields
        // an empty slice, for which `from_raw_parts` accepts any aligned non-null
        // pointer (`NonNull::dangling` is suitably aligned). The borrow is tied
        // to `&self`, so no `&mut` alias can coexist.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Zeroable> DerefMut for LargePageArray<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as `deref`, but `&mut self` guarantees exclusivity, so handing
        // out `&mut [T]` introduces no aliasing.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Zeroable + fmt::Debug> fmt::Debug for LargePageArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: Zeroable> Drop for LargePageArray<T> {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: `ptr` came from `alloc_zeroed_large` with exactly `layout`
            // (an empty array keeps `len == 0` and is skipped), and it is freed
            // exactly once because `LargePageArray` uniquely owns the block. `T:
            // Zeroable` needs no element drop glue, so freeing the raw bytes is
            // sufficient.
            unsafe {
                free_large(self.ptr.cast(), self.layout);
            }
        }
    }
}

/// A large-page-backed, zero-initialised single `T` — the analogue of `Box<T>`
/// (and of the reference's `make_unique_large_page<T>`). Keeps a fixed-shape
/// array's compile-time dimensions (so per-access index maths stay
/// bounds-check-free) while moving the backing store onto huge pages.
pub struct LargePageBox<T: Zeroable> {
    /// Base of the aligned allocation, holding one initialised `T`.
    ptr: NonNull<T>,
    /// Exact [`Layout`] the block was allocated with, replayed to [`free_large`]
    /// on drop.
    layout: Layout,
}

// SAFETY: `LargePageBox<T>` uniquely owns one heap `T`, freed once in `Drop`,
// just like `Box<T>`; so it is `Send`/`Sync` exactly when `T` is.
unsafe impl<T: Zeroable + Send> Send for LargePageBox<T> {}
unsafe impl<T: Zeroable + Sync> Sync for LargePageBox<T> {}

impl<T: Zeroable> LargePageBox<T> {
    /// A zeroed `T`, its base pointer on a [`LARGE_PAGE_ALIGN`] boundary.
    pub fn zeroed() -> Self {
        debug_assert!(size_of::<T>() > 0, "LargePageBox value must not be a ZST");
        debug_assert!(align_of::<T>() <= LARGE_PAGE_ALIGN);

        let (raw, layout) = alloc_zeroed_large(size_of::<T>());
        Self {
            ptr: raw.cast(),
            layout,
        }
    }
}

impl<T: Zeroable> Deref for LargePageBox<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `ptr` addresses one initialised (`alloc_zeroed`) `T` within a
        // live allocation; the borrow is tied to `&self`, so no `&mut` alias can
        // coexist.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: Zeroable> DerefMut for LargePageBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as `deref`, but `&mut self` guarantees exclusivity.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: Zeroable + fmt::Debug> fmt::Debug for LargePageBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: Zeroable> Drop for LargePageBox<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed_large` with exactly `layout`, and
        // it is freed exactly once because `LargePageBox` uniquely owns the
        // block. `T: Zeroable` needs no drop glue.
        unsafe {
            free_large(self.ptr.cast(), self.layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounded_size_rounds_up_to_alignment() {
        assert_eq!(rounded_size(1), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN - 1), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN + 1), 2 * LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(2 * LARGE_PAGE_ALIGN), 2 * LARGE_PAGE_ALIGN);
    }

    #[test]
    fn array_is_aligned_zeroed_and_correct_length() {
        // A length that does not divide the alignment (exercises the round-up)
        // and one that spans several alignment units.
        for &len in &[1usize, 7, 4096, LARGE_PAGE_ALIGN / 2 + 3, LARGE_PAGE_ALIGN] {
            let buf = LargePageArray::<i16>::zeroed(len);
            assert_eq!(buf.len(), len);
            assert_eq!(
                buf.as_ptr() as usize % LARGE_PAGE_ALIGN,
                0,
                "base pointer not {LARGE_PAGE_ALIGN}-aligned for len {len}",
            );
            assert!(
                buf.iter().all(|&x| x == 0),
                "buffer not zeroed for len {len}"
            );
        }
    }

    #[test]
    fn empty_array_is_valid_and_aligned() {
        let buf = LargePageArray::<i16>::zeroed(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.as_ptr() as usize % align_of::<i16>(), 0);
    }

    #[test]
    fn array_is_mutable_in_place() {
        let mut buf = LargePageArray::<i32>::zeroed(64);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_eq!(buf[63], 63);
        assert_eq!(buf.as_ptr() as usize % LARGE_PAGE_ALIGN, 0);
    }

    #[test]
    fn from_slice_preserves_bytes_and_aligns() {
        let src: Vec<i16> = (0..1000i32).map(|i| (i - 500) as i16).collect();
        let buf = LargePageArray::<i16>::from_slice(&src);
        assert_eq!(buf.as_ptr() as usize % LARGE_PAGE_ALIGN, 0);
        assert_eq!(&*buf, &src[..]);
    }

    #[test]
    fn box_is_aligned_and_zeroed() {
        let boxed = LargePageBox::<[[i16; 4]; 2]>::zeroed();
        assert_eq!(
            (&*boxed as *const _ as usize) % LARGE_PAGE_ALIGN,
            0,
            "box base pointer not {LARGE_PAGE_ALIGN}-aligned",
        );
        assert_eq!(*boxed, [[0i16; 4]; 2]);
    }

    #[test]
    fn box_is_mutable_and_keeps_fixed_dimensions() {
        let mut boxed = LargePageBox::<[[i16; 4]; 2]>::zeroed();
        boxed[1][3] = 42;
        assert_eq!(boxed[1][3], 42);
        assert_eq!(boxed[0][0], 0);
    }
}
