//! Shared huge-page-backed allocator, a port of `aligned_large_pages_alloc` /
//! `make_unique_large_page`. The transposition table, the search-layer history
//! tables and the eval-layer network parameters all allocate through it, as
//! `Box<[T], LargePageAlloc>` and `Box<T, LargePageAlloc>`.
//!
//! The policy: base alignment [`LARGE_PAGE_ALIGN`], the byte size rounded **up**
//! to a whole multiple of it, and zero-initialised storage.
//!
//! On Linux the block is a fresh anonymous private mapping with a
//! `madvise(MADV_HUGEPAGE)` hint over the whole rounded region, and allocation
//! writes none of it. The order matters: a page written before the hint is
//! faulted in as a 4 KiB page, and a page lands on the NUMA node of the thread
//! that first writes it. Leaving the first write to the owner lets the
//! transposition table's multi-threaded clear decide both. The hint's return
//! value is ignored, the hint being best-effort, as the reference ignores it.
//! A fresh anonymous mapping reads as zero, which is what makes the storage
//! zero-initialised without a write.
//!
//! Elsewhere, and under miri (which cannot `mmap` or `madvise`), the block comes
//! from the global allocator's `alloc_zeroed`.
//!
//! [`zeroed_slice`] and [`zeroed_box`] turn the allocator's zeroed block into a
//! `Box` without writing it, so the element type must have a valid all-zero bit
//! pattern; that is what [`Zeroable`] captures. Filling a `Vec` instead would
//! write every page at construction.

use std::alloc::{AllocError, Allocator, Layout, handle_alloc_error};
#[cfg(not(all(target_os = "linux", not(miri))))]
use std::alloc::{alloc_zeroed, dealloc};
#[cfg(all(target_os = "linux", not(miri)))]
use std::ptr;
use std::ptr::NonNull;

/// Base alignment of a large-page allocation: a 2 MiB huge-page boundary on
/// Linux, so that a `MADV_HUGEPAGE` hint can take effect, and a 4 KiB page
/// boundary elsewhere, where the huge-page path does not apply.
#[cfg(target_os = "linux")]
pub const LARGE_PAGE_ALIGN: usize = 2 * 1024 * 1024;
/// See the Linux definition; 4 KiB elsewhere.
#[cfg(not(target_os = "linux"))]
pub const LARGE_PAGE_ALIGN: usize = 4096;

/// Types whose all-zero bit pattern is a valid, fully initialised value, so
/// that a zeroed allocation is a valid `Self`.
///
/// # Safety
///
/// An all-zero byte pattern of `size_of::<Self>()` bytes must be a valid value
/// of `Self`.
pub unsafe trait Zeroable {}

// SAFETY: `0` is a valid value of each of these integer types.
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for u8 {}

// SAFETY: `AtomicI16` has the same layout as `i16`, and its all-zero pattern is
// the integer `0`, a valid initialised atomic.
unsafe impl Zeroable for core::sync::atomic::AtomicI16 {}

// SAFETY: an array is valid when every element is, and an all-zero array is an
// array of all-zero (valid) elements.
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}

/// Round `min_bytes` up to a whole multiple of [`LARGE_PAGE_ALIGN`]
/// (`size = ((allocSize + alignment - 1) / alignment) * alignment`).
pub(crate) const fn rounded_size(min_bytes: usize) -> usize {
    min_bytes.div_ceil(LARGE_PAGE_ALIGN) * LARGE_PAGE_ALIGN
}

/// Map `layout.size()` bytes on a `layout.align()` boundary: reserve the size
/// plus one alignment unit with no access, map the read-write region over the
/// first aligned address inside the reservation, then unmap the slack on both
/// sides. A plain `mmap` guarantees only page alignment, and a huge page can
/// back only a range aligned to its own size.
#[cfg(all(target_os = "linux", not(miri)))]
fn alloc_zeroed_block(layout: Layout) -> Option<NonNull<u8>> {
    use libc::{
        MAP_ANONYMOUS, MAP_FAILED, MAP_FIXED, MAP_PRIVATE, PROT_NONE, PROT_READ, PROT_WRITE,
        c_void, madvise, mmap, munmap,
    };

    let size = layout.size();
    let align = layout.align();
    let reservation_size = size.checked_add(align)?;

    // SAFETY: an anonymous mapping at a kernel-chosen address aliases nothing.
    let reservation = unsafe {
        mmap(
            ptr::null_mut(),
            reservation_size,
            PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if reservation == MAP_FAILED {
        return None;
    }

    let prefix = (reservation as usize).next_multiple_of(align) - reservation as usize;
    let suffix = reservation_size - prefix - size;
    // SAFETY: `prefix < align`, so the aligned address and the `size` bytes after
    // it lie inside the reservation.
    let aligned = unsafe { reservation.cast::<u8>().add(prefix) }.cast::<c_void>();

    // SAFETY: `MAP_FIXED` replaces part of the reservation this function owns,
    // and nothing else.
    let mapped = unsafe {
        mmap(
            aligned,
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
            -1,
            0,
        )
    };
    if mapped == MAP_FAILED {
        // SAFETY: the whole reservation is still this function's own mapping.
        unsafe { munmap(reservation, reservation_size) };
        return None;
    }

    // SAFETY: the prefix and the suffix are the unused ends of the reservation,
    // disjoint from the `size` bytes at `mapped`. `madvise` only adjusts the
    // kernel's paging policy for that range; it neither reads nor writes it.
    unsafe {
        if prefix != 0 {
            munmap(reservation, prefix);
        }
        if suffix != 0 {
            munmap(mapped.cast::<u8>().add(size).cast(), suffix);
        }
        madvise(mapped, size, libc::MADV_HUGEPAGE);
    }

    NonNull::new(mapped.cast())
}

#[cfg(not(all(target_os = "linux", not(miri))))]
fn alloc_zeroed_block(layout: Layout) -> Option<NonNull<u8>> {
    // SAFETY: `layout` has non-zero size: the allocator maps only non-empty
    // requests, each rounded up to at least `LARGE_PAGE_ALIGN`.
    NonNull::new(unsafe { alloc_zeroed(layout) })
}

/// Free a block that [`alloc_zeroed_block`] returned for `layout`.
///
/// # Safety
///
/// `ptr` must have come from [`alloc_zeroed_block`] with exactly `layout`, and
/// must not be freed more than once.
unsafe fn free_block(ptr: NonNull<u8>, layout: Layout) {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        // SAFETY: by the caller's contract the `layout.size()` bytes at `ptr`
        // are exactly one live mapping made by `alloc_zeroed_block`.
        let rc = unsafe { libc::munmap(ptr.as_ptr().cast(), layout.size()) };
        assert!(
            rc == 0,
            "munmap failed: {}",
            std::io::Error::last_os_error()
        );
    }
    #[cfg(not(all(target_os = "linux", not(miri))))]
    {
        // SAFETY: contract delegated to the caller (see the doc comment).
        unsafe { dealloc(ptr.as_ptr(), layout) }
    }
}

/// The large-page allocator: every non-empty block is [`LARGE_PAGE_ALIGN`]-
/// aligned, rounded up to a whole multiple of it, zeroed, and on Linux advised
/// for huge pages, as the module head describes. A request for a stricter
/// alignment than [`LARGE_PAGE_ALIGN`] fails.
#[derive(Clone, Copy, Debug, Default)]
pub struct LargePageAlloc;

impl LargePageAlloc {
    /// The block actually mapped for a non-empty request of `layout`.
    ///
    /// Every size from a request's own up to its rounded size maps to the same
    /// block. That is what lets `deallocate`, which receives the caller's layout
    /// rather than the rounded one, free the block `allocate` made.
    fn block_layout(layout: Layout) -> Result<Layout, AllocError> {
        if layout.align() > LARGE_PAGE_ALIGN {
            return Err(AllocError);
        }
        let size = layout
            .size()
            .checked_next_multiple_of(LARGE_PAGE_ALIGN)
            .ok_or(AllocError)?;
        Layout::from_size_align(size, LARGE_PAGE_ALIGN).map_err(|_| AllocError)
    }
}

// SAFETY: a non-empty block is a fresh mapping (or global-allocator block) of
// `block_layout(layout)`, aligned at least as strictly as `layout` asks and as
// long as the returned slice says. `deallocate` frees that same block: any size
// the caller may hand back, from its request up to the returned length, rounds
// to the same block layout. An empty request gets a dangling, suitably aligned
// pointer, which `deallocate` ignores. The allocator is a unit value, so any
// copy frees what another allocated.
unsafe impl Allocator for LargePageAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate_zeroed(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }
        let block = Self::block_layout(layout)?;
        let ptr = alloc_zeroed_block(block).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(ptr, block.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        let block = Self::block_layout(layout).expect("a layout that allocated maps again");
        // SAFETY: by the `Allocator` contract `ptr` is a live block this
        // allocator returned for a layout that `layout` fits, and `block` is
        // the layout it was mapped with.
        unsafe { free_block(ptr, block) }
    }
}

/// A zeroed, large-page-backed `[T]` of `len` elements, the analogue of
/// `make_unique_large_page<T[]>`. The slice covers exactly `len` elements; the
/// round-up tail of the block is not part of it. No byte of the block is
/// written.
pub fn zeroed_slice<T: Zeroable>(len: usize) -> Box<[T], LargePageAlloc> {
    let layout = Layout::array::<T>(len).expect("large-page array size overflows");
    let block = LargePageAlloc
        .allocate_zeroed(layout)
        .unwrap_or_else(|_| handle_alloc_error(layout));
    let ptr = NonNull::slice_from_raw_parts(block.cast::<T>(), len);
    // SAFETY: `block` is a live `LargePageAlloc` allocation for `layout`, the
    // layout of `len` elements of `T`, which is the layout the `Box` frees it
    // with. Its bytes are zero, and all-zero is a valid `T` by `T: Zeroable`.
    unsafe { Box::from_non_null_in(ptr, LargePageAlloc) }
}

/// A zeroed, large-page-backed `T`, the analogue of `make_unique_large_page<T>`.
/// Keeps a fixed-shape array's compile-time dimensions while moving its backing
/// store onto huge pages. No byte of the block is written.
pub fn zeroed_box<T: Zeroable>() -> Box<T, LargePageAlloc> {
    let layout = Layout::new::<T>();
    let block = LargePageAlloc
        .allocate_zeroed(layout)
        .unwrap_or_else(|_| handle_alloc_error(layout));
    // SAFETY: as in `zeroed_slice`, for the layout of one `T`.
    unsafe { Box::from_non_null_in(block.cast::<T>(), LargePageAlloc) }
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
    fn every_size_a_caller_may_free_with_maps_to_the_allocated_block() {
        for request in [1usize, 7, 4096, LARGE_PAGE_ALIGN - 1, LARGE_PAGE_ALIGN + 1] {
            for align in [1usize, 2, 64, LARGE_PAGE_ALIGN] {
                let layout = Layout::from_size_align(request, align).unwrap();
                let block = LargePageAlloc::block_layout(layout).unwrap();
                assert_eq!(block.size(), rounded_size(request));
                assert_eq!(block.align(), LARGE_PAGE_ALIGN);
                for size in [request, request + 1, block.size() - 1, block.size()] {
                    let fit = Layout::from_size_align(size, align).unwrap();
                    assert_eq!(LargePageAlloc::block_layout(fit), Ok(block));
                }
            }
        }
    }

    #[test]
    fn a_stricter_alignment_is_refused() {
        let layout = Layout::from_size_align(1, 2 * LARGE_PAGE_ALIGN).unwrap();
        assert_eq!(LargePageAlloc.allocate(layout), Err(AllocError));
    }

    #[test]
    fn a_block_freed_with_any_fitting_size_round_trips() {
        let request = Layout::from_size_align(LARGE_PAGE_ALIGN + 3, 8).unwrap();
        let block = LargePageAlloc.allocate(request).unwrap();
        assert_eq!(block.len(), 2 * LARGE_PAGE_ALIGN);
        assert_eq!(block.cast::<u8>().as_ptr() as usize % LARGE_PAGE_ALIGN, 0);
        let fit = Layout::from_size_align(block.len() - 1, 8).unwrap();
        // SAFETY: `block` came from `LargePageAlloc` for `request`, and `fit`
        // lies between the requested and the returned size.
        unsafe { LargePageAlloc.deallocate(block.cast(), fit) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn slice_is_aligned_zeroed_and_correct_length() {
        // A length that does not divide the alignment, exercising the round-up,
        // and one that spans several alignment units.
        for &len in &[1usize, 7, 4096, LARGE_PAGE_ALIGN / 2 + 3, LARGE_PAGE_ALIGN] {
            let buf = zeroed_slice::<i16>(len);
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
    fn empty_slice_is_valid_and_aligned() {
        let buf = zeroed_slice::<i16>(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.as_ptr() as usize % align_of::<i16>(), 0);
    }

    #[test]
    fn slice_is_mutable_in_place() {
        let mut buf = zeroed_slice::<i32>(64);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_eq!(buf[63], 63);
        assert_eq!(buf.as_ptr() as usize % LARGE_PAGE_ALIGN, 0);
    }

    #[test]
    fn copied_slice_preserves_bytes_and_aligns() {
        let src = Vec::from_fn(1000, |i| i as i16 - 500);
        let mut buf = zeroed_slice::<i16>(src.len());
        buf.copy_from_slice(&src);
        assert_eq!(buf.as_ptr() as usize % LARGE_PAGE_ALIGN, 0);
        assert_eq!(&*buf, &src[..]);
    }

    #[test]
    fn box_is_aligned_and_zeroed() {
        let boxed = zeroed_box::<[[i16; 4]; 2]>();
        assert_eq!(
            (&*boxed as *const _ as usize) % LARGE_PAGE_ALIGN,
            0,
            "box base pointer not {LARGE_PAGE_ALIGN}-aligned",
        );
        assert_eq!(*boxed, [[0i16; 4]; 2]);
    }

    #[test]
    fn box_is_mutable_and_keeps_fixed_dimensions() {
        let mut boxed = zeroed_box::<[[i16; 4]; 2]>();
        boxed[1][3] = 42;
        assert_eq!(boxed[1][3], 42);
        assert_eq!(boxed[0][0], 0);
    }
}
