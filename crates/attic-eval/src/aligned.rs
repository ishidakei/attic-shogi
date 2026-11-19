//! 64-byte-aligning allocator for NNUE weight/bias data, so that the AVX-512
//! kernels' unaligned 512-bit loads never split a cache line.

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;

/// Cache-line size assumed by the AVX-512 NNUE kernels.
const ALIGN: usize = 64;

/// The global allocator with every block's alignment raised to at least 64
/// bytes. A non-empty `Box<[T], Align64Alloc>` is a `Box<[T]>` whose base
/// pointer is cache-line aligned; an empty one allocates nothing, and its
/// dangling base is aligned for `T` only.
#[derive(Clone, Copy, Debug, Default)]
pub struct Align64Alloc;

/// `layout` with its alignment raised to [`ALIGN`]; the size is unchanged, so
/// every call to the global allocator sees one and the same layout for a block.
fn raised(layout: Layout) -> Result<Layout, AllocError> {
    layout.align_to(ALIGN).map_err(|_| AllocError)
}

// SAFETY: every method forwards to `Global` with the same raised layout, so a
// block is freed by `Global` with exactly the layout it was allocated with, and
// the returned block meets the caller's (weaker) alignment as well. A zero-size
// request gets `Global`'s dangling, now 64-aligned, pointer, which `Global`'s
// `deallocate` accepts.
unsafe impl Allocator for Align64Alloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(raised(layout)?)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate_zeroed(raised(layout)?)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let layout = raised(layout).expect("a layout that allocated raises again");
        // SAFETY: `ptr` came from `allocate`/`allocate_zeroed` above, which got
        // it from `Global` with this same raised layout.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

/// A 64-byte-aligned `[T]` holding a copy of `src`.
#[cfg(test)]
pub(crate) fn from_slice<T: Copy>(src: &[T]) -> Box<[T], Align64Alloc> {
    let mut v = Vec::with_capacity_in(src.len(), Align64Alloc);
    v.extend_from_slice(src);
    v.into_boxed_slice()
}

/// A 64-byte-aligned `[T]` of `len` default values.
pub(crate) fn defaulted<T: Copy + Default>(len: usize) -> Box<[T], Align64Alloc> {
    let mut v = Vec::with_capacity_in(len, Align64Alloc);
    v.resize(len, T::default());
    v.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_aligned<T>(buf: &[T]) {
        assert_eq!(
            buf.as_ptr() as usize % ALIGN,
            0,
            "base pointer is not 64-byte aligned"
        );
    }

    #[test]
    fn defaulted_is_aligned_and_zero() {
        for len in [1usize, 7, 15, 16, 17, 1_536, 24_576] {
            let buf = defaulted::<i16>(len);
            assert_eq!(buf.len(), len);
            assert_aligned(&buf);
            assert!(buf.iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn from_slice_preserves_bytes_and_aligns() {
        let src = Vec::from_fn(200, |i| (i as i32 - 100) as i8);
        let buf = from_slice(&src);
        assert_aligned(&buf);
        assert_eq!(&*buf, &src[..]);
    }

    #[test]
    fn empty_is_valid() {
        assert!(defaulted::<i16>(0).is_empty());
        assert!(from_slice::<i16>(&[]).is_empty());
    }

    #[test]
    fn deref_mut_allows_in_place_fill() {
        let mut buf = defaulted::<i32>(64);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_aligned(&buf);
        assert_eq!(buf[63], 63);
    }

    #[test]
    fn a_grown_vec_stays_aligned() {
        let mut v = Vec::new_in(Align64Alloc);
        for i in 0..1000u16 {
            v.push(i);
            assert_aligned(&v);
        }
        v.shrink_to_fit();
        assert_aligned(&v);
    }
}
