//! 64-byte-aligned heap buffer for NNUE weight/bias data: forces a 64-byte base
//! so the AVX-512 kernels' unaligned 512-bit loads never split a cache line.

use std::alloc::{self, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// Cache-line size assumed by the AVX-512 NNUE kernels.
const ALIGN: usize = 64;

/// A heap-allocated `[T]` whose base pointer is 64-byte aligned (`T: Copy`).
pub struct Aligned64<T: Copy> {
    /// Non-null for every length; length 0 is a dangling, aligned sentinel.
    ptr: NonNull<T>,
    len: usize,
}

// SAFETY: uniquely owns its allocation like `Box<[T]>`, so `Send`/`Sync` when `T` is.
unsafe impl<T: Copy + Send> Send for Aligned64<T> {}
unsafe impl<T: Copy + Sync> Sync for Aligned64<T> {}

impl<T: Copy> Aligned64<T> {
    /// `Layout` for `len` elements at 64-byte alignment.
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len * std::mem::size_of::<T>(), ALIGN)
            .expect("NNUE buffer layout overflow")
    }

    /// Allocates a zeroed, 64-byte-aligned buffer of `len` elements.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            // Use ALIGN as the sentinel so an empty buffer still reports a non-null, 64-aligned base.
            return Self {
                ptr: NonNull::new(ALIGN as *mut T).expect("ALIGN sentinel is non-null"),
                len: 0,
            };
        }
        let layout = Self::layout(len);
        // SAFETY: len > 0 non-ZST layout; alloc_zeroed gives a 64-aligned block of valid zero bytes.
        let raw = unsafe { alloc::alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    /// Copies `src` into a fresh 64-byte-aligned buffer of the same length.
    pub fn from_slice(src: &[T]) -> Self {
        let mut buf = Self::zeroed(src.len());
        // SAFETY: buf holds src.len() elements disjoint from src; T: Copy makes a bytewise copy valid.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), buf.ptr.as_ptr(), src.len()) };
        buf.len = src.len();
        buf
    }
}

impl<T: Copy> Deref for Aligned64<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: ptr addresses len contiguous initialised T (dangling-but-aligned when len == 0).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for Aligned64<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: see `deref`; `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Drop for Aligned64<T> {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: ptr came from alloc_zeroed with this exact layout; T: Copy has no destructor.
            unsafe { alloc::dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.len)) };
        }
    }
}

impl<T: Copy> From<Vec<T>> for Aligned64<T> {
    fn from(v: Vec<T>) -> Self {
        Self::from_slice(&v)
    }
}

impl<T: Copy> FromIterator<T> for Aligned64<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let v: Vec<T> = iter.into_iter().collect();
        Self::from_slice(&v)
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for Aligned64<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_aligned<T: Copy>(buf: &Aligned64<T>) {
        assert_eq!(
            buf.as_ptr() as usize % ALIGN,
            0,
            "base pointer is not 64-byte aligned"
        );
    }

    #[test]
    fn zeroed_is_aligned_and_zero() {
        for len in [1usize, 7, 15, 16, 17, 1_536, 24_576] {
            let buf = Aligned64::<i16>::zeroed(len);
            assert_eq!(buf.len(), len);
            assert_aligned(&buf);
            assert!(buf.iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn from_slice_preserves_bytes_and_aligns() {
        let src: Vec<i8> = (0..200i32).map(|i| (i - 100) as i8).collect();
        let buf = Aligned64::<i8>::from_slice(&src);
        assert_aligned(&buf);
        assert_eq!(&*buf, &src[..]);
    }

    #[test]
    fn from_vec_and_from_iter_match() {
        let v: Vec<i32> = (0..50).map(|i| i * 3 - 7).collect();
        let from_vec = Aligned64::<i32>::from(v.clone());
        let from_iter: Aligned64<i32> = v.iter().copied().collect();
        assert_aligned(&from_vec);
        assert_aligned(&from_iter);
        assert_eq!(&*from_vec, &v[..]);
        assert_eq!(&*from_iter, &v[..]);
    }

    #[test]
    fn empty_is_valid_and_aligned() {
        let buf = Aligned64::<i16>::zeroed(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_aligned(&buf);
        let from_empty: Aligned64<i16> = Vec::<i16>::new().into();
        assert!(from_empty.is_empty());
    }

    #[test]
    fn deref_mut_allows_in_place_fill() {
        let mut buf = Aligned64::<i32>::zeroed(64);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_aligned(&buf);
        assert_eq!(buf[63], 63);
    }
}
