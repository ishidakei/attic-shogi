//! AVX-512 (F + BW) feature-transformer accumulate/update kernels, each
//! bit-identical to its [`crate::simd::scalar`] counterpart.
//!
//! Unlike the reference, which cfg-gates the whole module on the target
//! features, this one is *compiled* on every `x86_64` target so that the parity
//! tests below can hold it against the scalar baseline on any AVX-512-capable
//! host. The `#[target_feature]` attribute on each entry point imposes no
//! whole-binary requirement; it only makes the compiler emit AVX-512 code for
//! that one function, which the linker drops when nothing calls it.

use std::arch::x86_64::{
    __m512i, _mm512_add_epi16, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_sub_epi16,
};

use crate::features::FeatureIndex;
use crate::types::HIDDEN_SIZE;

const NUM_CHUNKS: usize = HIDDEN_SIZE / 32;
const LANES: usize = 32;

/// # Safety
/// The running CPU must support `avx512f` and `avx512bw`. `out` and every
/// referenced weight column must be `HIDDEN_SIZE` long.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let base = idx as usize * HIDDEN_SIZE;
        let col = &weights[base..base + HIDDEN_SIZE];
        let col_ptr = col.as_ptr();
        for chunk in 0..NUM_CHUNKS {
            let offset = chunk * LANES;
            // SAFETY: `chunk*LANES+LANES <= HIDDEN_SIZE = out.len() = col.len()`,
            // and the unaligned 512-bit ops impose no alignment requirement.
            unsafe {
                let o = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
                let w = _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>());
                _mm512_storeu_si512(
                    out_ptr.add(offset).cast::<__m512i>(),
                    _mm512_add_epi16(o, w),
                );
            }
        }
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let base = idx as usize * HIDDEN_SIZE;
        let col = &weights[base..base + HIDDEN_SIZE];
        let col_ptr = col.as_ptr();
        for chunk in 0..NUM_CHUNKS {
            let offset = chunk * LANES;
            // SAFETY: see `add_features`.
            unsafe {
                let o = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
                let w = _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>());
                _mm512_storeu_si512(
                    out_ptr.add(offset).cast::<__m512i>(),
                    _mm512_sub_epi16(o, w),
                );
            }
        }
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn add_sub_features(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE = out.len()`.
        let mut acc = unsafe { _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>()) };
        for &idx in added {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: `col` is `HIDDEN_SIZE`-long, same offset bound.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_add_epi16(acc, w);
        }
        for &idx in removed {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_sub_epi16(acc, w);
        }
        // SAFETY: same offset bound on `out`.
        unsafe { _mm512_storeu_si512(out_ptr.add(offset).cast::<__m512i>(), acc) };
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn add_sub_sub_features(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed_a: &[FeatureIndex],
    removed_b: &[FeatureIndex],
) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE = out.len()`.
        let mut acc = unsafe { _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>()) };
        for &idx in added {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: `col` is `HIDDEN_SIZE`-long, same offset bound.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_add_epi16(acc, w);
        }
        for &idx in removed_a {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_sub_epi16(acc, w);
        }
        for &idx in removed_b {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_sub_epi16(acc, w);
        }
        // SAFETY: same offset bound on `out`.
        unsafe { _mm512_storeu_si512(out_ptr.add(offset).cast::<__m512i>(), acc) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;

    /// Skip the body unless the CPU has the features these kernels need, so
    /// that the parity checks are skipped rather than failed elsewhere.
    macro_rules! require_avx512bw {
        () => {
            if !(std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw"))
            {
                eprintln!("skipping AVX-512 parity test: avx512f/avx512bw unavailable");
                return;
            }
        };
    }

    fn fill_weights(weights: &mut [i16], seed: i16) {
        for (i, slot) in weights.iter_mut().enumerate() {
            *slot = ((i as i32 * 7 + seed as i32) % 29 - 14) as i16;
        }
    }

    fn seeded_initial(seed: i16) -> [i16; HIDDEN_SIZE] {
        let mut out = [0i16; HIDDEN_SIZE];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = ((i as i32 * 13 + seed as i32) % 41 - 20) as i16;
        }
        out
    }

    #[test]
    fn add_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 17);
        let indices: [FeatureIndex; 4] = [0, 2, 5, 7];
        let initial = seeded_initial(3);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_features(&mut avx, &weights, &indices) };

        let mut sca = initial;
        scalar::add_features(&mut sca, &weights, &indices);

        assert_eq!(avx, sca);
    }

    #[test]
    fn sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 29);
        let indices: [FeatureIndex; 3] = [1, 3, 6];
        let initial = seeded_initial(13);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { sub_features(&mut avx, &weights, &indices) };

        let mut sca = initial;
        scalar::sub_features(&mut sca, &weights, &indices);

        assert_eq!(avx, sca);
    }

    #[test]
    fn add_sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 23);
        let added: [FeatureIndex; 1] = [3];
        let removed: [FeatureIndex; 1] = [6];
        let initial = seeded_initial(7);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_sub_features(&mut avx, &weights, &added, &removed) };

        let mut sca = initial;
        scalar::add_sub_features(&mut sca, &weights, &added, &removed);

        assert_eq!(avx, sca);
    }

    #[test]
    fn add_sub_sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 31);
        let added: [FeatureIndex; 1] = [2];
        let removed_a: [FeatureIndex; 1] = [5];
        let removed_b: [FeatureIndex; 1] = [7];
        let initial = seeded_initial(19);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_sub_sub_features(&mut avx, &weights, &added, &removed_a, &removed_b) };

        let mut sca = initial;
        scalar::add_sub_sub_features(&mut sca, &weights, &added, &removed_a, &removed_b);

        assert_eq!(avx, sca);
    }
}
