//! Scalar/SIMD kernel selection for the NNUE forward pass.
//!
//! Ported from the read-only Rust NNUE reference implementation's SIMD kernels.
//! Like the reference, the backend is chosen **at compile time** from the CPU
//! features the build itself enables (`cfg(target_feature = ...)`). This repo
//! builds with `-C target-cpu=native` (see `.cargo/config.toml`) and runs each
//! binary on the machine that produced it, so "what the build enables" is the
//! host CPU's feature set; a run-time probe could never pick anything the build
//! had not already fixed. Two feature sets matter:
//!
//! - `avx512f` + `avx512bw` — the feature-transformer accumulate/update kernels
//!   ([`avx512`]) and the element-wise post-FT kernels ([`avx512_post_ft`]).
//! - the above plus `avx512vnni` — the fused fc_0/fc_1/fc_2 chain
//!   ([`avx512_post_ft::fused_fc_chain`], called from [`crate::network`]).
//!
//! Both backends stay **compiled unconditionally** (only the call sites are
//! `cfg`-gated) so the SIMD == scalar bit-equality tests in each backend module
//! keep exercising both. Those tests, and only those, still probe the CPU at run
//! time with `is_x86_feature_detected!` so they skip gracefully on a host
//! without the features. The kernels are exact, not approximate: whichever the
//! build selects, the evaluation output is bit-identical.
//!
//! ## Safety invariants
//!
//! Every AVX-512 entry point is an `unsafe fn` carrying a
//! `#[target_feature(enable = ...)]` attribute: calling it is sound only when
//! the named features are present on the running CPU. The wrappers in this
//! module are the sole callers in non-test code, and each `unsafe` call sits
//! behind a `cfg(target_feature = ...)` gate naming exactly the features its
//! callee enables — the code is compiled only into a build that already
//! requires those features of its host, so no `unsafe` escapes the SIMD modules.

use crate::features::FeatureIndex;

// `allow(dead_code)`: both backends are compiled unconditionally so the
// equivalence tests can compare them, which leaves whichever one this build did
// not select without a caller outside `cfg(test)`.
#[allow(dead_code)]
pub mod scalar;
#[allow(dead_code)]
pub mod scalar_post_ft;

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub mod avx512_post_ft;

/// Which kernel backend this build compiled into the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar baseline (always correct, always available).
    Scalar,
    /// AVX-512 F + BW + VNNI kernels (the full SIMD forward pass).
    Avx512Vnni,
}

/// The kernel backend baked into this build.
///
/// [`Backend::Avx512Vnni`] means the full forward pass (accumulator, output
/// transform, and layer stack) runs on AVX-512; [`Backend::Scalar`] means the
/// portable path — including the partial case of an `avx512f`+`avx512bw` build
/// without VNNI, where the feature transformer is SIMD but the layer stack is
/// not. The eval-parity gate uses this to assert that a build produced on an
/// AVX-512-VNNI host really did select the SIMD path (i.e. that
/// `-C target-cpu=native` was in effect) rather than silently compiling scalar.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
pub const fn active_backend() -> Backend {
    Backend::Avx512Vnni
}

/// The kernel backend baked into this build — non-VNNI arm (the VNNI arm above
/// carries the documentation).
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
)))]
pub const fn active_backend() -> Backend {
    Backend::Scalar
}

/// Feature-transformer accumulate/update kernels, selected at compile time.
///
/// The AVX-512 arm is compiled only when the build enables `avx512f` +
/// `avx512bw` (exactly the features [`super::avx512`]'s kernels declare); every
/// other build gets the scalar baseline. These two `cfg` arms are the only
/// place that condition is spelled out for the FT family.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub mod transformer_kernel {
    use super::FeatureIndex;
    use super::avx512;

    /// Add each active feature's FT weight column into `out`.
    #[inline]
    pub fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: this module is compiled only into a build that enables
        // avx512f + avx512bw — exactly the features the callee's
        // `#[target_feature]` attribute names — and such a build only ever runs
        // on a host providing them (`-C target-cpu=native`; build and run happen
        // on the same machine).
        unsafe { avx512::add_features(out, weights, indices) }
    }

    /// Subtract each feature's FT weight column from `out`.
    #[inline]
    pub fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: see `add_features`.
        unsafe { avx512::sub_features(out, weights, indices) }
    }

    /// Fused single-add / single-sub delta.
    #[inline]
    pub fn add_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx512::add_sub_features(out, weights, added, removed) }
    }

    /// Fused single-add / double-sub delta (capture-style updates).
    #[inline]
    pub fn add_sub_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed_a: &[FeatureIndex],
        removed_b: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx512::add_sub_sub_features(out, weights, added, removed_a, removed_b) }
    }
}

/// Feature-transformer accumulate/update kernels — scalar arm (the AVX-512 arm
/// above carries the selection rule).
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
pub mod transformer_kernel {
    pub use super::scalar::{add_features, add_sub_features, add_sub_sub_features, sub_features};
}

/// Output-transform and layer element-wise kernels, selected at compile time on
/// the same `avx512f` + `avx512bw` condition as [`transformer_kernel`]. The
/// affine and whole-chain forward paths live in [`crate::network`], which calls
/// [`avx512_post_ft::fused_fc_chain`] directly on a VNNI-enabled build.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub mod post_ft_kernel {
    use super::{avx512_post_ft, scalar_post_ft};

    /// Pairwise element-wise multiply for one perspective half.
    #[inline]
    pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
        // SAFETY: this module is compiled only into a build that enables
        // avx512f + avx512bw — exactly the features the callee's
        // `#[target_feature]` attribute names — and such a build only ever runs
        // on a host providing them.
        unsafe { avx512_post_ft::ewm_one_perspective(half, out) }
    }

    /// Clipped ReLU.
    #[inline]
    pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { avx512_post_ft::clipped_relu(input, output) }
    }

    /// Squared clipped ReLU.
    #[inline]
    pub fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { avx512_post_ft::sqr_clipped_relu(input, output) }
    }

    /// Integer affine transform. The AVX-512 form needs VNNI and is only worth
    /// its setup at the wide fc_0 shape, where the whole chain runs through
    /// [`avx512_post_ft::fused_fc_chain`]; this per-layer entry point therefore
    /// always uses the scalar kernel.
    #[inline]
    pub fn affine(
        output: &mut [i32],
        biases: &[i32],
        weights: &[i8],
        input: &[u8],
        in_dims: usize,
        padded_in: usize,
    ) {
        scalar_post_ft::affine(output, biases, weights, input, in_dims, padded_in);
    }
}

/// Output-transform and layer element-wise kernels — scalar arm (the AVX-512 arm
/// above carries the selection rule).
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
pub mod post_ft_kernel {
    pub use super::scalar_post_ft::{affine, clipped_relu, ewm_one_perspective, sqr_clipped_relu};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_backend_is_avx512_when_cpu_supports_vnni() {
        // Backend selection is compile-time, so this checks the *build* was
        // configured for its host: with `-C target-cpu=native` (this repo's
        // standing setting) a VNNI-capable CPU must yield a VNNI build, and
        // hence the SIMD backend. A scalar backend here means the build did not
        // enable the host's features — the silent-fallback case worth catching.
        // Runtime detection survives here because this is test code.
        #[cfg(target_arch = "x86_64")]
        {
            let has_vnni = std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni");
            if has_vnni {
                assert_eq!(
                    active_backend(),
                    Backend::Avx512Vnni,
                    "CPU reports AVX-512 VNNI but this build compiled the scalar \
                     backend — check that `-C target-cpu=native` is in effect",
                );
            } else {
                assert_eq!(active_backend(), Backend::Scalar);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(active_backend(), Backend::Scalar);
    }
}
