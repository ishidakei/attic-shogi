//! Finny tables: a per-worker cache of refreshed accumulator halves, keyed by
//! the perspective's own-king square.
//!
//! A `HalfKA_hm2` feature index embeds the perspective's own-king square, so a
//! perspective whose own king moved cannot be updated differentially: its half
//! is rebuilt from the FT biases plus all [`MAX_ACTIVE_FEATURES`] active
//! columns, an order of magnitude more work than a one-piece diff. King moves
//! are frequent in a shogi tree, and within one search the king revisits the
//! same squares over and over — so keeping one *refreshed* accumulator per
//! (perspective, king square), with the active-feature list it was built from,
//! turns most rebuilds into a diff of a handful of columns.
//!
//! Upstream added the same structure (`FinnyEntry` / `FinnyCache`) but ships it
//! dormant: `USE_FINNY_TABLES` is defined by no Makefile or config header, so a
//! default build never compiles it. This module is an adaptation of code the
//! reference does not run.
//!
//! ## The invariant
//!
//! For every initialised entry:
//!
//! ```text
//! entry.accumulation == ft_biases + sum over entry.active of ft_weights[column]
//! ```
//!
//! That mentions no position, only a feature multiset, which is why an entry
//! stays valid across nodes, searches and whole games. The one thing that can
//! invalidate it is the weights changing underneath, so the cache carries a
//! network-identity token and resets itself when the token moves.
//!
//! Diffing the entry's list against a new one preserves the invariant and
//! re-establishes the accumulator identity **exactly**: the accumulator is a sum
//! of `i16` columns under wrapping arithmetic, so any decomposition of the same
//! multiset of adds and subs is bit-identical. The cached rebuild is
//! value-invariant against a from-scratch refresh, not merely close.
//!
//! One cache per search worker, allocated at worker setup and shared with
//! nobody, so Lazy SMP needs no locking. Its entry table lives on large pages.

use attic_state::{Color, Position, Square};
use attic_storage::large_page::zeroed_box;
use attic_storage::{LargePageAlloc, Zeroable};

use crate::features::{
    DiffScratch, FeatureIndex, MAX_ACTIVE_FEATURES, active_features_into, changed_indices_into,
};
use crate::transformer::{apply_diff, refresh_perspective};
use crate::types::{HIDDEN_SIZE, NnueNetwork};

/// One cached refreshed half: the accumulation and the active-feature list it
/// was built from.
///
/// The 64-byte alignment keeps every entry's accumulation on a cache-line
/// boundary, so the AVX-512 kernels' 512-bit loads never split a line.
#[derive(Debug)]
#[repr(C, align(64))]
struct FinnyEntry {
    /// `ft_biases + sum(ft_weights[c] for c in active)`, valid only while
    /// [`Self::initialized`] is set.
    accumulation: [i16; HIDDEN_SIZE],
    /// The active-feature multiset [`Self::accumulation`] corresponds to. Every
    /// list is padded to exactly [`MAX_ACTIVE_FEATURES`] entries.
    active: [FeatureIndex; MAX_ACTIVE_FEATURES],
    /// Whether [`Self::accumulation`] currently satisfies the module invariant.
    initialized: bool,
}

// SAFETY: every field is an integer array or a `bool`, each valid all-zero
// (a zeroed entry is an uninitialised one), and none has drop glue.
unsafe impl Zeroable for FinnyEntry {}

/// `[perspective][own king square]`, over the raw 81 squares.
type FinnyTable = [[FinnyEntry; Square::COUNT]; Color::COUNT];

/// A worker-private finny table: one [`FinnyEntry`] per (perspective, own-king
/// square).
///
/// Build one per search worker with [`FinnyCache::new`] and hand it to
/// [`crate::Accumulator::derive_into_cached`] on every `do_move`.
#[derive(Debug)]
pub struct FinnyCache {
    /// `[perspective][own king square]`, over the raw 81 squares. The bucket key
    /// must be the untouched king square rather than the mirrored `sq_k_code`:
    /// two mirror-equivalent squares generate different index sets.
    entries: Box<FinnyTable, LargePageAlloc>,
    /// Identity of the network the entries were built against; see
    /// [`network_token`]. `None` until the first rebuild.
    token: Option<NetworkToken>,
    /// Reusable buffer for the post-move active-feature list.
    scratch_active: Vec<FeatureIndex>,
    /// Reusable buffers for the entry-vs-position feature diff.
    diff: DiffScratch,
}

/// Identity of the loaded network, as seen by the cache.
///
/// Both the `NnueNetwork` address and its FT weight-block base are recorded,
/// because a freed network's address can be reused by a later allocation. For
/// both to coincide, the new network's much larger parameter arena would have
/// to land on the old base as well.
type NetworkToken = (usize, usize);

fn network_token(net: &NnueNetwork) -> NetworkToken {
    (
        net as *const NnueNetwork as usize,
        net.ft_weights.as_ptr() as usize,
    )
}

impl FinnyCache {
    /// Allocates an empty cache. The entry table is ~0.5 MiB, rounded up to a
    /// whole large page, so this belongs at worker setup, not on the search
    /// path.
    pub fn new() -> Self {
        FinnyCache {
            entries: zeroed_box(),
            token: None,
            scratch_active: Vec::with_capacity(MAX_ACTIVE_FEATURES),
            diff: DiffScratch::default(),
        }
    }

    /// Drops every cached half. The entries keep their allocations and are
    /// rebuilt lazily.
    fn invalidate(&mut self) {
        for per_color in self.entries.iter_mut() {
            for entry in per_color.iter_mut() {
                entry.initialized = false;
            }
        }
    }

    /// Rebuild `perspective`'s half of the accumulator for `pos` into `dst`,
    /// going through this cache.
    ///
    /// Bit-identical to a from-scratch `refresh_perspective`; see the module
    /// head for why the decomposition cannot change the result.
    ///
    /// # Panics
    /// Panics if `pos` is missing `perspective`'s king.
    pub(crate) fn refresh_into(
        &mut self,
        net: &NnueNetwork,
        pos: &Position,
        perspective: Color,
        dst: &mut [i16],
    ) {
        let token = network_token(net);
        if self.token != Some(token) {
            self.invalidate();
            self.token = Some(token);
        }

        let FinnyCache {
            entries,
            scratch_active,
            diff,
            ..
        } = self;

        let bucket = pos
            .king_square(perspective)
            .expect("position must have the perspective's king")
            .index() as usize;
        let entry = &mut entries[perspective.index()][bucket];

        active_features_into(pos, perspective, scratch_active);

        if entry.initialized {
            // Within one bucket the own-king square — and so `sq_k_code`, and so
            // the `BONA_PIECE_ZERO` padding feature — is identical on both
            // sides, so the padding cancels in the multiset diff exactly as it
            // does for an ordinary incremental update.
            changed_indices_into(&entry.active, scratch_active, diff);
            apply_diff(
                &mut entry.accumulation,
                &net.ft_weights,
                &diff.added,
                &diff.removed,
            );
        } else {
            refresh_perspective(
                &mut entry.accumulation,
                &net.ft_biases,
                &net.ft_weights,
                scratch_active,
            );
            entry.initialized = true;
        }

        dst.copy_from_slice(&entry.accumulation);
        entry.active.copy_from_slice(scratch_active);
    }
}

impl Default for FinnyCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl FinnyCache {
    /// Whether the bucket currently holds a valid cached half.
    pub(crate) fn is_warm(&self, perspective: Color, king_sq: Square) -> bool {
        self.entries[perspective.index()][king_sq.index() as usize].initialized
    }

    /// Recompute `biases + sum(columns)` from `entry.active` and compare, an
    /// independent check of the module invariant.
    pub(crate) fn assert_invariant(&self, net: &NnueNetwork) {
        for per_color in self.entries.iter() {
            for entry in per_color.iter() {
                if !entry.initialized {
                    continue;
                }
                let mut expected: Vec<i16> = net.ft_biases.to_vec();
                for &idx in &entry.active {
                    let base = idx as usize * HIDDEN_SIZE;
                    let col = &net.ft_weights[base..base + HIDDEN_SIZE];
                    for (a, &w) in expected.iter_mut().zip(col.iter()) {
                        *a = a.wrapping_add(w);
                    }
                }
                assert_eq!(
                    entry.accumulation.as_slice(),
                    expected.as_slice(),
                    "finny entry violates the biases+columns invariant",
                );
            }
        }
    }
}
