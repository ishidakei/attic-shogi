//! Finny tables: a per-worker cache of refreshed accumulator halves, keyed by
//! the perspective's own-king square.
//!
//! ## Why
//!
//! A `HalfKA_hm2` feature index embeds the perspective's own-king square, so a
//! perspective whose own king moved cannot be updated differentially: its half
//! is rebuilt from scratch by [`crate::transformer::refresh_perspective`] —
//! copy the FT biases, then add all [`MAX_ACTIVE_FEATURES`] active columns. That
//! is roughly an order of magnitude more work than a one-piece diff, and king
//! moves are frequent in a shogi tree (evasions, endgame king walks).
//!
//! Within one search the king revisits the same squares over and over. So
//! instead of rebuilding from the biases every time, keep one *refreshed*
//! accumulator per (perspective, king square) together with the active-feature
//! list it was built from, and rebuild by diffing the new active list against
//! the cached one — usually a handful of columns instead of 40.
//!
//! This is Stockfish's `AccumulatorCaches` ("finny tables"). Upstream YaneuraOu
//! added the same thing in commit `72c91d8` (`nnue_feature_transformer.h`,
//! `FinnyEntry` / `FinnyCache` / `refresh_accumulator_using_finny_entry`), but
//! ships it dormant — `USE_FINNY_TABLES` is not defined by any Makefile or
//! config header, so a default upstream build never compiles it. This module is
//! therefore an **ahead-of-pin** adaptation, not a port of live pin code; see
//! the deviation note at [`crate::Accumulator::derive_into_cached`].
//!
//! ## The invariant
//!
//! For every initialised entry:
//!
//! ```text
//! entry.accumulation == ft_biases + sum over entry.active of ft_weights[column]
//! ```
//!
//! That statement mentions no position — only a feature multiset — which is why
//! an entry stays valid across nodes, searches and whole games. The only thing
//! that can invalidate it is the weights changing underneath, so the cache
//! carries a network-identity token and resets itself when the token moves.
//!
//! Applying `(removed, added) = changed_indices(entry.active, new_active)` to
//! the entry preserves the invariant for the new list, and re-establishes the
//! accumulator identity for the new position exactly: the accumulator is a sum
//! of `i16` columns under wrapping arithmetic, so any decomposition of the same
//! multiset of adds and subs is bit-identical. The cached rebuild is therefore
//! value-invariant against a from-scratch refresh, not merely close.
//!
//! ## Ownership
//!
//! One boxed cache per search worker, allocated once at worker setup and never
//! per node. Nothing is shared between workers, so Lazy SMP needs no locking.
//! Memory is `2 * 81 * HIDDEN_SIZE * 2` bytes of accumulation (~0.5 MiB) plus
//! the 162 small index lists. Upstream gates the analogous structure on
//! `kHalfDimensions <= 4096`; `HIDDEN_SIZE` is 1536, comfortably inside it.

use attic_state::{Color, Position, Square};

use crate::aligned::Aligned64;
use crate::features::{
    DiffScratch, FeatureIndex, MAX_ACTIVE_FEATURES, active_features_into, changed_indices_into,
};
use crate::transformer::{apply_diff, refresh_perspective};
use crate::types::{HIDDEN_SIZE, NnueNetwork};

/// One cached refreshed half: the accumulation and the active-feature list it
/// was built from. Cache-line aligned via [`Aligned64`], like every other NNUE
/// buffer in this crate.
#[derive(Debug)]
struct FinnyEntry {
    /// `ft_biases + sum(ft_weights[c] for c in active)`, valid only while
    /// [`Self::initialized`] is set.
    accumulation: Aligned64<i16>,
    /// The active-feature multiset [`Self::accumulation`] corresponds to.
    active: Vec<FeatureIndex>,
    /// Whether [`Self::accumulation`] currently satisfies the module invariant.
    initialized: bool,
}

impl FinnyEntry {
    fn new() -> Self {
        FinnyEntry {
            accumulation: Aligned64::<i16>::zeroed(HIDDEN_SIZE),
            active: Vec::with_capacity(MAX_ACTIVE_FEATURES),
            initialized: false,
        }
    }
}

/// A worker-private finny table: one [`FinnyEntry`] per (perspective, own-king
/// square).
///
/// Build one per search worker with [`FinnyCache::new`] and hand it to
/// [`crate::Accumulator::derive_into_cached`] on every `do_move`.
#[derive(Debug)]
pub struct FinnyCache {
    /// `[perspective][own king square]`. The raw 81 squares, matching upstream's
    /// `entries[trigger][perspective][SQ_NB]` (the bucket key is the untouched
    /// king square, not the mirrored `sq_k_code` — two mirror-equivalent king
    /// squares generate different index sets and must not share an entry).
    entries: [[FinnyEntry; Square::COUNT]; Color::COUNT],
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
/// Both the `NnueNetwork` address and its FT weight-block base are recorded:
/// a freed network's address can be reused by a later allocation, but the two
/// coinciding at once would additionally require the new network's parameter
/// arena to land on the old base — and the arena is a separate, much larger
/// allocation. A mismatch on either component resets the cache.
type NetworkToken = (usize, usize);

fn network_token(net: &NnueNetwork) -> NetworkToken {
    (
        net as *const NnueNetwork as usize,
        net.ft_weights.as_ptr() as usize,
    )
}

impl FinnyCache {
    /// Allocates an empty cache (every entry uninitialised) on the heap.
    ///
    /// Boxed because the entries own ~0.5 MiB of accumulation buffers; call this
    /// once per worker, never on the search path.
    pub fn new() -> Box<Self> {
        Box::new(FinnyCache {
            entries: std::array::from_fn(|_| std::array::from_fn(|_| FinnyEntry::new())),
            token: None,
            scratch_active: Vec::with_capacity(MAX_ACTIVE_FEATURES),
            diff: DiffScratch::default(),
        })
    }

    /// Drops every cached half. Called on a network-identity change; the entries
    /// keep their allocations and are rebuilt lazily.
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
    /// Bit-identical to `refresh_perspective(dst, biases, weights,
    /// active_features(pos, perspective))` — see the module docs for why the
    /// diff-from-a-cached-entry decomposition cannot change the result.
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

        // Destructure so the entry borrow and the scratch borrows are disjoint.
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
            // Warm entry: the cached half only differs by the pieces that moved
            // since it was built. Within one bucket the own-king square (hence
            // `sq_k_code`, hence the `BONA_PIECE_ZERO` padding feature) is
            // identical on both sides, so the padding cancels in the multiset
            // diff exactly as it does for an ordinary incremental update.
            changed_indices_into(&entry.active, scratch_active, diff);
            apply_diff(
                &mut entry.accumulation,
                &net.ft_weights,
                &diff.added,
                &diff.removed,
            );
        } else {
            // Cold entry: pay the from-scratch rebuild once for this bucket.
            refresh_perspective(
                &mut entry.accumulation,
                &net.ft_biases,
                &net.ft_weights,
                scratch_active,
            );
            entry.initialized = true;
        }

        dst.copy_from_slice(&entry.accumulation);
        // Adopt the new list; the old one becomes next call's scratch, so the
        // steady state allocates nothing.
        std::mem::swap(&mut entry.active, scratch_active);
    }
}

#[cfg(test)]
impl FinnyCache {
    /// Whether the (perspective, king square) bucket currently holds a valid
    /// cached half — i.e. whether the next rebuild there is a warm hit.
    pub(crate) fn is_warm(&self, perspective: Color, king_sq: Square) -> bool {
        self.entries[perspective.index()][king_sq.index() as usize].initialized
    }

    /// Independent check of the module invariant for every initialised entry:
    /// recompute `biases + sum(columns)` from `entry.active` and compare.
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
                    &*entry.accumulation,
                    expected.as_slice(),
                    "finny entry violates the biases+columns invariant",
                );
            }
        }
    }
}
