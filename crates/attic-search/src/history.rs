//! History tables the [`MovePicker`] consults for move ordering, mirroring the
//! reference's (`movepick.h`, `history.h`).
//!
//! Each is either zero-filled or filled once with the reference's `clear()`
//! init constant. **Those constants matter for parity**: before any update
//! lands, a uniformly filled table contributes a constant to every score, so
//! first-search move ordering is a pure function of them and the static terms —
//! which is what makes the depth-1 node counts reproducible.
//!
//! `pawnHistory` and `correctionHistory` do not live here: they are shared
//! between the worker threads of one NUMA node, so they sit in
//! [`SharedHistories`] with atomic entries. At a thread count of 1 that type is
//! byte-identical to a per-worker copy, so single-thread node counts are
//! unaffected by the sharing.
//!
//! Every table allocates through the shared huge-page allocator, as the
//! reference allocates its dynamically sized ones with `make_unique_large_page`
//! (`history.h`) and carries the fixed-shape ones inside its large-page
//! `Worker`. That affects placement only; element type, index maths, init
//! values and clear semantics are the reference's.

use std::sync::atomic::{AtomicI16, Ordering};

use attic_state::{Color, Move, Piece, Square};
use attic_storage::{LargePageArray, LargePageBox};

/// Number of low-ply planes the reference keeps (`LOW_PLY_HISTORY_SIZE`,
/// `history.h`): `lowPlyHistory` is indexed by `ply` for `ply < 5`.
pub const LOW_PLY_HISTORY_SIZE: usize = 5;

/// Per-table gravity limits `D` (the `StatsEntry<T, D>` template parameter,
/// `history.h`). See [`apply_gravity`].
pub const MAIN_HISTORY_D: i32 = 7183;
/// `lowPlyHistory` gravity limit (`ButterflyHistory` shares `D = 7183`).
pub const LOW_PLY_HISTORY_D: i32 = 7183;
/// `captureHistory` gravity limit (`history.h`).
pub const CAPTURE_HISTORY_D: i32 = 10692;
/// `continuationHistory` plane gravity limit (`PieceToHistory`, `history.h`).
pub const CONTINUATION_HISTORY_D: i32 = 30000;
/// `pawnHistory` gravity limit (`history.h`).
pub const PAWN_HISTORY_D: i32 = 8192;
/// Correction-history gravity limit `CORRECTION_HISTORY_LIMIT` (`history.h`).
pub const CORRECTION_HISTORY_D: i32 = 1024;
/// `ttMoveHistory` gravity limit (`history.h`).
pub const TT_MOVE_HISTORY_D: i32 = 8192;

/// Number of pawn-structure planes in one [`PawnHistory`] plane block
/// (`PAWN_HISTORY_BASE_SIZE`, `history.h`).
pub const PAWN_HISTORY_BASE_SIZE: usize = 8192;

/// Number of correction-history slots (`CORRHIST_BASE_SIZE`, `history.h`).
pub const CORRHIST_BASE_SIZE: usize = 65536;

/// The reference `StatsEntry::operator<<` gravity update (`history.h`):
/// clamp `bonus` to `[-d, d]`, then `entry += clamped − entry·|clamped|/d` in
/// integer arithmetic, whose division truncates toward zero as Rust's does.
///
/// The result satisfies `|entry| <= d`, so it always fits back into `i16`.
pub fn apply_gravity(entry: i16, bonus: i32, d: i32) -> i16 {
    debug_assert!(d > 0);
    let clamped = bonus.clamp(-d, d);
    let val = entry as i32;
    // `val * clamped.abs()` peaks at `d²`, well within `i32` for every table.
    let updated = val + clamped - val * clamped.abs() / d;
    debug_assert!(updated.abs() <= d, "gravity result {updated} exceeds D={d}");
    updated as i16
}

/// Number of distinct colored piece codes: `(kind, promoted, color)`.
const PIECE_NB: usize = 32;
/// Number of board squares.
const SQ_NB: usize = Square::COUNT;
/// Distinct captured-piece type codes: `(kind, promoted)`.
const CAPTURED_NB: usize = 16;

/// Dense index for a colored, possibly-promoted piece — the `pc` index.
fn piece_code(p: Piece) -> usize {
    p.kind.index() + if p.promoted { 8 } else { 0 } + if p.color == Color::White { 16 } else { 0 }
}

/// Dense index for a captured piece's *type*, collapsing colour but keeping the
/// promoted distinction. Index `0` is reserved for an empty target, as the
/// reference's `type_of(NO_PIECE) == 0` is: the main search reads that slot for
/// a non-capturing check (`yaneuraou-search.cpp`).
fn captured_code(p: Piece) -> usize {
    1 + p.kind.index() + if p.promoted { 8 } else { 0 }
}

/// `captureHistory[pc][to][type_of(captured)]` — the capture-ordering bonus for
/// "moved piece `pc` captures a `captured`-type piece on square `to`".
pub struct CapturePieceToHistory {
    table: LargePageArray<i16>,
}

impl Default for CapturePieceToHistory {
    fn default() -> Self {
        Self {
            table: LargePageArray::zeroed(PIECE_NB * SQ_NB * CAPTURED_NB),
        }
    }
}

impl CapturePieceToHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v`; the reference's init is `-678`
    /// (`yaneuraou-search.cpp`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    fn index(moved: Piece, to: Square, captured: Piece) -> usize {
        (piece_code(moved) * SQ_NB + to.index() as usize) * CAPTURED_NB + captured_code(captured)
    }

    /// The bonus for `moved` capturing `captured` on `to`.
    pub fn get(&self, moved: Piece, to: Square, captured: Piece) -> i32 {
        self.table[Self::index(moved, to, captured)] as i32
    }

    /// The entry for `moved` moving to an **empty** `to` — the index-`0` victim
    /// slot the main search reads for a non-capturing check.
    pub fn get_empty(&self, moved: Piece, to: Square) -> i32 {
        self.table[(piece_code(moved) * SQ_NB + to.index() as usize) * CAPTURED_NB] as i32
    }

    /// Gravity-update the entry for `moved` capturing `captured` on `to`.
    pub fn update(&mut self, moved: Piece, to: Square, captured: Piece, bonus: i32) {
        let i = Self::index(moved, to, captured);
        self.table[i] = apply_gravity(self.table[i], bonus, CAPTURE_HISTORY_D);
    }
}

/// `mainHistory[us][move]` — the butterfly quiet-move history for the side to
/// move, indexed by the low 16 bits of the packed move, which encode exactly
/// the pair the reference's `move.raw()` uses.
pub struct ButterflyHistory {
    table: LargePageArray<i16>,
}

impl Default for ButterflyHistory {
    fn default() -> Self {
        Self {
            table: LargePageArray::zeroed(Color::COUNT * (1 << 16)),
        }
    }
}

impl ButterflyHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v`; the reference's init is `0`.
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    fn index(us: Color, m: Move) -> usize {
        us.index() * (1 << 16) + (m.to_bits() & 0xFFFF) as usize
    }

    /// The quiet-move history for `m` played by `us`.
    pub fn get(&self, us: Color, m: Move) -> i32 {
        self.table[Self::index(us, m)] as i32
    }

    /// Gravity-update `mainHistory[us][move.raw16]`.
    pub fn update(&mut self, us: Color, m: Move, bonus: i32) {
        let i = Self::index(us, m);
        self.table[i] = apply_gravity(self.table[i], bonus, MAIN_HISTORY_D);
    }
}

/// `PieceToHistory[pc][to]` — one continuation-history plane.
pub struct PieceToHistory {
    table: Box<[i16]>,
}

impl Default for PieceToHistory {
    fn default() -> Self {
        Self {
            table: vec![0i16; PIECE_NB * SQ_NB].into_boxed_slice(),
        }
    }
}

impl PieceToHistory {
    /// A fresh, zero-filled plane.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v`; the reference's init is `-523`
    /// (`yaneuraou-search.cpp`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    /// Overwrite this plane's entries from `src`, a `PIECE_NB * SQ_NB` slice.
    fn copy_from_slice(&mut self, src: &[i16]) {
        self.table.copy_from_slice(src);
    }

    fn index(pc: Piece, to: Square) -> usize {
        piece_code(pc) * SQ_NB + to.index() as usize
    }

    /// The continuation bonus for piece `pc` moving to `to`.
    pub fn get(&self, pc: Piece, to: Square) -> i32 {
        self.table[Self::index(pc, to)] as i32
    }

    /// Gravity-update the `[pc][to]` continuation entry.
    pub fn update(&mut self, pc: Piece, to: Square, bonus: i32) {
        let i = Self::index(pc, to);
        self.table[i] = apply_gravity(self.table[i], bonus, CONTINUATION_HISTORY_D);
    }
}

/// `lowPlyHistory[ply][move]` — the near-root quiet bonus
/// (`movepick.cpp`), re-filled per `go` as the reference does. Indexed
/// like [`ButterflyHistory`] but with a ply axis instead of a colour one.
pub struct LowPlyHistory {
    table: LargePageArray<i16>,
}

impl Default for LowPlyHistory {
    fn default() -> Self {
        Self {
            table: LargePageArray::zeroed(LOW_PLY_HISTORY_SIZE * (1 << 16)),
        }
    }
}

impl LowPlyHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v`; the reference's init is `98`
    /// (`yaneuraou-search.cpp`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    fn index(ply: usize, m: Move) -> usize {
        ply * (1 << 16) + (m.to_bits() & 0xFFFF) as usize
    }

    /// The near-root bonus for `m` at `ply`. The caller guarantees
    /// `ply < LOW_PLY_HISTORY_SIZE`.
    pub fn get(&self, ply: usize, m: Move) -> i32 {
        self.table[Self::index(ply, m)] as i32
    }

    /// Gravity-update `lowPlyHistory[ply][move.raw16]`. The caller guarantees
    /// `ply < LOW_PLY_HISTORY_SIZE`.
    pub fn update(&mut self, ply: usize, m: Move, bonus: i32) {
        let i = Self::index(ply, m);
        self.table[i] = apply_gravity(self.table[i], bonus, LOW_PLY_HISTORY_D);
    }
}

/// The four correction channels of a `CorrectionBundle` (`history.h`).
/// Each unified-correction slot holds one `i16` per channel, per side to move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CorrChannel {
    /// By color and pawn structure (`pawn_key`-keyed slot).
    Pawn = 0,
    /// By color and minor-piece positions (`minor_piece_key`-keyed slot).
    Minor = 1,
    /// By White non-pawn positions (`non_pawn_key(WHITE)`-keyed slot).
    NonPawnWhite = 2,
    /// By Black non-pawn positions (`non_pawn_key(BLACK)`-keyed slot).
    NonPawnBlack = 3,
}

impl CorrChannel {
    const COUNT: usize = 4;
    fn index(self) -> usize {
        self as usize
    }
}

/// Reference `SharedHistories::clear()` init values
/// (`yaneuraou-search.cpp`): correction entries `0`, pawn entries
/// `-1238`.
const CORRECTION_INIT: i16 = 0;
const PAWN_INIT: i16 = -1238;

/// [`apply_gravity`] on an atomic entry: a plain relaxed load-modify-store
/// rather than a CAS loop, because the reference leaves its read-modify-write
/// unguarded too (`history.h`). **Two threads updating the same entry can
/// lose one update**, which is accepted by design — the histories are a
/// heuristic — so this port matches it rather than tightening it.
fn apply_gravity_atomic(cell: &AtomicI16, bonus: i32, d: i32) {
    debug_assert!(d > 0);
    let clamped = bonus.clamp(-d, d);
    let val = cell.load(Ordering::Relaxed) as i32;
    let updated = val + clamped - val * clamped.abs() / d;
    debug_assert!(updated.abs() <= d, "gravity result {updated} exceeds D={d}");
    cell.store(updated as i16, Ordering::Relaxed);
}

/// Number of `i16` channels a correction slot holds: one [`CorrChannel`] per
/// side to move.
const CORR_SLOT_LEN: usize = Color::COUNT * CorrChannel::COUNT;
/// Number of `i16` entries a pawn-history slot holds (`[pc][to]`,
/// `history.h`).
const PAWN_SLOT_LEN: usize = PIECE_NB * SQ_NB;

/// The reference `SharedHistories` (`history.h`): the unified correction
/// history and the pawn history, **shared between the worker threads of one NUMA
/// node** and sized by that node's thread count, so that a larger node gets a
/// proportionally larger, less-contended table.
///
/// `thread_count` must be a non-zero power of two, which is what makes slot
/// selection a single mask over the full 64-bit key.
pub struct SharedHistories {
    /// The node's thread count, the slot multiplier of both tables.
    thread_count: usize,
    /// Correction table, flat over `[slot][color][channel]`.
    correction: LargePageArray<AtomicI16>,
    /// `correctionHistory.get_size() - 1` — the correction slot mask.
    corr_mask: usize,
    /// Pawn table, flat over `[plane][pc][to]`.
    pawn: LargePageArray<AtomicI16>,
    /// `pawnHistory.get_size() - 1` — the pawn slot mask.
    pawn_mask: usize,
}

impl SharedHistories {
    /// Build the shared tables for a node of `thread_count` workers, applying
    /// the reference `clear()` init. `thread_count` must be a non-zero power of
    /// two, as the reference asserts (`history.h`).
    ///
    /// The allocation **and** the initial fill run here, so that calling this
    /// inside a node-bound thread first-touches every page on that node.
    pub fn new(thread_count: usize) -> Self {
        assert!(
            thread_count.is_power_of_two() && thread_count != 0,
            "SharedHistories thread_count must be a non-zero power of two, got {thread_count}"
        );
        let corr_slots = thread_count * CORRHIST_BASE_SIZE;
        let pawn_slots = thread_count * PAWN_HISTORY_BASE_SIZE;
        let out = Self {
            thread_count,
            correction: LargePageArray::zeroed(corr_slots * CORR_SLOT_LEN),
            corr_mask: corr_slots - 1,
            pawn: LargePageArray::zeroed(pawn_slots * PAWN_SLOT_LEN),
            pawn_mask: pawn_slots - 1,
        };
        // The correction fill of `0` is a no-op on the zeroed allocation, but is
        // written explicitly so that every page is first-touched on the node.
        out.fill_correction(CORRECTION_INIT);
        out.fill_pawn(PAWN_INIT);
        out
    }

    /// The node's thread count (the slot multiplier).
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    /// Number of correction slots.
    pub fn correction_slots(&self) -> usize {
        self.corr_mask + 1
    }

    /// Number of pawn slots.
    pub fn pawn_slots(&self) -> usize {
        self.pawn_mask + 1
    }

    /// Overwrite every correction entry with `v`.
    pub fn fill_correction(&self, v: i16) {
        for e in self.correction.iter() {
            e.store(v, Ordering::Relaxed);
        }
    }

    /// Overwrite every pawn entry with `v`.
    pub fn fill_pawn(&self, v: i16) {
        for e in self.pawn.iter() {
            e.store(v, Ordering::Relaxed);
        }
    }

    /// Flat index of the `channel` entry for side-to-move `color` in the slot
    /// keyed by `key`: `slot = key & corr_mask`, then `[color][channel]`.
    fn corr_index(&self, key: u64, color: Color, channel: CorrChannel) -> usize {
        let slot = (key as usize) & self.corr_mask;
        (slot * Color::COUNT + color.index()) * CorrChannel::COUNT + channel.index()
    }

    /// The `channel` correction value for side-to-move `color` in the slot keyed
    /// by `key`.
    pub fn correction_get(&self, key: u64, color: Color, channel: CorrChannel) -> i32 {
        self.correction[self.corr_index(key, color, channel)].load(Ordering::Relaxed) as i32
    }

    /// Gravity-update the `channel` entry for side-to-move `color` in the slot
    /// keyed by `key`.
    pub fn correction_update(&self, key: u64, color: Color, channel: CorrChannel, bonus: i32) {
        apply_gravity_atomic(
            &self.correction[self.corr_index(key, color, channel)],
            bonus,
            CORRECTION_HISTORY_D,
        );
    }

    /// Flat index of the `[pc][to]` entry in the plane keyed by `pawn_key`:
    /// `plane = pawn_key & pawn_mask`, then `[pc][to]`.
    fn pawn_index(&self, pawn_key: u64, pc: Piece, to: Square) -> usize {
        let plane = (pawn_key as usize) & self.pawn_mask;
        (plane * PIECE_NB + piece_code(pc)) * SQ_NB + to.index() as usize
    }

    /// The pawn-structure bonus for piece `pc` moving to `to` in the plane keyed
    /// by `pawn_key`.
    pub fn pawn_get(&self, pawn_key: u64, pc: Piece, to: Square) -> i32 {
        self.pawn[self.pawn_index(pawn_key, pc, to)].load(Ordering::Relaxed) as i32
    }

    /// Gravity-update the `[pc][to]` entry in the plane keyed by `pawn_key`.
    pub fn pawn_update(&self, pawn_key: u64, pc: Piece, to: Square, bonus: i32) {
        apply_gravity_atomic(
            &self.pawn[self.pawn_index(pawn_key, pc, to)],
            bonus,
            PAWN_HISTORY_D,
        );
    }
}

/// `CorrectionHistory<Continuation>` (`history.h`): a `[pc][to]`
/// table whose every cell is itself a `[pc][to]` table. The outer pair selects
/// a *plane*, the inner indexes within it, and [`Self::SENTINEL_PLANE`] is the
/// reference's default for pre-root search-stack cells.
pub struct ContinuationCorrectionHistory {
    /// Flat over `[plane][inner_pc][inner_to]`, with the inner dimensions fixed
    /// so that each access carries a compile-time length.
    table: LargePageBox<[[[i16; SQ_NB]; PIECE_NB]; ContinuationCorrectionHistory::NUM_PLANES]>,
}

impl Default for ContinuationCorrectionHistory {
    fn default() -> Self {
        Self {
            table: LargePageBox::zeroed(),
        }
    }
}

impl ContinuationCorrectionHistory {
    /// Number of outer planes.
    const NUM_PLANES: usize = PIECE_NB * SQ_NB;
    /// The `[NO_PIECE][0]` sentinel plane, which this port's dense piece
    /// encoding places at index `0`.
    pub const SENTINEL_PLANE: usize = 0;

    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v`; the reference's init is `6`
    /// (`yaneuraou-search.cpp`).
    pub fn fill(&mut self, v: i16) {
        self.table
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|e| *e = v);
    }

    /// The plane index selected by the outer `[pc][to]`.
    pub fn plane_index(pc: Piece, to: Square) -> usize {
        piece_code(pc) * SQ_NB + to.index() as usize
    }

    /// The inner `[pc][to]` value in `plane`.
    pub fn get_at(&self, plane: usize, pc: Piece, to: Square) -> i32 {
        self.table[plane][piece_code(pc)][to.index() as usize] as i32
    }

    /// Gravity-update the inner `[pc][to]` entry in `plane`.
    pub fn update_at(&mut self, plane: usize, pc: Piece, to: Square, bonus: i32) {
        let cell = &mut self.table[plane][piece_code(pc)][to.index() as usize];
        *cell = apply_gravity(*cell, bonus, CORRECTION_HISTORY_D);
    }
}

/// The worker's `continuationHistory`: planes keyed by
/// `(in_check, capture, moved_piece, to)`, each a [`PieceToHistory`]. A search
/// stack cell points at one such plane, and the update primitives write the
/// current move's `[pc][to]` within it.
pub struct ContinuationHistory {
    /// Flat over `[plane][inner_pc][inner_to]`.
    table: LargePageArray<i16>,
}

impl Default for ContinuationHistory {
    fn default() -> Self {
        Self {
            table: LargePageArray::zeroed(Self::NUM_PLANES * Self::PLANE_LEN),
        }
    }
}

impl ContinuationHistory {
    /// Entries per plane.
    const PLANE_LEN: usize = PIECE_NB * SQ_NB;
    /// Plane count, over `[in_check][capture][pc][to]`.
    const NUM_PLANES: usize = 2 * 2 * PIECE_NB * SQ_NB;

    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v`; the reference's init is `-523`
    /// (`yaneuraou-search.cpp`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    /// The plane index selected by `(in_check, capture, pc, to)`.
    pub fn plane_index(in_check: bool, capture: bool, pc: Piece, to: Square) -> usize {
        let ic = in_check as usize;
        let cap = capture as usize;
        ((ic * 2 + cap) * PIECE_NB + piece_code(pc)) * SQ_NB + to.index() as usize
    }

    fn cell(plane: usize, pc: Piece, to: Square) -> usize {
        plane * Self::PLANE_LEN + piece_code(pc) * SQ_NB + to.index() as usize
    }

    /// The inner `[pc][to]` value in `plane`.
    pub fn get_at(&self, plane: usize, pc: Piece, to: Square) -> i32 {
        self.table[Self::cell(plane, pc, to)] as i32
    }

    /// Gravity-update the inner `[pc][to]` entry in `plane`.
    pub fn update_at(&mut self, plane: usize, pc: Piece, to: Square, bonus: i32) {
        let i = Self::cell(plane, pc, to);
        self.table[i] = apply_gravity(self.table[i], bonus, CONTINUATION_HISTORY_D);
    }

    /// Copy `plane` out into a standalone [`PieceToHistory`], which is the form
    /// [`crate::MovePicker`] reads continuation planes in.
    pub fn clone_plane(&self, plane: usize) -> PieceToHistory {
        let mut out = PieceToHistory::new();
        let base = plane * Self::PLANE_LEN;
        out.copy_from_slice(&self.table[base..base + Self::PLANE_LEN]);
        out
    }
}

/// `TTMoveHistory` (`history.h`): a single gravity entry.
#[derive(Default)]
pub struct TtMoveHistory {
    entry: i16,
}

impl TtMoveHistory {
    /// A fresh, zero entry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The current value.
    pub fn get(&self) -> i32 {
        self.entry as i32
    }

    /// Gravity-update the entry.
    pub fn update(&mut self, bonus: i32) {
        self.entry = apply_gravity(self.entry, bonus, TT_MOVE_HISTORY_D);
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use attic_state::PieceKind;
    use std::sync::Arc;

    fn bp() -> Piece {
        Piece::new(PieceKind::Pawn, Color::Black)
    }
    fn to() -> Square {
        Square::new(4, 3).unwrap()
    }
    const CHANNELS: [CorrChannel; 4] = [
        CorrChannel::Pawn,
        CorrChannel::Minor,
        CorrChannel::NonPawnWhite,
        CorrChannel::NonPawnBlack,
    ];

    /// Table sizing scales the slot count by `thread_count` for both tables.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn sizing_scales_with_thread_count() {
        for &tc in &[1usize, 2, 4, 8] {
            let sh = SharedHistories::new(tc);
            assert_eq!(sh.thread_count(), tc);
            assert_eq!(sh.correction_slots(), tc * CORRHIST_BASE_SIZE);
            assert_eq!(sh.pawn_slots(), tc * PAWN_HISTORY_BASE_SIZE);
        }
    }

    /// The `thread_count` power-of-two assert (`history.h`).
    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two() {
        let _ = SharedHistories::new(3);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_zero_thread_count() {
        let _ = SharedHistories::new(0);
    }

    /// Init values: pawn `-1238`, correction `0`, across distinct slots.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn init_values() {
        let sh = SharedHistories::new(1);
        assert_eq!(sh.pawn_get(0, bp(), to()), -1238);
        assert_eq!(sh.pawn_get(0xFFFF, bp(), to()), -1238);
        for &color in &[Color::Black, Color::White] {
            for &ch in &CHANNELS {
                assert_eq!(sh.correction_get(0, color, ch), 0);
                assert_eq!(sh.correction_get(0xFFFF, color, ch), 0);
            }
        }
    }

    /// Two pawn keys equal mod the base size but differing above it must alias
    /// at one thread and separate at two.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn pawn_mask_widens_with_thread_count() {
        let key_a = 0u64;
        let key_b = PAWN_HISTORY_BASE_SIZE as u64; // 8192: equal mod 8192, differ mod 16384

        let one = SharedHistories::new(1);
        let b_before = one.pawn_get(key_b, bp(), to());
        one.pawn_update(key_a, bp(), to(), 1_000_000);
        assert_ne!(
            one.pawn_get(key_b, bp(), to()),
            b_before,
            "thread_count 1: keys 0 and 8192 must share one plane"
        );

        let two = SharedHistories::new(2);
        let b_before = two.pawn_get(key_b, bp(), to());
        two.pawn_update(key_a, bp(), to(), 1_000_000);
        assert_ne!(
            two.pawn_get(key_a, bp(), to()),
            b_before,
            "thread_count 2: the updated plane still moved"
        );
        assert_eq!(
            two.pawn_get(key_b, bp(), to()),
            b_before,
            "thread_count 2: keys 0 and 8192 must be in distinct planes"
        );
    }

    /// The same widening for the correction table.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn correction_mask_widens_with_thread_count() {
        let key_a = 0u64;
        let key_b = CORRHIST_BASE_SIZE as u64; // 65536

        let one = SharedHistories::new(1);
        one.correction_update(key_a, Color::Black, CorrChannel::Pawn, 1_000_000);
        assert_ne!(
            one.correction_get(key_b, Color::Black, CorrChannel::Pawn),
            0,
            "thread_count 1: correction keys 0 and 65536 share a slot"
        );

        let two = SharedHistories::new(2);
        two.correction_update(key_a, Color::Black, CorrChannel::Pawn, 1_000_000);
        assert_ne!(
            two.correction_get(key_a, Color::Black, CorrChannel::Pawn),
            0
        );
        assert_eq!(
            two.correction_get(key_b, Color::Black, CorrChannel::Pawn),
            0,
            "thread_count 2: correction keys 0 and 65536 are in distinct slots"
        );
    }

    /// Atomicity smoke: two threads hammering the same entry with `<<` terminates
    /// and leaves a value within `[-D, D]` (the reference accepts lost updates but
    /// the gravity bound always holds — `history.h`).
    #[test]
    #[cfg_attr(miri, ignore)]
    fn concurrent_updates_terminate_within_limit() {
        let sh = Arc::new(SharedHistories::new(2));
        let key = 12_345u64;
        let handles: Vec<_> = (0..2)
            .map(|t| {
                let sh = Arc::clone(&sh);
                std::thread::spawn(move || {
                    for i in 0..20_000i32 {
                        let bonus = if (i + t) % 2 == 0 { 5000 } else { -5000 };
                        sh.pawn_update(key, bp(), to(), bonus);
                        sh.correction_update(key, Color::White, CorrChannel::Minor, bonus);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("update thread must not panic");
        }
        let pv = sh.pawn_get(key, bp(), to());
        assert!(pv.abs() <= PAWN_HISTORY_D, "pawn value {pv} exceeds D");
        let cv = sh.correction_get(key, Color::White, CorrChannel::Minor);
        assert!(
            cv.abs() <= CORRECTION_HISTORY_D,
            "correction value {cv} exceeds D"
        );
    }
}
