//! History-table skeletons the [`MovePicker`] consults for move ordering.
//!
//! Each table is either zero-filled (via `new`) or filled once with the
//! reference's `clear()` init constant (via
//! [`fill`](CapturePieceToHistory::fill)). The init constants matter for parity:
//! before any update lands, a uniformly filled table contributes a constant to
//! every score, so first-search move ordering is a pure function of those
//! constants and the static terms (MVV for captures, the capture bias for
//! evasions) — which is what makes the depth-1 node counts reproducible.
//!
//! The types mirror the reference tables the `MovePicker` constructors take
//! (`source/movepick.h`, `history.h`):
//!
//! * [`CapturePieceToHistory`] — `captureHistory[pc][to][type_of(captured)]`,
//!   consulted by `score<CAPTURES>` (`movepick.cpp`). Init `-678`.
//! * [`ButterflyHistory`] — `mainHistory[us][move]`, consulted by the quiet
//!   branches of `score<QUIETS>` / `score<EVASIONS>` (`movepick.cpp`).
//!   Init `0`.
//! * [`PieceToHistory`] — one plane of the continuation history; the main-search
//!   quiet score reads planes `[0][1][2][3][5]` and the evasion quiet score
//!   reads plane `[0]` (`movepick.cpp`). Init `-523`.
//! * [`LowPlyHistory`] — `lowPlyHistory[ply][move]`, consulted by `score<QUIETS>`
//!   near the root (`movepick.cpp`). Init `98`, re-filled per `go`.
//!
//! The `pawnHistory` and `correctionHistory` tables do not live here: they are
//! SHARED between the worker threads of one NUMA node, so they sit in
//! [`SharedHistories`] with atomic entries. At `thread_count == 1` that type is
//! byte-identical to a per-worker `PawnHistory` / `UnifiedCorrectionHistory`
//! copy, so single-thread node counts are unaffected by the sharing.
//!
//! The entries are `i16` to match the reference's `StatsEntry` width. The
//! indexing helpers below are sized to hold every index the picker can present.
//!
//! # Backing store
//!
//! Each table's heap array is allocated through the shared huge-page allocator
//! ([`attic_storage::LargePageArray`] for the flat `i16` / atomic tables,
//! [`attic_storage::LargePageBox`] for the fixed-shape continuation-correction
//! table),
//! mirroring the reference, which allocates every dynamically sized history
//! table with `make_unique_large_page` (`history.h`) and carries the
//! fixed-shape ones inside the large-page-allocated `Worker` object
//! (`thread.cpp`). The allocator affects placement only — element type,
//! index maths, init values, and clear semantics are those of the reference.

use std::sync::atomic::{AtomicI16, Ordering};

use attic_state::{Color, Move, Piece, Square};
use attic_storage::{LargePageArray, LargePageBox};

/// Number of low-ply planes the reference keeps (`LOW_PLY_HISTORY_SIZE`,
/// `history.h`): `lowPlyHistory` is indexed by `ply` for `ply < 5`.
pub const LOW_PLY_HISTORY_SIZE: usize = 5;

/// Per-table gravity limits `D` (the `StatsEntry<T, D>` template parameter,
/// `history.h`), one constant per history table. Every update is clamped and
/// pulled toward zero relative to its table's `D` — see [`apply_gravity`].
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

/// Number of pawn-structure planes in [`PawnHistory`]
/// (`PAWN_HISTORY_BASE_SIZE`, `history.h`; a power of two, thread count 1 ×
/// base 8192). The reference multiplies this by the thread count; this engine
/// shares one table per NUMA node instead, so the base size stands.
pub const PAWN_HISTORY_BASE_SIZE: usize = 8192;

/// Number of correction-history slots (`CORRHIST_BASE_SIZE`, `history.h`;
/// `u16::MAX + 1`, a power of two, thread count 1).
pub const CORRHIST_BASE_SIZE: usize = 65536;

/// The reference `StatsEntry::operator<<` gravity update
/// (`history.h`): clamp `bonus` to `[-d, d]`, then
/// `entry += clampedBonus − entry·|clampedBonus|/d`, all integer arithmetic
/// (division truncates toward zero, as Rust's `/` does). The result is
/// guaranteed to satisfy `|entry| ≤ d`, so it always fits back into `i16`
/// (every `d` here is `< i16::MAX`). Implemented once, parameterised by `d`.
pub fn apply_gravity(entry: i16, bonus: i32, d: i32) -> i16 {
    debug_assert!(d > 0);
    let clamped = bonus.clamp(-d, d);
    let val = entry as i32;
    // `val * clamped.abs()` peaks at d² ≈ 9·10⁸ for the largest table, well
    // within i32.
    let updated = val + clamped - val * clamped.abs() / d;
    debug_assert!(updated.abs() <= d, "gravity result {updated} exceeds D={d}");
    updated as i16
}

/// Number of distinct colored piece codes: `(kind, promoted, color)`.
/// `kind` (0..8) + `promoted` (×8) + `color` (×16) ⇒ 0..32.
const PIECE_NB: usize = 32;
/// Number of board squares.
const SQ_NB: usize = Square::COUNT;
/// Distinct captured-piece type codes: `kind` (0..8) + `promoted` (×8) ⇒ 0..16.
const CAPTURED_NB: usize = 16;

/// Dense index for a colored, possibly-promoted piece (the `pc` index). Unique
/// per `(kind, color, promoted)`, in `0..PIECE_NB`.
fn piece_code(p: Piece) -> usize {
    p.kind.index() + if p.promoted { 8 } else { 0 } + if p.color == Color::White { 16 } else { 0 }
}

/// Dense index for a captured piece's *type* (`type_of(captured)`), collapsing
/// colour but keeping the promoted distinction, in `1..CAPTURED_NB`. Index `0`
/// is reserved for `NO_PIECE` (an empty target), matching the reference where
/// `type_of(NO_PIECE) == NO_PIECE_TYPE == 0` — the main search reads that slot
/// for a non-capturing check's `captHist` (`yaneuraou-search.cpp`).
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

    /// Overwrite every entry with `v` — the reference's `captureHistory.fill(v)`
    /// (`yaneuraou-search.cpp`, init `-678`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    fn index(moved: Piece, to: Square, captured: Piece) -> usize {
        (piece_code(moved) * SQ_NB + to.index() as usize) * CAPTURED_NB + captured_code(captured)
    }

    /// The bonus for `moved` capturing `captured` on `to`. `0` for the
    /// zero-filled table.
    pub fn get(&self, moved: Piece, to: Square, captured: Piece) -> i32 {
        self.table[Self::index(moved, to, captured)] as i32
    }

    /// The entry for `moved` moving to an **empty** `to` — the `NO_PIECE`
    /// (index `0`) victim slot the main search reads for a non-capturing check
    /// (`captureHistory[movedPiece][to][type_of(NO_PIECE)]`,
    /// `yaneuraou-search.cpp`).
    pub fn get_empty(&self, moved: Piece, to: Square) -> i32 {
        self.table[(piece_code(moved) * SQ_NB + to.index() as usize) * CAPTURED_NB] as i32
    }

    /// Gravity-update the entry for `moved` capturing `captured` on `to`
    /// (`D = 10692`).
    pub fn update(&mut self, moved: Piece, to: Square, captured: Piece, bonus: i32) {
        let i = Self::index(moved, to, captured);
        self.table[i] = apply_gravity(self.table[i], bonus, CAPTURE_HISTORY_D);
    }
}

/// `mainHistory[us][move]` — the butterfly (from-to) quiet-move history for the
/// side to move. Indexed by the low 16 bits of the packed move, which encode
/// exactly the `(from-or-dropped-type, to)` pair the reference's `move.raw()`
/// uses.
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

    /// Overwrite every entry with `v` — the reference's `mainHistory.fill(v)`
    /// (`yaneuraou-search.cpp`, init `0`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    fn index(us: Color, m: Move) -> usize {
        us.index() * (1 << 16) + (m.to_bits() & 0xFFFF) as usize
    }

    /// The quiet-move history for `m` played by `us`. `0` for the zero-filled
    /// table.
    pub fn get(&self, us: Color, m: Move) -> i32 {
        self.table[Self::index(us, m)] as i32
    }

    /// Gravity-update `mainHistory[us][move.raw16]` (`D = 7183`).
    pub fn update(&mut self, us: Color, m: Move, bonus: i32) {
        let i = Self::index(us, m);
        self.table[i] = apply_gravity(self.table[i], bonus, MAIN_HISTORY_D);
    }
}

/// `PieceToHistory[pc][to]` — one continuation-history plane. The qsearch
/// evasion quiet score reads only `continuationHistory[0]`, i.e. one such
/// plane.
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

    /// Overwrite every entry with `v` — the reference fills each
    /// `continuationHistory` plane with `-523` (`yaneuraou-search.cpp`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    /// Overwrite this plane's entries from `src` (a `PIECE_NB * SQ_NB` slice, the
    /// plane layout). Used by [`ContinuationHistory::clone_plane`].
    fn copy_from_slice(&mut self, src: &[i16]) {
        self.table.copy_from_slice(src);
    }

    fn index(pc: Piece, to: Square) -> usize {
        piece_code(pc) * SQ_NB + to.index() as usize
    }

    /// The continuation bonus for piece `pc` moving to `to`. `0` for the
    /// zero-filled plane.
    pub fn get(&self, pc: Piece, to: Square) -> i32 {
        self.table[Self::index(pc, to)] as i32
    }

    /// Gravity-update the `[pc][to]` continuation entry (`D = 30000`).
    pub fn update(&mut self, pc: Piece, to: Square, bonus: i32) {
        let i = Self::index(pc, to);
        self.table[i] = apply_gravity(self.table[i], bonus, CONTINUATION_HISTORY_D);
    }
}

/// `lowPlyHistory[ply][move]` — the near-root quiet bonus, consulted by
/// `score<QUIETS>` for `ply < LOW_PLY_HISTORY_SIZE` as
/// `8 * lowPlyHistory[ply][move] / (1 + ply)` (`movepick.cpp`). Init
/// `98`, re-filled per `go` in the reference (`yaneuraou-search.cpp`); the
/// per-`go` fill goes through [`fill`](LowPlyHistory::fill) at the root.
///
/// Indexed by `ply` (0..[`LOW_PLY_HISTORY_SIZE`]) and the low 16 bits of the
/// packed move (`move.raw()`), exactly like [`ButterflyHistory`] but with a ply
/// axis instead of a colour axis.
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

    /// Overwrite every entry with `v` — the reference's `lowPlyHistory.fill(v)`
    /// (`yaneuraou-search.cpp`, init `98`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    fn index(ply: usize, m: Move) -> usize {
        ply * (1 << 16) + (m.to_bits() & 0xFFFF) as usize
    }

    /// The near-root bonus for `m` at `ply`. The caller guarantees
    /// `ply < LOW_PLY_HISTORY_SIZE`. `0` for the zero-filled table.
    pub fn get(&self, ply: usize, m: Move) -> i32 {
        self.table[Self::index(ply, m)] as i32
    }

    /// Gravity-update `lowPlyHistory[ply][move.raw16]` (`D = 7183`). The caller
    /// guarantees `ply < LOW_PLY_HISTORY_SIZE`.
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

/// The atomic gravity update — the reference `StatsEntry<T, D, true>::operator<<`
/// (`history.h`) on an atomic entry. Identical arithmetic to
/// [`apply_gravity`], but a plain RELAXED load-modify-store rather than a CAS
/// loop: the reference (`history.h`) uses `memory_order_relaxed` for both
/// the load and the store and does **not** guard the read-modify-write against a
/// concurrent update. Two threads updating the same entry can therefore lose one
/// update — this is accepted by design (the histories are a heuristic), so this
/// port matches it byte-for-byte and takes the same relaxed, non-atomic-RMW
/// path. The result still satisfies `|entry| <= d`, so it always fits back into
/// `i16`.
fn apply_gravity_atomic(cell: &AtomicI16, bonus: i32, d: i32) {
    debug_assert!(d > 0);
    let clamped = bonus.clamp(-d, d);
    let val = cell.load(Ordering::Relaxed) as i32;
    let updated = val + clamped - val * clamped.abs() / d;
    debug_assert!(updated.abs() <= d, "gravity result {updated} exceeds D={d}");
    cell.store(updated as i16, Ordering::Relaxed);
}

/// Number of `i16` channels a correction slot holds: one [`CorrChannel`] per
/// side to move (`MultiArray<CorrectionBundle, COLOR_NB>`, `history.h`).
const CORR_SLOT_LEN: usize = Color::COUNT * CorrChannel::COUNT;
/// Number of `i16` entries a pawn-history slot holds (`[pc][to]`,
/// `history.h`).
const PAWN_SLOT_LEN: usize = PIECE_NB * SQ_NB;

/// The reference `SharedHistories` (`history.h`): the unified correction
/// history and the pawn history, **shared between the worker threads of one NUMA
/// node** and sized by that node's thread count.
///
/// Both tables are `DynStats` of atomic entries: their slot count is
/// `thread_count * BASE` (`BASE` = [`CORRHIST_BASE_SIZE`] / [`PAWN_HISTORY_BASE_SIZE`]),
/// so a larger node gets a proportionally larger, less-contended table.
/// `thread_count` must be a non-zero power of two (asserted) so slot selection is
/// a single mask — `key & (slots - 1)` — over the full 64-bit key. At
/// `thread_count == 1` the masks are `65535` / `8191`, exactly this port's
/// former per-worker `UnifiedCorrectionHistory` / `PawnHistory` shape, so
/// single-thread search is byte-identical (the parity gate).
///
/// Entries are [`AtomicI16`] with RELAXED access ([`apply_gravity_atomic`]);
/// concurrent lost updates are accepted by design. Held behind an [`Arc`] by the
/// driver and handed to every worker on the node.
pub struct SharedHistories {
    /// The node's thread count (a power of two); the slot multiplier of both
    /// tables.
    thread_count: usize,
    /// Correction table: `thread_count * CORRHIST_BASE_SIZE` slots, each
    /// [`CORR_SLOT_LEN`] atomic `i16`. Layout `[slot][color][channel]`, flat.
    correction: LargePageArray<AtomicI16>,
    /// `correctionHistory.get_size() - 1` — the correction slot mask.
    corr_mask: usize,
    /// Pawn table: `thread_count * PAWN_HISTORY_BASE_SIZE` slots, each
    /// [`PAWN_SLOT_LEN`] atomic `i16`. Layout `[plane][pc][to]`, flat.
    pawn: LargePageArray<AtomicI16>,
    /// `pawnHistory.get_size() - 1` — the pawn slot mask.
    pawn_mask: usize,
}

impl SharedHistories {
    /// Build the shared tables for a node of `thread_count` workers, applying the
    /// reference `clear()` init (correction `0`, pawn `-1238`). `thread_count`
    /// must be a non-zero power of two (the reference `assert`,
    /// `history.h`) — the driver passes `next_power_of_two(count)`.
    ///
    /// The allocation **and** the initial fill run here, so when the driver calls
    /// this inside a node-bound thread (`execute_on_numa_node`) the first-touch
    /// policy places every page on that node.
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
        // written explicitly so every page is first-touched on the (bound) node.
        out.fill_correction(CORRECTION_INIT);
        out.fill_pawn(PAWN_INIT);
        out
    }

    /// The node's thread count (the slot multiplier).
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    /// Number of correction slots (`thread_count * CORRHIST_BASE_SIZE`).
    pub fn correction_slots(&self) -> usize {
        self.corr_mask + 1
    }

    /// Number of pawn slots (`thread_count * PAWN_HISTORY_BASE_SIZE`).
    pub fn pawn_slots(&self) -> usize {
        self.pawn_mask + 1
    }

    /// Overwrite every correction entry with `v` (atomic RELAXED stores).
    pub fn fill_correction(&self, v: i16) {
        for e in self.correction.iter() {
            e.store(v, Ordering::Relaxed);
        }
    }

    /// Overwrite every pawn entry with `v` (atomic RELAXED stores).
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
    /// by `key`. `0` on a fresh table.
    pub fn correction_get(&self, key: u64, color: Color, channel: CorrChannel) -> i32 {
        self.correction[self.corr_index(key, color, channel)].load(Ordering::Relaxed) as i32
    }

    /// Gravity-update the `channel` entry for side-to-move `color` in the slot
    /// keyed by `key` (`D = 1024`).
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
    /// by `pawn_key`. `-1238` on a fresh table.
    pub fn pawn_get(&self, pawn_key: u64, pc: Piece, to: Square) -> i32 {
        self.pawn[self.pawn_index(pawn_key, pc, to)].load(Ordering::Relaxed) as i32
    }

    /// Gravity-update the `[pc][to]` entry in the plane keyed by `pawn_key`
    /// (`D = 8192`).
    pub fn pawn_update(&self, pawn_key: u64, pc: Piece, to: Square, bonus: i32) {
        apply_gravity_atomic(
            &self.pawn[self.pawn_index(pawn_key, pc, to)],
            bonus,
            PAWN_HISTORY_D,
        );
    }
}

/// `CorrectionHistory<Continuation>` (`history.h`): a `[pc][to]`
/// table whose every cell is itself a `[pc][to]` `i16` table, each a gravity
/// entry with `D = 1024`. Init fill `6` (`yaneuraou-search.cpp`).
///
/// The outer `[pc][to]` selects a *plane* (via [`Self::plane_index`], the plane
/// the search stack points a cell's `continuationCorrectionHistory` at); the
/// inner `[pc][to]` indexes within it. The `[NO_PIECE][0]` plane
/// ([`Self::SENTINEL_PLANE`]) is the reference's default the search stack seeds
/// pre-root cells with.
pub struct ContinuationCorrectionHistory {
    /// Layout: `[plane = outer_pc*SQ_NB + outer_to][inner_pc][inner_to]`. Fixed
    /// inner dimensions so each access carries a compile-time length: the outer
    /// `plane` (a stored index) keeps a bounds check against the constant plane
    /// count, but the per-access length load is gone. `inner_pc` (from the
    /// bounded piece code) is provably in range.
    table: LargePageBox<[[[i16; SQ_NB]; PIECE_NB]; ContinuationCorrectionHistory::NUM_PLANES]>,
}

impl Default for ContinuationCorrectionHistory {
    fn default() -> Self {
        // Huge-page-backed, zero-initialised; the compile-time dimensions are
        // preserved by [`LargePageBox`].
        Self {
            table: LargePageBox::zeroed(),
        }
    }
}

impl ContinuationCorrectionHistory {
    /// Number of outer planes (`PIECE_NB * SQ_NB`).
    const NUM_PLANES: usize = PIECE_NB * SQ_NB;
    /// The `[NO_PIECE][0]` sentinel plane index (reference default). `NO_PIECE`
    /// has piece code `0` in this port's dense encoding, so the sentinel is
    /// plane `0`.
    pub const SENTINEL_PLANE: usize = 0;

    /// A fresh, zero-filled table. Use [`Self::fill`] to apply the reference's
    /// init value of `6`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v` — the reference fills each continuation
    /// correction plane with `6` (`yaneuraou-search.cpp`).
    pub fn fill(&mut self, v: i16) {
        self.table
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|e| *e = v);
    }

    /// The plane index selected by the outer `[pc][to]` (the plane a search
    /// stack cell's `continuationCorrectionHistory` points at).
    pub fn plane_index(pc: Piece, to: Square) -> usize {
        piece_code(pc) * SQ_NB + to.index() as usize
    }

    /// The inner `[pc][to]` value in `plane`. The fill value on a filled table.
    pub fn get_at(&self, plane: usize, pc: Piece, to: Square) -> i32 {
        self.table[plane][piece_code(pc)][to.index() as usize] as i32
    }

    /// Gravity-update the inner `[pc][to]` entry in `plane` (`D = 1024`).
    pub fn update_at(&mut self, plane: usize, pc: Piece, to: Square, bonus: i32) {
        let cell = &mut self.table[plane][piece_code(pc)][to.index() as usize];
        *cell = apply_gravity(*cell, bonus, CORRECTION_HISTORY_D);
    }
}

/// The worker's `continuationHistory` (`history.h` selectors):
/// `[in_check][capture]` copies of a `ContinuationHistory`
/// (`MultiArray<PieceToHistory, PIECE_NB, SQUARE_NB>`), i.e. planes keyed by
/// `(in_check, capture, moved_piece, to)`, each plane a `PieceToHistory`
/// (`[pc][to]` `i16`, gravity `D = 30000`). Init fill `-523`
/// (`yaneuraou-search.cpp`).
///
/// A search stack cell's `continuationHistory` points at one such plane; the
/// update primitives write `[pc][to]` (the current move) within it.
pub struct ContinuationHistory {
    /// Layout: `[plane][inner_pc][inner_to]`, `plane` from [`Self::plane_index`].
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
    /// Entries per plane (`PIECE_NB * SQ_NB`).
    const PLANE_LEN: usize = PIECE_NB * SQ_NB;
    /// Plane count: `[in_check][capture][pc][to]` == `2 * 2 * PIECE_NB * SQ_NB`.
    const NUM_PLANES: usize = 2 * 2 * PIECE_NB * SQ_NB;

    /// A fresh, zero-filled table. Use [`Self::fill`] for the reference init.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v` — the reference fills each continuation
    /// plane with `-523` (`yaneuraou-search.cpp`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|e| *e = v);
    }

    /// The plane index selected by `(in_check, capture, pc, to)` — the plane a
    /// search stack cell's `continuationHistory` points at after a move.
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

    /// Gravity-update the inner `[pc][to]` entry in `plane` (`D = 30000`).
    pub fn update_at(&mut self, plane: usize, pc: Piece, to: Square, bonus: i32) {
        let i = Self::cell(plane, pc, to);
        self.table[i] = apply_gravity(self.table[i], bonus, CONTINUATION_HISTORY_D);
    }

    /// Copy `plane` out into a standalone [`PieceToHistory`]. The main-search
    /// [`crate::MovePicker`] reads continuation planes as `&PieceToHistory`
    /// references (the reference stores a pointer to a plane on the search
    /// stack); this materialises one such plane so a search that keeps its
    /// continuation table in this multi-plane form can hand the picker the six
    /// planes its `contHist` array names.
    pub fn clone_plane(&self, plane: usize) -> PieceToHistory {
        let mut out = PieceToHistory::new();
        let base = plane * Self::PLANE_LEN;
        out.copy_from_slice(&self.table[base..base + Self::PLANE_LEN]);
        out
    }
}

/// `TTMoveHistory` (`history.h`): a single gravity entry with `D = 8192`,
/// init `0` (cleared at `yaneuraou-search.cpp`).
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

    /// Gravity-update the entry (`D = 8192`).
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

    /// Mask widening: two pawn keys equal mod `8192` but different mod `16384`
    /// alias at `thread_count == 1` (mask `8191`) and separate at `thread_count
    /// == 2` (mask `16383`).
    #[test]
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

    /// The same widening for the correction table (`65536` slot base).
    #[test]
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
