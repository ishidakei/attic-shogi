//! History-table **update** machinery — a faithful port of the reference
//! `update_all_stats`, `update_quiet_histories`, `update_continuation_histories`
//! and `update_correction_history`
//! (`source/engine/yaneuraou-engine/yaneuraou-search.cpp`,
//! lines 748-772 and 5284-5414 at the pinned submodule).
//!
//! # Shape
//!
//! The functions here are decoupled from the search body: they operate on a
//! self-contained [`WorkerHistories`] bundle and a slice of
//! [`SearchStackCell`]s, with the reference's `(ss-N)->continuationHistory`
//! pointers modelled as flat indices into the continuation tables. `qsearch`
//! reconciles [`SearchStackCell`] with its live search stack and supplies the
//! real plane indices.
//!
//! Line numbers in the comments point into the reference file above; all
//! arithmetic is integer (division truncates toward zero, as Rust's `/` does),
//! matching the C++ exactly.

use std::sync::Arc;

use attic_state::{Color, Move, Piece, Position, Square};
use attic_storage::Value;

use crate::history::{
    ButterflyHistory, CapturePieceToHistory, ContinuationCorrectionHistory, ContinuationHistory,
    CorrChannel, LOW_PLY_HISTORY_SIZE, LowPlyHistory, SharedHistories, TtMoveHistory,
};

/// Reference `clear()` init constants (`yaneuraou-search.cpp`).
const MAIN_HISTORY_INIT: i16 = 0;
const CAPTURE_HISTORY_INIT: i16 = -678;
const CONTINUATION_INIT: i16 = -523;
const CONTINUATION_CORRECTION_INIT: i16 = 6;
/// The pawn-history init (`-1238`) is applied by the shared table
/// ([`SharedHistories::new`]); the tests below assert the bundle exposes it.
#[cfg(test)]
const PAWN_HISTORY_INIT: i16 = -1238;

/// Capacity of a searched-move list (`SEARCHEDLIST_CAPACITY`,
/// `yaneuraou-search.cpp`): both `quietsSearched` and `capturesSearched`
/// hold at most 32 moves.
pub const SEARCHED_LIST_CAPACITY: usize = 32;

/// A fixed-capacity list of the moves tried at a node (the reference's
/// `SearchedList quietsSearched` / `capturesSearched`, `movepick.h`). Backed by
/// an inline `[Move; SEARCHED_LIST_CAPACITY]` array so it lives on the search
/// stack with **no** per-node heap allocation; the search only ever
/// pushes while `moveCount <= SEARCHED_LIST_CAPACITY`, so it never overflows.
#[derive(Clone)]
pub struct SearchedList {
    moves: [Move; SEARCHED_LIST_CAPACITY],
    len: usize,
}

impl Default for SearchedList {
    fn default() -> Self {
        SearchedList {
            moves: [Move::none(); SEARCHED_LIST_CAPACITY],
            len: 0,
        }
    }
}

impl SearchedList {
    /// A new, empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `mv`. The caller guarantees at most [`SEARCHED_LIST_CAPACITY`]
    /// pushes per node (`moveCount <= SEARCHED_LIST_CAPACITY`), matching the
    /// reference's bounded `push_back`.
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < SEARCHED_LIST_CAPACITY, "searched list overflow");
        self.moves[self.len] = mv;
        self.len += 1;
    }

    /// The pushed moves in order.
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

/// The full set of history tables an update touches — the reference's
/// per-worker tables plus a handle to the node-shared correction / pawn tables
/// ([`SharedHistories`]). Bundled so the update functions can borrow
/// them together. [`WorkerHistories::new`] applies the reference `clear()` init.
///
/// The pawn and correction tables are not part of this bundle: they are SHARED
/// between the worker threads of one NUMA node and reached through
/// [`Self::shared`], a cheap [`Arc`] clone the driver hands each worker. At
/// `thread_count == 1` the shared tables are byte-identical to per-worker
/// copies, so single-thread search is unaffected by the sharing.
pub struct WorkerHistories {
    /// `mainHistory[us][move.raw16]` (init `0`).
    pub main: ButterflyHistory,
    /// `lowPlyHistory[ply][move.raw16]`. The reference re-fills this per `go`
    /// (to `98`); `clear()` itself leaves it, so this bundle leaves it zero and
    /// the per-`go` fill happens at the root.
    pub low_ply: LowPlyHistory,
    /// `captureHistory[pc][to][captured type]` (init `-678`).
    pub capture: CapturePieceToHistory,
    /// `continuationHistory[in_check][capture][pc][to]` planes (init `-523`).
    pub continuation: ContinuationHistory,
    /// `continuationCorrectionHistory[pc][to]` planes (init `6`).
    pub continuation_correction: ContinuationCorrectionHistory,
    /// `ttMoveHistory` — a single gravity entry (init `0`).
    pub tt_move: TtMoveHistory,
    /// The node-shared `correctionHistory` and `pawnHistory`. Every
    /// pawn / correction read and update routes here, so an interior update on
    /// one worker is visible to every worker on the same node.
    pub shared: Arc<SharedHistories>,
}

impl Default for WorkerHistories {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerHistories {
    /// A fresh bundle with its own single-thread [`SharedHistories`] and the
    /// reference `clear()` init values applied. This is the single-worker /
    /// test path; the shared tables are sized `thread_count == 1`, exactly the
    /// former per-worker shape. The driver's multi-thread path uses
    /// [`Self::with_shared`] to hand each worker its node's shared tables.
    pub fn new() -> Self {
        Self::with_shared(Arc::new(SharedHistories::new(1)))
    }

    /// A fresh bundle of per-worker tables (reference `clear()` init) attached to
    /// the given node-shared tables. The driver builds one [`SharedHistories`]
    /// per NUMA node and hands each worker on that node a cheap [`Arc`] clone.
    pub fn with_shared(shared: Arc<SharedHistories>) -> Self {
        let mut main = ButterflyHistory::new();
        main.fill(MAIN_HISTORY_INIT);
        let mut capture = CapturePieceToHistory::new();
        capture.fill(CAPTURE_HISTORY_INIT);
        let mut continuation = ContinuationHistory::new();
        continuation.fill(CONTINUATION_INIT);
        let mut continuation_correction = ContinuationCorrectionHistory::new();
        continuation_correction.fill(CONTINUATION_CORRECTION_INIT);
        Self {
            main,
            low_ply: LowPlyHistory::new(),
            capture,
            continuation,
            continuation_correction,
            tt_move: TtMoveHistory::new(),
            shared,
        }
    }

    /// Swap in a different node's shared tables, leaving the per-worker tables
    /// untouched. The driver calls this on a pool rebuild so the session
    /// (coordinator) worker keeps its game-scoped per-worker tables while picking
    /// up freshly (re)built shared tables for its assigned node.
    pub fn set_shared(&mut self, shared: Arc<SharedHistories>) {
        self.shared = shared;
    }
}

/// One cell of the search stack, holding exactly the fields the update /
/// correction functions read across plies. The search body writes these during
/// search; the reference defaults let the update functions be exercised in
/// isolation.
///
/// `cont_hist` / `cont_corr` are the flat plane indices the reference stores as
/// `continuationHistory` / `continuationCorrectionHistory` *pointers*. Their
/// pre-root defaults are the `[0][0][NO_PIECE][0]` / `[NO_PIECE][0]` sentinel
/// planes (both index `0`).
#[derive(Clone, Debug)]
pub struct SearchStackCell {
    /// `ss->currentMove` — the move played from this ply (`Move::none()` when
    /// unset; a null move counts as not-`is_ok`).
    pub current_move: Move,
    /// `ss->inCheck` — whether the side to move at this ply is in check.
    pub in_check: bool,
    /// `ss->ttHit` — whether the TT probe at this ply hit.
    pub tt_hit: bool,
    /// `ss->ttPv` — whether this ply is on a TT principal variation. Persistent
    /// across probes; the interior search reads `(ss-1)->ttPv` at fail-low.
    pub tt_pv: bool,
    /// `ss->moveCount` — moves tried at this ply.
    pub move_count: i32,
    /// `ss->statScore` — the reference default is `0`.
    pub stat_score: i32,
    /// `ss->ply` — distance from the search root.
    pub ply: i32,
    /// `ss->staticEval` — the (corrected) static evaluation at this ply, read
    /// across plies (`(ss-1)`/`(ss-2)`) by the interior search's improving /
    /// hindsight / fail-low logic. Defaults to `VALUE_NONE`.
    pub static_eval: Value,
    /// `ss->reduction` — the reduction applied to this ply's move (read back as
    /// `(ss-1)->reduction` = `priorReduction`).
    pub reduction: i32,
    /// `ss->cutoffCnt` — how many beta cutoffs happened at this ply; the parent
    /// reads `(ss+1)->cutoffCnt` when scaling reductions.
    pub cutoff_cnt: i32,
    /// `ss->excludedMove` — the singular-extension excluded move.
    pub excluded_move: Move,
    /// `ss->followPV` — whether this ply follows the previous iteration's PV.
    pub follow_pv: bool,
    /// `ss->pv` — the principal variation collected from this ply.
    pub pv: Vec<Move>,
    /// Plane index into [`ContinuationHistory`] this cell points at
    /// (`ss->continuationHistory`).
    pub cont_hist: usize,
    /// Plane index into [`ContinuationCorrectionHistory`] this cell points at
    /// (`ss->continuationCorrectionHistory`).
    pub cont_corr: usize,
}

impl Default for SearchStackCell {
    fn default() -> Self {
        Self {
            current_move: Move::none(),
            in_check: false,
            tt_hit: false,
            tt_pv: false,
            move_count: 0,
            stat_score: 0,
            ply: 0,
            // `VALUE_NONE` (`types.h`) — the reference pre-root sentinel.
            static_eval: 32002,
            reduction: 0,
            cutoff_cnt: 0,
            excluded_move: Move::none(),
            follow_pv: false,
            pv: Vec::new(),
            cont_hist: 0,
            cont_corr: ContinuationCorrectionHistory::SENTINEL_PLANE,
        }
    }
}

/// Plies-and-weights for [`update_continuation_histories`]
/// (`yaneuraou-search.cpp`): `{1:1157, 2:648, 3:288, 4:576, 5:140,
/// 6:441}`.
const CONTHIST_BONUSES: [(usize, i32); 6] =
    [(1, 1157), (2, 648), (3, 288), (4, 576), (5, 140), (6, 441)];

/// A *plain* capture test (`Position::capture`, `position.h`): a non-drop
/// landing on an occupied square. In this engine `capture_stage == capture`
/// (`position.h`), so this is the exact predicate `update_all_stats` uses.
fn is_capture(pos: &Position, m: Move) -> bool {
    !m.is_drop() && pos.board().get(m.to_sq()).is_some()
}

/// `update_continuation_histories(ss, pc, to, bonus)`
/// (`yaneuraou-search.cpp`): fold `bonus` into the continuation
/// planes of the previous plies formed with the current `(pc, to)`.
///
/// `ss` is the index of the *current* cell in `stack`; `(ss - i)` cells are
/// earlier plies. Each write is guarded by `(ss - i)->currentMove` being an ok
/// move; the `i > 2` plies are skipped entirely when the current cell is in
/// check. The caller must ensure `ss >= 6` so `stack[ss - i]` is in range (the
/// reference guarantees this with its `stack + 7` sentinel base).
pub fn update_continuation_histories(
    hist: &mut WorkerHistories,
    stack: &[SearchStackCell],
    ss: usize,
    pc: Piece,
    to: Square,
    bonus: i32,
) {
    let in_check = stack[ss].in_check;
    for (i, weight) in CONTHIST_BONUSES {
        // Only update the first 2 continuation histories if we are in check.
        if in_check && i > 2 {
            break;
        }
        let prev = &stack[ss - i];
        if prev.current_move.is_ok() {
            let value = (bonus * weight / 1024) + 88 * (i < 2) as i32;
            hist.continuation.update_at(prev.cont_hist, pc, to, value);
        }
    }
}

/// `update_quiet_histories(pos, ss, w, move, bonus)`
/// (`yaneuraou-search.cpp`): the reference's quiet-move heuristic
/// bump. `ss` is the current cell index in `stack`.
pub fn update_quiet_histories(
    hist: &mut WorkerHistories,
    pos: &Position,
    stack: &[SearchStackCell],
    ss: usize,
    m: Move,
    bonus: i32,
) {
    let us = pos.side_to_move();
    hist.main.update(us, m, bonus);

    let ply = stack[ss].ply;
    if (ply as usize) < LOW_PLY_HISTORY_SIZE {
        hist.low_ply.update(ply as usize, m, bonus * 761 / 1024);
    }

    // `pos.moved_piece(move)` is `moved_piece_after` in this engine.
    let moved = m.moved_piece_after();
    update_continuation_histories(hist, stack, ss, moved, m.to_sq(), bonus * 955 / 1024);

    // Asymmetric pawn-history bump: +850/1024 of a positive bonus, +550/1024 of
    // a non-positive one.
    let pawn_bonus = bonus * if bonus > 0 { 850 } else { 550 } / 1024;
    hist.shared
        .pawn_update(pos.pawn_key(), moved, m.to_sq(), pawn_bonus);
}

/// `update_all_stats(...)` (`yaneuraou-search.cpp`): the post-search
/// history-statistics update run when a best move is found.
///
/// `ss` is the current cell index in `stack` (`ss >= 7`, so the refutation
/// penalty's `update_continuation_histories(ss - 1, …)` stays in range).
/// `prev_sq` is the previous move's destination (`None` for `SQ_NONE`).
/// `prior_capture` mirrors `pos.captured_piece() != NO_PIECE` — whether the
/// move that reached this node captured. It is a caller argument because this
/// module does not model the Position's captured-piece stack.
#[allow(clippy::too_many_arguments)]
pub fn update_all_stats(
    hist: &mut WorkerHistories,
    pos: &Position,
    stack: &[SearchStackCell],
    ss: usize,
    best_move: Move,
    prev_sq: Option<Square>,
    quiets_searched: &[Move],
    captures_searched: &[Move],
    depth: i32,
    tt_move: Move,
    prior_capture: bool,
) {
    let moved_piece = best_move.moved_piece_after();

    let bonus = (128 * depth - 77).min(1529)
        + 353 * (best_move == tt_move) as i32
        + stack[ss - 1].stat_score / 32;
    let malus = (882 * depth - 204).min(2122);

    if !is_capture(pos, best_move) {
        update_quiet_histories(hist, pos, stack, ss, best_move, bonus * 806 / 1024);

        // Decrease stats for all non-best quiet moves.
        let mut actual_malus = malus * 1113 / 1024;
        for &mv in quiets_searched {
            actual_malus = actual_malus * 977 / 1024;
            update_quiet_histories(hist, pos, stack, ss, mv, -actual_malus);
        }
    } else if let Some(captured) = pos.board().get(best_move.to_sq()) {
        // Increase stats for the best move when it was a capture. A capture's
        // destination is always occupied, so the `if let` never falls through;
        // it just keeps the site panic-free instead of an `expect`.
        hist.capture.update(
            moved_piece,
            best_move.to_sq(),
            captured,
            bonus * 1286 / 1024,
        );
    }

    // Extra penalty for a quiet early move that was not a TT move in the
    // previous ply when it gets refuted.
    if let Some(prev_sq) = prev_sq
        && stack[ss - 1].move_count == 1 + stack[ss - 1].tt_hit as i32
        && !prior_capture
        && let Some(prev_piece) = pos.board().get(prev_sq)
    {
        update_continuation_histories(
            hist,
            stack,
            ss - 1,
            prev_piece,
            prev_sq,
            -malus * 616 / 1024,
        );
    }

    // Decrease stats for all non-best capture moves.
    for &mv in captures_searched {
        let moved = mv.moved_piece_after();
        if let Some(captured) = pos.board().get(mv.to_sq()) {
            hist.capture
                .update(moved, mv.to_sq(), captured, -malus * 1559 / 1024);
        }
    }
}

/// `update_correction_history(pos, ss, w, bonus)`
/// (`yaneuraou-search.cpp`): fold `bonus` into the correction channels.
/// `ss` is the current cell index in `stack` (`ss >= 4`).
pub fn update_correction_history(
    hist: &mut WorkerHistories,
    pos: &Position,
    stack: &[SearchStackCell],
    ss: usize,
    bonus: i32,
) {
    let us = pos.side_to_move();

    hist.shared
        .correction_update(pos.pawn_key(), us, CorrChannel::Pawn, bonus);
    hist.shared.correction_update(
        pos.minor_piece_key(),
        us,
        CorrChannel::Minor,
        bonus * 153 / 128,
    );
    hist.shared.correction_update(
        pos.non_pawn_key(Color::White),
        us,
        CorrChannel::NonPawnWhite,
        bonus * 187 / 128,
    );
    hist.shared.correction_update(
        pos.non_pawn_key(Color::Black),
        us,
        CorrChannel::NonPawnBlack,
        bonus * 187 / 128,
    );

    let m = stack[ss - 1].current_move;
    if m.is_ok() {
        let to = m.to_sq();
        if let Some(pc) = pos.board().get(to) {
            hist.continuation_correction.update_at(
                stack[ss - 2].cont_corr,
                pc,
                to,
                bonus * 126 / 128,
            );
            hist.continuation_correction.update_at(
                stack[ss - 4].cont_corr,
                pc,
                to,
                bonus * 63 / 128,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{
        CAPTURE_HISTORY_D, CONTINUATION_HISTORY_D, CORRECTION_HISTORY_D, MAIN_HISTORY_D,
        PAWN_HISTORY_D, apply_gravity,
    };
    use attic_state::{Move, PieceKind, Square, parse_sfen};

    // ---- gravity primitive ------------------------------------------------

    /// A small deterministic xorshift64* (the workspace bans `Math.random`-style
    /// nondeterminism).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    #[test]
    fn gravity_never_exceeds_d_over_random_walk() {
        // Property: repeatedly applying random bonuses keeps |entry| <= D.
        for &d in &[
            MAIN_HISTORY_D,
            CAPTURE_HISTORY_D,
            CONTINUATION_HISTORY_D,
            PAWN_HISTORY_D,
            CORRECTION_HISTORY_D,
        ] {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ d as u64);
            let mut entry: i16 = 0;
            for _ in 0..5000 {
                // Bonuses spanning well beyond [-D, D] to exercise the clamp.
                let bonus = (rng.next() as i64 % (4 * d as i64 + 1) - 2 * d as i64) as i32;
                entry = apply_gravity(entry, bonus, d);
                assert!(
                    (entry as i32).abs() <= d,
                    "entry {entry} exceeded D={d} after bonus",
                );
            }
        }
    }

    #[test]
    fn gravity_bonus_equal_to_d_pins_entry_to_d() {
        // bonus == D ⇒ entry becomes exactly D (regardless of prior value):
        //   val + D - val*D/D == val + D - val == D.
        for &d in &[MAIN_HISTORY_D, CONTINUATION_HISTORY_D, CORRECTION_HISTORY_D] {
            for &start in &[-d as i16 as i32, -37, 0, 91, d] {
                let e = apply_gravity(start as i16, d, d);
                assert_eq!(e as i32, d, "bonus=D must pin entry to D (start {start})");
                let e = apply_gravity(start as i16, -d, d);
                assert_eq!(
                    e as i32, -d,
                    "bonus=-D must pin entry to -D (start {start})"
                );
            }
        }
    }

    #[test]
    fn gravity_hand_computed_sequence() {
        // D = 1024. Start at 0.
        //   << 512:  0 + 512 - 0        = 512
        //   << 512:  512 + 512 - 512*512/1024 = 1024 - 256 = 768
        //   << -256: 768 - 256 - 768*256/1024 = 512 - 192 = 320
        let d = 1024;
        let mut e = 0i16;
        e = apply_gravity(e, 512, d);
        assert_eq!(e, 512);
        e = apply_gravity(e, 512, d);
        assert_eq!(e, 768);
        e = apply_gravity(e, -256, d);
        assert_eq!(e, 320);
    }

    #[test]
    fn gravity_clamps_out_of_range_bonus() {
        // bonus far above D is clamped to D before the update, so from 0 it
        // still pins to D.
        assert_eq!(apply_gravity(0, 1_000_000, 1024) as i32, 1024);
        assert_eq!(apply_gravity(0, -1_000_000, 1024) as i32, -1024);
    }

    // ---- WorkerHistories init/clear under the large-page allocation -------

    #[test]
    fn worker_histories_new_applies_reference_init_values() {
        // The tables live on the shared huge-page allocator; this pins that the
        // `clear()`-init values a fresh bundle exposes are byte-for-byte the
        // reference constants across every table, i.e. that the allocation path
        // does not disturb them. A representative index per table is sampled
        // (uniform fills make any index representative).
        let hist = WorkerHistories::new();

        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        let bs = Piece::new(PieceKind::Silver, Color::White);
        let to = Square::new(4, 3).unwrap();
        let m = Move::make(Square::new(4, 4).unwrap(), to, bp);

        // mainHistory init 0 (both a low and a high index).
        assert_eq!(hist.main.get(Color::Black, m), MAIN_HISTORY_INIT as i32);
        assert_eq!(hist.main.get(Color::White, m), MAIN_HISTORY_INIT as i32);
        // lowPlyHistory left zero by clear() (per-`go` fill is elsewhere).
        assert_eq!(hist.low_ply.get(0, m), 0);
        assert_eq!(hist.low_ply.get(LOW_PLY_HISTORY_SIZE - 1, m), 0);
        // captureHistory init -678 (occupied-victim and empty-victim slots).
        assert_eq!(hist.capture.get(bs, to, bp), CAPTURE_HISTORY_INIT as i32);
        assert_eq!(hist.capture.get_empty(bs, to), CAPTURE_HISTORY_INIT as i32);
        // pawnHistory init -1238, across distinct pawn-key planes.
        assert_eq!(hist.shared.pawn_get(0, bs, to), PAWN_HISTORY_INIT as i32);
        assert_eq!(
            hist.shared.pawn_get(0xFFFF, bs, to),
            PAWN_HISTORY_INIT as i32
        );
        // continuationHistory init -523, across distinct planes.
        assert_eq!(
            hist.continuation.get_at(0, bs, to),
            CONTINUATION_INIT as i32
        );
        let last_plane = ContinuationHistory::plane_index(true, true, bs, to);
        assert_eq!(
            hist.continuation.get_at(last_plane, bs, to),
            CONTINUATION_INIT as i32
        );
        // correctionHistory init 0 across all four channels and both colors.
        for &color in &[Color::Black, Color::White] {
            for &ch in &[
                CorrChannel::Pawn,
                CorrChannel::Minor,
                CorrChannel::NonPawnWhite,
                CorrChannel::NonPawnBlack,
            ] {
                assert_eq!(hist.shared.correction_get(0, color, ch), 0);
                assert_eq!(hist.shared.correction_get(0xFFFF, color, ch), 0);
            }
        }
        // continuationCorrectionHistory init 6, sentinel and a deep plane.
        assert_eq!(
            hist.continuation_correction.get_at(
                ContinuationCorrectionHistory::SENTINEL_PLANE,
                bs,
                to
            ),
            CONTINUATION_CORRECTION_INIT as i32,
        );
        let deep = ContinuationCorrectionHistory::plane_index(bs, to);
        assert_eq!(
            hist.continuation_correction.get_at(deep, bs, to),
            CONTINUATION_CORRECTION_INIT as i32,
        );
        // ttMoveHistory init 0.
        assert_eq!(hist.tt_move.get(), 0);
    }

    // ---- fixtures for the update-function scenarios -----------------------

    /// A two-king board plus a black pawn on 5e, black to move — a simple,
    /// fully-determined position for the quiet/correction scenarios.
    fn simple_pos() -> Position {
        parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b - 1").unwrap()
    }

    /// A minimal stack with `base` sentinel cells before the working cell, all
    /// defaulted. Returns the vector; index `base` is the "current" cell.
    fn fresh_stack(len: usize) -> Vec<SearchStackCell> {
        vec![SearchStackCell::default(); len]
    }

    // ---- update_continuation_histories ------------------------------------

    #[test]
    fn continuation_histories_writes_expected_cells() {
        // A stack where (ss-1)..(ss-6) all have distinct ok currentMoves and
        // distinct continuation planes, current cell NOT in check. Verify each
        // plane receives the weighted bonus with the +88 only on i==1.
        let mut hist = WorkerHistories::new();
        let ss = 7usize;
        let mut stack = fresh_stack(ss + 1);

        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let dummy = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), pawn);
        // Give each of the six previous plies an ok move and a unique plane.
        for i in 1..=6 {
            stack[ss - i].current_move = dummy;
            stack[ss - i].cont_hist = i; // distinct planes 1..=6
        }
        stack[ss].in_check = false;

        let pc = Piece::new(PieceKind::Silver, Color::Black);
        let to = Square::new(3, 3).unwrap();
        let bonus = 1000;

        // Pre-values are all the -523 continuation init.
        let pre: Vec<i32> = (1..=6)
            .map(|i| hist.continuation.get_at(i, pc, to))
            .collect();

        update_continuation_histories(&mut hist, &stack, ss, pc, to, bonus);

        for (idx, i) in (1..=6).enumerate() {
            let weight = CONTHIST_BONUSES[idx].1;
            let write = (bonus * weight / 1024) + 88 * (i < 2) as i32;
            let expected = apply_gravity(pre[idx] as i16, write, CONTINUATION_HISTORY_D) as i32;
            assert_eq!(
                hist.continuation.get_at(i, pc, to),
                expected,
                "continuation plane {i} mismatch",
            );
        }
    }

    #[test]
    fn continuation_histories_skips_deep_plies_when_in_check() {
        // With the current cell in check, only i <= 2 are updated.
        let mut hist = WorkerHistories::new();
        let ss = 7usize;
        let mut stack = fresh_stack(ss + 1);
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let dummy = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), pawn);
        for i in 1..=6 {
            stack[ss - i].current_move = dummy;
            stack[ss - i].cont_hist = i;
        }
        stack[ss].in_check = true;

        let pc = Piece::new(PieceKind::Silver, Color::Black);
        let to = Square::new(3, 3).unwrap();
        update_continuation_histories(&mut hist, &stack, ss, pc, to, 1000);

        // Planes 1,2 changed; 3..6 remain at the -523 init.
        assert_ne!(
            hist.continuation.get_at(1, pc, to),
            CONTINUATION_INIT as i32
        );
        assert_ne!(
            hist.continuation.get_at(2, pc, to),
            CONTINUATION_INIT as i32
        );
        for i in 3..=6 {
            assert_eq!(
                hist.continuation.get_at(i, pc, to),
                CONTINUATION_INIT as i32,
                "plane {i} must be untouched in check",
            );
        }
    }

    #[test]
    fn continuation_histories_skips_null_prev_moves() {
        // A ply whose currentMove is not ok is skipped.
        let mut hist = WorkerHistories::new();
        let ss = 7usize;
        let mut stack = fresh_stack(ss + 1);
        // Only (ss-1) has an ok move; the rest keep Move::none().
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        stack[ss - 1].current_move =
            Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), pawn);
        stack[ss - 1].cont_hist = 1;

        let pc = Piece::new(PieceKind::Silver, Color::Black);
        let to = Square::new(3, 3).unwrap();
        update_continuation_histories(&mut hist, &stack, ss, pc, to, 1000);

        assert_ne!(
            hist.continuation.get_at(1, pc, to),
            CONTINUATION_INIT as i32
        );
        // Plane 0 (the sentinel the other cells point at) is untouched.
        assert_eq!(
            hist.continuation.get_at(0, pc, to),
            CONTINUATION_INIT as i32
        );
    }

    // ---- update_quiet_histories -------------------------------------------

    #[test]
    fn quiet_histories_hand_computed() {
        let mut hist = WorkerHistories::new();
        let pos = simple_pos();
        let ss = 7usize;
        let mut stack = fresh_stack(ss + 1);
        stack[ss].ply = 2; // < LOW_PLY_HISTORY_SIZE, so lowPlyHistory updates.
        // Give (ss-1) an ok move + plane so a continuation write lands.
        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        stack[ss - 1].current_move =
            Move::make(Square::new(0, 6).unwrap(), Square::new(0, 5).unwrap(), bp);
        stack[ss - 1].cont_hist = 3;

        // The quiet move under test: black pawn 5e->5d (quiet, no capture).
        let mv = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), bp);
        let bonus = 800;

        let us = pos.side_to_move();
        let pk = pos.pawn_key();
        let moved = mv.moved_piece_after();
        let to = mv.to_sq();

        let main_pre = hist.main.get(us, mv) as i16;
        let low_pre = hist.low_ply.get(2, mv) as i16;
        let pawn_pre = hist.shared.pawn_get(pk, moved, to) as i16;
        let cont_pre = hist.continuation.get_at(3, moved, to) as i16;

        update_quiet_histories(&mut hist, &pos, &stack, ss, mv, bonus);

        assert_eq!(
            hist.main.get(us, mv),
            apply_gravity(main_pre, bonus, MAIN_HISTORY_D) as i32,
        );
        assert_eq!(
            hist.low_ply.get(2, mv),
            apply_gravity(low_pre, bonus * 761 / 1024, MAIN_HISTORY_D) as i32,
        );
        // continuation write uses bonus*955/1024 (+88 since i==1) at plane 3.
        let cont_write = (bonus * 955 / 1024 * CONTHIST_BONUSES[0].1 / 1024) + 88;
        assert_eq!(
            hist.continuation.get_at(3, moved, to),
            apply_gravity(cont_pre, cont_write, CONTINUATION_HISTORY_D) as i32,
        );
        // pawn write: bonus>0 ⇒ *850/1024.
        assert_eq!(
            hist.shared.pawn_get(pk, moved, to),
            apply_gravity(pawn_pre, bonus * 850 / 1024, PAWN_HISTORY_D) as i32,
        );
    }

    #[test]
    fn quiet_histories_negative_bonus_uses_550_pawn_scale() {
        let mut hist = WorkerHistories::new();
        let pos = simple_pos();
        let ss = 7usize;
        let stack = fresh_stack(ss + 1);
        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        let mv = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), bp);
        let bonus = -800;
        let pk = pos.pawn_key();
        let moved = mv.moved_piece_after();
        let to = mv.to_sq();
        let pawn_pre = hist.shared.pawn_get(pk, moved, to) as i16;
        update_quiet_histories(&mut hist, &pos, &stack, ss, mv, bonus);
        assert_eq!(
            hist.shared.pawn_get(pk, moved, to),
            apply_gravity(pawn_pre, bonus * 550 / 1024, PAWN_HISTORY_D) as i32,
        );
    }

    // ---- update_all_stats -------------------------------------------------

    #[test]
    fn all_stats_capture_best_updates_capture_history() {
        // Black rook on 5e captures a white pawn on 5d. bestMove is a capture,
        // so the capture branch runs: captureHistory[rook][5d][pawn] gets the
        // +bonus*1286/1024 bump; no quiet updates.
        let pos = parse_sfen("4k4/9/9/4p4/4R4/9/9/9/4K4 b - 1").unwrap();
        let mut hist = WorkerHistories::new();
        let ss = 8usize;
        let stack = fresh_stack(ss + 1);

        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let from = Square::new(4, 4).unwrap();
        let to = Square::new(4, 3).unwrap();
        let best = Move::make(from, to, rook);
        let captured = pos.board().get(to).unwrap();
        assert_eq!(captured.kind, PieceKind::Pawn);

        let depth = 6;
        // bestMove != ttMove here (ttMove is none), so the +353 term is 0.
        let bonus = (128 * depth - 77).min(1529) + stack[ss - 1].stat_score / 32;
        let pre = hist.capture.get(rook, to, captured) as i16;

        update_all_stats(
            &mut hist,
            &pos,
            &stack,
            ss,
            best,
            None,
            &[],
            &[],
            depth,
            Move::none(),
            false,
        );

        assert_eq!(
            hist.capture.get(rook, to, captured),
            apply_gravity(pre, bonus * 1286 / 1024, CAPTURE_HISTORY_D) as i32,
        );
    }

    #[test]
    fn all_stats_quiet_best_bumps_best_and_decays_others() {
        // bestMove is a quiet pawn push; one other quiet was searched. The best
        // gets +bonus*806/1024; the other gets the decayed negated malus.
        let pos = simple_pos();
        let mut hist = WorkerHistories::new();
        let ss = 8usize;
        let stack = fresh_stack(ss + 1);

        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        let best = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), bp);
        // Another quiet: black king step 5i->4i.
        let bk = Piece::new(PieceKind::King, Color::Black);
        let other = Move::make(Square::new(4, 8).unwrap(), Square::new(3, 8).unwrap(), bk);

        let depth = 5;
        let tt_move = best; // bestMove == ttMove ⇒ +353 in the bonus.
        let bonus = (128 * depth - 77).min(1529) + 353 + stack[ss - 1].stat_score / 32;
        let malus = (882 * depth - 204).min(2122);

        let us = pos.side_to_move();
        let best_main_pre = hist.main.get(us, best) as i16;
        let other_main_pre = hist.main.get(us, other) as i16;

        update_all_stats(
            &mut hist,
            &pos,
            &stack,
            ss,
            best,
            None,
            &[other],
            &[],
            depth,
            tt_move,
            false,
        );

        // Best move mainHistory: +bonus*806/1024.
        assert_eq!(
            hist.main.get(us, best),
            apply_gravity(best_main_pre, bonus * 806 / 1024, MAIN_HISTORY_D) as i32,
        );
        // The single non-best quiet: actualMalus starts malus*1113/1024, then
        // is decayed *977/1024 once, applied negated.
        let actual_malus = (malus * 1113 / 1024) * 977 / 1024;
        assert_eq!(
            hist.main.get(us, other),
            apply_gravity(other_main_pre, -actual_malus, MAIN_HISTORY_D) as i32,
        );
    }

    #[test]
    fn all_stats_non_best_capture_gets_malus() {
        // A non-best capture in capturesSearched gets captureHistory
        // -malus*1559/1024. Best is a quiet so the quiet branch runs for it.
        let pos = parse_sfen("4k4/9/9/4p4/3RR4/9/9/9/4K4 b - 1").unwrap();
        let mut hist = WorkerHistories::new();
        let ss = 8usize;
        let stack = fresh_stack(ss + 1);

        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        // Best: a quiet king move so the quiet branch runs.
        let bk = Piece::new(PieceKind::King, Color::Black);
        let best = Move::make(Square::new(4, 8).unwrap(), Square::new(3, 8).unwrap(), bk);
        let _ = bp;

        // A non-best capture: rook 5e->5d takes the white pawn.
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let cto = Square::new(4, 3).unwrap();
        let cap = Move::make(Square::new(4, 4).unwrap(), cto, rook);
        let captured = pos.board().get(cto).unwrap();

        let depth = 4;
        let malus = (882 * depth - 204).min(2122);
        let pre = hist.capture.get(rook, cto, captured) as i16;

        update_all_stats(
            &mut hist,
            &pos,
            &stack,
            ss,
            best,
            None,
            &[],
            &[cap],
            depth,
            Move::none(),
            false,
        );

        assert_eq!(
            hist.capture.get(rook, cto, captured),
            apply_gravity(pre, -malus * 1559 / 1024, CAPTURE_HISTORY_D) as i32,
        );
    }

    #[test]
    fn all_stats_refutation_penalty_hits_prev_ply_continuation() {
        // prevSq real, (ss-1).moveCount == 1 + ttHit, !priorCapture ⇒ the
        // previous ply's continuation plane is penalised at [piece_on(prevSq)]
        // [prevSq] with -malus*616/1024.
        let pos = simple_pos();
        let mut hist = WorkerHistories::new();
        let ss = 8usize;
        let mut stack = fresh_stack(ss + 1);

        // prevSq = 5e, occupied by the black pawn.
        let prev_sq = Square::new(4, 4).unwrap();
        let prev_piece = pos.board().get(prev_sq).unwrap();

        // (ss-1) is the refuted ply: moveCount 1, ttHit false ⇒ 1 == 1+0.
        stack[ss - 1].move_count = 1;
        stack[ss - 1].tt_hit = false;
        // update_continuation_histories(ss-1, ...) reads (ss-1).in_check and
        // (ss-1-i).currentMove/cont_hist. Give (ss-2) an ok move + plane.
        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        stack[ss - 2].current_move =
            Move::make(Square::new(0, 6).unwrap(), Square::new(0, 5).unwrap(), bp);
        stack[ss - 2].cont_hist = 5;

        // Best is a quiet so the quiet branch runs; depth for malus.
        let bk = Piece::new(PieceKind::King, Color::Black);
        let best = Move::make(Square::new(4, 8).unwrap(), Square::new(3, 8).unwrap(), bk);
        let depth = 4;
        let malus = (882 * depth - 204).min(2122);

        // (ss-1) continuation write: pc=prev_piece, to=prev_sq, plane from
        // (ss-2).cont_hist (=5), i==1 weight 1157, +88.
        let pre = hist.continuation.get_at(5, prev_piece, prev_sq) as i16;

        update_all_stats(
            &mut hist,
            &pos,
            &stack,
            ss,
            best,
            Some(prev_sq),
            &[],
            &[],
            depth,
            Move::none(),
            false,
        );

        let write = (-malus * 616 / 1024 * CONTHIST_BONUSES[0].1 / 1024) + 88;
        assert_eq!(
            hist.continuation.get_at(5, prev_piece, prev_sq),
            apply_gravity(pre, write, CONTINUATION_HISTORY_D) as i32,
        );
    }

    // ---- update_correction_history ----------------------------------------

    #[test]
    fn correction_history_hand_computed() {
        // A position with a black pawn already on 5d, so that (ss-1)'s move
        // landing there has `piece_on(to) == Some` (the continuation-correction
        // reads it). Parsed via SFEN so the partial keys are consistent.
        let pos2 = parse_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1").unwrap();
        let mut hist = WorkerHistories::new();
        let ss = 7usize;
        let mut stack = fresh_stack(ss + 1);

        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        let to = Square::new(4, 3).unwrap(); // 5d
        assert_eq!(
            pos2.board().get(to),
            Some(bp),
            "fixture places a pawn on 5d"
        );
        stack[ss - 1].current_move = Move::make(Square::new(4, 4).unwrap(), to, bp);
        stack[ss - 2].cont_corr = ContinuationCorrectionHistory::plane_index(bp, to);
        stack[ss - 4].cont_corr = ContinuationCorrectionHistory::plane_index(bp, to);

        let us = pos2.side_to_move();
        let bonus = 200;

        let pawn_pre = hist
            .shared
            .correction_get(pos2.pawn_key(), us, CorrChannel::Pawn) as i16;
        let minor_pre =
            hist.shared
                .correction_get(pos2.minor_piece_key(), us, CorrChannel::Minor) as i16;
        let wnp_pre = hist.shared.correction_get(
            pos2.non_pawn_key(Color::White),
            us,
            CorrChannel::NonPawnWhite,
        ) as i16;
        let bnp_pre = hist.shared.correction_get(
            pos2.non_pawn_key(Color::Black),
            us,
            CorrChannel::NonPawnBlack,
        ) as i16;
        let plane = ContinuationCorrectionHistory::plane_index(bp, to);
        let cc_pre = hist.continuation_correction.get_at(plane, bp, to) as i16;

        update_correction_history(&mut hist, &pos2, &stack, ss, bonus);

        assert_eq!(
            hist.shared
                .correction_get(pos2.pawn_key(), us, CorrChannel::Pawn),
            apply_gravity(pawn_pre, bonus, CORRECTION_HISTORY_D) as i32,
        );
        assert_eq!(
            hist.shared
                .correction_get(pos2.minor_piece_key(), us, CorrChannel::Minor),
            apply_gravity(minor_pre, bonus * 153 / 128, CORRECTION_HISTORY_D) as i32,
        );
        assert_eq!(
            hist.shared.correction_get(
                pos2.non_pawn_key(Color::White),
                us,
                CorrChannel::NonPawnWhite
            ),
            apply_gravity(wnp_pre, bonus * 187 / 128, CORRECTION_HISTORY_D) as i32,
        );
        assert_eq!(
            hist.shared.correction_get(
                pos2.non_pawn_key(Color::Black),
                us,
                CorrChannel::NonPawnBlack
            ),
            apply_gravity(bnp_pre, bonus * 187 / 128, CORRECTION_HISTORY_D) as i32,
        );
        // The two continuation-correction planes are the same plane here, so the
        // 126/128 write then the 63/128 write compose.
        let after_126 = apply_gravity(cc_pre, bonus * 126 / 128, CORRECTION_HISTORY_D);
        let after_63 = apply_gravity(after_126, bonus * 63 / 128, CORRECTION_HISTORY_D);
        assert_eq!(
            hist.continuation_correction.get_at(plane, bp, to),
            after_63 as i32,
        );
    }

    #[test]
    fn correction_history_skips_continuation_when_prev_move_not_ok() {
        // With (ss-1).currentMove not ok, only the four channel writes fire.
        let pos = simple_pos();
        let mut hist = WorkerHistories::new();
        let ss = 7usize;
        let stack = fresh_stack(ss + 1);
        let us = pos.side_to_move();
        let bonus = 128;
        let pawn_pre = hist
            .shared
            .correction_get(pos.pawn_key(), us, CorrChannel::Pawn) as i16;

        update_correction_history(&mut hist, &pos, &stack, ss, bonus);

        assert_eq!(
            hist.shared
                .correction_get(pos.pawn_key(), us, CorrChannel::Pawn),
            apply_gravity(pawn_pre, bonus, CORRECTION_HISTORY_D) as i32,
        );
        // The sentinel continuation-correction plane is untouched (fill 6).
        let sentinel = ContinuationCorrectionHistory::SENTINEL_PLANE;
        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        assert_eq!(
            hist.continuation_correction
                .get_at(sentinel, bp, Square::new(4, 3).unwrap()),
            CONTINUATION_CORRECTION_INIT as i32,
        );
    }
}
