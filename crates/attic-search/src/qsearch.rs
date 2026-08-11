//! Quiescence search — a faithful port of the reference
//! `Search::YaneuraOuWorker::qsearch`
//! (`source/engine/yaneuraou-engine/yaneuraou-search.cpp`,
//! lines 4432–5136 at the current submodule pin).
//!
//! Every structural detail here is derived from the **code** at that pin, not
//! from the stale design-note comment block at the top of the reference
//! function (which describes an older `DEPTH_QS_CHECKS` / `DEPTH_QS_RECAPTURES`
//! design that no longer exists — see the movepick module docs). Line numbers
//! in the comments below point into that reference file.
//!
//! # Shape
//!
//! The reference signature is `template<NodeType, bool ReadTT> Value
//! qsearch(Position&, Stack*, Value alpha, Value beta)` — no depth parameter.
//! Both `PvNode` and `ReadTT` are invariant down the whole recursion (qsearch
//! always recurses with the *same* node type, unlike the main search), so they
//! are stored once on [`QSearch`] rather than threaded through every call.
//!
//! # Evaluation
//!
//! Like the reference, this port maintains the NNUE accumulator **incrementally**
//! across `do_move` / `undo_move`. Each worker owns a private
//! accumulator stack ([`QSearch::acc_stack`]): the root position is refreshed
//! once, and every `do_move` derives the child accumulator from its parent by
//! rewriting only the feature columns the move touches (the move-derived
//! [`attic_eval::MoveDelta`]; a perspective whose own king moved is refreshed
//! from scratch), pushing it for the child and popping it on `undo_move`. A null
//! move touches no pieces, so the child reuses the parent slot unchanged. The
//! six evaluation sites then read the live accumulator through
//! [`attic_eval::evaluate_with`] — no per-node full refresh. `attic-eval`
//! guarantees the differential accumulator is **bit-identical** to a
//! from-scratch refresh, so the returned static evals are exactly the
//! reference's; a debug-only assertion in [`QSearch::static_eval`] re-checks
//! that identity against [`attic_eval::evaluate`] at every evaluation point.
//! Eager derivation on every `do_move` (rather than the reference's lazy
//! refresh-on-first-evaluation) is behaviourally identical — the accumulator a
//! node evaluates is the same either way — and simply skips the bookkeeping for
//! deferring it.
//!
//! # History tables
//!
//! qsearch reads **one** set of live worker history tables ([`WorkerHistories`])
//! — the same tables the interior [`QSearch::search`] updates — for its move
//! ordering (leaf [`MovePicker`]), its evasion continuation-plane scoring, and
//! its [`QSearch::correction_value`]. There are no qsearch-private history
//! duplicates: once the interior search updates `self.histories`, every
//! subsequent leaf qsearch sees those updates, exactly as the reference reads
//! the worker's tables everywhere. qsearch itself never *writes*
//! the tables (no history update occurs inside a pure-qsearch subtree), so with
//! untouched tables move ordering is pure MVV for captures and the capture-bias
//! for evasions, and `correction_value` is eval-neutral.
//!
//! # Transposition table
//!
//! The reference holds a single `TTWriter` from the Step-3 probe and writes
//! through it — including the *tail* write that happens after the whole move
//! loop. In safe Rust a writer borrowing the table cannot outlive the recursive
//! calls that also mutate the table, so instead the node captures the entry's
//! *location* ([`attic_storage::TtSlot`]) at the Step-3 [`TranspositionTable::locate`]
//! probe and every write site writes to that exact slot via
//! [`QSearch::tt_store`]. Because the slot is an index pair (not a re-probe), the
//! tail write lands on the reference's held-writer entry even after a child has
//! churned the cluster — a re-probe there would re-run the replacement selection
//! against the churned cluster and could pick a different slot, drifting the
//! stored TT state and the node counts of later probes (the depth-3 parity gate
//! is sensitive to exactly that).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use attic_eval::{Accumulator, FinnyCache, MoveDelta, NnueNetwork, evaluate_with};
use attic_state::{Color, Move, Piece, PieceKind, Position, RepetitionState, piece_value};
use attic_storage::{Bound, TranspositionTable, TtSlot, Value};

use crate::history::{ContinuationCorrectionHistory, ContinuationHistory, CorrChannel};
use crate::movepick::MovePicker;
use crate::root::{
    EnteringKingConfig, RootKind, RootMove, RootOutcome, declaration_win, generate_root_moves,
};
use crate::timeman::TimeManagement;
use crate::update::{
    SEARCHED_LIST_CAPACITY, SearchStackCell, SearchedList, WorkerHistories, update_all_stats,
    update_continuation_histories, update_correction_history, update_quiet_histories,
};

// -----------------------------------------------------------------------------
// Value / depth constants and helpers (source/types.h, source/config.h).
// -----------------------------------------------------------------------------

/// `MAX_PLY` (`config.h` → `types.h`): the standard engine build value.
const MAX_PLY: i32 = 246;
/// `VALUE_INFINITE` (`types.h`).
const VALUE_INFINITE: Value = 32001;
/// `VALUE_NONE` (`types.h`).
const VALUE_NONE: Value = 32002;
/// `VALUE_MATE` (`types.h`).
const VALUE_MATE: Value = 32000;
/// `VALUE_MATE_IN_MAX_PLY` == `VALUE_TB_WIN_IN_MAX_PLY` (`types.h`).
const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY; // 31754
/// `VALUE_MAX_EVAL` == `VALUE_SUPERIOR` (`types.h`).
const VALUE_MAX_EVAL: Value = VALUE_TB_WIN_IN_MAX_PLY - 1; // 31753
/// `VALUE_DRAW` (`types.h`).
const VALUE_DRAW: Value = 0;
/// `DEPTH_QS` (`types.h`).
const DEPTH_QS: i32 = 0;
/// `DEPTH_UNSEARCHED` (`types.h`).
const DEPTH_UNSEARCHED: i32 = -2;
/// Futility margin added to `ss->staticEval` (`yaneuraou-search.cpp`).
const FUTILITY_MARGIN: Value = 328;
/// SEE cutoff for a capture with no futility exemption
/// (`yaneuraou-search.cpp`).
const SEE_CAPTURE_MARGIN: i32 = -73;
/// The default-remapped `MaxMovesToDraw` (`yaneuraou-search.cpp`):
/// the `0` option default is rewritten to `100000`. This is the default
/// [`QSearch::max_moves_to_draw`] value, i.e. the fixed-depth parity path — the
/// driver overrides it per `go` from the `MaxMovesToDraw` option.
const MAX_MOVES_TO_DRAW: i32 = 100_000;

/// `mate_in(ply)` (`types.h`).
fn mate_in(ply: i32) -> Value {
    VALUE_MATE - ply
}

/// `mated_in(ply)` (`types.h`).
fn mated_in(ply: i32) -> Value {
    -VALUE_MATE + ply
}

/// `is_valid(v)` (`types.h`).
fn is_valid(v: Value) -> bool {
    v != VALUE_NONE
}

/// `is_win(v)` (`types.h`).
fn is_win(v: Value) -> bool {
    v >= VALUE_TB_WIN_IN_MAX_PLY
}

/// `is_loss(v)` (`types.h`).
fn is_loss(v: Value) -> bool {
    v <= -VALUE_TB_WIN_IN_MAX_PLY
}

/// `is_decisive(v)` (`types.h`).
fn is_decisive(v: Value) -> bool {
    is_win(v) || is_loss(v)
}

/// `value_to_tt(v, ply)` (`yaneuraou-search.cpp`): shift a mate score away
/// from the root before storing.
fn value_to_tt(v: Value, ply: i32) -> Value {
    if is_win(v) {
        v + ply
    } else if is_loss(v) {
        v - ply
    } else {
        v
    }
}

/// `value_from_tt(v, ply)` (`yaneuraou-search.cpp`, non-`STOCKFISH`
/// variant — no rule50 downgrade): shift a stored mate score back toward the
/// root.
fn value_from_tt(v: Value, ply: i32) -> Value {
    if !is_valid(v) {
        VALUE_NONE
    } else if is_win(v) {
        v - ply
    } else if is_loss(v) {
        v + ply
    } else {
        v
    }
}

/// `value_draw(nodes)` (`yaneuraou-search.cpp`): a ±1 dither keyed on bit 1
/// of the node counter, `VALUE_DRAW - 1 + (nodes & 0x2)`. Deterministic given
/// the counter.
fn value_draw(nodes: u64) -> Value {
    VALUE_DRAW - 1 + (nodes & 0x2) as Value
}

/// `RootMove::operator<` (`search.h`) as a stable-sort comparator: descending
/// by `score`, then descending by `previousScore`. Used for both the active-tail
/// re-sort (`yaneuraou-search.cpp`) and the finished-head sort (`1791`).
fn root_move_order(a: &RootMove, b: &RootMove) -> std::cmp::Ordering {
    if a.score != b.score {
        b.score.cmp(&a.score)
    } else {
        b.previous_score.cmp(&a.previous_score)
    }
}

/// The reference fail-high/low PV-output gate (`yaneuraou-search.cpp`),
/// factored out as a pure predicate so it can be unit-tested independently. The
/// pin gates the print on `mainThread && multiPV == 1 && (bestValue <= alpha ||
/// bestValue >= beta) && nodes > 10_000_000 && (rootDepth < 3 || interval
/// elapsed) && OutputFailLHPV`.
///
/// The `nodes > 10_000_000` conjunct and the `rootDepth < 3` disjunct are easy
/// to drop when reading the condition informally; both are in the pin, and both
/// are reproduced here.
#[allow(clippy::too_many_arguments)]
pub fn fail_lh_pv_gate(
    main_thread: bool,
    multi_pv: usize,
    best_value: Value,
    alpha: Value,
    beta: Value,
    nodes: u64,
    root_depth: i32,
    interval_elapsed: bool,
    output_fail_lh_pv: bool,
) -> bool {
    main_thread
        && multi_pv == 1
        && (best_value <= alpha || best_value >= beta)
        && nodes > 10_000_000
        && (root_depth < 3 || interval_elapsed)
        && output_fail_lh_pv
}

/// `to_corrected_static_eval(v, cv)` (`yaneuraou-search.cpp`):
/// `clamp(v + cv/131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)`.
/// With zero-filled correction tables `cv == 0` and this degenerates to a plain
/// clamp; the arithmetic and clamp shape are the pin's regardless.
fn to_corrected_static_eval(v: Value, cv: i32) -> Value {
    (v + cv / 131072).clamp(-VALUE_MAX_EVAL, VALUE_MAX_EVAL)
}

/// The low-16-bit move fragment stored in the TT (`Move16`): the reference
/// stores `Move::to_move16()`, which is the low 16 bits of the packed move.
fn move16_of(m: Move) -> u16 {
    (m.to_bits() & 0xFFFF) as u16
}

/// `ttData.bound & (want_lower ? BOUND_LOWER : BOUND_UPPER)` as a bool
/// (`BOUND_LOWER == 2`, `BOUND_UPPER == 1`, `BOUND_EXACT == 3`).
fn bound_matches(bound: Bound, want_lower: bool) -> bool {
    let mask = if want_lower {
        Bound::Lower as u8
    } else {
        Bound::Upper as u8
    };
    (bound as u8 & mask) != 0
}

// -----------------------------------------------------------------------------
// Search stack.
// -----------------------------------------------------------------------------

/// The number of sentinel entries before ply 0 (the reference uses `ss =
/// stack + 7`). Only `(ss - 1)` is read in this port (`prevSq`, `contHist`);
/// the deeper `(ss - 2)` / `(ss - 4)` reads live inside `correction_value`,
/// which is identically zero here. The extra sentinels are kept so the stack
/// layout matches the reference's.
const STACK_BASE: usize = 7;

/// Length of the fixed-size search stack. The reference declares `Stack
/// stack[MAX_PLY + 10] = {}` and walks `Stack* ss = stack + 7`
/// (`yaneuraou-search.cpp`, both iterative-deepening sites); we mirror the
/// shape with `STACK_BASE` (7) leading sentinels, `MAX_PLY` live plies, plus
/// two trailing cells so the deepest node's `(ss+2)->cutoffCnt` write is in
/// range. Being a compile-time constant (not a `Vec` length) lets the
/// optimizer range-analyze `si(ply)` indexes and drop the per-access bounds
/// checks that dominated the hot search function's panic paths.
const STACK_LEN: usize = STACK_BASE + MAX_PLY as usize + 2;

/// Length of the fixed-size differential-NNUE accumulator stack: one slot per
/// reachable do/undo depth (bounded by `MAX_PLY`) plus headroom. A boxed
/// fixed-size array keeps the (large) `Accumulator` slots on the heap while
/// making the length a compile-time constant, exactly as for [`STACK_LEN`].
const ACC_LEN: usize = MAX_PLY as usize + 8;

// The search-stack cell is [`SearchStackCell`] (defined in `crate::update`), the
// same cell the interior main search and its history-update machinery operate
// on. qsearch and the depth-1 root read only a handful of its fields —
// `current_move` (the child reads it as `(ss-1)->currentMove` for `prevSq`),
// `tt_pv` (`ss->ttPv`, read for the mate-in-1 TT write), `pv` (the parent reads
// the child's `(ss+1)->pv`), and `cont_corr` (the `(ss-2)`/`(ss-4)` planes
// `correction_value` reads). Sharing one cell type keeps qsearch, root, and the
// interior `search()` on a single stack, exactly as the reference does.

// -----------------------------------------------------------------------------
// QSearch context.
// -----------------------------------------------------------------------------

/// Outcome of a top-level [`QSearch::run`].
#[derive(Debug, Clone)]
pub struct QSearchOutcome {
    /// Search value from the root side-to-move's point of view.
    pub value: Value,
    /// Number of `do_move` calls made — the reference's node-count semantics
    /// (`yaneuraou-search.cpp`, the sole increment site).
    pub nodes: u64,
    /// Principal variation collected at the root (empty for a non-PV run or a
    /// fail-low).
    pub pv: Vec<Move>,
    /// Maximum selective depth reached (`selDepth`), counting from 1.
    pub sel_depth: i32,
}

/// How many `check_time` calls elapse between two real clock / node / flag
/// checks — the reference `SearchManager::callsCnt` reset value
/// (`yaneuraou-search.cpp`, the non-`nodes` case). Reading `Instant::now()`
/// on every node is too costly, so the deadline and the atomic stop flag are
/// only consulted once per this many `check_time` calls. The counter is seeded
/// to this value at search start (rather than the reference's `0`), so the
/// **first** checkpoint lands after a full interval: a short fixed-depth search
/// (e.g. the 30-node depth-1 startpos gate) never reaches a checkpoint, so an
/// asynchronously-set stop flag cannot perturb it — the fixed-depth parity path
/// stays bit-identical.
const CHECK_INTERVAL: i32 = 512;

/// The shared ponder state for one `go ponder` — the reference `SearchManager`'s
/// `ponder` atomic (`yaneuraou-search.h`) plus the `tm.ponderhitTime` the
/// `set_ponderhit` path stamps (`yaneuraou-search.cpp`). Held behind an
/// [`Arc`] so the USI driver thread can clear it (a `ponderhit`) while the search
/// worker polls it.
///
/// A pondering search never self-terminates: `check_time` returns before any stop
/// decision while [`Self::is_active`] holds, and the iterative-deepening budget
/// block sets `stopOnPonderhit` instead of the end time. Only `stop` (the shared
/// abort flag) or a `ponderhit` ends it.
pub struct PonderSignal {
    /// `SearchManager::ponder` — true while `go ponder` is pondering; cleared by
    /// [`Self::ponderhit`].
    active: AtomicBool,
    /// The instant a `ponderhit` arrived (`tm.ponderhitTime = now()`), stamped
    /// **before** the flag is cleared so a worker that observes `active == false`
    /// always sees the time — the reference ordering (`yaneuraou-search.cpp`).
    hit_at: Mutex<Option<Instant>>,
}

impl PonderSignal {
    /// A fresh signal, `active` seeded from `limits.ponderMode`
    /// (`pre_start_searching`, `yaneuraou-search.cpp`).
    pub fn new(active: bool) -> Self {
        PonderSignal {
            active: AtomicBool::new(active),
            hit_at: Mutex::new(None),
        }
    }

    /// Whether the search is still pondering.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// `set_ponderhit(false)` (`yaneuraou-search.cpp`): stamp the ponderhit
    /// instant, then clear the flag. The order matters — `check_time` /
    /// `set_search_end` read `ponderhitTime` after seeing `ponder == false`.
    pub fn ponderhit(&self) {
        *self.hit_at.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        self.active.store(false, Ordering::Release);
    }

    /// The stamped ponderhit instant, if a `ponderhit` has arrived.
    fn hit_at(&self) -> Option<Instant> {
        *self.hit_at.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Time / node / stop controls for one search, mapped by the USI driver from the
/// `go` limits. Every field absent (the [`Default`]) is the fixed-depth parity
/// path: no early termination, so [`QSearch::run_root`] runs to its `limit_depth`
/// with a search that is bit-identical to a build compiled without any of this
/// machinery (the flag is only ever *read* at a checkpoint, and only ever
/// *acted on* when a limit is actually configured).
#[derive(Clone, Default)]
pub struct SearchControl {
    /// Shared abort flag, set asynchronously by the driver on `stop` / `quit`
    /// and polled at the [`CHECK_INTERVAL`] granularity. `None` (the default)
    /// means no external stop is possible.
    pub stop: Option<Arc<AtomicBool>>,
    /// The shared `go ponder` state (`Some` only on the main worker of a `go
    /// ponder`). While it is active the search never self-terminates (only `stop`
    /// or a `ponderhit` ends it); a `ponderhit` clears it and stamps the ponderhit
    /// time so time management resumes. `None` (the default) on every non-ponder
    /// search and on every helper, leaving the parity path bit-identical.
    pub ponder: Option<Arc<PonderSignal>>,
    /// Hard node ceiling (`go nodes N`): abort once the node counter reaches it.
    pub node_limit: Option<u64>,
    /// The reference `TimeManagement` state plus the limit classification the
    /// search-side time control needs (`Some` only on the main worker of a `go`
    /// that has a time budget — `go movetime`, `go rtime`, or a real clock).
    /// `None` on helpers, on fixed-depth / nodes / infinite / mate `go`s, and on
    /// the fixed-depth parity path, where the whole time-management machinery is
    /// inert and the search is bit-identical to a build without it.
    pub time: Option<TimeControl>,
}

/// The main worker's time-management state for one `go`: the reference
/// `TimeManagement` (mutated in place — its `search_end` is written by the
/// iterative-deepening budget block and `check_time`) plus the pieces of the
/// reference `LimitsType` / `MainManager` the search-side control consults.
#[derive(Clone)]
pub struct TimeControl {
    /// The reference `TimeManagement` for this `go`.
    pub tm: TimeManagement,
    /// `limits.use_time_management()` (`search.h`): true only for a real
    /// clock / `go rtime`, i.e. not `movetime` / `depth` / `nodes` / `infinite` /
    /// `mate`. Gates the dynamic optimum-time block and the `maximum()` stop.
    pub use_time_management: bool,
    /// `limits.movetime` [ms] (`Some` only for `go movetime`): `check_time` stops
    /// the search once `elapsed >= movetime`.
    pub movetime: Option<i64>,
    /// `threads.size()` — the worker count, the divisor of the best-move
    /// instability factor (`yaneuraou-search.cpp`).
    pub n_threads: usize,
    /// `main_manager()->bestPreviousScore` — the previous `go`'s reported score
    /// (`VALUE_INFINITE` for the first move of a game), used to seed `iterValue`
    /// (`yaneuraou-search.cpp`).
    pub best_previous_score: Value,
    /// `main_manager()->bestPreviousAverageScore` — the previous `go`'s reported
    /// average score, used in `fallingEval` (`yaneuraou-search.cpp`).
    pub best_previous_average_score: Value,
    /// `main_manager()->previousTimeReduction` — the previous `go`'s final
    /// `timeReduction` (`0.85` for the first move), used in `reduction`
    /// (`yaneuraou-search.cpp`).
    pub previous_time_reduction: f64,
}

/// The search driver owning the network reference, the transposition table, the
/// search stack, and the one live set of worker history tables shared by the
/// root search, the interior `search`, and the leaf qsearch.
///
/// Despite the name, this type drives the whole `go depth 1..3` search:
/// [`Self::run_root`] runs iterative deepening at the root, [`Self::search`] is
/// the shared interior body, and [`Self::qsearch`] is the leaf quiescence search
/// they recurse into.
///
/// The transposition table must be sized (`resize`) before [`QSearch::run`];
/// [`TranspositionTable::probe`] panics on an unsized table.
pub struct QSearch<'a> {
    net: &'a NnueNetwork,
    /// The shared transposition table, borrowed `&self`: the table
    /// lives behind an `Arc` in the driver and every probe / write goes through
    /// its atomics, so the driver can hand each worker a cheap `Arc` clone.
    tt: &'a TranspositionTable,

    /// `nodes` counter — bumped once per `do_move` (`yaneuraou-search.cpp`).
    nodes: u64,
    /// `selDepth` (`yaneuraou-search.cpp`).
    sel_depth: i32,
    /// `nmpMinPly` (`yaneuraou-search.h`): while a null-move verification
    /// search is running, the ply below which null-move pruning stays disabled.
    /// Zero means "no verification in flight", which is also the value between
    /// searches — the reference's per-`go` reset (`yaneuraou-search.cpp`)
    /// sits inside `#if STOCKFISH` and is therefore dead in the live build, so
    /// the field is only ever seeded once at construction and restored to 0 by
    /// the verification block itself (Step 9).
    nmp_min_ply: i32,

    /// Whether this run is a PV search (`nodeType == PV`). Invariant down the
    /// recursion.
    pv_node: bool,
    /// Whether TT hits are honoured (`ReadTT`). Invariant down the recursion.
    read_tt: bool,

    /// Root side-to-move, used by [`Self::draw_value`] to reproduce the
    /// contempt-signed `drawValueTable[REPETITION_DRAW]` (set once per search
    /// from the root side, `yaneuraou-search.cpp`).
    root_us: Color,
    /// `drawValueTable[REPETITION_DRAW][root_us]`, i.e. the contempt draw score
    /// for the root side. With default options (`DrawValueBlack/White = -2`,
    /// `types.cpp` / `yaneuraou-search.cpp`) and `PawnValue = 90` this
    /// is `-2 * 90 / 100 == -1` (C++ truncation toward zero); the opponent side
    /// gets `-draw_contempt`.
    draw_contempt: Value,

    /// The persistent search stack (`STACK_BASE` sentinels + `MAX_PLY` + 1),
    /// a fixed-size boxed array (length [`STACK_LEN`]) mirroring the pin's
    /// `Stack stack[MAX_PLY + 10]`. The constant length lets the optimizer
    /// prove `si(ply)` indexes in range and elide bounds checks in the hot
    /// search function.
    stack: Box<[SearchStackCell; STACK_LEN]>,

    /// This worker's private differential-NNUE accumulator stack.
    /// `acc_stack[acc_depth]` is always the accumulator for the position the
    /// current node is searching: seeded once with a full refresh at the root,
    /// then derived one child at a time from its parent on every `do_move`
    /// (pushed) and discarded on `undo_move` (popped). A null move touches no
    /// pieces, so the child reuses the parent slot without a push. Each Lazy-SMP
    /// worker owns its own stack — never shared — so the six search evaluation
    /// sites read it through [`attic_eval::evaluate_with`] instead of rebuilding
    /// the accumulator from scratch. Preallocated so a node never allocates.
    /// A fixed-size boxed array (length [`ACC_LEN`]); the constant length lets
    /// the optimizer drop the per-access bounds check.
    acc_stack: Box<[Accumulator; ACC_LEN]>,
    /// Index of the current node's accumulator within [`Self::acc_stack`] (the
    /// live top of the do/undo stack). Incremented per `do_move`, decremented per
    /// `undo_move`; `0` at every root node.
    acc_depth: usize,
    /// This worker's private finny table: one cached refreshed
    /// accumulator half per (perspective, own-king square), so a `do_move` whose
    /// own king moved diffs against that cached half instead of summing all 40
    /// feature columns from the FT biases. Value-invariant — see
    /// [`Accumulator::derive_into_cached`]. Allocated once per worker (~0.5 MiB)
    /// and never touched off the `push_accumulator` path; never shared between
    /// Lazy-SMP workers, so it needs no synchronisation.
    finny: Box<FinnyCache>,
    /// Test-only: when set, [`Self::static_eval`] asserts the differential
    /// accumulator equals a from-scratch [`attic_eval::evaluate`] at every
    /// evaluation point (the accumulator-equivalence gate). Off by default, so
    /// production searches never invoke the refresh entry point; enabled via
    /// [`Self::set_verify_accumulator`] by the equivalence test.
    verify_accumulator: bool,

    /// The single set of live worker history tables read **everywhere** in the
    /// search tree — by the interior [`Self::search`] for move ordering (which
    /// also updates them in place: `mainHistory`, `captureHistory`,
    /// `continuationHistory`, `pawnHistory`, `lowPlyHistory`, the correction
    /// tables, `ttMoveHistory`), by the root picker, and by every leaf qsearch
    /// (its [`MovePicker`], its evasion continuation-plane scoring, and its
    /// [`Self::correction_value`]). One set of tables, so an interior update is
    /// visible to a later leaf qsearch — the reference contract.
    histories: WorkerHistories,
    /// `reductions[i] = int(2763/128.0 * ln(i))` for `i in 1..600`, `[0] == 0`
    /// (`yaneuraou-search.cpp`). Read by [`Self::reduction`].
    reductions: Vec<i32>,
    /// `rootDelta` — the width `beta - alpha` of the *root* aspiration window,
    /// read by [`Self::reduction`]. [`Self::run_root`] sets it before each
    /// `search<Root>` call; the default is the full-window width
    /// `2 * VALUE_INFINITE`.
    root_delta: Value,
    /// `rootDepth` — the current iterative-deepening depth, read by the Step-20
    /// nodes tie-break (`ss->ply + 2 >= rootDepth`). [`Self::run_root`] sets it
    /// per iteration; default `1`.
    root_depth: i32,
    /// `lastIterationPV` — the previous iteration's PV, consulted to seed
    /// `ss->followPV` from iteration 2 on. Cleared per `go`, then assigned at the
    /// end of each iterative-deepening iteration.
    last_iteration_pv: Vec<Move>,

    /// Time / node / stop controls. Empty by default (the
    /// fixed-depth parity path); the driver sets it via [`Self::set_control`]
    /// before an under-clock `go`.
    control: SearchControl,
    /// The entering-king declaration config for this `go`: the
    /// selected rule plus its precomputed per-side point thresholds, read by the
    /// two in-search `declaration_win` checks (`run_root`'s root shortcut and the
    /// interior Step-5 check). Defaults to `CSARule27` with the pre-option
    /// hardcoded thresholds, so the fixed-depth parity path is unchanged; the
    /// driver overrides it per `go` via [`Self::set_entering_king`].
    entering_king: EnteringKingConfig,
    /// The game ply past which the search adjudicates an unconditional draw
    /// (`search_options.max_moves_to_draw`, `yaneuraou-search.cpp` /
    /// `4616`). Already the `0 → 100000` remapped value: the driver passes the
    /// remapped `MaxMovesToDraw` option per `go` via [`Self::set_max_moves_to_draw`],
    /// so every worker (main + helpers) shares one snapshot. Defaults to the
    /// unlimited [`MAX_MOVES_TO_DRAW`] so the fixed-depth parity path is unchanged.
    max_moves_to_draw: i32,
    /// `generate_all_legal_moves` (`yaneuraou-search.cpp`): when true the
    /// search-facing move generators also yield the non-promoting moves the
    /// default generator suppresses (pawn/lance/knight non-promotions etc.). Set
    /// per `go` via [`Self::set_generate_all_legal_moves`]; `false` on the
    /// fixed-depth parity path, so generation stays bit-identical to today.
    generate_all_legal_moves: bool,
    /// `go mate` mode (`limits.mate != 0`). When set, the iterative-deepening
    /// early mate/mated break is disabled (the search keeps proving within its
    /// time budget, `yaneuraou-search.cpp`) and the mate-found stop rule
    /// (`1918-1923`) is armed so a proven mate terminates promptly. `false` on
    /// every non-mate `go`, so the parity path is unchanged. Set per `go` via
    /// [`Self::set_mate_mode`].
    mate_mode: bool,
    /// The reference `callsCnt` down-counter: `check_time` fires its real check
    /// once this reaches zero, then reloads it (see [`CHECK_INTERVAL`]).
    calls_cnt: i32,
    /// Latched abort state — set once a checkpoint observes the stop flag, the
    /// node ceiling, or the hard deadline. Every stop-check site early-returns on
    /// this; it stays `false` for a limit-free search, so those sites are inert.
    stopped: bool,
    /// `completedDepth` (`yaneuraou-search.cpp`), published so `check_time`
    /// can gate the time / node stops on "at least one iteration finished"
    /// (`5527`/`5532`). Reset to `0` per `go`; set to `rootDepth` at each
    /// completed iteration.
    completed_depth: i32,
    /// This worker's own `bestMoveChanges` (`yaneuraou-search.cpp`) since the
    /// last iteration — a `double` in the reference, only ever incremented by one,
    /// folded into `totBestMoveChanges` each iteration for time management. Used
    /// only on the single-worker [`Self::run_root`] path (no [`Self::best_move_tally`]);
    /// the Lazy-SMP path increments the shared per-worker slot instead so the main
    /// worker can sum every worker's count like the reference.
    best_move_changes: f64,
    /// `main_manager()->stopOnPonderhit` (`yaneuraou-search.cpp`). Without
    /// ponder it is never set true, but the writes are ported so the ponder path
    /// needs no search-side special case. Reset to `false` per `go`.
    stop_on_ponderhit: bool,
    /// Whether this worker has already copied the ponderhit instant out of the
    /// shared [`PonderSignal`] into `tm.ponderhitTime` (a one-time sync once a
    /// `ponderhit` clears the ponder flag). Reset to `false` per `go`; stays
    /// `false` on every non-ponder search, where the sync is a no-op.
    ponderhit_synced: bool,
    /// Lazy-SMP shared node counters — the slot vector for **all** workers plus
    /// this worker's index into it. Each `check_time` checkpoint
    /// publishes `self.nodes` to `slots[index]`, and the main worker's node
    /// ceiling sums the slots to reproduce the reference `threads.nodes_searched()`
    /// aggregate (`yaneuraou-search.cpp`). `None` on the direct single-worker
    /// [`Self::run_root`] path, where `self.nodes` is authoritative and the check
    /// is bit-identical to the pre-Lazy-SMP code.
    node_tally: Option<(Arc<Vec<AtomicU64>>, usize)>,
    /// Lazy-SMP shared best-move-change counters — the same slot-per-worker shape
    /// as [`Self::node_tally`] (one `AtomicU64` per worker plus this worker's
    /// index). Each worker `fetch_add`s its own slot at the root best-move-change
    /// site, and the **main** worker (index 0) folds *every* slot into
    /// `totBestMoveChanges` and zeroes each at every iteration end — the reference
    /// `for (auto&& th : threads) { totBestMoveChanges += yw->bestMoveChanges;
    /// yw->bestMoveChanges = 0; }` (`yaneuraou-search.cpp`, main-thread
    /// only via the `if (!mainThread) continue;` at `:1902`). Helpers never fold or
    /// reset — the main worker reads and clears their slots. Relaxed atomics: the
    /// reference's cross-thread reads are benign races. `None` on the direct
    /// single-worker [`Self::run_root`] path, where [`Self::best_move_changes`]
    /// carries the count and the behaviour is bit-identical to before.
    best_move_tally: Option<(Arc<Vec<AtomicU64>>, usize)>,

    /// `pvIdx` — the current MultiPV line index (`yaneuraou-search.cpp`).
    /// Read at the root by the interior-search hooks; `0` on every non-MultiPV
    /// path, so those hooks are no-ops and the single-PV search stays
    /// bit-identical.
    pv_idx: usize,
    /// `min(options["MultiPV"], rootMoves.size())` input — the raw `MultiPV`
    /// option value. `1` by default (the fixed-depth parity path runs one PV
    /// line). Set per `go` by [`Self::set_pv_output`] (main) / [`Self::set_multi_pv`]
    /// (helpers).
    multi_pv: usize,
    /// The per-iteration / final PV output sink (`main_manager()->pv()`). `None`
    /// on every worker but the main one, so helpers and the parity path emit
    /// nothing.
    pv_sink: Option<Box<dyn PvSink>>,
    /// The main worker's PV-output configuration for this `go` (`None` elsewhere).
    pv_config: Option<PvOutputConfig>,
    /// `lastPvInfoTime` — the last time a PV was emitted (`989`, refreshed at each
    /// `pv()` call). Seeded to `pv_config.start_time` per `go`; unused when
    /// `pv_config` is `None`.
    last_pv_info_time: Instant,
}

/// The result one worker's iterative deepening produces, consumed by
/// the driver's Lazy-SMP orchestration to vote for and report a single result.
#[derive(Clone, Debug)]
pub struct WorkerResult {
    /// The last completed iteration's `rootMoves[0]` (or, on an abort before any
    /// iteration completed, the best-so-far after the partial iteration's sort).
    pub best: RootMove,
    /// `completedDepth` — the last fully-completed iterative-deepening depth (`0`
    /// if none completed). Reported as `info depth` for the chosen worker and fed
    /// to the thread vote.
    pub completed_depth: i32,
    /// The previous iteration's `pv[1]` — the `extract_ponder_from_tt` fallback
    /// applied to the chosen worker's length-1 PV.
    pub ponder_candidate: Move,
    /// This worker's own node count (`do_move` calls). The driver sums every
    /// worker's count for the aggregated `info ... nodes` output.
    pub nodes: u64,
    /// Whether this worker (only meaningful for the main worker) already emitted
    /// the last completed iteration's final PV during iterative deepening — the
    /// reference `uciPvSent` return of `iterative_deepening`
    /// (`yaneuraou-search.cpp`). The coordinator's final-PV fallback
    /// (`1289`) keys off this so a fully throttled search still emits one PV
    /// before `bestmove`. Always `false` for a helper (helpers never emit).
    pub uci_pv_sent: bool,
    /// The last completed iteration's top-`multiPV` root moves, in score order —
    /// the lines the coordinator re-emits when the final PV was throttled
    /// (`uci_pv_sent == false`). `[best]` for the fixed-depth / helper paths.
    pub pv_lines: Vec<RootMove>,
    /// `timeReduction` after iterative deepening (`yaneuraou-search.cpp`) —
    /// the value the driver stores as `previousTimeReduction` for the next `go`.
    /// `1.0` when the time-management block never ran (fixed-depth / helper).
    pub time_reduction: f64,
}

/// USI `info` bound marker for one PV line — the reference `pv()` `isExact` /
/// `scoreLowerbound` / `scoreUpperbound` logic (`yaneuraou-search.cpp`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PvBound {
    /// An exact score (no `lowerbound` / `upperbound` marker).
    Exact,
    /// A fail-high lower bound (`lowerbound`).
    Lower,
    /// A fail-low upper bound (`upperbound`).
    Upper,
}

/// One PV line's data for a USI `info` output — the reference `InfoFull`
/// (`search.h`) as this port surfaces it. The Protocol layer formats it into
/// the wire line; the `Value`→`cp`/`mate` and `Move`→USI conversions live there.
#[derive(Clone, Debug)]
pub struct PvInfo {
    /// `info.depth`.
    pub depth: i32,
    /// `info.selDepth`.
    pub sel_depth: i32,
    /// `info.multiPV` — the **1-based** PV index (`i + 1`).
    pub multipv: usize,
    /// `info.score`, still a raw search [`Value`].
    pub score: Value,
    /// Whether the score is exact or a fail-high/low bound.
    pub bound: PvBound,
    /// `info.nodes`.
    pub nodes: u64,
    /// `info.pv` as moves (the Protocol layer joins them into USI text).
    pub pv: Vec<Move>,
}

/// Sink for per-iteration / final PV `info` output (the reference
/// `main_manager()->pv()` call sites). Only the **main** worker is given one;
/// helper workers and the direct fixed-depth [`QSearch::run_root`] path leave it
/// unset, so they never emit and the parity path is untouched.
pub trait PvSink: Send {
    /// Emit one PV line.
    fn emit(&mut self, info: &PvInfo);
}

/// The PV-output configuration snapshot for one `go` (the reference
/// registration at `yaneuraou-search.cpp` and the per-search derivation at
/// `989-997`). Installed on the main worker only, via [`QSearch::set_pv_output`].
#[derive(Clone)]
pub struct PvOutputConfig {
    /// `min(options["MultiPV"], rootMoves.size())` is applied inside the worker;
    /// this is the raw `MultiPV` option value.
    pub multi_pv: usize,
    /// `computed_pv_interval` (`993-997`): `0` (never suppress) when `go infinite`
    /// or `ConsiderationMode` is on, else the `PvInterval` option as a duration.
    pub pv_interval: Duration,
    /// `ConsiderationMode` (`88-92`): collect each PV from the transposition table
    /// instead of the searched PV array.
    pub consideration_mode: bool,
    /// `OutputFailLHPV` (`94-98`): emit a PV on a fail-high/low re-search.
    pub output_fail_lh_pv: bool,
    /// `limits.startTime` — the `lastPvInfoTime` seed (`989`).
    pub start_time: Instant,
}

/// `DrawValueBlack` / `DrawValueWhite` default (`yaneuraou-search.cpp`).
const DRAW_VALUE_OPTION_DEFAULT: i32 = -2;
/// `Eval::PawnValue` (`evaluate.h`), used to scale the contempt option.
const PAWN_VALUE: i32 = 90;

impl<'a> QSearch<'a> {
    /// Create a driver over `net` and a **pre-sized** `tt` with fresh history
    /// tables. The reference re-fills `lowPlyHistory` to 98 per `go`
    /// (`yaneuraou-search.cpp`); [`WorkerHistories::new`] leaves it zero, so
    /// it is seeded here so a bare [`Self::run`] / [`Self::run_search`] (which do
    /// not run the per-`go` refill) sees the reference value.
    pub fn new(net: &'a NnueNetwork, tt: &'a TranspositionTable) -> Self {
        let histories = {
            let mut h = WorkerHistories::new();
            h.low_ply.fill(98);
            h
        };
        Self::with_histories(net, tt, histories)
    }

    /// Create a driver over `net` and a **pre-sized** `tt`, taking ownership of
    /// externally-held `histories`.
    ///
    /// This is the driver's game-scoped path: the session owns
    /// one [`WorkerHistories`] that persists across `go`s within a game (the
    /// reference lifetime — histories are cleared only by `usinewgame` /
    /// `search_clear`) and hands it in here per `go`, reclaiming it afterwards
    /// with [`Self::into_histories`]. The per-`go` `lowPlyHistory` refill still
    /// happens inside [`Self::run_root`], so a fresh bundle and a carried-over
    /// bundle enter their first `go` identically.
    pub fn with_histories(
        net: &'a NnueNetwork,
        tt: &'a TranspositionTable,
        histories: WorkerHistories,
    ) -> Self {
        Self {
            net,
            tt,
            nodes: 0,
            sel_depth: 0,
            nmp_min_ply: 0,
            pv_node: false,
            read_tt: true,
            root_us: Color::Black,
            // Default draw contempt = the baked `DrawValueBlack/White = -2`
            // option scaled by `PawnValue`: `-2 * 90 / 100 == -1`. A bare
            // `run`/`run_search` (the fixed-depth parity path) never calls
            // `set_draw_value`, so this default keeps that path bit-identical.
            draw_contempt: DRAW_VALUE_OPTION_DEFAULT * PAWN_VALUE / 100,
            // A fixed-size boxed array; each cell's `pv` is preallocated so a
            // PV update never grows the buffer on the hot path. Built as a
            // `Vec` (each cell distinct — `SearchStackCell` is not `Copy`) then
            // converted to the boxed array without a stack copy.
            stack: (0..STACK_LEN)
                .map(|_| {
                    let mut cell = SearchStackCell::default();
                    cell.pv.reserve_exact(MAX_PLY as usize + 1);
                    cell
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
                .try_into()
                .map_err(|_| ())
                .expect("STACK_LEN cells collected"),
            // One accumulator slot per reachable do/undo depth (bounded by
            // `MAX_PLY`, plus headroom); `Accumulator` is not `Clone`, so build
            // the slots individually. A boxed fixed-size array keeps the large
            // slots on the heap (no stack copy) with a compile-time length.
            acc_stack: (0..ACC_LEN)
                .map(|_| Accumulator::new())
                .collect::<Vec<_>>()
                .into_boxed_slice()
                .try_into()
                .map_err(|_| ())
                .expect("ACC_LEN slots collected"),
            acc_depth: 0,
            finny: FinnyCache::new(),
            verify_accumulator: false,
            histories,
            reductions: {
                let mut r = vec![0i32; 600];
                for (i, slot) in r.iter_mut().enumerate().skip(1) {
                    *slot = (2763.0 / 128.0 * (i as f64).ln()) as i32;
                }
                r
            },
            root_delta: 2 * VALUE_INFINITE,
            root_depth: 1,
            last_iteration_pv: Vec::new(),
            control: SearchControl::default(),
            entering_king: EnteringKingConfig::default(),
            max_moves_to_draw: MAX_MOVES_TO_DRAW,
            generate_all_legal_moves: false,
            mate_mode: false,
            calls_cnt: CHECK_INTERVAL,
            stopped: false,
            completed_depth: 0,
            best_move_changes: 0.0,
            stop_on_ponderhit: false,
            ponderhit_synced: false,
            node_tally: None,
            best_move_tally: None,
            pv_idx: 0,
            multi_pv: 1,
            pv_sink: None,
            pv_config: None,
            last_pv_info_time: Instant::now(),
        }
    }

    /// Test-only (the accumulator-equivalence gate): when enabled, every
    /// evaluation point re-checks the differential accumulator against a
    /// from-scratch [`attic_eval::evaluate`]. Never enabled on a production
    /// search, so the refresh entry point is never called there.
    #[cfg(test)]
    pub fn set_verify_accumulator(&mut self, verify: bool) {
        self.verify_accumulator = verify;
    }

    /// Install the time / node / stop [`SearchControl`] for the coming search.
    /// With the default (empty) control every stop-check site is inert and the
    /// search runs the fixed-depth parity path.
    pub fn set_control(&mut self, control: SearchControl) {
        self.control = control;
    }

    /// Install the entering-king declaration config for this `go`.
    /// Called by the driver after snapshotting the `EnteringKingRule` option and
    /// computing the per-side thresholds from the root position; every worker
    /// (main + helpers) gets the same config, since the material total is
    /// invariant across the search.
    pub fn set_entering_king(&mut self, config: EnteringKingConfig) {
        self.entering_king = config;
    }

    /// Install the `MaxMovesToDraw` horizon for this `go`.
    /// `value` must already be the `0 → 100000` remapped value (the driver does
    /// the remap): the game ply past which every interior / qsearch node returns
    /// the forced-draw score. Every worker (main + helpers) gets the same value.
    pub fn set_max_moves_to_draw(&mut self, value: i32) {
        self.max_moves_to_draw = value;
    }

    /// Install the root-side draw contempt `drawValueTable[REPETITION_DRAW][us]`
    /// for this `go`. `contempt` must already be the
    /// pawn-scaled `option[root_us] * PawnValue / 100` (the driver does the scale
    /// from the root side to move). [`Self::draw_value`] returns `+contempt` for
    /// the root side and `-contempt` for the opponent, reproducing the reference's
    /// symmetric `±draw_value` (`yaneuraou-search.cpp`). Every worker
    /// gets the same value; the fixed-depth parity path never calls this and keeps
    /// the baked `-1` default.
    pub fn set_draw_value(&mut self, contempt: Value) {
        self.draw_contempt = contempt;
    }

    /// Install the `GenerateAllLegalMoves` flag for this `go`:
    /// when true the search-facing generators also yield the non-promoting moves
    /// the default generator suppresses. Every worker gets the same flag; `false`
    /// (the default) leaves generation bit-identical to the parity path.
    pub fn set_generate_all_legal_moves(&mut self, all: bool) {
        self.generate_all_legal_moves = all;
    }

    /// Install `go mate` mode for this `go`: disables the
    /// iterative-deepening early mate/mated break and arms the mate-found stop
    /// rule. Every worker gets the same flag; `false` (the default) leaves the
    /// iterative-deepening loop bit-identical to the parity path.
    pub fn set_mate_mode(&mut self, mate: bool) {
        self.mate_mode = mate;
    }

    /// Install the Lazy-SMP shared node counters: `slots` holds one
    /// [`AtomicU64`] per worker and `index` is this worker's slot. Each
    /// `check_time` checkpoint publishes `self.nodes` there so the main worker's
    /// `go nodes N` ceiling can sum every worker's count (the reference
    /// `threads.nodes_searched()`). Leave unset for the single-worker path.
    pub fn set_node_tally(&mut self, slots: Arc<Vec<AtomicU64>>, index: usize) {
        self.node_tally = Some((slots, index));
    }

    /// Install the Lazy-SMP shared best-move-change counters: `slots` holds one
    /// [`AtomicU64`] per worker and `index` is this worker's slot. Each worker
    /// `fetch_add`s its own slot at the root best-move-change site; the main worker
    /// (index 0) folds every slot into `totBestMoveChanges` at each iteration end
    /// (the reference `1936-1941` aggregate). Leave unset for the single-worker
    /// path, where [`Self::best_move_changes`] carries the count. See
    /// [`Self::best_move_tally`].
    pub fn set_best_move_tally(&mut self, slots: Arc<Vec<AtomicU64>>, index: usize) {
        self.best_move_tally = Some((slots, index));
    }

    /// Install the raw `MultiPV` option value for this `go` (helpers). The main
    /// worker uses [`Self::set_pv_output`] instead, which also sets the sink. Both
    /// clamp to `rootMoves.size()` inside [`Self::run_worker`]. Leave unset (`1`)
    /// for the fixed-depth parity path.
    pub fn set_multi_pv(&mut self, multi_pv: usize) {
        self.multi_pv = multi_pv.max(1);
    }

    /// Install the PV-output configuration and sink on the **main** worker (the
    /// reference `main_manager()->pv()` owner). This sets `MultiPV`, the computed
    /// PV interval, the consideration / fail-LH flags, and seeds `lastPvInfoTime`
    /// to the search start; the sink receives each emitted line.
    pub fn set_pv_output(&mut self, config: PvOutputConfig, sink: Box<dyn PvSink>) {
        self.multi_pv = config.multi_pv.max(1);
        self.last_pv_info_time = config.start_time;
        self.pv_config = Some(config);
        self.pv_sink = Some(sink);
    }

    /// Consume the driver and return its history tables so the session can carry
    /// them into the next `go`. Consuming `self` also ends the
    /// mutable borrow of the transposition table, freeing the caller to reclaim
    /// it too.
    pub fn into_histories(self) -> WorkerHistories {
        self.histories
    }

    /// The `callsCnt` reload value (`yaneuraou-search.cpp`): the standard
    /// [`CHECK_INTERVAL`], but capped tighter when a small node ceiling is set so
    /// the check rate stays at least ~0.1% of the ceiling.
    fn calls_reset(&self) -> i32 {
        match self.control.node_limit {
            Some(n) => CHECK_INTERVAL.min((n / 1024) as i32).max(1),
            None => CHECK_INTERVAL,
        }
    }

    /// The aggregate node count against the `go nodes` ceiling
    /// (`worker.threads.nodes_searched()`). With a tally installed, sum every
    /// worker's published slot; without one (single worker), this worker's own
    /// `nodes` is the whole search — bit-identical to the pre-Lazy-SMP check.
    fn counted_nodes(&self) -> u64 {
        match &self.node_tally {
            Some((slots, _)) => slots.iter().map(|s| s.load(Ordering::Relaxed)).sum(),
            None => self.nodes,
        }
    }

    /// Whether this search is currently pondering (the shared [`PonderSignal`] is
    /// installed and still active). `false` on every non-ponder search.
    fn is_pondering(&self) -> bool {
        self.control.ponder.as_ref().is_some_and(|p| p.is_active())
    }

    /// Copy the shared [`PonderSignal`]'s stamped ponderhit instant into
    /// `tm.ponderhitTime` the first time we observe the ponder flag cleared — the
    /// port's stand-in for the reference `set_ponderhit` writing `tm.ponderhitTime`
    /// on the USI thread (`yaneuraou-search.cpp`). A no-op on a non-ponder
    /// search, while still pondering, or after the one-time sync has run.
    /// Fold this iteration's best-move changes into `tot` and reset for the next
    /// iteration — the reference `for (auto&& th : threads) { totBestMoveChanges +=
    /// yw->bestMoveChanges; yw->bestMoveChanges = 0; }` (`yaneuraou-search.cpp`),
    /// which runs on the main thread only. Under Lazy-SMP the main worker (slot 0)
    /// sums and zeroes EVERY worker's slot; helpers reach this fold too (the port
    /// has no `if (!mainThread) continue;`) but take the no-op arm, leaving their
    /// slot for the main worker to read+zero. The single-worker [`Self::run_root`]
    /// path (no tally) folds its own scalar — bit-identical to the pre-Lazy-SMP code.
    fn fold_best_move_changes(&mut self, tot: &mut f64) {
        match &self.best_move_tally {
            Some((slots, 0)) => {
                for s in slots.iter() {
                    *tot += s.swap(0, Ordering::Relaxed) as f64;
                }
            }
            Some(_) => {}
            None => {
                *tot += self.best_move_changes;
                self.best_move_changes = 0.0;
            }
        }
    }

    fn sync_ponderhit(&mut self) {
        if self.ponderhit_synced {
            return;
        }
        // Read (and copy out) the stamped instant without holding a borrow on
        // `self.control` across the `self.control.time` mutation below.
        let hit = match self.control.ponder.as_ref() {
            Some(p) if !p.is_active() => p.hit_at(),
            _ => return,
        };
        self.ponderhit_synced = true;
        if let (Some(hit), Some(tc)) = (hit, self.control.time.as_mut()) {
            tc.tm.ponderhit_time = hit;
        }
    }

    /// The reference `SearchManager::check_time` (`yaneuraou-search.cpp`):
    /// count down and, once per [`Self::calls_reset`] calls, consult the stop
    /// flag, then — only after the first iteration has completed
    /// (`completedDepth >= 1`, `5527`/`5532`) — the `movetime` / node / `search_end`
    /// stops, or set `search_end` from `maximum()` under time management. A no-op
    /// cost aside, it changes nothing when the control is empty — the parity
    /// guarantee.
    fn check_time(&mut self) {
        if self.stopped {
            return;
        }
        self.calls_cnt -= 1;
        if self.calls_cnt > 0 {
            return;
        }
        self.calls_cnt = self.calls_reset();

        // Publish this worker's node count so the main worker's aggregate ceiling
        // (and the final aggregated `nodes` output) can sum every worker — the
        // reference reads a per-worker atomic `nodes` each checkpoint
        // (`threads.nodes_searched()`); this port publishes at the checkpoint
        // rather than per node. Inert (no slot) on the single-worker path.
        if let Some((slots, idx)) = &self.node_tally {
            slots[*idx].store(self.nodes, Ordering::Relaxed);
        }

        // The external stop flag (GUI `stop` / `gameover`, the coordinator's
        // end-of-search signal, a sibling worker's abort) is honoured
        // unconditionally — it is how every worker terminates.
        if let Some(flag) = &self.control.stop
            && flag.load(Ordering::Relaxed)
        {
            self.stopped = true;
            return;
        }

        // While pondering, make no stop decision at all — a `go ponder` search
        // self-terminates only on `stop` (checked above) or a `ponderhit`
        // (`yaneuraou-search.cpp`). The stop check deliberately precedes
        // this so `stop` still ends a pondering search.
        if self.control.ponder.as_ref().is_some_and(|p| p.is_active()) {
            return;
        }
        // Not (or no longer) pondering: if a `ponderhit` just cleared the flag,
        // copy its stamped instant into `tm.ponderhitTime` once, before the
        // `set_search_end` below reads it (the reference set_ponderhit ordering).
        self.sync_ponderhit();

        // The reference gates every time / node stop on completedDepth >= 1 so a
        // `bestmove` is always backed by at least one finished iteration.
        if self.completed_depth < 1 {
            return;
        }

        // 3./4. movetime elapsed or node ceiling reached ⇒ stop immediately.
        let counted = self.counted_nodes();
        if let Some(limit) = self.control.node_limit
            && counted >= limit
        {
            self.request_abort();
            return;
        }
        let Some(tc) = self.control.time.as_ref() else {
            return;
        };
        let elapsed = tc.tm.elapsed_from(Instant::now());
        if let Some(movetime) = tc.movetime
            && elapsed >= movetime
        {
            self.request_abort();
            return;
        }
        // 5. the TimeManagement-decided end time has arrived ⇒ stop immediately.
        if tc.tm.search_end != 0 {
            if tc.tm.search_end <= elapsed {
                self.request_abort();
            }
            return;
        }
        // 1./2. the maximum think time (or stopOnPonderhit) is exceeded ⇒ round up
        // to a whole second via set_search_end rather than stopping now.
        if tc.use_time_management && (elapsed > tc.tm.maximum() || self.stop_on_ponderhit) {
            self.control
                .time
                .as_mut()
                .expect("time control present")
                .tm
                .set_search_end(elapsed);
        }
    }

    /// Latch the abort and, if a shared stop flag is installed, publish it so an
    /// external observer (the driver's next `stop`/`quit`, a sibling reader) sees
    /// the search has terminated itself — the reference's
    /// `threads.stop = threads.abortedSearch = true`.
    fn request_abort(&mut self) {
        self.stopped = true;
        if let Some(flag) = &self.control.stop {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Run quiescence search on `pos` from ply 0.
    ///
    /// `alpha < beta` must hold, and `pv_node || alpha == beta - 1` (the
    /// reference's `ASSERT_LV3`). `read_tt == false` reproduces the `ReadTT ==
    /// false` template instantiation (TT hits are ignored, but writes still
    /// happen).
    pub fn run(
        &mut self,
        pos: &mut Position,
        alpha: Value,
        beta: Value,
        pv_node: bool,
        read_tt: bool,
    ) -> QSearchOutcome {
        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || alpha == beta - 1);

        self.nodes = 0;
        self.sel_depth = 0;
        self.pv_node = pv_node;
        self.read_tt = read_tt;
        self.root_us = pos.side_to_move();
        self.stopped = false;
        self.calls_cnt = CHECK_INTERVAL;

        for cell in self.stack.iter_mut() {
            cell.current_move = Move::none();
            cell.tt_pv = false;
            cell.pv.clear();
        }

        // Seed the root accumulator; every node below derives from it.
        self.seed_accumulator(pos);

        let value = self.qsearch(pos, 0, alpha, beta);

        QSearchOutcome {
            value,
            nodes: self.nodes,
            pv: self.stack[Self::si(0)].pv.clone(),
            sel_depth: self.sel_depth,
        }
    }

    /// Stack index for search ply `ply`.
    #[inline]
    fn si(ply: i32) -> usize {
        STACK_BASE + ply as usize
    }

    /// The current node's live accumulator (`acc_stack[acc_depth]`), read by the
    /// six evaluation sites via [`attic_eval::evaluate_with`].
    #[inline]
    fn acc(&self) -> &Accumulator {
        &self.acc_stack[self.acc_depth]
    }

    /// The node's NNUE static evaluation, read through the differentially
    /// maintained accumulator (`evaluate_with`) rather than a per-node full
    /// refresh — the whole point of the differential accumulator. Every one of
    /// the search's six evaluation sites routes through here.
    ///
    /// When [`Self::verify_accumulator`] is set (a test-only knob, off in
    /// production so the refresh entry point is never called on a real search),
    /// it additionally asserts the differential result equals a from-scratch
    /// [`attic_eval::evaluate`] of the current position — the in-search half of
    /// the accumulator-equivalence gate, checked at *every* evaluation point.
    #[inline]
    fn static_eval(&self, pos: &Position) -> Value {
        let value = evaluate_with(self.net, self.acc(), pos);
        if self.verify_accumulator {
            assert_eq!(
                value,
                attic_eval::evaluate(self.net, pos),
                "differential NNUE accumulator diverged from a full refresh",
            );
        }
        value
    }

    /// Full-refresh the root accumulator into slot `0` and reset the do/undo
    /// depth. Called once at the start of every fresh search from a root
    /// position (`run` / `run_search` / each worker's iterative deepening); every
    /// deeper node's accumulator is derived incrementally from here.
    #[inline]
    fn seed_accumulator(&mut self, pos: &Position) {
        let net = self.net;
        self.acc_depth = 0;
        self.acc_stack[0].refresh(net, pos);
    }

    /// Derive the child accumulator from the current top and advance the depth,
    /// mirroring a `do_move`. `post_pos` is the position *after* the move (used
    /// only to rebuild a perspective whose own king moved); `delta` is the
    /// move-derived add/sub feature delta captured from the pre-move position.
    ///
    /// The king-move rebuild goes through this worker's finny table
    /// ([`Self::finny`]), which turns most such rebuilds into a few-column diff
    /// against the cached refreshed half for that king square. Bit-identical to
    /// the uncached [`Accumulator::derive_into`] either way.
    #[inline]
    fn push_accumulator(&mut self, post_pos: &Position, delta: &MoveDelta) {
        // Software-prefetch the child position's TT cluster. The
        // reference prefetches inside `Position::do_move` the moment the
        // post-move key is known; this port's `Position` can't reach the TT, so
        // the hint is issued here — the accumulator seam every real-search
        // `do_move` funnels through (qsearch move loop, ProbCut, main move loop)
        // — a few nanoseconds after `do_move` returns but well before the child
        // probes. `post_pos` is already the position after the move, so its
        // `key()` / `side_to_move()` mirror the reference's `first_entry(key,
        // them)`. Pure hint: zero effect on node counts or any output.
        self.tt
            .prefetch(post_pos.key(), post_pos.side_to_move().index() as u8);

        let net = self.net;
        let d = self.acc_depth;
        let (parent_slots, child_slots) = self.acc_stack.split_at_mut(d + 1);
        Accumulator::derive_into_cached(
            &parent_slots[d],
            &mut child_slots[0],
            net,
            post_pos,
            delta,
            &mut self.finny,
        );
        self.acc_depth = d + 1;
    }

    /// Pop the child accumulator, mirroring an `undo_move`. The parent slot is
    /// never mutated by [`Self::push_accumulator`], so the parent accumulator is
    /// restored simply by dropping back to it.
    #[inline]
    fn pop_accumulator(&mut self) {
        self.acc_depth -= 1;
    }

    /// `drawValueTable[rs][c]` (`types.cpp` defaults, with the
    /// `REPETITION_DRAW` row overwritten from contempt at search start,
    /// `yaneuraou-search.cpp`).
    fn draw_value(&self, rs: RepetitionState, c: Color) -> Value {
        match rs {
            RepetitionState::None => VALUE_DRAW,
            RepetitionState::Win => VALUE_MATE,
            RepetitionState::Lose => -VALUE_MATE,
            RepetitionState::Draw => {
                if c == self.root_us {
                    self.draw_contempt
                } else {
                    -self.draw_contempt
                }
            }
            // VALUE_SUPERIOR == VALUE_MAX_EVAL (`types.h`).
            RepetitionState::Superior => VALUE_MAX_EVAL,
            RepetitionState::Inferior => -VALUE_MAX_EVAL,
        }
    }

    /// Write a TT entry to the exact `slot` captured by the node's initial
    /// [`TranspositionTable::locate`] probe, reproducing the reference's single-
    /// `TTWriter`-per-node discipline. Writing by a fixed slot (rather than a
    /// re-probe) keeps the stored TT state bit-identical to the reference even
    /// when a child has since churned the cluster — the write-slot-drift fix.
    #[allow(clippy::too_many_arguments)]
    fn tt_store(
        &mut self,
        slot: TtSlot,
        key: u64,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: i32,
        mv: u16,
        eval: Value,
    ) {
        let generation = self.tt.generation();
        self.tt
            .write_at(slot, key, value, pv, bound, depth, mv, eval, generation);
    }

    /// Widen a stored 16-bit move against `pos` and validate it: returns the
    /// unique **legal** move whose `Move16` equals `move16`, or `None`.
    ///
    /// Retained as the **`#[cfg(test)]` oracle** for the O(1)
    /// [`Position::to_move`] + [`Position::pseudo_legal`] chain that replaced it
    /// — production widens through that chain now. This
    /// generate-and-match form both reconstructs the full move (piece bits and
    /// all) and proves legality in one step, and is used by the torn-entry
    /// totality / oracle tests to pin the new chain's accepted set. It widens
    /// through [`Position::generate_legal_all`], which is
    /// repetition-blind exactly like the production `to_move` + `pseudo_legal`
    /// chain — so the two agree on every move, including one that continues a
    /// perpetual check.
    #[cfg(test)]
    fn widen_tt_move(pos: &Position, move16: u16) -> Option<Move> {
        if move16 == 0 {
            return None;
        }
        let mut legal: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut legal);
        Self::select_tt_move(&legal, move16)
    }

    /// Select, from a position's already-generated `legal` moves, the unique one
    /// whose `Move16` fragment equals `move16`, or `None`. This is the only step
    /// that consumes the raw fragment, and it is a total `move16_of` comparison
    /// (no bit is decoded into a `Move`). Split out from [`widen_tt_move`] so the
    /// torn-entry totality test can drive all 65536 fragments
    /// against one generated list without re-generating per pattern.
    #[cfg(test)]
    fn select_tt_move(legal: &[Move], move16: u16) -> Option<Move> {
        legal.iter().copied().find(|&m| move16_of(m) == move16)
    }

    /// The core recursive qsearch (`yaneuraou-search.cpp`).
    fn qsearch(&mut self, pos: &mut Position, ply: i32, mut alpha: Value, beta: Value) -> Value {
        let pv_node = self.pv_node;

        // Poll the stop flag / hard deadline at the reference `check_time`
        // granularity. The reference only checks in the interior
        // `search`, but polling in qsearch too bounds the abort latency inside a
        // deep leaf tree. On abort the return value is discarded — the interior
        // caller unwinds immediately via its own post-move stop check — so any
        // value is fine. Inert on the parity path (`stopped` never latches).
        self.check_time();
        if self.stopped {
            return VALUE_DRAW;
        }

        // -----------------------------------------------------------------
        // Step 1. Initialize node (4515-4556).
        // -----------------------------------------------------------------
        if pv_node {
            self.stack[Self::si(ply)].pv.clear();
        }
        let in_check = pos.in_check();
        let mut move_count = 0;
        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        // -----------------------------------------------------------------
        // Step 2. Immediate draw / max ply (4558-4621).
        // -----------------------------------------------------------------
        let us = pos.side_to_move();
        let draw_type = pos.is_repetition(ply as u16);
        if draw_type != RepetitionState::None {
            if draw_type == RepetitionState::Draw {
                // Ordinary repetition: the ±1 dither (4601).
                return self.draw_value(RepetitionState::Draw, us) + value_draw(self.nodes);
            } else {
                // Superior / inferior / perpetual-check: map to a root-relative
                // score (4603).
                return value_from_tt(self.draw_value(draw_type, us), ply);
            }
        }
        // The `depth <= -16 → draw` measure at 4607-4613 is `#if 0`: not ported.
        if ply >= MAX_PLY || pos.ply() as i32 > self.max_moves_to_draw {
            return self.draw_value(RepetitionState::Draw, us) + value_draw(self.nodes);
        }

        // -----------------------------------------------------------------
        // Step 3. Transposition table lookup (4623-4679).
        // -----------------------------------------------------------------
        let pos_key = pos.key();
        let side = us.index() as u8;
        // Capture the entry location once (like the reference's Step-3
        // `ttWriter`); every write below targets this exact slot.
        let (found, tt_data, tt_slot) = self.tt.locate(pos_key, side);
        // `if constexpr (!ReadTT) ttHit = false;` (4632).
        let tt_hit = found && self.read_tt;
        // `ttData.move = ttHit ? to_move(ttData.move) : Move::none();` (4640):
        // O(1) widening (`Position::to_move`), no legal-move generation. The
        // MovePicker's TT stage re-validates with `pseudo_legal` + `is_legal`.
        let tt_move = if tt_hit {
            pos.to_move(tt_data.move16)
        } else {
            None
        };
        // `ttData.value = ttHit ? value_from_tt(...) : VALUE_NONE;` (4642-4646).
        let tt_value = if tt_hit {
            value_from_tt(tt_data.value, ply)
        } else {
            VALUE_NONE
        };
        let pv_hit = tt_hit && tt_data.is_pv;

        // Non-PV early TT cutoff (4661-4670).
        if !pv_node
            && tt_data.depth >= DEPTH_QS
            && is_valid(tt_value)
            && bound_matches(tt_data.bound, tt_value >= beta)
        {
            return tt_value;
        }

        // -----------------------------------------------------------------
        // Step 4. Static evaluation (4681-4830).
        // -----------------------------------------------------------------
        let mut unadjusted_static_eval = VALUE_NONE;
        let mut best_value: Value;
        let futility_base: Value;

        if in_check {
            // Every evasion is generated, so start from -infinity (4687-4696).
            best_value = -VALUE_INFINITE;
            futility_base = -VALUE_INFINITE;
        } else {
            let correction_value = self.correction_value(pos, ply);

            if tt_hit {
                // Trust nothing about the stored eval (4701-4736).
                unadjusted_static_eval = tt_data.eval;
                if !is_valid(unadjusted_static_eval) {
                    unadjusted_static_eval = self.static_eval(pos);
                } else if pv_node {
                    // USE_CLASSIC_EVAL re-eval on PV nodes (4712-4718): the
                    // classic-eval branch is active in the reference build.
                    unadjusted_static_eval = self.static_eval(pos);
                }
                best_value = to_corrected_static_eval(unadjusted_static_eval, correction_value);

                // ttValue as a better position eval (4734-4736).
                if is_valid(tt_value)
                    && !is_decisive(tt_value)
                    && bound_matches(tt_data.bound, tt_value > best_value)
                {
                    best_value = tt_value;
                }
            } else {
                // 🌈 1-ply mate check, only when the TT missed (4738-4775).
                if let Some(mate_move) = pos.mate_1ply() {
                    best_value = mate_in(ply + 1);
                    // Note: the stored value is the RAW root-relative
                    // `mate_in(ply+1)`, deliberately NOT `value_to_tt`-converted
                    // (4768), and the pv flag is the persistent `ss->ttPv`.
                    let tt_pv = self.stack[Self::si(ply)].tt_pv;
                    self.tt_store(
                        tt_slot,
                        pos_key,
                        best_value,
                        tt_pv,
                        Bound::Exact,
                        DEPTH_QS,
                        move16_of(mate_move),
                        unadjusted_static_eval,
                    );
                    return best_value;
                }

                unadjusted_static_eval = self.static_eval(pos);
                best_value = to_corrected_static_eval(unadjusted_static_eval, correction_value);
            }

            // Stand pat (4793-4815).
            if best_value >= beta {
                if !is_decisive(best_value) {
                    best_value = (best_value + beta) / 2;
                }
                if !tt_hit {
                    self.tt_store(
                        tt_slot,
                        pos_key,
                        value_to_tt(best_value, ply),
                        false,
                        Bound::Lower,
                        DEPTH_UNSEARCHED,
                        0,
                        unadjusted_static_eval,
                    );
                }
                return best_value;
            }

            if best_value > alpha {
                alpha = best_value;
            }

            // `ss->staticEval` is `best_value` before the stand-pat clamp; here
            // `best_value` still equals the corrected static eval (the only
            // mutation above was the `ttValue`-refinement, which the reference
            // applies to `best_value`, not `ss->staticEval`). The reference's
            // `futilityBase = ss->staticEval + 328` uses the corrected eval; we
            // recompute it identically.
            futility_base = to_corrected_static_eval(unadjusted_static_eval, correction_value)
                + FUTILITY_MARGIN;
        }

        // -----------------------------------------------------------------
        // Step 5-8. Move loop (4832-5025).
        // -----------------------------------------------------------------
        // `prevSq` from `(ss-1)->currentMove` (4842).
        let prev_move = self.stack[Self::si(ply) - 1].current_move;
        let prev_sq = if prev_move.is_ok() {
            Some(prev_move.to_sq())
        } else {
            None
        };

        // `contHist[] = {(ss-1)->continuationHistory}` (`yaneuraou-search.cpp`).
        // The qsearch evasion score reads plane `[0]`, the previous ply's REAL
        // continuation plane; with untouched tables it holds the uniform -523
        // fill, so the depth-1 evasion ordering is a constant shift.
        let cont_planes: [usize; 6] =
            std::array::from_fn(|i| self.stack[Self::si(ply) - 1 - i].cont_hist);
        let mut mp =
            MovePicker::new_qsearch(pos, tt_move, cont_planes, self.generate_all_legal_moves);

        let mut best_move = Move::none();

        while let Some(mv) = mp.next_move(pos, &self.histories) {
            // The MovePicker yields only legal moves, so the reference's
            // `if (!pos.legal(move)) continue;` (4880) is already applied — an
            // illegal move is skipped without counting, exactly as here.

            let gives_check = pos.gives_check(mv);
            // `capture_stage` is plain `capture` at the pin: a non-drop landing
            // on an occupied (enemy) square (4886).
            let capture = !mv.is_drop() && pos.board().get(mv.to_sq()).is_some();
            move_count += 1;

            // Step 6. Pruning (4890-4982), only while not already losing.
            if !is_loss(best_value) {
                if !gives_check && Some(mv.to_sq()) != prev_sq && !is_loss(futility_base) {
                    // moveCount pruning: 3rd non-recapture onward (4919-4920).
                    if move_count > 2 {
                        continue;
                    }
                    // `PieceValue[piece_on(to)]`; 0 for a quiet (empty) target
                    // (4928-4929).
                    let futility_value =
                        futility_base + pos.board().get(mv.to_sq()).map_or(0, piece_value);
                    if futility_value <= alpha {
                        best_value = best_value.max(futility_value);
                        continue;
                    }
                    if !pos.see_ge(mv, alpha - futility_base) {
                        best_value = best_value.max(alpha.min(futility_base));
                        continue;
                    }
                }

                // Skip non-captures — this also drops a quiet TT move (4966-4967).
                if !capture {
                    continue;
                }

                // Skip captures with bad SEE (4980-4981).
                if !pos.see_ge(mv, SEE_CAPTURE_MARGIN) {
                    continue;
                }
            }

            // Step 7. Make and search (4984-4994).
            // Capture the accumulator delta from the pre-move position, then push
            // the derived child accumulator after the move (popped after undo).
            let acc_delta = MoveDelta::from_move(pos, mv);
            self.nodes += 1; // sole node-count increment (worker do_move, 2072).
            let undo = pos.do_move_with_check(mv, gives_check);
            // `ss->currentMove` and the continuation-history pointers, all set by
            // the reference *inside* `do_move` (2090-2104) for every move —
            // qsearch moves included. A deeper node's `correction_value` reads the
            // continuation-correction plane at `(ss-2)` / `(ss-4)`, and an interior
            // node's move ordering / pruning reads the continuation-history plane
            // there too, so a qsearch ply that leaves these stale feeds a wrong
            // `cntcv` (and, transitively, a wrong corrected eval) once the
            // correction / continuation tables warm up — the second, non-pawn-key
            // half of that historical divergence, which only surfaces from depth 6.
            let moved = mv.moved_piece_after();
            self.stack[Self::si(ply)].current_move = mv; // ss->currentMove (2090).
            self.stack[Self::si(ply)].cont_hist =
                ContinuationHistory::plane_index(in_check, capture, moved, mv.to_sq());
            self.stack[Self::si(ply)].cont_corr =
                ContinuationCorrectionHistory::plane_index(moved, mv.to_sq());
            self.push_accumulator(pos, &acc_delta);
            let value = -self.qsearch(pos, ply + 1, -beta, -alpha);
            pos.undo_move(mv, undo);
            self.pop_accumulator();

            // An abort that fired inside this child's subtree makes `value`
            // untrustworthy; bail immediately rather than fold it in and keep
            // looping (the interior caller discards this node's value on abort).
            // Inert on the parity path.
            if self.stopped {
                return best_value;
            }

            // Step 8. New best move (4998-5024).
            if value > best_value {
                best_value = value;
                if value > alpha {
                    best_move = mv;
                    if pv_node {
                        self.update_pv(ply, mv);
                    }
                    if value < beta {
                        alpha = value;
                    } else {
                        break; // fail high
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // Step 9. Mate check + tail write (5027-5135).
        // -----------------------------------------------------------------
        // In check with no legal move ⇒ checkmate; the `moveCount == 0` form
        // (not the Stockfish `bestValue == -VALUE_INFINITE` form), 5073-5082.
        if in_check && move_count == 0 {
            return mated_in(ply);
        }

        if !is_decisive(best_value) && best_value > beta {
            best_value = (best_value + beta) / 2;
        }

        // Final TT write (5115-5117): `value_to_tt`-converted, the locally
        // computed `pvHit`, and `BOUND_LOWER`/`BOUND_UPPER` by fail-high.
        self.tt_store(
            tt_slot,
            pos_key,
            value_to_tt(best_value, ply),
            pv_hit,
            if best_value >= beta {
                Bound::Lower
            } else {
                Bound::Upper
            },
            DEPTH_QS,
            move16_of(best_move),
            unadjusted_static_eval,
        );

        best_value
    }

    /// `ss->pv->update(move, (ss+1)->pv)` (5015): the node's PV becomes `move`
    /// followed by the child's PV.
    fn update_pv(&mut self, ply: i32, mv: Move) {
        // `split_at_mut` hands out disjoint borrows of this node's cell and its
        // child's, so the child PV is copied in place (no `clone()` heap
        // allocation) into the parent's preallocated buffer.
        let parent = Self::si(ply);
        let (head, tail) = self.stack.split_at_mut(parent + 1);
        let cell = &mut head[parent];
        let child_pv = &tail[0].pv;
        cell.pv.clear();
        cell.pv.push(mv);
        cell.pv.extend_from_slice(child_pv);
    }
}

// -----------------------------------------------------------------------------
// Root search (`go depth 1..3`): iterative deepening + aspiration windows.
// -----------------------------------------------------------------------------
//
// `run_root` drives the reference `iterative_deepening` loop, entering the
// shared [`QSearch::search`] body at `nodeType == Root` for each iteration and
// aspiration re-search, reusing the same `QSearch` — its node counter,
// transposition table, search stack, worker histories, and NNUE network — so the
// interior search and child qsearch it recurses into share that state. See
// `crate::root` for scope and for the value types.

impl QSearch<'_> {
    /// Run the `go depth <limit_depth>` root path on `pos` (single thread,
    /// `MultiPV == 1`, no book) and return the [`RootOutcome`] the USI layer
    /// reports. The caller must have sized the transposition table (the fixture
    /// gate resizes it to the default 1024 MiB and clears it per `usinewgame`).
    ///
    /// Reproduces the reference `start_searching` → `iterative_deepening`
    /// control flow: TT generation bump, full search-stack reset, resign /
    /// declaration-win pre-search exits, then the iterative-deepening loop —
    /// real aspiration windows around each iteration's [`Self::search`] at
    /// `nodeType == Root`, the stable re-sort, the early mate/mated break, and
    /// the final ponder-extended PV.
    pub fn run_root(&mut self, pos: &Position, limit_depth: i32) -> RootOutcome {
        // --- start_searching ---
        // Advance the TT generation before searching (tt.new_search()). On the
        // Lazy-SMP path the driver hoists this bump out — one per
        // `go`, before any helper launches — and calls [`Self::run_worker`]
        // directly. Here, the direct single-worker path used by the parity
        // gates, it stays inline so the observable sequence (bump, then search)
        // is unchanged.
        self.tt.new_search();

        // The root-move list is built once from the legal moves (the reference's
        // `start_thinking`). No legal move ⇒ `bestmove resign` with mated_in(1).
        let root_moves = generate_root_moves(pos, self.generate_all_legal_moves);
        if root_moves.is_empty() {
            return RootOutcome {
                best_move: Move::resign(),
                score: mated_in(1),
                nodes: 0,
                pv: vec![Move::resign()],
                depth: 0,
                sel_depth: 0,
                kind: RootKind::Resign,
            };
        }

        // Declaration win at the root ⇒ `bestmove win` with mate_in(1). (1-ply
        // mate is intentionally *not* probed at the root.) This direct path is
        // the fixed-depth parity route and is only ever driven with the point /
        // `None` rules, for which `declaration_win` returns `Move::win()` or
        // nothing; the `TryRule` root shortcut (which emits an actual king move)
        // is owned by the driver's coordinator (`run_coordinated`), not here.
        if let Some(mv) = declaration_win(pos, &self.entering_king) {
            debug_assert_eq!(
                mv,
                Move::win(),
                "run_root is only driven with point/None entering-king rules"
            );
            return RootOutcome {
                best_move: Move::win(),
                score: mate_in(1),
                nodes: 0,
                pv: vec![Move::win()],
                depth: 0,
                sel_depth: 0,
                kind: RootKind::DeclarationWin,
            };
        }
        // (No book probe: no book.)

        // The single worker runs the whole iterative deepening.
        let result = self.run_worker(pos, root_moves, limit_depth);

        // --- back in start_searching (bestThread == this for a single worker) ---
        let mut best = result.best;
        let mut work = pos.clone();
        // Extend a length-1 PV with a ponder move (TT, then ponder_candidate).
        self.extract_ponder(&mut work, &mut best, result.ponder_candidate);

        RootOutcome {
            best_move: best.mv,
            score: best.uci_score,
            nodes: result.nodes,
            pv: best.pv.clone(),
            // At least 1: an abort during iteration 1 leaves `completed_depth`
            // zero, but a `bestmove` still went out at (partial) depth 1.
            depth: result.completed_depth.max(1),
            sel_depth: best.sel_depth,
            kind: RootKind::Normal,
        }
    }

    /// Run one Lazy-SMP worker's iterative deepening and return its
    /// [`WorkerResult`]. Shared by the direct [`Self::run_root`] path and the
    /// driver's parallel orchestration: the main worker and every helper call
    /// this on their own `QSearch`, each with its own copy of `root_moves`.
    ///
    /// The caller has already bumped the TT generation exactly once for this
    /// `go` (`tt.new_search()`) and built `root_moves` from the root's legal
    /// moves; this method performs the per-worker init (histories low-ply refill,
    /// stack reset, node/selDepth reset) the reference `pre_start_searching` /
    /// `iterative_deepening` prologue does, then runs the aspiration-windowed
    /// iterative-deepening loop. Every time-management block is gated on the
    /// installed [`SearchControl`] (a helper's is stop-only), so a helper simply
    /// never enters the soft-budget short-circuit.
    pub fn run_worker(
        &mut self,
        root_pos: &Position,
        mut root_moves: Vec<RootMove>,
        limit_depth: i32,
    ) -> WorkerResult {
        self.root_us = root_pos.side_to_move();
        self.read_tt = true;
        self.nodes = 0;
        self.sel_depth = 0;
        self.last_iteration_pv.clear();
        // Fresh abort state for this `go`; the counter is seeded so the first
        // checkpoint lands a full interval in (see [`CHECK_INTERVAL`]).
        self.stopped = false;
        self.calls_cnt = CHECK_INTERVAL;
        // Fresh time-management bookkeeping for this `go` (reference
        // `pre_start_searching` / per-thread reset, `yaneuraou-search.cpp`).
        self.completed_depth = 0;
        self.best_move_changes = 0.0;
        self.stop_on_ponderhit = false;
        self.ponderhit_synced = false;
        // lowPlyHistory is refilled to 98 per `go` (1525 / 2175).
        self.histories.low_ply.fill(98);

        // Full search-stack reset (1441-1463): every cell returns to the
        // pristine sentinel state the interior search's improving / hindsight /
        // continuation logic assumes — `static_eval == VALUE_NONE`, `follow_pv
        // == false`, the sentinel continuation planes, `stat_score` / counters
        // zeroed. A stale `static_eval` from a previous `go` would flip
        // `improving`. `ply` is then set to the signed distance from the root
        // (the reference's `(ss + i)->ply = i`; the sentinels keep ply 0).
        for cell in self.stack.iter_mut() {
            // Reset to the pristine sentinel state while preserving the
            // preallocated `pv` buffer so a later PV update never reallocates.
            let mut pv = std::mem::take(&mut cell.pv);
            pv.clear();
            *cell = SearchStackCell::default();
            cell.pv = pv;
        }
        for (i, cell) in self.stack.iter_mut().enumerate().skip(STACK_BASE) {
            cell.ply = (i - STACK_BASE) as i32;
        }

        // A private working copy; the caller's position is never mutated.
        let mut work = root_pos.clone();

        // Seed this worker's accumulator stack from the root once. `search<Root>`
        // restores `work` to the root after every iteration and never mutates
        // slot 0 (it is only ever the derivation source), so this stays valid for
        // the whole iterative-deepening loop; `acc_depth` is reset to 0 before
        // each root search below for good measure.
        self.seed_accumulator(&work);

        // --- iterative_deepening (1519-1918) ---
        //
        // `searchAgainCounter` is incremented only when `increaseDepth` is false
        // (1586-1591); `increaseDepth` is initialised true (`pre_start_searching`)
        // and cleared only inside the time-managed block, which a fixed-depth
        // `go` never enters — so both stay at their init here and the
        // `adjustedDepth` formula is unchanged on the parity path.
        let mut search_again_counter: i32 = 0;
        let mut increase_depth = true;
        // Reference iterative-deepening locals (`yaneuraou-search.cpp`):
        // the eval-stability time reduction, the aged best-move-change statistic,
        // the per-iteration best-value ring, and the depth at which the best move
        // last changed (`1410`).
        let mut time_reduction: f64 = 1.0;
        let mut tot_best_move_changes: f64 = 0.0;
        let mut iter_idx: usize = 0;
        let mut last_best_move_depth: i32 = 0;

        // Main-thread persistent inputs for the time-management block
        // (`yaneuraou-search.cpp`), carried in via the time control; the
        // fixed-depth / helper path uses the first-move-of-a-game sentinels, but
        // never reads them (the block is time-gated).
        let (best_previous_score, best_previous_average_score, previous_time_reduction, n_threads) =
            match &self.control.time {
                Some(tc) => (
                    tc.best_previous_score,
                    tc.best_previous_average_score,
                    tc.previous_time_reduction,
                    tc.n_threads,
                ),
                None => (VALUE_INFINITE, VALUE_INFINITE, 0.85, 1),
            };
        // `iterValue.fill(bestPreviousScore == VALUE_INFINITE ? 0 : ...)`
        // (`yaneuraou-search.cpp`).
        let iter_seed = if best_previous_score == VALUE_INFINITE {
            0
        } else {
            best_previous_score
        };
        let mut iter_value = [iter_seed; 4];

        // `ponder_candidate` — the previous iteration's `pv[1]` (1909-1910), a
        // fallback for `extract_ponder_from_tt` on a length-1 final PV.
        let mut ponder_candidate = Move::none();

        // `multiPV = min(options["MultiPV"], rootMoves.size())` (1489, 1509-1510).
        // The Skill-driven `max(multiPV, 4)` bump (1500-1507) is Skill-gated in
        // the pin (`if (skill.enabled())`, and Skill is `Skill(20, 0)` — disabled
        // — in the non-Stockfish build), so it is out of scope and skipped.
        let multi_pv = self.multi_pv.min(root_moves.len()).max(1);

        // The last *completed* iteration's root move — the stable result an
        // aborted search rolls back to. A `go depth N` (no
        // clock) never aborts, so after the loop this always holds the final
        // iteration's `root_moves[0]`, identical to reading `root_moves[0]`
        // directly — the fixed-depth parity path is unchanged.
        let mut completed_best: Option<RootMove> = None;
        // The last completed iteration's top-`multiPV` lines (for the coordinator's
        // throttled-final-PV re-emit).
        let mut completed_lines: Vec<RootMove> = Vec::new();
        // `rootDepth` of the last completed iteration — reported as `info depth`.
        let mut completed_depth = 0;
        // `uciPvSent` (1522, 1129): whether the current iteration's final PV was
        // emitted to the GUI. Read after the loop by the coordinator's fallback.
        let mut uci_pv_sent = false;

        let mut root_depth = 0;
        // `while (++rootDepth < MAX_PLY && !threads.stop && rootDepth <=
        // limits.depth)` (1535): the guard is on the *post-increment* value, so
        // `root_depth + 1 < MAX_PLY` and `root_depth < limit_depth`
        // (== `root_depth + 1 <= limit_depth`). `!self.stopped` mirrors
        // `!threads.stop`; it is never set on the fixed-depth path.
        while root_depth + 1 < MAX_PLY && !self.stopped && root_depth < limit_depth {
            root_depth += 1;

            // Age out the PV-variability metric each iteration (1564) and, when the
            // previous iteration did not deepen (increaseDepth cleared by the time
            // block), count a repeated depth (1587-1591). On the parity path
            // `increase_depth` is always true, so `search_again_counter` stays 0.
            tot_best_move_changes /= 2.0;
            if !increase_depth {
                search_again_counter += 1;
            }

            // Save the previous iteration's scores for the aspiration seed
            // (1576-1577); all `-VALUE_INFINITE` on the first iteration. Done once
            // per iteration, before the MultiPV loop.
            for rm in &mut root_moves {
                rm.previous_score = rm.score;
            }

            // `uciPvSent = false` at the start of each iteration (1565).
            uci_pv_sent = false;

            // The iteration's last search value (the last PV line's), read by the
            // MultiPV==1 early-mate break below (the reference's function-level
            // `bestValue`).
            let mut iter_best_value = -VALUE_INFINITE;

            // MultiPV loop (1595): a full root search per PV line. At MultiPV == 1
            // this runs once with `pv_idx == 0`, keeping the single-PV path
            // bit-identical.
            for pv_idx in 0..multi_pv {
                self.pv_idx = pv_idx;
                // pvFirst = pvIdx, pvLast = rootMoves.size() (1615-1616): shogi
                // uses no tbRank banding, so the finished head is [0..pvIdx] and
                // the active tail is [pvIdx..].
                self.sel_depth = 0;

                // Aspiration window (1655-1658), seeded from `rootMoves[pvIdx]`. On
                // iteration 1 the sentinels (averageScore == -VALUE_INFINITE,
                // meanSquaredScore == -VALUE_INFINITE²) make `delta` huge, so
                // alpha/beta clamp to the full (-INF, +INF) window naturally.
                let mut delta: Value =
                    5 + (root_moves[pv_idx].mean_squared_score.unsigned_abs() / 9000) as Value;
                let avg = root_moves[pv_idx].average_score;
                let mut alpha = (avg - delta).max(-VALUE_INFINITE);
                let mut beta = (avg + delta).min(VALUE_INFINITE);

                // fail-high count: each fail high shaves a ply off adjustedDepth.
                let mut failed_high_cnt = 0;

                // This PV line's search value, set by every aspiration attempt.
                let mut best_value;

                // Re-search until the value lands inside the window (1681-1783).
                loop {
                    let adjusted_depth =
                        1.max(root_depth - failed_high_cnt - 3 * (search_again_counter + 1) / 4);
                    self.root_delta = beta - alpha;
                    self.root_depth = root_depth;
                    // The root search always begins at accumulator depth 0; slot 0
                    // still holds the root refresh seeded above.
                    self.acc_depth = 0;
                    best_value = self.search(
                        &mut work,
                        0,
                        alpha,
                        beta,
                        adjusted_depth,
                        false,
                        true,
                        None,
                        Some(&mut root_moves),
                    );

                    // Stable re-sort of the active tail `rootMoves[pvIdx..]`
                    // (1714): score desc, previousScore desc — the
                    // RootMove::operator< equivalent. Non-PV moves carry
                    // -VALUE_INFINITE, so only the PV rises; the rest keep their
                    // order (and the finished head [0..pvIdx] is untouched).
                    root_moves[pv_idx..].sort_by(root_move_order);

                    // Aborted mid-window ⇒ break at once (1724). The sort above ran
                    // (it is safe: `root_moves` still refers to the previous
                    // iteration), so `root_moves[0]` remains a legal best-so-far.
                    if self.stopped {
                        break;
                    }

                    // Fail-high/low PV update before the re-search (1738-1758):
                    // main worker, MultiPV == 1, over a fail bound, past the node
                    // floor and interval gate, with OutputFailLHPV on.
                    if self.should_output_fail_lh(multi_pv, best_value, alpha, beta, root_depth) {
                        self.emit_pv(root_pos, &root_moves, pv_idx, root_depth, multi_pv);
                        self.last_pv_info_time = Instant::now();
                    }

                    // Fail low / high ⇒ widen and re-search; else break
                    // (1762-1778). The fail-low form is `beta = alpha; alpha =
                    // bestValue - delta` (NOT the Stockfish `(alpha+beta)/2`).
                    if best_value <= alpha {
                        beta = alpha;
                        alpha = (best_value - delta).max(-VALUE_INFINITE);
                        failed_high_cnt = 0;
                        // Reset stopOnPonderhit on a fail low (1768-1769).
                        // Without ponder it is already false; the write is
                        // ported so the ponder path needs no special case here.
                        self.stop_on_ponderhit = false;
                    } else if best_value >= beta {
                        alpha = (beta - delta).max(alpha);
                        beta = (best_value + delta).min(VALUE_INFINITE);
                        failed_high_cnt += 1;
                    } else {
                        break;
                    }
                    delta += delta / 3;
                }

                iter_best_value = best_value;

                // Stable-sort the finished head `rootMoves[..pvIdx+1]` (1791): a
                // later line may have out-scored an earlier one.
                root_moves[..pv_idx + 1].sort_by(root_move_order);

                // End-of-line PV update (1793-1823): main worker, on a stop / the
                // last line / a huge node count, not an aborted proven loss, past
                // the interval gate. `uciPvSent` records whether the *final* line
                // of the iteration was the one emitted.
                if self.pv_sink.is_some()
                    && (self.stopped
                        || pv_idx + 1 == multi_pv
                        || self.aggregate_nodes() > 10_000_000)
                    && !(self.stopped && is_loss(root_moves[0].uci_score))
                    && self.pv_interval_elapsed()
                {
                    self.emit_pv(root_pos, &root_moves, pv_idx, root_depth, multi_pv);
                    self.last_pv_info_time = Instant::now();
                    uci_pv_sent = pv_idx + 1 == multi_pv;
                }

                if self.stopped {
                    break;
                }
            } // MultiPV loop

            // Restore the neutral pvIdx so any later interior use is bit-neutral.
            self.pv_idx = 0;

            // The MultiPV loop broke on an abort (1826): this iteration is
            // incomplete, so discard it and keep the last completed ordering.
            if self.stopped {
                break;
            }

            // End of a completed iteration (the reference's `if (!threads.stop)`
            // block, 1831): seed `ss->followPV` for the next iteration (1838),
            // record the depth at which the best move last changed (1835-1836),
            // and snapshot this iteration's result as the new stable best.
            if self.last_iteration_pv.is_empty() || root_moves[0].pv[0] != self.last_iteration_pv[0]
            {
                last_best_move_depth = root_depth;
            }
            self.last_iteration_pv = root_moves[0].pv.clone();
            completed_best = Some(root_moves[0].clone());
            completed_lines = root_moves[..multi_pv].to_vec();
            completed_depth = root_depth;
            // Publish for `check_time`'s `completedDepth >= 1` gate (1833).
            self.completed_depth = root_depth;

            // Early mate / mated termination (1885-1900), MultiPV == 1 only: stop
            // once the search depth outruns 2.5× the mate distance. It cannot fire
            // before rootDepth 8, but it is cheap and faithful. Suppressed for
            // MultiPV > 1 (one PV finding a mate must not stop the others, 1882),
            // and under `go mate` (`!limits.mate` at 1892/1897): a mate search
            // keeps proving within its time budget rather than retiring early.
            if multi_pv == 1 && !self.mate_mode {
                if iter_best_value >= VALUE_TB_WIN_IN_MAX_PLY
                    && (VALUE_MATE - iter_best_value + 2) * 5 / 2 < root_depth
                {
                    break;
                }
                if iter_best_value <= -VALUE_TB_WIN_IN_MAX_PLY
                    && (iter_best_value + VALUE_MATE + 2) * 5 / 2 < root_depth
                {
                    break;
                }
            }

            // Mate-found stop under `go mate` (`yaneuraou-search.cpp`).
            // The pin keeps this branch under `#if STOCKFISH` because its normal
            // engine defers mate proofs to a separate mate-search engine; this
            // port has no such engine, so the branch is ported here so `go mate`
            // actually terminates on a proof instead of hanging (a bare / infinite
            // `go mate` has no time bound). In USI `limits.mate` is a millisecond
            // time budget, which makes the reference's
            // `VALUE_MATE - score <= 2 * limits.mate` distance bound degenerate
            // (any mate distance is far below any practical ms budget), so it
            // reduces to: once a completed iteration's exact score is a decisive
            // mate or mated, stop. `iter_best_value` here is the just-completed
            // iteration's converged score (the reference `score == usiScore`
            // exactness guard is implied by being at iteration completion).
            if self.mate_mode && (is_win(iter_best_value) || is_loss(iter_best_value)) {
                break;
            }

            // Ponder candidate (1909-1910): remember pv[1] for the fallback.
            if root_moves[0].pv.len() > 1 {
                ponder_candidate = root_moves[0].pv[1];
            }

            // Fold this iteration's best-move changes into the aged statistic and
            // reset for the next iteration (1936-1941).
            self.fold_best_move_changes(&mut tot_best_move_changes);

            // Do we have time for the next iteration, or should we stop now?
            // (`yaneuraou-search.cpp`). Only the main worker under active
            // time management, and only until the end time is fixed.
            let time_managed = self
                .control
                .time
                .as_ref()
                .is_some_and(|tc| tc.use_time_management && tc.tm.search_end == 0);
            if time_managed && !self.stopped && !self.stop_on_ponderhit {
                // Close the ponderhit checkpoint lag: the reference stamps
                // `tm.ponderhitTime` on the USI thread (`set_ponderhit`,
                // `yaneuraou-search.cpp`), so it is visible to *every*
                // subsequent time decision immediately. The port instead reconciles
                // the stamped instant into `tm.ponderhitTime` lazily, and until now
                // only inside `check_time` (~every 512 nodes). That left a window
                // where this budget block could reach `set_search_end` below with an
                // un-synced (go-origin) ponderhit time. Run the same one-time sync
                // here, before the decision, so the rounding origin always reflects
                // an arrived ponderhit. A no-op when no `ponderhit` arrived.
                self.sync_ponderhit();

                // A pondering search always deepens and, when over budget, arms
                // `stopOnPonderhit` instead of fixing the end time
                // (`yaneuraou-search.cpp`).
                let pondering = self.is_pondering();
                // Values needed for the budget; read before the mutable borrow.
                // nodesEffort divides by this worker's OWN node counter
                // (`size_t(nodes)`, the base-Worker member — `1954-1955`), not the
                // all-worker aggregate.
                let own_nodes = self.nodes.max(1);
                let effort = root_moves[0].effort;
                let single_root_move = root_moves.len() == 1;
                let tc = self.control.time.as_ref().expect("time control present");
                let optimum = tc.tm.optimum() as f64;
                let maximum = tc.tm.maximum() as f64;
                let elapsed = tc.tm.elapsed_from(Instant::now());

                // nodesEffort (1954-1955): the best root move's share of the nodes,
                // scaled to 100000.
                let nodes_effort = effort * 100_000 / own_nodes;

                // fallingEval (1957-1961): >1 when the score is dropping vs the
                // previous iterations, <1 when rising; clamped.
                let falling_eval = ((11.325
                    + 2.115 * (best_previous_average_score - iter_best_value) as f64
                    + 0.987 * (iter_value[iter_idx] - iter_best_value) as f64)
                    / 100.0)
                    .clamp(0.5688, 1.5698);

                // timeReduction (1966-1968): shorter when the best move is stable
                // across iterations.
                let k = 0.5189;
                let center = last_best_move_depth as f64 + 11.57;
                time_reduction =
                    0.723 + 0.79 / (1.104 + (-k * (completed_depth as f64 - center)).exp());
                let reduction = (1.455 + previous_time_reduction) / (2.2375 * time_reduction);

                // bestMoveInstability (1971) and highBestMoveEffort (1972).
                let best_move_instability =
                    1.04 + 1.8956 * tot_best_move_changes / n_threads as f64;
                let high_best_move_effort = if completed_depth >= 10 && nodes_effort >= 92425 {
                    0.666
                } else {
                    1.0
                };

                let mut total_time = optimum
                    * falling_eval
                    * reduction
                    * best_move_instability
                    * high_best_move_effort;

                // Cap the used time for a single legal root move (1977-1981).
                if single_root_move {
                    total_time = total_time.min(502.0);
                }

                // Over the budget (or the maximum)? While pondering, arm
                // `stopOnPonderhit` so the first `check_time` after the ponderhit
                // stops; otherwise fix the end time, rounding up to a whole second,
                // rather than stopping now (`2026-2036`). Else decide whether to
                // deepen — a pondering search always does.
                if elapsed as f64 > total_time.min(maximum) {
                    if pondering {
                        self.stop_on_ponderhit = true;
                    } else {
                        self.control
                            .time
                            .as_mut()
                            .expect("time control present")
                            .tm
                            .set_search_end(elapsed);
                    }
                } else {
                    increase_depth = pondering || elapsed as f64 <= total_time * 0.503;
                }
            }

            // iterValue ring update (2040-2041).
            iter_value[iter_idx] = iter_best_value;
            iter_idx = (iter_idx + 1) & 3;
        }

        // The worker's reported move is the last completed iteration's root move.
        // Only an abort during iteration 1 (before any iteration completed)
        // leaves `completed_best` empty, in which case `root_moves[0]` — the
        // best-so-far after the partial iteration's sort — is still a legal move.
        let best = completed_best.unwrap_or_else(|| root_moves[0].clone());
        if completed_lines.is_empty() {
            completed_lines = vec![best.clone()];
        }

        WorkerResult {
            best,
            completed_depth,
            ponder_candidate,
            nodes: self.nodes,
            uci_pv_sent,
            pv_lines: completed_lines,
            time_reduction,
        }
    }

    /// The aggregate node count across every worker (`threads.nodes_searched()`).
    /// With a Lazy-SMP tally installed this sums the helpers' last-published slots
    /// plus this worker's live `self.nodes`; on the single-worker path it is just
    /// `self.nodes`. Used for the PV-output `nodes` field and the `nodes > 10M`
    /// gates.
    fn aggregate_nodes(&self) -> u64 {
        match &self.node_tally {
            Some((slots, idx)) => {
                let mut total = self.nodes;
                for (i, s) in slots.iter().enumerate() {
                    if i != *idx {
                        total += s.load(Ordering::Relaxed);
                    }
                }
                total
            }
            None => self.nodes,
        }
    }

    /// `lastPvInfoTime + computed_pv_interval <= now()` (`yaneuraou-search.cpp`):
    /// whether the PV-output interval has elapsed since the last emit. Without a
    /// PV config (helpers / parity path) the gate is vacuously satisfied.
    fn pv_interval_elapsed(&self) -> bool {
        match &self.pv_config {
            Some(cfg) => self.last_pv_info_time + cfg.pv_interval <= Instant::now(),
            None => true,
        }
    }

    /// The reference fail-high/low PV-output gate (`yaneuraou-search.cpp`)
    /// bound to this worker's live state. Only the main worker (which owns the
    /// sink and PV config) can pass it.
    fn should_output_fail_lh(
        &self,
        multi_pv: usize,
        best_value: Value,
        alpha: Value,
        beta: Value,
        root_depth: i32,
    ) -> bool {
        let Some(cfg) = &self.pv_config else {
            return false;
        };
        fail_lh_pv_gate(
            self.pv_sink.is_some(),
            multi_pv,
            best_value,
            alpha,
            beta,
            self.aggregate_nodes(),
            root_depth,
            self.last_pv_info_time + cfg.pv_interval <= Instant::now(),
            cfg.output_fail_lh_pv,
        )
    }

    /// Build and emit the per-line PV `info` output — the reference
    /// `main_manager()->pv()` (`yaneuraou-search.cpp`). Builds one [`PvInfo`]
    /// per PV index `0..multiPV` (respecting the `depth == 1 && !updated && i > 0`
    /// skip and the ConsiderationMode TT-walk), then hands each to the sink. A
    /// no-op when no sink is installed (helpers / parity path).
    fn emit_pv(
        &mut self,
        root_pos: &Position,
        root_moves: &[RootMove],
        pv_idx: usize,
        depth: i32,
        multi_pv: usize,
    ) {
        if self.pv_sink.is_none() {
            return;
        }
        let nodes = self.aggregate_nodes();
        let infos = self.build_pv_infos(root_pos, root_moves, pv_idx, depth, multi_pv, nodes);
        if let Some(sink) = self.pv_sink.as_mut() {
            for info in &infos {
                sink.emit(info);
            }
        }
    }

    /// Assemble the `info` lines for `pv()` (`yaneuraou-search.cpp`): one
    /// per PV index, carrying the reported depth/score/bound and PV. The
    /// `updated`/`isExact` logic decides the bound marker, and ConsiderationMode
    /// routes the PV through [`Self::consideration_pv`].
    ///
    /// Public so the coordinator can build the final-PV fallback lines
    /// (`1289-1315`) from the chosen worker's result: it passes `pv_idx ==
    /// lines.len()`, which makes every line exact (no bound), and the aggregate
    /// node total.
    pub fn build_pv_infos(
        &self,
        root_pos: &Position,
        root_moves: &[RootMove],
        pv_idx: usize,
        depth: i32,
        multi_pv: usize,
        nodes: u64,
    ) -> Vec<PvInfo> {
        let consideration = self
            .pv_config
            .as_ref()
            .is_some_and(|c| c.consideration_mode);
        let mut out = Vec::with_capacity(multi_pv);
        for (i, rm) in root_moves.iter().enumerate().take(multi_pv) {
            // `updated` — this line has a real score this iteration (5712).
            let updated = rm.score != -VALUE_INFINITE;
            // Skip an un-searched non-first line at depth 1 (5714).
            if depth == 1 && !updated && i > 0 {
                continue;
            }
            // Reported depth / value (5717-5721).
            let d = if updated { depth } else { 1.max(depth - 1) };
            let mut v = if updated {
                rm.uci_score
            } else {
                rm.previous_score
            };
            if v == -VALUE_INFINITE {
                v = VALUE_DRAW;
            }
            // `isExact = i != pvIdx || !updated` (5738): only the currently-searched
            // line can carry a fail bound; every other line is exact.
            let is_exact = i != pv_idx || !updated;
            let bound = if is_exact {
                PvBound::Exact
            } else if rm.score_lowerbound {
                PvBound::Lower
            } else if rm.score_upperbound {
                PvBound::Upper
            } else {
                PvBound::Exact
            };
            let pv = if consideration {
                self.consideration_pv(root_pos, &rm.pv)
            } else {
                rm.pv.clone()
            };
            out.push(PvInfo {
                depth: d,
                sel_depth: rm.sel_depth,
                multipv: i + 1,
                score: v,
                bound,
                nodes,
                pv,
            });
        }
        out
    }

    /// The ConsiderationMode PV collector (`yaneuraou-search.cpp`): walk
    /// the root PV as far as it goes, then extend from the transposition table,
    /// stopping at a repetition (after ply 0), a TT miss, an illegal / `none` TT
    /// move, or a non-move sentinel. Returns the collected move sequence.
    ///
    /// The pin appends a repetition/terminal text marker (`to_usi_string(rep)`)
    /// to the PV string; this port surfaces the PV as moves only and stops at the
    /// same points — the marker is a cosmetic diagnostic, dropped here (a
    /// documented, minor divergence noted in the PR).
    fn consideration_pv(&self, root_pos: &Position, root_pv: &[Move]) -> Vec<Move> {
        let mut pos = root_pos.clone();
        let mut moves: Vec<Move> = Vec::new();
        let mut applied: Vec<(Move, attic_state::Undo)> = Vec::new();
        let mut ply = 0usize;
        while ply < MAX_PLY as usize {
            // Repetition ends the PV, except at the root (5761-5767).
            if ply >= 1 && pos.is_repetition(ply as u16) != RepetitionState::None {
                break;
            }
            // First follow the searched PV, then fall back to the TT (5773-5792).
            let m = if ply < root_pv.len() {
                root_pv[ply]
            } else {
                let key = pos.key();
                let side = pos.side_to_move().index() as u8;
                let (found, data, _writer) = self.tt.probe(key, side);
                if !found {
                    break;
                }
                // Widen O(1) and validate via the reference chain (`to_move` +
                // `pseudo_legal` + `legal`, 5789-5791): a hit only extends the PV
                // when the stored move is playable here.
                match pos.to_move(data.move16) {
                    Some(mm)
                        if mm.is_ok()
                            && pos.pseudo_legal(mm, self.generate_all_legal_moves)
                            && pos.is_legal(mm) =>
                    {
                        mm
                    }
                    _ => break,
                }
            };
            // A resign/win sentinel is not playable — stop before `do_move` (5796).
            if !m.is_ok() {
                break;
            }
            moves.push(m);
            let undo = pos.do_move(m);
            applied.push((m, undo));
            ply += 1;
        }
        // Unwind the walk so `root_pos`'s clone is left balanced (5808-5809).
        while let Some((m, undo)) = applied.pop() {
            pos.undo_move(m, undo);
        }
        moves
    }

    /// `RootMove::extract_ponder_from_tt` (`yaneuraou-search.cpp`): when the
    /// final PV is a bare bestmove, play it and look for a legal ponder move —
    /// first the child TT entry's move, and on a TT miss the `ponder_candidate`
    /// (the previous iteration's `pv[1]`, a YaneuraOu extension). The move is
    /// pushed only if legal; nothing here touches the node counter.
    ///
    /// Public so the Lazy-SMP driver can apply it to the *chosen*
    /// worker's PV after the thread vote, using this (the main worker's) shared
    /// transposition table.
    pub fn extract_ponder(
        &mut self,
        pos: &mut Position,
        best: &mut RootMove,
        ponder_candidate: Move,
    ) {
        if best.pv.len() != 1 {
            return;
        }
        let pv0 = best.pv[0];
        if !pv0.is_ok() {
            return;
        }
        let undo = pos.do_move(pv0);
        let key = pos.key();
        let side = pos.side_to_move().index() as u8;
        let (found, data, _writer) = self.tt.probe(key, side);
        if found {
            // Widen O(1) and validate via the reference chain (`to_move` +
            // `pseudo_legal` + `legal`): push the child TT move only if playable.
            if let Some(m) = pos.to_move(data.move16)
                && m.is_ok()
                && pos.pseudo_legal(m, self.generate_all_legal_moves)
                && pos.is_legal(m)
            {
                best.pv.push(m);
            }
        } else if ponder_candidate.is_ok() {
            // TT miss ⇒ fall back to the previous iteration's pv[1] (5896-5901),
            // pushed only if it is legal in the child position.
            let mut legal: Vec<Move> = Vec::new();
            pos.generate_legal_all(&mut legal);
            if legal.contains(&ponder_candidate) {
                best.pv.push(ponder_candidate);
            }
        }
        pos.undo_move(pv0, undo);
    }
}

// -----------------------------------------------------------------------------
// Interior main search (`search<PV/NonPV>`, non-root).
// -----------------------------------------------------------------------------
//
// A faithful port of the reference shared `search` body
// (`yaneuraou-search.cpp`). It runs at every ply reachable from `go
// depth 2..3`: [`Self::run_root`] enters it at `nodeType == Root` (ply 0), and
// its move loop recurses into it at `PV` / `NonPV` for the interior plies until
// `newDepth` reaches 0, where it dives into qsearch. The structurally-dead
// blocks (the singular-extension family, null-move verification, tablebase /
// upcoming-repetition, the IIR condition beyond its depth-6 guard) are omitted
// or inert exactly as the reference guards make them, with a `debug_assert`
// tripwire where the singular block would begin.
//
// This search reads and updates the one live [`WorkerHistories`]
// (`self.histories`) for its move ordering; the leaf qsearch it recurses into
// reads those same live tables, so an interior update is visible to
// a later qsearch. The depth-3 parity gate (`tests/depth3_parity.rs`) pins the
// whole path to the reference.

/// `NO_PIECE` continuation plane (`continuationHistory[0][0][NO_PIECE][SQ_ZERO]`):
/// the null-move sentinel plane, index `0` in this port's flat layout.
const NULL_MOVE_CONT_PLANE: usize = 0;

impl QSearch<'_> {
    /// True iff `m` is a *plain* capture in `pos` (`Position::capture`): a
    /// non-drop landing on an occupied square. In this engine `capture_stage ==
    /// capture`.
    fn is_capture(pos: &Position, m: Move) -> bool {
        !m.is_drop() && pos.board().get(m.to_sq()).is_some()
    }

    /// `is_shuffling(move, ss, pos)` (`yaneuraou-search.cpp`, the
    /// non-`STOCKFISH` shogi branch): whether `move` merely shuffles a piece
    /// back and forth, so its singular extension should be suppressed. Shogi has
    /// no 50-move rule and a drop is not a "round trip", so captures and drops
    /// are excluded outright. `capture` is the caller's already-computed
    /// `capture_stage(move)` (`capture_stage == capture` in this engine). It
    /// cannot return true below ply 18, but the full predicate is ported.
    fn is_shuffling(&self, mv: Move, capture: bool, ply: i32, pos: &Position) -> bool {
        if capture || mv.is_drop() {
            return false;
        }
        if pos.plies_from_null() <= 6 || ply < 18 {
            return false;
        }
        let s = Self::si(ply);
        let move2 = self.stack[s - 2].current_move;
        let move4 = self.stack[s - 4].current_move;
        if !move2.is_ok() || !move4.is_ok() || move2.is_drop() || move4.is_drop() {
            return false;
        }
        mv.from_sq() == move2.to_sq() && move2.from_sq() == move4.to_sq()
    }

    /// `ss->statScore` for one move (`yaneuraou-search.cpp`): the
    /// capture branch scores `863·PieceValue[captured]/128 + captureHistory`, the
    /// quiet branch `2·mainHistory + contHist[0] + contHist[1]`. `s` is the
    /// current stack index; `captured` is `undo.captured()` (the victim, present
    /// iff `capture`).
    fn move_stat_score(
        &self,
        us: Color,
        moved_piece: Piece,
        mv: Move,
        s: usize,
        capture: bool,
        captured: Option<Piece>,
    ) -> i32 {
        if let (true, Some(cap_piece)) = (capture, captured) {
            863 * piece_value(cap_piece) / 128
                + self
                    .histories
                    .capture
                    .get(moved_piece, mv.to_sq(), cap_piece)
        } else if capture {
            // A capture always carries a victim, so this arm is unreachable;
            // returning `0` keeps it panic-free instead of an `expect`.
            0
        } else {
            2 * self.histories.main.get(us, mv)
                + self.histories.continuation.get_at(
                    self.stack[s - 1].cont_hist,
                    moved_piece,
                    mv.to_sq(),
                )
                + self.histories.continuation.get_at(
                    self.stack[s - 2].cont_hist,
                    moved_piece,
                    mv.to_sq(),
                )
        }
    }

    /// `reduction(i, d, mn, delta)` (`yaneuraou-search.cpp`). Returns
    /// the reduction scaled by 1024. Reads [`Self::root_delta`].
    fn reduction(&self, improving: bool, d: i32, mn: i32, delta: i32) -> i32 {
        let reduction_scale = self.reductions[d as usize] * self.reductions[mn as usize];
        reduction_scale - delta * 585 / self.root_delta
            + (!improving as i32) * reduction_scale * 206 / 512
            + 1133
    }

    /// `correction_value(*this, pos, ss)` (`yaneuraou-search.cpp`):
    /// `12153·pcv + 8620·micv + 12355·(wnpcv + bnpcv) + 7982·cntcv`, where the
    /// first four are the side-to-move channel reads keyed by the position's
    /// partial keys and `cntcv` sums the `(ss-2)` / `(ss-4)` continuation
    /// correction reads at `[piece_on(prev_to)][prev_to]` when `(ss-1)`'s move is
    /// ok, else the constant `8`.
    ///
    /// Reads the **one** live correction set ([`Self::histories`]) — the same
    /// tables the interior search updates — so a leaf qsearch and an interior
    /// node compute the correction from identical state. On fresh
    /// tables (unified channels `0`, continuation-correction `6`) `cntcv` is `12`
    /// (prev move ok) or `8`, so `|cv| < 131072` and `cv / 131072 == 0` in
    /// [`to_corrected_static_eval`]; a unit test asserts this and the depth-1
    /// parity fixtures gate it end to end.
    fn correction_value(&self, pos: &Position, ply: i32) -> i32 {
        let us = pos.side_to_move();
        let pcv = self
            .histories
            .shared
            .correction_get(pos.pawn_key(), us, CorrChannel::Pawn);
        let micv =
            self.histories
                .shared
                .correction_get(pos.minor_piece_key(), us, CorrChannel::Minor);
        let wnpcv = self.histories.shared.correction_get(
            pos.non_pawn_key(Color::White),
            us,
            CorrChannel::NonPawnWhite,
        );
        let bnpcv = self.histories.shared.correction_get(
            pos.non_pawn_key(Color::Black),
            us,
            CorrChannel::NonPawnBlack,
        );

        let s = Self::si(ply);
        let prev_move = self.stack[s - 1].current_move;
        let cntcv = if prev_move.is_ok() {
            let to = prev_move.to_sq();
            match pos.board().get(to) {
                Some(pc) => {
                    self.histories.continuation_correction.get_at(
                        self.stack[s - 2].cont_corr,
                        pc,
                        to,
                    ) + self.histories.continuation_correction.get_at(
                        self.stack[s - 4].cont_corr,
                        pc,
                        to,
                    )
                }
                None => 8,
            }
        } else {
            8
        };

        12153 * pcv + 8620 * micv + 12355 * (wnpcv + bnpcv) + 7982 * cntcv
    }

    /// Top-level entry to the interior search for tests / smoke checks: reset the
    /// per-run counters and stack, then run `search` at ply 0 as a *non-root*
    /// (`root_node == false`) PV / NonPV node.
    ///
    /// The transposition table must already be sized. `root_delta` / `root_depth`
    /// are seeded from this window and depth; the returned value is the node's
    /// search score. The real search enters through [`Self::run_root`] instead —
    /// this helper drives the shared body directly for unit tests.
    pub fn run_search(
        &mut self,
        pos: &mut Position,
        alpha: Value,
        beta: Value,
        depth: i32,
        cut_node: bool,
        pv_node: bool,
    ) -> Value {
        self.nodes = 0;
        self.sel_depth = 0;
        self.root_us = pos.side_to_move();
        self.read_tt = true;
        self.root_delta = (beta - alpha).max(1);
        self.root_depth = depth;
        self.last_iteration_pv.clear();
        self.stopped = false;
        self.calls_cnt = CHECK_INTERVAL;
        for cell in self.stack.iter_mut() {
            // Preserve each cell's preallocated `pv` buffer across the reset.
            let mut pv = std::mem::take(&mut cell.pv);
            pv.clear();
            *cell = SearchStackCell::default();
            cell.pv = pv;
        }
        self.seed_accumulator(pos);
        self.search(pos, 0, alpha, beta, depth, cut_node, pv_node, None, None)
    }

    /// The shared `search<Root/PV/NonPV>` body (`yaneuraou-search.cpp`).
    /// `prior_captured` is the piece the move that reached this node captured
    /// (`pos.captured_piece()`), threaded from the caller's `do_move`.
    ///
    /// `root_moves` is `Some` **only** for the root call (`nodeType == Root`,
    /// `ss->ply == 0`); it carries the live [`RootMove`] list that the reference
    /// `iterative_deepening` owns, and its presence is the `rootNode` flag. Every
    /// recursive call passes `None`, so `root_node` is false below the root.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        pos: &mut Position,
        ply: i32,
        mut alpha: Value,
        mut beta: Value,
        mut depth: i32,
        cut_node: bool,
        pv_node: bool,
        prior_captured: Option<Piece>,
        mut root_moves: Option<&mut Vec<RootMove>>,
    ) -> Value {
        // `rootNode == (nodeType == Root)`; carried by the root-move list's
        // presence. `Root` is a PV node, so `pv_node` is also true at the root.
        let root_node = root_moves.is_some();
        let all_node = !(pv_node || cut_node);

        // Dive into qsearch when the depth reaches zero (2240-2241), preserving
        // PV-ness. qsearch runs at the *same* ply (ss), not ss+1.
        if depth <= 0 {
            self.pv_node = pv_node;
            self.read_tt = true;
            return self.qsearch(pos, ply, alpha, beta);
        }

        depth = depth.min(MAX_PLY - 1);
        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || alpha == beta - 1);
        debug_assert!(0 < depth && depth < MAX_PLY);
        debug_assert!(!(pv_node && cut_node));

        let s = Self::si(ply);

        // Timer / stop check (`yaneuraou-search.cpp`, the main-thread
        // `check_time`). Placed once per interior node; it only counts down and,
        // at a checkpoint, latches `stopped` from the flag / node ceiling / hard
        // deadline. Inert on the parity path.
        self.check_time();

        // -----------------------------------------------------------------
        // Step 1. Initialize node (2333-2410).
        // -----------------------------------------------------------------
        let in_check = pos.in_check();
        self.stack[s].in_check = in_check;
        let prior_capture = prior_captured.is_some();
        let us = pos.side_to_move();
        self.stack[s].move_count = 0;
        self.stack[s].ply = ply;
        let mut best_value = -VALUE_INFINITE;

        // ss->followPV (2355-2357): always true at the root; below the root it
        // tracks the previous iteration's PV via `last_iteration_pv`.
        let follow_pv = root_node
            || (ply >= 1
                && self.stack[s - 1].follow_pv
                && ((ply - 1) as usize) < self.last_iteration_pv.len()
                && self.stack[s - 1].current_move == self.last_iteration_pv[(ply - 1) as usize]);
        self.stack[s].follow_pv = follow_pv;

        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        // Steps 2 and 3 are both inside the reference `if (!rootNode)` block
        // (2416-2524): the root never draws or mate-distance-prunes.
        if !root_node {
            // -------------------------------------------------------------
            // Step 2. Immediate draw / max ply (2419-2461, non-root).
            // -------------------------------------------------------------
            let draw_type = pos.is_repetition(ply as u16);
            if draw_type != RepetitionState::None {
                if draw_type == RepetitionState::Draw {
                    return self.draw_value(RepetitionState::Draw, us) + value_draw(self.nodes);
                }
                return value_from_tt(self.draw_value(draw_type, us), ply);
            }
            // The reference folds `threads.stop` into this return
            // (`yaneuraou-search.cpp`): an aborted non-root node yields the
            // draw score without touching the TT. `stopped` is never set on the
            // parity path, so this reduces to the original max-ply guard there.
            if self.stopped || ply >= MAX_PLY || pos.ply() as i32 > self.max_moves_to_draw {
                return self.draw_value(RepetitionState::Draw, us) + value_draw(self.nodes);
            }

            // -------------------------------------------------------------
            // Step 3. Mate distance pruning (2520-2523).
            // -------------------------------------------------------------
            alpha = mated_in(ply).max(alpha);
            beta = mate_in(ply + 1).min(beta);
            if alpha >= beta {
                return alpha;
            }
        }

        // -----------------------------------------------------------------
        // Stack preparation (2535-2540).
        // -----------------------------------------------------------------
        let prev_move = self.stack[s - 1].current_move;
        let prev_sq = if prev_move.is_ok() {
            Some(prev_move.to_sq())
        } else {
            None
        };
        let mut best_move = Move::none();
        let prior_reduction = self.stack[s - 1].reduction;
        self.stack[s - 1].reduction = 0;
        self.stack[s].stat_score = 0;
        self.stack[s + 2].cutoff_cnt = 0;

        // -----------------------------------------------------------------
        // Step 4. Transposition table lookup (2543-2783).
        // -----------------------------------------------------------------
        let excluded_move = self.stack[s].excluded_move;
        let pos_key = pos.key();
        let side = us.index() as u8;
        // Capture the entry location once (the reference's Step-4 `ttWriter`);
        // every write in this node targets this exact slot, not a re-probe.
        let (tt_hit, tt_data, tt_slot) = self.tt.locate(pos_key, side);
        self.stack[s].tt_hit = tt_hit;
        // `ttData.move = rootNode ? rootMoves[pvIdx].pv[0] : ttHit ? ttData.move
        // : Move::none()` (2625). At the root the current PV line's move
        // (`rootMoves[pvIdx].pv[0]`) is treated as the TT move regardless of what
        // the probe returned; from iteration 2 the probe still HITS the entry
        // iteration 1 wrote, so `tt_data` (value/eval/bound/depth) is consumed as
        // usual. At MultiPV == 1 `pv_idx == 0`, so this is `rootMoves[0].pv[0]` —
        // bit-identical to the single-PV path.
        let tt_move = if let Some(rms) = root_moves.as_deref() {
            Some(rms[self.pv_idx].pv[0])
        } else if tt_hit {
            // O(1) widening (`Position::to_move`); the MovePicker's TT stage
            // re-validates with `pseudo_legal` + `is_legal`.
            pos.to_move(tt_data.move16)
        } else {
            None
        };
        let tt_value = if tt_hit {
            value_from_tt(tt_data.value, ply)
        } else {
            VALUE_NONE
        };
        if !excluded_move.is_ok() {
            self.stack[s].tt_pv = pv_node || (tt_hit && tt_data.is_pv);
        }
        // Snapshot of `ss->ttPv` for the pre-move-loop readers. Refreshed after
        // Step 9's verification search (which can flip the live field) and never
        // read past the start of the move loop, where every site reads live.
        let mut ttpv = self.stack[s].tt_pv;
        let tt_capture = tt_move.is_some_and(|m| Self::is_capture(pos, m));

        // Non-PV early TT cutoff (2675-2783).
        if !pv_node
            && !excluded_move.is_ok()
            && tt_data.depth > depth - (tt_value <= beta) as i32
            && is_valid(tt_value)
            && bound_matches(tt_data.bound, tt_value >= beta)
            && (cut_node == (tt_value >= beta) || depth > 5)
        {
            if let Some(ttm) = tt_move
                && tt_value >= beta
            {
                if !tt_capture {
                    update_quiet_histories(
                        &mut self.histories,
                        pos,
                        &self.stack[..],
                        s,
                        ttm,
                        (130 * depth - 71).min(1043),
                    );
                }
                if let Some(psq) = prev_sq
                    && self.stack[s - 1].move_count <= 4
                    && !prior_capture
                    && let Some(pc) = pos.board().get(psq)
                {
                    update_continuation_histories(
                        &mut self.histories,
                        &self.stack[..],
                        s - 1,
                        pc,
                        psq,
                        -2142,
                    );
                }
            }
            return tt_value;
        }

        // -----------------------------------------------------------------
        // Step 5. Mate-in-1 and declaration win (2881-2985).
        // -----------------------------------------------------------------
        let mut unadjusted_static_eval = VALUE_NONE;
        if !root_node
            && !tt_hit
            && !excluded_move.is_ok()
            && !in_check
            && let Some(mate_move) = pos.mate_1ply()
        {
            best_value = mate_in(ply + 1);
            self.tt_store(
                tt_slot,
                pos_key,
                best_value,
                ttpv,
                Bound::Exact,
                (MAX_PLY - 1).min(depth + 6),
                move16_of(mate_move),
                unadjusted_static_eval,
            );
            return best_value;
        }
        if (tt_move.is_none() || pv_node) && declaration_win(pos, &self.entering_king).is_some() {
            return mate_in(ply + 1);
        }

        // -----------------------------------------------------------------
        // Step 6. Static evaluation (2990-3110).
        // -----------------------------------------------------------------
        let correction_value = self.correction_value(pos, ply);
        let eval: Value;
        // Mirrors `ss->staticEval`. Kept in a local because the reference reads
        // it constantly, but re-synced after Step 9's verification search, which
        // re-enters this node and rewrites the stack cell.
        let mut static_eval: Value;
        let mut improving: bool;

        if in_check {
            static_eval = self.stack[s - 2].static_eval;
            self.stack[s].static_eval = static_eval;
            improving = false;
            // goto moves_loop (below): Steps 6b-11 (the only readers of `eval`)
            // are skipped in check.
        } else {
            if excluded_move.is_ok() {
                // Kept faithful to the pin: eval is the outer search's
                // ss->staticEval.
                static_eval = self.stack[s].static_eval;
                unadjusted_static_eval = static_eval;
                eval = static_eval;
            } else if tt_hit {
                unadjusted_static_eval = tt_data.eval;
                if !is_valid(unadjusted_static_eval) {
                    unadjusted_static_eval = self.static_eval(pos);
                } else if pv_node {
                    // USE_CLASSIC_EVAL re-eval on PV nodes (3032-3045).
                    unadjusted_static_eval = self.static_eval(pos);
                }
                let corrected = to_corrected_static_eval(unadjusted_static_eval, correction_value);
                static_eval = corrected;
                eval = if is_valid(tt_value) && bound_matches(tt_data.bound, tt_value > corrected) {
                    tt_value
                } else {
                    corrected
                };
            } else {
                unadjusted_static_eval = self.static_eval(pos);
                static_eval = to_corrected_static_eval(unadjusted_static_eval, correction_value);
                eval = static_eval;
                self.tt_store(
                    tt_slot,
                    pos_key,
                    VALUE_NONE,
                    ttpv,
                    Bound::None,
                    DEPTH_UNSEARCHED,
                    0,
                    unadjusted_static_eval,
                );
            }
            self.stack[s].static_eval = static_eval;

            // Eval-diff history update (3111-3118).
            if self.stack[s - 1].current_move.is_ok()
                && !self.stack[s - 1].in_check
                && !prior_capture
            {
                let eval_diff =
                    (-(self.stack[s - 1].static_eval + static_eval)).clamp(-214, 171) + 60;
                self.histories.main.update(
                    us.flip(),
                    self.stack[s - 1].current_move,
                    eval_diff * 10,
                );
                if !tt_hit
                    && let Some(psq) = prev_sq
                    && let Some(pc) = pos.board().get(psq)
                {
                    let not_pawn = pc.kind != PieceKind::Pawn || pc.promoted;
                    if not_pawn && !self.stack[s - 1].current_move.is_promote() {
                        self.histories
                            .shared
                            .pawn_update(pos.pawn_key(), pc, psq, eval_diff * 12);
                    }
                }
            }

            improving = static_eval > self.stack[s - 2].static_eval;
            let opponent_worsening = static_eval > -self.stack[s - 1].static_eval;

            // Hindsight depth adjustment (3161-3163).
            if prior_reduction >= 3 && !opponent_worsening {
                depth += 1;
            }
            if prior_reduction >= 2
                && depth >= 2
                && static_eval + self.stack[s - 1].static_eval > 173
            {
                depth -= 1;
            }

            // Step 7. Razoring (3176-3177).
            if !pv_node && eval < alpha - 502 - 306 * depth * depth {
                self.pv_node = false;
                self.read_tt = true;
                return self.qsearch(pos, ply, alpha, beta);
            }

            // Step 8. Futility pruning (3202-3212).
            let futility_mult = 76 - 21 * (!tt_hit) as i32;
            let margin = futility_mult * depth
                - (2686 * improving as i32 + 362 * opponent_worsening as i32) * futility_mult
                    / 1024
                + correction_value.abs() / 180600;
            if !ttpv
                && depth < 15
                && eval >= beta
                && eval - margin >= beta
                && (tt_move.is_none() || tt_capture)
                && !is_loss(beta)
                && !is_win(eval)
            {
                return (2 * beta + eval) / 3;
            }

            // Step 9. Null-move search with verification search (3234-3305).
            // The reference's `pos.non_pawn_material(us)` term stays inside
            // `#if STOCKFISH` at the pin, so it is absent here; `ss->ply >=
            // nmpMinPly` is live and is what disables the pass while a
            // verification search is in flight.
            if cut_node
                && static_eval >= beta - 16 * depth - 53 * improving as i32 + 378
                && !excluded_move.is_ok()
                && ply >= self.nmp_min_ply
                && !is_loss(beta)
            {
                let r = 7 + depth / 3;
                self.stack[s].current_move = Move::null();
                self.stack[s].cont_hist = NULL_MOVE_CONT_PLANE;
                self.stack[s].cont_corr = ContinuationCorrectionHistory::SENTINEL_PLANE;
                pos.do_null_move();
                // Prefetch the child TT cluster, matching the
                // reference's prefetch inside `do_null_move`
                // (`position.cpp` ~2483). A null move touches no accumulator, so
                // it bypasses `push_accumulator`'s prefetch and needs this one.
                // `do_null_move` has already flipped the side and updated the
                // key, so `pos.key()` / `pos.side_to_move()` are the post-null
                // child's. Pure hint: no effect on the search.
                self.tt
                    .prefetch(pos.key(), pos.side_to_move().index() as u8);
                let null_value = -self.search(
                    pos,
                    ply + 1,
                    -beta,
                    -beta + 1,
                    depth - r,
                    false,
                    false,
                    None,
                    None,
                );
                pos.undo_null_move();
                // Do not return unproven mate scores (3269-3271).
                if null_value >= beta && !is_win(null_value) {
                    // Shallow nodes, and nodes already inside a verification
                    // search, trust the pass outright (3278-3279).
                    if self.nmp_min_ply != 0 || depth < 16 {
                        return null_value;
                    }
                    debug_assert_eq!(
                        self.nmp_min_ply, 0,
                        "recursive null-move verification is not allowed (3281)"
                    );

                    // Verify the pass by re-searching this SAME node (same ply,
                    // same stack cell, no `do_move`) at the reduced depth with
                    // null-move pruning disabled until `ply` climbs past
                    // `nmp_min_ply`, and *not* as a cut node (3290-3295).
                    self.nmp_min_ply = ply + 3 * (depth - r) / 4;
                    let v = self.search(
                        pos,
                        ply,
                        beta - 1,
                        beta,
                        depth - r,
                        false,
                        false,
                        prior_captured,
                        None,
                    );
                    self.nmp_min_ply = 0;

                    if v >= beta {
                        return null_value;
                    }
                }
            }
            // The verification search above re-entered on this node's own `ss`,
            // so `ss->staticEval` / `ss->ttPv` may have been rewritten under us
            // (a fail-low re-entry applies `ss->ttPv |= (ss-1)->ttPv`). Every
            // reference read from here on is a LIVE `ss->` read, so refresh the
            // snapshots rather than carrying the pre-Step-9 values. Without a
            // verification search both are unchanged, so the parity path is
            // bit-identical.
            static_eval = self.stack[s].static_eval;
            ttpv = self.stack[s].tt_pv;
            improving |= static_eval >= beta;

            // Step 10. Internal iterative reduction (3310-3311).
            if !self.stack[s].follow_pv
                && !all_node
                && depth >= 6
                && tt_move.is_none()
                && prior_reduction <= 3
            {
                depth -= 1;
            }

            // Step 11. ProbCut (3348-3416).
            let prob_cut_beta = beta + 224 - 61 * improving as i32;
            if depth >= 3 && !is_decisive(beta) && !(is_valid(tt_value) && tt_value < prob_cut_beta)
            {
                let prob_cut_depth = depth - 4;
                let mut mp = MovePicker::new_probcut(
                    pos,
                    tt_move,
                    prob_cut_beta - static_eval,
                    self.generate_all_legal_moves,
                );
                while let Some(mv) = mp.next_move(pos, &self.histories) {
                    if mv == excluded_move || !pos.is_legal(mv) {
                        continue;
                    }
                    let acc_delta = MoveDelta::from_move(pos, mv);
                    self.nodes += 1;
                    let undo = pos.do_move(mv);
                    let moved = mv.moved_piece_after();
                    self.stack[s].current_move = mv;
                    self.stack[s].cont_hist =
                        ContinuationHistory::plane_index(in_check, true, moved, mv.to_sq());
                    self.stack[s].cont_corr =
                        ContinuationCorrectionHistory::plane_index(moved, mv.to_sq());
                    self.push_accumulator(pos, &acc_delta);
                    self.pv_node = false;
                    self.read_tt = true;
                    let mut value = -self.qsearch(pos, ply + 1, -prob_cut_beta, -prob_cut_beta + 1);
                    if value >= prob_cut_beta && prob_cut_depth > 0 {
                        value = -self.search(
                            pos,
                            ply + 1,
                            -prob_cut_beta,
                            -prob_cut_beta + 1,
                            prob_cut_depth,
                            !cut_node,
                            false,
                            undo.captured(),
                            None,
                        );
                    }
                    pos.undo_move(mv, undo);
                    self.pop_accumulator();
                    if value >= prob_cut_beta {
                        self.tt_store(
                            tt_slot,
                            pos_key,
                            value_to_tt(value, ply),
                            ttpv,
                            Bound::Lower,
                            prob_cut_depth + 1,
                            move16_of(mv),
                            unadjusted_static_eval,
                        );
                        if !is_decisive(value) {
                            return value - (prob_cut_beta - beta);
                        }
                    }
                }
            }
        }

        // ---- moves_loop: (in-check nodes resume here) ----

        // Step 12. Small ProbCut (3426-3429).
        let prob_cut_beta = beta + 416;
        if bound_matches(tt_data.bound, true)
            && tt_data.depth >= depth - 4
            && tt_value >= prob_cut_beta
            && !is_decisive(beta)
            && is_valid(tt_value)
            && !is_decisive(tt_value)
        {
            return prob_cut_beta;
        }

        // -----------------------------------------------------------------
        // Move-loop preparation (3439-3455).
        // -----------------------------------------------------------------
        // `contHist[i] = (ss-1-i)->continuationHistory` — held as flat plane
        // indices into the live `continuationHistory`, not snapshots, so a plane
        // updated by an earlier move's subtree is seen when a later stage
        // (QUIET_INIT) scores against it.
        let cont_planes: [usize; 6] = std::array::from_fn(|i| self.stack[s - 1 - i].cont_hist);
        let mut mp = MovePicker::new_main_search(
            pos,
            tt_move,
            depth,
            ply,
            cont_planes,
            self.generate_all_legal_moves,
        );

        let mut move_count = 0i32;
        let mut quiets_searched = SearchedList::new();
        let mut captures_searched = SearchedList::new();

        // -----------------------------------------------------------------
        // Step 13. Loop through the moves (3467-4248).
        // -----------------------------------------------------------------
        while let Some(mv) = mp.next_move(pos, &self.histories) {
            if mv == excluded_move {
                continue;
            }
            // The MovePicker yields only legal moves (the reference's
            // `if (!pos.legal(move)) continue;` is already applied).
            //
            // rootNode: skip moves not in the still-active tail
            // `rootMoves[pvIdx..]` — the reference `std::count(rootMoves.begin() +
            // pvIdx, rootMoves.begin() + pvLast, move)` membership test
            // (3502-3512), with `pvLast == rootMoves.size()`. This skips moves the
            // earlier PV lines already fixed. At pvIdx == 0 the tail is the whole
            // list, so at MultiPV == 1 (or the first line) it is bit-identical to
            // the previous whole-list membership test.
            if let Some(rms) = root_moves.as_deref()
                && !rms[self.pv_idx..].iter().any(|rm| rm.mv == mv)
            {
                continue;
            }
            move_count += 1;
            self.stack[s].move_count = move_count;
            if pv_node {
                self.stack[s + 1].pv.clear();
            }

            let mut extension = 0;
            let capture = Self::is_capture(pos, mv);
            let moved_piece = mv.moved_piece_after();
            let gives_check = pos.gives_check(mv);
            let mut new_depth = depth - 1;
            let delta = beta - alpha;
            let mut r = self.reduction(improving, depth, move_count, delta);
            // `ss->ttPv` is read LIVE here and at every in-loop site below
            // (`3564`): an earlier move's singular re-entry can flip it via its
            // fail-low `ss->ttPv |= (ss-1)->ttPv`, and the flip persists for the
            // rest of this node's move loop. The pre-loop `ttpv` snapshot must
            // NOT be used past this point.
            if self.stack[s].tt_pv {
                r += 1013;
            }

            // Step 14. Pruning at shallow depths (3577-3691). Skipped at the
            // root (`!rootNode`, 3577): every root move must be searched.
            if !root_node && !is_loss(best_value) {
                if move_count >= (3 + depth * depth) / (2 - improving as i32) {
                    mp.skip_quiet_moves();
                }
                let mut lmr_depth = new_depth - r / 1024;
                if capture || gives_check {
                    let victim = pos.board().get(mv.to_sq());
                    let capt_hist = match victim {
                        Some(v) => self.histories.capture.get(moved_piece, mv.to_sq(), v),
                        None => self.histories.capture.get_empty(moved_piece, mv.to_sq()),
                    };
                    if !gives_check && lmr_depth < 7 {
                        let futility_value = static_eval
                            + 218
                            + 223 * lmr_depth
                            + victim.map_or(0, piece_value)
                            + 131 * capt_hist / 1024;
                        if futility_value <= alpha {
                            continue;
                        }
                    }
                    let margin = (167 * depth + capt_hist * 34 / 1024).max(0);
                    if alpha >= VALUE_DRAW && !pos.see_ge(mv, -margin) {
                        continue;
                    }
                } else if !self.stack[s].follow_pv || !pv_node {
                    let mut history = self.histories.continuation.get_at(
                        self.stack[s - 1].cont_hist,
                        moved_piece,
                        mv.to_sq(),
                    ) + self.histories.continuation.get_at(
                        self.stack[s - 2].cont_hist,
                        moved_piece,
                        mv.to_sq(),
                    ) + self.histories.shared.pawn_get(
                        pos.pawn_key(),
                        moved_piece,
                        mv.to_sq(),
                    );
                    if history < -4097 * depth {
                        continue;
                    }
                    history += 71 * self.histories.main.get(us, mv) / 32;
                    lmr_depth += history / 3220;
                    let futility_value = static_eval
                        + 42
                        + 151 * (!best_move.is_ok()) as i32
                        + 120 * lmr_depth
                        + 86 * (static_eval > alpha) as i32;
                    if !in_check && lmr_depth < 13 && futility_value <= alpha {
                        if best_value <= futility_value
                            && !is_decisive(best_value)
                            && !is_win(futility_value)
                        {
                            best_value = futility_value;
                        }
                        continue;
                    }
                    lmr_depth = lmr_depth.max(0);
                    if !pos.see_ge(mv, -25 * lmr_depth * lmr_depth) {
                        continue;
                    }
                }
            }

            // Step 15. Singular extension (3736-3850). Search the ttMove's
            // singularity by re-entering THIS node with `move` excluded: if every
            // other move fails low under a reduced null-window search, the ttMove
            // is singular and we extend it. `(ttData.bound & BOUND_LOWER)` is a
            // BIT test — Exact carries the LOWER bit, so Exact and Lower both
            // pass. The reference reassigns `ttData.value = value_from_tt(...)` at
            // Step 4 (2635), so every `ttData.value` here is the ply-adjusted
            // value — `tt_value`, not the raw `tt_data.value`.
            if !root_node
                && Some(mv) == tt_move
                && !excluded_move.is_ok()
                && depth >= 6 + self.stack[s].tt_pv as i32
                && is_valid(tt_value)
                && !is_decisive(tt_value)
                && bound_matches(tt_data.bound, true)
                && tt_data.depth >= depth - 3
                && !self.is_shuffling(mv, capture, ply, pos)
            {
                let singular_beta =
                    tt_value - (60 + 66 * (self.stack[s].tt_pv && !pv_node) as i32) * depth / 55;
                let singular_depth = new_depth / 2;

                // Re-enter on the SAME node (no do_move, same ply, same stack
                // cell) with `move` excluded. Any ss field the inner search
                // overwrites is intentionally shared, exactly as the reference
                // re-enters on the same ss.
                self.stack[s].excluded_move = mv;
                self.pv_node = false;
                self.read_tt = true;
                let s_value = self.search(
                    pos,
                    ply,
                    singular_beta - 1,
                    singular_beta,
                    singular_depth,
                    cut_node,
                    false,
                    prior_captured,
                    None,
                );
                self.stack[s].excluded_move = Move::none();

                if s_value < singular_beta {
                    let corr_val_adj = correction_value.abs() / 210590;
                    let double_margin = -4 + 212 * pv_node as i32
                        - 182 * (!tt_capture) as i32
                        - corr_val_adj
                        - 906 * self.histories.tt_move.get() / 116517
                        - (ply > self.root_depth) as i32 * 44;
                    // `92 * ss->ttPv` reads ss->ttPv LIVE, after the re-entry:
                    // the inner search failed low here (value < singularBeta ==
                    // its beta, so value <= its alpha), so it applied
                    // `ss->ttPv |= (ss-1)->ttPv`. The guard and `singular_beta`
                    // above ran before the re-entry and see the original value.
                    let triple_margin = 73 + 320 * pv_node as i32 - 218 * (!tt_capture) as i32
                        + 92 * self.stack[s].tt_pv as i32
                        - corr_val_adj
                        - (ply > self.root_depth) as i32 * 45;

                    extension = 1
                        + (s_value < singular_beta - double_margin) as i32
                        + (s_value < singular_beta - triple_margin) as i32;

                    // Increment the NODE's remaining depth: the remaining moves'
                    // LMR and the final TT-store depth observe this bump.
                    depth += 1;
                }
                // Multi-cut pruning (3808-3811): the ttMove is assumed to fail
                // high, and if excluding it still fails high over beta, this is
                // not a singular node — prune the subtree with a softbound.
                else if s_value >= beta && !is_decisive(s_value) {
                    self.histories
                        .tt_move
                        .update((-424 - 107 * depth).max(-3375));
                    return s_value;
                }
                // Negative extensions (3832-3841). `ttData.value` is the
                // ply-adjusted value (see the guard note).
                else if tt_value >= beta {
                    extension = -3;
                } else if cut_node {
                    extension = -2;
                }
            }

            // Step 16. Make the move (3858-3932).
            // Snapshot the accumulator delta from the pre-move position, then
            // push the derived child after `do_move` (popped after `undo_move`).
            let acc_delta = MoveDelta::from_move(pos, mv);
            self.nodes += 1;
            let undo = pos.do_move_with_check(mv, gives_check);
            self.stack[s].current_move = mv;
            self.stack[s].cont_hist =
                ContinuationHistory::plane_index(in_check, capture, moved_piece, mv.to_sq());
            self.stack[s].cont_corr =
                ContinuationCorrectionHistory::plane_index(moved_piece, mv.to_sq());
            self.push_accumulator(pos, &acc_delta);
            new_depth += extension;

            // `nodeCount = rootNode ? nodes : 0` (3865): the node count *after*
            // this move's `do_move` increment, so `rm.effort` sums only the
            // subtree below the move. Unused off the root.
            let node_count = self.nodes;

            // `ss->ttPv` LIVE (3870): reflects any singular flip from this or an
            // earlier move in the loop.
            if self.stack[s].tt_pv {
                r -= 2819
                    + pv_node as i32 * 973
                    + (tt_value > alpha) as i32 * 905
                    + (tt_data.depth >= depth) as i32 * (935 + cut_node as i32 * 959);
            }
            r += 691;
            r -= move_count * 65;
            r -= correction_value.abs() / 25600;
            if cut_node {
                r += 3611 + 985 * tt_move.is_none() as i32;
            }
            if tt_capture {
                r += 1054;
            }
            if self.stack[s + 1].cutoff_cnt > 1 {
                r +=
                    251 + 1124 * (self.stack[s + 1].cutoff_cnt > 2) as i32 + 1042 * all_node as i32;
            }
            if Some(mv) == tt_move {
                r -= 2239;
            }
            let stat_score = self.move_stat_score(us, moved_piece, mv, s, capture, undo.captured());
            self.stack[s].stat_score = stat_score;
            r -= stat_score * 428 / 4096;
            if all_node {
                r += r * 273 / (256 * depth + 260);
            }

            // Step 17. Late-move reduction / extension (3945-4001).
            let mut value: Value = best_value;
            if depth >= 2 && move_count > 1 {
                let d = ((new_depth - r / 1024).min(new_depth + 2)).max(1) + pv_node as i32;
                self.stack[s].reduction = new_depth - d;
                self.pv_node = false;
                self.read_tt = true;
                value = -self.search(
                    pos,
                    ply + 1,
                    -(alpha + 1),
                    -alpha,
                    d,
                    true,
                    false,
                    undo.captured(),
                    None,
                );
                self.stack[s].reduction = 0;
                if value > alpha {
                    let do_deeper = d < new_depth && value > best_value + 48;
                    let do_shallower = value < best_value + 9;
                    new_depth += do_deeper as i32 - do_shallower as i32;
                    if new_depth > d {
                        value = -self.search(
                            pos,
                            ply + 1,
                            -(alpha + 1),
                            -alpha,
                            new_depth,
                            !cut_node,
                            false,
                            undo.captured(),
                            None,
                        );
                    }
                    update_continuation_histories(
                        &mut self.histories,
                        &self.stack[..],
                        s,
                        moved_piece,
                        mv.to_sq(),
                        1426,
                    );
                }
            }
            // Step 18. Full-depth search when LMR is skipped (4008-4021).
            else if !pv_node || move_count > 1 {
                if tt_move.is_none() {
                    r += 1057;
                }
                let nd = new_depth - (r > 4628) as i32 - (r > 5772 && new_depth > 2) as i32;
                self.pv_node = false;
                self.read_tt = true;
                value = -self.search(
                    pos,
                    ply + 1,
                    -(alpha + 1),
                    -alpha,
                    nd,
                    !cut_node,
                    false,
                    undo.captured(),
                    None,
                );
            }

            // PV full-window search (4034-4051).
            if pv_node && (move_count == 1 || value > alpha) {
                self.stack[s + 1].pv.clear();
                if Some(mv) == tt_move
                    && ((is_valid(tt_value) && is_decisive(tt_value) && tt_data.depth > 0)
                        || tt_data.depth > 1)
                {
                    new_depth = new_depth.max(1);
                }
                self.pv_node = true;
                self.read_tt = true;
                value = -self.search(
                    pos,
                    ply + 1,
                    -beta,
                    -alpha,
                    new_depth,
                    false,
                    true,
                    undo.captured(),
                    None,
                );
            }

            // Step 19. Undo move (4059).
            pos.undo_move(mv, undo);
            self.pop_accumulator();

            // A stop that fired inside this move's subtree makes the returned
            // value untrustworthy (`yaneuraou-search.cpp`): bail out at once,
            // updating neither the best move, the PV, the root-move list, nor the
            // TT. The caller unwinds the same way; `run_root` then keeps the last
            // completed iteration's ordering. `VALUE_DRAW == VALUE_ZERO`. Inert on
            // the parity path.
            if self.stopped {
                return VALUE_DRAW;
            }

            // Step 20. root-node special processing (4080-4172): fold this
            // move's result into its RootMove before the generic best-move
            // logic. `root_moves` is disjoint from `self`, so reading
            // `self.sel_depth` / `self.stack` while holding `rm` is sound.
            // A searched root move is always present in the root-move list, so
            // `find` never misses; the `if let` keeps the miss panic-free
            // instead of an `expect`.
            if let Some(rms) = root_moves.as_deref_mut()
                && let Some(rm) = rms.iter_mut().find(|rm| rm.mv == mv)
            {
                let ms_init: i64 = -(VALUE_INFINITE as i64 * VALUE_INFINITE as i64);
                rm.effort += self.nodes - node_count;
                rm.average_score = if rm.average_score != -VALUE_INFINITE {
                    (value + rm.average_score) / 2
                } else {
                    value
                };
                let value_sq = value as i64 * (value as i64).abs();
                rm.mean_squared_score = if rm.mean_squared_score != ms_init {
                    (value_sq + rm.mean_squared_score) / 2
                } else {
                    value_sq
                };

                if move_count == 1 || value > alpha {
                    rm.score = value;
                    rm.uci_score = value;
                    rm.sel_depth = self.sel_depth;
                    rm.score_lowerbound = false;
                    rm.score_upperbound = false;
                    if value >= beta {
                        rm.score_lowerbound = true;
                        rm.uci_score = beta;
                    } else if value <= alpha {
                        rm.score_upperbound = true;
                        rm.uci_score = alpha;
                    }
                    rm.pv.clear();
                    rm.pv.push(mv);
                    rm.pv.extend(self.stack[s + 1].pv.iter().copied());

                    // Record how often the best move changes within an iteration,
                    // for time management (4149-4150). MultiPV must only count the
                    // first PV line (`!pvIdx`), and only a *change* (`moveCount > 1`).
                    // Under Lazy-SMP each worker bumps its own shared slot so the
                    // main worker can sum every worker's count (`1936-1941`); the
                    // single-worker path keeps its own scalar.
                    if move_count > 1 && self.pv_idx == 0 {
                        match &self.best_move_tally {
                            Some((slots, idx)) => {
                                slots[*idx].fetch_add(1, Ordering::Relaxed);
                            }
                            None => self.best_move_changes += 1.0,
                        }
                    }
                } else {
                    // Not the PV: sink to the lowest value; the stable sort in
                    // `run_root` keeps its position.
                    rm.score = -VALUE_INFINITE;
                }
            }

            // Step 20 (cont). Check for a new best move (4185-4247).
            let inc = ((value == best_value)
                && (ply + 2 >= self.root_depth)
                && ((self.nodes as i32) & 14) == 0
                && !is_win(value.abs() + 1)) as i32;
            if value + inc > best_value {
                best_value = value;
                if value + inc > alpha {
                    best_move = mv;
                    // Update the node PV even on a fail high, but NOT at the root
                    // (`PvNode && !rootNode`, 4196): the root PV is the RootMove's.
                    if pv_node && !root_node {
                        self.update_pv(ply, mv);
                    }
                    if value >= beta {
                        self.stack[s].cutoff_cnt += ((extension < 2) || pv_node) as i32;
                        break;
                    }
                    if depth > 2 && depth < 14 && !is_decisive(value) {
                        depth -= 2;
                    }
                    alpha = value;
                }
            }
            if mv != best_move && move_count <= SEARCHED_LIST_CAPACITY as i32 {
                if capture {
                    captures_searched.push(mv);
                } else {
                    quiets_searched.push(mv);
                }
            }
        }

        // -----------------------------------------------------------------
        // Step 21-23. Mate check, stat updates, TT write (4270-4414).
        // -----------------------------------------------------------------
        if best_value >= beta && !is_decisive(best_value) && !is_decisive(alpha) {
            best_value = (best_value * depth + beta) / (depth + 1);
        }

        if move_count == 0 {
            best_value = if excluded_move.is_ok() {
                alpha
            } else {
                mated_in(ply)
            };
        } else if best_move.is_ok() {
            update_all_stats(
                &mut self.histories,
                pos,
                &self.stack[..],
                s,
                best_move,
                prev_sq,
                quiets_searched.as_slice(),
                captures_searched.as_slice(),
                depth,
                tt_move.unwrap_or(Move::none()),
                prior_capture,
            );
            if !pv_node {
                self.histories
                    .tt_move
                    .update(if Some(best_move) == tt_move {
                        805
                    } else {
                        -787
                    });
            }
        } else if !prior_capture && let Some(psq) = prev_sq {
            let mut bonus_scale = -232;
            bonus_scale -= self.stack[s - 1].stat_score / 108;
            bonus_scale += (59 * depth).min(454);
            bonus_scale += 169 * (self.stack[s - 1].move_count > 8) as i32;
            bonus_scale += 145 * (!in_check && best_value <= static_eval - 110) as i32;
            bonus_scale += 154
                * (!self.stack[s - 1].in_check && best_value <= -self.stack[s - 1].static_eval - 73)
                    as i32;
            bonus_scale = bonus_scale.max(0);
            let scaled_bonus = (135 * depth - 80).min(1400) * bonus_scale;
            if let Some(pc) = pos.board().get(psq) {
                update_continuation_histories(
                    &mut self.histories,
                    &self.stack[..],
                    s - 1,
                    pc,
                    psq,
                    scaled_bonus * 221 / 16384,
                );
                self.histories.main.update(
                    us.flip(),
                    self.stack[s - 1].current_move,
                    scaled_bonus * 235 / 32768,
                );
                let not_pawn = pc.kind != PieceKind::Pawn || pc.promoted;
                if not_pawn && !self.stack[s - 1].current_move.is_promote() {
                    self.histories.shared.pawn_update(
                        pos.pawn_key(),
                        pc,
                        psq,
                        scaled_bonus * 290 / 8192,
                    );
                }
            }
        } else if prior_capture
            && let Some(psq) = prev_sq
            && let Some(pc) = pos.board().get(psq)
            && let Some(cap) = prior_captured
        {
            self.histories.capture.update(pc, psq, cap, 1018);
        }

        if best_value <= alpha {
            self.stack[s].tt_pv = self.stack[s].tt_pv || self.stack[s - 1].tt_pv;
        }

        // TT write, guarded by `!excludedMove && !(rootNode && pvIdx)` (4387): at
        // the root, PV lines beyond the first (`pvIdx > 0`) must NOT overwrite the
        // TT — their reduced windows would poison the entry the first line wrote.
        // At MultiPV == 1 `pv_idx == 0`, so the second clause never fires: this is
        // bit-identical to the single-PV path.
        let skip_tt_write = excluded_move.is_ok() || (root_node && self.pv_idx != 0);
        if !skip_tt_write {
            let bound = if best_value >= beta {
                Bound::Lower
            } else if pv_node && best_move.is_ok() {
                Bound::Exact
            } else {
                Bound::Upper
            };
            let store_depth = if move_count != 0 {
                depth
            } else {
                (MAX_PLY - 1).min(depth + 6)
            };
            self.tt_store(
                tt_slot,
                pos_key,
                value_to_tt(best_value, ply),
                self.stack[s].tt_pv,
                bound,
                store_depth,
                move16_of(best_move),
                unadjusted_static_eval,
            );
        }

        // Correction-history update (4401-4409). CORRECTION_HISTORY_LIMIT/4 == 256.
        if !(in_check || (best_move.is_ok() && Self::is_capture(pos, best_move)))
            && (best_value > static_eval) == best_move.is_ok()
        {
            let sign = if best_move.is_ok() { 12 } else { 17 };
            let bonus = ((best_value - static_eval) * depth * sign / 128).clamp(-256, 256);
            update_correction_history(
                &mut self.histories,
                pos,
                &self.stack[..],
                s,
                1069 * bonus / 1024,
            );
        }

        best_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attic_eval::{
        FC_0_PADDED_INPUT_DIMS, HIDDEN_SIZE, HIDDEN1_DIMS, NetHeader, NnueNetwork,
        NnueNetworkBuilder,
    };
    use attic_state::{Piece, PieceKind, Square, parse_sfen};
    use attic_storage::{Bound, TTData, TranspositionTable};

    // --- synthetic zero-eval network -------------------------------------
    // A network whose every position evaluates to 0. The layer stacks route
    // `transformed[LANE_A] + transformed[LANE_B]` to the score (the same
    // shortcut the lib.rs tests use); with a zero feature transformer those
    // bytes are 0, so `evaluate` returns 0 everywhere. This lets the qsearch
    // arithmetic be exercised against a known static eval.
    const LANE_A: usize = 0;
    const LANE_B: usize = HIDDEN_SIZE / 2;

    fn zero_net() -> NnueNetwork {
        // One arena; the FT stays zero (so every position evaluates to 0) and
        // each stack routes `transformed[LANE_A] + transformed[LANE_B]` to the
        // score through the fc_0 shortcut row.
        let header = NetHeader {
            version: 0,
            hash: 0,
            arch_id: "synthetic".to_string(),
        };
        let mut b = NnueNetworkBuilder::new(header, [0u8; 32]);
        let row = HIDDEN1_DIMS * FC_0_PADDED_INPUT_DIMS;
        for s in 0..b.layer_stacks() {
            let w = b.fc_0_weights_mut(s);
            w[row + LANE_A] = 1;
            w[row + LANE_B] = 1;
        }
        b.build()
    }

    fn pos(sfen: &str) -> Position {
        parse_sfen(sfen).expect("valid SFEN")
    }

    fn fresh_tt() -> TranspositionTable {
        let mut t = TranspositionTable::new();
        t.resize(1); // 1 MiB is plenty for these tiny trees.
        t
    }

    fn legal_moves(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_legal_all(&mut v);
        v
    }

    fn captures(p: &Position) -> Vec<Move> {
        legal_moves(p)
            .into_iter()
            .filter(|&m| !m.is_drop() && p.board().get(m.to_sq()).is_some())
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn prewrite(
        table: &mut TranspositionTable,
        p: &Position,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: i32,
        mv: u16,
        eval: Value,
    ) {
        let key = p.key();
        let side = p.side_to_move().index() as u8;
        let generation = table.generation();
        let (_f, _d, w) = table.probe(key, side);
        w.write(key, value, pv, bound, depth, mv, eval, generation);
    }

    fn probe_root(table: &mut TranspositionTable, p: &Position) -> (bool, TTData) {
        let (f, d, _w) = table.probe(p.key(), p.side_to_move().index() as u8);
        (f, d)
    }

    /// Path to the real (never-committed) SFNN-1536 network, mirroring the
    /// `real_network.rs` integration test.
    fn real_nn_bin() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../eval/nn.bin")
    }

    /// Accumulator gate 2 (in-search equivalence): a test-mode search over a set
    /// of fixtures at a small fixed depth, with the accumulator self-check armed,
    /// so `evaluate_with(acc) == evaluate(refresh)` is asserted at EVERY
    /// evaluation point along the real search. Uses the real network (nonzero
    /// weights, so a wrong accumulator would change the eval and trip the check);
    /// skipped when `nn.bin` is absent, exactly like the other real-network tests.
    #[test]
    fn differential_accumulator_matches_refresh_at_every_eval_site() {
        let path = real_nn_bin();
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let net = attic_eval::load_network(&path).expect("real nn.bin loads");
        let tt = fresh_tt();

        // Start position plus hand-heavy / sparse fixtures that exercise drops,
        // promoted pieces, and frequent king moves (the refresh path).
        const FIXTURES: [&str; 3] = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
            "2k6/9/4+R4/8P/9/1n7/9/9/6K2 b B2Pgp 1",
        ];
        for sfen in FIXTURES {
            let mut p = pos(sfen);
            let mut q = QSearch::new(&net, &tt);
            q.set_verify_accumulator(true);
            // A small fixed depth is enough to cover qsearch, the interior move
            // loop, null moves, and ProbCut — every one of the six eval sites.
            q.run_search(&mut p, -VALUE_INFINITE, VALUE_INFINITE, 5, false, true);
        }
    }

    // -----------------------------------------------------------------
    // Value / depth helper unit tests.
    // -----------------------------------------------------------------

    #[test]
    fn value_tt_roundtrip_shifts_mate_scores_by_ply() {
        // Non-mate values are unchanged.
        assert_eq!(value_to_tt(123, 5), 123);
        assert_eq!(value_from_tt(123, 5), 123);
        // A win is pushed away from the root on store, pulled back on load.
        let win = mate_in(20); // 31980
        assert!(is_win(win));
        assert_eq!(value_to_tt(win, 5), win + 5);
        assert_eq!(value_from_tt(value_to_tt(win, 5), 5), win);
        // A loss symmetrically.
        let loss = mated_in(20); // -31980
        assert!(is_loss(loss));
        assert_eq!(value_to_tt(loss, 5), loss - 5);
        assert_eq!(value_from_tt(value_to_tt(loss, 5), 5), loss);
        // VALUE_NONE never validates.
        assert_eq!(value_from_tt(VALUE_NONE, 5), VALUE_NONE);
    }

    #[test]
    fn decisive_boundaries_match_the_pin() {
        assert!(is_win(VALUE_TB_WIN_IN_MAX_PLY));
        assert!(!is_win(VALUE_TB_WIN_IN_MAX_PLY - 1)); // == VALUE_MAX_EVAL
        assert!(is_loss(-VALUE_TB_WIN_IN_MAX_PLY));
        assert!(!is_loss(-VALUE_TB_WIN_IN_MAX_PLY + 1));
        assert!(is_decisive(mate_in(1)));
        assert!(!is_decisive(0));
        assert_eq!(VALUE_MAX_EVAL, 31753);
        assert_eq!(mate_in(1), 31999);
        assert_eq!(mated_in(0), -32000);
    }

    #[test]
    fn value_draw_dither_is_keyed_on_bit_one() {
        assert_eq!(value_draw(0), -1);
        assert_eq!(value_draw(1), -1); // bit 1 clear
        assert_eq!(value_draw(2), 1); // bit 1 set
        assert_eq!(value_draw(3), 1);
        assert_eq!(value_draw(4), -1);
    }

    #[test]
    fn draw_value_table_matches_defaults_and_contempt() {
        let net = zero_net();
        let table = fresh_tt();
        let mut q = QSearch::new(&net, &table);
        // Emulate a root search with Black to move.
        q.root_us = Color::Black;
        q.draw_contempt = DRAW_VALUE_OPTION_DEFAULT * PAWN_VALUE / 100; // -1

        // REPETITION_DRAW is contempt-signed: -1 for the root side, +1 for the
        // opponent.
        assert_eq!(q.draw_value(RepetitionState::Draw, Color::Black), -1);
        assert_eq!(q.draw_value(RepetitionState::Draw, Color::White), 1);
        // The other rows are the fixed `drawValueTable` defaults.
        assert_eq!(q.draw_value(RepetitionState::Win, Color::Black), VALUE_MATE);
        assert_eq!(
            q.draw_value(RepetitionState::Lose, Color::Black),
            -VALUE_MATE
        );
        assert_eq!(
            q.draw_value(RepetitionState::Superior, Color::Black),
            VALUE_MAX_EVAL
        );
        assert_eq!(
            q.draw_value(RepetitionState::Inferior, Color::Black),
            -VALUE_MAX_EVAL
        );
        // Contempt default truncates toward zero: -2 * 90 / 100 == -1.
        assert_eq!(q.draw_contempt, -1);
    }

    #[test]
    fn set_draw_value_signs_the_repetition_draw_row_per_side() {
        // `set_draw_value(contempt)` installs the pawn-scaled
        // per-`go` draw contempt; the REPETITION_DRAW row is `+contempt` for the
        // root side and `-contempt` for the opponent (the reference's symmetric
        // `drawValueTable[REPETITION_DRAW][us] = +dv`, `[~us] = -dv`).
        let net = zero_net();
        let table = fresh_tt();
        let mut q = QSearch::new(&net, &table);

        // Black to move at the root, DrawValueBlack = 500 ⇒ contempt = 500*90/100.
        q.root_us = Color::Black;
        let contempt = 500 * PAWN_VALUE / 100; // 450
        q.set_draw_value(contempt);
        assert_eq!(q.draw_value(RepetitionState::Draw, Color::Black), contempt);
        assert_eq!(q.draw_value(RepetitionState::Draw, Color::White), -contempt);

        // White to move at the root: the same mechanism, opposite root side.
        q.root_us = Color::White;
        let contempt_w = 300 * PAWN_VALUE / 100; // 270
        q.set_draw_value(contempt_w);
        assert_eq!(
            q.draw_value(RepetitionState::Draw, Color::White),
            contempt_w
        );
        assert_eq!(
            q.draw_value(RepetitionState::Draw, Color::Black),
            -contempt_w
        );

        // The win/loss/superior/inferior rows are untouched by contempt.
        assert_eq!(q.draw_value(RepetitionState::Win, Color::Black), VALUE_MATE);
        assert_eq!(
            q.draw_value(RepetitionState::Superior, Color::White),
            VALUE_MAX_EVAL
        );
    }

    // -----------------------------------------------------------------
    // Stand-pat.
    // -----------------------------------------------------------------

    const TWO_KINGS: &str = "4k4/9/9/9/9/9/9/9/4K4 b - 1";

    #[test]
    fn stand_pat_cutoff_adjusts_and_writes_unsearched_lower_bound() {
        let net = zero_net();
        let mut table = fresh_tt();
        let p = pos(TWO_KINGS);
        assert!(!p.in_check());

        // best_value == eval == 0 >= beta (-4); not decisive → (0 + -4)/2 = -2.
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), -5, -4, false, true)
        };
        assert_eq!(out.value, -2);
        assert_eq!(out.nodes, 0, "stand pat returns before any do_move");

        let (found, data) = probe_root(&mut table, &p);
        assert!(found);
        assert_eq!(data.depth, DEPTH_UNSEARCHED);
        assert_eq!(data.bound, Bound::Lower);
        assert_eq!(data.value, value_to_tt(-2, 0));
        assert_eq!(data.eval, 0);
        assert!(!data.is_pv);
    }

    // -----------------------------------------------------------------
    // Futility / SEE pruning — one position, three alpha regimes.
    // -----------------------------------------------------------------

    // Black lance on 9c captures the white pawn on 9b, which is defended by the
    // white gold on 9a: SEE == PawnValue - LanceValue == 90 - 315 == -225. It
    // is the only capture, is non-checking, and there is no 1-ply mate.
    const LANCE_SEE: &str = "g7k/p8/L8/9/9/9/9/9/8K b - 1";

    fn lance_capture(p: &Position) -> Move {
        // The lance captures the pawn on 9b (internal (8,1)). Landing on the
        // enemy 2nd rank, `generate_legal_all` lists both the promoting and
        // non-promoting variants; both share the same `from` (the lance) and
        // `to`, hence the same SEE and victim, so either represents the capture
        // the qsearch MovePicker searches (it takes the promotion-only one).
        let target = Square::new(8, 1).unwrap();
        captures(p)
            .into_iter()
            .find(|&m| m.to_sq() == target)
            .expect("the lance capture of 9b must exist")
    }

    #[test]
    fn futility_see_preconditions_hold() {
        let p = pos(LANCE_SEE);
        assert!(!p.in_check());
        assert!(p.mate_1ply().is_none());
        let m = lance_capture(&p);
        // futilityBase == staticEval(0) + 328 == 328; the victim is a pawn.
        assert_eq!(p.board().get(m.to_sq()).map(piece_value), Some(90));
        assert!(!p.see_ge(m, -73), "SEE(-225) must fail the -73 gate");
        assert!(p.see_ge(m, -328), "SEE(-225) must clear a -328 gate");
        assert!(!p.see_ge(m, -128), "SEE(-225) must fail a -128 gate");
    }

    #[test]
    fn futility_first_prune_floors_bestvalue_at_futility_value() {
        // alpha == 418 == futilityBase(328) + PawnValue(90): the first futility
        // test `futilityValue <= alpha` fires, flooring bestValue at 418.
        let net = zero_net();
        let table = fresh_tt();
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut pos(LANCE_SEE), 418, 419, false, true)
        };
        assert_eq!(out.value, 418);
        assert_eq!(out.nodes, 0, "the only capture is futility-pruned");
    }

    #[test]
    fn futility_see_branch_floors_bestvalue_at_min_alpha_base() {
        // alpha == 200: futilityValue(418) > alpha, but SEE(-225) < alpha -
        // futilityBase (== -128), so the SEE branch fires with the floor
        // `min(alpha, futilityBase) == min(200, 328) == 200`.
        let net = zero_net();
        let table = fresh_tt();
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut pos(LANCE_SEE), 200, 201, false, true)
        };
        assert_eq!(out.value, 200);
        assert_eq!(out.nodes, 0);
    }

    #[test]
    fn see_minus_73_skips_a_losing_capture() {
        // alpha == 0: futility does not prune (futilityValue 418 > 0, SEE clears
        // alpha - base == -328), so the capture reaches the `!see_ge(m, -73)`
        // gate and is skipped there (SEE -225 < -73). No node is searched.
        let net = zero_net();
        let table = fresh_tt();
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut pos(LANCE_SEE), 0, 1, false, true)
        };
        assert_eq!(out.value, 0);
        assert_eq!(out.nodes, 0, "the losing capture is SEE(-73)-skipped");
    }

    // -----------------------------------------------------------------
    // moveCount pruning + its prevSq / givesCheck exemptions.
    // -----------------------------------------------------------------

    // Black rook on 5e has three MVV-distinct captures (bishop 855, gold 540,
    // pawn 90), none checking, and no 1-ply mate. After either of the two
    // top-MVV captures White has no capture, so those children stand pat
    // immediately.
    const THREE_CAPTURES: &str = "6k2/9/9/9/b3R3g/9/4p4/9/K8 b - 1";

    #[test]
    fn movecount_prune_drops_the_third_capture() {
        let p = pos(THREE_CAPTURES);
        assert!(!p.in_check());
        assert!(p.mate_1ply().is_none());
        assert_eq!(captures(&p).len(), 3, "fixture must offer three captures");

        let net = zero_net();
        let table = fresh_tt();
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), 0, 1, false, true)
        };
        // Two captures searched (each child stand-pats: no deeper do_move); the
        // 3rd (lowest-MVV pawn) is pruned by `moveCount > 2` — it is NOT
        // SEE/futility-pruned (its SEE and futilityValue both clear the gates),
        // so `nodes == 2` isolates the moveCount prune.
        assert_eq!(out.nodes, 2);
        assert_eq!(out.value, 0);
    }

    #[test]
    fn prevsq_exemption_lets_a_recapture_past_movecount_pruning() {
        // Same position, but drive qsearch directly with `(ss-1)->currentMove`
        // set so `prevSq == 5g` (the pawn-capture's target). The 3rd capture is
        // then a "recapture" (to == prevSq), exempt from the moveCount block, so
        // it is searched too: `nodes == 3` rather than 2.
        let net = zero_net();
        let table = fresh_tt();
        let mut p = pos(THREE_CAPTURES);
        let mut q = QSearch::new(&net, &table);
        q.root_us = p.side_to_move();
        q.draw_contempt = DRAW_VALUE_OPTION_DEFAULT * PAWN_VALUE / 100;
        q.pv_node = false;
        q.read_tt = true;
        q.nodes = 0;
        // A dummy previous move whose destination is 5g (internal (4,6)).
        let prev = Move::make(
            Square::new(4, 5).unwrap(),
            Square::new(4, 6).unwrap(),
            Piece::new(PieceKind::Pawn, Color::White),
        );
        q.stack[STACK_BASE - 1].current_move = prev;
        let _ = q.qsearch(&mut p, 0, 0, 1);
        assert_eq!(q.nodes, 3, "the recapture is exempt from moveCount pruning");
    }

    #[test]
    fn correction_value_is_eval_neutral_on_fresh_tables() {
        // On fresh correction tables (unified channels all 0,
        // continuation-correction all 6), the correction contributes nothing to
        // the eval: `cv / 131072 == 0`, so `to_corrected_static_eval` matches
        // the uncorrected clamp. Checked both at ply 0 (prev move not ok ⇒
        // cntcv == 8) and at a deeper ply with a real previous move (cntcv ==
        // 12).
        let net = zero_net();
        let table = fresh_tt();
        let mut q = QSearch::new(&net, &table);

        const SFENS: &[&str] = &[
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "4k4/9/9/4p4/4G4/9/9/9/4K4 b - 1",
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        ];
        for sfen in SFENS {
            let p = pos(sfen);

            // Ply 0: (ss-1) has no move ⇒ cntcv == 8.
            q.stack[QSearch::si(0) - 1].current_move = Move::none();
            let cv0 = q.correction_value(&p, 0);
            assert_eq!(cv0 / 131072, 0, "cv/131072 must be 0 at ply 0 (`{sfen}`)");

            // A deeper ply with a real previous move landing on an occupied
            // square ⇒ cntcv == 12. Use black pawn 5e→5d on the second fixture,
            // but any move whose `to` holds a piece works; drive it structurally
            // by pushing the side-to-move's own piece key. Here we just set a
            // previous move whose destination is occupied on the board.
            let occupied = (0..81u8)
                .filter_map(attic_state::Square::from_index)
                .find(|&sq| p.board().get(sq).is_some())
                .unwrap();
            let piece = p.board().get(occupied).unwrap();
            let from = (0..81u8)
                .filter_map(attic_state::Square::from_index)
                .find(|&sq| p.board().get(sq).is_none())
                .unwrap();
            let prev = Move::make(from, occupied, piece);
            q.stack[QSearch::si(2) - 1].current_move = prev;
            let cv2 = q.correction_value(&p, 2);
            assert_eq!(cv2 / 131072, 0, "cv/131072 must be 0 at ply 2 (`{sfen}`)");

            // And the corrected eval equals the uncorrected one for a spread of
            // static evals.
            for v in [-31000, -500, 0, 123, 30000] {
                assert_eq!(
                    to_corrected_static_eval(v, cv0),
                    to_corrected_static_eval(v, 0),
                    "corrected eval must be unchanged (`{sfen}`, v={v})",
                );
                assert_eq!(
                    to_corrected_static_eval(v, cv2),
                    to_corrected_static_eval(v, 0),
                );
            }
        }
    }

    #[test]
    fn leaf_qsearch_correction_value_reads_live_worker_update() {
        // A leaf `correction_value` read must reflect an update made to the one
        // live worker correction table *before* the call — there is
        // no qsearch-private correction duplicate to fall out of sync.
        let net = zero_net();
        let table = fresh_tt();
        let mut q = QSearch::new(&net, &table);
        let p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        q.stack[QSearch::si(0) - 1].current_move = Move::none();

        let before = q.correction_value(&p, 0);
        let us = p.side_to_move();
        // Gravity-update the pawn channel to its limit (`+1024`), read live and
        // weighted by 12153 in `correction_value`.
        q.histories
            .shared
            .correction_update(p.pawn_key(), us, CorrChannel::Pawn, 1_000_000);
        let after = q.correction_value(&p, 0);
        assert_eq!(
            after - before,
            12153 * 1024,
            "correction_value must read the live worker pawn channel (+1024 × weight 12153)"
        );
    }

    #[test]
    fn leaf_qsearch_value_reflects_worker_correction_update() {
        // End-to-end: a quiet, not-in-check startpos qsearch stand-pats to the
        // corrected static eval. With the zero-eval network the uncorrected eval
        // is 0; a live worker correction update made before `run` must shift the
        // returned value, proving the leaf reads the one live table set.
        let net = zero_net();
        let p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");

        let base = {
            let table = fresh_tt();
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), -VALUE_INFINITE, VALUE_INFINITE, true, true)
                .value
        };

        let bumped = {
            let table = fresh_tt();
            let mut q = QSearch::new(&net, &table);
            let us = p.side_to_move();
            // Saturate the pawn channel so `cv / 131072` is a nonzero shift.
            for _ in 0..64 {
                q.histories.shared.correction_update(
                    p.pawn_key(),
                    us,
                    CorrChannel::Pawn,
                    1_000_000,
                );
            }
            q.run(&mut p.clone(), -VALUE_INFINITE, VALUE_INFINITE, true, true)
                .value
        };

        assert_eq!(base, 0, "fresh tables ⇒ eval-neutral correction");
        assert_ne!(
            base, bumped,
            "leaf qsearch value must reflect the live worker correction update"
        );
    }

    // Black gold on 9c captures the white pawn on 9b and thereby delivers
    // checkmate (the black lance on 9i backs it; the white king on 9a cannot
    // escape or capture). The capturing move gives check.
    const CAPTURE_MATE: &str = "k8/p8/G8/9/9/9/9/9/L7K b - 1";

    #[test]
    fn givescheck_exemption_searches_a_checking_capture_under_futility() {
        let p = pos(CAPTURE_MATE);
        assert!(!p.in_check());
        let caps = captures(&p);
        assert_eq!(caps.len(), 1);
        let m = caps[0];
        assert!(p.gives_check(m), "the mating capture gives check");
        assert!(p.see_ge(m, -73), "the mating capture clears the -73 gate");

        // Pre-write a (non-cutoff) TT entry so the node is a TT hit and the
        // 1-ply mate short-circuit is skipped — the mate must be found through
        // the *move loop*. alpha == 418 == futilityBase + PawnValue, so a
        // non-checking capture here would be futility-pruned; the givesCheck
        // exemption skips that block and searches the capture, finding mate.
        let net = zero_net();
        let mut table = fresh_tt();
        prewrite(
            &mut table,
            &p,
            0,
            false,
            Bound::None,
            DEPTH_UNSEARCHED,
            0,
            0,
        );
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), 418, 419, false, true)
        };
        assert_eq!(out.value, mate_in(1), "the checking capture mates");
        assert_eq!(out.nodes, 1, "only the mating capture is searched");
    }

    // -----------------------------------------------------------------
    // Quiet-move skip (a quiet TT move dropped by `!capture`).
    // -----------------------------------------------------------------

    const QUIET_CHECK: &str = "k8/9/9/9/4R4/9/9/9/8K b - 1";

    #[test]
    fn quiet_tt_move_is_dropped_by_the_capture_filter() {
        let p = pos(QUIET_CHECK);
        assert!(!p.in_check());
        assert!(captures(&p).is_empty(), "no captures in this position");
        // Rook 5e→9e: a quiet move that checks the white king on 9a.
        let quiet_check = Move::make(
            Square::new(4, 4).unwrap(),
            Square::new(8, 4).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert!(p.gives_check(quiet_check));
        assert!(legal_moves(&p).contains(&quiet_check));

        let net = zero_net();
        let mut table = fresh_tt();
        // Store the quiet check as the TT move (non-cutoff entry).
        prewrite(
            &mut table,
            &p,
            0,
            false,
            Bound::None,
            DEPTH_UNSEARCHED,
            move16_of(quiet_check),
            0,
        );
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), 0, 1, false, true)
        };
        // The MovePicker yields the quiet check first; givesCheck skips the
        // futility block, then `!capture` drops it. Nothing is searched.
        assert_eq!(out.nodes, 0);
        assert_eq!(out.value, 0);
    }

    // -----------------------------------------------------------------
    // Mate paths.
    // -----------------------------------------------------------------

    // Black to move and checkmated: a head-gold wall backed by the white king.
    const BLACK_MATED: &str = "4K4/3ggg3/4k4/9/9/9/9/9/9 b - 1";

    #[test]
    fn no_evasion_in_check_returns_mated_in_ply() {
        let p = pos(BLACK_MATED);
        assert!(p.in_check());
        assert!(legal_moves(&p).is_empty(), "fixture must be checkmate");

        let net = zero_net();
        let table = fresh_tt();
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), -VALUE_INFINITE, VALUE_INFINITE, true, true)
        };
        assert_eq!(out.value, mated_in(0)); // -VALUE_MATE
        assert_eq!(out.nodes, 0);
    }

    // Black to move with a 1-ply gold-drop mate (G*8a).
    const MATE_IN_1: &str = "k8/9/G1N6/9/9/9/9/9/8K b G 1";

    #[test]
    fn mate_1ply_short_circuits_with_exact_tt_write() {
        let p = pos(MATE_IN_1);
        assert!(!p.in_check());
        assert!(p.mate_1ply().is_some());

        let net = zero_net();
        let mut table = fresh_tt();
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), -VALUE_INFINITE, VALUE_INFINITE, true, true)
        };
        assert_eq!(out.value, mate_in(1)); // mate_in(ss->ply + 1) at ply 0
        assert_eq!(out.nodes, 0, "the mate is found before any do_move");

        let (found, data) = probe_root(&mut table, &p);
        assert!(found);
        assert_eq!(data.bound, Bound::Exact);
        assert_eq!(data.depth, DEPTH_QS);
        assert_eq!(data.value, mate_in(1), "raw mate score, not value_to_tt'd");
        assert_eq!(data.eval, VALUE_NONE, "unadjustedStaticEval is still NONE");
        assert_eq!(data.move16, move16_of(p.mate_1ply().unwrap()));
    }

    // -----------------------------------------------------------------
    // Repetition draws + the ±1 dither.
    // -----------------------------------------------------------------

    #[test]
    fn max_ply_returns_draw_with_dither() {
        let net = zero_net();
        let table = fresh_tt();
        let mut p = pos(TWO_KINGS);
        let mut q = QSearch::new(&net, &table);
        q.root_us = p.side_to_move(); // Black
        q.draw_contempt = DRAW_VALUE_OPTION_DEFAULT * PAWN_VALUE / 100; // -1
        q.pv_node = false;
        q.read_tt = true;

        // At ply == MAX_PLY the node returns draw_value(DRAW, us) + dither.
        q.nodes = 0;
        assert_eq!(q.qsearch(&mut p, MAX_PLY, -1, 0), -1 + value_draw(0)); // -2
        q.nodes = 2;
        assert_eq!(q.qsearch(&mut p, MAX_PLY, -1, 0), -1 + value_draw(2)); // 0
    }

    #[test]
    fn repetition_draw_is_detected_with_dither() {
        let net = zero_net();
        let table = fresh_tt();
        // Shuffle both kings back and forth. The four-ply cycle is period-4, so
        // the position after six plies equals the position after two plies — a
        // repetition four plies back whose earlier occurrence (ply 2) is
        // strictly *after* the search root (ply 0). The reference reports that
        // as an ordinary draw at search ply 6 (`repetition == 4 < 6`); a bare
        // two-fold landing *on* the root would not be scored (`4 < 4` is false).
        let mut p = pos(TWO_KINGS);
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let step = |from: (u8, u8), to: (u8, u8), pc: Piece| {
            Move::make(
                Square::new(from.0, from.1).unwrap(),
                Square::new(to.0, to.1).unwrap(),
                pc,
            )
        };
        p.do_move(step((4, 8), (3, 8), bk));
        p.do_move(step((4, 0), (3, 0), wk));
        p.do_move(step((3, 8), (4, 8), bk));
        p.do_move(step((3, 0), (4, 0), wk));
        // Two more plies open the next cycle so ply 6 == ply 2 (dist 4).
        p.do_move(step((4, 8), (3, 8), bk));
        p.do_move(step((4, 0), (3, 0), wk));
        assert_eq!(
            p.is_repetition(6),
            RepetitionState::Draw,
            "a repetition strictly after the search root is an ordinary draw"
        );
        // The `ply == distance` case is the one the two repetition
        // configurations disagree on: the non-QUICK_DRAW gate (`repetition <
        // ply`) suppresses it, QUICK_DRAW has no gate at all. Either way the
        // qsearch assertion below (at ply 6) is unaffected.
        #[cfg(not(feature = "quick-draw"))]
        assert_eq!(
            p.is_repetition(4),
            RepetitionState::None,
            "the same repetition reaching to the root (ply == distance) is not a draw"
        );
        #[cfg(feature = "quick-draw")]
        assert_eq!(
            p.is_repetition(4),
            RepetitionState::Draw,
            "QUICK_DRAW adjudicates the same repetition without a root-distance gate"
        );

        let mut q = QSearch::new(&net, &table);
        q.root_us = p.side_to_move(); // Black
        q.draw_contempt = DRAW_VALUE_OPTION_DEFAULT * PAWN_VALUE / 100; // -1
        q.pv_node = false;
        q.read_tt = true;
        q.nodes = 0;
        // draw_value(DRAW, Black=root_us) + value_draw(0) == -1 + -1 == -2.
        assert_eq!(q.qsearch(&mut p, 6, -1, 0), -2);
    }

    // -----------------------------------------------------------------
    // ReadTT=false ignores hits; determinism.
    // -----------------------------------------------------------------

    #[test]
    fn read_tt_false_ignores_a_cutoff_entry() {
        let net = zero_net();
        let mut table = fresh_tt();
        let p = pos(TWO_KINGS);
        // A lower-bound entry at DEPTH_QS with value 500 >= beta triggers the
        // non-PV early cutoff when ReadTT is honoured.
        prewrite(&mut table, &p, 500, false, Bound::Lower, DEPTH_QS, 0, 0);

        let with_tt = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), 399, 400, false, true)
        };
        assert_eq!(with_tt.value, 500, "TT cutoff returns the stored bound");
        assert_eq!(with_tt.nodes, 0);

        let without_tt = {
            let mut q = QSearch::new(&net, &table);
            q.run(&mut p.clone(), 399, 400, false, false)
        };
        assert_eq!(without_tt.value, 0, "ReadTT=false ignores the entry");
    }

    #[test]
    fn search_is_deterministic_across_runs() {
        let net = zero_net();
        let run_once = || {
            let table = fresh_tt();
            let mut q = QSearch::new(&net, &table);
            let out = q.run(&mut pos(THREE_CAPTURES), 0, 1, false, true);
            (out.value, out.nodes)
        };
        assert_eq!(run_once(), run_once());
    }

    // The reference folds EVERY worker's `bestMoveChanges` into `totBestMoveChanges`
    // and zeroes each, on the main thread only (`yaneuraou-search.cpp`).
    #[test]
    fn fold_best_move_changes_sums_and_zeroes_every_slot() {
        let net = zero_net();
        let table = fresh_tt();

        // Four workers; the main worker (slot 0) folds them all.
        let slots: Arc<Vec<AtomicU64>> = Arc::new((0..4).map(|_| AtomicU64::new(0)).collect());
        slots[0].store(2, Ordering::Relaxed);
        slots[1].store(3, Ordering::Relaxed);
        slots[2].store(0, Ordering::Relaxed);
        slots[3].store(5, Ordering::Relaxed);

        let mut main = QSearch::new(&net, &table);
        main.set_best_move_tally(Arc::clone(&slots), 0);
        let mut tot = 1.0; // a pre-existing aged statistic is added to, not replaced
        main.fold_best_move_changes(&mut tot);
        assert_eq!(tot, 1.0 + 10.0, "main folds every worker's count (2+3+0+5)");
        for (i, s) in slots.iter().enumerate() {
            assert_eq!(s.load(Ordering::Relaxed), 0, "slot {i} reset by the fold");
        }

        // A helper (slot != 0) neither folds nor resets — the main worker owns that.
        slots[1].store(7, Ordering::Relaxed);
        let mut helper = QSearch::new(&net, &table);
        helper.set_best_move_tally(Arc::clone(&slots), 1);
        let mut htot = 4.0;
        helper.fold_best_move_changes(&mut htot);
        assert_eq!(htot, 4.0, "a helper folds nothing");
        assert_eq!(
            slots[1].load(Ordering::Relaxed),
            7,
            "a helper leaves its slot for the main worker to read+zero"
        );

        // The single-worker path (no tally) folds its own scalar and zeroes it —
        // bit-identical to the pre-Lazy-SMP code.
        let mut solo = QSearch::new(&net, &table);
        solo.best_move_changes = 6.0;
        let mut stot = 2.0;
        solo.fold_best_move_changes(&mut stot);
        assert_eq!(stot, 8.0);
        assert_eq!(solo.best_move_changes, 0.0);
    }

    // A `ponderhit` arriving between `check_time` checkpoints must be reflected by
    // the very next iterative-deepening budget decision — not only after the next
    // checkpoint. `sync_ponderhit`, now called at the budget-block entry as well as
    // in `check_time`, copies the stamped instant into `tm.ponderhit_time` the
    // moment the ponder flag is seen cleared (the reference `set_ponderhit`
    // ordering, `yaneuraou-search.cpp`).
    #[test]
    fn sync_ponderhit_copies_the_stamped_instant_before_the_budget_decision() {
        let net = zero_net();
        let table = fresh_tt();
        let start = Instant::now();

        // A ponderhit has arrived (flag cleared, instant stamped) but no checkpoint
        // has synced it yet.
        let sig = Arc::new(PonderSignal::new(true));
        sig.ponderhit();
        let stamped = sig.hit_at().expect("ponderhit stamped an instant");

        let input = crate::timeman::TimeInput {
            time_us: 0,
            inc_us: 0,
            byoyomi_us: 0,
            movetime: 1000,
            rtime: 0,
            network_delay: 0,
            network_delay2: 0,
            minimum_thinking_time: 0,
            slow_mover: 100,
            round_up_to_fullsecond: false,
            usi_ponder: true,
            stochastic_ponder: false,
            ply: 1,
            max_moves_to_draw: 100_000,
            start_time: start,
        };
        let tm = TimeManagement::init(&input, &mut crate::book::Prng::new(1));
        assert_eq!(
            tm.ponderhit_time, tm.start_time,
            "unsynced: the rounding origin is still go-time"
        );

        let mut q = QSearch::new(&net, &table);
        q.set_control(SearchControl {
            stop: None,
            ponder: Some(Arc::clone(&sig)),
            node_limit: None,
            time: Some(TimeControl {
                tm,
                use_time_management: true,
                movetime: None,
                n_threads: 1,
                best_previous_score: VALUE_INFINITE,
                best_previous_average_score: VALUE_INFINITE,
                previous_time_reduction: 0.85,
            }),
        });

        // The budget-block sync copies the stamped instant into `tm.ponderhit_time`.
        q.sync_ponderhit();
        let synced = q.control.time.as_ref().unwrap().tm.ponderhit_time;
        assert_eq!(
            synced, stamped,
            "the sync copies the stamped ponderhit instant"
        );
        assert_ne!(synced, start, "the rounding origin advanced off go-time");

        // Idempotent: a later `check_time` sync is a no-op.
        q.sync_ponderhit();
        assert_eq!(q.control.time.as_ref().unwrap().tm.ponderhit_time, stamped);
    }

    // -----------------------------------------------------------------
    // Depth-1 root — the pre-search skipped exits. Both return before any
    // evaluation or do_move, so a synthetic zero network suffices.
    // -----------------------------------------------------------------

    #[test]
    fn run_root_resigns_with_no_legal_move() {
        let net = zero_net();
        let table = fresh_tt();
        // Black is checkmated: no legal move ⇒ bestmove resign.
        let p = pos("4K4/3ggg3/4k4/9/9/9/9/9/9 b - 1");
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run_root(&p, 1)
        };
        assert_eq!(out.kind, RootKind::Resign);
        assert_eq!(out.best_move, Move::resign());
        assert_eq!(out.score, mated_in(1));
        assert_eq!(out.nodes, 0);
    }

    #[test]
    fn run_root_declares_a_nyugyoku_win() {
        let net = zero_net();
        let table = fresh_tt();
        // Black king on 5b inside the enemy field with a 32-point entering-king
        // score and legal moves available (so resign does not take precedence).
        let p = pos("+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/4k4 b R 1");
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.run_root(&p, 1)
        };
        assert_eq!(out.kind, RootKind::DeclarationWin);
        assert_eq!(out.best_move, Move::win());
        assert_eq!(out.score, mate_in(1));
        assert_eq!(out.nodes, 0);
    }

    // -----------------------------------------------------------------
    // MaxMovesToDraw horizon. The `0 → 100000` remap is
    // the driver's job; here the search field is set directly.
    // -----------------------------------------------------------------

    #[test]
    fn qsearch_forced_draw_past_max_moves_to_draw_is_exact() {
        // A quiet, non-repetition position at game_ply 60. With the horizon set
        // below it, the ply-0 qsearch node adjudicates an unconditional draw
        // before any eval or do_move (`yaneuraou-search.cpp`). The value is
        // exactly `draw_value(REPETITION_DRAW, root_us) + value_draw(0)`:
        //   draw_contempt (-2*90/100 = -1) + (VALUE_DRAW - 1 + (0 & 2) = -1) = -2.
        let net = zero_net();
        let table = fresh_tt();
        let mut p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 60");
        let out = {
            let mut q = QSearch::new(&net, &table);
            q.set_max_moves_to_draw(50); // game_ply 60 > 50 → forced draw at ply 0.
            q.run(&mut p, -VALUE_INFINITE, VALUE_INFINITE, true, true)
        };
        assert_eq!(
            out.value, -2,
            "forced-draw value = draw_contempt + value_draw(0)"
        );
        assert_eq!(out.nodes, 0, "the horizon draw returns before any do_move");
    }

    #[test]
    fn qsearch_default_horizon_does_not_force_draw() {
        // The same position under the default (unlimited) horizon runs a real
        // qsearch: no captures at startpos, zero-eval stand-pat ⇒ value 0. The
        // point is it did NOT take the -2 forced-draw exit — the horizon is what
        // changed the outcome.
        let net = zero_net();
        let table = fresh_tt();
        let mut p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 60");
        let out = {
            let mut q = QSearch::new(&net, &table); // default max_moves_to_draw.
            q.run(&mut p, -VALUE_INFINITE, VALUE_INFINITE, true, true)
        };
        assert_eq!(
            out.value, 0,
            "unlimited horizon ⇒ zero-eval stand-pat, not the draw exit"
        );
    }

    #[test]
    fn run_root_max_moves_to_draw_suppresses_a_mate() {
        // A gold-drop head mate (`G*8a`) at a high game ply — the White king on
        // 9a is not itself in check at the root (so no root move captures it),
        // and dropping the supported gold beside it is mate. Under the default
        // horizon the search finds the mate (a decisive score); with the horizon
        // set below the game ply, every interior node adjudicates a draw before
        // the mate is seen, so the score collapses to the draw band. This
        // exercises the interior-search horizon site (2460) at depth 2 and the
        // qsearch site (4616) at depth 1, both driven by the configured field.
        let net = zero_net();
        let p = pos("k8/9/G1N6/9/9/9/9/9/8K b G 100");

        let unlimited = {
            let table = fresh_tt();
            let mut q = QSearch::new(&net, &table);
            q.run_root(&p, 2)
        };
        assert!(
            is_win(unlimited.score),
            "unlimited horizon must find the mate, got score {}",
            unlimited.score
        );

        let capped = {
            let table = fresh_tt();
            let mut q = QSearch::new(&net, &table);
            q.set_max_moves_to_draw(50); // game_ply 100 > 50 at every interior node.
            q.run_root(&p, 2)
        };
        assert!(
            !is_decisive(capped.score),
            "the horizon must suppress the mate, got score {}",
            capped.score
        );
        assert!(
            capped.score.abs() <= 2,
            "capped score must be in the draw band, got {}",
            capped.score
        );
    }

    // -----------------------------------------------------------------
    // Interior main search (`QSearch::search`). Values here are hand-computed
    // against the reference's formulas, driven by the zero-eval network so every
    // static eval is 0.
    // -----------------------------------------------------------------

    #[test]
    fn reductions_table_and_reduction_formula() {
        let net = zero_net();
        let table = fresh_tt();
        let mut q = QSearch::new(&net, &table);

        // reductions[i] == int(2763/128.0 * ln(i)), reductions[0] == 0.
        assert_eq!(q.reductions[0], 0);
        assert_eq!(q.reductions[1], 0); // ln(1) == 0
        assert_eq!(q.reductions[2], 14);
        assert_eq!(q.reductions[3], 23);
        assert_eq!(q.reductions[4], 29);
        assert_eq!(q.reductions[8], 44);
        assert_eq!(q.reductions[10], 49);

        // reduction(i, d, mn, delta) = rs - delta*585/rootDelta
        //                            + (!i)*rs*206/512 + 1133, rs = red[d]*red[mn].
        q.root_delta = 1000;
        let rs = q.reductions[8] * q.reductions[4];
        assert_eq!(q.reduction(true, 8, 4, 100), rs - 100 * 585 / 1000 + 1133);
        assert_eq!(
            q.reduction(false, 8, 4, 100),
            rs - 100 * 585 / 1000 + rs * 206 / 512 + 1133,
        );
        q.root_delta = 200;
        let rs2 = q.reductions[10] * q.reductions[2];
        assert_eq!(
            q.reduction(false, 10, 2, 50),
            rs2 - 50 * 585 / 200 + rs2 * 206 / 512 + 1133,
        );
    }

    #[test]
    fn move_stat_score_capture_and_quiet() {
        let net = zero_net();
        let table = fresh_tt();
        let q = QSearch::new(&net, &table);
        let s = QSearch::si(2); // sentinels (ss-1)/(ss-2) below it exist.

        // Quiet: 2*mainHistory(0) + contHist[0] + contHist[1]. Both continuation
        // planes are the sentinel plane 0, filled -523.
        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        let quiet = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), bp);
        // 2*mainHistory(0) == 0, plus the two sentinel continuation planes.
        assert_eq!(
            q.move_stat_score(Color::Black, bp, quiet, s, false, None),
            -523 + -523,
        );

        // Capture: 863*PieceValue[pawn]/128 + captureHistory init(-678).
        let wp = Piece::new(PieceKind::Pawn, Color::White);
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let cap = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), rook);
        assert_eq!(
            q.move_stat_score(Color::Black, rook, cap, s, true, Some(wp)),
            863 * 90 / 128 - 678,
        );
    }

    #[test]
    fn razoring_boundary_returns_qsearch_without_do_move() {
        let net = zero_net();
        // eval == 0, depth 1 ⇒ razoring fires when alpha - 502 - 306 > 0, i.e.
        // alpha > 808. It returns qsearch<NonPV>, which for two bare kings makes
        // no do_move.
        let fired_tt = fresh_tt();
        let fired_nodes = {
            let mut q = QSearch::new(&net, &fired_tt);
            q.run_search(&mut pos(TWO_KINGS), 809, 810, 1, false, false);
            q.nodes
        };
        assert_eq!(fired_nodes, 0, "razoring returns qsearch with no do_move");

        // At the boundary (alpha 808 ⇒ 0 < 0 is false) razoring does not fire and
        // the move loop searches the king moves.
        let not_tt = fresh_tt();
        let not_nodes = {
            let mut q = QSearch::new(&net, &not_tt);
            q.run_search(&mut pos(TWO_KINGS), 808, 809, 1, false, false);
            q.nodes
        };
        assert!(not_nodes > 0, "at the boundary razoring must not fire");
    }

    #[test]
    fn futility_boundary_returns_two_beta_plus_eval_over_three() {
        let net = zero_net();
        // depth 1, !ttHit ⇒ futilityMult 55, improving false, opponentWorsening
        // true ⇒ margin = 55 - 362*55/1024 = 36. Futility fires when eval >= beta
        // and eval - 36 >= beta; with eval 0 that is beta <= -36. It returns
        // (2*beta + eval)/3.
        let tt = fresh_tt();
        let (v, n) = {
            let mut q = QSearch::new(&net, &tt);
            let v = q.run_search(&mut pos(TWO_KINGS), -37, -36, 1, false, false);
            (v, q.nodes)
        };
        assert_eq!(v, (2 * -36) / 3); // -24
        assert_eq!(n, 0, "futility returns before any do_move");

        // beta == -35: eval - margin (-36) < beta, so futility does not fire.
        let tt2 = fresh_tt();
        let n2 = {
            let mut q = QSearch::new(&net, &tt2);
            q.run_search(&mut pos(TWO_KINGS), -36, -35, 1, false, false);
            q.nodes
        };
        assert!(n2 > 0, "at the boundary futility must not fire");
    }

    #[test]
    fn null_move_boundary_returns_null_value() {
        let net = zero_net();
        let p = pos(TWO_KINGS);
        // A quiet TT move suppresses the futility return (its `!ttMove` term is
        // false), letting the null-move step be reached. Stored non-cutoff
        // (BOUND_NONE, VALUE_NONE) so Step 4 does not cut and the eval is a fresh
        // evaluate() == 0.
        let quiet = legal_moves(&p)
            .into_iter()
            .find(|&m| !m.is_drop() && p.board().get(m.to_sq()).is_none())
            .expect("a quiet king move exists");

        // depth 1, improving false ⇒ null fires when 0 >= beta + 378 - 16, i.e.
        // beta <= -362. R = 7, depth-R < 0 ⇒ the child is qsearch (no do_move).
        let mut tt = fresh_tt();
        prewrite(
            &mut tt,
            &p,
            VALUE_NONE,
            false,
            Bound::None,
            DEPTH_UNSEARCHED,
            move16_of(quiet),
            VALUE_NONE,
        );
        let (v, n) = {
            let mut q = QSearch::new(&net, &tt);
            let v = q.run_search(&mut p.clone(), -363, -362, 1, true, false);
            (v, q.nodes)
        };
        assert_eq!(v, 0, "null search of two kings returns 0");
        assert_eq!(n, 0, "null move + qsearch make no counted do_move");

        // beta == -361: 0 >= -361 - 16 + 378 == 1 is false ⇒ null does not fire.
        let mut tt2 = fresh_tt();
        prewrite(
            &mut tt2,
            &p,
            VALUE_NONE,
            false,
            Bound::None,
            DEPTH_UNSEARCHED,
            move16_of(quiet),
            VALUE_NONE,
        );
        let n2 = {
            let mut q = QSearch::new(&net, &tt2);
            q.run_search(&mut p.clone(), -362, -361, 1, true, false);
            q.nodes
        };
        assert!(n2 > 0, "at the boundary null move must not fire");
    }

    #[test]
    fn probcut_returns_value_minus_margin() {
        let net = zero_net();
        // Black rook (5e) can capture the undefended white pawn (5d); the white
        // king (1a) is far, so SEE(Rxp) == +90 and there is no recapture. A
        // ttPv-marked non-cutoff TT entry (no move) makes ss->ttPv true so Step 8
        // futility is skipped, letting ProbCut run.
        let p = pos("8k/9/9/4p4/4R4/9/9/9/K8 b - 1");
        assert!(!p.in_check());
        assert_eq!(captures(&p).len(), 1, "exactly one capture");

        let mut tt = fresh_tt();
        prewrite(
            &mut tt,
            &p,
            VALUE_NONE,
            true, // is_pv ⇒ ss->ttPv true
            Bound::None,
            DEPTH_UNSEARCHED,
            0,
            VALUE_NONE,
        );
        // depth 4 ⇒ probCutDepth 0 (no verification search); improving becomes
        // true (0 >= beta), so probCutBeta = beta + 224 - 61 = -61. The capture's
        // qsearch value is 0 >= -61, so ProbCut returns value - (probCutBeta -
        // beta) = 0 - (-61 - (-224)) = -163, having made exactly one do_move.
        let (v, n) = {
            let mut q = QSearch::new(&net, &tt);
            let v = q.run_search(&mut p.clone(), -225, -224, 4, false, false);
            (v, q.nodes)
        };
        assert_eq!(v, -163);
        assert_eq!(n, 1, "one ProbCut capture is searched");
    }

    #[test]
    fn fail_low_writes_the_documented_bonuses() {
        let net = zero_net();
        let tt = fresh_tt();
        let mut q = QSearch::new(&net, &tt);

        // Drive `search` directly at ply 1 with a hand-set (ss-1) cell so the
        // fail-low branch (bestMove none, prevSq real, !priorCapture) fires. White
        // to move; the previous (black) move landed the black king (uppercase `K`)
        // on 5a, so `piece_on(prevSq)` is that black king.
        let mut p = pos("4K4/9/9/9/9/9/9/9/4k4 w - 1");
        q.nodes = 0;
        q.sel_depth = 0;
        q.root_us = Color::White;
        q.draw_contempt = DRAW_VALUE_OPTION_DEFAULT * PAWN_VALUE / 100;
        q.read_tt = true;
        q.root_delta = 2 * VALUE_INFINITE;
        q.root_depth = 1;

        let bk = Piece::new(PieceKind::King, Color::Black);
        let prev_sq = Square::new(4, 0).unwrap(); // 5a, where the black king sits
        let prev = Move::make(Square::new(3, 0).unwrap(), prev_sq, bk);
        let s0 = QSearch::si(0);
        q.stack[s0].current_move = prev;
        q.stack[s0].in_check = true; // suppresses the Step-6 eval-diff main update
        q.stack[s0].stat_score = -20000;
        q.stack[s0].move_count = 0;

        let pawn_key = p.pawn_key();
        assert_eq!(q.histories.main.get(Color::Black, prev), 0);
        let pawn_before = q.histories.shared.pawn_get(pawn_key, bk, prev_sq);
        let corr_before =
            q.histories
                .shared
                .correction_get(pawn_key, Color::White, CorrChannel::Pawn);

        // A small zero-window above 0 makes every white king move (value 0) fail
        // low without tripping razoring (which needs eval < alpha - 808).
        let v = q.search(&mut p, 1, 1, 2, 1, false, false, None, None);
        assert_eq!(v, 0);

        // bonusScale = -232 - (-20000/108) + 59 = 12; scaledBonus = 55*12 = 660.
        // mainHistory[~White=Black][prev] << 660*235/32768 == 4 (from 0).
        assert_eq!(q.histories.main.get(Color::Black, prev), 4);
        // pawn plane [blackKing][5a] << 660*290/8192 == 23 (moves off -1238).
        assert_ne!(
            q.histories.shared.pawn_get(pawn_key, bk, prev_sq),
            pawn_before
        );

        // Correction-history guard: !inCheck, bestMove is none so
        // (bestValue > staticEval) == false == bool(bestMove); the guard fires but
        // with bestValue == staticEval the bonus is 0, leaving the table unchanged.
        assert_eq!(
            q.histories
                .shared
                .correction_get(pawn_key, Color::White, CorrChannel::Pawn),
            corr_before,
        );
    }

    #[test]
    fn interior_smoke_runs_and_leaves_state_balanced() {
        let net = zero_net();
        const SFEN: &str =
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1";
        for depth in [1, 2] {
            let orig = pos(SFEN);
            let mut work = orig.clone();
            let mut tt = fresh_tt();
            let (v, nodes) = {
                let mut q = QSearch::new(&net, &tt);
                let v = q.run_search(
                    &mut work,
                    -VALUE_INFINITE,
                    VALUE_INFINITE,
                    depth,
                    false,
                    true,
                );
                (v, q.nodes)
            };
            assert!(
                -VALUE_INFINITE < v && v < VALUE_INFINITE,
                "depth {depth}: value {v} out of range"
            );
            assert!(nodes > 0, "depth {depth}: interior body must search moves");
            assert_eq!(work, orig, "depth {depth}: the position stack must balance");
            let (found, _data) = probe_root(&mut tt, &orig);
            assert!(found, "depth {depth}: the root node writes a TT entry");
        }
    }

    // -----------------------------------------------------------------
    // Torn-entry totality (Lazy SMP).
    //
    // Under Lazy SMP several workers share the TT through relaxed atomics, so a
    // decoded `TTData` can pair a stale key fragment with a `move16` written for
    // a DIFFERENT position — a torn entry, i.e. an arbitrary `u16`. The single
    // place a stored `move16` is turned back into a `Move` is `widen_tt_move`,
    // and it never widens raw bits into a `Move`: it generates the position's
    // legal moves and returns the one whose `Move16` fragment matches, or `None`.
    // Every consumption point — the qsearch and interior MovePicker TT-move
    // stages, the interior TT-cutoff, and the ponder extraction — funnels
    // through this one gate, so a `Move` is only ever built by the move
    // generator, never decoded from garbage bits (no piece-kind table lookup,
    // enum cast, or square construction is ever indexed by a raw fragment). This
    // test proves the totality end to end: all 65536 patterns against a diverse
    // position set, no panic (debug or release), and every accepted move ⊆ the
    // position's legal set — the port's gate contract, which is stricter than
    // the pin's `pseudo_legal` (accepted ⊆ legal ⊆ pseudo_legal).
    // -----------------------------------------------------------------

    /// The 6 parity-fixture SFENs, which between them already cover the two
    /// structural cases the audit singled out (an in-check position and several
    /// hand-heavy ones); the test asserts that coverage explicitly below.
    const TORN_ENTRY_SFENS: &[&str] = &[
        // startpos
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        // drop-heavy — pieces in hand for both sides
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        // mid-game-tactical — pieces in hand, dense board
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        // check-evasion — side to move is IN CHECK, several pieces in hand
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        // promotion-zone-edges — promoted pieces, near the back ranks
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        // sennichite base — 18 pawns in hand
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    fn total_hand_count(p: &Position) -> u32 {
        let h = p.hand(p.side_to_move());
        [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Gold,
            PieceKind::Bishop,
            PieceKind::Rook,
        ]
        .iter()
        .map(|&k| h.count(k) as u32)
        .sum()
    }

    #[test]
    fn torn_tt_move_decode_is_total_over_all_patterns() {
        let positions: Vec<Position> = TORN_ENTRY_SFENS.iter().map(|s| pos(s)).collect();

        // The set must genuinely exercise the two cases the audit called out.
        assert!(
            positions.iter().any(|p| p.in_check()),
            "position set must include an in-check case"
        );
        assert!(
            positions.iter().any(|p| total_hand_count(p) >= 3),
            "position set must include a hand-heavy case"
        );

        let hist = WorkerHistories::new();

        for (p, sfen) in positions.iter().zip(TORN_ENTRY_SFENS) {
            let mut legal = Vec::new();
            p.generate_legal_all(&mut legal);
            let legal_set: std::collections::HashSet<Move> = legal.iter().copied().collect();

            // (Decode gate — shared by all three consumption points.) Drive the
            // fragment-consuming step (`select_tt_move`, the exact code
            // `widen_tt_move` runs after generation) over every one of the 65536
            // patterns against this one generated list. Assert no panic (the
            // loop completes), and that any accepted move is legal in `p` and
            // round-trips back to its own fragment.
            let mut accepted: std::collections::HashSet<Move> = std::collections::HashSet::new();
            for bits in 0u32..=0xFFFF {
                let m16 = bits as u16;
                if let Some(m) = QSearch::select_tt_move(&legal, m16) {
                    assert!(
                        legal_set.contains(&m),
                        "{sfen}: select_tt_move accepted {m:?} for move16={m16:#06x}, not legal"
                    );
                    assert_eq!(
                        move16_of(m),
                        m16,
                        "{sfen}: select_tt_move({m16:#06x}) returned a move with a different fragment"
                    );
                    accepted.insert(m);
                }
            }

            // Tie the exhaustive `select_tt_move` sweep back to the real
            // `widen_tt_move` (generate + select), which every consumption point
            // actually calls: they agree on `move16 == 0` (early `None` vs. no
            // legal move carrying fragment 0), on every accepted fragment, and
            // on a strided sample of rejecting ones. Generation is
            // pattern-independent, so this transfers the sweep's totality to the
            // real decode without 65536 re-generations.
            let mut sample: Vec<u16> = vec![0];
            sample.extend(accepted.iter().map(|&m| move16_of(m)));
            sample.extend((0u32..=0xFFFF).step_by(97).map(|b| b as u16));
            for m16 in sample {
                assert_eq!(
                    QSearch::widen_tt_move(p, m16),
                    QSearch::select_tt_move(&legal, m16),
                    "{sfen}: widen_tt_move and select_tt_move disagree on move16={m16:#06x}"
                );
            }

            // (Consumption gate.) The MovePicker TT-move stage takes the decode's
            // output; its output is a pure function of `(pos, widened_move)`, and
            // `widened_move` ranges over exactly `{None}` ∪ `accepted`, so driving
            // both pickers over that finite input set covers every one of the
            // 65536 patterns. Drain each fully: no panic, every yield legal.
            let mut inputs: Vec<Option<Move>> = vec![None];
            inputs.extend(accepted.iter().map(|&m| Some(m)));
            for tt_move in inputs {
                for mut mp in [
                    MovePicker::new_qsearch(p, tt_move, [0; 6], false),
                    MovePicker::new_main_search(p, tt_move, 6, 0, [0; 6], false),
                ] {
                    while let Some(m) = mp.next_move(p, &hist) {
                        assert!(
                            legal_set.contains(&m),
                            "{sfen}: MovePicker yielded illegal {m:?} for tt_move={tt_move:?}"
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // O(1) `to_move` + `pseudo_legal` widen chain.
    //
    // The generate-and-match `widen_tt_move` was replaced in production by the
    // reference chain `Position::to_move` (O(1) fragment widen) →
    // `Position::pseudo_legal` → `Position::is_legal`. These tests pin the new
    // chain against the old widen (kept as the `#[cfg(test)]` oracle):
    //   * Gate 1 (totality): all 65536 `u16` fragments widen-validate without
    //     panic under both `all` modes, and every accepted move is pseudo-legal
    //     ∧ legal with a round-tripping fragment.
    //   * Gate 2 (oracle): every legal move round-trips through `to_move` and is
    //     `pseudo_legal` under both modes; every fragment the widen oracle
    //     accepted is accepted by the new chain with the SAME move. Both the
    //     oracle (`generate_legal_all`) and the production chain
    //     are repetition-blind, so the strict search-legal set is a subset of
    //     the all-legal set with NO perpetual-check exception.
    // -----------------------------------------------------------------

    /// The strict search-generated legal set — the moves `pseudo_legal(_, false)`
    /// admits, i.e. the moves the search generators actually produce (and thus
    /// the only moves that can ever be stored as a TT move). Evasions when in
    /// check, else captures ∪ quiets, filtered by [`Position::is_legal`]. Unlike
    /// [`Position::generate_legal_all`] (the lenient all-legal set) it prunes
    /// the "useless" non-promotions and drops onto cannot-move squares.
    fn strict_search_legal(p: &Position) -> Vec<Move> {
        let mut pseudo: Vec<attic_state::ExtMove> = Vec::new();
        if p.in_check() {
            p.generate_evasions(false, &mut pseudo);
        } else {
            p.generate_captures(false, &mut pseudo);
            p.generate_quiets(false, &mut pseudo);
        }
        pseudo
            .into_iter()
            .map(|e| e.mv)
            .filter(|&m| p.is_legal(m))
            .collect()
    }

    /// Gate 2 production oracle at one position: every strict search-legal move
    /// (the only moves that reach the TT) round-trips through `to_move`, is
    /// `pseudo_legal(all=false)`, and the production chain accepts exactly it.
    fn legal_move_chain_oracle(p: &Position, ctx: &str) {
        for m in strict_search_legal(p) {
            let f = move16_of(m);
            assert_eq!(
                p.to_move(f),
                Some(m),
                "{ctx}: to_move({f:#06x}) != its own move"
            );
            assert!(
                p.pseudo_legal(m, false),
                "{ctx}: strict-legal {m:?} is not pseudo_legal(all=false)"
            );
            assert!(
                p.pseudo_legal(m, true),
                "{ctx}: strict-legal {m:?} is not pseudo_legal(all=true)"
            );
            assert!(p.is_legal(m), "{ctx}: strict-legal {m:?} is not legal");
            // The production chain (all=false) accepts exactly `m` for `f`.
            let accepted = p
                .to_move(f)
                .filter(|&mm| mm.is_ok() && p.pseudo_legal(mm, false) && p.is_legal(mm));
            assert_eq!(
                accepted,
                Some(m),
                "{ctx}: chain does not accept {m:?} for {f:#06x}"
            );
        }
    }

    /// Gate 1 (totality) + gate 2 (widen-oracle equivalence) at one position.
    fn widen_chain_full_gates(p: &Position, ctx: &str) {
        legal_move_chain_oracle(p, ctx);

        // The all-legal set is exactly what the generate-and-match widen oracle
        // accepted, and the strict set is what the production chain admits.
        let mut perft_legal = Vec::new();
        p.generate_legal_all(&mut perft_legal);
        let perft_set: std::collections::HashSet<Move> = perft_legal.iter().copied().collect();
        let strict: std::collections::HashSet<Move> = strict_search_legal(p).into_iter().collect();

        // Drive every fragment through the chain, both `all` modes. The
        // loop completing proves totality (no panic); the `if` guard makes the
        // acceptance predicate pseudo-legal ∧ legal by construction. (A torn
        // drop fragment carrying a stray promote bit widens to the same clean
        // drop, so `move16` is not asserted to round-trip over the full sweep —
        // only over real moves, in `legal_move_chain_oracle`.) The count sanity
        // confirms the sweep actually reaches the real moves rather than
        // short-circuiting: every strict-legal move must be among the accepted.
        for all in [false, true] {
            let mut accepted = 0usize;
            for bits in 0u32..=0xFFFF {
                let m16 = bits as u16;
                if let Some(m) = p.to_move(m16)
                    && m.is_ok()
                    && p.pseudo_legal(m, all)
                    && p.is_legal(m)
                {
                    accepted += 1;
                }
            }
            assert!(
                accepted >= strict.len(),
                "{ctx}: {accepted} acceptances (all={all}) < {} strict-legal moves",
                strict.len()
            );
        }

        // Gate 2 equivalence: every fragment the OLD widen accepted (a perft-legal
        // move — lenient promotion rules, so compared under `all == true`) is
        // accepted by the new chain with the SAME move. `select_tt_move` over the
        // generated list is the old widen without 65536 re-generations (tied to
        // the real `widen_tt_move` by the sibling test).
        for bits in 0u32..=0xFFFF {
            let m16 = bits as u16;
            if let Some(old) = QSearch::select_tt_move(&perft_legal, m16) {
                let new = p
                    .to_move(m16)
                    .filter(|&m| m.is_ok() && p.pseudo_legal(m, true) && p.is_legal(m));
                assert_eq!(
                    new,
                    Some(old),
                    "{ctx}: old widen accepted {old:?} for {m16:#06x}, new chain gives {new:?}"
                );
            }
        }

        // `strict ⊆ perft` with NO exception: the all-legal set
        // (`generate_legal_all`) and the strict search set are both
        // repetition-blind, so the only reason `strict` is
        // smaller is its promotion / cannot-move pruning — every strict-legal
        // move is an all-legal move.
        let missing: Vec<Move> = strict.difference(&perft_set).copied().collect();
        assert!(
            missing.is_empty(),
            "{ctx}: strict search-legal moves absent from the all-legal set: {missing:?}"
        );
    }

    #[test]
    fn to_move_widen_chain_totality_and_oracle() {
        for sfen in TORN_ENTRY_SFENS {
            let p = pos(sfen);
            widen_chain_full_gates(&p, sfen);
        }
    }

    #[test]
    fn to_move_widen_chain_oracle_over_playouts() {
        // Gate 2 requires the legal-move oracle to hold along a deterministic
        // playout of >= 30 plies from each fixture. The move choice rotates by
        // ply so the line advances rather than shuffling in place.
        for sfen in TORN_ENTRY_SFENS {
            let mut p = pos(sfen);
            for ply in 0..30usize {
                legal_move_chain_oracle(&p, &format!("{sfen} @ ply {ply}"));
                let mut legal = Vec::new();
                p.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    break; // terminal position (mate / stalemate)
                }
                let pick = legal[ply % legal.len()];
                let _ = p.do_move(pick);
            }
        }
    }
}
