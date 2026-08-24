//! Port of the reference `MovePicker` (`source/movepick.cpp`
//! at the current submodule pin), covering both the quiescence-search stages and
//! the main-search stages.
//!
//! # Stage sequences
//!
//! Not in check (`movepick.cpp`):
//!
//! * main search (`depth > 0`):
//!   `MAIN_TT → CAPTURE_INIT → GOOD_CAPTURE → QUIET_INIT → GOOD_QUIET →
//!    BAD_CAPTURE → BAD_QUIET`
//! * qsearch (`depth == 0`): `QSEARCH_TT → QCAPTURE_INIT → QCAPTURE`
//!
//! In check (both searches): `EVASION_TT → EVASION_INIT → EVASION`.
//!
//! ProbCut: `PROBCUT_TT → PROBCUT_INIT → PROBCUT`.
//!
//! # Single-buffer stage machine
//!
//! Like the reference, this picker runs the whole stage machine on **one** fixed
//! [`ExtMove`] buffer (`buf`, capacity [`MAX_MOVES`]) with index boundaries that
//! mirror the pin's pointers (`movepick.h`):
//!
//! * `cur`             — the next move to return (`select`'s cursor).
//! * `end_cur`         — the end of the segment `select` currently walks.
//! * `end_bad_captures`— the end of the SEE-losing capture region compacted into
//!   the buffer's **front** during `GOOD_CAPTURE`.
//! * `end_captures`    — the end of the captures region (`GOOD_QUIET` walks the
//!   quiets that follow it).
//! * `end_generated`   — the end of all generated moves.
//!
//! `GOOD_CAPTURE` walks the sorted captures and, for each SEE-loser, swaps it to
//! the front region `[0, end_bad_captures)` (`std::swap(*endBadCaptures++, *cur)`,
//! `movepick.cpp`), so the good ones are yielded and the bad ones accumulate
//! for `BAD_CAPTURE` to replay. The quiets are generated into the buffer *after*
//! the captures (`[end_captures, end_generated)`); `GOOD_QUIET` yields those
//! scoring `> -14000`, `BAD_QUIET` replays the rest. For qsearch / evasion /
//! ProbCut there is no good/bad split — a single `select` loop over the whole
//! sorted list reproduces `QCAPTURE` / `EVASION` / `PROBCUT`
//! (`movepick.cpp`).
//!
//! ## Generation writes `buf` directly
//!
//! Like the pin's generators (`ExtMove* generateMoves(const Position&,
//! ExtMove*)`, `movegen.h`), this port's search generators
//! ([`Position::generate_captures`] etc.) write [`ExtMove`] straight into
//! `buf`'s tail with `value: 0`, and scoring then fills each `value` **in
//! place** at the `*_INIT` stage entry. There is no intermediate
//! `Move`-typed staging vector and no per-move restaging copy. `buf` is drawn
//! from the per-thread scratch pool, so steady-state search performs **zero**
//! per-node heap allocation.
//!
//! # Staged-lazy scoring
//!
//! The reference is a lazy state machine: it scores a stage's moves *when the
//! stage is entered*, which — crucially — is **after** the earlier stages'
//! moves have been emitted and their subtrees searched (`movepick.cpp`).
//! At `depth >= 2` those subtrees run `update_all_stats` / TT-cutoff /
//! eval-diff updates that write the very history cells a later stage's scoring
//! reads (grandchildren move the same colour, touching `mainHistory[us]` and the
//! `(ss-1..4)` continuation planes at `[us piece][to]`), so a construction-time
//! snapshot of the scores orders moves differently from the reference. This port
//! therefore mirrors the reference exactly: [`MovePicker::next_move`] takes the
//! live [`WorkerHistories`] and scores each stage at stage-entry time, reading
//! whatever those tables hold at that moment:
//!
//! * `CAPTURE_INIT` scores the captures on the first `next_move` after the TT
//!   move's subtree was searched (`movepick.cpp`).
//! * `QUIET_INIT` scores the quiets after the good captures' subtrees were
//!   searched (`movepick.cpp`).
//!
//! The continuation planes are held as flat plane **indices** into the live
//! [`ContinuationHistory`] (`cont_planes`, the reference's `contHist` array of
//! `(ss-1-i)->continuationHistory` pointers, `yaneuraou-search.cpp`),
//! never as snapshots, so an update to those planes between stages is seen by
//! the later stage's scoring exactly as in the reference.
//!
//! *Move generation* itself (not scoring) is a pure function of the position and
//! carries no history dependence, so *when* a list is materialized never changes
//! its contents or order. Following the reference, every list is generated at
//! its `*_INIT` stage entry, not at construction: `CAPTURE_INIT` /
//! `QCAPTURE_INIT` / `PROBCUT_INIT` generate the captures and `EVASION_INIT`
//! generates the evasions at stage entry (`movepick.cpp`), and
//! `QUIET_INIT` generates the quiets. A picker abandoned
//! at the `*_TT` stage by a TT-move beta cutoff therefore never pays for any
//! generation — the reference's whole point in deferring it. The `skipQuiets`
//! flag is honoured at `QUIET_INIT` (skip the quiet generation and scoring) and
//! at `GOOD_QUIET` / `BAD_QUIET` (skip emitting), exactly as the reference
//! checks it (`movepick.cpp`).
//!
//! # Lazy SEE at yield
//!
//! SEE is evaluated **lazily, at yield time**, exactly as the pin does: the
//! `GOOD_CAPTURE` stage calls `see_ge(m, -value / 18)` on each capture as
//! `select` reaches it (`movepick.cpp`), and `PROBCUT` calls
//! `see_ge(m, threshold)` per move (`movepick.cpp`). On an early beta cutoff
//! the remaining captures' SEE is never computed — work that SEE-scoring every
//! capture at `CAPTURE_INIT` would pay regardless of the cutoff. QSearch /
//! evasion lists carry no SEE split and are simply full-sorted.
//!
//! # Scoring (`movepick.cpp`)
//!
//! * Captures: `captureHistory[pc][to][type_of(captured)] + 7 * PieceValue[captured]`.
//!   `GOOD_CAPTURE` keeps a capture iff `see_ge(m, -value / 18)` where `value` is
//!   that move's own score (`movepick.cpp`).
//! * Quiets: `2*mainHistory + 2*pawnHistory + contHist[0] + contHist[1] +
//!   contHist[2] + contHist[3] + contHist[5] + (direct-check ? 16384 : 0) +
//!   (ply < LOW_PLY_HISTORY_SIZE ? 8*lowPlyHistory/(1+ply) : 0)`. `QUIET_INIT`
//!   sorts with `partial_insertion_sort` limit `-3560 * depth`; `GOOD_QUIET`
//!   keeps `value > -14000`.
//! * Evasions: capturing evasion `PieceValue[victim] + (1 << 28)` (outranks every
//!   quiet), quiet evasion `mainHistory + continuationHistory[0]`.
//!
//! # TT move
//!
//! The `*_TT` stages yield the TT move first, and every later stage skips it
//! (`select`'s `*cur != ttMove` guard, `movepick.cpp`). The reference gates
//! the TT move on `pseudo_legal` (`movepick.cpp`) and lets the search
//! loop drop an illegal one with `pos.legal(move)`. This port applies **both**
//! at construction — [`Position::pseudo_legal`] then [`Position::is_legal`]
//! — because this engine's search loops rely on the picker's
//! "yielded moves are legal" contract rather than re-testing legality per move.
//! The accepted set is therefore exactly the reference's (pseudo-legal ∧ legal),
//! with one deliberate exception: a TT move that *continues* a perpetual check
//! is accepted here, as in the pin — repetition is handled by scoring
//! (`is_repetition`), never by filtering the move out of the picker.
//!
//! # Legality filtering: at generation vs at yield
//!
//! The reference generators are pseudo-legal; the search loop drops illegal
//! moves with `if (!pos.legal(move)) continue`. This port keeps the "yielded
//! moves are legal" contract but must place the legality filter where it does
//! not disturb move *ordering* (unchanged by that placement):
//!
//! * **Captures / evasions / ProbCut** are **full-sorted** (limit `i32::MIN`),
//!   so removing illegal moves at generation is order-neutral (every element is
//!   promoted, so `partial_insertion_sort`'s promotion swap `*p = *++sortedEnd`
//!   is a self-assignment — see [`MovePicker::next_move`]'s `CAPTURE_INIT` arm).
//! * **Quiets** are **partial-sorted** with the depth-scaled limit
//!   `-3560 * depth`, whose promotion swaps relocate tail elements. The set of
//!   elements in the buffer therefore changes the emitted order, so the raw
//!   pseudo-legal buffer (illegal quiets and the TT move included) is scored and
//!   sorted, and legality is filtered only at yield in `GOOD_QUIET` / `BAD_QUIET`
//!   — exactly mirroring the reference's `MoveList<QUIETS>` + search-loop gate.

use std::cell::RefCell;

use attic_state::{CheckSquares, ExtMove, Move, Position, piece_value};

use crate::history::LOW_PLY_HISTORY_SIZE;
use crate::update::WorkerHistories;

/// `goodQuietThreshold` (`movepick.cpp`): quiets scoring above this go to
/// `GOOD_QUIET`, the rest to `BAD_QUIET`.
const GOOD_QUIET_THRESHOLD: i32 = -14000;

/// The reference `MovePicker`'s fixed buffer capacity (`movepick.h`). The
/// maximum number of legal moves in a shogi position is 593; 600 leaves a small
/// margin. `debug_assert`ed on generation overflow.
const MAX_MOVES: usize = 600;

/// The reusable per-node move buffers a [`MovePicker`] draws from — the port's
/// analogue of the reference `MovePicker`'s fixed `ExtMove moves[MAX_MOVES]`
/// stack buffer (`movepick.h`). A picker is constructed at every search
/// node, so allocating these `Vec`s afresh each time faults a fresh heap page
/// per node. Instead every picker borrows a `PickerScratch` from a
/// per-thread pool ([`SCRATCH_POOL`]) at construction and returns it on
/// [`Drop`]; the `Vec`s are cleared (never shrunk) between uses, so their
/// capacity persists and steady-state search performs **zero** per-node heap
/// allocation here. The pool grows to at most the live recursion depth (a
/// parent picker holds its scratch while its children search with their own),
/// bounded by `MAX_PLY`.
#[derive(Default)]
struct PickerScratch {
    /// The single stage-machine buffer: captures (or evasions) first, then — for
    /// the main not-in-check picker — the raw quiets. Scoring fills each `value`
    /// in place at stage entry; the sort, bad-capture compaction, and segment
    /// replays all walk index boundaries into this one buffer. The search-side
    /// generators ([`Position::generate_captures`] etc.) emit [`ExtMove`]
    /// straight into this buffer's tail, so there is no separate
    /// `Move`-typed staging vector and no per-move restaging copy.
    buf: Vec<ExtMove>,
}

impl PickerScratch {
    /// Empty the buffer without releasing capacity — the reuse contract that
    /// keeps steady-state allocation at zero.
    fn clear(&mut self) {
        self.buf.clear();
    }
}

thread_local! {
    /// Per-worker (per-thread) free-list of [`PickerScratch`] buffers. Each Lazy
    /// SMP worker runs on its own thread, so a thread-local pool is exactly a
    /// per-worker pool. Popped at [`MovePicker`] construction, pushed back on
    /// [`Drop`]; capacity accumulates to the deepest concurrent recursion.
    static SCRATCH_POOL: RefCell<Vec<PickerScratch>> = const { RefCell::new(Vec::new()) };
}

/// Borrow a cleared scratch from the thread-local pool (or a fresh one if the
/// pool is empty — a one-time cost as the pool warms to recursion depth). Both
/// buffers are reserved to [`MAX_MOVES`] so no reallocation occurs mid-search.
fn take_scratch() -> PickerScratch {
    let mut scratch = SCRATCH_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    scratch.clear();
    scratch.buf.reserve(MAX_MOVES);
    scratch
}

/// The reference `partial_insertion_sort` (`movepick.cpp`): a stable
/// descending insertion sort over the elements whose `value >= limit` (those
/// below `limit` are left toward the tail in unspecified order). Equal-scored
/// elements keep their input order, so with a full sort (`limit == i32::MIN`)
/// the tie-break is the generation order of the input.
fn partial_insertion_sort(a: &mut [ExtMove], limit: i32) {
    if a.is_empty() {
        return;
    }
    let mut sorted_end = 0usize;
    for p in 1..a.len() {
        if a[p].value >= limit {
            let tmp = a[p];
            sorted_end += 1;
            a[p] = a[sorted_end];
            let mut q = sorted_end;
            while q != 0 && a[q - 1].value < tmp.value {
                a[q] = a[q - 1];
                q -= 1;
            }
            a[q] = tmp;
        }
    }
}

/// `score<CAPTURES>` for a single capture (`movepick.cpp`).
///
/// A `CAPTURES` move always lands on an enemy piece, so the victim is present.
/// The moving-piece index is `pos.moved_piece(m)`, which in YaneuraOu aliases
/// `moved_piece_after()` — the **after-promotion** piece (position.h,
/// "moved_piece_after()にしたほうが強い"). Using the pre-move piece would index
/// `captureHistory` wrongly for promoting captures. Reads the live
/// `captureHistory` at call (stage-entry) time.
fn score_capture(pos: &Position, m: Move, hist: &WorkerHistories) -> i32 {
    let to = m.to_sq();
    let moved = m.moved_piece_after();
    // A capture's destination always holds the victim, so this is never `None`;
    // returning `0` on the impossible miss keeps the hot scorer panic-free.
    let Some(victim) = pos.board().get(to) else {
        return 0;
    };
    hist.capture.get(moved, to, victim) + 7 * piece_value(victim)
}

/// `score<EVASIONS>` for a single evasion (`movepick.cpp`).
///
/// `capture_stage(m)` at the pin is plain `capture(m)`: a non-drop landing on an
/// occupied square. Capturing evasions get `PieceValue[victim] + (1 << 28)` so
/// they outrank every quiet; quiet evasions get `mainHistory[us][m] +
/// (*continuationHistory[0])[pc][to]`, both read live at call time.
/// `cont_plane0` is the flat index of `(ss-1)->continuationHistory` into `hist`.
fn score_evasion(pos: &Position, m: Move, hist: &WorkerHistories, cont_plane0: usize) -> i32 {
    let to = m.to_sq();
    let victim = if m.is_drop() {
        None
    } else {
        pos.board().get(to)
    };
    if let Some(v) = victim {
        piece_value(v) + (1 << 28)
    } else {
        let us = pos.side_to_move();
        // `continuationHistory` is indexed by `pos.moved_piece(m)` ==
        // `moved_piece_after()` (the after-promotion piece), which is well
        // defined for drops too (the dropped piece with the side-to-move's
        // colour) — the same reasoning applies in `score_capture`.
        let pc = m.moved_piece_after();
        hist.main.get(us, m) + hist.continuation.get_at(cont_plane0, pc, to)
    }
}

/// `score<QUIETS>` for a single quiet move (`movepick.cpp`).
///
/// `pc = pos.moved_piece(m)` is the after-promotion piece (a promoting quiet
/// indexes its promoted form); for a drop it is the dropped piece of the side to
/// move. The quiet score reads the five continuation planes
/// `continuationHistory[0][1][2][3][5]` (index `4` deliberately absent), passed
/// as flat plane indices into the live `hist.continuation`. The Stockfish
/// "threat by a lesser piece" term (`movepick.cpp`) is `#if STOCKFISH`-
/// only and absent at this pin, so it is omitted.
fn score_quiet(
    pos: &Position,
    m: Move,
    ply: i32,
    hist: &WorkerHistories,
    cont_planes: [usize; 6],
    check_squares: &CheckSquares,
) -> i32 {
    let us = pos.side_to_move();
    let to = m.to_sq();
    let pc = m.moved_piece_after();

    let mut value = 2 * hist.main.get(us, m);
    // `pawn_entry(pos)` selects the plane by `pos.pawn_key()` (`history.h`),
    // then indexes `[pc][to]` (`movepick.cpp`).
    value += 2 * hist.shared.pawn_get(pos.pawn_key(), pc, to);
    value += hist.continuation.get_at(cont_planes[0], pc, to);
    value += hist.continuation.get_at(cont_planes[1], pc, to);
    value += hist.continuation.get_at(cont_planes[2], pc, to);
    value += hist.continuation.get_at(cont_planes[3], pc, to);
    value += hist.continuation.get_at(cont_planes[5], pc, to);

    // Bonus for a direct check that is not a losing sacrifice
    // (`movepick.cpp`). The direct-check term reads the `checkSquares`
    // snapshot taken once at `QUIET_INIT` entry — the exact reference form
    // `(pos.check_squares(pt) & to) && pos.see_ge(m, -75)` — instead of
    // re-entering the lazy `check_info()` accessor per scored quiet.
    if check_squares.gives_direct_check(m) && pos.see_ge(m, -75) {
        value += 16384;
    }

    // lowPlyHistory near the root (`movepick.cpp`). Integer division
    // truncates toward zero exactly as the C++ `/` does.
    if (ply as usize) < LOW_PLY_HISTORY_SIZE {
        value += 8 * hist.low_ply.get(ply as usize, m) / (1 + ply);
    }

    value
}

/// Which family of moves this picker emits, and how its capture-like init stage
/// scores and partitions them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Main search, not in check: captures split by SEE (`GOOD`/`BAD_CAPTURE`),
    /// then quiets split by the `-14000` threshold.
    Main,
    /// Quiescence, not in check: captures only, full sort, no split.
    QSearch,
    /// In check (main or qsearch): evasions, full sort, no split.
    Evasion,
    /// ProbCut: captures with `see_ge(threshold)`, full sort, no split.
    ProbCut,
}

/// The current emission stage, one arm per reference `Stages` constant
/// (`movepick.cpp`). `*_INIT` are the lazy score-and-sort stages; the rest
/// drain a buffer segment via `select`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// The shared `*_TT` phase: yield the TT move (if any), then dispatch to the
    /// kind's init stage.
    Tt,
    CaptureInit,
    GoodCapture,
    QuietInit,
    GoodQuiet,
    BadCapture,
    BadQuiet,
    QcaptureInit,
    Qcapture,
    EvasionInit,
    Evasion,
    ProbcutInit,
    Probcut,
    Done,
}

/// Yields the moves of a search node in the reference `MovePicker` order.
///
/// Construct with [`MovePicker::new_qsearch`] (quiescence),
/// [`MovePicker::new_main_search`] (main search) or [`MovePicker::new_probcut`]
/// (ProbCut), then pull moves with [`MovePicker::next_move`], **passing the live
/// [`WorkerHistories`] each call**, until it returns `None`. Scoring happens at
/// each stage's entry against those live tables, so history updates performed by
/// earlier moves' subtrees (between `next_move` calls) are honoured — this is
/// the staged-lazy fidelity the eager design could not provide. All
/// yielded moves are legal (the reference search skips illegal moves without
/// counting a node; filtering them here is equivalent).
pub struct MovePicker {
    kind: Kind,
    tt: Option<Move>,
    /// Search depth (`> 0` for the main picker) — the `QUIET_INIT` partial-sort
    /// limit is `-3560 * depth`. `0` for qsearch / ProbCut.
    depth: i32,
    /// `ss->ply`, read by `score<QUIETS>`'s `lowPlyHistory` term. Only the main
    /// picker scores quiets, so it is irrelevant for the other kinds.
    ply: i32,
    /// ProbCut SEE threshold (unused by the other kinds).
    threshold: i32,
    /// Flat plane indices of `(ss-1-i)->continuationHistory` into
    /// [`WorkerHistories::continuation`] (`contHist`, `yaneuraou-search.cpp`).
    cont_planes: [usize; 6],

    /// The reusable per-node move buffers, borrowed from the thread-local pool
    /// at construction and returned on [`Drop`] (see [`PickerScratch`]). `buf` is
    /// filled with the generated moves at construction; scoring fills each
    /// `value` in place at the `*_INIT` stages.
    scratch: PickerScratch,

    skip_quiets: bool,
    stage: Stage,

    /// The `GenerateAllLegalMoves` flag, stashed so the deferred quiet
    /// generation at `QUIET_INIT` can reproduce the construction-time call
    /// `generate_quiets(all, …)`. Only the `Main` kind reaches `QUIET_INIT`, so
    /// it is unread for the other kinds.
    all: bool,

    // The pin's pointers, as indices into `scratch.buf` (`movepick.h`).
    /// The next move to return.
    cur: usize,
    /// The end of the segment `select` currently walks.
    end_cur: usize,
    /// The end of the SEE-losing capture region compacted into the buffer front.
    end_bad_captures: usize,
    /// The end of the captures region (quiets follow it).
    end_captures: usize,
    /// The end of all generated moves.
    end_generated: usize,
}

impl Drop for MovePicker {
    /// Return the scratch buffers (capacity intact) to the thread-local pool for
    /// the next node's picker to reuse. Running on `Drop` makes reclamation
    /// robust to every early return in the search loops.
    fn drop(&mut self) {
        let scratch = std::mem::take(&mut self.scratch);
        SCRATCH_POOL.with(|pool| pool.borrow_mut().push(scratch));
    }
}

impl MovePicker {
    /// Build a qsearch picker for `pos` with an optional transposition-table
    /// move `tt_move` and the current node's continuation planes `cont_planes`
    /// (`(ss-1-i)->continuationHistory`; qsearch scores only read plane `[0]`).
    ///
    /// When `pos` is in check the evasion stages run; otherwise the capture
    /// stages run. A legal `tt_move` is yielded first and de-duplicated from the
    /// generated stage. There is no good/bad split in qsearch — the whole sorted
    /// list is emitted best-first.
    pub fn new_qsearch(
        pos: &Position,
        tt_move: Option<Move>,
        cont_planes: [usize; 6],
        all: bool,
    ) -> Self {
        let in_check = pos.in_check();
        let tt = tt_move.filter(|&m| m.is_ok() && pos.pseudo_legal(m, all) && pos.is_legal(m));
        // The capture / evasion list is *not* materialized here. The reference
        // generates it only at the `QCAPTURE_INIT` / `EVASION_INIT` stage entry
        // (`movepick.cpp`); a node that cuts off at the TT stage
        // never pays for it. Generation lives at those `next_move` arms, so
        // the buffer starts empty (`end_captures == end_generated == 0`).
        let scratch = take_scratch();
        Self::from_parts(
            if in_check {
                Kind::Evasion
            } else {
                Kind::QSearch
            },
            tt,
            0,
            0,
            0,
            cont_planes,
            scratch,
            all,
        )
    }

    /// Build a main-search picker for `pos` at `depth` (`> 0`) and `ply` with an
    /// optional transposition-table move `tt_move` and the node's continuation
    /// planes `cont_planes`.
    ///
    /// In check, the evasion stages run (identical to qsearch, but scored with
    /// the main-search histories). Otherwise the capture / quiet stages run with
    /// the SEE-based good/bad capture split and the `-14000` good/bad quiet
    /// split. The quiet score reads planes `[0][1][2][3][5]`
    /// (`movepick.cpp`).
    pub fn new_main_search(
        pos: &Position,
        tt_move: Option<Move>,
        depth: i32,
        ply: i32,
        cont_planes: [usize; 6],
        all: bool,
    ) -> Self {
        let in_check = pos.in_check();
        let tt = tt_move.filter(|&m| m.is_ok() && pos.pseudo_legal(m, all) && pos.is_legal(m));
        // Neither list is materialized here. The reference generates the captures
        // (or, in check, the evasions) only at `CAPTURE_INIT` / `EVASION_INIT`
        // stage entry, and the quiets only at `QUIET_INIT`
        // (`movepick.cpp`, the latter two inside
        // `if (!skipQuiets)`): a node that
        // cuts off during the TT / good-capture stages — or that had `skipQuiets`
        // set by late-move pruning before `QUIET_INIT` — never pays for the
        // generation it does not reach. Generating at construction would be
        // cutoff-independent prepaid work; the captures/evasions instead
        // generate at the corresponding `*_INIT` arm of `next_move` and the
        // quiets at `Stage::QuietInit`. The buffer therefore starts
        // empty (`end_captures == end_generated == 0`); `CAPTURE_INIT` fills the
        // captures into `[0, end_captures)`, then `QUIET_INIT` appends the quiets
        // after them and extends `end_generated`.
        let scratch = take_scratch();
        let kind = if in_check { Kind::Evasion } else { Kind::Main };
        Self::from_parts(kind, tt, depth, ply, 0, cont_planes, scratch, all)
    }

    /// Build a ProbCut picker for `pos` with an optional transposition-table
    /// move `tt_move` and SEE `threshold` (`movepick.cpp`).
    ///
    /// Generates **captures only**, scored by `score<CAPTURES>` and fully sorted,
    /// then yields those with `see_ge(m, threshold)` (SEE evaluated lazily at
    /// yield). The TT move leads iff it is a legal capture (it is exempt from the
    /// SEE filter, matching the reference `PROBCUT_TT` stage, which gates only on
    /// `pos.capture(ttm)` & pseudo-legality). ProbCut is only entered when not in
    /// check, so there is no evasion path.
    ///
    /// `all` is the `GenerateAllLegalMoves` flag; the reference `PROBCUT_INIT`
    /// generates `CAPTURES_ALL` when it is on, `CAPTURES` otherwise
    /// (`movepick.cpp`, shared with `CAPTURE_INIT`), so this picker passes
    /// it through. The pin's asymmetry note (`yaneuraou-search.cpp`) —
    /// that ProbCut should never return a pawn *non-promotion the generator would
    /// not produce* — is satisfied by construction: `generate_captures` targets
    /// enemy squares only, so no quiet pawn push (promoting or not) is ever
    /// generated here regardless of `all`.
    pub fn new_probcut(pos: &Position, tt_move: Option<Move>, threshold: i32, all: bool) -> Self {
        let is_capture = |m: Move| !m.is_drop() && pos.board().get(m.to_sq()).is_some();
        let tt = tt_move
            .filter(|&m| m.is_ok() && is_capture(m) && pos.pseudo_legal(m, all) && pos.is_legal(m));
        // The capture list is generated at `PROBCUT_INIT` stage entry, not here
        // (`movepick.cpp`): the buffer starts empty and is
        // filled at that `next_move` arm.
        let scratch = take_scratch();
        Self::from_parts(Kind::ProbCut, tt, 0, 0, threshold, [0; 6], scratch, all)
    }

    /// Generate the legal, TT-deduped capture (or, for the `Evasion` kind,
    /// evasion) list into `scratch.buf` at `*_INIT` stage entry — the deferred
    /// analogue of the reference generating `MoveList<CAPTURES>` /
    /// `MoveList<EVASIONS>` only at `CAPTURE_INIT` / `QCAPTURE_INIT` /
    /// `EVASION_INIT` / `PROBCUT_INIT` (`movepick.cpp`),
    /// never at construction. Generation is a pure function of the position, and
    /// the board at INIT-stage entry is identical to the board at construction
    /// (the search restores it around each `next_move`), so deferring it changes
    /// only *when* the list is materialized, never its contents or order.
    ///
    /// The generators emit unscored [`ExtMove`]s (`value: 0`) straight into
    /// `buf`, which is empty at this point; the illegal / TT moves are then
    /// filtered out in place with [`Vec::retain`] (order-preserving), and the
    /// surviving `value`s are filled immediately after by the `*_INIT` scoring.
    /// Sets `end_captures = end_generated = buf.len()`; `QUIET_INIT` later
    /// appends the quiets and extends `end_generated`.
    fn generate_capture_list(&mut self, pos: &Position) {
        let in_check = self.kind == Kind::Evasion;
        let tt = self.tt;
        let all = self.all;
        if in_check {
            pos.generate_evasions(all, &mut self.scratch.buf);
        } else {
            pos.generate_captures(all, &mut self.scratch.buf);
        }
        self.scratch
            .buf
            .retain(|e| pos.is_legal(e.mv) && Some(e.mv) != tt);
        debug_assert!(
            self.scratch.buf.len() <= MAX_MOVES,
            "move buffer overflow: {} > {MAX_MOVES}",
            self.scratch.buf.len()
        );
        self.end_captures = self.scratch.buf.len();
        self.end_generated = self.end_captures;
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        kind: Kind,
        tt: Option<Move>,
        depth: i32,
        ply: i32,
        threshold: i32,
        cont_planes: [usize; 6],
        scratch: PickerScratch,
        all: bool,
    ) -> Self {
        MovePicker {
            kind,
            tt,
            depth,
            ply,
            threshold,
            cont_planes,
            scratch,
            skip_quiets: false,
            stage: Stage::Tt,
            all,
            cur: 0,
            end_cur: 0,
            end_bad_captures: 0,
            // The lists are generated at their `*_INIT` stage, so the buffer
            // starts empty; `generate_capture_list` / `QUIET_INIT` fill these.
            end_captures: 0,
            end_generated: 0,
        }
    }

    /// The reference `select` (`movepick.cpp`): advance `cur` over
    /// `[cur, end_cur)`, returning the first move that is not the TT move and
    /// passes `filter`. `filter` receives `&mut self` (so `GOOD_CAPTURE` can
    /// compact SEE-losers to the front) and the position (for the SEE / legality
    /// predicates), and inspects the current move via `self.cur`.
    fn select<F>(&mut self, pos: &Position, mut filter: F) -> Option<Move>
    where
        F: FnMut(&mut Self, &Position) -> bool,
    {
        while self.cur < self.end_cur {
            let m = self.scratch.buf[self.cur].mv;
            if Some(m) != self.tt && filter(self, pos) {
                // `return *cur++`: the true-path filters never mutate `buf[cur]`,
                // so this equals `m` (the `GOOD_CAPTURE` swap only fires on the
                // false path).
                let r = self.scratch.buf[self.cur].mv;
                self.cur += 1;
                return Some(r);
            }
            self.cur += 1;
        }
        None
    }

    /// Score the captures / evasions region `[0, end_captures)` in place against
    /// the *live* `hist` (`score<CAPTURES>` / `score<EVASIONS>`).
    fn score_captures_in_place(&mut self, pos: &Position, hist: &WorkerHistories) {
        let evasion = self.kind == Kind::Evasion;
        for i in 0..self.end_captures {
            let m = self.scratch.buf[i].mv;
            self.scratch.buf[i].value = if evasion {
                score_evasion(pos, m, hist, self.cont_planes[0])
            } else {
                score_capture(pos, m, hist)
            };
        }
    }

    /// Score the quiets region `[end_captures, end_generated)` in place against
    /// the *live* `hist` (`score<QUIETS>`).
    fn score_quiets_in_place(&mut self, pos: &Position, hist: &WorkerHistories) {
        // Snapshot the `checkSquares` table once for the whole quiet region — one
        // `check_info()` borrow instead of one per scored move. The
        // board is fixed for the picker's lifetime, so the snapshot is valid for
        // every quiet scored here.
        let check_squares = pos.check_squares();
        for i in self.end_captures..self.end_generated {
            let m = self.scratch.buf[i].mv;
            self.scratch.buf[i].value =
                score_quiet(pos, m, self.ply, hist, self.cont_planes, &check_squares);
        }
    }

    /// The next move in picker order, or `None` when exhausted. `hist` are the
    /// live worker history tables, read at each stage's entry to score that
    /// stage (see the module docs on staged-lazy scoring).
    pub fn next_move(&mut self, pos: &Position, hist: &WorkerHistories) -> Option<Move> {
        loop {
            match self.stage {
                // MAIN_TT / EVASION_TT / QSEARCH_TT / PROBCUT_TT
                // (`movepick.cpp`): yield the TT move, then advance to the
                // kind's init stage.
                Stage::Tt => {
                    self.stage = match self.kind {
                        Kind::Main => Stage::CaptureInit,
                        Kind::QSearch => Stage::QcaptureInit,
                        Kind::Evasion => Stage::EvasionInit,
                        Kind::ProbCut => Stage::ProbcutInit,
                    };
                    if let Some(m) = self.tt {
                        return Some(m);
                    }
                }

                // CAPTURE_INIT (`movepick.cpp`): generate the captures
                // (deferred here from construction), score them, full
                // sort, reset the front/bad-capture region. `partial_insertion_sort`
                // with `i32::MIN` promotes every element, so its promotion swap is
                // a self-assignment — a pure descending permutation, which is why
                // legality could be pre-filtered at generation without perturbing
                // order (see the module docs).
                Stage::CaptureInit => {
                    self.generate_capture_list(pos);
                    self.score_captures_in_place(pos, hist);
                    partial_insertion_sort(&mut self.scratch.buf[0..self.end_captures], i32::MIN);
                    self.cur = 0;
                    self.end_bad_captures = 0;
                    self.end_cur = self.end_captures;
                    self.stage = Stage::GoodCapture;
                }

                // GOOD_CAPTURE (`movepick.cpp`): yield captures passing
                // `see_ge(m, -value / 18)`; compact SEE-losers to `[0,
                // end_bad_captures)` for `BAD_CAPTURE`. SEE is evaluated here, at
                // yield, so an early cutoff never pays the remaining captures' SEE.
                Stage::GoodCapture => {
                    if let Some(m) = self.select(pos, |s, p| {
                        let e = s.scratch.buf[s.cur];
                        if p.see_ge(e.mv, -e.value / 18) {
                            true
                        } else {
                            s.scratch.buf.swap(s.end_bad_captures, s.cur);
                            s.end_bad_captures += 1;
                            false
                        }
                    }) {
                        return Some(m);
                    }
                    self.stage = Stage::QuietInit;
                }

                // QUIET_INIT (`movepick.cpp`): generate the raw quiets,
                // score them, and partial-sort with the depth-scaled limit — all
                // inside `if (!skipQuiets)`. Deferring generation to here (rather
                // than to construction) matches the reference: a node that cut
                // off during the TT / good-capture stages never reaches this arm
                // and so never pays quiet generation, and a node with
                // `skip_quiets` already set (late-move pruning) generates nothing.
                // The `GOOD_QUIET` pointer setup below is still performed either
                // way (unused when skipping). `cur` / `end_cur` point at the
                // quiets region `[end_captures, end_generated)`, which is empty
                // (`end_generated == end_captures`) when no quiets were generated.
                Stage::QuietInit => {
                    if !self.skip_quiets {
                        // The reference scores and partial-sorts the *raw*
                        // generated quiet buffer (`MoveList<QUIETS>`,
                        // `movepick.cpp`): both the TT move and
                        // pseudo-legal-but-illegal (pinned-piece) quiets are
                        // present at `QUIET_INIT`, and legality is only tested
                        // later at the search loop's `if (!pos.legal(move))
                        // continue`. So neither is filtered here — they are
                        // dropped at yield time in `GoodQuiet` / `BadQuiet`,
                        // matching the reference `select`'s `*cur != ttMove` plus
                        // the search loop's legality gate.
                        //
                        // Their presence in the buffer is load-bearing:
                        // `partial_insertion_sort` promotes every element whose
                        // score clears the depth-scaled limit, and each promotion
                        // swaps a tail element into the promoted slot's old place
                        // (`*p = *++sortedEnd`, `movepick.cpp`). Which
                        // elements sit in the buffer therefore changes the
                        // *unsorted tail's* final order — so a high-scoring TT
                        // move or illegal quiet reshapes the emitted order of the
                        // legal quiets even though it is never itself yielded.
                        // Dropping either here would silently reorder the
                        // surviving quiets. (Contrast the capture / evasion /
                        // ProbCut lists, which are full-sorted.)
                        //
                        // The quiets are appended after the captures region
                        // (`[end_captures, …)`), the output-equivalent deviation
                        // from the pin's overwrite-from-`endBadCaptures` layout
                        // documented for the single-buffer layout; only the
                        // generation *timing* moved to this arm.
                        pos.generate_quiets(self.all, &mut self.scratch.buf);
                        debug_assert!(
                            self.scratch.buf.len() <= MAX_MOVES,
                            "move buffer overflow: {} > {MAX_MOVES}",
                            self.scratch.buf.len()
                        );
                        self.end_generated = self.scratch.buf.len();

                        self.score_quiets_in_place(pos, hist);
                        partial_insertion_sort(
                            &mut self.scratch.buf[self.end_captures..self.end_generated],
                            -3560 * self.depth,
                        );
                    }
                    self.cur = self.end_captures;
                    self.end_cur = self.end_generated;
                    self.stage = Stage::GoodQuiet;
                }

                // GOOD_QUIET (`movepick.cpp`): yield quiets scoring above
                // the threshold, skipping the TT move and — the port's yield-time
                // legality gate — pseudo-legal-but-illegal quiets. Then reset `cur`
                // / `end_cur` to the bad-capture front region for `BAD_CAPTURE`.
                Stage::GoodQuiet => {
                    if !self.skip_quiets
                        && let Some(m) = self.select(pos, |s, p| {
                            let e = s.scratch.buf[s.cur];
                            e.value > GOOD_QUIET_THRESHOLD && p.is_legal(e.mv)
                        })
                    {
                        return Some(m);
                    }
                    self.cur = 0;
                    self.end_cur = self.end_bad_captures;
                    self.stage = Stage::BadCapture;
                }

                // BAD_CAPTURE (`movepick.cpp`): replay the compacted
                // SEE-losing captures (already TT-deduped at generation), then
                // reset `cur` / `end_cur` to the full quiets region for `BAD_QUIET`.
                Stage::BadCapture => {
                    if let Some(m) = self.select(pos, |_, _| true) {
                        return Some(m);
                    }
                    self.cur = self.end_captures;
                    self.end_cur = self.end_generated;
                    self.stage = Stage::BadQuiet;
                }

                // BAD_QUIET (`movepick.cpp`): yield quiets at or below the
                // threshold (legal, non-TT). With `skip_quiets` set the reference
                // returns `Move::none()` here.
                Stage::BadQuiet => {
                    if self.skip_quiets {
                        self.stage = Stage::Done;
                        continue;
                    }
                    return self.select(pos, |s, p| {
                        let e = s.scratch.buf[s.cur];
                        e.value <= GOOD_QUIET_THRESHOLD && p.is_legal(e.mv)
                    });
                }

                // QCAPTURE_INIT / EVASION_INIT / PROBCUT_INIT
                // (`movepick.cpp`): generate the single list
                // (deferred here from construction), score it, full
                // sort, then a lone `select` loop with no good/bad split.
                Stage::QcaptureInit => {
                    self.generate_capture_list(pos);
                    self.score_captures_in_place(pos, hist);
                    partial_insertion_sort(&mut self.scratch.buf[0..self.end_captures], i32::MIN);
                    self.cur = 0;
                    self.end_cur = self.end_captures;
                    self.stage = Stage::Qcapture;
                }
                Stage::EvasionInit => {
                    self.generate_capture_list(pos);
                    self.score_captures_in_place(pos, hist);
                    partial_insertion_sort(&mut self.scratch.buf[0..self.end_captures], i32::MIN);
                    self.cur = 0;
                    self.end_cur = self.end_captures;
                    self.stage = Stage::Evasion;
                }
                Stage::ProbcutInit => {
                    self.generate_capture_list(pos);
                    self.score_captures_in_place(pos, hist);
                    partial_insertion_sort(&mut self.scratch.buf[0..self.end_captures], i32::MIN);
                    self.cur = 0;
                    self.end_cur = self.end_captures;
                    self.stage = Stage::Probcut;
                }

                // QCAPTURE / EVASION (`movepick.cpp`): best-first, no
                // filter (both lists are pre-filtered legal and TT-deduped).
                Stage::Qcapture | Stage::Evasion => {
                    return self.select(pos, |_, _| true);
                }

                // PROBCUT (`movepick.cpp`): yield captures with
                // `see_ge(m, threshold)`, SEE evaluated lazily per move.
                Stage::Probcut => {
                    return self.select(pos, |s, p| {
                        let e = s.scratch.buf[s.cur];
                        p.see_ge(e.mv, s.threshold)
                    });
                }

                Stage::Done => return None,
            }
        }
    }

    /// Skip the remaining quiet stages (`GOOD_QUIET` / `BAD_QUIET`), matching the
    /// reference `skip_quiet_moves()` (`movepick.cpp`). Deferred bad captures
    /// are still replayed. Once set the flag stays set.
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }
}

/// The retained pre-single-buffer multi-`Vec` `MovePicker`, kept verbatim as a
/// test-only
/// twin. The single-buffer production picker must yield an **identical** move
/// sequence to this twin in every state (the single-buffer gate); the equality
/// tests
/// in the parent `tests` module drive both side by side.
#[cfg(test)]
mod twin {
    use std::cell::RefCell;

    use attic_state::{Move, Position};

    use super::{
        ExtMove, GOOD_QUIET_THRESHOLD, Kind, partial_insertion_sort, score_capture, score_evasion,
        score_quiet,
    };
    use crate::update::WorkerHistories;

    /// The pre-single-buffer multi-`Vec` scratch (raw + scored + four segment
    /// vectors).
    #[derive(Default)]
    struct PickerScratch {
        raw_captures: Vec<Move>,
        raw_quiets: Vec<Move>,
        good_captures: Vec<Move>,
        bad_captures: Vec<Move>,
        good_quiets: Vec<Move>,
        bad_quiets: Vec<Move>,
        scored: Vec<ExtMove>,
    }

    impl PickerScratch {
        fn clear(&mut self) {
            self.raw_captures.clear();
            self.raw_quiets.clear();
            self.good_captures.clear();
            self.bad_captures.clear();
            self.good_quiets.clear();
            self.bad_quiets.clear();
            self.scored.clear();
        }
    }

    thread_local! {
        static SCRATCH_POOL: RefCell<Vec<PickerScratch>> = const { RefCell::new(Vec::new()) };
    }

    fn take_scratch() -> PickerScratch {
        let mut scratch = SCRATCH_POOL
            .with(|pool| pool.borrow_mut().pop())
            .unwrap_or_default();
        scratch.clear();
        scratch
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Stage {
        Tt,
        CaptureInit,
        GoodCapture,
        QuietInit,
        GoodQuiet,
        BadCapture,
        BadQuiet,
        Done,
    }

    pub(super) struct TwinMovePicker {
        kind: Kind,
        tt: Option<Move>,
        depth: i32,
        ply: i32,
        threshold: i32,
        cont_planes: [usize; 6],
        scratch: PickerScratch,
        skip_quiets: bool,
        stage: Stage,
        idx: usize,
    }

    impl Drop for TwinMovePicker {
        fn drop(&mut self) {
            let scratch = std::mem::take(&mut self.scratch);
            SCRATCH_POOL.with(|pool| pool.borrow_mut().push(scratch));
        }
    }

    impl TwinMovePicker {
        pub(super) fn new_qsearch(
            pos: &Position,
            tt_move: Option<Move>,
            cont_planes: [usize; 6],
            all: bool,
        ) -> Self {
            let in_check = pos.in_check();
            let tt = tt_move.filter(|&m| m.is_ok() && pos.pseudo_legal(m, all) && pos.is_legal(m));
            let mut scratch = take_scratch();
            Self::generate_into(pos, in_check, tt, all, &mut scratch.raw_captures);
            Self::from_parts(
                if in_check {
                    Kind::Evasion
                } else {
                    Kind::QSearch
                },
                tt,
                0,
                0,
                0,
                cont_planes,
                scratch,
            )
        }

        pub(super) fn new_main_search(
            pos: &Position,
            tt_move: Option<Move>,
            depth: i32,
            ply: i32,
            cont_planes: [usize; 6],
            all: bool,
        ) -> Self {
            let in_check = pos.in_check();
            let tt = tt_move.filter(|&m| m.is_ok() && pos.pseudo_legal(m, all) && pos.is_legal(m));
            let mut scratch = take_scratch();
            if in_check {
                Self::generate_into(pos, true, tt, all, &mut scratch.raw_captures);
                return Self::from_parts(Kind::Evasion, tt, depth, ply, 0, cont_planes, scratch);
            }
            Self::generate_into(pos, false, tt, all, &mut scratch.raw_captures);
            let mut tmp: Vec<ExtMove> = Vec::new();
            pos.generate_quiets(all, &mut tmp);
            scratch.raw_quiets.extend(tmp.into_iter().map(|e| e.mv));
            Self::from_parts(Kind::Main, tt, depth, ply, 0, cont_planes, scratch)
        }

        pub(super) fn new_probcut(
            pos: &Position,
            tt_move: Option<Move>,
            threshold: i32,
            all: bool,
        ) -> Self {
            let is_capture = |m: Move| !m.is_drop() && pos.board().get(m.to_sq()).is_some();
            let tt = tt_move.filter(|&m| {
                m.is_ok() && is_capture(m) && pos.pseudo_legal(m, all) && pos.is_legal(m)
            });
            let mut scratch = take_scratch();
            Self::generate_into(pos, false, tt, all, &mut scratch.raw_captures);
            Self::from_parts(Kind::ProbCut, tt, 0, 0, threshold, [0; 6], scratch)
        }

        fn generate_into(
            pos: &Position,
            in_check: bool,
            tt: Option<Move>,
            all: bool,
            out: &mut Vec<Move>,
        ) {
            // The generators emit `ExtMove`; the twin keeps its raw list as
            // `Move`, so unwrap `.mv` at this boundary — the twin's ordering
            // logic never sees an `ExtMove`.
            let mut tmp: Vec<ExtMove> = Vec::new();
            if in_check {
                pos.generate_evasions(all, &mut tmp);
            } else {
                pos.generate_captures(all, &mut tmp);
            }
            for e in tmp {
                if pos.is_legal(e.mv) && Some(e.mv) != tt {
                    out.push(e.mv);
                }
            }
        }

        fn from_parts(
            kind: Kind,
            tt: Option<Move>,
            depth: i32,
            ply: i32,
            threshold: i32,
            cont_planes: [usize; 6],
            scratch: PickerScratch,
        ) -> Self {
            TwinMovePicker {
                kind,
                tt,
                depth,
                ply,
                threshold,
                cont_planes,
                scratch,
                skip_quiets: false,
                stage: Stage::Tt,
                idx: 0,
            }
        }

        fn init_captures(&mut self, pos: &Position, hist: &WorkerHistories) {
            self.scratch.scored.clear();
            for i in 0..self.scratch.raw_captures.len() {
                let m = self.scratch.raw_captures[i];
                let value = if self.kind == Kind::Evasion {
                    score_evasion(pos, m, hist, self.cont_planes[0])
                } else {
                    score_capture(pos, m, hist)
                };
                self.scratch.scored.push(ExtMove { mv: m, value });
            }
            partial_insertion_sort(&mut self.scratch.scored, i32::MIN);

            match self.kind {
                Kind::Main => {
                    for i in 0..self.scratch.scored.len() {
                        let e = self.scratch.scored[i];
                        if pos.see_ge(e.mv, -e.value / 18) {
                            self.scratch.good_captures.push(e.mv);
                        } else {
                            self.scratch.bad_captures.push(e.mv);
                        }
                    }
                }
                Kind::ProbCut => {
                    for i in 0..self.scratch.scored.len() {
                        let e = self.scratch.scored[i];
                        if pos.see_ge(e.mv, self.threshold) {
                            self.scratch.good_captures.push(e.mv);
                        }
                    }
                }
                Kind::QSearch | Kind::Evasion => {
                    for i in 0..self.scratch.scored.len() {
                        let mv = self.scratch.scored[i].mv;
                        self.scratch.good_captures.push(mv);
                    }
                }
            }
        }

        fn init_quiets(&mut self, pos: &Position, hist: &WorkerHistories) {
            self.scratch.scored.clear();
            let check_squares = pos.check_squares();
            for i in 0..self.scratch.raw_quiets.len() {
                let m = self.scratch.raw_quiets[i];
                let value = score_quiet(pos, m, self.ply, hist, self.cont_planes, &check_squares);
                self.scratch.scored.push(ExtMove { mv: m, value });
            }
            partial_insertion_sort(&mut self.scratch.scored, -3560 * self.depth);
            for i in 0..self.scratch.scored.len() {
                let e = self.scratch.scored[i];
                if e.value > GOOD_QUIET_THRESHOLD {
                    self.scratch.good_quiets.push(e.mv);
                } else {
                    self.scratch.bad_quiets.push(e.mv);
                }
            }
        }

        pub(super) fn next_move(&mut self, pos: &Position, hist: &WorkerHistories) -> Option<Move> {
            loop {
                match self.stage {
                    Stage::Tt => {
                        self.stage = Stage::CaptureInit;
                        if let Some(m) = self.tt {
                            return Some(m);
                        }
                    }
                    Stage::CaptureInit => {
                        self.init_captures(pos, hist);
                        self.stage = Stage::GoodCapture;
                        self.idx = 0;
                    }
                    Stage::GoodCapture => {
                        if let Some(&m) = self.scratch.good_captures.get(self.idx) {
                            self.idx += 1;
                            return Some(m);
                        }
                        self.stage = Stage::QuietInit;
                        self.idx = 0;
                    }
                    Stage::QuietInit => {
                        if self.kind == Kind::Main && !self.skip_quiets {
                            self.init_quiets(pos, hist);
                        }
                        self.stage = Stage::GoodQuiet;
                        self.idx = 0;
                    }
                    Stage::GoodQuiet => {
                        if self.skip_quiets {
                            self.stage = Stage::BadCapture;
                            self.idx = 0;
                            continue;
                        }
                        while let Some(&m) = self.scratch.good_quiets.get(self.idx) {
                            self.idx += 1;
                            if Some(m) != self.tt && pos.is_legal(m) {
                                return Some(m);
                            }
                        }
                        self.stage = Stage::BadCapture;
                        self.idx = 0;
                    }
                    Stage::BadCapture => {
                        if let Some(&m) = self.scratch.bad_captures.get(self.idx) {
                            self.idx += 1;
                            return Some(m);
                        }
                        self.stage = Stage::BadQuiet;
                        self.idx = 0;
                    }
                    Stage::BadQuiet => {
                        if self.skip_quiets {
                            self.stage = Stage::Done;
                            continue;
                        }
                        while let Some(&m) = self.scratch.bad_quiets.get(self.idx) {
                            self.idx += 1;
                            if Some(m) != self.tt && pos.is_legal(m) {
                                return Some(m);
                            }
                        }
                        self.stage = Stage::Done;
                    }
                    Stage::Done => return None,
                }
            }
        }

        pub(super) fn skip_quiet_moves(&mut self) {
            self.skip_quiets = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::ContinuationHistory;
    use attic_state::{Color, Piece, PieceKind, Square, format_usi_move, parse_sfen};

    fn pos(sfen: &str) -> Position {
        parse_sfen(sfen).expect("valid SFEN")
    }

    /// The six sentinel continuation planes at a node with untouched stack cells
    /// (all pointing at plane `0`, the `[0][0][NO_PIECE][SQ_ZERO]` sentinel).
    const SENTINEL_PLANES: [usize; 6] = [0; 6];

    /// A `WorkerHistories` with the reference `clear()` init values, plus the
    /// per-`go` `lowPlyHistory` fill of `98`.
    fn init_histories() -> WorkerHistories {
        let mut h = WorkerHistories::new();
        h.low_ply.fill(98);
        h
    }

    /// Drive a picker to exhaustion against `hist`, collecting USI strings.
    fn collect_usi(mut mp: MovePicker, p: &Position, hist: &WorkerHistories) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(m) = mp.next_move(p, hist) {
            out.push(format_usi_move(m));
        }
        out
    }

    /// Drive a picker to exhaustion, collecting moves.
    fn collect_moves(mut mp: MovePicker, p: &Position, hist: &WorkerHistories) -> Vec<Move> {
        let mut out = Vec::new();
        while let Some(m) = mp.next_move(p, hist) {
            out.push(m);
        }
        out
    }

    fn legal_moves(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_legal_all(&mut v);
        v
    }

    fn is_capture(p: &Position, m: Move) -> bool {
        !m.is_drop() && p.board().get(m.to_sq()).is_some()
    }

    // ------------------------------------------------------------------
    //   qsearch picker (init histories): MVV / capture-first behaviour
    // ------------------------------------------------------------------

    fn qpicker(p: &Position, tt: Option<Move>) -> MovePicker {
        MovePicker::new_qsearch(p, tt, SENTINEL_PLANES, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn qsearch_captures_are_ordered_by_victim_value_descending() {
        let p = pos("9/9/9/9/b3R3g/9/4p4/9/K7k b - 1");
        let h = init_histories();
        assert!(!p.in_check());
        assert_eq!(
            collect_usi(qpicker(&p, None), &p, &h),
            vec!["5e9e".to_string(), "5e1e".to_string(), "5e5g".to_string()],
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn qsearch_equal_value_victims_keep_reference_generation_order() {
        let p = pos("9/9/9/2p1p4/2S1G4/9/9/9/K7k b - 1");
        let h = init_histories();
        assert_eq!(
            collect_usi(qpicker(&p, None), &p, &h),
            vec!["7e7d".to_string(), "5e5d".to_string()],
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn qsearch_tt_move_is_yielded_first_and_not_duplicated() {
        let p = pos("9/9/9/9/b3R3g/9/4p4/9/K7k b - 1");
        let h = init_histories();
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let tt = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 6).unwrap(), rook);
        assert_eq!(format_usi_move(tt), "5e5g");

        let order = collect_usi(qpicker(&p, Some(tt)), &p, &h);
        assert_eq!(
            order,
            vec!["5e5g".to_string(), "5e9e".to_string(), "5e1e".to_string()],
        );
        assert_eq!(order.iter().filter(|s| *s == "5e5g").count(), 1);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn qsearch_not_in_check_yields_only_captures_no_quiet_checks() {
        let p = pos("k8/9/9/9/4R4/9/4p4/9/8K b - 1");
        let h = init_histories();
        assert!(!p.in_check());
        let order = collect_usi(qpicker(&p, None), &p, &h);
        assert_eq!(order, vec!["5e5g".to_string()]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn qsearch_generate_all_legal_moves_adds_capture_nonpromotion() {
        // A Black lance on 5d capturing a White pawn on 5b (rank 1, enemy second
        // rank). The default qsearch picker yields only the promotion; with
        // `all == true` the suppressed non-promotion is additionally yielded.
        // This is the MovePicker-level enumeration the GenerateAllLegalMoves
        // option asks for.
        let p = pos("k8/4p4/9/4L4/9/9/9/9/8K b - 1");
        let h = init_histories();
        assert!(!p.in_check());

        let off = collect_usi(
            MovePicker::new_qsearch(&p, None, SENTINEL_PLANES, false),
            &p,
            &h,
        );
        assert_eq!(off, vec!["5d5b+".to_string()], "default: promotion only");

        let on = collect_usi(
            MovePicker::new_qsearch(&p, None, SENTINEL_PLANES, true),
            &p,
            &h,
        );
        assert!(
            on.contains(&"5d5b".to_string()),
            "all-mode must yield the lance capture non-promotion: {on:?}",
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn qsearch_evasions_are_exactly_the_legal_set_with_captures_first() {
        let p = pos("4r4/5G3/9/9/9/9/9/9/4K3k b - 1");
        let h = init_histories();
        assert!(p.in_check());

        let picker_moves = collect_moves(qpicker(&p, None), &p, &h);
        let picker_set: std::collections::HashSet<Move> = picker_moves.iter().copied().collect();
        let legal_set: std::collections::HashSet<Move> = legal_moves(&p).into_iter().collect();
        assert_eq!(picker_set, legal_set);

        assert_eq!(format_usi_move(picker_moves[0]), "4b5a");
        let first_quiet = picker_moves
            .iter()
            .position(|&m| !is_capture(&p, m))
            .unwrap();
        assert!(
            picker_moves[..first_quiet]
                .iter()
                .all(|&m| is_capture(&p, m))
        );
        assert!(
            picker_moves[first_quiet..]
                .iter()
                .all(|&m| !is_capture(&p, m))
        );
    }

    // ------------------------------------------------------------------
    //   main-search picker with reference clear() histories
    // ------------------------------------------------------------------

    fn main_picker(p: &Position, tt: Option<Move>, depth: i32, ply: i32) -> MovePicker {
        MovePicker::new_main_search(p, tt, depth, ply, SENTINEL_PLANES, false)
    }

    /// The union of legal captures and legal quiets — the moves the not-in-check
    /// main picker is responsible for.
    fn legal_capture_and_quiet_set(p: &Position) -> std::collections::HashSet<Move> {
        let mut caps: Vec<ExtMove> = Vec::new();
        p.generate_captures(false, &mut caps);
        let mut quiets: Vec<ExtMove> = Vec::new();
        p.generate_quiets(false, &mut quiets);
        caps.into_iter()
            .chain(quiets)
            .map(|e| e.mv)
            .filter(|&m| p.is_legal(m))
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_capture_order_is_mvv_with_initial_histories() {
        let p = pos("9/9/9/9/b3R3g/9/4p4/9/K7k b - 1");
        let h = init_histories();
        let order = collect_usi(main_picker(&p, None, 8, 0), &p, &h);
        assert_eq!(
            &order[..3],
            &["5e9e".to_string(), "5e1e".to_string(), "5e5g".to_string()],
        );
        // Everything after the three captures is a quiet rook move.
        assert!(
            order[3..]
                .iter()
                .all(|s| s.starts_with("5e") || s.starts_with("9i"))
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_tt_move_leads_and_is_never_duplicated() {
        let p = pos("9/9/9/9/b3R3g/9/4p4/9/K7k b - 1");
        let h = init_histories();
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let tt = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 6).unwrap(), rook);
        assert_eq!(format_usi_move(tt), "5e5g");

        let order = collect_usi(main_picker(&p, Some(tt), 8, 0), &p, &h);
        assert_eq!(order[0], "5e5g");
        assert_eq!(order.iter().filter(|s| *s == "5e5g").count(), 1);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_emits_every_legal_move_exactly_once_not_in_check() {
        let p =
            pos("l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1");
        let h = init_histories();
        assert!(!p.in_check());
        let emitted = collect_moves(main_picker(&p, None, 6, 3), &p, &h);

        let emitted_set: std::collections::HashSet<Move> = emitted.iter().copied().collect();
        assert_eq!(emitted_set.len(), emitted.len(), "a move was emitted twice");

        // Exactly the legal captures ∪ quiets.
        assert_eq!(emitted_set, legal_capture_and_quiet_set(&p));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_emits_every_legal_move_exactly_once_with_tt() {
        let p =
            pos("l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1");
        let h = init_histories();

        // Pick an arbitrary legal quiet move as the TT move.
        let mut caps: Vec<ExtMove> = Vec::new();
        p.generate_captures(false, &mut caps);
        let cap_set: std::collections::HashSet<Move> = caps.iter().map(|e| e.mv).collect();
        let tt = legal_moves(&p)
            .into_iter()
            .find(|m| !cap_set.contains(m) && !m.is_drop())
            .expect("a legal quiet move exists");

        let emitted = collect_moves(main_picker(&p, Some(tt), 6, 3), &p, &h);
        assert_eq!(emitted[0], tt, "TT move must lead");
        let emitted_set: std::collections::HashSet<Move> = emitted.iter().copied().collect();
        assert_eq!(emitted_set.len(), emitted.len(), "a move was emitted twice");
        assert_eq!(emitted_set, legal_capture_and_quiet_set(&p));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_good_bad_capture_split_respects_see_boundary() {
        let p = pos("6rkg/8G/9/9/9/9/9/9/K8 b - 1");
        let h = init_histories();
        let emitted = collect_moves(main_picker(&p, None, 6, 0), &p, &h);
        let boundary = split_and_check(&p, &h, &emitted);
        assert!(
            boundary,
            "good/bad capture split violated the see_ge boundary"
        );
    }

    /// Verify the good-capture / bad-capture ordering invariant on an emitted
    /// list from an all-initial-history main picker with no quiets skipped:
    /// every capture emitted *before* the first quiet must pass
    /// `see_ge(m, -value/18)`, and every capture emitted *after* a quiet must
    /// fail it.
    fn split_and_check(p: &Position, hist: &WorkerHistories, emitted: &[Move]) -> bool {
        let is_cap = |m: Move| !m.is_drop() && p.board().get(m.to_sq()).is_some();
        let first_quiet = emitted.iter().position(|&m| !is_cap(m));
        let cap_score = |m: Move| score_capture(p, m, hist);
        match first_quiet {
            None => emitted
                .iter()
                .all(|&m| !is_cap(m) || p.see_ge(m, -cap_score(m) / 18)),
            Some(fq) => {
                let good_ok = emitted[..fq]
                    .iter()
                    .all(|&m| !is_cap(m) || p.see_ge(m, -cap_score(m) / 18));
                let bad_ok = emitted[fq..]
                    .iter()
                    .filter(|&&m| is_cap(m))
                    .all(|&m| !p.see_ge(m, -cap_score(m) / 18));
                good_ok && bad_ok
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_skip_quiet_moves_drops_quiets_but_keeps_bad_captures() {
        let p =
            pos("l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1");
        let h = init_histories();

        // Full run to know the bad-capture set.
        let full = collect_moves(main_picker(&p, None, 6, 3), &p, &h);
        let is_cap = |m: Move| !m.is_drop() && p.board().get(m.to_sq()).is_some();

        // Skip quiets from the very start: only captures (good + bad) remain.
        let mut mp = main_picker(&p, None, 6, 3);
        mp.skip_quiet_moves();
        let mut kept = Vec::new();
        while let Some(m) = mp.next_move(&p, &h) {
            kept.push(m);
        }

        assert!(
            kept.iter().all(|&m| is_cap(m)),
            "skip_quiet_moves must leave only captures"
        );
        let full_caps: std::collections::HashSet<Move> =
            full.into_iter().filter(|&m| is_cap(m)).collect();
        let kept_set: std::collections::HashSet<Move> = kept.iter().copied().collect();
        assert_eq!(kept_set, full_caps, "captures must be unaffected by skip");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_skip_mid_iteration_still_replays_bad_captures() {
        let p =
            pos("l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1");
        let h = init_histories();
        let is_cap = |m: Move| !m.is_drop() && p.board().get(m.to_sq()).is_some();

        let mut mp = main_picker(&p, None, 6, 3);
        // Pull a few moves, then skip quiets.
        let mut seen = Vec::new();
        for _ in 0..2 {
            if let Some(m) = mp.next_move(&p, &h) {
                seen.push(m);
            }
        }
        mp.skip_quiet_moves();
        while let Some(m) = mp.next_move(&p, &h) {
            seen.push(m);
        }
        // No quiet appears after the skip point; and the set of captures equals
        // the captures of a never-skipped run.
        let full_caps: std::collections::HashSet<Move> =
            collect_moves(main_picker(&p, None, 6, 3), &p, &h)
                .into_iter()
                .filter(|&m| is_cap(m))
                .collect();
        let seen_caps: std::collections::HashSet<Move> =
            seen.iter().copied().filter(|&m| is_cap(m)).collect();
        assert_eq!(seen_caps, full_caps, "bad captures must still be replayed");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_in_check_emits_exactly_the_legal_evasions() {
        let p = pos("4r4/5G3/9/9/9/9/9/9/4K3k b - 1");
        let h = init_histories();
        assert!(p.in_check());
        let emitted = collect_moves(main_picker(&p, None, 6, 0), &p, &h);

        let emitted_set: std::collections::HashSet<Move> = emitted.iter().copied().collect();
        let legal_set: std::collections::HashSet<Move> = legal_moves(&p).into_iter().collect();
        assert_eq!(emitted_set.len(), emitted.len(), "duplicate evasion");
        assert_eq!(emitted_set, legal_set, "evasions must equal the legal set");

        // Capturing evasion (gold takes rook) leads.
        assert_eq!(format_usi_move(emitted[0]), "4b5a");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn main_quiet_direct_check_bonus_orders_a_checking_move_first() {
        let p = pos("4k4/9/R8/9/9/9/9/9/8K b - 1");
        let h = init_histories();
        assert!(!p.in_check());
        let emitted = collect_moves(main_picker(&p, None, 6, 0), &p, &h);

        let is_quiet = |m: Move| m.is_drop() || p.board().get(m.to_sq()).is_none();
        let quiets: Vec<Move> = emitted.into_iter().filter(|&m| is_quiet(m)).collect();
        let checking: Vec<usize> = quiets
            .iter()
            .enumerate()
            .filter(|&(_, &m)| p.gives_direct_check(m) && p.see_ge(m, -75))
            .map(|(i, _)| i)
            .collect();
        assert!(
            !checking.is_empty(),
            "the rook has at least one quiet checking move"
        );
        let last_check = *checking.iter().max().unwrap();
        let first_noncheck = quiets
            .iter()
            .position(|&m| !(p.gives_direct_check(m) && p.see_ge(m, -75)));
        if let Some(fnc) = first_noncheck {
            assert!(
                last_check < fnc,
                "a checking quiet ({last_check}) sorted after a non-checking quiet ({fnc})"
            );
        }
    }

    // ------------------------------------------------------------------
    //   Staged-lazy scoring: a table update performed between
    //   two stages provably reorders the later stage.
    // ------------------------------------------------------------------

    /// A picker scores the quiets at `QUIET_INIT`, *after* the captures have been
    /// emitted. If `mainHistory` for one quiet is bumped between draining the
    /// captures and entering `QUIET_INIT`, the eager (construction-time) design
    /// would miss it, but the staged design must surface the bumped quiet first.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn quiet_scoring_reads_table_update_made_after_captures() {
        // Rook on 5e: one good capture (the pawn on 5g) then many quiet rook /
        // king moves. No quiet gives check, so on untouched tables every quiet
        // ties and is emitted in generation order.
        let p = pos("9/9/9/9/4R4/9/4p4/9/K7k b - 1");
        let us = p.side_to_move();

        // Baseline quiet order on untouched tables; the genuinely-last quiet can
        // only be moved forward by a score bump, so it is the discriminator.
        let h0 = init_histories();
        let mut base_quiets = Vec::new();
        {
            let mut mp0 = main_picker(&p, None, 6, 5);
            while let Some(m) = mp0.next_move(&p, &h0) {
                if !is_capture(&p, m) {
                    base_quiets.push(m);
                }
            }
        }
        // Compare two *non-checking* quiets: a checking quiet carries the
        // +16384 direct-check bonus, which a gravity-clamped mainHistory bump
        // cannot overcome. Among the tie-scored non-checking quiets, `a` leads
        // `b` in generation order on untouched tables.
        let noncheck: Vec<Move> = base_quiets
            .iter()
            .copied()
            .filter(|&m| !(p.gives_direct_check(m) && p.see_ge(m, -75)))
            .collect();
        assert!(noncheck.len() >= 2, "need two non-checking quiets");
        let a = noncheck[0];
        let b = *noncheck.last().unwrap();
        assert_ne!(a, b);

        // Drive a fresh picker: drain the capture, THEN bump mainHistory[b]
        // before the first quiet is pulled (i.e. before QUIET_INIT runs). Staged
        // scoring must read the bump and surface `b` ahead of `a`.
        let mut h = init_histories();
        let mut mp = main_picker(&p, None, 6, 5);
        let first = mp.next_move(&p, &h).unwrap();
        assert!(is_capture(&p, first), "the good capture must lead");

        h.main.update(us, b, 20_000);

        let mut quiets = Vec::new();
        while let Some(m) = mp.next_move(&p, &h) {
            if !is_capture(&p, m) {
                quiets.push(m);
            }
        }
        let pos_a = quiets.iter().position(|&m| m == a).unwrap();
        let pos_b = quiets.iter().position(|&m| m == b).unwrap();
        assert!(
            pos_b < pos_a,
            "the quiet bumped after the capture stage must sort ahead — eager scoring would have kept `a` first"
        );
    }

    /// The capture stage scores at `CAPTURE_INIT`, on the first `next_move` after
    /// the TT move. A `captureHistory` bump applied between constructing the
    /// picker (TT emitted) and the first non-TT `next_move` must reorder the
    /// good captures — the property the eager snapshot could not honour.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn capture_scoring_reads_table_update_made_after_tt() {
        // Two equal-victim captures so the tie-break is decided by captureHistory:
        // silver 7e x 7d(pawn) and gold 5e x 5d(pawn).
        let p = pos("9/9/9/2p1p4/2S1G4/9/9/9/K7k b - 1");
        let mut h = init_histories();

        let cap_a = Move::make(
            Square::new(6, 4).unwrap(),
            Square::new(6, 3).unwrap(),
            Piece::new(PieceKind::Silver, Color::Black),
        );
        let cap_b = Move::make(
            Square::new(4, 4).unwrap(),
            Square::new(4, 3).unwrap(),
            Piece::new(PieceKind::Gold, Color::Black),
        );
        assert!(legal_moves(&p).contains(&cap_a));
        assert!(legal_moves(&p).contains(&cap_b));

        // Use the silver capture as a (legal) TT move so there is a TT stage to
        // sit between construction and CAPTURE_INIT. It is emitted first and
        // deduped; the remaining capture (cap_b) is scored at CAPTURE_INIT.
        // Bump captureHistory for cap_b *after* construction; a plain re-run
        // with the bump present shows the order the eager design would have
        // frozen instead.
        let victim = p.board().get(cap_b.to_sq()).unwrap();
        let moved = cap_b.moved_piece_after();
        h.capture.update(moved, cap_b.to_sq(), victim, 12_000);

        // Fresh picker, no TT, both captures scored at CAPTURE_INIT against the
        // updated table: cap_b (bumped) must now precede cap_a.
        let order = collect_moves(main_picker(&p, None, 6, 0), &p, &h);
        let ia = order.iter().position(|&m| m == cap_a).unwrap();
        let ib = order.iter().position(|&m| m == cap_b).unwrap();
        assert!(ib < ia, "the capture with the bumped history must lead");

        // Now drive a picker whose CAPTURE_INIT runs *after* the bump is applied
        // between TT emission and the first non-TT next_move, and confirm it
        // reflects the same (post-bump) ordering — i.e. scoring is not frozen at
        // construction.
        let mut h2 = init_histories();
        let mut mp = MovePicker::new_main_search(&p, Some(cap_a), 6, 0, SENTINEL_PLANES, false);
        let tt_out = mp.next_move(&p, &h2).unwrap();
        assert_eq!(tt_out, cap_a, "TT move leads");
        // Bump cap_b between the TT stage and CAPTURE_INIT.
        h2.capture.update(moved, cap_b.to_sq(), victim, 12_000);
        let rest = {
            let mut v = Vec::new();
            while let Some(m) = mp.next_move(&p, &h2) {
                v.push(m);
            }
            v
        };
        // cap_b is the only remaining capture; it is emitted (proving CAPTURE_INIT
        // ran after the TT stage, reading the live table).
        assert!(rest.contains(&cap_b), "the non-TT capture must be emitted");
        assert!(is_capture(&p, rest[0]), "captures precede quiets");
        assert_eq!(rest[0], cap_b, "the sole remaining capture leads the rest");
    }

    /// Sanity: the continuation planes are read live. Two sibling quiets whose
    /// only score difference is one continuation plane cell must reorder when
    /// that cell is bumped between construction and `QUIET_INIT`.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn quiet_scoring_reads_live_continuation_plane() {
        let p = pos("9/9/9/9/4R4/9/9/9/K7k b - 1");
        // Point cont plane index 0 at a real (non-sentinel) plane so the update
        // lands somewhere the score reads (planes [0][1][2][3][5]).
        let plane = ContinuationHistory::plane_index(
            false,
            false,
            Piece::new(PieceKind::Rook, Color::Black),
            Square::new(4, 4).unwrap(),
        );
        let cont_planes = [plane, 0, 0, 0, 0, 0];
        let mut h = init_histories();

        let q_target = Move::make(
            Square::new(4, 4).unwrap(),
            Square::new(4, 3).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert!(legal_moves(&p).contains(&q_target));

        let mut mp = MovePicker::new_main_search(&p, None, 6, 0, cont_planes, false);
        // No captures in this position, so the first next_move enters QUIET_INIT.
        // Bump the plane cell for q_target before that first call.
        h.continuation.update_at(
            plane,
            q_target.moved_piece_after(),
            q_target.to_sq(),
            30_000,
        );
        let first = mp.next_move(&p, &h).unwrap();
        assert_eq!(
            first, q_target,
            "the quiet whose live continuation cell was bumped must lead"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn illegal_quiet_in_buffer_reorders_the_legal_quiets() {
        // Hand-computed partial-sort tail swap driven purely by a
        // pseudo-legal-but-illegal quiet's presence in the scored buffer.
        //
        // Position (Black to move): a Black knight on 1g is pinned to the Black
        // king on 1i by the White lance on 1a, so its only pseudo-legal move
        // (N-1g-2e) is illegal. That knight move also gives direct check to the
        // White king on 3g, so it scores the +16384 direct-check bonus; every
        // other quiet scores the uniform `clear()` value
        //   2*0 + 2*(-1238) + 5*(-523) = -5091.
        // At depth 1 the `QUIET_INIT` limit is `-3560 * 1 = -3560`: the -5091
        // quiets fall *below* it (never promoted by `partial_insertion_sort`),
        // while the illegal knight move at 11293 clears it and is promoted.
        //
        // Generation order (`generate_quiets`, file-major) puts the three legal
        // pawn pushes at buffer indices 0..3 and the illegal knight move at
        // index 3. The single promotion executes `a[3] = a[1]` (relocating the
        // second pawn) then inserts the knight at the front — so filtering the
        // illegal move back out leaves the pawns in the order p0, p2, p1 rather
        // than p0, p1, p2. A legal-only buffer would never promote anything and
        // would emit p0, p1, p2, so this test fails against that shape.
        let p = pos("8l/9/6k2/9/9/PPP6/8N/9/8K b - 1");
        assert!(!p.in_check(), "Black is not in check → quiet stages run");

        // The illegal knight move is generated (pseudo-legal) but not legal.
        let knight_illegal = "1g2e";
        let mut raw_ext: Vec<ExtMove> = Vec::new();
        p.generate_quiets(false, &mut raw_ext);
        let raw_quiets: Vec<Move> = raw_ext.into_iter().map(|e| e.mv).collect();
        let raw_usi: Vec<String> = raw_quiets.iter().map(|&m| format_usi_move(m)).collect();
        assert_eq!(
            raw_usi.iter().position(|s| s == knight_illegal),
            Some(3),
            "the illegal knight move sits at buffer index 3 (after the 3 pawns)",
        );
        let knight_move = raw_quiets[3];
        assert!(
            !p.is_legal(knight_move),
            "the pinned knight move is illegal"
        );
        assert!(
            p.gives_direct_check(knight_move) && p.see_ge(knight_move, -75),
            "the knight move carries the +16384 direct-check bonus",
        );
        // The three pawn pushes are legal and lead the buffer.
        for usi in ["7f7e", "8f8e", "9f9e"] {
            let idx = raw_usi
                .iter()
                .position(|s| s == usi)
                .expect("pawn push present");
            assert!(idx < 3, "{usi} is one of the three leading legal quiets");
        }

        // Drive the real picker with the reference `clear()` histories (no
        // lowPlyHistory fill, so the -5091 baseline is exact).
        let h = WorkerHistories::new();
        let mut mp = main_picker(&p, None, 1, 0);
        let mut emitted = Vec::new();
        while let Some(m) = mp.next_move(&p, &h) {
            emitted.push(format_usi_move(m));
        }

        // No captures in this position, so the emitted stream is the quiets.
        // The illegal knight move must never appear.
        assert!(
            !emitted.contains(&knight_illegal.to_string()),
            "the illegal quiet is filtered at yield: {emitted:?}",
        );
        // The tail swap: p0, then p2 and p1 in *swapped* order. Under a
        // legal-only buffer this would be 7f7e, 8f8e, 9f9e.
        assert_eq!(
            &emitted[..3],
            &["7f7e".to_string(), "9f9e".to_string(), "8f8e".to_string()],
            "the illegal quiet's promotion swaps the last two pawns; \
             legal-only ordering (7f7e, 8f8e, 9f9e) must fail here",
        );
    }

    // ------------------------------------------------------------------
    //   Single-buffer gate: the single-buffer production picker must yield an
    //   IDENTICAL sequence to the retained multi-Vec twin in every state.
    // ------------------------------------------------------------------

    use super::twin::TwinMovePicker;

    /// A deterministic non-zero fill of every history table `mp` scoring reads,
    /// parameterised by `seed` so distinct profiles exercise the good/bad splits
    /// (SEE good/bad captures, the `-14000` good/bad quiet threshold) differently.
    /// Uniform fills alone would leave both pickers scoring identically anyway;
    /// the point is to move moves across the segment boundaries so the stage
    /// machine — not just the shared scorer — is what the equality asserts.
    fn filled_histories(seed: i16) -> WorkerHistories {
        let mut h = WorkerHistories::new();
        h.main.fill(3 * seed);
        h.capture.fill(11 * seed);
        h.shared.fill_pawn(-2 * seed);
        h.low_ply.fill(98 + seed);
        // A single continuation fill drives all five read planes (SENTINEL_PLANES
        // point every read at plane 0). Large-magnitude values push quiets across
        // the `-14000` threshold for the negative seeds.
        h.continuation.fill(-1700 * seed);
        h
    }

    /// How a gate case selects its TT move — either a fixed USI string (resolved
    /// against the legal set) or a role picked programmatically so it is always
    /// legal in the fixture.
    enum TtPick {
        None,
        Usi(&'static str),
        FirstLegalQuiet,
    }

    /// The construction inputs a gate case feeds *identically* to both pickers.
    struct GateCase {
        sfen: &'static str,
        tt: TtPick,
        depth: i32,
        ply: i32,
    }

    /// Resolve a fixed USI TT move against the legal set (panics on a typo).
    fn tt_move(p: &Position, tt_usi: Option<&str>) -> Option<Move> {
        tt_usi.map(|u| {
            legal_moves(p)
                .into_iter()
                .find(|&m| format_usi_move(m) == u)
                .unwrap_or_else(|| panic!("gate TT move {u} is not legal in the fixture"))
        })
    }

    /// Resolve a [`TtPick`] into a concrete (always-legal) TT move.
    fn tt_pick(p: &Position, pick: &TtPick) -> Option<Move> {
        match pick {
            TtPick::None => None,
            TtPick::Usi(u) => tt_move(p, Some(u)),
            TtPick::FirstLegalQuiet => legal_moves(p)
                .into_iter()
                .find(|&m| !is_capture(p, m) && !m.is_drop()),
        }
    }

    /// Drive both pickers with the identical `skip_at` schedule (call
    /// `skip_quiet_moves` immediately before the `skip_at`-th `next_move`) and
    /// assert an identical yielded sequence.
    fn assert_gate_main(case: &GateCase, hist: &WorkerHistories, skip_at: Option<usize>) {
        let p = pos(case.sfen);
        let tt = tt_pick(&p, &case.tt);
        let mut prod =
            MovePicker::new_main_search(&p, tt, case.depth, case.ply, SENTINEL_PLANES, false);
        let mut twin =
            TwinMovePicker::new_main_search(&p, tt, case.depth, case.ply, SENTINEL_PLANES, false);

        let mut prod_seq = Vec::new();
        let mut twin_seq = Vec::new();
        let mut n = 0usize;
        loop {
            if Some(n) == skip_at {
                prod.skip_quiet_moves();
                twin.skip_quiet_moves();
            }
            let a = prod.next_move(&p, hist);
            let b = twin.next_move(&p, hist);
            assert_eq!(
                a,
                b,
                "sequence diverged at step {n} for {sfen} (skip_at={skip_at:?})",
                sfen = case.sfen
            );
            match a {
                Some(m) => {
                    prod_seq.push(m);
                    twin_seq.push(m);
                    n += 1;
                }
                None => break,
            }
        }
        assert_eq!(prod_seq, twin_seq);
    }

    /// Drive both qsearch pickers (in check → evasion, else qcapture) to
    /// exhaustion and assert identical sequences.
    fn assert_gate_qsearch(sfen: &str, tt_usi: Option<&str>, hist: &WorkerHistories) {
        let p = pos(sfen);
        let tt = tt_move(&p, tt_usi);
        let mut prod = MovePicker::new_qsearch(&p, tt, SENTINEL_PLANES, false);
        let mut twin = TwinMovePicker::new_qsearch(&p, tt, SENTINEL_PLANES, false);
        let mut n = 0usize;
        loop {
            let a = prod.next_move(&p, hist);
            let b = twin.next_move(&p, hist);
            assert_eq!(a, b, "qsearch diverged at step {n} for {sfen}");
            if a.is_none() {
                break;
            }
            n += 1;
        }
    }

    /// Drive both ProbCut pickers to exhaustion and assert identical sequences.
    fn assert_gate_probcut(sfen: &str, threshold: i32, hist: &WorkerHistories) {
        let p = pos(sfen);
        let mut prod = MovePicker::new_probcut(&p, None, threshold, false);
        let mut twin = TwinMovePicker::new_probcut(&p, None, threshold, false);
        let mut n = 0usize;
        loop {
            let a = prod.next_move(&p, hist);
            let b = twin.next_move(&p, hist);
            assert_eq!(
                a, b,
                "probcut diverged at step {n} for {sfen} (th={threshold})"
            );
            if a.is_none() {
                break;
            }
            n += 1;
        }
    }

    /// The gate fixture set: a spread of not-in-check main positions (with and
    /// without a TT move, capture-rich and quiet-rich), plus in-check evasion
    /// nodes, exercising every picker kind.
    const GATE_MAIN_CASES: &[GateCase] = &[
        // Rook + bishop capture fan, one quiet-heavy tail.
        GateCase {
            sfen: "9/9/9/9/b3R3g/9/4p4/9/K7k b - 1",
            tt: TtPick::None,
            depth: 8,
            ply: 0,
        },
        GateCase {
            sfen: "9/9/9/9/b3R3g/9/4p4/9/K7k b - 1",
            tt: TtPick::Usi("5e5g"),
            depth: 3,
            ply: 2,
        },
        // Dense midgame node: many captures + quiets + a drop.
        GateCase {
            sfen: "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
            tt: TtPick::None,
            depth: 6,
            ply: 3,
        },
        // Same node, with a programmatically-chosen legal quiet as the TT move.
        GateCase {
            sfen: "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
            tt: TtPick::FirstLegalQuiet,
            depth: 1,
            ply: 0,
        },
        // A SEE-losing capture available (gold defended by gold) → bad captures.
        GateCase {
            sfen: "6rkg/8G/9/9/9/9/9/9/K8 b - 1",
            tt: TtPick::None,
            depth: 6,
            ply: 0,
        },
        // In check → evasion picker.
        GateCase {
            sfen: "4r4/5G3/9/9/9/9/9/9/4K3k b - 1",
            tt: TtPick::None,
            depth: 6,
            ply: 0,
        },
        // Quiet-heavy with a quiet checking move (direct-check bonus path).
        GateCase {
            sfen: "4k4/9/R8/9/9/9/9/9/8K b - 1",
            tt: TtPick::None,
            depth: 6,
            ply: 4,
        },
    ];

    #[test]
    #[cfg_attr(miri, ignore)]
    fn gate_main_pickers_match_twin_across_histories_and_skip_schedules() {
        // Distinct history profiles: fresh, and several signed fills that push
        // quiets above / below the `-14000` split and reshape the capture order.
        let profiles = [
            WorkerHistories::new(),
            init_histories(),
            filled_histories(1),
            filled_histories(4),
            filled_histories(-3),
            filled_histories(-9),
        ];
        for case in GATE_MAIN_CASES {
            for hist in &profiles {
                // No skip, plus skips at assorted iteration points fed identically
                // to both pickers.
                for skip_at in [None, Some(0), Some(1), Some(2), Some(4), Some(8)] {
                    assert_gate_main(case, hist, skip_at);
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn gate_qsearch_pickers_match_twin() {
        let cases: &[(&str, Option<&str>)] = &[
            ("9/9/9/9/b3R3g/9/4p4/9/K7k b - 1", None),
            ("9/9/9/9/b3R3g/9/4p4/9/K7k b - 1", Some("5e5g")),
            ("9/9/9/2p1p4/2S1G4/9/9/9/K7k b - 1", None),
            // In check → evasion path in qsearch.
            ("4r4/5G3/9/9/9/9/9/9/4K3k b - 1", None),
            (
                "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
                None,
            ),
        ];
        let profiles = [
            WorkerHistories::new(),
            init_histories(),
            filled_histories(2),
            filled_histories(-5),
        ];
        for &(sfen, tt) in cases {
            for hist in &profiles {
                assert_gate_qsearch(sfen, tt, hist);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn gate_probcut_pickers_match_twin() {
        let sfens = [
            "9/9/9/9/b3R3g/9/4p4/9/K7k b - 1",
            "6rkg/8G/9/9/9/9/9/9/K8 b - 1",
            "9/9/9/2p1p4/2S1G4/9/9/9/K7k b - 1",
        ];
        let profiles = [WorkerHistories::new(), filled_histories(3)];
        for sfen in sfens {
            for hist in &profiles {
                for threshold in [-2000, 0, 1, 500, 5000] {
                    assert_gate_probcut(sfen, threshold, hist);
                }
            }
        }
    }
}
