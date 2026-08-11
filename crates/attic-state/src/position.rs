use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::color::Color;
use crate::hand::Hand;
use crate::move_::Move;
use crate::movegen::try_find_king;
use crate::piece::{Piece, PieceKind};
use crate::square::Square;

/// The seven piece kinds that can be held in hand, in the order the hand-key
/// recomputation walks them. Order is irrelevant to the resulting key (the
/// steps are summed), but fixing it keeps the recomputation deterministic.
const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Maximum look-back (in plies) for repetition detection. Mirrors the
/// reference's `Position::max_repetition_ply` (`position.cpp`, default 16):
/// searching further back for a repetition is rarely productive and measurably
/// slows the engine, so the walk is capped here regardless of `plies_from_null`.
const MAX_REPETITION_PLY: i32 = 16;

/// Classification of a repeated position, ported from upstream YaneuraOu's
/// `RepetitionState` (`source/types.h` ~line 1107).
///
/// # WIN / LOSE viewpoint convention
///
/// `Win` and `Lose` describe the **perpetual-check** outcome **from the
/// viewpoint of the side to move** in the position `is_repetition` is asked
/// about — *not* from the viewpoint of whoever delivered the checks:
///
/// - [`RepetitionState::Lose`] — the **side to move** is the one that has been
///   delivering check on every one of its moves across the repeated cycle.
///   Perpetual check is prohibited for the checking side, so the side to move
///   **loses**. (Reference: `i <= continuousCheck[sideToMove]`.)
/// - [`RepetitionState::Win`] — the **opponent** of the side to move has been
///   delivering the perpetual check, so the opponent loses and the side to
///   move **wins**. (Reference: `i <= continuousCheck[~sideToMove]`.)
///
/// Getting this backwards flips the sign of the score in search, so the mapping
/// above is the load-bearing contract for any downstream consumer.
///
/// The remaining variants are check-independent:
///
/// - [`RepetitionState::Draw`] — an ordinary (non-perpetual-check) repetition
///   of the same board *and* hands with the same side to move.
/// - [`RepetitionState::Superior`] — same board, but the side to move holds an
///   equal-or-greater hand in every piece kind (and strictly greater in at
///   least one) than at the earlier occurrence: a superior position.
/// - [`RepetitionState::Inferior`] — same board, but the side to move's hand is
///   the equal-or-lesser one: an inferior position.
/// - [`RepetitionState::None`] — no qualifying repetition was found within the
///   lookback bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepetitionState {
    /// Not a repetition.
    None,
    /// Perpetual-check repetition won by the side to move (opponent checked).
    Win,
    /// Perpetual-check repetition lost by the side to move (side to move checked).
    Lose,
    /// Ordinary (non-perpetual-check) fourfold-style repetition.
    Draw,
    /// Same board, side to move holds the strictly superior hand.
    Superior,
    /// Same board, side to move holds the strictly inferior hand.
    Inferior,
}

/// `true` iff `superior` holds an equal-or-greater count of **every** hand
/// piece kind than `inferior`. Mirrors the reference's
/// `hand_is_equal_or_superior` (`types.h` ~line 1039), used to split a
/// board-equal / hand-unequal repetition into `Superior` vs `Inferior`.
fn hand_is_equal_or_superior(superior: &Hand, inferior: &Hand) -> bool {
    HAND_KINDS
        .iter()
        .all(|&k| superior.count(k) >= inferior.count(k))
}

/// The piece sitting on `from` *before* a board move `m`, given the piece that
/// will occupy `to` after it (`piece_after`, read off the move encoding). For a
/// promotion the pre-move piece is the unpromoted form; otherwise it is
/// `piece_after` unchanged. Drops have no `from` square and never reach here.
fn piece_before_move(m: Move, piece_after: Piece) -> Piece {
    if m.is_promote() {
        Piece::new(piece_after.kind, piece_after.color)
    } else {
        piece_after
    }
}

/// A post-move `gives_check` probe: read off the just-updated board,
/// is `side_to_move`'s king attacked by the opposite side? Kept solely as
/// the debug-only equivalence oracle for the flag passed into
/// [`Position::do_move_with_check`] (the reference `ASSERT_LV3`,
/// `position.cpp`). If the side-to-move's king is absent (a pseudo-legal
/// probe move captured it), the probe is trivially `false`, matching the
/// predicate.
#[cfg(debug_assertions)]
fn post_move_gives_check(board: &Board, side_to_move: Color) -> bool {
    match try_find_king(board, side_to_move) {
        Some(king_sq) => crate::movegen::is_attacked_by(board, king_sq, side_to_move.flip()),
        None => false,
    }
}

/// Snapshot pushed onto `Position::history` after every `do_move`.
///
/// This record stores **no board copy** — mirroring the reference `StateInfo`,
/// which keeps only small per-state scalars and reconstructs the board by
/// reverse-applying the move in `undo_move`. The repetition equality key is
/// `(board_key, hands, side_to_move)`: `board_key` is the board half of the
/// Zobrist key (side-to-move term included), used by `is_repetition` /
/// `position_occurrences` as the board identity, and `hands` is compared for
/// the exact `Draw` vs `Superior`/`Inferior` split — the reference accepts the
/// hash-collision risk of a board_key match (position.cpp `is_repetition`,
/// ENABLE_QUICK_DRAW branch), and so do we. `gives_check` is the per-step
/// check-flag used by `history_since_last_distinct` consumers (perpetual-check
/// filter): it records whether the side that just moved put the now-side-to-move
/// in check.
///
/// The remaining fields port the reference's incrementally-maintained
/// repetition state (`source/position.cpp`, computed in
/// `do_move` ~2111-2211, reset in `do_null_move` / `set`):
///
/// - `plies_from_null` — plies since the last null move (or the setup root);
///   `set`/root = 0, `do_move` = previous + 1, `do_null_move` = 0. Bounds the
///   repetition look-back to `min(16, plies_from_null)`, so detection never
///   crosses a null move (`st->pliesFromNull`).
/// - `continuous_check[color]` — the run (counted in twos) of consecutive
///   checking moves by `color` ending at this state; `+= 2` on a checking move
///   by `color`, reset to `0` otherwise; the other colour is inherited from the
///   previous state. `do_null_move` zeroes the null-mover's entry
///   (`st->continuousCheck`).
///
/// The last three fields exist only in the **non**-`quick-draw` configuration.
/// Upstream compiles the whole precompute out under `ENABLE_QUICK_DRAW`
/// (`position.h`, `position.cpp`), because the QUICK_DRAW
/// `is_repetition` walks the chain at query time and never reads them:
///
/// - `repetition` — signed ply distance to the previous occurrence of this
///   position (0 if none); negative in the forced-fourfold case
///   (`repetition_times >= 3`). `is_repetition` reads it directly
///   (`st->repetition`).
/// - `repetition_times` — how many earlier occurrences this position chains to
///   (`st->repetition_times`); the fourfold sign flips once it reaches 3.
/// - `repetition_type` — the classification (`st->repetition_type`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StateInfo {
    pub hands: [Hand; Color::COUNT],
    pub side_to_move: Color,
    pub gives_check: bool,
    pub board_key: u64,
    pub plies_from_null: i32,
    pub continuous_check: [u16; Color::COUNT],
    #[cfg(not(feature = "quick-draw"))]
    pub repetition: i32,
    #[cfg(not(feature = "quick-draw"))]
    pub repetition_times: i32,
    #[cfg(not(feature = "quick-draw"))]
    pub repetition_type: RepetitionState,
}

#[derive(Debug, Clone)]
pub struct Position {
    board: Board,
    hands: [Hand; Color::COUNT],
    side_to_move: Color,
    ply: u16,
    history: Vec<StateInfo>,
    /// Snapshot of the position *before the first move of the current line*
    /// (the SFEN / setup root). `history` records only post-move states, so the
    /// root is not otherwise recoverable — yet `is_repetition` must be able to
    /// look back to it (the reference walks past the root up to `ply` plies; a
    /// four-ply cycle recurring straight onto the root is detected there).
    /// Captured lazily on the first `do_move` of an empty history, so a root
    /// re-established after a full unwind (or a direct board mutation) is
    /// re-snapshotted on the next `do_move`.
    root: Option<StateInfo>,
    /// Zobrist board hash: XOR of per-`(piece, square)` terms plus the
    /// side-to-move term. See [`crate::key`] for the scheme.
    board_key: u64,
    /// Zobrist hand hash: sum of per-`(color, kind)` steps, one per held copy.
    hand_key: u64,
    /// Partial position key over **board pawns only** (piece type PAWN; a
    /// promoted pawn is not a pawn here). Seeded with the non-zero
    /// [`crate::key::no_pawns`] value, then XORed by `psq[pawn][sq]` per board
    /// pawn. Consumed by the search's pawn-structure history / correction
    /// histories (mirrors the reference `StateInfo::pawnKey`). Pieces in hand
    /// never contribute.
    pawn_key: u64,
    /// Partial position key over board **minor pieces** (lance, knight, silver,
    /// gold and their promotions — see [`crate::key::is_minor_piece`]). Empty
    /// value zero. Mirrors the reference `StateInfo::minorPieceKey`.
    minor_piece_key: u64,
    /// Partial position keys over **all non-pawn board pieces** of each colour,
    /// **king included** (a board pawn is the sole exclusion). Indexed by
    /// [`Color::index`]. Empty value zero. Mirrors `StateInfo::nonPawnKey`.
    non_pawn_key: [u64; Color::COUNT],
    // NOTE: the reference also maintains an additive `materialKey`
    // (`position.cpp`), but no consumer exists in
    // `yaneuraou-search.cpp` / `movepick.cpp`, so it is deliberately not ported.
    /// Eagerly-computed per-state check info of the *current* position (reference
    /// `StateInfo::{checkSquares, blockersForKing}`), read by the `gives_check`
    /// / `gives_direct_check` / `is_legal` predicates in
    /// [`crate::search_movegen`]. Recomputed on every state change — `do_move` /
    /// `do_null_move` save the parent value on [`Self::check_info_stack`] and
    /// compute the child's via `compute_check_info`; `undo_move` /
    /// `undo_null_move` pop the parent back (O(1), no recompute); `refresh_keys`
    /// recomputes it. A plain per-state field (no `RefCell`, no `Option`),
    /// mirroring the reference's eager `set_check_info` stored in `StateInfo`.
    /// Excluded from [`PartialEq`] as a derived cache.
    check_info: crate::search_movegen::CheckInfo,
    /// The stack of parent [`Self::check_info`] values, one entry per live
    /// `do_move` / `do_null_move`, so `undo_move` / `undo_null_move` can restore
    /// the parent state's check info in O(1) rather than recomputing it. Mirrors
    /// `history`'s depth. Excluded from [`PartialEq`] as auxiliary cache state.
    check_info_stack: Vec<crate::search_movegen::CheckInfo>,
}

/// Structural equality over the *primary* state (`board`, `hands`,
/// `side_to_move`, `ply`, `history`). `board_key` / `hand_key` are a derived
/// cache of `(board, hands, side_to_move)` — recomputable at any time — and are
/// deliberately excluded so that a position built by the direct setters (which
/// leave the cache stale until `refresh_keys`) still compares equal to the same
/// state built through `do_move` or SFEN parsing. `root` is likewise excluded:
/// it is auxiliary lookback state for `is_repetition`, captured lazily on the
/// first `do_move`, so a position that has taken a move-then-undo round trip
/// (root `Some`) must still compare equal to the untouched snapshot it was
/// cloned from (root `None`). Key correctness is pinned by the dedicated
/// Zobrist gate, not by this predicate.
impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.board == other.board
            && self.hands == other.hands
            && self.side_to_move == other.side_to_move
            && self.ply == other.ply
            && self.history == other.history
    }
}

impl Eq for Position {}

impl Position {
    pub const fn empty() -> Self {
        // An empty board with Black (the side whose term is *not* XORed in) to
        // move and no hand pieces hashes to zero on both halves. The partial
        // keys carry their empty-board values: `pawn_key` seeds with the
        // non-zero `noPawns` constant, the others with zero.
        Self {
            board: Board::empty(),
            hands: [Hand::empty(), Hand::empty()],
            side_to_move: Color::Black,
            ply: 1,
            history: Vec::new(),
            root: None,
            board_key: 0,
            hand_key: 0,
            pawn_key: crate::key::NO_PAWNS_SEED,
            minor_piece_key: 0,
            non_pawn_key: [0; Color::COUNT],
            // The empty board's check info (no kings, no attacks); recomputed
            // eagerly on the first state change (SFEN parse / `do_move`).
            check_info: crate::search_movegen::CheckInfo::EMPTY,
            check_info_stack: Vec::new(),
        }
    }

    pub fn startpos() -> Self {
        crate::sfen::parse_sfen(crate::sfen::STARTPOS_SFEN)
            .expect("STARTPOS_SFEN is a hard-coded valid sfen")
    }

    pub const fn board(&self) -> &Board {
        &self.board
    }

    /// Direct mutable access to the board, through a [`BoardMut`] guard.
    ///
    /// **Warning:** this bypasses the incremental Zobrist maintenance that
    /// `do_move` / `undo_move` perform, so any mutation through this reference
    /// leaves the cached `board_key` (and therefore `key()`) stale. Call
    /// `refresh_keys()` afterwards to recompute the cache — as `sfen.rs` does
    /// after parsing a board.
    ///
    /// The returned guard derefs to the [`Board`] and, on drop, recomputes the
    /// per-state [`CheckInfo`] eagerly (mirroring the reference's
    /// `set_check_info` after a board edit) so a hand-built position's check
    /// info stays fresh without a `refresh_keys` round trip. This is a cold path
    /// — the search's `do_move` mutates `self.board` directly, never through
    /// `board_mut`.
    pub fn board_mut(&mut self) -> BoardMut<'_> {
        BoardMut { pos: self }
    }

    pub const fn hand(&self, color: Color) -> &Hand {
        &self.hands[color.index()]
    }

    /// Direct mutable access to a color's hand.
    ///
    /// **Warning:** this bypasses the incremental Zobrist maintenance that
    /// `do_move` / `undo_move` perform, so any mutation through this reference
    /// leaves the cached `hand_key` (and therefore `key()`) stale. Call
    /// `refresh_keys()` afterwards to recompute the cache — as `sfen.rs` does
    /// after parsing the hands.
    pub fn hand_mut(&mut self, color: Color) -> &mut Hand {
        &mut self.hands[color.index()]
    }

    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// Locate `color`'s king in O(1) by reading the incrementally-maintained
    /// KING piece set (via [`try_find_king`]), returning `None` when that side
    /// has no king on the board. Mirrors the reference's `king_square(c)` read
    /// of `kingSquare[c]`; callers that require the king (the evaluator) unwrap.
    pub fn king_square(&self, color: Color) -> Option<Square> {
        try_find_king(&self.board, color)
    }

    /// Set the side to move directly.
    ///
    /// **Warning:** the side-to-move term is part of `board_key`, so this
    /// bypasses the incremental Zobrist maintenance that `do_move` /
    /// `undo_move` perform and leaves the cached `board_key` (and therefore
    /// `key()`) stale. Call `refresh_keys()` afterwards to recompute the cache
    /// — as `sfen.rs` does after parsing the side field.
    ///
    /// The side to move selects which king is "own" vs "enemy" in the per-state
    /// [`CheckInfo`], so this recomputes it eagerly (mirroring the reference's
    /// `set_check_info`) — keeping a hand-built position's check info fresh
    /// without a `refresh_keys` round trip.
    pub fn set_side_to_move(&mut self, color: Color) {
        self.side_to_move = color;
        self.check_info = self.compute_check_info();
    }

    pub const fn ply(&self) -> u16 {
        self.ply
    }

    /// Set the ply counter directly.
    ///
    /// Unlike the other direct mutators, `ply` is intentionally not part of the
    /// Zobrist key, so this cannot leave `board_key` / `hand_key` stale — no
    /// `refresh_keys()` call is needed.
    pub const fn set_ply(&mut self, ply: u16) {
        self.ply = ply;
    }

    /// `st->pliesFromNull` of the current (top-of-stack) state — plies since the
    /// last null move (or the setup root); `0` before any move. Mirrors the
    /// reference `Position::state()->pliesFromNull`, consumed by the search's
    /// `is_shuffling` singular-extension guard.
    pub fn plies_from_null(&self) -> i32 {
        self.current_plies_from_null()
    }

    /// Full 64-bit Zobrist key: `board_key ^ hand_key`. A pure function of
    /// `(board, hands, side_to_move)` — two positions with the same triple hash
    /// equal, and the incremental maintenance in `do_move` / `undo_move` keeps
    /// this in agreement with a from-scratch recomputation.
    pub const fn key(&self) -> u64 {
        self.board_key ^ self.hand_key
    }

    /// Board half of the key: XOR of per-`(piece, square)` terms plus the
    /// side-to-move term. Mirrors the reference's `board_key`.
    pub const fn board_key(&self) -> u64 {
        self.board_key
    }

    /// The per-state [`crate::search_movegen::CheckInfo`] of the current
    /// position, recomputed eagerly on every state change and stored as a plain
    /// field. The `search_movegen` predicates read it through this accessor — a
    /// zero-ceremony `&CheckInfo` field borrow.
    pub(crate) fn check_info(&self) -> &crate::search_movegen::CheckInfo {
        debug_assert_eq!(
            self.check_info.board_key,
            self.board_key(),
            "cached check_info is stale relative to the current board_key",
        );
        &self.check_info
    }

    /// Hand half of the key: sum of per-`(color, kind)` steps. Mirrors the
    /// reference's `hand_key`.
    pub const fn hand_key(&self) -> u64 {
        self.hand_key
    }

    /// Partial key over board pawns only (see the field docs). Mirrors
    /// `Position::pawn_key()`.
    pub const fn pawn_key(&self) -> u64 {
        self.pawn_key
    }

    /// Partial key over board minor pieces (see the field docs). Mirrors
    /// `Position::minor_piece_key()`.
    pub const fn minor_piece_key(&self) -> u64 {
        self.minor_piece_key
    }

    /// Partial key over `color`'s board non-pawn pieces, king included (see the
    /// field docs). Mirrors `Position::non_pawn_key(Color)`.
    pub const fn non_pawn_key(&self, color: Color) -> u64 {
        self.non_pawn_key[color.index()]
    }

    /// XOR `piece`'s `psq` term into the partial keys it belongs to. Every board
    /// piece is in exactly one of `pawn_key` / `non_pawn_key[color]`; a minor
    /// piece additionally toggles `minor_piece_key`. XOR is self-inverse, so the
    /// same call both places and removes a piece from the partial keys — the
    /// reference's `xor_piece_for_partial_key` (`position.cpp`).
    fn xor_piece_partial(&mut self, piece: Piece, sq: Square) {
        let term = crate::key::psq(piece, sq);
        if piece.kind == PieceKind::Pawn && !piece.promoted {
            self.pawn_key ^= term;
        } else {
            if crate::key::is_minor_piece(piece) {
                self.minor_piece_key ^= term;
            }
            self.non_pawn_key[piece.color.index()] ^= term;
        }
    }

    /// Recompute `(board_key, hand_key)` from scratch over the current board,
    /// hands and side-to-move. Used to (re)seed the keys after direct mutation
    /// (SFEN parsing) and by tests to validate the incremental path.
    fn recomputed_keys(&self) -> (u64, u64) {
        let mut board_key = 0u64;
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if let Some(piece) = self.board.get(sq) {
                board_key ^= crate::key::psq(piece, sq);
            }
        }
        if self.side_to_move == Color::White {
            board_key ^= crate::key::side();
        }

        let mut hand_key = 0u64;
        for color in [Color::Black, Color::White] {
            for kind in HAND_KINDS {
                let n = self.hands[color.index()].count(kind) as u64;
                hand_key =
                    hand_key.wrapping_add(crate::key::hand_step(color, kind).wrapping_mul(n));
            }
        }
        (board_key, hand_key)
    }

    /// Recompute the three partial keys (`pawn_key`, `minor_piece_key`,
    /// `non_pawn_key`) from scratch over the current board. Mirrors the
    /// per-piece walk in `Position::set()` (`position.cpp`): `pawn_key`
    /// seeds with `noPawns`, the rest with zero, and each board piece is folded
    /// in via the same classification as the incremental path.
    fn recomputed_partial_keys(&self) -> (u64, u64, [u64; Color::COUNT]) {
        let mut scratch = Position::empty();
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if let Some(piece) = self.board.get(sq) {
                scratch.xor_piece_partial(piece, sq);
            }
        }
        (
            scratch.pawn_key,
            scratch.minor_piece_key,
            scratch.non_pawn_key,
        )
    }

    /// Reseed the stored keys from a from-scratch recomputation. Called after
    /// SFEN parsing, whose direct board / hand / side mutations bypass the
    /// incremental path.
    pub(crate) fn refresh_keys(&mut self) {
        let (board_key, hand_key) = self.recomputed_keys();
        self.board_key = board_key;
        self.hand_key = hand_key;
        let (pawn_key, minor_piece_key, non_pawn_key) = self.recomputed_partial_keys();
        self.pawn_key = pawn_key;
        self.minor_piece_key = minor_piece_key;
        self.non_pawn_key = non_pawn_key;
        // The direct board / hand / side mutations that precede `refresh_keys`
        // (SFEN parsing) invalidate the per-state check info; recompute it
        // eagerly once the keys are current (the child position's board is
        // final at this point).
        self.check_info = self.compute_check_info();
    }

    /// Play `m`, deciding check status from the parent's cached check info
    /// *before* touching the board — the reference convenience overload
    /// `do_move(m, newSt)` = `do_move(m, newSt, gives_check(m))` (`position.h`).
    ///
    /// Every cold caller (root, book, protocol driver, movegen scratch legality
    /// loops, mate helpers, tests) uses this wrapper; the two hot search sites
    /// call [`Self::do_move_with_check`] directly with the flag they already
    /// computed for their own pruning.
    pub fn do_move(&mut self, m: Move) -> Undo {
        let gc = self.gives_check(m);
        self.do_move_with_check(m, gc)
    }

    /// Play `m` given the pre-computed `gives_check` predicate (the reference's
    /// `do_move(m, newSt, givesCheck)`, `position.h`). `gives_check` must
    /// equal `self.gives_check(m)` evaluated from the pre-move position; the
    /// debug-only oracle below re-derives it with a post-move full attack probe
    /// and asserts equality (mirroring the reference `ASSERT_LV3`,
    /// `position.cpp`).
    pub fn do_move_with_check(&mut self, m: Move, gives_check: bool) -> Undo {
        // Snapshot the pre-move state as the lookback root the first time a move
        // is played from an empty history. Re-captured on every fresh start (an
        // empty history after a full unwind, or after direct board mutation) so
        // the root always reflects the actual head of the current line.
        if self.history.is_empty() {
            self.root = Some(self.root_state());
        }

        // The state this move is played from — the look-back "previous" whose
        // `plies_from_null` / `continuous_check` the new state extends. It is the
        // top of the history stack, or the freshly-captured root when empty.
        let prev = self
            .history
            .last()
            .cloned()
            .unwrap_or_else(|| self.root.as_ref().expect("root captured above").clone());

        let mover = self.side_to_move;
        let to = m.to_sq();

        // Debug oracle gate: whether the side about to be checked (the mover's
        // opponent) was *already* in check in this pre-move position. Captured
        // before any mutation. When it holds, the pre-existing checker is a
        // stepper the differential `checkersBB` reconstruction neither can nor
        // needs to reproduce (the reference `ASSERT_LV3` fires identically
        // there), so the strict-equality half of the oracle is skipped and only
        // the subset direction is asserted. Impossible in legal play — a side
        // in check cannot have just been given check by its own move — so this
        // is exercised only by contrived fixtures / movegen scratch loops.
        #[cfg(debug_assertions)]
        let enemy_already_in_check = post_move_gives_check(&self.board, mover.flip());

        let undo = if m.is_drop() {
            let kind = m.dropped_piece_kind();
            let piece = Piece::new(kind, mover);
            self.board.set(to, Some(piece));
            self.hands[mover.index()].decrement(kind);
            // Key: piece appears at `to` (XOR in), one copy leaves the hand
            // (subtract its step).
            self.board_key ^= crate::key::psq(piece, to);
            self.hand_key = self
                .hand_key
                .wrapping_sub(crate::key::hand_step(mover, kind));
            // Partial keys: the dropped piece enters the board (the copy that
            // left the hand contributed to no partial key). A pawn drop hence
            // enters `pawn_key`.
            self.xor_piece_partial(piece, to);
            Undo { captured: None }
        } else {
            let from = m.from_sq();
            let captured = self.board.get(to);
            let piece_after = m.moved_piece_after();
            let piece_before = piece_before_move(m, piece_after);
            self.board.set(to, Some(piece_after));
            self.board.set(from, None);
            // Key: remove any captured piece from `to` (and add a copy of its
            // kind to the mover's hand), then move the piece off `from` and
            // onto `to` in its post-move (possibly promoted) form.
            if let Some(cap) = captured {
                self.hands[mover.index()].increment(cap.kind);
                self.board_key ^= crate::key::psq(cap, to);
                self.hand_key = self
                    .hand_key
                    .wrapping_add(crate::key::hand_step(mover, cap.kind));
                // Partial keys: the captured piece leaves the board (its move
                // into the hand touches no partial key).
                self.xor_piece_partial(cap, to);
            }
            self.board_key ^= crate::key::psq(piece_before, from);
            self.board_key ^= crate::key::psq(piece_after, to);
            // Partial keys: the mover leaves `from` in its pre-move form and
            // enters `to` in its post-move (possibly promoted) form. A promoting
            // pawn thus leaves `pawn_key` and enters `non_pawn_key` (+minor).
            self.xor_piece_partial(piece_before, from);
            self.xor_piece_partial(piece_after, to);
            Undo { captured }
        };

        self.side_to_move = mover.flip();
        self.ply = self.ply.wrapping_add(1);
        // Side-to-move flipped; toggle the side term in `board_key`.
        self.board_key ^= crate::key::side();

        // `gives_check` was decided from the *pre-move* position by the caller
        // (the table-lookup predicate `gives_check(m)`), so no post-move full
        // attack probe runs here. The debug-only oracle below mirrors the
        // reference `ASSERT_LV3` (`position.cpp`), which lives *inside* the
        // `if (givesCheck)` branch: when a check is claimed, the just-updated
        // board must actually show the now-side-to-move's king under attack.
        // The reference asserts nothing in the `!givesCheck` branch, and neither
        // do we — a bidirectional check would wrongly fire on the (illegal, but
        // used by movegen scratch loops and contrived test fixtures) positions
        // where the side-not-to-move was *already* in check by some other piece
        // before the move: the predicate correctly reports "this move gives no
        // new check" while the post-move full probe still sees the pre-existing
        // checker. `gives_check ⟹ opponent-king-attacked` holds universally
        // (a real check always leaves the opponent's king attacked), so this
        // direction is safe on every position the suite reaches, including the
        // enemy-king-capture probe moves (there `gives_check` is `false`).
        #[cfg(debug_assertions)]
        if gives_check {
            debug_assert!(
                post_move_gives_check(&self.board, self.side_to_move),
                "gives_check flag was set for move {m:?} but the post-move attack \
                 probe finds no checker on the now-side-to-move's king",
            );
        }

        // `continuousCheck`: inherit the previous state's runs, then extend the
        // *mover's* run by 2 on a checking move (reset to 0 otherwise). The
        // non-mover's run is inherited unchanged (`position.cpp`).
        let mut continuous_check = prev.continuous_check;
        continuous_check[mover.index()] = if gives_check {
            prev.continuous_check[mover.index()] + 2
        } else {
            0
        };

        // Save the parent state's check info so the matching `undo_move`
        // restores it in O(1), then compute the child state's info eagerly
        // (reference `set_check_info` inside `do_move`). The board / side /
        // `board_key` are already final, so the tables are built against the
        // resulting position; the child's `in_check` is taken directly from
        // `gives_check` (a move gives check iff the resulting side to move is in
        // check), avoiding a second full attack probe.
        // Build the child `checkersBB` differentially from the parent info + the
        // move (reference `do_move_impl`, `position.cpp`)
        // instead of a full reverse-attack probe, which the check-giving path
        // would otherwise run on every checking transition. `self.check_info` is
        // still the parent value here (pushed just below); the board / side are
        // already final. Empty when the move gives no check.
        let child_checkers = if gives_check {
            self.differential_child_checkers(m, mover, &self.check_info)
        } else {
            Bitboard::EMPTY
        };

        // Debug-only oracle for the differential set: the built checkers must be
        // a subset of the full reverse-attack
        // probe on the now-side-to-move's king (every constructed checker is a
        // real attacker), and — unless the checked side was already in check
        // before this move — must equal it. The surplus in the already-in-check
        // case is pre-existing steppers neither this differential nor the
        // reference's `ASSERT_LV3` can reproduce.
        #[cfg(debug_assertions)]
        if gives_check {
            let oks = try_find_king(&self.board, self.side_to_move)
                .expect("gives_check implies the now-side-to-move has a king");
            let probed = crate::movegen::attackers_bb(&self.board, oks, mover);
            debug_assert!(
                (child_checkers & !probed).is_empty(),
                "differential checkers {child_checkers:?} are not a subset of the \
                 probed attackers {probed:?} for move {m:?}",
            );
            if !enemy_already_in_check {
                debug_assert_eq!(
                    child_checkers, probed,
                    "differential checkers disagree with the full probe for move {m:?}",
                );
            }
        }

        self.check_info_stack.push(self.check_info);
        self.check_info =
            self.compute_check_info_with_in_check_and_checkers(gives_check, child_checkers);

        self.history.push(StateInfo {
            hands: self.hands,
            side_to_move: self.side_to_move,
            gives_check,
            board_key: self.board_key,
            // `++st->pliesFromNull` (`position.cpp`).
            plies_from_null: prev.plies_from_null + 1,
            continuous_check,
            // Filled in by `store_current_repetition` below.
            #[cfg(not(feature = "quick-draw"))]
            repetition: 0,
            #[cfg(not(feature = "quick-draw"))]
            repetition_times: 0,
            #[cfg(not(feature = "quick-draw"))]
            repetition_type: RepetitionState::None,
        });

        // Now that the new state is on the stack, compute its repetition info by
        // walking the chain (`position.cpp`) and write it back. Under
        // `quick-draw` the reference `#if`s this precompute out entirely
        // (`position.cpp`) — the walk happens at query time instead — so the
        // per-move cost disappears with it.
        #[cfg(not(feature = "quick-draw"))]
        self.store_current_repetition();

        undo
    }

    /// The setup-root `StateInfo`: the `set()`-equivalent initial state, with
    /// zeroed `plies_from_null` / `continuous_check` / repetition fields
    /// (`position.cpp` `set` / `set_state`).
    fn root_state(&self) -> StateInfo {
        StateInfo {
            hands: self.hands,
            side_to_move: self.side_to_move,
            gives_check: false,
            board_key: self.board_key,
            plies_from_null: 0,
            continuous_check: [0; Color::COUNT],
            #[cfg(not(feature = "quick-draw"))]
            repetition: 0,
            #[cfg(not(feature = "quick-draw"))]
            repetition_times: 0,
            #[cfg(not(feature = "quick-draw"))]
            repetition_type: RepetitionState::None,
        }
    }

    /// Compute the repetition info for the current (top-of-stack) state via
    /// [`Self::compute_repetition`] and store it into that `StateInfo`. Invoked
    /// by `do_move` after the push (and, in tests, by hand-assembled histories).
    #[cfg(not(feature = "quick-draw"))]
    fn store_current_repetition(&mut self) {
        let (repetition, repetition_times, repetition_type) = self.compute_repetition();
        if let Some(top) = self.history.last_mut() {
            top.repetition = repetition;
            top.repetition_times = repetition_times;
            top.repetition_type = repetition_type;
        }
    }

    pub fn undo_move(&mut self, m: Move, undo: Undo) {
        // Pop history first so the round-trip invariant
        // `do_move(m); undo_move(m, undo) ⇒ identical Position` holds
        // element-by-element (history pop cancels the do_move push).
        let _ = self.history.pop();
        // Restore the parent state's check info in O(1) (the value `do_move`
        // saved), avoiding a recompute. An empty stack (an `undo_move` with no
        // matching `do_move`) falls back to a fresh compute.
        self.check_info = match self.check_info_stack.pop() {
            Some(ci) => ci,
            None => self.compute_check_info(),
        };
        self.side_to_move = self.side_to_move.flip();
        self.ply = self.ply.wrapping_sub(1);
        // Side-to-move flipped back; toggle the side term to match `do_move`.
        self.board_key ^= crate::key::side();
        let mover = self.side_to_move;
        let to = m.to_sq();

        if m.is_drop() {
            let kind = m.dropped_piece_kind();
            let piece = Piece::new(kind, mover);
            self.board.set(to, None);
            self.hands[mover.index()].increment(kind);
            // Reverse the drop: remove the piece from `to`, return its step to
            // the hand.
            self.board_key ^= crate::key::psq(piece, to);
            self.hand_key = self
                .hand_key
                .wrapping_add(crate::key::hand_step(mover, kind));
            // Reverse the partial-key drop (XOR is self-inverse).
            self.xor_piece_partial(piece, to);
            return;
        }

        let from = m.from_sq();
        let after = m.moved_piece_after();
        let piece_before = piece_before_move(m, after);
        self.board.set(from, Some(piece_before));
        self.board.set(to, undo.captured);
        // Reverse the board move: lift the moved piece off `to`, restore it at
        // `from`, then put any captured piece back at `to` (and remove the copy
        // that `do_move` added to the mover's hand).
        self.board_key ^= crate::key::psq(after, to);
        self.board_key ^= crate::key::psq(piece_before, from);
        // Reverse the partial-key board move (XOR is self-inverse): the mover
        // leaves `to` in its post-move form and re-enters `from` in its
        // pre-move form.
        self.xor_piece_partial(after, to);
        self.xor_piece_partial(piece_before, from);
        if let Some(cap) = undo.captured {
            self.hands[mover.index()].decrement(cap.kind);
            self.board_key ^= crate::key::psq(cap, to);
            self.hand_key = self
                .hand_key
                .wrapping_sub(crate::key::hand_step(mover, cap.kind));
            // Restore the captured piece to the board in the partial keys.
            self.xor_piece_partial(cap, to);
        }
    }

    /// Play a null move: pass the turn without touching the board or hands.
    ///
    /// Mirrors the reference `Position::do_null_move` (`position.cpp`), used by
    /// the main search's Step 9 null-move pruning. Only the side-to-move (and its
    /// Zobrist term), the ply counter, and the game history are updated; the
    /// board, hands and partial keys are unchanged. The pushed `StateInfo` records
    /// `gives_check = false` — a null move can never give check, and Step 9 is
    /// only reached when the side to move is *not* already in check.
    ///
    /// The reference resets `pliesFromNull` to 0 here (`position.cpp`) so
    /// that repetition detection never looks back across the null move — the
    /// look-back bound `min(16, plies_from_null)` collapses to 0 immediately
    /// after a null and rebuilds one ply at a time. It also zeroes the
    /// null-mover's `continuousCheck` (`position.cpp`) and clears the
    /// repetition fields (`repetition`/`repetition_times`, 2503-2506). Undo with
    /// [`Self::undo_null_move`].
    pub fn do_null_move(&mut self) {
        if self.history.is_empty() {
            self.root = Some(self.root_state());
        }

        // The null move's "previous" state — the top of the stack, or the root.
        let prev = self
            .history
            .last()
            .cloned()
            .unwrap_or_else(|| self.root.as_ref().expect("root captured above").clone());

        let null_mover = self.side_to_move;
        self.side_to_move = self.side_to_move.flip();
        self.ply = self.ply.wrapping_add(1);
        self.board_key ^= crate::key::side();

        // Inherit the previous `continuousCheck`, then zero the null-mover's run:
        // a null move is never a check, so any streak by the side that just
        // passed is broken (`st->continuousCheck[~sideToMove] = 0`). The new
        // side to move's run is 0 by the reference invariant (you cannot null
        // out of check), and is inherited unchanged here.
        let mut continuous_check = prev.continuous_check;
        continuous_check[null_mover.index()] = 0;

        // The board is unchanged, but passing the turn swaps which king is the
        // enemy, so the check info differs; save the parent value for O(1) undo
        // and compute the new side's info eagerly. A null move never gives check
        // and is only tried when the side to move is not already in check, so the
        // child's `in_check` is `false` (injected, not probed).
        self.check_info_stack.push(self.check_info);
        self.check_info = self.compute_check_info_with_in_check(false);

        self.history.push(StateInfo {
            hands: self.hands,
            side_to_move: self.side_to_move,
            gives_check: false,
            board_key: self.board_key,
            // `st->pliesFromNull = 0` (`position.cpp`).
            plies_from_null: 0,
            continuous_check,
            // `st->repetition = st->repetition_times = 0` (2503-2506); a state
            // reached directly by a null move is never itself a repetition.
            #[cfg(not(feature = "quick-draw"))]
            repetition: 0,
            #[cfg(not(feature = "quick-draw"))]
            repetition_times: 0,
            #[cfg(not(feature = "quick-draw"))]
            repetition_type: RepetitionState::None,
        });
    }

    /// Undo the [`Self::do_null_move`] that reached the current position.
    pub fn undo_null_move(&mut self) {
        let _ = self.history.pop();
        // Restore the pre-null check info in O(1).
        self.check_info = match self.check_info_stack.pop() {
            Some(ci) => ci,
            None => self.compute_check_info(),
        };
        self.side_to_move = self.side_to_move.flip();
        self.ply = self.ply.wrapping_sub(1);
        self.board_key ^= crate::key::side();
    }

    /// True iff `s`'s `(board_key, hands, side_to_move)` matches the current
    /// position's. The single equality predicate shared by both repetition
    /// queries below — kept in one place so the two surfaces can never drift.
    ///
    /// A per-state record carries no board copy: board identity is carried by
    /// `board_key` (the side-to-move term is folded into it, so the explicit
    /// `side_to_move` check is belt-and-suspenders). This inherits the
    /// reference's accepted hash-collision risk: two distinct boards that hash
    /// to the same `board_key` and hold equal hands compare equal here.
    fn matches_current_state(&self, s: &StateInfo) -> bool {
        s.board_key == self.board_key
            && s.hands == self.hands
            && s.side_to_move == self.side_to_move
    }

    /// Number of `history` entries whose `(board, hands, side_to_move)`
    /// matches the current position. After a `do_move`, the pushed entry
    /// itself counts — so a freshly-reached-once state returns 1, a 4-fold
    /// completion returns 4.
    pub fn position_occurrences(&self) -> usize {
        self.history
            .iter()
            .filter(|s| self.matches_current_state(s))
            .count()
    }

    /// Yield the history entries between (exclusive) the most-recent prior
    /// occurrence of the current position in `history[..len-1]` and `history`'s
    /// end (inclusive). Iteration order is **oldest-to-newest** — i.e., the
    /// natural slice order, so consumers can pair entries with the moves that
    /// produced them by index. If no prior occurrence exists, the iterator is
    /// empty.
    ///
    /// `#[cfg(test)]`: no production path consumes it (movegen is
    /// repetition-blind, so there is no movegen-time perpetual-check filter).
    /// It is test-support API for the repetition / cycle-boundary unit tests,
    /// which exercise the oldest-to-newest cycle walk directly.
    #[cfg(test)]
    pub(crate) fn history_since_last_distinct(&self) -> impl Iterator<Item = &StateInfo> + '_ {
        let len = self.history.len();
        let start = if len == 0 {
            len
        } else {
            match self.history[..len - 1]
                .iter()
                .rposition(|s| self.matches_current_state(s))
            {
                Some(p) => p + 1,
                None => len,
            }
        };
        self.history[start..len].iter()
    }

    /// The state exactly `dist` plies back from the current position, or `None`
    /// if that reaches past the recorded root. The timeline of states is
    /// `[root, history[0], …, history[len-1] == current]`, so distance `dist`
    /// back is timeline index `len - dist`: the root when `dist == len`, and
    /// `history[len - 1 - dist]` otherwise.
    fn state_back(&self, dist: usize) -> Option<&StateInfo> {
        let len = self.history.len();
        match len.cmp(&dist) {
            std::cmp::Ordering::Less => None,
            std::cmp::Ordering::Equal => self.root.as_ref(),
            std::cmp::Ordering::Greater => Some(&self.history[len - 1 - dist]),
        }
    }

    /// `plies_from_null` of the current (top-of-stack) state — 0 before any
    /// move (the setup root). The repetition look-back bound.
    fn current_plies_from_null(&self) -> i32 {
        self.history.last().map_or(0, |s| s.plies_from_null)
    }

    /// `continuous_check[color]` of the current (top-of-stack) state — 0 before
    /// any move.
    fn current_continuous_check(&self, color: Color) -> i32 {
        self.history
            .last()
            .map_or(0, |s| s.continuous_check[color.index()] as i32)
    }

    /// Compute the reference's incremental repetition triple `(repetition,
    /// repetition_times, repetition_type)` for the **current** position, walking
    /// the `StateInfo` chain back `min(16, plies_from_null)` plies in steps of
    /// two (`position.cpp`). Reads the *stored* `repetition_times` /
    /// `repetition_type` of the nearest prior occurrence, so the chain it builds
    /// on must already carry them (every state reached through `do_move` does).
    ///
    /// The walk classifies the **first** (nearest) matching prior occurrence:
    ///
    /// - same board **and** hands → a genuine repetition. `repetition_times`
    ///   chains off that occurrence's count; the sign of `repetition` is negative
    ///   once the count reaches 3 (the forced-fourfold case). The type is
    ///   `Lose` / `Win` for a perpetual check (see [`RepetitionState`] for the
    ///   viewpoint convention), otherwise `Draw` — and is downgraded to `Draw`
    ///   when the nearest occurrence classified differently (a cycle that was not
    ///   perpetual check the whole way is an ordinary draw, `position.cpp`).
    /// - same board, different hands → `Superior` / `Inferior` when one side to
    ///   move's hand dominates; otherwise a coincidental board match, keep
    ///   walking.
    /// - nothing found → `(0, 0, None)`.
    #[cfg(not(feature = "quick-draw"))]
    fn compute_repetition(&self) -> (i32, i32, RepetitionState) {
        let end = MAX_REPETITION_PLY.min(self.current_plies_from_null());
        // A repetition needs at least a four-ply cycle to return.
        if end < 4 {
            return (0, 0, RepetitionState::None);
        }

        let stm = self.side_to_move;
        let cc_stm = self.current_continuous_check(stm);
        let cc_opp = self.current_continuous_check(stm.flip());

        let mut i = 4i32;
        while i <= end {
            if let Some(prev) = self.state_back(i as usize) {
                // `board_key` is the board identity (side-to-move term folded
                // in) — no board copy is stored, matching the
                // reference exactly (position.cpp `is_repetition`,
                // ENABLE_QUICK_DRAW branch): a `board_key` hit is trusted
                // directly, then the stored hands decide DRAW vs
                // Superior/Inferior. This inherits the reference's accepted
                // hash-collision risk.
                if prev.board_key == self.board_key {
                    if prev.hands == self.hands {
                        // Same board and hands (same side to move — `i` is even):
                        // a genuine repetition of this position.
                        let times = prev.repetition_times + 1;
                        // 1..3rd occurrence: positive distance; 4th onward
                        // (`times >= 3`): negative, the forced-fourfold marker.
                        let repetition = if times >= 3 { -i } else { i };
                        let mut typ = if i <= cc_stm {
                            RepetitionState::Lose
                        } else if i <= cc_opp {
                            RepetitionState::Win
                        } else {
                            RepetitionState::Draw
                        };
                        // A cycle that was perpetual check only part of the way
                        // (its nearest prior occurrence classified differently)
                        // is an ordinary draw.
                        if prev.repetition_times != 0 && typ != prev.repetition_type {
                            typ = RepetitionState::Draw;
                        }
                        return (repetition, times, typ);
                    }
                    // Same board, different hands: superior/inferior is decided
                    // on the side to move's hand alone (by piece conservation
                    // the opponent's hand moves oppositely). `repetition_times`
                    // stays 0 here, matching the reference (2192-2205).
                    let cur_hand = &self.hands[stm.index()];
                    let prev_hand = &prev.hands[stm.index()];
                    if hand_is_equal_or_superior(cur_hand, prev_hand) {
                        return (i, 0, RepetitionState::Superior);
                    }
                    if hand_is_equal_or_superior(prev_hand, cur_hand) {
                        return (i, 0, RepetitionState::Inferior);
                    }
                    // Neither dominates: a coincidental board match. Keep walking.
                }
            }
            i += 2;
        }

        (0, 0, RepetitionState::None)
    }

    /// Classify the current position as a repetition from the search's viewpoint
    /// at search distance `ply` from the root.
    ///
    /// Ported from upstream YaneuraOu's `Position::is_repetition`
    /// (`position.cpp`): it reads the precomputed `st->repetition` /
    /// `st->repetition_type` (see [`Self::compute_repetition`]) and reports the
    /// type only when `repetition != 0 && repetition < ply`. That gate means:
    ///
    /// - a **positive** `repetition` (the 2nd/3rd occurrence) is reported only
    ///   when the earlier occurrence lies strictly *after* the search root
    ///   (`repetition < ply`); a two-fold that reaches back to or before the
    ///   root is not scored as an immediate draw;
    /// - a **negative** `repetition` (the 4th occurrence — a forced fourfold) is
    ///   always `< ply`, so it is reported regardless of where the earlier
    ///   occurrences sit, including entirely within the game history before the
    ///   root. This is the game-history repetition the previous ply-bounded
    ///   design could not see.
    ///
    /// The look-back that produced `repetition` spans the whole `do_move` chain
    /// (the `position … moves …` prefix included), bounded only by
    /// `min(16, plies_from_null)`.
    ///
    /// This is the **non**-`quick-draw` configuration; see the sibling
    /// definition for the `ENABLE_QUICK_DRAW` one the tournament reference
    /// actually plays.
    #[cfg(not(feature = "quick-draw"))]
    pub fn is_repetition(&self, ply: u16) -> RepetitionState {
        match self.history.last() {
            Some(s) if s.repetition != 0 && s.repetition < ply as i32 => s.repetition_type,
            _ => RepetitionState::None,
        }
    }

    /// Classify the current position as a repetition — the `ENABLE_QUICK_DRAW`
    /// variant (`position.cpp`), which the reference's
    /// `FOR_TOURNAMENT` build compiles (`config.h`) and which this crate
    /// selects by default.
    ///
    /// The differences from the non-QUICK_DRAW variant above are all deliberate,
    /// and each is what upstream ships:
    ///
    /// - **`ply` is ignored entirely.** There is no `repetition < ply` root gate,
    ///   so a repetition whose earlier occurrence lies *at* or *before* the
    ///   search root adjudicates just like one inside the search window.
    ///   Upstream measures looking back past the root as worth roughly +R5
    ///   (`position.cpp`). The parameter is kept in the signature so both
    ///   configurations present one API to the search.
    /// - **The 2nd occurrence already adjudicates.** There is no
    ///   `repetition_times` chain and hence no fourfold requirement: the first
    ///   `board_key` hit on the backward walk decides, full stop. By the same
    ///   token there is no "the cycle was not perpetual check the whole way"
    ///   downgrade to `Draw` — that rule only exists in the incremental variant.
    /// - **The walk runs at query time**, over the same
    ///   `min(16, plies_from_null)` window (`position.cpp`; the cap is a
    ///   literal 16 here, not `max_repetition_ply`). Fixing the cap at 16 rather
    ///   than the search ply is likewise worth about +R5 upstream
    ///   (`position.cpp`).
    ///
    /// Everything downstream is unchanged: the returned [`RepetitionState`]
    /// feeds the search's existing state-to-score mapping in both
    /// configurations.
    #[cfg(feature = "quick-draw")]
    pub fn is_repetition(&self, _ply: u16) -> RepetitionState {
        // `int end = std::min(16, st->pliesFromNull);`
        let end = MAX_REPETITION_PLY.min(self.current_plies_from_null());
        // `if (end < 4) return REPETITION_NONE;` — a repetition needs at least a
        // four-ply cycle to return to the same position with the same side to
        // move.
        if end < 4 {
            return RepetitionState::None;
        }

        let stm = self.side_to_move;
        let cc_stm = self.current_continuous_check(stm);
        let cc_opp = self.current_continuous_check(stm.flip());

        // `for (int i = 4; i <= end; i += 2)`, stepping two plies so the side to
        // move always matches.
        let mut i = 4i32;
        while i <= end {
            // `i <= end <= plies_from_null` keeps this in range for every state
            // reached through `do_move`; the `if let` is the total-function form
            // of the reference's raw `stp` chase.
            if let Some(prev) = self.state_back(i as usize)
                && prev.board_key == self.board_key
            {
                if prev.hands == self.hands {
                    // Same board and same hands: a genuine repetition. Note the
                    // reference compares only `st->hand` (the side to move's
                    // hand) — with the board equal and the side to move equal,
                    // piece conservation makes that equivalent to comparing both.
                    return if i <= cc_stm {
                        // The side to move delivered check on every one of its
                        // moves across the cycle, so it loses.
                        RepetitionState::Lose
                    } else if i <= cc_opp {
                        RepetitionState::Win
                    } else {
                        RepetitionState::Draw
                    };
                }
                // Same board, different hands: superior/inferior, decided on the
                // side to move's hand alone.
                let cur_hand = &self.hands[stm.index()];
                let prev_hand = &prev.hands[stm.index()];
                if hand_is_equal_or_superior(cur_hand, prev_hand) {
                    return RepetitionState::Superior;
                }
                if hand_is_equal_or_superior(prev_hand, cur_hand) {
                    return RepetitionState::Inferior;
                }
                // Neither dominates: a coincidental board match. Keep walking —
                // the reference falls through the `else` block here too.
            }
            i += 2;
        }

        RepetitionState::None
    }
}

/// The mutable-board guard returned by [`Position::board_mut`]. Derefs to the
/// underlying [`Board`] for direct edits and, on drop, recomputes the position's
/// per-state [`crate::search_movegen::CheckInfo`] so the eager cache never goes
/// stale behind a direct board mutation. See `board_mut` for the full contract.
pub struct BoardMut<'a> {
    pos: &'a mut Position,
}

impl std::ops::Deref for BoardMut<'_> {
    type Target = Board;

    fn deref(&self) -> &Board {
        &self.pos.board
    }
}

impl std::ops::DerefMut for BoardMut<'_> {
    fn deref_mut(&mut self) -> &mut Board {
        &mut self.pos.board
    }
}

impl Drop for BoardMut<'_> {
    fn drop(&mut self) {
        // The board just changed; recompute the eager check info from the
        // (possibly still key-stale) board — `compute_check_info` reads the
        // board and side to move, both current, and stamps `board_key` from
        // `self.board_key()` so the freshness `debug_assert` stays consistent.
        self.pos.check_info = self.pos.compute_check_info();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Undo {
    pub(crate) captured: Option<Piece>,
}

impl Undo {
    /// The piece captured by the move this `Undo` reverses, or `None` for a
    /// non-capturing move — the reference's `pos.captured_piece()` for the move
    /// that reached the resulting position.
    pub fn captured(self) -> Option<Piece> {
        self.captured
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Piece, PieceKind};
    use crate::square::Square;

    #[test]
    fn empty_has_black_to_move_and_ply_one() {
        let p = Position::empty();
        assert_eq!(p.side_to_move(), Color::Black);
        assert_eq!(p.ply(), 1);
    }

    #[test]
    fn empty_has_no_pieces_or_hands() {
        let p = Position::empty();
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            assert!(p.board().get(sq).is_none());
        }
        for color in [Color::Black, Color::White] {
            for kind in [
                PieceKind::Pawn,
                PieceKind::Lance,
                PieceKind::Knight,
                PieceKind::Silver,
                PieceKind::Gold,
                PieceKind::Bishop,
                PieceKind::Rook,
            ] {
                assert_eq!(p.hand(color).count(kind), 0);
            }
        }
    }

    #[test]
    fn mutating_one_hand_does_not_disturb_the_other() {
        let mut p = Position::empty();
        p.hand_mut(Color::Black).increment(PieceKind::Pawn);
        assert_eq!(p.hand(Color::Black).count(PieceKind::Pawn), 1);
        assert_eq!(p.hand(Color::White).count(PieceKind::Pawn), 0);
    }

    #[test]
    fn board_mut_writes_through_to_board() {
        let mut p = Position::empty();
        let sq = Square::new(0, 0).unwrap();
        let piece = Piece::new(PieceKind::King, Color::Black);
        p.board_mut().set(sq, Some(piece));
        assert_eq!(p.board().get(sq), Some(piece));
    }

    #[test]
    fn do_undo_round_trips_pawn_push_at_startpos() {
        let mut p = Position::startpos();
        let snapshot = p.clone();
        let from = Square::new(6, 6).unwrap(); // 7g
        let to = Square::new(6, 5).unwrap(); // 7f
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(from, to, pawn);
        let undo = p.do_move(m);
        assert_eq!(p.side_to_move(), Color::White);
        assert_eq!(p.ply(), 2);
        assert_eq!(p.board().get(to), Some(pawn));
        assert!(p.board().get(from).is_none());
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn do_undo_round_trips_capture() {
        let mut p = Position::empty();
        let from = Square::new(4, 4).unwrap();
        let to = Square::new(4, 3).unwrap();
        let attacker = Piece::new(PieceKind::Rook, Color::Black);
        let victim = Piece::new(PieceKind::Pawn, Color::White);
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        p.board_mut().set(from, Some(attacker));
        p.board_mut().set(to, Some(victim));
        p.board_mut().set(Square::new(0, 8).unwrap(), Some(bk));
        p.board_mut().set(Square::new(0, 0).unwrap(), Some(wk));
        let snapshot = p.clone();
        let m = Move::make(from, to, attacker);
        let undo = p.do_move(m);
        assert_eq!(p.board().get(to), Some(attacker));
        assert!(p.board().get(from).is_none());
        assert_eq!(p.hand(Color::Black).count(PieceKind::Pawn), 1);
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn do_undo_round_trips_promotion() {
        let mut p = Position::empty();
        let from = Square::new(4, 3).unwrap(); // rank 3, just outside promotion zone
        let to = Square::new(4, 2).unwrap(); // rank 2, in promotion zone
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        p.board_mut().set(from, Some(pawn));
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        let snapshot = p.clone();
        let m = Move::make_promote(from, to, pawn);
        let undo = p.do_move(m);
        let placed = p.board().get(to).unwrap();
        assert_eq!(placed.kind, PieceKind::Pawn);
        assert!(placed.promoted);
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn do_undo_round_trips_drop() {
        let mut p = Position::empty();
        p.hand_mut(Color::Black).increment(PieceKind::Pawn);
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        let snapshot = p.clone();
        let to = Square::new(4, 4).unwrap();
        let m = Move::make_drop(PieceKind::Pawn, Color::Black, to);
        let undo = p.do_move(m);
        assert_eq!(
            p.board().get(to),
            Some(Piece::new(PieceKind::Pawn, Color::Black))
        );
        assert_eq!(p.hand(Color::Black).count(PieceKind::Pawn), 0);
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    // ---- Position-history infrastructure (Phase 1) -----------------------

    /// Two-king board with no other pieces. Used as the base for the
    /// king-shuffle cycle tests below; both kings have free space to step
    /// sideways without check geometry getting in the way.
    fn setup_king_shuffle_pos() -> Position {
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p
    }

    /// Four moves that, applied in order from `setup_king_shuffle_pos`, return
    /// the position to its starting state. Black-K and White-K each step right
    /// then back; side-to-move alternates each move.
    fn shuffle_cycle() -> [Move; 4] {
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        [
            Move::make(Square::new(0, 8).unwrap(), Square::new(1, 8).unwrap(), bk),
            Move::make(Square::new(0, 0).unwrap(), Square::new(1, 0).unwrap(), wk),
            Move::make(Square::new(1, 8).unwrap(), Square::new(0, 8).unwrap(), bk),
            Move::make(Square::new(1, 0).unwrap(), Square::new(0, 0).unwrap(), wk),
        ]
    }

    #[test]
    fn do_undo_round_trips_capture_promotion() {
        // Black pawn at (4,1) captures a White knight at (4,0) and promotes
        // to Tokin (forced — last rank). Round-trip must restore board, hands,
        // and history exactly.
        let mut p = Position::empty();
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let target = Piece::new(PieceKind::Knight, Color::White);
        p.board_mut().set(Square::new(8, 8).unwrap(), Some(bk));
        p.board_mut().set(Square::new(0, 0).unwrap(), Some(wk));
        p.board_mut().set(Square::new(4, 1).unwrap(), Some(pawn));
        p.board_mut().set(Square::new(4, 0).unwrap(), Some(target));
        let snapshot = p.clone();
        let from = Square::new(4, 1).unwrap();
        let to = Square::new(4, 0).unwrap();
        let m = Move::make_promote(from, to, pawn);
        let undo = p.do_move(m);
        let placed = p.board().get(to).unwrap();
        assert_eq!(placed.kind, PieceKind::Pawn);
        assert!(placed.promoted);
        assert_eq!(p.hand(Color::Black).count(PieceKind::Knight), 1);
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn do_move_grows_history_by_one() {
        let mut p = Position::startpos();
        let prior_len = p.history.len();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(Square::new(6, 6).unwrap(), Square::new(6, 5).unwrap(), pawn);
        p.do_move(m);
        assert_eq!(p.history.len(), prior_len + 1);
    }

    #[test]
    fn undo_move_shrinks_history_by_one() {
        let mut p = Position::startpos();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(Square::new(6, 6).unwrap(), Square::new(6, 5).unwrap(), pawn);
        let undo = p.do_move(m);
        let post_do_len = p.history.len();
        p.undo_move(m, undo);
        assert_eq!(p.history.len(), post_do_len - 1);
    }

    #[test]
    fn startpos_history_is_empty() {
        assert_eq!(Position::startpos().history.len(), 0);
        assert_eq!(Position::empty().history.len(), 0);
        let parsed = crate::sfen::parse_sfen(crate::sfen::STARTPOS_SFEN).unwrap();
        assert_eq!(parsed.history.len(), 0);
    }

    #[test]
    fn position_occurrences_zero_in_fresh_state() {
        assert_eq!(Position::startpos().position_occurrences(), 0);
        assert_eq!(Position::empty().position_occurrences(), 0);
    }

    #[test]
    fn position_occurrences_after_round_trip_cycle() {
        // King-shuffle: four moves return to start. After the full cycle,
        // `history.last()` equals start; every other entry differs.
        let mut p = setup_king_shuffle_pos();
        for m in shuffle_cycle() {
            p.do_move(m);
        }
        assert_eq!(p.position_occurrences(), 1);
    }

    #[test]
    fn gives_check_true_for_check_move() {
        // Black rook slides along rank 8 onto file 4; the rook's column then
        // attacks the white king at (4,0) through empty squares.
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(8, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let m = Move::make(Square::new(0, 8).unwrap(), Square::new(4, 8).unwrap(), rook);
        p.do_move(m);
        assert!(
            p.history.last().unwrap().gives_check,
            "rook to file 4 should put white king in check",
        );
    }

    #[test]
    fn gives_check_false_for_quiet_move() {
        // 7g7f at startpos puts no white piece in check.
        let mut p = Position::startpos();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(Square::new(6, 6).unwrap(), Square::new(6, 5).unwrap(), pawn);
        p.do_move(m);
        assert!(
            !p.history.last().unwrap().gives_check,
            "7g7f is a quiet move",
        );
    }

    #[test]
    fn gives_check_true_for_discovered_check() {
        // Black knight at (4,5) blocks Black rook at (4,8) from attacking
        // white king at (4,0). Knight steps off-file to (3,3): doesn't itself
        // attack (4,0), but uncovers the rook's attack along file 4. Asserts
        // gives_check tracks attackers via `is_attacked_by` (not just direct
        // mover-piece attack).
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p.board_mut().set(
            Square::new(4, 8).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 5).unwrap(),
            Some(Piece::new(PieceKind::Knight, Color::Black)),
        );
        let knight = Piece::new(PieceKind::Knight, Color::Black);
        let m = Move::make(
            Square::new(4, 5).unwrap(),
            Square::new(3, 3).unwrap(),
            knight,
        );
        p.do_move(m);
        assert!(
            p.history.last().unwrap().gives_check,
            "knight move should uncover rook discovered-check",
        );
    }

    #[test]
    fn history_since_last_distinct_walks_short_cycle() {
        // Two king-shuffle cycles → start state appears twice in history. The
        // iterator yields the *most-recent* cycle: 4 entries, oldest-to-newest,
        // ending at history.last() (== current state). Callers index into that
        // sequence expecting the order — the per-element assertions below pin
        // the contract.
        let mut p = setup_king_shuffle_pos();
        let moves = shuffle_cycle();
        for _ in 0..2 {
            for m in moves {
                p.do_move(m);
            }
        }
        assert_eq!(p.position_occurrences(), 2);

        // Replay one cycle from a fresh fixture-pos so we have a known-good
        // sequence of expected post-move states to compare against. A record
        // carries no board copy, so board identity is checked through
        // `board_key` — the equality key repetition itself uses.
        let mut expected = setup_king_shuffle_pos();
        let mut expected_states: Vec<(u64, Color)> = Vec::new();
        for m in moves {
            expected.do_move(m);
            expected_states.push((expected.board_key(), expected.side_to_move()));
        }

        let cycle: Vec<&StateInfo> = p.history_since_last_distinct().collect();
        assert_eq!(cycle.len(), 4, "cycle: {cycle:?}");
        for (i, s) in cycle.iter().enumerate() {
            assert_eq!(
                (s.board_key, s.side_to_move),
                expected_states[i],
                "cycle[{i}] does not match expected post-move state",
            );
        }
        // And the last entry is the current state.
        assert_eq!(cycle[3].board_key, p.board_key());
        assert_eq!(cycle[3].side_to_move, p.side_to_move());
    }

    #[test]
    fn history_since_last_distinct_is_empty_when_no_prior_occurrence() {
        // Single move from startpos. The pushed state is the first occurrence
        // of itself; no prior occurrence exists in `history[..len-1]`.
        let mut p = Position::startpos();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(Square::new(6, 6).unwrap(), Square::new(6, 5).unwrap(), pawn);
        p.do_move(m);
        let cycle: Vec<&StateInfo> = p.history_since_last_distinct().collect();
        assert!(cycle.is_empty(), "expected empty cycle, got {cycle:?}");
    }

    // ---- Zobrist key gate ------------------------------------------------

    use crate::sfen::parse_sfen;

    /// The six perft-fixture SFENs, matching `tests/fixtures/perft/*.json`. The
    /// randomized-playout gate seeds one deterministic game from each.
    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",                // sennichite
    ];

    /// Small deterministic xorshift64* — mirrors the driver in
    /// `attic-eval/tests/incremental_parity.rs`. `Math.random`-style
    /// nondeterminism is banned in this workspace.
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

        fn pick(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Assert the incrementally-maintained keys equal a from-scratch
    /// recomputation, and that `key() == board_key ^ hand_key`.
    fn assert_key_consistent(p: &Position, ctx: &str) {
        let (board_key, hand_key) = p.recomputed_keys();
        assert_eq!(p.board_key(), board_key, "{ctx}: board_key diverged");
        assert_eq!(p.hand_key(), hand_key, "{ctx}: hand_key diverged");
        assert_eq!(
            p.key(),
            board_key ^ hand_key,
            "{ctx}: key() != board_key ^ hand_key",
        );

        // Partial keys: incremental value must equal the from-scratch walk.
        let (pawn_key, minor_piece_key, non_pawn_key) = p.recomputed_partial_keys();
        assert_eq!(p.pawn_key(), pawn_key, "{ctx}: pawn_key diverged");
        assert_eq!(
            p.minor_piece_key(),
            minor_piece_key,
            "{ctx}: minor_piece_key diverged",
        );
        assert_eq!(
            p.non_pawn_key(Color::Black),
            non_pawn_key[Color::Black.index()],
            "{ctx}: non_pawn_key(Black) diverged",
        );
        assert_eq!(
            p.non_pawn_key(Color::White),
            non_pawn_key[Color::White.index()],
            "{ctx}: non_pawn_key(White) diverged",
        );
    }

    #[test]
    fn incremental_key_matches_from_scratch_on_random_playouts() {
        const MIN_PLIES: usize = 30;

        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
            assert_key_consistent(&pos, &format!("fixture {fi} root"));

            // Seed derived from the fixture index so each game is distinct yet
            // reproducible run-to-run.
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, Undo)> = Vec::new();
            let mut legal: Vec<Move> = Vec::new();

            let mut plies = 0usize;
            while plies < MIN_PLIES {
                legal.clear();
                pos.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    // Terminal: unwind fully (exercising undo) and restart from
                    // the root, exactly as the eval parity driver does.
                    while let Some((m, u)) = stack.pop() {
                        pos.undo_move(m, u);
                        assert_key_consistent(&pos, &format!("fixture {fi} restart undo"));
                    }
                    continue;
                }
                let m = legal[rng.pick(legal.len())];
                let u = pos.do_move(m);
                assert_key_consistent(&pos, &format!("fixture {fi} ply {plies} after do"));
                stack.push((m, u));
                plies += 1;
            }

            while let Some((m, u)) = stack.pop() {
                pos.undo_move(m, u);
                assert_key_consistent(&pos, &format!("fixture {fi} after undo"));
            }
        }
    }

    #[test]
    fn transposition_reaches_identical_key() {
        // Two black pawn pushes and two white pawn pushes, interleaved in two
        // different (turn-respecting) orders. Both reach the same board, hands,
        // and side-to-move, so the keys must agree.
        let line_a = ["2g2f", "8c8d", "7g7f", "3c3d"];
        let line_b = ["7g7f", "3c3d", "2g2f", "8c8d"];

        let play = |line: &[&str]| {
            let mut p = Position::startpos();
            for usi in line {
                let m = crate::move_::parse_usi_move(usi, &p).expect("legal move");
                p.do_move(m);
            }
            p
        };

        let a = play(&line_a);
        let b = play(&line_b);

        // Sanity: the two lines genuinely transpose.
        assert_eq!(a.board(), b.board());
        assert_eq!(a.side_to_move(), b.side_to_move());
        for color in [Color::Black, Color::White] {
            assert_eq!(a.hand(color), b.hand(color));
        }

        assert_eq!(a.key(), b.key(), "transposed positions must share a key");
        assert_key_consistent(&a, "transposition line a");
        assert_key_consistent(&b, "transposition line b");
    }

    #[test]
    fn hand_key_capture_then_drop_cancels() {
        // Board: black rook 5e (4,4), white pawn 5d (4,3), kings tucked in the
        // corners. Black captures the pawn (rook 5e→5d): a pawn enters Black's
        // hand. White makes a throwaway king step. Black then drops a pawn: the
        // pawn leaves Black's hand. The add and subtract are inverses, so
        // `hand_key` returns to its pre-capture value.
        let mut p = parse_sfen("k8/9/9/4p4/4R4/9/9/9/8K b - 1").unwrap();
        let h0 = p.hand_key();
        assert_key_consistent(&p, "capture/drop root");

        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let capture = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), rook);
        p.do_move(capture);
        assert_ne!(p.hand_key(), h0, "capture should push a pawn into hand");
        assert_key_consistent(&p, "after capture");

        let wk = Piece::new(PieceKind::King, Color::White);
        let wait = Move::make(Square::new(8, 0).unwrap(), Square::new(7, 0).unwrap(), wk);
        p.do_move(wait);
        assert_key_consistent(&p, "after white wait");

        let drop = Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(4, 4).unwrap());
        p.do_move(drop);
        assert_key_consistent(&p, "after drop");
        assert_eq!(
            p.hand_key(),
            h0,
            "capture then drop of the same kind must restore hand_key",
        );
    }

    #[test]
    fn hand_count_changes_the_key() {
        // Two positions identical but for one pawn's presence in Black's hand.
        let one = parse_sfen("9/9/9/9/9/9/9/9/9 b P 1").unwrap();
        let two = parse_sfen("9/9/9/9/9/9/9/9/9 b 2P 1").unwrap();
        assert_ne!(one.key(), two.key(), "differing hand counts must differ");
        assert_ne!(one.hand_key(), two.hand_key());
        // The board halves are identical (same empty board, same side).
        assert_eq!(one.board_key(), two.board_key());
    }

    #[test]
    fn side_to_move_changes_the_key() {
        // Same board and (empty) hands, opposite side to move.
        let black = parse_sfen("9/9/9/9/4P4/9/9/9/9 b - 1").unwrap();
        let white = parse_sfen("9/9/9/9/4P4/9/9/9/9 w - 1").unwrap();
        assert_ne!(black.key(), white.key(), "side-to-move must change the key");
        assert_ne!(black.board_key(), white.board_key());
        // The side term lives in board_key; hands are identical.
        assert_eq!(black.hand_key(), white.hand_key());
    }

    #[test]
    fn partial_keys_ignore_hand_contents() {
        // Two positions with the same board / side but different hands must
        // carry identical partial keys — pieces in hand contribute to none of
        // them (only the full `hand_key` distinguishes the two).
        let a = parse_sfen("9/9/9/9/4P4/9/9/9/9 b - 1").unwrap();
        let b = parse_sfen("9/9/9/9/4P4/9/9/9/9 b 2Rb 1").unwrap();
        assert_eq!(a.pawn_key(), b.pawn_key(), "pawn_key must ignore hands");
        assert_eq!(
            a.minor_piece_key(),
            b.minor_piece_key(),
            "minor_piece_key must ignore hands",
        );
        assert_eq!(a.non_pawn_key(Color::Black), b.non_pawn_key(Color::Black));
        assert_eq!(a.non_pawn_key(Color::White), b.non_pawn_key(Color::White));
        // But the full key differs (hands differ).
        assert_ne!(a.key(), b.key());
    }

    #[test]
    fn pawn_key_empty_when_no_board_pawns() {
        // A board with no pawns hashes `pawn_key` to the dedicated `noPawns`
        // seed (a non-zero constant), regardless of other pieces present.
        let p = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        assert_eq!(p.pawn_key(), crate::key::NO_PAWNS_SEED);
        // A board with a pawn differs from the empty-pawn seed.
        let q = parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b - 1").unwrap();
        assert_ne!(q.pawn_key(), crate::key::NO_PAWNS_SEED);
    }

    #[test]
    fn promoting_pawn_leaves_pawn_key_and_enters_minor_and_nonpawn() {
        // Black pawn on 5c (rank 2, in the promotion zone edge) promotes forward
        // to 5b. Before: the pawn is in `pawn_key`, not in `non_pawn_key` /
        // `minor_piece_key`. After: it has left `pawn_key` (back to the empty
        // seed on an otherwise pawnless board) and entered both the promoted
        // piece's `minor_piece_key` and Black's `non_pawn_key`.
        let mut p = parse_sfen("4k4/9/4P4/9/9/9/9/9/4K4 b - 1").unwrap();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        // 5c == file 4 (0-based), rank 2; 5b == rank 1.
        let from = Square::new(4, 2).unwrap();
        let to = Square::new(4, 1).unwrap();
        let minor_before = p.minor_piece_key();
        let nonpawn_before = p.non_pawn_key(Color::Black);
        assert_ne!(p.pawn_key(), crate::key::NO_PAWNS_SEED, "pawn present");

        let m = Move::make_promote(from, to, pawn);
        let undo = p.do_move(m);
        assert_key_consistent(&p, "after promoting pawn push");
        // The only board pawn promoted away, so `pawn_key` is back to the seed.
        assert_eq!(
            p.pawn_key(),
            crate::key::NO_PAWNS_SEED,
            "promoted pawn must leave pawn_key",
        );
        assert_ne!(
            p.minor_piece_key(),
            minor_before,
            "promoted pawn (a minor piece) must enter minor_piece_key",
        );
        assert_ne!(
            p.non_pawn_key(Color::Black),
            nonpawn_before,
            "promoted pawn must enter Black's non_pawn_key",
        );

        p.undo_move(m, undo);
        assert_key_consistent(&p, "after undo of promoting pawn push");
        assert_ne!(p.pawn_key(), crate::key::NO_PAWNS_SEED, "pawn restored");
        assert_eq!(p.minor_piece_key(), minor_before);
        assert_eq!(p.non_pawn_key(Color::Black), nonpawn_before);
    }

    #[test]
    fn pawn_drop_enters_pawn_key() {
        // A pawn drop places a board pawn, so `pawn_key` must change (the copy
        // that left the hand contributed to no partial key). Round-trips exact.
        let mut p = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
        assert_eq!(p.pawn_key(), crate::key::NO_PAWNS_SEED, "no board pawn yet");
        let to = Square::new(4, 4).unwrap();
        let m = Move::make_drop(PieceKind::Pawn, Color::Black, to);
        let undo = p.do_move(m);
        assert_key_consistent(&p, "after pawn drop");
        assert_ne!(
            p.pawn_key(),
            crate::key::NO_PAWNS_SEED,
            "dropped pawn must enter pawn_key",
        );
        p.undo_move(m, undo);
        assert_key_consistent(&p, "after undo pawn drop");
        assert_eq!(p.pawn_key(), crate::key::NO_PAWNS_SEED);
    }

    #[test]
    fn key_agrees_with_repetition_equality_on_sennichite() {
        // Along the sennichite perft fixture's move list, every recurrence that
        // `position_occurrences` detects (structural equality of board, hands,
        // side-to-move) must carry a key equal to its earlier occurrences —
        // i.e. the key is a faithful function of that triple.
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/perft/sennichite.json",
        ))
        .expect("read sennichite fixture");
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse fixture");
        let sfen = json["sfen"].as_str().expect("fixture sfen");
        let moves: Vec<String> = json["moves"]
            .as_array()
            .expect("fixture moves")
            .iter()
            .map(|m| m.as_str().expect("move string").to_string())
            .collect();

        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        // One (structural signature, key) entry per pushed history state,
        // mirroring `history` exactly.
        type Signature = (Board, [Hand; Color::COUNT], Color);
        let mut seen: Vec<(Signature, u64)> = Vec::new();
        let mut any_recurrence = false;

        for usi in &moves {
            let m = crate::move_::parse_usi_move(usi, &pos).expect("fixture move parses");
            pos.do_move(m);
            assert_key_consistent(&pos, &format!("sennichite `{usi}`"));

            let sig: Signature = (
                *pos.board(),
                [*pos.hand(Color::Black), *pos.hand(Color::White)],
                pos.side_to_move(),
            );
            let key = pos.key();

            let priors: Vec<u64> = seen
                .iter()
                .filter(|(s, _)| *s == sig)
                .map(|(_, k)| *k)
                .collect();
            // `position_occurrences` counts the just-pushed entry plus every
            // prior structural match — so it must equal priors + 1.
            assert_eq!(
                pos.position_occurrences(),
                priors.len() + 1,
                "occurrence count vs structural signature disagree at `{usi}`",
            );
            for prior_key in &priors {
                any_recurrence = true;
                assert_eq!(
                    *prior_key, key,
                    "recurrence at `{usi}` has a different key than its earlier occurrence",
                );
            }
            seen.push((sig, key));
        }

        assert!(
            any_recurrence,
            "sennichite fixture should exercise at least one recurrence",
        );
    }

    // ---- RepetitionState / is_repetition ---------------------------------

    /// Load the sennichite perft fixture's `(sfen, moves)`.
    fn load_sennichite() -> (String, Vec<String>) {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/perft/sennichite.json",
        ))
        .expect("read sennichite fixture");
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse fixture");
        let sfen = json["sfen"].as_str().expect("fixture sfen").to_string();
        let moves = json["moves"]
            .as_array()
            .expect("fixture moves")
            .iter()
            .map(|m| m.as_str().expect("move string").to_string())
            .collect();
        (sfen, moves)
    }

    /// Build a `StateInfo` for `(board, hands, side)` with its `board_key`
    /// computed the same way the incremental path would, so hand-assembled
    /// histories agree with `Position::board_key` on the board half. The
    /// incremental repetition fields are zeroed (`assemble` fills in the
    /// `plies_from_null` chain and recomputes the top state's repetition).
    fn make_state(
        board: Board,
        hands: [Hand; Color::COUNT],
        side: Color,
        gives_check: bool,
    ) -> StateInfo {
        let mut scratch = Position::empty();
        *scratch.board_mut() = board;
        scratch.hands = hands;
        scratch.set_side_to_move(side);
        scratch.refresh_keys();
        StateInfo {
            hands,
            side_to_move: side,
            gives_check,
            board_key: scratch.board_key(),
            plies_from_null: 0,
            continuous_check: [0; Color::COUNT],
            #[cfg(not(feature = "quick-draw"))]
            repetition: 0,
            #[cfg(not(feature = "quick-draw"))]
            repetition_times: 0,
            #[cfg(not(feature = "quick-draw"))]
            repetition_type: RepetitionState::None,
        }
    }

    #[test]
    fn is_repetition_none_before_and_draw_at_sennichite_recurrence() {
        // The sennichite fixture is a four-ply king shuffle: the root recurs
        // every four plies. The reference reports NONE until the first
        // recurrence (which lands straight on the root, so it must be found via
        // the root snapshot), then DRAW from the four-ply mark onward.
        let (sfen, moves) = load_sennichite();
        let mut pos = parse_sfen(&sfen).expect("fixture sfen parses");
        for (idx, usi) in moves.iter().enumerate() {
            let m = crate::move_::parse_usi_move(usi, &pos).expect("fixture move parses");
            pos.do_move(m);
            let depth = idx + 1;
            let rep = pos.is_repetition(16);
            if depth < 4 {
                assert_eq!(
                    rep,
                    RepetitionState::None,
                    "no repetition should be reported at depth {depth}",
                );
            } else {
                assert_eq!(
                    rep,
                    RepetitionState::Draw,
                    "plain sennichite must classify as DRAW at depth {depth}",
                );
            }
        }
    }

    #[test]
    fn is_repetition_lose_when_side_to_move_is_perpetual_checker() {
        // Black rook oscillates giving check to a shuffling White king; the
        // position returns to the root after four plies. At that point Black
        // (the side to move) is the one that checked on every move, so the
        // repetition is a LOSE for the side to move.
        let mut p = Position::empty();
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let br = Piece::new(PieceKind::Rook, Color::Black);
        p.board_mut().set(Square::new(0, 8).unwrap(), Some(bk)); // Black king, far corner
        p.board_mut().set(Square::new(4, 0).unwrap(), Some(wk)); // White king K0
        p.board_mut().set(Square::new(8, 1).unwrap(), Some(br)); // Black rook A
        p.refresh_keys();

        // A=(8,1) checks a king on rank 1; B=(8,0) checks a king on rank 0.
        let a = Square::new(8, 1).unwrap();
        let b = Square::new(8, 0).unwrap();
        let k0 = Square::new(4, 0).unwrap();
        let k1 = Square::new(4, 1).unwrap();
        p.do_move(Move::make(a, b, br)); // Black: rook A→B, checks WK@K0
        p.do_move(Move::make(k0, k1, wk)); // White: king escapes to K1
        p.do_move(Move::make(b, a, br)); // Black: rook B→A, checks WK@K1
        p.do_move(Move::make(k1, k0, wk)); // White: king back to K0 (== root)

        assert_eq!(
            p.is_repetition(16),
            RepetitionState::Lose,
            "the perpetually-checking side to move must be classified LOSE",
        );
    }

    #[test]
    fn is_repetition_win_when_opponent_is_perpetual_checker() {
        // Mirror of the LOSE case: a White rook perpetually checks a shuffling
        // Black king. The position recurs with Black to move; the opponent
        // (White) was the perpetual checker, so it is a WIN for the side to
        // move.
        let mut p = Position::empty();
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let wr = Piece::new(PieceKind::Rook, Color::White);
        p.board_mut().set(Square::new(0, 8).unwrap(), Some(wk)); // White king, far corner
        p.board_mut().set(Square::new(4, 0).unwrap(), Some(bk)); // Black king K0 (in check)
        p.board_mut().set(Square::new(8, 0).unwrap(), Some(wr)); // White rook B, checks K0
        p.set_side_to_move(Color::Black);
        p.refresh_keys();

        let a = Square::new(8, 1).unwrap();
        let b = Square::new(8, 0).unwrap();
        let k0 = Square::new(4, 0).unwrap();
        let k1 = Square::new(4, 1).unwrap();
        p.do_move(Move::make(k0, k1, bk)); // Black: king escapes to K1
        p.do_move(Move::make(b, a, wr)); // White: rook B→A, checks BK@K1
        p.do_move(Move::make(k1, k0, bk)); // Black: king back to K0
        p.do_move(Move::make(a, b, wr)); // White: rook A→B, checks BK@K0 (== root)

        assert_eq!(
            p.is_repetition(16),
            RepetitionState::Win,
            "the checked side to move must be classified WIN",
        );
    }

    /// The two-king board shared by every superior/inferior fixture state. All
    /// such states differ only in their hands, so they share this board (and
    /// therefore its `board_key`) — which is exactly the "same board, different
    /// hands" case `is_repetition` must classify.
    fn superiority_board() -> Board {
        let mut board = Board::empty();
        board.set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        board.set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        board
    }

    /// A [`superiority_board`] state plus the given per-color hand-pawn counts,
    /// Black to move. Used to assemble superior/inferior histories directly.
    fn superiority_state(black_pawns: u8, white_pawns: u8) -> StateInfo {
        let board = superiority_board();
        let mut hands = [Hand::empty(), Hand::empty()];
        for _ in 0..black_pawns {
            hands[Color::Black.index()].increment(PieceKind::Pawn);
        }
        for _ in 0..white_pawns {
            hands[Color::White.index()].increment(PieceKind::Pawn);
        }
        make_state(board, hands, Color::Black, false)
    }

    /// Assemble a `Position` on `board` whose current state is `states.last()`,
    /// whose `history` is `states`, and whose lookback root is `root`. Bypasses
    /// `do_move` so a controlled superior/inferior history can be fed to
    /// `is_repetition` directly. `board` is supplied separately because the
    /// per-state records carry no board copy; every state in a
    /// superior/inferior fixture shares the one board.
    ///
    /// Each state's `plies_from_null` is set to its timeline distance from the
    /// root (1, 2, …), so the top state's repetition look-back reaches the whole
    /// assembled chain; the top state's repetition triple is then computed via
    /// `store_current_repetition`, exactly as `do_move` would.
    fn assemble(board: Board, states: &[StateInfo], root: StateInfo) -> Position {
        let last = states.last().expect("non-empty history").clone();
        let mut p = Position::empty();
        *p.board_mut() = board;
        p.hands = last.hands;
        p.set_side_to_move(last.side_to_move);
        p.refresh_keys();
        p.history = states.to_vec();
        for (idx, s) in p.history.iter_mut().enumerate() {
            s.plies_from_null = idx as i32 + 1;
        }
        p.root = Some(root);
        // Under `quick-draw` there is no precompute to seed: `is_repetition`
        // walks this assembled chain itself.
        #[cfg(not(feature = "quick-draw"))]
        p.store_current_repetition();
        p
    }

    #[test]
    fn is_repetition_superior_when_side_to_move_gained_a_hand_pawn() {
        // Earlier occurrence (the root): Black holds no hand pawn. Current: the
        // same board, but Black (the side to move) now holds one — a superior
        // position. The four-ply distance makes the root the candidate.
        let root = superiority_state(0, 1);
        let filler = superiority_state(9, 9); // never inspected (only dist 4 checked)
        let current = superiority_state(1, 0);
        let p = assemble(
            superiority_board(),
            &[filler.clone(), filler.clone(), filler, current],
            root,
        );
        assert_eq!(p.is_repetition(16), RepetitionState::Superior);
    }

    #[test]
    fn is_repetition_inferior_when_side_to_move_lost_a_hand_pawn() {
        // Mirror of the superior case: Black now holds fewer hand pawns than at
        // the earlier occurrence.
        let root = superiority_state(1, 0);
        let filler = superiority_state(9, 9);
        let current = superiority_state(0, 1);
        let p = assemble(
            superiority_board(),
            &[filler.clone(), filler.clone(), filler, current],
            root,
        );
        assert_eq!(p.is_repetition(16), RepetitionState::Inferior);
    }

    /// Both kings walk a triangle and return to the root only after six plies,
    /// so the nearest prior occurrence of the final position is the root itself,
    /// six plies back. Shared by the two configuration-specific ply-gate tests.
    fn six_ply_triangle_cycle() -> Position {
        let mut p = Position::empty();
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        p.board_mut().set(Square::new(0, 8).unwrap(), Some(bk));
        p.board_mut().set(Square::new(0, 0).unwrap(), Some(wk));
        p.refresh_keys();

        // Black king triangle (0,8)->(1,8)->(0,7)->(0,8); White king triangle
        // (0,0)->(1,0)->(0,1)->(0,0). Interleaved: B,W,B,W,B,W.
        let bmoves = [
            (Square::new(0, 8).unwrap(), Square::new(1, 8).unwrap()),
            (Square::new(1, 8).unwrap(), Square::new(0, 7).unwrap()),
            (Square::new(0, 7).unwrap(), Square::new(0, 8).unwrap()),
        ];
        let wmoves = [
            (Square::new(0, 0).unwrap(), Square::new(1, 0).unwrap()),
            (Square::new(1, 0).unwrap(), Square::new(0, 1).unwrap()),
            (Square::new(0, 1).unwrap(), Square::new(0, 0).unwrap()),
        ];
        for i in 0..3 {
            p.do_move(Move::make(bmoves[i].0, bmoves[i].1, bk));
            p.do_move(Move::make(wmoves[i].0, wmoves[i].1, wk));
        }
        p
    }

    #[cfg(not(feature = "quick-draw"))]
    #[test]
    fn is_repetition_gated_by_search_ply() {
        // The nearest prior occurrence is the root, six plies back, so
        // `repetition == 6`. The non-QUICK_DRAW gate reports the draw only when
        // `repetition < ply` — i.e. only when the earlier occurrence lies
        // strictly *after* the search root:
        //   ply <= 6 → the recurrence reaches to/before the root → NONE;
        //   ply  > 6 → the root sits strictly inside the search window → DRAW.
        let p = six_ply_triangle_cycle();

        assert_eq!(
            p.is_repetition(6),
            RepetitionState::None,
            "a recurrence reaching the root (repetition == ply) is not an immediate draw",
        );
        assert_eq!(
            p.is_repetition(7),
            RepetitionState::Draw,
            "with the root strictly inside the search window the recurrence is a draw",
        );
        assert_eq!(p.is_repetition(16), RepetitionState::Draw);
    }

    #[cfg(feature = "quick-draw")]
    #[test]
    fn is_repetition_ignores_search_ply_under_quick_draw() {
        // Same six-ply cycle. QUICK_DRAW has no `repetition < ply` gate at all
        // (`position.cpp` never reads `ply`), so the recurrence onto
        // the root is a draw at *every* search ply — including ply 0, where the
        // whole cycle lies before the root.
        let p = six_ply_triangle_cycle();

        for ply in [0u16, 1, 6, 7, 16] {
            assert_eq!(
                p.is_repetition(ply),
                RepetitionState::Draw,
                "QUICK_DRAW ignores ply (ply {ply})",
            );
        }
    }

    #[cfg(not(feature = "quick-draw"))]
    #[test]
    fn is_repetition_forced_fourfold_is_seen_across_the_search_root() {
        // Play the four-ply king shuffle three times: the root position occurs
        // a fourth time (the current position). `repetition_times` chains up to
        // 3, which flips `repetition` negative — the forced-fourfold marker.
        let mut p = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        for _ in 0..3 {
            for m in cycle {
                p.do_move(m);
            }
        }

        let top = p.history.last().expect("history is non-empty");
        assert_eq!(
            top.repetition_times, 3,
            "the fourth occurrence chains repetition_times to 3",
        );
        assert!(
            top.repetition < 0,
            "a forced fourfold marks repetition negative (got {})",
            top.repetition,
        );

        // Because `repetition` is negative it is `< ply` for *every* search ply,
        // so the draw is reported even when all earlier occurrences lie before
        // the search root (the game-history repetition the previous ply-bounded
        // design could not see). `ply == 1` places the entire cycle before the
        // root.
        assert_eq!(
            p.is_repetition(1),
            RepetitionState::Draw,
            "a forced fourfold is reported regardless of the search ply",
        );
    }

    #[cfg(not(feature = "quick-draw"))]
    #[test]
    fn is_repetition_two_fold_before_root_is_not_reported() {
        // One four-ply cycle: the root recurs once, `repetition == 4` (positive,
        // second occurrence). At a search ply that places the earlier occurrence
        // at or before the root (`4 <= ply` is false, i.e. ply <= 4), the
        // reference does not score it as an immediate draw — only the fourfold
        // path crosses the root.
        let mut p = setup_king_shuffle_pos();
        for m in shuffle_cycle() {
            p.do_move(m);
        }
        let top = p.history.last().expect("history is non-empty");
        assert_eq!(top.repetition, 4, "second occurrence: positive distance");
        assert_eq!(top.repetition_times, 1);

        assert_eq!(
            p.is_repetition(4),
            RepetitionState::None,
            "a two-fold reaching the root (ply == distance) is not an immediate draw",
        );
        assert_eq!(
            p.is_repetition(5),
            RepetitionState::Draw,
            "with the earlier occurrence strictly after the root it is a draw",
        );
    }

    #[cfg(feature = "quick-draw")]
    #[test]
    fn is_repetition_two_fold_adjudicates_immediately_under_quick_draw() {
        // One four-ply cycle, i.e. the *second* occurrence of the root. Under
        // QUICK_DRAW that already adjudicates — there is no fourfold requirement
        // and no root-distance gate — so it is a draw at every search ply,
        // including the plies the non-QUICK_DRAW gate suppresses (ply <= 4).
        let mut p = setup_king_shuffle_pos();
        for m in shuffle_cycle() {
            p.do_move(m);
        }

        for ply in [0u16, 1, 4, 5, 16] {
            assert_eq!(
                p.is_repetition(ply),
                RepetitionState::Draw,
                "the 2nd occurrence adjudicates under QUICK_DRAW (ply {ply})",
            );
        }
    }

    #[cfg(feature = "quick-draw")]
    #[test]
    fn is_repetition_inferior_reaches_back_to_the_root_under_quick_draw() {
        // The divergence this configuration exists to close: a
        // four-ply line whose end position repeats the *root* board with the
        // side to move's hand strictly poorer. `repetition == 4`, so the
        // non-QUICK_DRAW gate suppresses it at search ply 4 (`4 < 4` is false)
        // while QUICK_DRAW adjudicates INFERIOR straight away.
        let root = superiority_state(1, 0);
        let filler = superiority_state(9, 9);
        let current = superiority_state(0, 1);
        let p = assemble(
            superiority_board(),
            &[filler.clone(), filler.clone(), filler, current],
            root,
        );
        assert_eq!(
            p.is_repetition(4),
            RepetitionState::Inferior,
            "an inferior recurrence landing on the root is reported at ply == distance",
        );
    }

    #[test]
    fn plies_from_null_blocks_repetition_detection_across_a_null_move() {
        // A four-ply king shuffle that returns to the root — ordinarily a draw.
        let mut p = setup_king_shuffle_pos();
        for m in shuffle_cycle() {
            p.do_move(m);
        }
        assert_eq!(
            p.is_repetition(16),
            RepetitionState::Draw,
            "the closed four-ply cycle is an ordinary draw",
        );

        // Pass with two null moves: the board returns to the root (Black to
        // move) yet again — identical to the position two and six plies back —
        // but `do_null_move` reset `plies_from_null` to 0, collapsing the
        // look-back window. The reference never looks back across a null move,
        // so the repetition is invisible.
        p.do_null_move(); // Black passes
        p.do_null_move(); // White passes → root board, Black to move
        assert_eq!(
            p.current_plies_from_null(),
            0,
            "a null move resets plies_from_null",
        );
        assert_eq!(
            p.is_repetition(16),
            RepetitionState::None,
            "plies_from_null == 0 blocks detection across the null move",
        );

        // Replaying the cycle after the nulls rebuilds the window one ply at a
        // time; detection resumes only once four real plies have accumulated,
        // and reaches back only to the post-null occurrence.
        for m in shuffle_cycle() {
            p.do_move(m);
        }
        assert_eq!(
            p.is_repetition(16),
            RepetitionState::Draw,
            "four real plies past the null restore detection",
        );
    }

    #[test]
    fn gives_check_matches_stored_history_flag_over_playouts() {
        // The oracle: `Position::gives_check(m)` must equal the post-`do_move`
        // `gives_check` flag the move-history machinery records.
        // This test lives here (not in `search_movegen`) so it can read the
        // private `history`. It compares, for every legal move over seeded
        // playouts from each perft fixture, `gives_check(m)` against the flag
        // stored by `do_move(m)`.
        const MIN_PLIES: usize = 30;
        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
            let mut rng = Rng(0x2360_ED05_1FC6_5DA4 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, Undo)> = Vec::new();
            let mut legal: Vec<Move> = Vec::new();
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                legal.clear();
                pos.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    while let Some((m, u)) = stack.pop() {
                        pos.undo_move(m, u);
                    }
                    continue;
                }
                for &m in &legal {
                    let predicted = pos.gives_check(m);
                    let u = pos.do_move(m);
                    let stored = pos.history.last().unwrap().gives_check;
                    pos.undo_move(m, u);
                    assert_eq!(
                        predicted, stored,
                        "fixture {fi} ply {plies}: gives_check(m) != stored flag",
                    );
                }
                let m = legal[rng.pick(legal.len())];
                let u = pos.do_move(m);
                stack.push((m, u));
                plies += 1;
            }
        }
    }

    #[test]
    fn is_repetition_survives_do_undo_round_trip() {
        // Record is_repetition at every ply on the way down the sennichite
        // line, then unwind and assert each restored depth reports the same
        // state — history + root + board_key restoration must be exact.
        let (sfen, moves) = load_sennichite();
        let mut pos = parse_sfen(&sfen).expect("fixture sfen parses");
        let mut stack: Vec<(Move, Undo)> = Vec::new();
        let mut forward: Vec<RepetitionState> = Vec::new();
        for usi in &moves {
            let m = crate::move_::parse_usi_move(usi, &pos).expect("fixture move parses");
            let u = pos.do_move(m);
            forward.push(pos.is_repetition(16));
            stack.push((m, u));
        }

        while let Some((m, u)) = stack.pop() {
            pos.undo_move(m, u);
            let depth = stack.len();
            let expected = if depth >= 1 {
                forward[depth - 1]
            } else {
                RepetitionState::None
            };
            assert_eq!(
                pos.is_repetition(16),
                expected,
                "is_repetition mismatch after unwinding to depth {depth}",
            );
        }
    }

    /// Assert two positions are identical element-by-element: the squares
    /// array, the derived piece sets (`occupied` / `by_color` / `by_pattern` —
    /// which `Board`'s `PartialEq` deliberately excludes), every key, side to
    /// move, ply, hands, the lookback root, and the repetition-relevant history
    /// vector. This is the round-trip oracle for the board copy's removal:
    /// `undo_move` reverse-applies through `Board::set`, so the piece sets and
    /// incremental keys must land back exactly where the pre-move snapshot had
    /// them.
    fn assert_positions_identical(a: &Position, b: &Position, ctx: &str) {
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            assert_eq!(a.board().get(sq), b.board().get(sq), "{ctx}: square {sq:?}");
        }
        // Derived piece sets, compared explicitly (Board::eq skips them).
        assert_eq!(
            a.board().occupied(),
            b.board().occupied(),
            "{ctx}: occupied set",
        );
        for color in [Color::Black, Color::White] {
            assert_eq!(
                a.board().pieces_color(color),
                b.board().pieces_color(color),
                "{ctx}: by_color {color:?}",
            );
            for pat in 0..crate::board::PATTERN_COUNT {
                assert_eq!(
                    a.board().pieces_pattern(color, pat),
                    b.board().pieces_pattern(color, pat),
                    "{ctx}: by_pattern {color:?} {pat}",
                );
            }
        }
        // Every key.
        assert_eq!(a.key(), b.key(), "{ctx}: key");
        assert_eq!(a.board_key(), b.board_key(), "{ctx}: board_key");
        assert_eq!(a.hand_key(), b.hand_key(), "{ctx}: hand_key");
        assert_eq!(a.pawn_key(), b.pawn_key(), "{ctx}: pawn_key");
        assert_eq!(
            a.minor_piece_key(),
            b.minor_piece_key(),
            "{ctx}: minor_piece_key",
        );
        for color in [Color::Black, Color::White] {
            assert_eq!(
                a.non_pawn_key(color),
                b.non_pawn_key(color),
                "{ctx}: non_pawn_key {color:?}",
            );
        }
        // Scalars, hands, and the repetition-relevant history vector.
        assert_eq!(a.side_to_move(), b.side_to_move(), "{ctx}: side_to_move");
        assert_eq!(a.ply(), b.ply(), "{ctx}: ply");
        for color in [Color::Black, Color::White] {
            assert_eq!(a.hand(color), b.hand(color), "{ctx}: hand {color:?}");
        }
        // NOTE: `root` is deliberately NOT compared. It is lazy lookback state
        // captured on the first move from an empty history and intentionally
        // left populated across an undo — the same asymmetry `Position`'s
        // `PartialEq` excludes so a move-then-undo round trip still equals its
        // untouched snapshot. Every repetition-relevant field lives in the
        // `history` vector, compared above.
        assert_eq!(a.history, b.history, "{ctx}: history");
    }

    #[test]
    fn do_undo_round_trip_is_identity_over_playouts() {
        // The playout leg: from every parity fixture, walk a deterministic
        // >= 40-ply line and, at each position, prove that `do_move` then the
        // matching `undo_move` restores a Position identical to a pre-move deep
        // snapshot — element by element, piece sets and keys included. A null
        // move do/undo is exercised at every not-in-check position too (Step 9
        // is only reached out of check).
        const MIN_PLIES: usize = 40;

        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
            let mut rng = Rng(0x1D87_2B41_9C6F_00A5 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, Undo)> = Vec::new();
            let mut legal: Vec<Move> = Vec::new();

            let mut plies = 0usize;
            while plies < MIN_PLIES {
                // Null move round-trip at every not-in-check position.
                if !pos.in_check() {
                    let snapshot = pos.clone();
                    pos.do_null_move();
                    pos.undo_null_move();
                    assert_positions_identical(
                        &pos,
                        &snapshot,
                        &format!("fixture {fi} ply {plies} null round-trip"),
                    );
                }

                legal.clear();
                pos.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    // Terminal: unwind fully (also exercising undo) and restart.
                    while let Some((m, u)) = stack.pop() {
                        pos.undo_move(m, u);
                    }
                    continue;
                }

                let m = legal[rng.pick(legal.len())];

                // Do/undo round-trip identity against a pre-move deep snapshot.
                let snapshot = pos.clone();
                let u = pos.do_move(m);
                pos.undo_move(m, u);
                assert_positions_identical(
                    &pos,
                    &snapshot,
                    &format!("fixture {fi} ply {plies} do/undo round-trip"),
                );

                // Advance the playout for real.
                let u = pos.do_move(m);
                stack.push((m, u));
                plies += 1;
            }

            // Full unwind at the end, asserting identity back to each snapshot
            // is unnecessary here (covered above); just exercise the undo path.
            while let Some((m, u)) = stack.pop() {
                pos.undo_move(m, u);
            }
        }
    }
}
