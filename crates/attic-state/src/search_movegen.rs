//! Search-oriented move generation and predicates: the reference-ordered
//! `CAPTURES` / `EVASIONS` candidate lists the qsearch `MovePicker` consumes,
//! plus the `gives_check` / `in_check` / `is_legal` predicates it (and qsearch)
//! consult.
//!
//! # The generators, and the unified legal list built on them
//!
//! The per-stage generators ([`Position::generate_captures`] /
//! [`Position::generate_quiets`] / [`Position::generate_evasions`] /
//! [`Position::generate_non_evasions`]) emit **pseudo-legal** candidates in the
//! reference `MovePicker` order — the `MoveList<CAPTURES>` / `MoveList<EVASIONS>`
//! / `MoveList<NON_EVASIONS>` produced by upstream's `generate_general` /
//! `generate_evasions` (`source/movegen.cpp`),
//! piece-type-major with ascending from- then to-square and the
//! `make_move_target` promotion/non-promotion suppression. Because move
//! ordering drives search cutoffs, node-count parity depends on reproducing
//! that emission order, so this module ports it directly.
//!
//! The single all-legal entry point, [`Position::generate_legal_all`], is built
//! on the same generators — it is the `generate<LEGAL_ALL>` port
//! (`movegen.cpp`): pick the check-state generator (evasions when in
//! check, else non-evasions) with `all == true`, then retain the candidates
//! passing [`Position::is_legal`]. It is the sole legal-move generator in the
//! engine; perft, USI `position ... moves` validation, book widening, and the
//! test oracles all route through it. It is **repetition-blind**, exactly like
//! the reference (no sennichite / perpetual-check term).
//!
//! The per-stage generators produce **pseudo-legal** moves (king-safety is not
//! checked), exactly like the reference's `MoveList`. The `MovePicker` filters
//! each with [`Position::is_legal`] as it yields it — equivalent to the
//! reference qsearch skipping illegal moves without counting a node.
//!
//! # `CAPTURES` at the pin (`generate_general<CAPTURES, Us, false>`)
//!
//! `target = pos.pieces(Them)` — destinations are exactly the enemy-occupied
//! squares (`movegen.cpp`); there are no drops (`movegen.cpp`, drops
//! are only generated for `QUIETS` / `NON_EVASIONS`) and no non-capturing pawn
//! promotions (that is `CAPTURES_PRO_PLUS`, a different generator). Piece-type
//! order (`movegen.cpp`): PAWN, LANCE, KNIGHT, SILVER, then
//! bishop+rook (`GPM_BR`, unpromoted only, interleaved by square), then
//! gold-likes+horse+dragon+king (`GPM_GHDK`, interleaved by square).
//!
//! With `All == false` (the option default, `generate_all_legal_moves`), the
//! `make_move_target` rules (`movegen.cpp`) suppress various
//! non-promoting variants:
//!
//! * PAWN — inside the promotion zone only the promotion is generated
//!   (`movegen.cpp`).
//! * LANCE — every capture into the enemy field promotes; the non-promotion is
//!   suppressed on the enemy first/second rank (kept from the third rank back)
//!   (`movegen.cpp`).
//! * KNIGHT — promotes where it can; the non-promotion is suppressed on the
//!   enemy first/second ranks where a non-promoted knight would be stuck
//!   (`movegen.cpp`).
//! * SILVER — always emits both promotion and non-promotion when a promotion
//!   is available (`movegen.cpp`).
//! * BISHOP / ROOK — promote-only when entering or leaving the zone; the
//!   non-promotion is `All`-only (`movegen.cpp`).
//! * GOLD-likes / horse / dragon / king — never promote (`movegen.cpp`).
//!
//! # `EVASIONS` at the pin (`generate_evasions<Us, false>`)
//!
//! King moves first (`movegen.cpp`), then — for a single check —
//! non-king moves in the same piece-type order as `CAPTURES` (but the
//! gold-group excludes the king, already emitted) restricted to the
//! block/capture squares, then interposition drops (`movegen.cpp`).
//! This port target-restricts generation exactly as the pin does:
//! the king steps are pre-masked by the union of every checker's attack rays
//! (`sliderAttacks`); on a single check the non-king pieces are masked to
//! `target2 = between(checksq, ksq) | checksq` (interpose or capture the
//! checker) and the drops to `target1 = between(checksq, ksq)` (interpose only);
//! on a double check only king moves are emitted, matching `checkersCnt <= 1`
//! (`movegen.cpp`). Every emitted move is a subsequence of what an
//! unrestricted emission would produce, so the [`Position::is_legal`]-filtered
//! output — and its order — is byte-identical to generating everything and then
//! filtering; the remaining suicide king steps (ray squares behind the king) are
//! caught by `is_legal`, as the pin's comment notes.
//!
//! One deliberate, output-preserving deviation from the pin: `sliderAttacks` is
//! accumulated under the *current* occupancy (king still on the board) rather
//! than the pin's king-removed `occ = pieces() ^ ksq`. With the king in the
//! occupancy a slider's ray stops at the king, so the square directly behind the
//! king survives the king-step mask instead of being pruned at generation; it is
//! a suicide step, which `is_legal` (vacating the king's from-square) rejects
//! anyway. Both forms therefore yield the identical legal king-move set, in the
//! identical ascending order.

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::board::pat;
use crate::color::Color;
use crate::move_::Move;
#[cfg(test)]
use crate::movegen::{dr_sign_for, movement, step_signed};
use crate::movegen::{is_attacked_by, is_in_promotion_zone, try_find_king};
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;

/// A move paired with its ordering score, matching the reference `ExtMove`
/// (`struct ExtMove : public Move`, `movegen.h`). The search-side generators
/// emit `ExtMove` **directly** into the `MovePicker`'s buffer — `value` is
/// initialised to `0` and the picker fills it in later at its scoring stage —
/// mirroring the reference's `ExtMove* generateMoves(const Position&, ExtMove*)`
/// layering (`movegen.h`), which writes the picker's array in place. There
/// is therefore no intermediate `Move` → `ExtMove` restaging pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExtMove {
    pub mv: Move,
    pub value: i32,
}

/// `#[cfg(test)]` scanning oracles for the piece-set move generators:
/// scan-based twins of the four search generators, built from
/// [`emit_group_scan`] / [`reachable_scan`], against which the production
/// generators are pinned move-for-move by the sequence-equality gate.
#[cfg(test)]
mod scan_oracle;

/// The seven droppable piece kinds in the order the reference's
/// `GenerateDropMoves` lays them into its `drops[]` array
/// (`movegen.cpp`): knight, lance, then silver, gold, bishop, rook.
/// Pawn drops are generated separately (and first).
const DROP_ORDER: [PieceKind; 6] = [
    PieceKind::Knight,
    PieceKind::Lance,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Apply `m` to a bare `board` for `mover`, mirroring the board half of
/// [`Position::do_move`] (no keys / hands / history). Used only by the
/// `#[cfg(test)]` copy-apply-scan oracles (`leaves_own_king_safe`,
/// `gives_check_reference`, `is_legal_reference`), which need the resulting
/// occupancy; the production predicates are board-mutation-free.
#[cfg(test)]
fn apply_to_board(board: &mut Board, m: Move, mover: Color) {
    let to = m.to_sq();
    if m.is_drop() {
        board.set(to, Some(Piece::new(m.dropped_piece_kind(), mover)));
    } else {
        board.set(to, Some(m.moved_piece_after()));
        board.set(m.from_sq(), None);
    }
}

/// The all-81-squares mask — the identity for the evasion destination
/// restriction, passed by every generator that imposes no `target2` / `target1`
/// mask (`CAPTURES`, `QUIETS`, `NON_EVASIONS`, and the unrestricted scan twins).
const ALL_SQUARES: Bitboard = Bitboard::FULL;

/// Which destination squares a move generator keeps, mirroring the reference
/// `generate_general` `target` bitboard: `CAPTURES` = enemy-occupied squares
/// only, `QUIETS` = empty squares only, and the `EVASIONS` block-or-capture
/// target = both (empty interposition squares and the checker's square).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    /// `pos.pieces(Them)` — enemy pieces only (`CAPTURES`).
    Captures,
    /// `pos.empties()` — empty squares only (`QUIETS`).
    Quiets,
    /// Empty *and* enemy squares — the `EVASIONS` block-or-capture set.
    BlockOrCapture,
}

/// The pseudo-legal destinations of the piece standing on `from`, ascending by
/// square index (mirroring the reference's `bb.pop()` / `foreach` iteration,
/// least-significant square first). `target` selects which squares are kept
/// (see [`Target`]).
///
/// Reads the piece's attack set straight off the occupancy-limited bitboard
/// queries ([`attacks_bb`]) instead of re-walking the movement rays, then masks
/// it with the piece sets: `Captures` keeps the enemy-occupied destinations,
/// `Quiets` the empty ones, and `BlockOrCapture` every non-own-occupied
/// destination (empty interposition squares plus enemy captures). The attack set
/// already stops each slider at (and including) the first blocker, so an
/// own-piece blocker square is present in the raw set and dropped by every mask.
/// The masked bitboard is returned directly (no intermediate `Vec<Square>`); its
/// `.squares()` yields the survivors in ascending-index order, and the emit
/// helpers iterate it in place. An independent movement-walk form is kept as the
/// `#[cfg(test)]` oracle [`reachable_scan`], and the sequence-equality test pins
/// the two together.
fn reachable(board: &Board, from: Square, piece: Piece, target: Target) -> Bitboard {
    let occ = board.occupied();
    let attacks = attacks_bb(from, piece, occ);
    match target {
        Target::Captures => attacks & board.pieces_color(piece.color.flip()),
        Target::Quiets => attacks & !occ,
        Target::BlockOrCapture => attacks & !board.pieces_color(piece.color),
    }
}

/// A movement-walk form of [`reachable`], derived independently of the bitboard
/// tables so it can serve as the sequence-equivalence oracle for the emitters.
#[cfg(test)]
fn reachable_scan(board: &Board, from: Square, piece: Piece, target: Target) -> Vec<Square> {
    let mut out: Vec<Square> = Vec::new();
    let dr_sign = dr_sign_for(piece.color);
    let (steps, slides) = movement(piece);
    for &(df, dr) in steps {
        if let Some(to) = step_signed(from, df, dr * dr_sign) {
            match board.get(to) {
                None => {
                    if target != Target::Captures {
                        out.push(to);
                    }
                }
                Some(p) => {
                    if p.color != piece.color && target != Target::Quiets {
                        out.push(to);
                    }
                }
            }
        }
    }
    for &(df, dr) in slides {
        let mut cur = from;
        loop {
            cur = match step_signed(cur, df, dr * dr_sign) {
                Some(s) => s,
                None => break,
            };
            match board.get(cur) {
                None => {
                    if target != Target::Captures {
                        out.push(cur);
                    }
                }
                Some(p) => {
                    if p.color != piece.color && target != Target::Quiets {
                        out.push(cur);
                    }
                    break;
                }
            }
        }
    }
    out.sort_unstable_by_key(|s| s.index());
    out
}

fn push_promote(from: Square, to: Square, piece: Piece, out: &mut Vec<ExtMove>) {
    out.push(ExtMove {
        mv: Move::make_promote(from, to, piece),
        value: 0,
    });
}

fn push_plain(from: Square, to: Square, piece: Piece, out: &mut Vec<ExtMove>) {
    out.push(ExtMove {
        mv: Move::make(from, to, piece),
        value: 0,
    });
}

/// The last rank a piece of `color` can reach — where a pawn / lance
/// non-promotion is always illegal (Black rank 0, White rank 8).
fn last_rank(color: Color) -> u8 {
    match color {
        Color::Black => 0,
        Color::White => 8,
    }
}

/// Non-promotion is allowed for a knight landing on `to`: suppressed on the
/// enemy first and second ranks (Black `rank >= 2`, White `rank <= 6`), where a
/// non-promoted knight would be stuck. Mirrors the knight guard
/// (`movegen.cpp`); this is `All`-independent (a stuck knight is never
/// generated regardless of the option).
fn nonpromote_rank_ok(to: Square, color: Color) -> bool {
    match color {
        Color::Black => to.rank() >= 2,
        Color::White => to.rank() <= 6,
    }
}

/// Non-promotion is allowed for a lance landing on `to`. With `All == false`
/// (the default) the non-promotion is suppressed on the enemy first and second
/// ranks (`movegen.cpp`, `ForwardRanksBB[…][RANK_2]`), coinciding with the
/// knight mask. With `All == true` only the very last rank — where the lance
/// would be stuck — is suppressed (`movegen.cpp`, `ForwardRanksBB[…][RANK_1]`),
/// so the enemy second rank's non-promotion is additionally generated.
fn lance_nonpromote_rank_ok(to: Square, color: Color, all: bool) -> bool {
    if all {
        match color {
            Color::Black => to.rank() >= 1,
            Color::White => to.rank() <= 7,
        }
    } else {
        nonpromote_rank_ok(to, color)
    }
}

/// Emit a pawn's move to its single forward `to` (`movegen.cpp`). With
/// `All == true` a pawn pushing into the promotion zone additionally emits its
/// non-promotion, except on the last rank where it would be stuck (`249`).
fn emit_pawn(from: Square, targets: Bitboard, piece: Piece, all: bool, out: &mut Vec<ExtMove>) {
    for to in targets.squares() {
        if is_in_promotion_zone(to, piece.color) {
            push_promote(from, to, piece, out);
            if all && to.rank() != last_rank(piece.color) {
                push_plain(from, to, piece, out);
            }
        } else {
            push_plain(from, to, piece, out);
        }
    }
}

/// Emit a lance's moves: all promotions into the enemy field first, then the
/// rank-masked non-promotions (`movegen.cpp`). The non-promotion rank mask
/// widens under `All` (see [`lance_nonpromote_rank_ok`]).
fn emit_lance(from: Square, targets: Bitboard, piece: Piece, all: bool, out: &mut Vec<ExtMove>) {
    for to in targets.squares() {
        if is_in_promotion_zone(to, piece.color) {
            push_promote(from, to, piece, out);
        }
    }
    for to in targets.squares() {
        if lance_nonpromote_rank_ok(to, piece.color, all) {
            push_plain(from, to, piece, out);
        }
    }
}

/// Emit a knight's moves, per destination: promotion where legal, then the
/// non-promotion where a non-promoted knight is not stuck (`movegen.cpp`).
/// `All`-independent: the reference knight guard carries no `All` term.
fn emit_knight(from: Square, targets: Bitboard, piece: Piece, out: &mut Vec<ExtMove>) {
    for to in targets.squares() {
        if is_in_promotion_zone(to, piece.color) {
            push_promote(from, to, piece, out);
        }
        if nonpromote_rank_ok(to, piece.color) {
            push_plain(from, to, piece, out);
        }
    }
}

/// Emit a silver's moves (`movegen.cpp`). A silver always keeps its
/// non-promotion; it additionally promotes whenever a promotion is available
/// (`from` or `to` in the enemy field). When `from` is not in the enemy field
/// the reference emits the promotable (into-zone) destinations first, then the
/// plain ones — reproduced here as two passes.
fn emit_silver(from: Square, targets: Bitboard, piece: Piece, out: &mut Vec<ExtMove>) {
    if is_in_promotion_zone(from, piece.color) {
        for to in targets.squares() {
            push_promote(from, to, piece, out);
            push_plain(from, to, piece, out);
        }
    } else {
        for to in targets.squares() {
            if is_in_promotion_zone(to, piece.color) {
                push_promote(from, to, piece, out);
                push_plain(from, to, piece, out);
            }
        }
        for to in targets.squares() {
            if !is_in_promotion_zone(to, piece.color) {
                push_plain(from, to, piece, out);
            }
        }
    }
}

/// Emit a bishop's / rook's moves (`movegen.cpp`). These always promote
/// when they can (`from` or `to` in the enemy field); the non-promotion is
/// `All`-only. With `All == false` it is emitted only for moves that stay
/// entirely outside the zone; with `All == true` the in-zone non-promotion is
/// additionally emitted, interleaved right after each promotion (matching the
/// reference `if (All) *mlist++ = make_move(...)` per destination, `154`/`162`).
fn emit_bishop_rook(
    from: Square,
    targets: Bitboard,
    piece: Piece,
    all: bool,
    out: &mut Vec<ExtMove>,
) {
    if is_in_promotion_zone(from, piece.color) {
        for to in targets.squares() {
            push_promote(from, to, piece, out);
            if all {
                push_plain(from, to, piece, out);
            }
        }
    } else {
        for to in targets.squares() {
            if is_in_promotion_zone(to, piece.color) {
                push_promote(from, to, piece, out);
                if all {
                    push_plain(from, to, piece, out);
                }
            }
        }
        for to in targets.squares() {
            if !is_in_promotion_zone(to, piece.color) {
                push_plain(from, to, piece, out);
            }
        }
    }
}

/// Emit a never-promoting piece's moves (gold-likes, horse, dragon, king)
/// (`movegen.cpp`).
fn emit_plain_only(from: Square, targets: Bitboard, piece: Piece, out: &mut Vec<ExtMove>) {
    for to in targets.squares() {
        push_plain(from, to, piece, out);
    }
}

/// The reference `generate_general` piece groups, in emission order. `king`
/// records whether the gold-group carries the king: it does for `CAPTURES`
/// (`GPM_GHDK`), but not for `EVASIONS` (`GPM_GHD` — the king was emitted
/// first).
///
/// This is the runtime, `match`-dispatched form. The hot search generators no
/// longer use it — they name a compile-time [`GroupSpec`] type so `pieces` /
/// `emit` resolve statically (see that trait). `Group` survives only as the
/// `#[cfg(test)]` membership / emit oracle for the scanning [`emit_group_scan`].
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Group {
    Pawn,
    Lance,
    Knight,
    Silver,
    BishopRook,
    GoldHdk { king: bool },
}

#[cfg(test)]
impl Group {
    /// Does `piece` (already known to be the side to move's) belong to this
    /// group? The groups partition the moving side's pieces exactly as the
    /// reference bitboards `pieces(Us, PAWN)`, …, `pieces(Us, BISHOP, ROOK)`,
    /// `pieces(Us, GOLDS, HDK)` do — the same partition [`group_pieces`] reads
    /// off the board's pattern sets. Retained as the membership oracle for the
    /// scanning [`emit_group_scan`].
    #[cfg(test)]
    fn contains(self, piece: Piece) -> bool {
        match self {
            Group::Pawn => piece.kind == PieceKind::Pawn && !piece.promoted,
            Group::Lance => piece.kind == PieceKind::Lance && !piece.promoted,
            Group::Knight => piece.kind == PieceKind::Knight && !piece.promoted,
            Group::Silver => piece.kind == PieceKind::Silver && !piece.promoted,
            Group::BishopRook => {
                matches!(piece.kind, PieceKind::Bishop | PieceKind::Rook) && !piece.promoted
            }
            Group::GoldHdk { king } => {
                if piece.kind == PieceKind::King {
                    king
                } else {
                    piece.promoted || piece.kind == PieceKind::Gold
                }
            }
        }
    }

    fn emit(
        self,
        from: Square,
        targets: Bitboard,
        piece: Piece,
        all: bool,
        out: &mut Vec<ExtMove>,
    ) {
        match self {
            Group::Pawn => emit_pawn(from, targets, piece, all, out),
            Group::Lance => emit_lance(from, targets, piece, all, out),
            Group::Knight => emit_knight(from, targets, piece, out),
            Group::Silver => emit_silver(from, targets, piece, out),
            Group::BishopRook => emit_bishop_rook(from, targets, piece, all, out),
            Group::GoldHdk { .. } => emit_plain_only(from, targets, piece, out),
        }
    }
}

/// Compile-time specialization of one piece group's emit pipeline. Each search
/// generator names a concrete zero-sized `GroupSpec` type per call site, so the
/// piece-set read ([`GroupSpec::pieces`]) and the per-piece emitter
/// ([`GroupSpec::emit`]) resolve to a direct call inside each monomorphized
/// [`emit_group`] / [`emit_group_masked`] instantiation — no runtime `match` on a
/// [`Group`] value, hence no jump-table load and no indirect branch in the
/// per-from-square loop. This mirrors the reference's template specializations
/// `GeneratePieceMoves<…, Pt, …>` / `make_move_target<Pt, …>`
/// (`movegen.cpp`), as opposed to the runtime `make_move_target_general`
/// switch (`movegen.cpp`) the hot generators never take. The runtime `Group`
/// enum survives only as the `#[cfg(test)]` oracle for [`emit_group_scan`].
trait GroupSpec {
    /// The side-to-move's pieces belonging to this group, as a bitboard read
    /// from the board's incrementally maintained pattern sets — the reference
    /// bitboards `pieces(Us, PAWN)`, …, `pieces(Us, BISHOP, ROOK)`,
    /// `pieces(Us, GOLDS, HDK)`. Iterated ascending it yields the same
    /// from-squares, in the same order, as a 0..81 scan filtered by
    /// `Group::contains`: each pattern slot holds exactly the pieces of that
    /// group (the four promoted minors collapse into the GOLD slot, horse /
    /// dragon are distinct, the king is its own slot).
    fn pieces(board: &Board, stm: Color) -> Bitboard;

    /// Emit one `from`-square piece's pseudo-moves onto `targets` — the static
    /// counterpart of `Group::emit`, resolving to the single per-piece emitter.
    fn emit(from: Square, targets: Bitboard, piece: Piece, all: bool, out: &mut Vec<ExtMove>);
}

/// Zero-sized group markers, one per emission group. `GoldHdk` is const-generic
/// over whether the group carries the king (it does for `CAPTURES` / `QUIETS` /
/// `NON_EVASIONS`, the `GPM_GHDK` case, but not for `EVASIONS`, the `GPM_GHD`
/// case where the king was emitted first).
enum PawnG {}
enum LanceG {}
enum KnightG {}
enum SilverG {}
enum BishopRookG {}
enum GoldHdkG<const KING: bool> {}

impl GroupSpec for PawnG {
    fn pieces(board: &Board, stm: Color) -> Bitboard {
        board.pieces_pattern(stm, pat::PAWN)
    }
    fn emit(from: Square, targets: Bitboard, piece: Piece, all: bool, out: &mut Vec<ExtMove>) {
        emit_pawn(from, targets, piece, all, out);
    }
}

impl GroupSpec for LanceG {
    fn pieces(board: &Board, stm: Color) -> Bitboard {
        board.pieces_pattern(stm, pat::LANCE)
    }
    fn emit(from: Square, targets: Bitboard, piece: Piece, all: bool, out: &mut Vec<ExtMove>) {
        emit_lance(from, targets, piece, all, out);
    }
}

impl GroupSpec for KnightG {
    fn pieces(board: &Board, stm: Color) -> Bitboard {
        board.pieces_pattern(stm, pat::KNIGHT)
    }
    fn emit(from: Square, targets: Bitboard, piece: Piece, _all: bool, out: &mut Vec<ExtMove>) {
        emit_knight(from, targets, piece, out);
    }
}

impl GroupSpec for SilverG {
    fn pieces(board: &Board, stm: Color) -> Bitboard {
        board.pieces_pattern(stm, pat::SILVER)
    }
    fn emit(from: Square, targets: Bitboard, piece: Piece, _all: bool, out: &mut Vec<ExtMove>) {
        emit_silver(from, targets, piece, out);
    }
}

impl GroupSpec for BishopRookG {
    fn pieces(board: &Board, stm: Color) -> Bitboard {
        board.pieces_pattern(stm, pat::BISHOP) | board.pieces_pattern(stm, pat::ROOK)
    }
    fn emit(from: Square, targets: Bitboard, piece: Piece, all: bool, out: &mut Vec<ExtMove>) {
        emit_bishop_rook(from, targets, piece, all, out);
    }
}

impl<const KING: bool> GroupSpec for GoldHdkG<KING> {
    fn pieces(board: &Board, stm: Color) -> Bitboard {
        let mut bb = board.pieces_pattern(stm, pat::GOLD)
            | board.pieces_pattern(stm, pat::HORSE)
            | board.pieces_pattern(stm, pat::DRAGON);
        if KING {
            bb |= board.pieces_pattern(stm, pat::KING);
        }
        bb
    }
    fn emit(from: Square, targets: Bitboard, piece: Piece, _all: bool, out: &mut Vec<ExtMove>) {
        emit_plain_only(from, targets, piece, out);
    }
}

/// Iterate the side-to-move's group-`G` pieces by ascending square and emit their
/// pseudo-moves onto the `target` squares (see [`Target`]). `all` is the
/// `GenerateAllLegalMoves` flag, widening the suppressed non-promotions. The
/// generating pieces come from [`GroupSpec::pieces`] (the board's piece sets)
/// rather than an 81-square scan; the scan form is kept as the sequence oracle
/// [`emit_group_scan`].
fn emit_group<G: GroupSpec>(
    board: &Board,
    stm: Color,
    target: Target,
    all: bool,
    out: &mut Vec<ExtMove>,
) {
    emit_group_masked::<G>(board, stm, target, ALL_SQUARES, all, out);
}

/// [`emit_group`] with the reachable destinations additionally intersected with
/// `restrict` — the reference `target2` mask the evasion generator threads in
/// (interpose-or-capture-the-checker). The non-evasion generators pass
/// [`ALL_SQUARES`], an identity intersection. Because `restrict` only removes
/// destinations, the emitted moves stay a subsequence of the unrestricted
/// emission, preserving the ascending square / promotion order.
fn emit_group_masked<G: GroupSpec>(
    board: &Board,
    stm: Color,
    target: Target,
    restrict: Bitboard,
    all: bool,
    out: &mut Vec<ExtMove>,
) {
    for from in G::pieces(board, stm).squares() {
        let piece = board
            .get(from)
            .expect("group_pieces bit implies a piece stands on the square");
        let targets = reachable(board, from, piece, target) & restrict;
        G::emit(from, targets, piece, all, out);
    }
}

/// A 0..81 scan form of [`emit_group`], the sequence oracle for the piece-set
/// emitter. Uses [`reachable_scan`] so it shares no code with the production
/// destination path.
#[cfg(test)]
fn emit_group_scan(
    board: &Board,
    stm: Color,
    group: Group,
    target: Target,
    all: bool,
    out: &mut Vec<ExtMove>,
) {
    for index in 0..Square::COUNT as u8 {
        let from = Square::from_index(index).unwrap();
        let piece = match board.get(from) {
            Some(p) if p.color == stm && group.contains(p) => p,
            _ => continue,
        };
        let mut targets = Bitboard::empty();
        for sq in reachable_scan(board, from, piece, target) {
            targets.set(sq);
        }
        group.emit(from, targets, piece, all, out);
    }
}

/// `file → the side to move already holds an own un-promoted pawn on it` — the
/// nifu (二歩) mask. Reads the un-promoted-pawn pattern set (a promoted pawn /
/// tokin lives in the GOLD slot and does not count) ANDed against each file
/// mask. [`Position::emit_drops_masked`] folds the same pawn bitboard into a
/// full-file exclusion mask inline; this per-file bool
/// form is a `#[cfg(test)]` helper for the drop-emitter oracle
/// [`Position::emit_drops_masked_scan`] and the equivalence gates, pinned equal
/// to the `#[cfg(test)]` scan oracle [`nifu_blocked_files_scan`] along the
/// fixture playouts.
#[cfg(test)]
fn nifu_blocked_files(board: &Board, stm: Color) -> [bool; Square::FILES as usize] {
    use crate::bitboard::file_mask;
    let own_pawns = board.pieces_pattern(stm, crate::board::pat::PAWN);
    let mut blocked = [false; Square::FILES as usize];
    for (file, b) in blocked.iter_mut().enumerate() {
        *b = !(own_pawns & file_mask(file as u8)).is_empty();
    }
    blocked
}

/// A nine-rank-per-file scan form of [`nifu_blocked_files`], the equivalence
/// oracle for the file-mask form.
#[cfg(test)]
fn nifu_blocked_files_scan(board: &Board, stm: Color) -> [bool; Square::FILES as usize] {
    let mut blocked = [false; Square::FILES as usize];
    for (file, b) in blocked.iter_mut().enumerate() {
        for rank in 0..Square::RANKS {
            let sq = Square::new(file as u8, rank).unwrap();
            if let Some(p) = board.get(sq)
                && p.color == stm
                && p.kind == PieceKind::Pawn
                && !p.promoted
            {
                *b = true;
                break;
            }
        }
    }
    blocked
}

// ===========================================================================
// Per-position-state check info (reference `Position::set_check_info`)
// ===========================================================================
//
// The reference precomputes, once per position STATE (in `do_move` /
// `set_check_info`, `position.cpp` non-STOCKFISH branch ~242-284), the data the
// per-move predicates `gives_check` / `gives_direct_check` / `legal` consult:
//
//   * `checkSquares[pt]` — the destination squares from which a piece of type
//     `pt` of the side to move would check the enemy king, under the *current*
//     occupancy. The predicates then reduce to a table lookup AND the move's
//     destination.
//   * `blockersForKing[c]` — the single blockers between `c`'s king and an
//     enemy slider (the pin set), reused verbatim from the SEE module's
//     [`crate::see::slider_blockers`] rather than duplicated.
//
// This module caches that per-state info on [`Position`] (recomputed on
// `do_move` / `undo_move` / `set`; see `position.rs`) so the three predicates
// below are constant-time lookups rather than a copy-apply-scan.

/// The number of distinct check-attack patterns keyed by [`check_pattern`].
const CHECK_PATTERN_COUNT: usize = crate::board::PATTERN_COUNT;

/// Map a concrete piece to its `checkSquares` pattern slot — the shared
/// [`crate::board::pattern_of`] partition (the four promoted minors collapse to
/// GOLD, horse / dragon are distinct, KING has its own slot), exactly the SEE
/// attacker-bucket partition and the [`crate::board`] piece-set partition.
fn check_pattern(piece: Piece) -> usize {
    crate::board::pattern_of(piece)
}

/// The set of squares `piece` (standing on `from`) attacks under occupancy
/// `occ`. Steppers land on their fixed offsets (from the foundation tables);
/// sliders walk to (and including) the first occupied square (the occupancy-
/// limited [`crate::bitboard`] queries). Used to fill `checkSquares` via the
/// reverse-attack trick and so, like the reference's `bishopEffect(ksq, occ)`
/// etc., it is computed against the current occupancy. A board-scanning form is
/// kept as the `#[cfg(test)]` oracle [`attack_set_from_scan`].
fn attacks_bb(from: Square, piece: Piece, occ: Bitboard) -> Bitboard {
    use crate::bitboard::{
        bishop_attacks, dragon_attacks, gold_attacks, horse_attacks, king_attacks, knight_attacks,
        lance_attacks, pawn_attacks, rook_attacks, silver_attacks,
    };
    let c = piece.color;
    match (piece.kind, piece.promoted) {
        (PieceKind::Pawn, false) => pawn_attacks(c, from),
        (PieceKind::Lance, false) => lance_attacks(c, from, occ),
        (PieceKind::Knight, false) => knight_attacks(c, from),
        (PieceKind::Silver, false) => silver_attacks(c, from),
        (PieceKind::Gold, _)
        | (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => {
            gold_attacks(c, from)
        }
        (PieceKind::Bishop, false) => bishop_attacks(from, occ),
        (PieceKind::Bishop, true) => horse_attacks(from, occ),
        (PieceKind::Rook, false) => rook_attacks(from, occ),
        (PieceKind::Rook, true) => dragon_attacks(from, occ),
        (PieceKind::King, _) => king_attacks(c, from),
    }
}

/// A board-scanning form of [`attacks_bb`], derived independently of the
/// bitboard tables so it can serve as their equivalence oracle.
#[cfg(test)]
fn attack_set_from_scan(board: &Board, from: Square, piece: Piece) -> Bitboard {
    let mut set = Bitboard::empty();
    let dr_sign = dr_sign_for(piece.color);
    let (steps, slides) = movement(piece);
    for &(df, dr) in steps {
        if let Some(to) = step_signed(from, df, dr * dr_sign) {
            set |= Bitboard::from_square(to);
        }
    }
    for &(df, dr) in slides {
        let mut cur = from;
        loop {
            cur = match step_signed(cur, df, dr * dr_sign) {
                Some(s) => s,
                None => break,
            };
            set |= Bitboard::from_square(cur);
            if board.get(cur).is_some() {
                break;
            }
        }
    }
    set
}

/// The unit ray direction from `king` to `sq` when the two lie on one of the
/// eight queen-lines, else `None`. Mirrors `Effect8::directions_of`. A table
/// lookup into the precomputed [`crate::bitboard`] direction table, which
/// `bitboard`'s tests pin exhaustively against the arithmetic derivation.
fn ray_dir(king: Square, sq: Square) -> Option<(i8, i8)> {
    crate::bitboard::ray_dir(king, sq)
}

/// The reference `aligned(s1, s2, ksq)` (`types.h`): are `s1` and `s2` on
/// the same ray *emanating from* `ksq` — the same one of the eight directions,
/// on the same side (a straight line passing through the king does not count)?
/// Used to decide whether moving a pinned / blocking piece stays on its pin ray
/// (so it neither exposes its own king nor un-blocks a discovered check).
fn aligned(king: Square, s1: Square, s2: Square) -> bool {
    match (ray_dir(king, s1), ray_dir(king, s2)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// The reference `between_bb(a, b)`: the squares strictly between `a` and `b`
/// when the two lie on one of the eight queen-lines, else empty. Endpoints are
/// excluded. Used by [`Position::pseudo_legal`] to test whether an in-check move
/// interposes on the checker's ray. A table lookup into the precomputed
/// [`crate::bitboard`] between-table, which `bitboard`'s tests pin exhaustively
/// against a per-call `step_signed` walk.
fn between_set(a: Square, b: Square) -> Bitboard {
    crate::bitboard::between(a, b)
}

/// The reference `is_non_promotable_piece(pc)` (`type_of(pc) >= GOLD`): a piece
/// that can never promote — gold, king, or any already-promoted piece.
fn is_non_promotable_piece(pc: Piece) -> bool {
    pc.promoted || matches!(pc.kind, PieceKind::Gold | PieceKind::King)
}

/// The per-position-state check info cached on [`Position`], ported from the
/// reference's `StateInfo::{checkSquares, blockersForKing}` (filled by
/// `Position::set_check_info`). Recomputed wherever the state changes — see
/// `Position::do_move` / `undo_move` / `refresh_keys` in `position.rs` — so the
/// predicates that read it are constant-time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CheckInfo {
    /// The `board_key` this info was computed against; a debug guard against a
    /// stale cache slipping through an unmaintained mutation path.
    pub(crate) board_key: u64,
    /// The enemy (opponent-of-side-to-move) king square, if present.
    enemy_king: Option<Square>,
    /// The side-to-move's own king square, if present.
    own_king: Option<Square>,
    /// Whether the side to move is currently in check.
    in_check: bool,
    /// `checkSquares[pattern]`: squares from which a side-to-move piece of the
    /// given [`check_pattern`] checks the enemy king, under the current
    /// occupancy.
    check_squares: [Bitboard; CHECK_PATTERN_COUNT],
    /// `blockersForKing[color]`: the pieces singly blocking a slider check on
    /// `color`'s king. Indexed by [`Color::index`].
    blockers: [Bitboard; Color::COUNT],
    /// `pinners[color]`: the enemy sliders pinning a `color`-blocker to
    /// `color`'s king. Indexed by [`Color::index`], mirroring the reference
    /// `StateInfo::pinners[2]`. `slider_blockers(c)` returns
    /// `(blockersForKing[c], pinners[~c])`, so the value returned alongside
    /// `blockersForKing[c]` is stored at `pinners[~c]`.
    pinners: [Bitboard; Color::COUNT],
    /// `checkersBB`: the squares of the enemy pieces giving check to the side to
    /// move's king. Empty unless [`Self::in_check`]. Read by
    /// [`Position::pseudo_legal`] to decide the single-checker interposition /
    /// double-check rules the reference `pseudo_legal_s` applies.
    checkers: Bitboard,
}

impl CheckInfo {
    /// The check info of the empty board (`board_key == 0`, no kings, no
    /// attacks) — the value [`Position::empty`] seeds its `check_info` field
    /// with. Matches `compute_check_info` on an empty board: both kings absent,
    /// so every `check_squares` slot, both `blockers`/`pinners`, and `checkers`
    /// are empty and `in_check` is false. Recomputed eagerly on the first state
    /// change (SFEN parse via `refresh_keys`, or `do_move`).
    pub(crate) const EMPTY: CheckInfo = CheckInfo {
        board_key: 0,
        enemy_king: None,
        own_king: None,
        in_check: false,
        check_squares: [Bitboard::EMPTY; CHECK_PATTERN_COUNT],
        blockers: [Bitboard::EMPTY; Color::COUNT],
        pinners: [Bitboard::EMPTY; Color::COUNT],
        checkers: Bitboard::EMPTY,
    };

    /// The enemy (opponent-of-side-to-move) king square, if present. Equal to
    /// `try_find_king(board, side_to_move.flip())`.
    pub(crate) fn enemy_king(&self) -> Option<Square> {
        self.enemy_king
    }

    /// The side-to-move's own king square, if present. Equal to
    /// `try_find_king(board, side_to_move)`.
    pub(crate) fn own_king(&self) -> Option<Square> {
        self.own_king
    }

    /// `blockersForKing[color]`: the pieces singly blocking a slider check on
    /// `color`'s king (the `.0` of [`crate::see::slider_blockers`]).
    pub(crate) fn blockers(&self, color: Color) -> Bitboard {
        self.blockers[color.index()]
    }

    /// `pinners[color]`: the enemy sliders pinning a `color`-blocker to
    /// `color`'s king. `slider_blockers(c)` returns `pinners[~c]` as its `.1`,
    /// so `pinners(c)` equals `slider_blockers(c.flip()).1`.
    pub(crate) fn pinners(&self, color: Color) -> Bitboard {
        self.pinners[color.index()]
    }
}

/// A by-value snapshot of a position's `checkSquares` table, the
/// cross-crate analogue of the reference `StateInfo::checkSquares`. Copied out
/// of the cached [`CheckInfo`] with a single `check_info()` borrow (mirroring
/// how the SEE blockers/pinners were lifted out of the `Ref` guard), so the
/// per-move direct-check test needs no `RefCell` borrow-flag check, no `Option`
/// probe, and no `Position` access at all.
///
/// `checkSquares` is a pure function of the position, and the position is
/// unchanged for a [`MovePicker`]'s entire lifetime (each subtree
/// `do_move`/`undo_move` pair restores the board before the next scoring call),
/// so a snapshot taken once at `QUIET_INIT` stage entry orders moves exactly as
/// the reference's per-move `pos.check_squares(pt) & to` does (`movepick.cpp`)
/// — the same output-equivalence argument documented for evaluating SEE at
/// stage entry.
#[derive(Clone, Copy, Debug)]
pub struct CheckSquares {
    /// The snapshotted `checkSquares[pattern]` table, keyed by [`check_pattern`].
    table: [Bitboard; CHECK_PATTERN_COUNT],
}

impl CheckSquares {
    /// True iff `m` gives a **direct** check by the moved piece — identical to
    /// [`Position::gives_direct_check`] but reading the snapshot table rather
    /// than re-entering the lazy `check_info()` accessor: the exact reference
    /// term `check_squares(type_of(moved_piece(m))) & to` (`movepick.cpp`).
    pub fn gives_direct_check(&self, m: Move) -> bool {
        self.table[check_pattern(m.moved_piece_after())].test(m.to_sq())
    }
}

impl Position {
    /// Compute this position's [`CheckInfo`] from scratch — the port of
    /// `Position::set_check_info` (`position.cpp` ~242-284). Called once per
    /// position state by the maintenance in `do_move` / `undo_move` /
    /// `refresh_keys`; never on the per-move hot path.
    /// Fully self-derived check info: `in_check` is probed from the board. Used
    /// by the cold entry points (`set` / `refresh_keys` / direct setters / the
    /// empty-stack undo fallback) that have no pre-computed check flag.
    pub(crate) fn compute_check_info(&self) -> CheckInfo {
        self.compute_check_info_impl(None, None)
    }

    /// Check info for the child position reached by a move whose check status is
    /// already known — `in_check` is taken from `gives_check` instead of a fresh
    /// `is_attacked_by` probe (a null move injects `false`). The tables are built
    /// exactly as [`Self::compute_check_info`]; only the `in_check` derivation is
    /// replaced. A debug-only oracle pins the injected flag against the full
    /// probe in the same one-directional style as the `do_move` `ASSERT_LV3`
    /// mirror (`position.cpp`).
    pub(crate) fn compute_check_info_with_in_check(&self, in_check: bool) -> CheckInfo {
        self.compute_check_info_impl(Some(in_check), None)
    }

    /// Like [`Self::compute_check_info_with_in_check`], but the child
    /// `checkersBB` is *also* supplied by the caller — built differentially from
    /// the parent check info and the move (the reference's `do_move_impl`
    /// `st->checkersBB` update, `position.cpp`) — instead of
    /// re-derived here by a full reverse-attack probe. Used by the
    /// `do_move_with_check` check-giving path; the caller owns the debug oracle
    /// pinning the injected set against that probe.
    pub(crate) fn compute_check_info_with_in_check_and_checkers(
        &self,
        in_check: bool,
        checkers: Bitboard,
    ) -> CheckInfo {
        self.compute_check_info_impl(Some(in_check), Some(checkers))
    }

    fn compute_check_info_impl(
        &self,
        injected_in_check: Option<bool>,
        injected_checkers: Option<Bitboard>,
    ) -> CheckInfo {
        let board = self.board();
        let stm = self.side_to_move();
        let enemy = stm.flip();
        let own_king = try_find_king(board, stm);
        let enemy_king = try_find_king(board, enemy);

        // checkSquares[pt]: an enemy-coloured `pt` piece placed on the enemy
        // king attacks exactly the squares from which a side-to-move `pt` piece
        // checks that king (attack reverse-symmetry). The sliders are computed
        // once and HORSE / DRAGON derived from them (`HORSE = BISHOP | king
        // ring`, `DRAGON = ROOK | king ring`, mirroring `set_check_info`), which
        // avoids re-walking the bishop / rook rays. The promoted minors reuse
        // the GOLD slot via [`check_pattern`], so only ten slots are stored.
        //
        // Slot 9 (KING) holds the actual king ring rather than the reference's
        // hard `0`: a king move can never be a *legal* check, but the ring keeps
        // `gives_check` / `gives_direct_check` byte-identical to the
        // scratch-scan oracle on every pseudo-legal king move the generators
        // emit (a king stepping next to the enemy king is pseudo-legal though
        // illegal). The pin zeroes it because such moves never reach
        // `gives_check` in real search; this port preserves the oracle's
        // observable behaviour instead — a deliberate, unobservable-in-legal-play
        // deviation.
        let occ = board.occupied();
        let mut check_squares = [Bitboard::EMPTY; CHECK_PATTERN_COUNT];
        if let Some(eks) = enemy_king {
            // Call the effect primitives directly with the constant piece kinds,
            // mirroring the reference `set_check_info` (`position.cpp`)
            // — bit-identical to an `attacks_bb(eks, Piece { kind, … })`
            // per-kind dispatch (these are the exact match arms `attacks_bb`
            // selects for each constant kind/promoted), but straight-line with no
            // shared-symbol re-dispatch on every call.
            use crate::bitboard::{
                bishop_attacks, gold_attacks, king_attacks, knight_attacks, lance_attacks,
                pawn_attacks, rook_attacks, silver_attacks,
            };
            let bishop = bishop_attacks(eks, occ);
            let rook = rook_attacks(eks, occ);
            let king_ring = king_attacks(enemy, eks);
            check_squares[check_pattern(Piece::new(PieceKind::Pawn, enemy))] =
                pawn_attacks(enemy, eks);
            check_squares[check_pattern(Piece::new(PieceKind::Lance, enemy))] =
                lance_attacks(enemy, eks, occ);
            check_squares[check_pattern(Piece::new(PieceKind::Knight, enemy))] =
                knight_attacks(enemy, eks);
            check_squares[check_pattern(Piece::new(PieceKind::Silver, enemy))] =
                silver_attacks(enemy, eks);
            check_squares[check_pattern(Piece::new(PieceKind::Gold, enemy))] =
                gold_attacks(enemy, eks);
            check_squares[check_pattern(Piece::new(PieceKind::Bishop, enemy))] = bishop;
            check_squares[check_pattern(Piece::new(PieceKind::Rook, enemy))] = rook;
            check_squares[check_pattern(Piece::promoted(PieceKind::Bishop, enemy).unwrap())] =
                bishop | king_ring;
            check_squares[check_pattern(Piece::promoted(PieceKind::Rook, enemy).unwrap())] =
                rook | king_ring;
            check_squares[check_pattern(Piece::new(PieceKind::King, enemy))] = king_ring;
        }

        // blockersForKing[c] and pinners[~c] for both kings — reused from the
        // SEE module. `slider_blockers(c)` returns `(blockersForKing[c],
        // pinners[~c])`, so the pinners half from `slider_blockers(Black)` is
        // `pinners[White]` and vice versa (mirroring the reference StateInfo,
        // which holds `blockersForKing[2]` and `pinners[2]`). `see_ge` /
        // `legal_drop` read both halves back from here instead of recomputing.
        let (blk_black, pin_white) = crate::see::slider_blockers(board, Color::Black);
        let (blk_white, pin_black) = crate::see::slider_blockers(board, Color::White);
        let mut blockers = [Bitboard::EMPTY; Color::COUNT];
        blockers[Color::Black.index()] = blk_black;
        blockers[Color::White.index()] = blk_white;
        let mut pinners = [Bitboard::EMPTY; Color::COUNT];
        pinners[Color::Black.index()] = pin_black;
        pinners[Color::White.index()] = pin_white;

        // `in_check` for the side to move. The cold path probes it from the
        // board; the do_move path injects the caller's `gives_check` (equal to
        // this probe in legal play, since a move gives check iff the resulting
        // side to move is in check). The debug-only oracle is one-directional
        // for the same reason the `do_move` `ASSERT_LV3` mirror is: a claimed
        // check must be a real check, but the reverse legitimately differs on
        // the illegal scratch / fixture positions where the side to move was
        // already in check before the move that reached here.
        let in_check = match injected_in_check {
            Some(v) => {
                #[cfg(debug_assertions)]
                {
                    let derived = own_king.is_some_and(|k| is_attacked_by(board, k, enemy));
                    debug_assert!(
                        !v || derived,
                        "injected in_check was set but the board shows no attacker \
                         on the side-to-move king",
                    );
                }
                v
            }
            None => own_king.is_some_and(|k| is_attacked_by(board, k, enemy)),
        };

        // `checkersBB`: only meaningful (and only needed by `pseudo_legal`) when
        // the side to move is in check. The `do_move_with_check` check-giving
        // path supplies it pre-built (differentially, from the parent info + the
        // move — the reference's `do_move_impl` update), so the full probe is
        // skipped there. Cold entry points inject nothing and fall back to the
        // reverse-attack lookup from the king square (`attackers_bb`), gated on
        // `in_check` so the common not-in-check path stays a no-op.
        let checkers = match injected_checkers {
            Some(c) => c,
            None => {
                if in_check {
                    let oks = own_king.expect("in_check implies the side to move has a king");
                    crate::movegen::attackers_bb(board, oks, enemy)
                } else {
                    Bitboard::EMPTY
                }
            }
        };

        CheckInfo {
            board_key: self.board_key(),
            enemy_king,
            own_king,
            in_check,
            check_squares,
            blockers,
            pinners,
            checkers,
        }
    }

    /// Build the child state's `checkersBB` differentially from the PARENT check
    /// info (`parent`) and the move just played, mirroring the reference
    /// `do_move_impl` (`position.cpp` — board moves, then drops).
    /// Called only on the `gives_check` path, with `self` already advanced to
    /// the post-move position (board and side final) but `parent` still the
    /// pre-move check info — whose `enemy_king` / `check_squares` / `blockers`
    /// describe the mover's frame (`mover` = the side that just moved, `Us` in
    /// the reference; its `blockers[~mover]` = `blockersForKing[them]`).
    ///
    /// * **Direct part**: `{to}` iff `to` lies in the parent's
    ///   `check_squares[check_pattern(moved_piece_after)]` — the identical test
    ///   the [`Position::gives_check`] predicate performs
    ///   (`prevSt->checkSquares[type_of(moved_after_pc)] & to`), KING-slot
    ///   semantics included, so the built set matches `attackers_bb` on every
    ///   position real search reaches.
    /// * **Discovered part** (board moves only): iff the from-square singly
    ///   blocks one of the mover's sliders aimed at the enemy king
    ///   (`blockersForKing[them] & from`) and the move leaves that ray
    ///   (`!aligned`), add the revealed slider — the first mover piece on the
    ///   enemy-king→from ray over the POST-move occupancy. The reference walks
    ///   `directEffect(from, direct_of(ksq, from), pieces())` from `from` away
    ///   from the king; equivalently, the occupancy-limited slider attack *from*
    ///   the king (the between-squares are empty because `from` was the sole
    ///   blocker) restricted to that single ray finds the same square.
    pub(crate) fn differential_child_checkers(
        &self,
        m: Move,
        mover: Color,
        parent: &CheckInfo,
    ) -> Bitboard {
        let board = self.board();
        let to = m.to_sq();

        // Direct check: `{to}` iff the landed piece's type checks from `to`.
        let mut checkers = if parent.check_squares[check_pattern(m.moved_piece_after())].test(to) {
            Bitboard::from_square(to)
        } else {
            Bitboard::EMPTY
        };

        // Discovered check (board moves only): the from-square singly blocks one
        // of the mover's sliders aimed at the enemy king, and the move leaves
        // that ray.
        if let Some(eks) = parent.enemy_king
            && !m.is_drop()
            && parent.blockers(mover.flip()).test(m.from_sq())
            && !aligned(eks, m.from_sq(), to)
        {
            let from = m.from_sq();
            let occ = board.occupied();
            let (df, dr) =
                ray_dir(eks, from).expect("a blocker for the enemy king is aligned with it");
            // Orthogonal vs diagonal ray → rook- vs bishop-shaped slider.
            let slider = if df != 0 && dr != 0 {
                crate::bitboard::bishop_attacks(eks, occ)
            } else {
                crate::bitboard::rook_attacks(eks, occ)
            };
            checkers |= slider & crate::bitboard::ray_toward(eks, from) & board.pieces_color(mover);
        }

        checkers
    }

    /// True iff the side to move's king is currently attacked — the reference's
    /// `Position::in_check()`. Used by the `MovePicker` constructor to pick
    /// between the capture and evasion stages.
    ///
    /// Reads the per-state check info: the search calls this once at node entry
    /// (before any per-move predicate), so it warms the cache the `is_legal` /
    /// `gives_check` / `gives_direct_check` calls at that node then reuse.
    pub fn in_check(&self) -> bool {
        self.check_info().in_check
    }

    /// True iff `sq` is attacked by any piece of `attacker`, with `discount`
    /// treated as empty — the reference's `effected_to(attacker, sq, discount)`
    /// (`position.h`), used by the try-rule entering-king declaration.
    ///
    /// Removing `discount` from the occupancy models the moving king vacating
    /// its from-square: it both drops the king itself as a (would-be) defender
    /// and reveals any enemy slider the king was blocking. `discount == sq` is
    /// harmless — an enemy piece already on the try square is a capture target,
    /// not a defender, and clearing it only prevents it from "attacking itself".
    pub fn is_attacked_discounting(&self, sq: Square, attacker: Color, discount: Square) -> bool {
        let mut board = *self.board();
        board.set(discount, None);
        is_attacked_by(&board, sq, attacker)
    }

    /// True iff playing `m` would leave the opponent's king in check — the
    /// reference's `Position::gives_check(m)` (direct checks, discovered
    /// checks, and checking drops). Faithful to the definition "a move that
    /// leaves the opponent king in check": `m` is applied to a scratch board
    /// and the opponent king is tested against the full attacker scan, so a
    /// slider uncovered by the move (discovered check) is detected the same way
    /// as a piece that directly attacks the king.
    ///
    /// Matches, by construction, the post-`do_move` `gives_check` flag recorded
    /// on the move history (both decide the same predicate); that flag is the
    /// oracle the unit test checks this against. If `m` captures the opponent's
    /// king (a pseudo-legal probe move, never a real move) there is no king to
    /// check and the result is `false`.
    ///
    /// Ported to the reference's constant-time form (`position.cpp`):
    /// a **direct** check iff the moved piece's destination lies in
    /// `checkSquares[type_of(moved_piece_after(m))]`; else, for a board move, a
    /// **discovered** check iff the from-square singly blocks one of our sliders
    /// aimed at the enemy king (`blockersForKing(~stm) & from`) and the move does
    /// not slide along that ray (`!aligned(from, to, enemy_king)`).
    pub fn gives_check(&self, m: Move) -> bool {
        let ci = self.check_info();
        let to = m.to_sq();

        // Capturing the enemy king (a pseudo-legal probe move only) leaves no
        // king to check — the scratch-board oracle finds no enemy king here.
        if ci.enemy_king == Some(to) {
            return false;
        }

        // Direct check: a piece of the moved type, on `to`, attacks the king.
        if ci.check_squares[check_pattern(m.moved_piece_after())].test(to) {
            return true;
        }

        // Discovered check: board moves only. `from` must singly block one of
        // our sliders from the enemy king, and the move must leave that ray.
        if m.is_drop() {
            return false;
        }
        let eks = match ci.enemy_king {
            Some(k) => k,
            None => return false,
        };
        let from = m.from_sq();
        let enemy = self.side_to_move().flip();
        ci.blockers[enemy.index()].test(from) && !aligned(eks, from, to)
    }

    /// True iff `m` gives a **direct** check by the moved piece — the reference
    /// main-search quiet-ordering term `check_squares(type_of(moved_piece(m))) &
    /// to` (`movepick.cpp`).
    ///
    /// The reference precomputes, per position, `checkSquares[pt]` = the squares
    /// from which a `pt` piece of the side to move attacks the enemy king under
    /// the *current* occupancy (`position.cpp`), then ANDs it with the
    /// move's destination. This reproduces that pointwise: a piece of type
    /// `moved_piece_after(m)` (the after-promotion piece — the reference's
    /// `pos.moved_piece(m)`) is imagined on `to`, and it checks iff the enemy
    /// king lies on one of its steps, or on a slider ray from `to` unobstructed
    /// by the current board. Like `checkSquares`, this is computed against the
    /// occupancy *before* the move (the mover still on `from`), so a slider
    /// whose own vacated `from` square lay on the ray is not counted — matching
    /// the reference's approximation. Discovered checks are deliberately *not*
    /// detected (they are not direct checks). Used only to score quiet moves, so
    /// `to` is empty; a drop's dropped piece is handled the same way.
    ///
    /// Now a single lookup into the cached `checkSquares` — the exact
    /// `check_squares(type_of(moved_piece(m))) & to` term (`movepick.cpp`).
    pub fn gives_direct_check(&self, m: Move) -> bool {
        let ci = self.check_info();
        ci.check_squares[check_pattern(m.moved_piece_after())].test(m.to_sq())
    }

    /// Snapshot the cached `checkSquares` table by value into a [`CheckSquares`]
    /// — one `check_info()` borrow, no `Position` access
    /// thereafter. Taken once at the picker's `QUIET_INIT` stage entry so the
    /// per-quiet check bonus becomes a bare table lookup
    /// ([`CheckSquares::gives_direct_check`]) instead of re-entering the lazy
    /// `RefCell` accessor per scored move — the reference reads the precomputed
    /// `StateInfo::checkSquares` field with the same zero ceremony
    /// (`movepick.cpp`). See [`CheckSquares`] for the output-equivalence
    /// argument.
    pub fn check_squares(&self) -> CheckSquares {
        CheckSquares {
            table: self.check_info().check_squares,
        }
    }

    /// True iff `m` is legal for the side to move in the sense the reference
    /// qsearch uses (`pos.legal(m)`): after the move, the mover's own king is
    /// not left in check. This covers moving a pinned piece off its pin ray and
    /// a king stepping into an attacked square.
    ///
    /// It is repetition-blind — no perpetual-check or sennichite term, exactly
    /// like the reference search-move generators (`MoveList`) and the
    /// [`Position::generate_legal_all`] list built on them. The uchifuzume
    /// (drop-pawn-mate) rule, however, *is* enforced — but at
    /// generation, inside [`Position::generate_evasions`] /
    /// [`Position::generate_captures`] (captures carry no drops), exactly as the
    /// reference's `GenerateDropMoves` does; so no uchifuzume move ever reaches
    /// this predicate.
    ///
    /// # Structure (reference `Position::legal`, `position.cpp`)
    ///
    /// This is now the pin's O(1), board-mutation-free `legal` — the same three
    /// arms in and out of check (the reference has no in-check special case):
    ///
    /// * **King move** — legal iff the destination is unattacked with the king's
    ///   from-square vacated (`!effected_to(~us, to, from)`), i.e.
    ///   [`Position::is_attacked_discounting`]. Valid in and out of check.
    /// * **Drop** — legal (`return true`). A drop adds a piece and can never
    ///   expose the own king. When in check this leans on the entry contract:
    ///   the restricted [`Position::generate_evasions`] only emits interposition
    ///   drops, and [`Position::pseudo_legal`] (`pseudo_legal_s`) rejects any
    ///   other in-check drop (single-checker interposition only; every drop
    ///   under double check), so every drop reaching here already resolves the
    ///   check.
    /// * **Other board move** — legal iff the moving piece does not singly block
    ///   a slider aimed at the own king, or it slides along that ray
    ///   (`!(blockersForKing(us) & from) || aligned(from, to, own_king)`). When
    ///   in check this again leans on the contract: generation (and
    ///   `pseudo_legal_s` for a TT candidate) restricts an in-check board move to
    ///   capture-the-checker-or-interpose, so the only remaining way it can be
    ///   illegal is by leaving a pin ray — exactly this pinned-blocker test.
    ///
    /// # Contract (audited call sites)
    ///
    /// The argument must come from one of the search move generators for the
    /// current check state (evasions when in check, captures / quiets /
    /// non-evasions when not), or have passed [`Position::pseudo_legal`]. Every
    /// non-test caller honours this:
    ///
    /// * `MovePicker` (`movepick.rs`): the capture/evasion and quiet stages emit
    ///   only their generators' moves (evasions when `in_check`); the TT and
    ///   probcut stages gate on `pseudo_legal` first. The quiet stages run only
    ///   when not in check.
    /// * `generate_root_moves` (`root.rs`): evasions when in check, else
    ///   non-evasions.
    /// * `qsearch` PV / probcut (`qsearch.rs`): PV extension gates on
    ///   `pseudo_legal`; probcut is skipped in check (Steps 6b–11 are bypassed
    ///   when `in_check`), so its moves are the not-in-check capture generator's.
    ///
    /// No non-test caller can hand this predicate an in-check move that neither
    /// resolves the check nor passed `pseudo_legal`.
    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.side_to_move();

        // King move: unattacked destination with the from-square vacated.
        if !m.is_drop() && m.moved_piece_after().kind == PieceKind::King {
            return !self.is_attacked_discounting(m.to_sq(), us.flip(), m.from_sq());
        }

        // A drop adds a piece and can never expose the king. In check this holds
        // by the entry contract (only interposition drops reach here).
        if m.is_drop() {
            return true;
        }

        // A board move exposes the king only by moving a sole slider-blocker
        // off its pin ray. In check the destination is contract-restricted to
        // capture-or-interpose, so this pin test is the only remaining illegality.
        let (own_king, pinned) = {
            let ci = self.check_info();
            (ci.own_king, ci.blockers[us.index()])
        };
        let from = m.from_sq();
        let oks = match own_king {
            Some(k) => k,
            None => return false,
        };
        !pinned.test(from) || aligned(oks, from, m.to_sq())
    }

    /// True iff playing `m` leaves the side to move's own king unattacked — a
    /// copy-apply-scan `is_legal`, kept purely as a `#[cfg(test)]`
    /// equivalence oracle (the restricted-evasion sequence gate filters the
    /// unrestricted oracle twin with it, and the predicate-equivalence gate pins
    /// `is_legal` against [`Self::is_legal_reference`], its verbatim copy).
    #[cfg(test)]
    fn leaves_own_king_safe(&self, m: Move) -> bool {
        let mover = self.side_to_move();
        let mut board = *self.board();
        apply_to_board(&mut board, m, mover);
        match try_find_king(&board, mover) {
            Some(king_sq) => !is_attacked_by(&board, king_sq, mover.flip()),
            None => false,
        }
    }

    /// Widen a stored 16-bit TT fragment into a full [`Move`] in **O(1)** — the
    /// reference `Position::to_move(Move16)` (`position.cpp`). This
    /// attaches the moving-piece bits the `move16` layout drops, without
    /// generating any move list and without proving
    /// legality (that is [`Self::pseudo_legal`] + [`Self::is_legal`]'s job).
    ///
    /// * `MOVE_NONE` (fragment `0`) → `None`.
    /// * A non-`is_ok` fragment (`MOVE_WIN` / from == to) → returned verbatim as
    ///   `Some`, mirroring the pin's `if (!m.is_ok()) return m;`. It is never a
    ///   legal move, so [`Self::pseudo_legal`] rejects it downstream; keeping it
    ///   `Some` (rather than folding it to `None`) matches the pin's
    ///   `ttData.move` being non-`MOVE_NONE` for such a value, so the search's
    ///   `tt_move.is_none()` gates agree with the reference `!ttData.move`.
    /// * A drop attaches `make_piece(stm, dropped)`; a board move requires
    ///   `piece_on(from)` to be the mover's own piece (else `None`); a promotion
    ///   of a non-promotable piece yields `None`, else the promoted piece is
    ///   attached.
    ///
    /// **Totality.** The pin trusts the stored dropped-piece bits ("we wrote
    /// them"); this engine's TT admits torn fragments, so this stays total —
    /// an out-of-range dropped-piece field returns `None` instead of indexing a
    /// table or panicking. Every returned `Move` is well-formed (built through
    /// the `Move` constructors), so [`Move::moved_piece_after`] on the result
    /// never panics.
    pub fn to_move(&self, m16: u16) -> Option<Move> {
        if m16 == 0 {
            return None;
        }
        let m = Move::from_bits(m16 as u32);
        if !m.is_ok() {
            // MOVE_WIN / MOVE_NULL / a from == to fragment: the pin returns the
            // move unchanged and lets `pseudo_legal` reject it.
            return Some(m);
        }
        let stm = self.side_to_move();

        // Totality: a torn fragment can carry a to-square in 81..127, which
        // `Move::to_sq` would panic on — validate it up front.
        let to = Square::from_index((m16 & 0x7f) as u8)?;

        if m.is_drop() {
            // Totality: an out-of-range dropped-piece field is not a real drop.
            let kind = m.dropped_piece_kind_checked()?;
            return Some(Move::make_drop(kind, stm, to));
        }

        // Board move: the from-square must be on the board and hold the mover's
        // own piece.
        let from = Square::from_index(((m16 >> 7) & 0x7f) as u8)?;
        let moved = self.board().get(from)?;
        if moved.color != stm {
            return None;
        }
        if m.is_promote() {
            if is_non_promotable_piece(moved) {
                return None;
            }
            return Some(Move::make_promote(from, to, moved));
        }
        Some(Move::make(from, to, moved))
    }

    /// True iff `m` is pseudo-legal for the side to move — the reference
    /// `Position::pseudo_legal(m, generate_all_legal_moves)` =
    /// `pseudo_legal_s<All>` (`position.cpp`). Pseudo-legal allows a
    /// king-suicide (that is [`Self::is_legal`]'s job); it is the pre-`do_move`
    /// guard for a TT / killer move, deciding whether the fragment is even
    /// well-shaped for this position.
    ///
    /// `all` is the per-`go` `GenerateAllLegalMoves` flag: with `all == false`
    /// the generator's non-promotion bans apply (a pawn/bishop/rook must promote
    /// when touching the enemy field, a lance must promote into the enemy first
    /// two ranks); with `all == true` only the cannot-move-otherwise bans remain
    /// (a pawn/lance may not sit on the last rank un-promoted).
    ///
    /// Guaranteed to return `false` for a non-`is_ok` move (`MOVE_WIN` etc.),
    /// mirroring the pin's contract, and total (never panics) on any well-formed
    /// move produced by [`Self::to_move`] or the generators.
    pub fn pseudo_legal(&self, m: Move, all: bool) -> bool {
        // The pin guarantees `pseudo_legal(m) == false` for `!is_ok(m)`; this
        // also keeps `moved_piece_after` below off the sentinel bit patterns.
        if !m.is_ok() {
            return false;
        }
        let us = self.side_to_move();
        let to = m.to_sq();
        let board = self.board();

        if m.is_drop() {
            let pr = match m.dropped_piece_kind_checked() {
                Some(k) => k,
                None => return false,
            };
            // The stored after-move piece must be `make_piece(us, pr)`.
            if m.moved_piece_after() != Piece::new(pr, us) {
                return false;
            }
            // The target must be empty and the piece held in hand.
            if board.get(to).is_some() || self.hand(us).count(pr) == 0 {
                return false;
            }
            // In check: the drop must interpose on the single checker's ray;
            // double check rejects every drop.
            let (in_check, checkers, own_king) = {
                let ci = self.check_info();
                (ci.in_check, ci.checkers, ci.own_king)
            };
            if in_check {
                if checkers.popcount() != 1 {
                    return false;
                }
                let checksq = checkers.squares().next().unwrap();
                let oks = match own_king {
                    Some(k) => k,
                    None => return false,
                };
                if !between_set(checksq, oks).test(to) {
                    return false;
                }
            }
            // A pawn drop must pass the nifu + drop-pawn-mate predicate.
            if pr == PieceKind::Pawn && !self.legal_pawn_drop(us, to) {
                return false;
            }
            return true;
        }

        // Board move.
        let pc = match board.get(m.from_sq()) {
            Some(p) if p.color == us => p,
            _ => return false,
        };
        // `to` must be reachable by `pc` from `from` under the current occupancy
        // (`effects_from`), and must not hold one of our own pieces.
        if !attacks_bb(m.from_sq(), pc, board.occupied()).test(to) {
            return false;
        }
        if board.get(to).is_some_and(|t| t.color == us) {
            return false;
        }

        if m.is_promote() {
            if is_non_promotable_piece(pc) {
                return false;
            }
            let promoted = match Piece::promoted(pc.kind, pc.color) {
                Some(p) => p,
                None => return false,
            };
            if m.moved_piece_after() != promoted {
                return false;
            }
        } else {
            if m.moved_piece_after() != pc {
                return false;
            }
            // Non-promotion bans (`pseudo_legal_s`'s `All` switch).
            if all {
                // Pawn / lance may not sit un-promoted on the last rank.
                if matches!(pc.kind, PieceKind::Pawn | PieceKind::Lance) && !pc.promoted {
                    let last_rank = if us == Color::Black { 0 } else { 8 };
                    if to.rank() == last_rank {
                        return false;
                    }
                }
            } else {
                match (pc.kind, pc.promoted) {
                    // Pawn: no un-promoted move into the enemy field.
                    (PieceKind::Pawn, false) if is_in_promotion_zone(to, us) => return false,
                    // Lance: no un-promoted move into the enemy first two ranks.
                    (PieceKind::Lance, false) => {
                        let banned = if us == Color::Black {
                            to.rank() <= 1
                        } else {
                            to.rank() >= 7
                        };
                        if banned {
                            return false;
                        }
                    }
                    // Bishop / rook: no un-promoted move touching the enemy field
                    // (from or to).
                    (PieceKind::Bishop, false) | (PieceKind::Rook, false)
                        if is_in_promotion_zone(m.from_sq(), us)
                            || is_in_promotion_zone(to, us) =>
                    {
                        return false;
                    }
                    _ => {}
                }
            }
        }

        // In check with a non-king mover: reject double check; else the move must
        // capture the checker or interpose on its ray. King moves fall through —
        // their suicide check is [`Self::is_legal`]'s job.
        if pc.kind != PieceKind::King {
            let (in_check, checkers, own_king) = {
                let ci = self.check_info();
                (ci.in_check, ci.checkers, ci.own_king)
            };
            if in_check {
                if checkers.popcount() > 1 {
                    return false;
                }
                let checksq = checkers.squares().next().unwrap();
                let oks = match own_king {
                    Some(k) => k,
                    None => return false,
                };
                let target = between_set(checksq, oks) | Bitboard::from_square(checksq);
                if !target.test(to) {
                    return false;
                }
            }
        }

        true
    }

    /// The reference `legal_pawn_drop(us, to)` (`position.cpp`): a pawn
    /// drop on `to` is legal iff it is not nifu (二歩 — no own un-promoted pawn
    /// already on the file) and not uchifuzume (打ち歩詰め — an unanswerable
    /// drop-pawn-mate). Reuses the same drop-pawn-mate probe the generators use
    /// ([`Self::pawn_drop_is_uchifuzume`]), fired only when the dropped pawn
    /// would actually attack the enemy king.
    fn legal_pawn_drop(&self, us: Color, to: Square) -> bool {
        let board = self.board();
        // Nifu: an own un-promoted pawn already stands on this file.
        for rank in 0..Square::RANKS {
            let sq = Square::new(to.file(), rank).unwrap();
            if let Some(p) = board.get(sq)
                && p.color == us
                && p.kind == PieceKind::Pawn
                && !p.promoted
            {
                return false;
            }
        }
        // Uchifuzume: only possible when the dropped pawn checks the enemy king,
        // i.e. `to` is the unique square from which our pawn attacks that king.
        if let Some(ek) = try_find_king(board, us.flip()) {
            let dr: i16 = if us == Color::Black { 1 } else { -1 };
            let r = ek.rank() as i16 + dr;
            if (0..Square::RANKS as i16).contains(&r)
                && Square::new(ek.file(), r as u8) == Some(to)
                && self.pawn_drop_is_uchifuzume(to)
            {
                return false;
            }
        }
        true
    }

    /// Append the reference-ordered pseudo-legal `CAPTURES` candidates (see the
    /// module docs) to `out`. Piece-type-major, ascending from- then to-square,
    /// with the `make_move_target` promotion/non-promotion rules; no drops.
    /// The caller filters legality with [`Position::is_legal`].
    pub fn generate_captures(&self, all: bool, out: &mut Vec<ExtMove>) {
        let board = self.board();
        let stm = self.side_to_move();
        emit_group::<PawnG>(board, stm, Target::Captures, all, out);
        emit_group::<LanceG>(board, stm, Target::Captures, all, out);
        emit_group::<KnightG>(board, stm, Target::Captures, all, out);
        emit_group::<SilverG>(board, stm, Target::Captures, all, out);
        emit_group::<BishopRookG>(board, stm, Target::Captures, all, out);
        emit_group::<GoldHdkG<true>>(board, stm, Target::Captures, all, out);
    }

    /// Append the reference-ordered pseudo-legal `QUIETS` candidates to `out`:
    /// non-capturing piece moves (every side-to-move piece to an empty square,
    /// same piece-type-major order and `make_move_target` promotion rules as
    /// [`Position::generate_captures`], with the gold-group including the king),
    /// then every pseudo-legal drop on an empty square. This is
    /// `generate_general<QUIETS, Us, false>` at the pin (`movegen.cpp`):
    /// the move target is `pos.empties()` and the drop target is `pos.empties()`.
    /// The caller filters legality with [`Position::is_legal`].
    ///
    /// Non-capturing pawn promotions (a pawn pushing into the enemy field onto
    /// an empty square) belong here, not to `generate_captures`: at the pin the
    /// main-search capture stage uses plain `CAPTURES` (target `pieces(Them)`),
    /// not `CAPTURES_PRO_PLUS`, so the two generators partition the destinations
    /// (enemy vs empty) with no overlap.
    pub fn generate_quiets(&self, all: bool, out: &mut Vec<ExtMove>) {
        let board = self.board();
        let stm = self.side_to_move();
        emit_group::<PawnG>(board, stm, Target::Quiets, all, out);
        emit_group::<LanceG>(board, stm, Target::Quiets, all, out);
        emit_group::<KnightG>(board, stm, Target::Quiets, all, out);
        emit_group::<SilverG>(board, stm, Target::Quiets, all, out);
        emit_group::<BishopRookG>(board, stm, Target::Quiets, all, out);
        emit_group::<GoldHdkG<true>>(board, stm, Target::Quiets, all, out);
        self.emit_drops(out);
    }

    /// Append the reference-ordered pseudo-legal `EVASIONS` candidates (see the
    /// module docs) to `out`: king moves first, then — on a single check — the
    /// non-king piece moves (gold-group excluding the king) restricted to
    /// capture-or-interpose, then interposition drops. The caller filters
    /// legality with [`Position::is_legal`], which removes the remaining suicide
    /// king steps.
    ///
    /// **Entry contract:** the side to move is in check (the reference
    /// `generate_evasions` asserts `pos.in_check()`). The `MovePicker` /
    /// `generate_root_moves` only call this when [`Position::in_check`] holds.
    pub fn generate_evasions(&self, all: bool, out: &mut Vec<ExtMove>) {
        let board = self.board();
        let stm = self.side_to_move();

        let (ksq, checkers) = {
            let ci = self.check_info();
            debug_assert!(
                ci.in_check,
                "generate_evasions requires the side to move to be in check"
            );
            (ci.own_king, ci.checkers)
        };
        let ksq = match ksq {
            Some(k) => k,
            None => return,
        };
        let king = board.get(ksq).unwrap();

        // sliderAttacks: the union of every checker's attack rays, so the king
        // cannot step onto a still-attacked square (movegen.cpp). Under
        // the *current* occupancy (king in) rather than the pin's king-removed
        // occupancy — an output-preserving deviation, see the module docs.
        let occ = board.occupied();
        let mut slider_attacks = Bitboard::empty();
        for checksq in checkers.squares() {
            let cp = board
                .get(checksq)
                .expect("a checker square holds an enemy piece");
            slider_attacks |= attacks_bb(checksq, cp, occ);
        }

        // King moves first: king steps onto neither own pieces nor sliderAttacks
        // (movegen.cpp). Ascending square order.
        let king_targets = reachable(board, ksq, king, Target::BlockOrCapture) & !slider_attacks;
        for to in king_targets.squares() {
            push_plain(ksq, to, king, out);
        }

        // Double check: only king moves evade (movegen.cpp).
        if checkers.popcount() >= 2 {
            return;
        }

        // Single check: capture the checker or interpose (movegen.cpp).
        let checksq = checkers
            .squares()
            .next()
            .expect("in check implies at least one checker");
        let target1 = crate::bitboard::between(checksq, ksq); // interposition squares
        let target2 = target1 | Bitboard::from_square(checksq); // + capture the checker

        // Non-king piece moves, same order as CAPTURES but the gold-group
        // excludes the king, masked to target2.
        emit_group_masked::<PawnG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<LanceG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<KnightG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<SilverG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<BishopRookG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<GoldHdkG<false>>(board, stm, Target::BlockOrCapture, target2, all, out);

        // Interposition drops last, masked to target1 (movegen.cpp).
        self.emit_drops_masked(target1, out);
    }

    /// Append the reference-ordered pseudo-legal `NON_EVASIONS` candidates to
    /// `out` — `generate_general<NON_EVASIONS, Us, false>` at the pin
    /// (`movegen.cpp`). This is the generator behind `MoveList<LEGAL>`
    /// for a not-in-check position (`movegen.cpp`), and thus the source of
    /// the root-move list the depth-1 root search consumes.
    ///
    /// The single move target is `~pos.pieces(Us)` (empty *and* enemy squares
    /// together, i.e. [`Target::BlockOrCapture`]), so — unlike the
    /// captures/quiets split — captures and quiets are **interleaved per piece
    /// and per destination square**: each side-to-move piece (in the
    /// [`Position::generate_captures`] piece-type order, gold-group *including*
    /// the king) emits all of its reachable squares in ascending order,
    /// whatever they hold, with the `make_move_target` promotion suppression.
    /// Drops (on empty squares) come last, in the shared `emit_drops` order.
    ///
    /// The moves are pseudo-legal; the caller filters king-safety with
    /// [`Position::is_legal`]. Reproducing the interleaved order (rather than
    /// concatenating `generate_captures` then `generate_quiets`, which would put
    /// every capture ahead of every quiet) is what makes the *first* legal move
    /// — the root search's TT move — match the reference.
    pub fn generate_non_evasions(&self, all: bool, out: &mut Vec<ExtMove>) {
        let board = self.board();
        let stm = self.side_to_move();
        emit_group::<PawnG>(board, stm, Target::BlockOrCapture, all, out);
        emit_group::<LanceG>(board, stm, Target::BlockOrCapture, all, out);
        emit_group::<KnightG>(board, stm, Target::BlockOrCapture, all, out);
        emit_group::<SilverG>(board, stm, Target::BlockOrCapture, all, out);
        emit_group::<BishopRookG>(board, stm, Target::BlockOrCapture, all, out);
        emit_group::<GoldHdkG<true>>(board, stm, Target::BlockOrCapture, all, out);
        self.emit_drops(out);
    }

    /// Append every legal move for the side to move to `out`, stripped from
    /// `ExtMove` to bare [`Move`]. The buffer is **not** cleared first (the
    /// caller may pre-allocate or reuse it across calls) — the same contract the
    /// former `Position::generate_legal_moves` exposed.
    ///
    /// This is exactly the reference `generate<LEGAL_ALL>`
    /// (`source/movegen.cpp`): pick the check-state
    /// generator (evasions when [`Position::in_check`], else non-evasions) with
    /// the `All == true` flag, then drop every candidate failing
    /// [`Position::is_legal`]. The `is_legal` entry contract
    /// (check-state-appropriate generator output) is satisfied by construction.
    ///
    /// **Repetition-blind**, exactly like the reference: there is no sennichite
    /// or perpetual-check (連続王手の千日手) term. Plain 4-fold makes the *game*
    /// drawn, not the move illegal, and a perpetual-check repetition is scored
    /// by the search (and adjudicated by the server in real games), never
    /// removed here. Uchifuzume (打ち歩詰め) and nifu (二歩) are excluded at
    /// drop-generation time inside the generators (mirroring the reference
    /// `GenerateDropMoves` / `legal_drop`), so they never reach `is_legal`.
    pub fn generate_legal_all(&self, out: &mut Vec<Move>) {
        let mut buf: Vec<ExtMove> = Vec::with_capacity(64);
        if self.in_check() {
            self.generate_evasions(true, &mut buf);
        } else {
            self.generate_non_evasions(true, &mut buf);
        }
        for em in buf {
            if self.is_legal(em.mv) {
                out.push(em.mv);
            }
        }
    }

    /// Emit the pseudo-legal drops on empty squares in the reference
    /// `GenerateDropMoves` order (`movegen.cpp`): pawn drops (ascending
    /// square) first, then the other kinds in rank bands. Shared by
    /// [`Position::generate_evasions`] (where the caller's legality filter keeps
    /// only the check-blocking drops) and [`Position::generate_quiets`] (where
    /// every drop is a genuine quiet); the reference passes `pos.empties()` as
    /// the drop target in both the `EVASIONS` and `QUIETS` paths, so the same
    /// nifu / last-rank / uchifuzume-filtered generation serves both.
    fn emit_drops(&self, out: &mut Vec<ExtMove>) {
        self.emit_drops_masked(ALL_SQUARES, out);
    }

    /// [`Position::emit_drops`] with the empty-square drop target additionally
    /// intersected with `restrict` — the reference `target1` (interposition
    /// squares only) the evasion generator threads in (`movegen.cpp`). The
    /// quiets / non-evasions paths pass [`ALL_SQUARES`], an identity
    /// intersection. Because `restrict` only removes candidate squares, the
    /// emitted drops stay a subsequence of the unrestricted emission, preserving
    /// the reference rank-band order.
    fn emit_drops_masked(&self, restrict: Bitboard, out: &mut Vec<ExtMove>) {
        use crate::bitboard::rank_mask;

        let board = self.board();
        let stm = self.side_to_move();
        let hand = self.hand(stm);

        // Base droppable target: the restriction intersected with the empty
        // squares. Every drop pass narrows this further; popping a resulting
        // bitboard yields ascending square indices — the same sequence the
        // former 81-square scan walked (`movegen.cpp` drives the whole
        // `GenerateDropMoves` off bitboards this way).
        let base = restrict & !board.occupied();

        let (back_rank, second_rank): (u8, u8) = if stm == Color::Black { (0, 1) } else { (8, 7) };

        // --- Pawn drops: empty, not the last rank, not a nifu file, and not
        // uchifuzume (打ち歩詰め). A dropped pawn checks only the single square
        // directly ahead of it, so the *only* drop square that can be
        // pawn-drop-mate is the one from which our pawn would attack the enemy
        // king; that square is tested for mate and excluded, mirroring the
        // reference `GenerateDropMoves`' `legal_drop` (`movegen.cpp`, pawn-drop
        // branch). Emitting it would hand the caller an illegal move (the
        // qsearch legality filter [`Position::is_legal`] is uchifuzume-blind).
        if hand.count(PieceKind::Pawn) > 0 {
            // Nifu (二歩) file exclusion: fold the side's own un-promoted-pawn
            // bitboard into a full-file mask (a promoted pawn / tokin lives in
            // the GOLD slot and does not count), then subtract it from the
            // target — the bitboard form of the per-file `nifu_blocked_files`.
            let own_pawns = board.pieces_pattern(stm, crate::board::pat::PAWN);
            let mut nifu_files = Bitboard::empty();
            for psq in own_pawns.squares() {
                nifu_files |= crate::bitboard::file_mask(psq.file());
            }
            // Last rank (from which a dropped pawn can never move) is removed too.
            let mut pawn_target = base & !rank_mask(back_rank) & !nifu_files;

            // The unique square a dropped pawn could deliver mate from: directly
            // "behind" the enemy king along our pawn's line of advance (Black
            // advances toward rank 0, so it is one rank above the king; White
            // one rank below). Probe it for uchifuzume only when it survives as
            // a genuine pawn-drop square, and remove it if it mates — at most one
            // probe per node, matching the scan twin's short-circuit.
            let uchi_candidate = try_find_king(board, stm.flip()).and_then(|ek| {
                let dr: i16 = if stm == Color::Black { 1 } else { -1 };
                let r = ek.rank() as i16 + dr;
                if (0..Square::RANKS as i16).contains(&r) {
                    Square::new(ek.file(), r as u8)
                } else {
                    None
                }
            });
            if let Some(c) = uchi_candidate
                && pawn_target.test(c)
                && self.pawn_drop_is_uchifuzume(c)
            {
                pawn_target.clear(c);
            }

            for sq in pawn_target.squares() {
                out.push(ExtMove {
                    mv: Move::make_drop(PieceKind::Pawn, stm, sq),
                    value: 0,
                });
            }
        }

        // --- Other kinds, laid into `drops` in the reference order, then
        // emitted in rank bands so lance / knight are excluded from the ranks
        // on which they would be stuck.
        // Fixed-capacity scratch (at most the six [`DROP_ORDER`] kinds); avoids a
        // per-node heap allocation.
        let mut drops_buf = [PieceKind::Knight; DROP_ORDER.len()];
        let mut drops_len = 0usize;
        for &kind in &DROP_ORDER {
            if hand.count(kind) > 0 {
                drops_buf[drops_len] = kind;
                drops_len += 1;
            }
        }
        let drops = &drops_buf[..drops_len];
        if drops.is_empty() {
            return;
        }
        // Index just past the leading knight / knight+lance entries (they lead
        // `DROP_ORDER`), matching the reference's `nextToKnight` / `nextToLance`.
        let next_to_knight = usize::from(drops.first() == Some(&PieceKind::Knight));
        let next_to_lance =
            next_to_knight + usize::from(drops.get(next_to_knight) == Some(&PieceKind::Lance));

        if next_to_lance == 0 {
            // No lance or knight in hand: every kind can go on every empty
            // square (movegen.cpp).
            for sq in base.squares() {
                for &kind in drops {
                    out.push(ExtMove {
                        mv: Move::make_drop(kind, stm, sq),
                        value: 0,
                    });
                }
            }
            return;
        }

        // Band 1 — own back rank: the kinds that may sit there (silver, gold,
        // bishop, rook — i.e. from `next_to_lance`), movegen.cpp.
        for sq in (base & rank_mask(back_rank)).squares() {
            for &kind in &drops[next_to_lance..] {
                out.push(ExtMove {
                    mv: Move::make_drop(kind, stm, sq),
                    value: 0,
                });
            }
        }
        // Band 2 — own second rank: lance too, but not knight (from
        // `next_to_knight`), movegen.cpp.
        for sq in (base & rank_mask(second_rank)).squares() {
            for &kind in &drops[next_to_knight..] {
                out.push(ExtMove {
                    mv: Move::make_drop(kind, stm, sq),
                    value: 0,
                });
            }
        }
        // Band 3 — the remaining ranks (all but the back and second ranks):
        // every kind, movegen.cpp.
        let band3 = base & !(rank_mask(back_rank) | rank_mask(second_rank));
        for sq in band3.squares() {
            for &kind in drops {
                out.push(ExtMove {
                    mv: Move::make_drop(kind, stm, sq),
                    value: 0,
                });
            }
        }
    }

    /// An 81-square band-scanning form of [`Position::emit_drops_masked`], the
    /// `#[cfg(test)]` equivalence oracle for the bitboard-driven emitter — the
    /// same discipline as [`nifu_blocked_files_scan`] and the `see_ge` reference
    /// twin. The bitboard emitter must produce the byte-identical move Vec on
    /// every restriction mask.
    #[cfg(test)]
    fn emit_drops_masked_scan(&self, restrict: Bitboard, out: &mut Vec<Move>) {
        let board = self.board();
        let stm = self.side_to_move();
        let hand = self.hand(stm);

        if hand.count(PieceKind::Pawn) > 0 {
            let nifu_file = nifu_blocked_files(board, stm);
            let last_rank = if stm == Color::Black { 0 } else { 8 };

            let uchi_candidate = try_find_king(board, stm.flip()).and_then(|ek| {
                let dr: i16 = if stm == Color::Black { 1 } else { -1 };
                let r = ek.rank() as i16 + dr;
                if (0..Square::RANKS as i16).contains(&r) {
                    Square::new(ek.file(), r as u8)
                } else {
                    None
                }
            });

            for index in 0..Square::COUNT as u8 {
                let sq = Square::from_index(index).unwrap();
                if restrict.test(sq)
                    && board.get(sq).is_none()
                    && sq.rank() != last_rank
                    && !nifu_file[sq.file() as usize]
                    && !(Some(sq) == uchi_candidate && self.pawn_drop_is_uchifuzume(sq))
                {
                    out.push(Move::make_drop(PieceKind::Pawn, stm, sq));
                }
            }
        }

        let mut drops_buf = [PieceKind::Knight; DROP_ORDER.len()];
        let mut drops_len = 0usize;
        for &kind in &DROP_ORDER {
            if hand.count(kind) > 0 {
                drops_buf[drops_len] = kind;
                drops_len += 1;
            }
        }
        let drops = &drops_buf[..drops_len];
        if drops.is_empty() {
            return;
        }
        let next_to_knight = usize::from(drops.first() == Some(&PieceKind::Knight));
        let next_to_lance =
            next_to_knight + usize::from(drops.get(next_to_knight) == Some(&PieceKind::Lance));

        let (back_rank, second_rank): (u8, u8) = if stm == Color::Black { (0, 1) } else { (8, 7) };
        let in_band3 = |rank: u8| {
            if stm == Color::Black {
                rank >= 2
            } else {
                rank <= 6
            }
        };

        if next_to_lance == 0 {
            for index in 0..Square::COUNT as u8 {
                let sq = Square::from_index(index).unwrap();
                if restrict.test(sq) && board.get(sq).is_none() {
                    for &kind in drops {
                        out.push(Move::make_drop(kind, stm, sq));
                    }
                }
            }
            return;
        }

        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if sq.rank() == back_rank && restrict.test(sq) && board.get(sq).is_none() {
                for &kind in &drops[next_to_lance..] {
                    out.push(Move::make_drop(kind, stm, sq));
                }
            }
        }
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if sq.rank() == second_rank && restrict.test(sq) && board.get(sq).is_none() {
                for &kind in &drops[next_to_knight..] {
                    out.push(Move::make_drop(kind, stm, sq));
                }
            }
        }
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if in_band3(sq.rank()) && restrict.test(sq) && board.get(sq).is_none() {
                for &kind in drops {
                    out.push(Move::make_drop(kind, stm, sq));
                }
            }
        }
    }

    /// True iff dropping a side-to-move pawn on `sq` is uchifuzume (打ち歩詰め)
    /// — an immediate, unanswerable checkmate against the enemy king, which the
    /// rules forbid. This is now answered in place by the ported
    /// [`crate::movegen::drop_is_uchifuzume`] / [`Position::legal_drop`] — no
    /// board clone, no inner move generation — the same drop-legality port the
    /// drop generators enforce at generation time, so the two surfaces agree by
    /// construction. Callers restrict this to the single
    /// geometric candidate square, so the probe fires at most once per node.
    fn pawn_drop_is_uchifuzume(&self, sq: Square) -> bool {
        let m = Move::make_drop(PieceKind::Pawn, self.side_to_move(), sq);
        crate::movegen::drop_is_uchifuzume(self, m)
    }
}

/// Copy-apply-scan implementations of the three predicates, derived
/// independently of the cached check info so they can serve as its equivalence
/// oracles (the same discipline as `see_ge_reference`). The cached
/// predicates must agree with these on every move the engine's own generators
/// emit, along a deterministic playout; the `equivalence` test below enforces it.
#[cfg(test)]
impl Position {
    /// Oracle: apply `m` to a scratch board and test the enemy king against the
    /// full attacker scan (direct, discovered and dropping checks alike).
    pub(crate) fn gives_check_reference(&self, m: Move) -> bool {
        let mover = self.side_to_move();
        let mut board = *self.board();
        apply_to_board(&mut board, m, mover);
        match try_find_king(&board, mover.flip()) {
            Some(king_sq) => is_attacked_by(&board, king_sq, mover),
            None => false,
        }
    }

    /// Oracle: does a piece of the moved type, standing on `to`, attack the
    /// enemy king under the pre-move occupancy (direct checks only)?
    pub(crate) fn gives_direct_check_reference(&self, m: Move) -> bool {
        let board = self.board();
        let stm = self.side_to_move();
        let eks = match try_find_king(board, stm.flip()) {
            Some(k) => k,
            None => return false,
        };
        let piece = m.moved_piece_after();
        let to = m.to_sq();
        let dr_sign = dr_sign_for(piece.color);
        let (steps, slides) = movement(piece);
        for &(df, dr) in steps {
            if step_signed(to, df, dr * dr_sign) == Some(eks) {
                return true;
            }
        }
        for &(df, dr) in slides {
            let mut cur = to;
            loop {
                cur = match step_signed(cur, df, dr * dr_sign) {
                    Some(s) => s,
                    None => break,
                };
                if cur == eks {
                    return true;
                }
                if board.get(cur).is_some() {
                    break;
                }
            }
        }
        false
    }

    /// Oracle: apply `m` to a scratch board and test that the mover's own king
    /// is not left in check.
    pub(crate) fn is_legal_reference(&self, m: Move) -> bool {
        let mover = self.side_to_move();
        let mut board = *self.board();
        apply_to_board(&mut board, m, mover);
        match try_find_king(&board, mover) {
            Some(king_sq) => !is_attacked_by(&board, king_sq, mover.flip()),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::move_::format_usi_move;
    use crate::sfen::parse_sfen;

    fn pos(sfen: &str) -> Position {
        parse_sfen(sfen).expect("valid SFEN")
    }

    /// Unwrap the [`ExtMove`]s the generators emit into a plain `Move` Vec, so
    /// the assertions below (written against `Move`) stay unchanged.
    fn unwrap_ext(v: Vec<ExtMove>) -> Vec<Move> {
        v.into_iter().map(|e| e.mv).collect()
    }

    fn captures(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_captures(false, &mut v);
        unwrap_ext(v)
    }

    fn captures_all(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_captures(true, &mut v);
        unwrap_ext(v)
    }

    fn quiets_all(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_quiets(true, &mut v);
        unwrap_ext(v)
    }

    fn legal_captures(p: &Position) -> Vec<Move> {
        captures(p).into_iter().filter(|&m| p.is_legal(m)).collect()
    }

    fn evasions(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_evasions(false, &mut v);
        unwrap_ext(v)
    }

    fn legal_moves(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_legal_all(&mut v);
        v
    }

    fn usi(moves: &[Move]) -> Vec<String> {
        moves.iter().map(|&m| format_usi_move(m)).collect()
    }

    // ---- gives_check / in_check / is_legal --------------------------------

    /// The six perft-fixture SFENs (matching `tests/fixtures/perft/*.json`),
    /// reused for the seeded-playout oracle gate.
    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    /// Small deterministic xorshift64* (banned `Math.random`-style
    /// nondeterminism), mirroring the driver used elsewhere in this crate.
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

    /// The oracle: whether the position is in check *after* `m` is played — the
    /// same post-`do_move` fact the move-history machinery records. Computed
    /// via a `do_move`/`undo_move` round trip so it shares no code with
    /// `gives_check`'s bare-board application.
    fn gives_check_oracle(p: &Position, m: Move) -> bool {
        let mut scratch = p.clone();
        let undo = scratch.do_move(m);
        let checked = scratch.in_check();
        scratch.undo_move(m, undo);
        checked
    }

    #[test]
    fn gives_check_matches_oracle_over_random_playouts() {
        const MIN_PLIES: usize = 30;
        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut p = pos(sfen);
            let mut rng = Rng(0xD1B5_4A32_D192_ED03 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, crate::position::Undo)> = Vec::new();
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                let legal = legal_moves(&p);
                if legal.is_empty() {
                    while let Some((m, u)) = stack.pop() {
                        p.undo_move(m, u);
                    }
                    continue;
                }
                // Every legal move's `gives_check` must equal the oracle.
                for &m in &legal {
                    assert_eq!(
                        p.gives_check(m),
                        gives_check_oracle(&p, m),
                        "fixture {fi} ply {plies}: gives_check disagrees with oracle for {}",
                        format_usi_move(m),
                    );
                }
                let m = legal[rng.pick(legal.len())];
                let u = p.do_move(m);
                stack.push((m, u));
                plies += 1;
            }
        }
    }

    /// Every pseudo-legal move each of the engine's own generators can emit at
    /// `p`, across both `all` settings and every generator, concatenated. Used
    /// by the cached-check-info equivalence gate so the `gives_check` /
    /// `gives_direct_check` predicates are exercised on the exact moves the
    /// search would hand them. The target-restricted `generate_evasions` asserts
    /// `in_check` at entry, so it is only invoked on genuine
    /// in-check positions.
    fn all_generator_moves(p: &Position) -> Vec<Move> {
        let mut ext: Vec<ExtMove> = Vec::new();
        for all in [false, true] {
            p.generate_captures(all, &mut ext);
            p.generate_quiets(all, &mut ext);
            if p.in_check() {
                p.generate_evasions(all, &mut ext);
            }
            p.generate_non_evasions(all, &mut ext);
        }
        let mut v = unwrap_ext(ext);
        p.generate_legal_all(&mut v);
        v
    }

    /// The moves `is_legal` may legitimately be handed at `p` — its O(1)
    /// contract: the check-state-appropriate search generator's
    /// output (evasions when in check, else captures / quiets / non-evasions),
    /// plus every candidate that passes `pseudo_legal` (the TT / probcut path).
    /// Outside this set the O(1) `is_legal` may legitimately disagree with the
    /// copy-apply-scan oracle (e.g. a non-check-resolving in-check drop), so the
    /// equivalence gate must not exercise it there — no non-test caller does.
    fn is_legal_contract_moves(p: &Position) -> Vec<Move> {
        let mut ext: Vec<ExtMove> = Vec::new();
        for all in [false, true] {
            if p.in_check() {
                p.generate_evasions(all, &mut ext);
            } else {
                p.generate_captures(all, &mut ext);
                p.generate_quiets(all, &mut ext);
                p.generate_non_evasions(all, &mut ext);
            }
        }
        let mut v = unwrap_ext(ext);
        // Every all-generator move that passes `pseudo_legal` also meets the
        // contract — this covers the TT-widened candidates the search validates
        // through `pseudo_legal` before calling `is_legal`.
        for m in all_generator_moves(p) {
            if p.pseudo_legal(m, false) || p.pseudo_legal(m, true) {
                v.push(m);
            }
        }
        v
    }

    /// The cached-check-info predicates (`gives_check`,
    /// `gives_direct_check`, `is_legal`) must agree with the verbatim scratch
    /// oracles for *every* move produced by *every* generator, at each fixture
    /// and along a deterministic playout of at least 30 plies from each.
    /// `is_legal` is exercised only over its contract set (see
    /// [`is_legal_contract_moves`]).
    #[test]
    fn cached_check_info_predicates_match_scratch_oracles_over_playouts() {
        const MIN_PLIES: usize = 30;
        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut p = pos(sfen);
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, crate::position::Undo)> = Vec::new();
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                for &m in &all_generator_moves(&p) {
                    assert_eq!(
                        p.gives_check(m),
                        p.gives_check_reference(m),
                        "fixture {fi} ply {plies}: gives_check mismatch for {}",
                        format_usi_move(m),
                    );
                    assert_eq!(
                        p.gives_direct_check(m),
                        p.gives_direct_check_reference(m),
                        "fixture {fi} ply {plies}: gives_direct_check mismatch for {}",
                        format_usi_move(m),
                    );
                }
                for &m in &is_legal_contract_moves(&p) {
                    assert_eq!(
                        p.is_legal(m),
                        p.is_legal_reference(m),
                        "fixture {fi} ply {plies}: is_legal mismatch for {}",
                        format_usi_move(m),
                    );
                }

                let legal = legal_moves(&p);
                if legal.is_empty() {
                    while let Some((m, u)) = stack.pop() {
                        p.undo_move(m, u);
                    }
                    continue;
                }
                let m = legal[rng.pick(legal.len())];
                let u = p.do_move(m);
                stack.push((m, u));
                plies += 1;
            }
        }
    }

    #[test]
    fn in_check_true_only_when_king_attacked() {
        // White rook on the 5-file gives check to the black king below it.
        let checked = pos("4r4/9/9/9/9/9/9/9/4K4 b - 1");
        assert!(checked.in_check());
        // Same board, rook shifted one file: no check.
        let quiet = pos("3r5/9/9/9/9/9/9/9/4K4 b - 1");
        assert!(!quiet.in_check());
    }

    #[test]
    fn is_legal_agrees_with_legal_set_for_pin_and_king_step() {
        // Black king 5a, black silver 5b, white rook 5i on the open 5-file; the
        // silver is pinned. A diagonal silver step exposes the king (illegal); a
        // sideways black-king step to an unattacked square is legal. `is_legal`
        // must agree with the full legal-move set on both.
        let p = pos("4K3k/4S4/9/9/9/9/9/9/4r4 b - 1");
        let legal: std::collections::HashSet<Move> = legal_moves(&p).into_iter().collect();

        let silver = Piece::new(PieceKind::Silver, Color::Black);
        let off_ray = Move::make(
            Square::new(4, 1).unwrap(),
            Square::new(3, 2).unwrap(),
            silver,
        );
        assert!(
            !p.is_legal(off_ray),
            "moving the pinned silver off the file must be illegal"
        );
        assert!(!legal.contains(&off_ray));

        let king = Piece::new(PieceKind::King, Color::Black);
        let king_step = Move::make(Square::new(4, 0).unwrap(), Square::new(5, 0).unwrap(), king);
        assert!(p.is_legal(king_step), "a safe king step must be legal");
        assert!(legal.contains(&king_step));
    }

    // ---- CAPTURES generator -----------------------------------------------

    #[test]
    fn pawn_capture_in_zone_is_promotion_only() {
        // Black pawn 5c captures a white pawn 5b (rank 1, inside the zone).
        // Only the promotion is generated; the non-promotion is suppressed.
        let p = pos("k8/4p4/4P4/9/9/9/9/9/8K b - 1");
        let caps = captures(&p);
        assert_eq!(caps.len(), 1, "expected one capture, got {:?}", usi(&caps));
        assert!(
            caps[0].is_promote(),
            "pawn capture into zone must promote: {:?}",
            usi(&caps)
        );
        assert_eq!(format_usi_move(caps[0]), "5c5b+");
    }

    #[test]
    fn lance_capture_on_enemy_second_rank_is_promotion_only() {
        // Black lance 5d captures a white piece on 5b (rank 1). Promotion only:
        // the non-promotion is suppressed on the enemy first/second rank.
        let p = pos("k8/4p4/9/4L4/9/9/9/9/8K b - 1");
        let caps = captures(&p);
        assert_eq!(caps.len(), 1, "got {:?}", usi(&caps));
        assert!(caps[0].is_promote());
        assert_eq!(format_usi_move(caps[0]), "5d5b+");
    }

    #[test]
    fn lance_capture_on_enemy_third_rank_keeps_both_variants() {
        // Black lance 5e captures a white piece on 5c (rank 2). Both the
        // promotion and the non-promotion are generated (rank 2 keeps the
        // non-promotion), promotion first.
        let p = pos("k8/9/4p4/9/4L4/9/9/9/8K b - 1");
        let caps = captures(&p);
        assert_eq!(usi(&caps), vec!["5e5c+".to_string(), "5e5c".to_string()]);
    }

    // ---- GenerateAllLegalMoves (all == true) ------------------------------

    #[test]
    fn all_pawn_capture_in_zone_adds_nonpromotion() {
        // Same position as `pawn_capture_in_zone_is_promotion_only`: a Black pawn
        // 5c captures a white pawn on 5b (rank 1, inside the zone). With the flag
        // OFF only the promotion is generated; with the flag ON the non-promotion
        // is additionally generated (rank 1 is not the last rank), promotion
        // first (`movegen.cpp`).
        let p = pos("k8/4p4/4P4/9/9/9/9/9/8K b - 1");
        assert_eq!(usi(&captures(&p)), vec!["5c5b+".to_string()]);
        assert_eq!(
            usi(&captures_all(&p)),
            vec!["5c5b+".to_string(), "5c5b".to_string()],
            "all-mode adds the pawn non-promotion",
        );
    }

    #[test]
    fn all_lance_capture_on_enemy_second_rank_adds_nonpromotion() {
        // `lance_capture_on_enemy_second_rank_is_promotion_only`: Black lance 5d
        // captures on 5b (rank 1). The default suppresses the non-promotion on
        // the enemy first/second rank; `all` widens the mask so only the very
        // last rank is suppressed, so the rank-1 non-promotion now appears.
        let p = pos("k8/4p4/9/4L4/9/9/9/9/8K b - 1");
        assert_eq!(usi(&captures(&p)), vec!["5d5b+".to_string()]);
        assert_eq!(
            usi(&captures_all(&p)),
            vec!["5d5b+".to_string(), "5d5b".to_string()],
            "all-mode adds the lance non-promotion on the enemy second rank",
        );
    }

    #[test]
    fn all_pawn_last_rank_push_is_still_promotion_only() {
        // A Black pawn on 5b pushing to the empty 5a (rank 0, the last rank): a
        // non-promotion there would be a stuck pawn, so it is suppressed even
        // with `all` on (`movegen.cpp`, `rank_of(to) != T_RANK1`).
        let p = pos("k8/4P4/9/9/9/9/9/9/8K b - 1");
        let promo = "5b5a+".to_string();
        assert_eq!(
            usi(&quiets_all(&p))
                .into_iter()
                .filter(|s| s.starts_with("5b5a"))
                .collect::<Vec<_>>(),
            vec![promo],
            "last-rank pawn push stays promotion-only even in all-mode",
        );
    }

    #[test]
    fn all_bishop_promotion_interleaves_nonpromotion() {
        // A Black bishop on 5e capturing into the zone: with the flag off only
        // the promotion is generated for an in-zone destination; with the flag on
        // the non-promotion is emitted right after the promotion, per destination
        // (`movegen.cpp`). Bishop 5e, white pawn on 3c (a diagonal in the
        // zone).
        let p = pos("k8/9/6p2/9/4B4/9/9/9/8K b - 1");
        let off: Vec<String> = usi(&captures(&p));
        let on: Vec<String> = usi(&captures_all(&p));
        assert!(
            off.contains(&"5e3c+".to_string()) && !off.contains(&"5e3c".to_string()),
            "default: promotion only, got {off:?}",
        );
        // In all-mode the non-promotion follows its promotion immediately.
        let i = on
            .iter()
            .position(|s| s == "5e3c+")
            .expect("promotion present");
        assert_eq!(
            on.get(i + 1).map(String::as_str),
            Some("5e3c"),
            "all-mode interleaves the bishop non-promotion after the promotion: {on:?}",
        );
    }

    #[test]
    fn captures_are_piece_type_major_and_carry_no_drops() {
        // Three independent captures outside the promotion zone (so each is a
        // single non-promoting move): white victims on rank e (our rank 4),
        // black attackers directly behind on rank f (our rank 5).
        //   USI file 9 (our file 0): pawn victim / pawn attacker
        //   USI file 7 (our file 2): silver victim / silver attacker
        //   USI file 5 (our file 4): gold victim / gold attacker
        // Black also holds a pawn in hand — CAPTURES must emit no drop.
        let p = pos("k8/9/9/9/p1s1g4/P1S1G4/9/9/7K1 b P 1");
        let caps = captures(&p);
        // Expected piece-type-major order: pawn, then silver, then gold.
        assert_eq!(
            usi(&caps),
            vec!["9f9e".to_string(), "7f7e".to_string(), "5f5e".to_string()],
            "captures must be piece-type-major with no drops",
        );
        assert!(
            caps.iter().all(|m| !m.is_drop()),
            "CAPTURES must not include drops"
        );
    }

    #[test]
    fn captures_equal_legal_captures_when_none_are_promotions() {
        // Sanity cross-check on a middlegame fixture: every generated capture is
        // a capture (lands on an enemy piece), and the legal ones are a subset
        // of the full legal move list.
        let p =
            pos("l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1");
        let board = p.board();
        for &m in &captures(&p) {
            assert!(!m.is_drop());
            let victim = board.get(m.to_sq());
            assert!(
                victim.is_some_and(|v| v.color != p.side_to_move()),
                "CAPTURES move {} does not land on an enemy piece",
                format_usi_move(m),
            );
        }
        let legal: std::collections::HashSet<Move> = legal_moves(&p).into_iter().collect();
        for &m in &legal_captures(&p) {
            assert!(
                legal.contains(&m),
                "legal capture {} missing from legal set",
                format_usi_move(m)
            );
        }
    }

    // ---- EVASIONS generator -----------------------------------------------

    #[test]
    fn evasions_are_generated_only_when_in_check() {
        // `generate_evasions` asserts `in_check` at entry: the
        // picker only calls it when `in_check()` holds, so this position is
        // never handed to it.
        let quiet = pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        assert!(!quiet.in_check());
        // In check: there is at least one legal evasion.
        let checked = pos("4r4/9/9/9/9/9/9/9/4K4 b - 1");
        assert!(checked.in_check());
        let legal_ev: Vec<Move> = evasions(&checked)
            .into_iter()
            .filter(|&m| checked.is_legal(m))
            .collect();
        assert!(
            !legal_ev.is_empty(),
            "an in-check position must have legal evasions"
        );
    }

    #[test]
    fn evasion_king_moves_come_first_and_set_matches_legal() {
        // Black king 5i is checked by the white rook on the 5-file. Evasions:
        // king steps off the file (the rook can be neither captured nor
        // blocked here). Every legal evasion is a king move, emitted first, and
        // the legal-filtered evasion set equals the full legal move set.
        let p = pos("4r4/9/9/9/9/9/9/9/4K4 b - 1");
        let legal_ev: Vec<Move> = evasions(&p)
            .into_iter()
            .filter(|&m| p.is_legal(m))
            .collect();
        let legal_all: std::collections::HashSet<Move> = legal_moves(&p).into_iter().collect();
        let ev_set: std::collections::HashSet<Move> = legal_ev.iter().copied().collect();
        assert_eq!(
            ev_set, legal_all,
            "legal evasions must equal the legal move set"
        );
        // All are king moves (from 5i), and king moves lead the emission.
        let king_from = Square::new(4, 8).unwrap();
        assert!(
            legal_ev
                .iter()
                .all(|m| !m.is_drop() && m.from_sq() == king_from),
            "every evasion here is a king move: {:?}",
            usi(&legal_ev),
        );
    }

    #[test]
    fn evasions_include_blocking_drops_in_check() {
        // Black king 5i, white rook 5a checking down the open 5-file, black
        // holds a gold. The seven interposition gold-drops on the 5-file are
        // legal evasions; they appear after the king moves.
        let p = pos("4r4/9/9/9/9/9/9/9/4K4 b G 1");
        let legal_ev: Vec<Move> = evasions(&p)
            .into_iter()
            .filter(|&m| p.is_legal(m))
            .collect();
        let drops: Vec<Move> = legal_ev.iter().copied().filter(|m| m.is_drop()).collect();
        assert_eq!(
            drops.len(),
            7,
            "seven blocking gold-drops expected: {:?}",
            usi(&drops)
        );
        assert!(
            drops
                .iter()
                .all(|m| m.to_sq().file() == 4 && (1..=7).contains(&m.to_sq().rank())),
            "gold drops must interpose on the 5-file: {:?}",
            usi(&drops),
        );
        // King moves precede every drop in the emission order.
        let first_drop = legal_ev.iter().position(|m| m.is_drop()).unwrap();
        assert!(
            legal_ev[..first_drop].iter().all(|m| !m.is_drop()),
            "king / piece moves must precede drops: {:?}",
            usi(&legal_ev),
        );
    }

    #[test]
    fn drop_generation_excludes_uchifuzume() {
        // A Black pawn dropped at (8,1) — directly in front of the White king on
        // 8a — is uchifuzume (unanswerable pawn-drop mate). The shared drop
        // generator [`Position::emit_drops`] (fed by both `generate_evasions`
        // and `generate_quiets`) must exclude it at generation, exactly as the
        // reference `GenerateDropMoves`' `legal_drop` does; the qsearch legality
        // filter `is_legal` is uchifuzume-blind, so leaving it in would surface
        // an illegal move. Black is not in check here, so the shared filter is
        // exercised through the quiets path (the restricted `generate_evasions`
        // now asserts `in_check`, but shares the same `emit_drops`).
        // (Same mating shape as `movegen::tests::uchifuzume_filters_pawn_drop_mate`.)
        let p = pos("k8/9/G1N6/9/9/9/9/9/8K b P 1");
        assert!(!p.in_check());
        let qs = quiets(&p);

        let mating_drop =
            Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            !qs.contains(&mating_drop),
            "uchifuzume pawn drop leaked into drop generation: {:?}",
            usi(&qs),
        );

        // The exclusion is surgical: an ordinary pawn drop elsewhere on the same
        // (non-nifu) file is still emitted.
        let other_drop = Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 3).unwrap());
        assert!(
            qs.contains(&other_drop),
            "a non-uchifuzume pawn drop was wrongly excluded: {:?}",
            usi(&qs),
        );
    }

    // ---- QUIETS generator -------------------------------------------------

    fn quiets(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_quiets(false, &mut v);
        unwrap_ext(v)
    }

    #[test]
    fn quiets_land_only_on_empty_squares_and_include_drops() {
        // Black rook 5e (quiet lifts to empty squares), a black pawn in hand
        // (drops), and a lone enemy pawn on 5g the rook could capture — the
        // capture must NOT appear among the quiets.
        let p = pos("k8/9/9/9/4R4/9/4p4/9/8K b P 1");
        let qs = quiets(&p);
        // No quiet lands on an occupied square.
        for &m in &qs {
            if !m.is_drop() {
                assert!(
                    p.board().get(m.to_sq()).is_none(),
                    "quiet {} lands on an occupied square",
                    format_usi_move(m),
                );
            }
        }
        // The rook's capture of the 5g pawn is a capture, not a quiet.
        let cap = Move::make(
            Square::new(4, 4).unwrap(),
            Square::new(4, 6).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert!(!qs.contains(&cap), "a capture leaked into QUIETS");
        // Some pawn drops are generated.
        assert!(
            qs.iter().any(|m| m.is_drop()),
            "expected pawn drops among the quiets"
        );
    }

    #[test]
    fn quiets_include_non_capturing_pawn_promotion() {
        // Black pawn 5d pushes to the empty 5c (rank 2, inside the zone): a
        // non-capturing promotion, which belongs to QUIETS (plain CAPTURES,
        // targeting enemy pieces only, would miss it).
        let p = pos("k8/9/9/4P4/9/9/9/9/8K b - 1");
        let qs = quiets(&p);
        let promo = Move::make_promote(
            Square::new(4, 3).unwrap(),
            Square::new(4, 2).unwrap(),
            Piece::new(PieceKind::Pawn, Color::Black),
        );
        assert_eq!(format_usi_move(promo), "5d5c+");
        assert!(
            qs.contains(&promo),
            "non-capturing pawn promotion missing from QUIETS: {:?}",
            usi(&qs),
        );
    }

    #[test]
    fn captures_and_quiets_partition_the_legal_moves_no_divergence() {
        // A position free of promotion-suppression divergence (golds, kings,
        // and pieces outside the promotion zone): the legal captures ∪ legal
        // quiets equal the full legal move set, and the two are disjoint.
        let p = pos("9/9/3g1g3/9/4G4/9/9/9/K7k b G 1");
        let legal: std::collections::HashSet<Move> = legal_moves(&p).into_iter().collect();
        let lc: std::collections::HashSet<Move> = legal_captures(&p).into_iter().collect();
        let lq: std::collections::HashSet<Move> =
            quiets(&p).into_iter().filter(|&m| p.is_legal(m)).collect();
        assert!(lc.is_disjoint(&lq), "a move is both a capture and a quiet");
        let union: std::collections::HashSet<Move> = lc.union(&lq).copied().collect();
        assert_eq!(union, legal, "captures ∪ quiets must equal the legal set");
    }

    // ---- gives_direct_check ----------------------------------------------

    #[test]
    fn gives_direct_check_true_for_quiet_rook_check() {
        // Black rook 9c; a quiet move 9c→5c gives a direct check to the white
        // king on 5a along the (empty) 5-file.
        let p = pos("4k4/9/R8/9/9/9/9/9/8K b - 1");
        let check = Move::make(
            Square::new(8, 2).unwrap(),
            Square::new(4, 2).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert_eq!(format_usi_move(check), "9c5c");
        assert!(p.gives_direct_check(check));
        // A sideways lift that does not reach the king's file is not a check.
        let quiet = Move::make(
            Square::new(8, 2).unwrap(),
            Square::new(7, 2).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert!(!p.gives_direct_check(quiet));
    }

    #[test]
    fn gives_direct_check_ignores_discovered_checks() {
        // Black rook 5e behind a black gold 5c, white king 5a on the same file.
        // Moving the gold off the file (5c→4c) uncovers the rook → a discovered
        // check, but NOT a *direct* one by the gold, so gives_direct_check is
        // false (it models `check_squares(pt) & to`, direct checks only).
        let p = pos("4k4/9/4G4/9/4R4/9/9/9/8K b - 1");
        let discover = Move::make(
            Square::new(4, 2).unwrap(),
            Square::new(3, 2).unwrap(),
            Piece::new(PieceKind::Gold, Color::Black),
        );
        assert_eq!(format_usi_move(discover), "5c4c");
        assert!(p.is_legal(discover));
        assert!(
            !p.gives_direct_check(discover),
            "a discovered check is not a direct check"
        );
        // Sanity: the full gives_check (direct + discovered) does see it.
        assert!(p.gives_check(discover));
    }

    // ---- generate_non_evasions --------------------------------------------

    /// The pseudo-legal `NON_EVASIONS` list holds exactly the union of the
    /// `CAPTURES` and `QUIETS` lists (same moves, no duplicates) — the single
    /// `~pieces(Us)` target is the union of the enemy-only and empty-only
    /// targets — for the not-in-check fixtures.
    #[test]
    fn non_evasions_set_equals_captures_union_quiets() {
        for sfen in FIXTURE_SFENS {
            let p = pos(sfen);
            if p.in_check() {
                continue;
            }
            let mut non_ev = Vec::new();
            p.generate_non_evasions(false, &mut non_ev);
            let non_ev = unwrap_ext(non_ev);
            let mut caps = Vec::new();
            p.generate_captures(false, &mut caps);
            let caps = unwrap_ext(caps);
            let mut quiets = Vec::new();
            p.generate_quiets(false, &mut quiets);
            let quiets = unwrap_ext(quiets);

            let ne_set: std::collections::HashSet<Move> = non_ev.iter().copied().collect();
            let cq_set: std::collections::HashSet<Move> =
                caps.iter().chain(quiets.iter()).copied().collect();
            assert_eq!(
                ne_set.len(),
                non_ev.len(),
                "{sfen}: NON_EVASIONS has a duplicate"
            );
            assert_eq!(
                ne_set, cq_set,
                "{sfen}: NON_EVASIONS set must equal CAPTURES ∪ QUIETS"
            );
            // And the legal subsets agree (the set the root move list / picker
            // ultimately search).
            let ne_legal: std::collections::HashSet<Move> =
                non_ev.into_iter().filter(|&m| p.is_legal(m)).collect();
            let cq_legal: std::collections::HashSet<Move> = caps
                .into_iter()
                .chain(quiets)
                .filter(|&m| p.is_legal(m))
                .collect();
            assert_eq!(
                ne_legal, cq_legal,
                "{sfen}: legal NON_EVASIONS set mismatch"
            );
        }
    }

    /// `NON_EVASIONS` interleaves captures and quiets per piece and per
    /// destination square, so — for a position that offers both — its order is
    /// *not* the captures-then-quiets concatenation, even though the underlying
    /// set is identical. This is exactly the distinction that fixes the root
    /// search's TT move (`rootMoves[0]`).
    #[test]
    fn non_evasions_interleaves_rather_than_concatenating() {
        let p =
            pos("l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1");
        assert!(!p.in_check());

        let mut caps = Vec::new();
        p.generate_captures(false, &mut caps);
        let caps = unwrap_ext(caps);
        assert!(!caps.is_empty(), "fixture must offer at least one capture");

        let mut non_ev = Vec::new();
        p.generate_non_evasions(false, &mut non_ev);
        let non_ev = unwrap_ext(non_ev);
        let concat: Vec<Move> = {
            let mut quiets = Vec::new();
            p.generate_quiets(false, &mut quiets);
            let quiets = unwrap_ext(quiets);
            caps.iter().chain(quiets.iter()).copied().collect()
        };

        assert_eq!(non_ev.len(), concat.len(), "same move count");
        assert_ne!(
            usi(&non_ev),
            usi(&concat),
            "NON_EVASIONS must interleave captures and quiets, not concatenate them"
        );
    }
}

// ===========================================================================
// Bitboard-rewrite gates: piece-set consistency and attack-query equivalence
// ===========================================================================
#[cfg(test)]
mod gate_262 {
    use super::{CHECK_PATTERN_COUNT, attack_set_from_scan, attacks_bb, check_pattern};
    use crate::bitboard::Bitboard;
    use crate::board::{Board, PATTERN_COUNT, pattern_of};
    use crate::color::Color;
    use crate::movegen::{attackers_bb, is_attacked_by, is_attacked_by_scan, try_find_king};
    use crate::piece::{Piece, PieceKind};
    use crate::position::Position;
    use crate::sfen::parse_sfen;
    use crate::square::Square;

    /// The fixtures the gates sweep — the perft parity positions plus the bench
    /// set: startpos-like, midgame-tactical, promotion-zone edges, drop-heavy,
    /// check-evasion, and the move-generation "festival" position.
    const FIXTURES: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",
        "l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",
        "6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",
        "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
    ];

    fn all_squares() -> impl Iterator<Item = Square> {
        (0..Square::COUNT as u8).map(|i| Square::from_index(i).unwrap())
    }

    /// Rebuild `(occupied, by_color, by_pattern)` from scratch by scanning the
    /// board — the equivalence oracle for the incrementally maintained sets.
    fn scan_sets(board: &Board) -> (Bitboard, [Bitboard; 2], [[Bitboard; PATTERN_COUNT]; 2]) {
        let mut occ = Bitboard::empty();
        let mut by_color = [Bitboard::empty(); 2];
        let mut by_pattern = [[Bitboard::empty(); PATTERN_COUNT]; 2];
        for sq in all_squares() {
            if let Some(p) = board.get(sq) {
                let bit = Bitboard::from_square(sq);
                occ |= bit;
                by_color[p.color.index()] |= bit;
                by_pattern[p.color.index()][pattern_of(p)] |= bit;
            }
        }
        (occ, by_color, by_pattern)
    }

    /// The incrementally maintained sets equal a from-scratch scan.
    fn assert_sets_consistent(pos: &Position, ctx: &str) {
        let board = pos.board();
        let (occ, by_color, by_pattern) = scan_sets(board);
        assert_eq!(board.occupied(), occ, "occupied @ {ctx}");
        for color in [Color::Black, Color::White] {
            assert_eq!(
                board.pieces_color(color),
                by_color[color.index()],
                "by_color {color:?} @ {ctx}",
            );
            for (pat, &expected) in by_pattern[color.index()].iter().enumerate() {
                assert_eq!(
                    board.pieces_pattern(color, pat),
                    expected,
                    "by_pattern {color:?} {pat} @ {ctx}",
                );
            }
        }
    }

    /// A checkers scan (reverse-of-`attack_set_from_scan`), the oracle for the
    /// `attackers_bb`-based checkers query.
    fn checkers_scan(board: &Board, king: Square, enemy: Color) -> Bitboard {
        let king_bb = Bitboard::from_square(king);
        let mut set = Bitboard::empty();
        for sq in all_squares() {
            if let Some(p) = board.get(sq)
                && p.color == enemy
                && !(attack_set_from_scan(board, sq, p) & king_bb).is_empty()
            {
                set |= Bitboard::from_square(sq);
            }
        }
        set
    }

    /// Every attack query equals its scanning oracle at `pos`.
    fn assert_attack_equiv(pos: &Position, ctx: &str) {
        let board = pos.board();
        let occ = board.occupied();

        // 2a: is_attacked_by == scan, all squares × both colours.
        for sq in all_squares() {
            for attacker in [Color::Black, Color::White] {
                assert_eq!(
                    is_attacked_by(board, sq, attacker),
                    is_attacked_by_scan(board, sq, attacker),
                    "is_attacked_by {sq:?} {attacker:?} @ {ctx}",
                );
            }
        }

        // 2b: is_attacked_discounting == the scan with `discount` vacated, over
        // every occupied square as the discount square.
        for discount in all_squares() {
            if board.get(discount).is_none() {
                continue;
            }
            let mut vacated = *board;
            vacated.set(discount, None);
            for sq in all_squares() {
                for attacker in [Color::Black, Color::White] {
                    assert_eq!(
                        pos.is_attacked_discounting(sq, attacker, discount),
                        is_attacked_by_scan(&vacated, sq, attacker),
                        "is_attacked_discounting {sq:?} {attacker:?} discount {discount:?} @ {ctx}",
                    );
                }
            }
        }

        // 2c: attacks_bb == attack_set_from_scan for every piece on the board,
        // and for each check-pattern piece imagined on the enemy king square.
        for from in all_squares() {
            if let Some(p) = board.get(from) {
                assert_eq!(
                    attacks_bb(from, p, occ),
                    attack_set_from_scan(board, from, p),
                    "attacks_bb piece {p:?} from {from:?} @ {ctx}",
                );
            }
        }
        for enemy in [Color::Black, Color::White] {
            if let Some(eks) = try_find_king(board, enemy) {
                // The ten check-pattern representative pieces the check-info fill
                // imagines on the enemy king.
                let reps = [
                    Piece::new(PieceKind::Pawn, enemy),
                    Piece::new(PieceKind::Lance, enemy),
                    Piece::new(PieceKind::Knight, enemy),
                    Piece::new(PieceKind::Silver, enemy),
                    Piece::new(PieceKind::Gold, enemy),
                    Piece::new(PieceKind::Bishop, enemy),
                    Piece::new(PieceKind::Rook, enemy),
                    Piece::promoted(PieceKind::Bishop, enemy).unwrap(),
                    Piece::promoted(PieceKind::Rook, enemy).unwrap(),
                    Piece::new(PieceKind::King, enemy),
                ];
                for p in reps {
                    assert_eq!(
                        attacks_bb(eks, p, occ),
                        attack_set_from_scan(board, eks, p),
                        "attacks_bb check-pattern {p:?} on king {eks:?} @ {ctx}",
                    );
                }
            }
        }

        // Checkers rewrite: attackers_bb from each king equals the scan.
        for color in [Color::Black, Color::White] {
            if let Some(king) = try_find_king(board, color) {
                assert_eq!(
                    attackers_bb(board, king, color.flip()),
                    checkers_scan(board, king, color.flip()),
                    "attackers_bb (checkers) king {color:?} @ {ctx}",
                );
            }
        }

        // The pattern maps line up (defends the numeric slots against drift).
        assert_eq!(CHECK_PATTERN_COUNT, PATTERN_COUNT);
        for p in [
            Piece::new(PieceKind::Gold, Color::Black),
            Piece::promoted(PieceKind::Pawn, Color::White).unwrap(),
            Piece::promoted(PieceKind::Rook, Color::Black).unwrap(),
        ] {
            assert_eq!(check_pattern(p), pattern_of(p));
        }
    }

    /// Run both gates at `pos`.
    fn assert_all(pos: &Position, ctx: &str) {
        assert_sets_consistent(pos, ctx);
        assert_attack_equiv(pos, ctx);
    }

    /// A tiny deterministic LCG so the playout is reproducible without a `rand`
    /// dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
    }

    #[test]
    fn gate_set_consistency_and_attack_equivalence_along_playouts() {
        const PLIES: usize = 32;
        for (fx, sfen) in FIXTURES.iter().enumerate() {
            let mut pos = parse_sfen(sfen).unwrap_or_else(|e| panic!("fixture {sfen}: {e:?}"));
            let mut rng = Lcg(0x9E3779B97F4A7C15 ^ (fx as u64).wrapping_mul(0x1000_0001));
            let mut buf = Vec::new();
            for ply in 0..PLIES {
                let ctx = format!("fx{fx} ply{ply}");
                assert_all(&pos, &ctx);

                // Null-move do/undo at positions not in check (board unchanged →
                // sets must be untouched, and restored after undo).
                if !pos.in_check() {
                    pos.do_null_move();
                    assert_sets_consistent(&pos, &format!("{ctx} post-null"));
                    pos.undo_null_move();
                    assert_sets_consistent(&pos, &format!("{ctx} post-null-undo"));
                }

                buf.clear();
                pos.generate_legal_all(&mut buf);
                if buf.is_empty() {
                    break;
                }
                let m = buf[(rng.next() >> 33) as usize % buf.len()];

                // do/undo round trip: sets consistent after do AND restored
                // after undo.
                let undo = pos.do_move(m);
                assert_sets_consistent(&pos, &format!("{ctx} post-do {m:?}"));
                pos.undo_move(m, undo);
                assert_sets_consistent(&pos, &format!("{ctx} post-undo {m:?}"));

                // Commit the move and walk on.
                pos.do_move(m);
            }
        }
    }
}

// ===========================================================================
// Move-generator emission-sequence equivalence
// ===========================================================================
//
// The binding constraint on the piece-set generators is the *emission
// sequence*: node-count parity depends on the exact order moves come out, so the
// production generators (piece-set bitboards) must emit the byte-identical move
// Vec the scanning oracles do. This gate compares them
// move-for-move — for every parity fixture, along a deterministic playout that
// visits in-check positions (exercising the evasion path), across every
// generator and BOTH `all` settings — and pins the file-mask nifu mask against
// its scanning oracle.
#[cfg(test)]
mod gate_267 {
    use super::{ExtMove, nifu_blocked_files, nifu_blocked_files_scan};
    use crate::color::Color;
    use crate::move_::{Move, format_usi_move};
    use crate::position::{Position, Undo};
    use crate::sfen::parse_sfen;

    /// The six perft / bench parity fixtures (all two-king, so the playout's
    /// `generate_legal_all` is well-defined). The second — a black king on 5a
    /// checked by a white rook on 5c — starts *in check*, so the evasion
    /// generator is exercised on a genuine in-check state from ply 0; the random
    /// playouts visit further checks along the way.
    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

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

    /// The four search generators, keyed by index, run for a given `all`. The
    /// generators emit [`ExtMove`]; this oracle unwraps `.mv` so
    /// the move-for-move gate comparisons below are unchanged.
    fn production(p: &Position, which: usize, all: bool) -> Vec<Move> {
        let mut v: Vec<ExtMove> = Vec::new();
        match which {
            0 => p.generate_captures(all, &mut v),
            1 => p.generate_quiets(all, &mut v),
            2 => p.generate_evasions(all, &mut v),
            3 => p.generate_non_evasions(all, &mut v),
            _ => unreachable!(),
        }
        v.into_iter().map(|e| e.mv).collect()
    }

    /// The matching scanning oracle for generator `which`.
    fn scan(p: &Position, which: usize, all: bool) -> Vec<Move> {
        let mut v: Vec<ExtMove> = Vec::new();
        match which {
            0 => p.generate_captures_scan(all, &mut v),
            1 => p.generate_quiets_scan(all, &mut v),
            2 => p.generate_evasions_scan(all, &mut v),
            3 => p.generate_non_evasions_scan(all, &mut v),
            _ => unreachable!(),
        }
        v.into_iter().map(|e| e.mv).collect()
    }

    const GEN_NAMES: [&str; 4] = ["captures", "quiets", "evasions", "non_evasions"];

    /// Every generator × both `all` settings emits the exact same move Vec as
    /// its scanning oracle; the nifu mask agrees with its scan for both colours.
    ///
    /// The evasion generator (index 2) is target-restricted, so its
    /// *raw* emission is a subsequence of the unrestricted scan twin. The binding
    /// invariant there is that the **legal-filtered** sequences match element-for-
    /// element: production filtered by the O(1) `is_legal`, the unrestricted twin
    /// filtered by the copy-apply-scan oracle `leaves_own_king_safe`. It is only
    /// compared on genuine in-check positions (its entry contract).
    fn assert_all_generators_match(p: &Position, ctx: &str) {
        for (which, name) in GEN_NAMES.iter().enumerate() {
            let is_evasions = which == 2;
            if is_evasions && !p.in_check() {
                continue;
            }
            for all in [false, true] {
                let mut prod = production(p, which, all);
                let mut scanned = scan(p, which, all);
                if is_evasions {
                    prod.retain(|&m| p.is_legal(m));
                    scanned.retain(|&m| p.leaves_own_king_safe(m));
                }
                assert_eq!(
                    prod.len(),
                    scanned.len(),
                    "{ctx}: {name} (all={all}) length differs — prod {:?} vs scan {:?}",
                    prod.iter().map(|&m| format_usi_move(m)).collect::<Vec<_>>(),
                    scanned
                        .iter()
                        .map(|&m| format_usi_move(m))
                        .collect::<Vec<_>>(),
                );
                for (i, (&a, &b)) in prod.iter().zip(scanned.iter()).enumerate() {
                    assert_eq!(
                        a,
                        b,
                        "{ctx}: {name} (all={all}) diverges at index {i}: prod {} vs scan {}",
                        format_usi_move(a),
                        format_usi_move(b),
                    );
                }
            }
        }
        for stm in [Color::Black, Color::White] {
            assert_eq!(
                nifu_blocked_files(p.board(), stm),
                nifu_blocked_files_scan(p.board(), stm),
                "{ctx}: nifu mask ({stm:?}) differs from scan",
            );
        }
    }

    #[test]
    fn generators_emit_byte_identical_sequences_over_playouts() {
        const MIN_PLIES: usize = 30;
        let mut in_check_visits = 0usize;
        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut p = parse_sfen(sfen).expect("valid SFEN");
            let mut rng = Rng(0x1D8E_3A55_9C6B_2F17 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, Undo)> = Vec::new();
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                if p.in_check() {
                    in_check_visits += 1;
                }
                assert_all_generators_match(&p, &format!("fixture {fi} ply {plies}"));

                let mut legal = Vec::new();
                p.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    // Terminal (mate/stalemate) — unwind and continue the walk.
                    match stack.pop() {
                        Some((m, u)) => {
                            p.undo_move(m, u);
                            continue;
                        }
                        None => break,
                    }
                }
                let m = legal[rng.pick(legal.len())];
                let u = p.do_move(m);
                stack.push((m, u));
                plies += 1;
            }
        }
        assert!(
            in_check_visits > 0,
            "playout never visited an in-check position — evasion path unexercised",
        );
    }
}

// ===========================================================================
// Drop-emitter gate: bitboard-driven `emit_drops_masked` == its scan twin
// ===========================================================================
//
// The drop emitter iterates droppable-target bitboards rather than walking the
// 81 square indices four times. Node-count parity depends on the exact emission
// order, so the bitboard emitter must produce the byte-identical move Vec the
// band scan does — on both restriction masks it is
// called with (the unrestricted `ALL_SQUARES` quiets/non-evasions path, and the
// single-checker interposition `target1` the evasion path threads in), and in
// positions where the nifu and uchifuzume exclusions actually fire. This gate
// pins the production emitter move-for-move against the retained scan oracle.
#[cfg(test)]
mod gate_280 {
    use super::{ALL_SQUARES, ExtMove, nifu_blocked_files};
    use crate::bitboard::Bitboard;
    use crate::move_::{Move, format_usi_move};
    use crate::piece::PieceKind;
    use crate::position::{Position, Undo};
    use crate::sfen::parse_sfen;

    /// Reuses the six perft / bench parity fixtures (mirrors `gate_267`): the
    /// second starts in check, so the evasion `target1` mask is exercised from
    /// ply 0, and the last is a pawns-in-hand-heavy position (`9P9p`). A seventh
    /// hand-crafted fixture supplies lance + knight in hand (the perft six never
    /// carry those for the side to move) so every rank band pass runs.
    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
        // Lance + knight + pawn in hand with an own pawn already on file 5 — from
        // ply 0 this drives every band pass (`next_to_lance != 0`) and fires the
        // nifu file exclusion, coverage the six perft fixtures never reach.
        "4k4/9/9/9/4P4/9/9/9/4K4 b LN2P 1",
    ];

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

    /// Coverage witnesses accumulated across the playouts, asserted non-zero at
    /// the end so the gate can never silently skip a required exclusion path.
    #[derive(Default)]
    struct Coverage {
        with_pawn: usize,
        with_lance: usize,
        with_knight: usize,
        empty_hand: usize,
        single_check: usize,
        nifu_fired: usize,
    }

    /// Assert the bitboard emitter and the scan twin emit the identical Vec for
    /// `restrict`, and return the production Vec for further inspection.
    fn assert_drops_match(p: &Position, restrict: Bitboard, ctx: &str) -> Vec<Move> {
        let mut prod_ext: Vec<ExtMove> = Vec::new();
        p.emit_drops_masked(restrict, &mut prod_ext);
        let prod: Vec<Move> = prod_ext.into_iter().map(|e| e.mv).collect();
        let mut scanned = Vec::new();
        p.emit_drops_masked_scan(restrict, &mut scanned);
        assert_eq!(
            prod.len(),
            scanned.len(),
            "{ctx}: drop count differs — bitboard {:?} vs scan {:?}",
            prod.iter().map(|&m| format_usi_move(m)).collect::<Vec<_>>(),
            scanned
                .iter()
                .map(|&m| format_usi_move(m))
                .collect::<Vec<_>>(),
        );
        for (i, (&a, &b)) in prod.iter().zip(scanned.iter()).enumerate() {
            assert_eq!(
                a,
                b,
                "{ctx}: drops diverge at index {i}: bitboard {} vs scan {}",
                format_usi_move(a),
                format_usi_move(b),
            );
        }
        prod
    }

    /// The single-checker interposition mask (`target1`) the evasion generator
    /// threads into `emit_drops_masked`, or `None` when not a single-check state.
    fn interposition_mask(p: &Position) -> Option<Bitboard> {
        let ci = p.check_info();
        if !ci.in_check {
            return None;
        }
        let checkers = ci.checkers;
        if checkers.popcount() != 1 {
            return None;
        }
        let ksq = ci.own_king.expect("in check implies own king on board");
        let checksq = checkers.squares().next().unwrap();
        Some(crate::bitboard::between(checksq, ksq))
    }

    /// Record which exclusion / hand conditions this position exercises, so the
    /// end-of-test assertions can prove every required path was visited.
    fn note_coverage(p: &Position, cov: &mut Coverage) {
        let stm = p.side_to_move();
        let hand = p.hand(stm);
        if hand.count(PieceKind::Pawn) > 0 {
            cov.with_pawn += 1;
        }
        if hand.count(PieceKind::Lance) > 0 {
            cov.with_lance += 1;
        }
        if hand.count(PieceKind::Knight) > 0 {
            cov.with_knight += 1;
        }
        if DROP_KINDS.iter().all(|&k| hand.count(k) == 0) {
            cov.empty_hand += 1;
        }
        if interposition_mask(p).is_some() {
            cov.single_check += 1;
        }
        // Nifu "fired" iff a pawn is in hand and some file is nifu-blocked, so at
        // least one candidate pawn-drop square is removed by the file mask.
        if hand.count(PieceKind::Pawn) > 0 {
            let blocked = nifu_blocked_files(p.board(), stm);
            if blocked.iter().any(|&b| b) {
                cov.nifu_fired += 1;
            }
        }
    }

    const DROP_KINDS: [PieceKind; 7] = [
        PieceKind::Pawn,
        PieceKind::Lance,
        PieceKind::Knight,
        PieceKind::Silver,
        PieceKind::Gold,
        PieceKind::Bishop,
        PieceKind::Rook,
    ];

    #[test]
    fn drop_emitter_matches_scan_over_playouts() {
        const MIN_PLIES: usize = 40;
        let mut cov = Coverage::default();
        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut p = parse_sfen(sfen).expect("valid SFEN");
            let mut rng = Rng(0x51ED_270B_9C6E_D14D ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, Undo)> = Vec::new();
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                note_coverage(&p, &mut cov);
                // Unrestricted mask — the quiets / non-evasions drop path.
                assert_drops_match(&p, ALL_SQUARES, &format!("fixture {fi} ply {plies} [all]"));
                // Interposition mask — the evasion drop path (single check only).
                if let Some(target1) = interposition_mask(&p) {
                    assert_drops_match(
                        &p,
                        target1,
                        &format!("fixture {fi} ply {plies} [interpose]"),
                    );
                }

                let mut legal = Vec::new();
                p.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    match stack.pop() {
                        Some((m, u)) => {
                            p.undo_move(m, u);
                            continue;
                        }
                        None => break,
                    }
                }
                let m = legal[rng.pick(legal.len())];
                let u = p.do_move(m);
                stack.push((m, u));
                plies += 1;
            }
        }
        assert!(cov.with_pawn > 0, "no pawn-in-hand position visited");
        assert!(cov.with_lance > 0, "no lance-in-hand position visited");
        assert!(cov.with_knight > 0, "no knight-in-hand position visited");
        assert!(cov.empty_hand > 0, "no empty-hand position visited");
        assert!(
            cov.single_check > 0,
            "no single-check interposition visited"
        );
        assert!(cov.nifu_fired > 0, "nifu exclusion never fired");
    }

    /// Dedicated positions where the nifu and uchifuzume exclusions each remove a
    /// genuine candidate square, checked to (a) still match the scan twin and (b)
    /// actually drop the offending square from the emission.
    #[test]
    fn nifu_and_uchifuzume_exclusions_fire_and_match() {
        // Nifu: a black pawn already on file 5 (5e) with a pawn in hand — no pawn
        // drop may land on file 5.
        let p = parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b P 1").expect("valid SFEN");
        let drops = assert_drops_match(&p, ALL_SQUARES, "nifu fixture");
        assert!(
            drops.iter().any(|&m| m.is_drop()),
            "nifu fixture should still emit pawn drops on other files",
        );
        assert!(
            !drops.iter().any(|&m| m.is_drop()
                && m.dropped_piece_kind() == PieceKind::Pawn
                && m.to_sq().file() == 4),
            "nifu exclusion failed: a pawn drop landed on the occupied file",
        );

        // Uchifuzume: dropping a pawn in front of the cornered white king is mate,
        // so that one square must be excluded while the emitter otherwise matches.
        let p = parse_sfen("k8/9/G1N6/9/9/9/9/9/8K b P 1").expect("valid SFEN");
        let mate_sq = crate::square::Square::new(8, 1).unwrap(); // 9b, in front of 9a king
        let drops = assert_drops_match(&p, ALL_SQUARES, "uchifuzume fixture");
        assert!(
            !drops.iter().any(|&m| m.is_drop() && m.to_sq() == mate_sq),
            "uchifuzume exclusion failed: the pawn-drop-mate square was emitted",
        );
        // Sanity: the exclusion actually removed a square the geometry allows —
        // the scan twin agrees (asserted above) and a non-mating pawn drop nearby
        // is still present, proving pawns are otherwise generated.
        assert!(
            drops.iter().any(|&m| m.is_drop()
                && m.dropped_piece_kind() == PieceKind::Pawn
                && m.to_sq() != mate_sq),
            "uchifuzume fixture emitted no other pawn drops — exclusion untested",
        );
    }
}
