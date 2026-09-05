//! Search-oriented move generation and predicates: the reference-ordered
//! `CAPTURES` / `QUIETS` / `EVASIONS` / `NON_EVASIONS` candidate lists the
//! `MovePicker` consumes, plus the `gives_check` / `in_check` / `is_legal`
//! predicates it consults.
//!
//! The per-stage generators emit **pseudo-legal** candidates, exactly as the
//! reference's `MoveList` does, and the picker filters each with
//! [`Position::is_legal`] as it yields it. They reproduce the reference's
//! emission order (`movegen.cpp`) move for move: piece-type-major, ascending
//! from- then to-square, with the `make_move_target` promotion suppression
//! stated at each emit function below. Move ordering drives search cutoffs, so
//! node-count parity depends on that order.
//!
//! [`Position::generate_legal_all`] is the engine's sole legal-move generator
//! (`generate<LEGAL_ALL>`, `movegen.cpp`), built on the same
//! generators. Like the reference it is **repetition-blind**: it carries no
//! sennichite or perpetual-check term.
//!
//! Evasions are target-restricted at generation, which is safe because every
//! emitted move is a subsequence of what an unrestricted emission would produce.
//! One deviation from the reference is deliberate and output-preserving: the
//! checkers' attack rays are accumulated under the *current* occupancy rather
//! than its king-removed `occ = pieces() ^ ksq`. With the king still in the
//! occupancy a slider's ray stops at it, so the square directly behind the king
//! survives the king-step mask instead of being pruned; it is a suicide step,
//! which `is_legal` rejects anyway.

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

/// A move paired with its ordering score (`ExtMove`, `movegen.h`). The
/// generators emit these directly into the `MovePicker`'s buffer with `value`
/// at `0`; the picker fills it in at its scoring stage.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExtMove {
    pub mv: Move,
    pub value: i32,
}

/// Scanning twins of the four search generators, against which the production
/// ones are pinned move-for-move.
#[cfg(test)]
mod scan_oracle;

/// The droppable piece kinds in the order `GenerateDropMoves` lays them into
/// its `drops[]` array (`movegen.cpp`). Pawn drops are generated
/// separately, and first.
const DROP_ORDER: [PieceKind; 6] = [
    PieceKind::Knight,
    PieceKind::Lance,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Apply `m` to a bare `board` — the board half of [`Position::do_move`], with
/// no keys, hands or history. Only the copy-apply-scan oracles need it; the
/// production predicates are board-mutation-free.
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

/// The identity for the evasion destination restriction, passed by every
/// generator that imposes no mask.
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

/// The pseudo-legal destinations of the piece standing on `from`, as a bitboard
/// whose `.squares()` yields them ascending by index — the reference's
/// `bb.pop()` order. `target` selects which squares are kept.
///
/// The attack set already stops each slider at and including the first blocker,
/// so an own-piece blocker square is present in the raw set and dropped by
/// every mask.
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
/// tables so that it can serve as their oracle.
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
/// enemy first and second ranks, where a non-promoted knight would be stuck
/// (`movegen.cpp`). Independent of `All` — a stuck knight is never
/// generated.
fn nonpromote_rank_ok(to: Square, color: Color) -> bool {
    match color {
        Color::Black => to.rank() >= 2,
        Color::White => to.rank() <= 6,
    }
}

/// Non-promotion is allowed for a lance landing on `to`. Under `All == false`
/// it is suppressed on the enemy first and second ranks (`movegen.cpp`);
/// under `All == true` only on the very last rank, where the lance would be
/// stuck (`movegen.cpp`).
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
/// non-promotion, and additionally promotes whenever a promotion is available.
/// When `from` is outside the enemy field the reference emits the into-zone
/// destinations first, then the rest — hence the two passes here.
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
/// when they can; the non-promotion is `All`-only, and under `All` it is
/// interleaved right after each promotion rather than emitted in a second pass.
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
/// (`GPM_GHDK`) but not for `EVASIONS` (`GPM_GHD`), where the king was already
/// emitted.
///
/// The runtime, `match`-dispatched form, kept as the oracle for the scanning
/// generators. The production ones name a compile-time [`GroupSpec`] instead.
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
    /// Does `piece`, already known to be the side to move's, belong to this
    /// group? The groups partition the moving side's pieces exactly as the
    /// reference bitboards `pieces(Us, PAWN)`, …, `pieces(Us, GOLDS, HDK)` do.
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

/// Compile-time specialization of one piece group's emit pipeline, mirroring
/// the reference's `GeneratePieceMoves<…, Pt, …>` template specializations
/// (`movegen.cpp`). Naming a concrete `GroupSpec` per call site keeps
/// the per-from-square loop free of an indirect branch.
trait GroupSpec {
    /// The side-to-move's pieces belonging to this group. Iterated ascending it
    /// yields the same from-squares, in the same order, as a 0..81 scan filtered
    /// by `Group::contains`.
    fn pieces(board: &Board, stm: Color) -> Bitboard;

    /// The static counterpart of `Group::emit`.
    fn emit(from: Square, targets: Bitboard, piece: Piece, all: bool, out: &mut Vec<ExtMove>);
}

/// Zero-sized group markers, one per emission group. `GoldHdk` is const-generic
/// over whether the group carries the king; it does not for `EVASIONS`, where
/// the king was emitted first.
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

/// Iterate the side-to-move's group-`G` pieces by ascending square and emit
/// their pseudo-moves onto the `target` squares. `all` is the
/// `GenerateAllLegalMoves` flag, which widens the suppressed non-promotions.
fn emit_group<G: GroupSpec>(
    board: &Board,
    stm: Color,
    target: Target,
    all: bool,
    out: &mut Vec<ExtMove>,
) {
    emit_group_masked::<G>(board, stm, target, ALL_SQUARES, all, out);
}

/// [`emit_group`] with the destinations additionally intersected with
/// `restrict` — the reference `target2` mask, interpose-or-capture-the-checker.
/// Because `restrict` only removes destinations, the emitted moves stay a
/// subsequence of the unrestricted emission.
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

/// A 0..81 scan form of [`emit_group`], sharing no code with the production
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

/// `file → the side to move already holds an own un-promoted pawn on it`, the
/// nifu (二歩) mask. A tokin lives in the GOLD pattern slot and so does not
/// count. The production emitter folds the same pawn bitboard into a full-file
/// exclusion mask inline; this per-file form serves its oracle.
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

/// A scan form of [`nifu_blocked_files`], the oracle for the file-mask form.
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

// The per-position-state check info (`Position::set_check_info`,
// `position.cpp`), cached on [`Position`], reduces the `gives_check` /
// `gives_direct_check` / `is_legal` predicates below to table lookups.

/// The number of distinct check-attack patterns keyed by [`check_pattern`].
const CHECK_PATTERN_COUNT: usize = crate::board::PATTERN_COUNT;

/// Map a concrete piece to its `checkSquares` pattern slot — the shared
/// [`crate::board::pattern_of`] partition.
fn check_pattern(piece: Piece) -> usize {
    crate::board::pattern_of(piece)
}

/// The set of squares `piece`, standing on `from`, attacks under occupancy
/// `occ`. Sliders stop at and including the first occupied square.
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
/// bitboard tables so that it can serve as their oracle.
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
/// eight queen-lines, else `None` (`Effect8::directions_of`).
fn ray_dir(king: Square, sq: Square) -> Option<(i8, i8)> {
    crate::bitboard::ray_dir(king, sq)
}

/// The reference `aligned(s1, s2, ksq)` (`types.h`): are `s1` and `s2` on
/// the same ray *emanating from* `ksq`? A straight line passing through the
/// king does not count — the two must be on the same side of it.
fn aligned(king: Square, s1: Square, s2: Square) -> bool {
    match (ray_dir(king, s1), ray_dir(king, s2)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// The reference `between_bb(a, b)`: the squares strictly between `a` and `b`
/// when the two lie on one of the eight queen-lines, else empty.
fn between_set(a: Square, b: Square) -> Bitboard {
    crate::bitboard::between(a, b)
}

/// The reference `is_non_promotable_piece(pc)` (`type_of(pc) >= GOLD`): a piece
/// that can never promote — gold, king, or any already-promoted piece.
fn is_non_promotable_piece(pc: Piece) -> bool {
    pc.promoted || matches!(pc.kind, PieceKind::Gold | PieceKind::King)
}

/// The per-position-state check info cached on [`Position`]
/// (`StateInfo::{checkSquares, blockersForKing}`).
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
    /// `color`'s king. Note the index flip: [`crate::see::slider_blockers`]
    /// returns `pinners[~c]` alongside `blockersForKing[c]`.
    pinners: [Bitboard; Color::COUNT],
    /// `checkersBB`: the squares of the enemy pieces giving check to the side to
    /// move's king. Empty unless [`Self::in_check`].
    checkers: Bitboard,
}

impl CheckInfo {
    /// The check info of the empty board, which [`Position::empty`] seeds its
    /// field with. Equal to what `compute_check_info` yields there.
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

    /// The enemy king square, if present.
    pub(crate) fn enemy_king(&self) -> Option<Square> {
        self.enemy_king
    }

    /// The side-to-move's own king square, if present.
    pub(crate) fn own_king(&self) -> Option<Square> {
        self.own_king
    }

    /// `blockersForKing[color]`: the pieces singly blocking a slider check on
    /// `color`'s king.
    pub(crate) fn blockers(&self, color: Color) -> Bitboard {
        self.blockers[color.index()]
    }

    /// `pinners[color]`: the enemy sliders pinning a `color`-blocker to
    /// `color`'s king.
    pub(crate) fn pinners(&self, color: Color) -> Bitboard {
        self.pinners[color.index()]
    }
}

/// A by-value snapshot of a position's `checkSquares` table, so that a
/// per-move direct-check test needs no `Position` access at all.
///
/// `checkSquares` is a pure function of the position, and the position is
/// unchanged for a move picker's whole lifetime — each subtree's
/// `do_move`/`undo_move` pair restores the board before the next scoring call —
/// so one snapshot orders moves exactly as the reference's per-move
/// `pos.check_squares(pt) & to` does (`movepick.cpp`).
#[derive(Clone, Copy, Debug)]
pub struct CheckSquares {
    /// The snapshotted `checkSquares[pattern]` table, keyed by [`check_pattern`].
    table: [Bitboard; CHECK_PATTERN_COUNT],
}

impl CheckSquares {
    /// True iff `m` gives a **direct** check by the moved piece — as
    /// [`Position::gives_direct_check`], but read off the snapshot.
    pub fn gives_direct_check(&self, m: Move) -> bool {
        self.table[check_pattern(m.moved_piece_after())].test(m.to_sq())
    }
}

impl Position {
    /// This position's [`CheckInfo`] computed from scratch
    /// (`Position::set_check_info`, `position.cpp`), with `in_check`
    /// probed from the board. For the cold entry points, which have no
    /// pre-computed check flag.
    pub(crate) fn compute_check_info(&self) -> CheckInfo {
        self.compute_check_info_impl(None, None)
    }

    /// Check info for a child position whose check status is already known:
    /// `in_check` is taken from `gives_check` rather than probed.
    pub(crate) fn compute_check_info_with_in_check(&self, in_check: bool) -> CheckInfo {
        self.compute_check_info_impl(Some(in_check), None)
    }

    /// Like [`Self::compute_check_info_with_in_check`], but with the child
    /// `checkersBB` also supplied by the caller, built differentially rather
    /// than re-derived by a full reverse-attack probe.
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

        // By attack reverse-symmetry, an enemy-coloured `pt` piece placed on the
        // enemy king attacks exactly the squares from which a side-to-move `pt`
        // piece checks that king.
        //
        // Slot 9 (KING) holds the actual king ring rather than the reference's
        // hard `0`. A king move can never be a *legal* check, so the reference
        // never reaches this slot; the ring keeps the predicates byte-identical
        // to the scratch-scan oracle on the pseudo-legal-but-illegal king steps
        // the generators do emit.
        let occ = board.occupied();
        let mut check_squares = [Bitboard::EMPTY; CHECK_PATTERN_COUNT];
        if let Some(eks) = enemy_king {
            // The effect primitives are called directly with constant piece
            // kinds, which is bit-identical to dispatching through
            // [`attacks_bb`] but leaves no re-dispatch per call.
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

        // `slider_blockers(c)` returns `(blockersForKing[c], pinners[~c])`, so
        // the pinners half from `slider_blockers(Black)` is `pinners[White]`.
        let (blk_black, pin_white) = crate::see::slider_blockers(board, Color::Black);
        let (blk_white, pin_black) = crate::see::slider_blockers(board, Color::White);
        let mut blockers = [Bitboard::EMPTY; Color::COUNT];
        blockers[Color::Black.index()] = blk_black;
        blockers[Color::White.index()] = blk_white;
        let mut pinners = [Bitboard::EMPTY; Color::COUNT];
        pinners[Color::Black.index()] = pin_black;
        pinners[Color::White.index()] = pin_white;

        // The injected flag equals this probe in legal play, a move giving check
        // iff the resulting side to move is in check. The oracle is
        // one-directional because the reverse legitimately differs on the
        // illegal scratch positions where the side to move was already in check.
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

        // `checkersBB` is only meaningful when the side to move is in check.
        // Cold entry points inject nothing and fall back to the reverse-attack
        // lookup from the king square.
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

    /// Build the child state's `checkersBB` from the **parent** check info and
    /// the move just played (`do_move_impl`, `position.cpp`).
    /// `self` is already the post-move position, but `parent` still describes
    /// the mover's frame.
    ///
    /// The discovered part looks for the revealed slider from the king rather
    /// than walking outward from `from` as the reference does. The two agree
    /// because `from` was the sole blocker, so the squares between are empty.
    pub(crate) fn differential_child_checkers(
        &self,
        m: Move,
        mover: Color,
        parent: &CheckInfo,
    ) -> Bitboard {
        let board = self.board();
        let to = m.to_sq();

        let mut checkers = if parent.check_squares[check_pattern(m.moved_piece_after())].test(to) {
            Bitboard::from_square(to)
        } else {
            Bitboard::EMPTY
        };

        // A discovery needs the from-square to singly block one of the mover's
        // sliders aimed at the enemy king, and the move to leave that ray.
        if let Some(eks) = parent.enemy_king
            && !m.is_drop()
            && parent.blockers(mover.flip()).test(m.from_sq())
            && !aligned(eks, m.from_sq(), to)
        {
            let from = m.from_sq();
            let occ = board.occupied();
            let (df, dr) =
                ray_dir(eks, from).expect("a blocker for the enemy king is aligned with it");
            let slider = if df != 0 && dr != 0 {
                crate::bitboard::bishop_attacks(eks, occ)
            } else {
                crate::bitboard::rook_attacks(eks, occ)
            };
            checkers |= slider & crate::bitboard::ray_toward(eks, from) & board.pieces_color(mover);
        }

        checkers
    }

    /// True iff the side to move's king is currently attacked
    /// (`Position::in_check()`).
    pub fn in_check(&self) -> bool {
        self.check_info().in_check
    }

    /// True iff `sq` is attacked by any piece of `attacker`, with `discount`
    /// treated as empty (`effected_to(attacker, sq, discount)`, `position.h`).
    ///
    /// Removing `discount` models the moving king vacating its from-square,
    /// which both drops it as a would-be defender and reveals any enemy slider
    /// it was blocking. `discount == sq` is harmless: a piece already on the try
    /// square is a capture target, not a defender.
    pub fn is_attacked_discounting(&self, sq: Square, attacker: Color, discount: Square) -> bool {
        let mut board = *self.board();
        board.set(discount, None);
        is_attacked_by(&board, sq, attacker)
    }

    /// True iff playing `m` would leave the opponent's king in check —
    /// `Position::gives_check(m)` (`position.cpp`), covering direct
    /// checks, discovered checks and checking drops.
    ///
    /// If `m` captures the opponent's king — a pseudo-legal probe move, never a
    /// real one — there is no king to check and the result is `false`.
    pub fn gives_check(&self, m: Move) -> bool {
        let ci = self.check_info();
        let to = m.to_sq();

        if ci.enemy_king == Some(to) {
            return false;
        }

        if ci.check_squares[check_pattern(m.moved_piece_after())].test(to) {
            return true;
        }

        // Discoveries need `from` to singly block one of our sliders from the
        // enemy king, and the move to leave that ray.
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

    /// True iff `m` gives a **direct** check by the moved piece — the
    /// quiet-ordering term `check_squares(type_of(moved_piece(m))) & to`
    /// (`movepick.cpp`).
    ///
    /// `checkSquares` is computed against the occupancy *before* the move, so a
    /// slider whose own vacated from-square lay on the ray is not counted;
    /// discovered checks are likewise not detected. Both match the reference's
    /// approximation.
    pub fn gives_direct_check(&self, m: Move) -> bool {
        let ci = self.check_info();
        ci.check_squares[check_pattern(m.moved_piece_after())].test(m.to_sq())
    }

    /// Snapshot the cached `checkSquares` table by value, so that the per-quiet
    /// check bonus becomes a bare table lookup. See [`CheckSquares`].
    pub fn check_squares(&self) -> CheckSquares {
        CheckSquares {
            table: self.check_info().check_squares,
        }
    }

    /// True iff `m` leaves the mover's own king out of check
    /// (`Position::legal`, `position.cpp`), covering a pinned piece
    /// moving off its ray and a king stepping into an attacked square.
    ///
    /// # Contract
    ///
    /// `m` must come from one of the search move generators for the current
    /// check state — evasions when in check, captures / quiets / non-evasions
    /// when not — or have passed [`Position::pseudo_legal`]. Both restrict an
    /// in-check move to capture-the-checker-or-interpose, which is what lets
    /// the two non-king arms below be as cheap as they are.
    ///
    /// It is repetition-blind, like the generators. Uchifuzume *is* enforced,
    /// but at generation time, so no such move ever reaches this predicate.
    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.side_to_move();

        if !m.is_drop() && m.moved_piece_after().kind == PieceKind::King {
            return !self.is_attacked_discounting(m.to_sq(), us.flip(), m.from_sq());
        }

        // A drop adds a piece and can never expose the king; in check, the
        // contract means only interposition drops reach here.
        if m.is_drop() {
            return true;
        }

        // A board move exposes the king only by moving a sole slider-blocker off
        // its pin ray; in check, the contract leaves that the only illegality.
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

    /// A copy-apply-scan form of `is_legal`, kept as its oracle.
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

    /// Widen a stored 16-bit TT fragment into a full [`Move`] by attaching the
    /// moving-piece bits the `move16` layout drops
    /// (`Position::to_move(Move16)`, `position.cpp`). Legality is not
    /// proved here.
    ///
    /// A non-`is_ok` fragment such as `MOVE_WIN` comes back verbatim as `Some`
    /// rather than folding to `None`, so that the search's `tt_move.is_none()`
    /// gates agree with the reference's `!ttData.move`; [`Self::pseudo_legal`]
    /// rejects it downstream.
    ///
    /// The reference trusts the stored bits because it wrote them. This TT
    /// admits torn fragments, so this stays total: a malformed field returns
    /// `None` rather than indexing a table.
    pub fn to_move(&self, m16: u16) -> Option<Move> {
        if m16 == 0 {
            return None;
        }
        let m = Move::from_bits(m16 as u32);
        if !m.is_ok() {
            return Some(m);
        }
        let stm = self.side_to_move();

        // A torn fragment can carry a to-square in 81..127, which `Move::to_sq`
        // would panic on.
        let to = Square::from_index((m16 & 0x7f) as u8)?;

        if m.is_drop() {
            let kind = m.dropped_piece_kind_checked()?;
            return Some(Move::make_drop(kind, stm, to));
        }

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

    /// True iff `m` is pseudo-legal for the side to move
    /// (`pseudo_legal_s<All>`, `position.cpp`) — the pre-`do_move`
    /// guard for a TT or killer move. A king-suicide passes; that is
    /// [`Self::is_legal`]'s business.
    ///
    /// Returns `false` for a non-`is_ok` move, and is total on any well-formed
    /// move produced by [`Self::to_move`] or the generators.
    pub fn pseudo_legal(&self, m: Move, all: bool) -> bool {
        // This also keeps `moved_piece_after` below off the sentinel patterns.
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
            if m.moved_piece_after() != Piece::new(pr, us) {
                return false;
            }
            if board.get(to).is_some() || self.hand(us).count(pr) == 0 {
                return false;
            }
            // In check, a drop must interpose on the single checker's ray;
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
            if pr == PieceKind::Pawn && !self.legal_pawn_drop(us, to) {
                return false;
            }
            return true;
        }

        let pc = match board.get(m.from_sq()) {
            Some(p) if p.color == us => p,
            _ => return false,
        };
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
            if all {
                if matches!(pc.kind, PieceKind::Pawn | PieceKind::Lance) && !pc.promoted {
                    let last_rank = if us == Color::Black { 0 } else { 8 };
                    if to.rank() == last_rank {
                        return false;
                    }
                }
            } else {
                match (pc.kind, pc.promoted) {
                    (PieceKind::Pawn, false) if is_in_promotion_zone(to, us) => return false,
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

        // King moves fall through: their suicide check is [`Self::is_legal`]'s.
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

    /// The reference `legal_pawn_drop(us, to)` (`position.cpp`): a
    /// pawn drop on `to` is legal iff it is neither nifu (二歩) nor uchifuzume
    /// (打ち歩詰め).
    fn legal_pawn_drop(&self, us: Color, to: Square) -> bool {
        let board = self.board();
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
        // Uchifuzume is only possible when the dropped pawn checks the enemy
        // king, which fixes `to` to a single square.
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

    /// Append the pseudo-legal `CAPTURES` candidates to `out`: destinations are
    /// exactly the enemy-occupied squares, so there are no drops and no
    /// non-capturing pawn promotions (`movegen.cpp`).
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

    /// Append the pseudo-legal `QUIETS` candidates to `out`: non-capturing
    /// piece moves, then every pseudo-legal drop
    /// (`generate_general<QUIETS, Us, false>`, `movegen.cpp`).
    ///
    /// Non-capturing pawn promotions belong here rather than to
    /// [`Position::generate_captures`], whose target is `pieces(Them)`, so that
    /// the two generators partition the destinations with no overlap.
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

    /// Append the pseudo-legal `EVASIONS` candidates to `out`: king moves
    /// first, then — on a single check — the non-king piece moves restricted to
    /// capture-or-interpose, then interposition drops. The caller's
    /// [`Position::is_legal`] filter removes the remaining suicide king steps.
    ///
    /// **Entry contract:** the side to move is in check, as the reference
    /// `generate_evasions` asserts.
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

        // The union of every checker's attack rays, so the king cannot step onto
        // a still-attacked square (`movegen.cpp`). Accumulated under the
        // current occupancy; see the module head for why that is equivalent.
        let occ = board.occupied();
        let mut slider_attacks = Bitboard::empty();
        for checksq in checkers.squares() {
            let cp = board
                .get(checksq)
                .expect("a checker square holds an enemy piece");
            slider_attacks |= attacks_bb(checksq, cp, occ);
        }

        // King moves first (`movegen.cpp`).
        let king_targets = reachable(board, ksq, king, Target::BlockOrCapture) & !slider_attacks;
        for to in king_targets.squares() {
            push_plain(ksq, to, king, out);
        }

        // Under double check only king moves evade (`movegen.cpp`).
        if checkers.popcount() >= 2 {
            return;
        }

        // Otherwise: capture the checker or interpose (`movegen.cpp`).
        let checksq = checkers
            .squares()
            .next()
            .expect("in check implies at least one checker");
        let target1 = crate::bitboard::between(checksq, ksq); // interposition squares
        let target2 = target1 | Bitboard::from_square(checksq); // + capture the checker

        // The gold-group excludes the king, already emitted above.
        emit_group_masked::<PawnG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<LanceG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<KnightG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<SilverG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<BishopRookG>(board, stm, Target::BlockOrCapture, target2, all, out);
        emit_group_masked::<GoldHdkG<false>>(board, stm, Target::BlockOrCapture, target2, all, out);

        // Interposition drops last (`movegen.cpp`).
        self.emit_drops_masked(target1, out);
    }

    /// Append the pseudo-legal `NON_EVASIONS` candidates to `out`
    /// (`generate_general<NON_EVASIONS, Us, false>`, `movegen.cpp`).
    ///
    /// Its single move target is `~pos.pieces(Us)`, so captures and quiets come
    /// out **interleaved** per piece and per destination square rather than in
    /// two blocks. That interleaving is what makes the *first* legal move — the
    /// root search's TT move — match the reference.
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

    /// Append every legal move for the side to move to `out`
    /// (`generate<LEGAL_ALL>`, `movegen.cpp`). The buffer is **not**
    /// cleared first, so a caller may reuse it across calls.
    ///
    /// **Repetition-blind**, like the reference: plain 4-fold makes the *game*
    /// drawn, not the move illegal, and a perpetual-check repetition is scored
    /// by the search rather than removed here.
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

    /// Emit the pseudo-legal drops on empty squares in `GenerateDropMoves`
    /// order (`movegen.cpp`): pawn drops first, then the other kinds in
    /// rank bands.
    fn emit_drops(&self, out: &mut Vec<ExtMove>) {
        self.emit_drops_masked(ALL_SQUARES, out);
    }

    /// [`Position::emit_drops`] with the target additionally intersected with
    /// `restrict` — the reference `target1`, interposition squares only
    /// (`movegen.cpp`). Because `restrict` only removes candidate squares,
    /// the emitted drops stay a subsequence of the unrestricted emission.
    fn emit_drops_masked(&self, restrict: Bitboard, out: &mut Vec<ExtMove>) {
        use crate::bitboard::rank_mask;

        let board = self.board();
        let stm = self.side_to_move();
        let hand = self.hand(stm);

        let base = restrict & !board.occupied();

        let (back_rank, second_rank): (u8, u8) = if stm == Color::Black { (0, 1) } else { (8, 7) };

        // A dropped pawn checks only the square directly ahead of it, so at most
        // one drop square can be uchifuzume. It is excluded here because the
        // caller's [`Position::is_legal`] filter is uchifuzume-blind.
        if hand.count(PieceKind::Pawn) > 0 {
            // A tokin lives in the GOLD pattern slot, so it does not block a
            // file for nifu (二歩) purposes.
            let own_pawns = board.pieces_pattern(stm, crate::board::pat::PAWN);
            let mut nifu_files = Bitboard::empty();
            for psq in own_pawns.squares() {
                nifu_files |= crate::bitboard::file_mask(psq.file());
            }
            // A pawn dropped on the last rank could never move again.
            let mut pawn_target = base & !rank_mask(back_rank) & !nifu_files;

            // That square lies directly "behind" the enemy king along our pawn's
            // line of advance. Probing it only once it survives as a genuine
            // drop square keeps the mate test to at most one call per node.
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

        // The other kinds are emitted in rank bands, so that lance and knight
        // are excluded from the ranks on which they would be stuck. The scratch
        // is fixed-capacity to keep the node allocation-free.
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
        // Knight and lance lead `DROP_ORDER`, so a band that excludes them is a
        // suffix — the reference's `nextToKnight` / `nextToLance`.
        let next_to_knight = usize::from(drops.first() == Some(&PieceKind::Knight));
        let next_to_lance =
            next_to_knight + usize::from(drops.get(next_to_knight) == Some(&PieceKind::Lance));

        if next_to_lance == 0 {
            // With no lance or knight in hand every kind can go on every empty
            // square (`movegen.cpp`).
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

        // Own back rank (`movegen.cpp`).
        for sq in (base & rank_mask(back_rank)).squares() {
            for &kind in &drops[next_to_lance..] {
                out.push(ExtMove {
                    mv: Move::make_drop(kind, stm, sq),
                    value: 0,
                });
            }
        }
        // Own second rank: lance too, but not knight (`movegen.cpp`).
        for sq in (base & rank_mask(second_rank)).squares() {
            for &kind in &drops[next_to_knight..] {
                out.push(ExtMove {
                    mv: Move::make_drop(kind, stm, sq),
                    value: 0,
                });
            }
        }
        // Everywhere else: every kind (`movegen.cpp`).
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

    /// An 81-square band-scanning form of [`Position::emit_drops_masked`],
    /// which must produce the byte-identical move list on every mask.
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

    /// True iff dropping a side-to-move pawn on `sq` is uchifuzume (打ち歩詰め),
    /// the unanswerable drop-pawn-mate the rules forbid.
    fn pawn_drop_is_uchifuzume(&self, sq: Square) -> bool {
        let m = Move::make_drop(PieceKind::Pawn, self.side_to_move(), sq);
        crate::movegen::drop_is_uchifuzume(self, m)
    }
}

/// Copy-apply-scan forms of the three predicates, derived independently of the
/// cached check info so that they can serve as its oracles.
#[cfg(test)]
impl Position {
    /// Apply `m` to a scratch board and test the enemy king against the full
    /// attacker scan.
    pub(crate) fn gives_check_reference(&self, m: Move) -> bool {
        let mover = self.side_to_move();
        let mut board = *self.board();
        apply_to_board(&mut board, m, mover);
        match try_find_king(&board, mover.flip()) {
            Some(king_sq) => is_attacked_by(&board, king_sq, mover),
            None => false,
        }
    }

    /// Does a piece of the moved type, standing on `to`, attack the enemy king
    /// under the pre-move occupancy?
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

    /// Apply `m` to a scratch board and test that the mover's own king is not
    /// left in check.
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

    /// Unwrap the [`ExtMove`]s the generators emit into a plain `Move` list.
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

    /// The six perft-fixture SFENs, reused for the seeded-playout oracle test.
    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    /// A small deterministic xorshift64*.
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

    /// Whether the position is in check *after* `m`, computed through a
    /// `do_move`/`undo_move` round trip so that it shares no code with
    /// `gives_check`'s bare-board application.
    fn gives_check_oracle(p: &Position, m: Move) -> bool {
        let mut scratch = p.clone();
        let undo = scratch.do_move(m);
        let checked = scratch.in_check();
        scratch.undo_move(m, undo);
        checked
    }

    #[test]
    #[cfg_attr(miri, ignore)]
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

    /// Every pseudo-legal move the generators can emit at `p`, across both
    /// `all` settings — the exact moves the search would hand the predicates.
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

    /// The moves `is_legal`'s contract admits at `p`. Outside this set it may
    /// legitimately disagree with the copy-apply-scan oracle — on a
    /// non-check-resolving in-check drop, say — so the test must not go there.
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
        // A move passing `pseudo_legal` also meets the contract, which covers
        // the TT-widened candidates.
        for m in all_generator_moves(p) {
            if p.pseudo_legal(m, false) || p.pseudo_legal(m, true) {
                v.push(m);
            }
        }
        v
    }

    /// The cached-check-info predicates must agree with the scratch oracles for
    /// every move every generator produces, along a playout from each fixture.
    #[test]
    #[cfg_attr(miri, ignore)]
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
        let checked = pos("4r4/9/9/9/9/9/9/9/4K4 b - 1");
        assert!(checked.in_check());
        let quiet = pos("3r5/9/9/9/9/9/9/9/4K4 b - 1");
        assert!(!quiet.in_check());
    }

    #[test]
    fn is_legal_agrees_with_legal_set_for_pin_and_king_step() {
        // The silver is pinned to its king by a rook down the open file, so a
        // diagonal silver step exposes the king while a sideways king step does
        // not.
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

    #[test]
    fn pawn_capture_in_zone_is_promotion_only() {
        // The capture lands on rank 1, inside the zone.
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
        // The capture lands on the enemy second rank.
        let p = pos("k8/4p4/9/4L4/9/9/9/9/8K b - 1");
        let caps = captures(&p);
        assert_eq!(caps.len(), 1, "got {:?}", usi(&caps));
        assert!(caps[0].is_promote());
        assert_eq!(format_usi_move(caps[0]), "5d5b+");
    }

    #[test]
    fn lance_capture_on_enemy_third_rank_keeps_both_variants() {
        // Rank 2 keeps the non-promotion, which follows the promotion.
        let p = pos("k8/9/4p4/9/4L4/9/9/9/8K b - 1");
        let caps = captures(&p);
        assert_eq!(usi(&caps), vec!["5e5c+".to_string(), "5e5c".to_string()]);
    }

    #[test]
    fn all_pawn_capture_in_zone_adds_nonpromotion() {
        // Rank 1 is inside the zone but is not the last rank, so `all` adds the
        // non-promotion after the promotion (`movegen.cpp`).
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
        // `all` widens the lance mask to the very last rank only, so the rank-1
        // non-promotion appears.
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
        // A non-promotion on the last rank would be a stuck pawn, so it is
        // suppressed even under `all` (`movegen.cpp`).
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
        // Under `all` the non-promotion is emitted right after its promotion,
        // per destination (`movegen.cpp`).
        let p = pos("k8/9/6p2/9/4B4/9/9/9/8K b - 1");
        let off: Vec<String> = usi(&captures(&p));
        let on: Vec<String> = usi(&captures_all(&p));
        assert!(
            off.contains(&"5e3c+".to_string()) && !off.contains(&"5e3c".to_string()),
            "default: promotion only, got {off:?}",
        );
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
        // Three independent captures outside the promotion zone, by a pawn, a
        // silver and a gold, so each is a single non-promoting move. Black also
        // holds a pawn in hand, and CAPTURES must emit no drop.
        let p = pos("k8/9/9/9/p1s1g4/P1S1G4/9/9/7K1 b P 1");
        let caps = captures(&p);
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

    #[test]
    fn evasions_are_generated_only_when_in_check() {
        // `generate_evasions` asserts `in_check` at entry, so this position is
        // never handed to it.
        let quiet = pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        assert!(!quiet.in_check());
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
        // The checking rook can be neither captured nor blocked, so every legal
        // evasion is a king step off the file.
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
        // A rook checks down an open file with a gold in hand, so the seven
        // interposition drops are legal evasions.
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
        let first_drop = legal_ev.iter().position(|m| m.is_drop()).unwrap();
        assert!(
            legal_ev[..first_drop].iter().all(|m| !m.is_drop()),
            "king / piece moves must precede drops: {:?}",
            usi(&legal_ev),
        );
    }

    #[test]
    fn drop_generation_excludes_uchifuzume() {
        // A pawn dropped in front of the White king here is uchifuzume, and the
        // shared drop generator must exclude it: the caller's `is_legal` filter
        // is uchifuzume-blind. Black is not in check, so this exercises the
        // filter through the quiets path.
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

        // The exclusion is surgical: an ordinary drop on the same file stays.
        let other_drop = Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 3).unwrap());
        assert!(
            qs.contains(&other_drop),
            "a non-uchifuzume pawn drop was wrongly excluded: {:?}",
            usi(&qs),
        );
    }

    fn quiets(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_quiets(false, &mut v);
        unwrap_ext(v)
    }

    #[test]
    fn quiets_land_only_on_empty_squares_and_include_drops() {
        // A rook with both quiet lifts and one available capture, plus a pawn in
        // hand for the drops.
        let p = pos("k8/9/9/9/4R4/9/4p4/9/8K b P 1");
        let qs = quiets(&p);
        for &m in &qs {
            if !m.is_drop() {
                assert!(
                    p.board().get(m.to_sq()).is_none(),
                    "quiet {} lands on an occupied square",
                    format_usi_move(m),
                );
            }
        }
        let cap = Move::make(
            Square::new(4, 4).unwrap(),
            Square::new(4, 6).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert!(!qs.contains(&cap), "a capture leaked into QUIETS");
        assert!(
            qs.iter().any(|m| m.is_drop()),
            "expected pawn drops among the quiets"
        );
    }

    #[test]
    fn quiets_include_non_capturing_pawn_promotion() {
        // A non-capturing promotion belongs to QUIETS; CAPTURES, targeting enemy
        // pieces only, would miss it.
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
        // A position free of promotion-suppression divergence, where the two
        // lists partition the legal move set exactly.
        let p = pos("9/9/3g1g3/9/4G4/9/9/9/K7k b G 1");
        let legal: std::collections::HashSet<Move> = legal_moves(&p).into_iter().collect();
        let lc: std::collections::HashSet<Move> = legal_captures(&p).into_iter().collect();
        let lq: std::collections::HashSet<Move> =
            quiets(&p).into_iter().filter(|&m| p.is_legal(m)).collect();
        assert!(lc.is_disjoint(&lq), "a move is both a capture and a quiet");
        let union: std::collections::HashSet<Move> = lc.union(&lq).copied().collect();
        assert_eq!(union, legal, "captures ∪ quiets must equal the legal set");
    }

    #[test]
    fn gives_direct_check_true_for_quiet_rook_check() {
        // The rook lifts onto the white king's file, which is empty between.
        let p = pos("4k4/9/R8/9/9/9/9/9/8K b - 1");
        let check = Move::make(
            Square::new(8, 2).unwrap(),
            Square::new(4, 2).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert_eq!(format_usi_move(check), "9c5c");
        assert!(p.gives_direct_check(check));
        let quiet = Move::make(
            Square::new(8, 2).unwrap(),
            Square::new(7, 2).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert!(!p.gives_direct_check(quiet));
    }

    #[test]
    fn gives_direct_check_ignores_discovered_checks() {
        // Moving the gold off the file uncovers the rook behind it — a
        // discovered check, but not a direct one by the gold.
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
        assert!(p.gives_check(discover));
    }

    /// The `NON_EVASIONS` list holds exactly the union of the `CAPTURES` and
    /// `QUIETS` lists: its single `~pieces(Us)` target is the union of the
    /// enemy-only and empty-only ones.
    #[test]
    #[cfg_attr(miri, ignore)]
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

    /// `NON_EVASIONS` interleaves captures and quiets, so its order is *not*
    /// the captures-then-quiets concatenation even though the set is identical.
    /// That distinction fixes the root search's TT move.
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

#[cfg(test)]
mod attack_query_equivalence {
    use super::{CHECK_PATTERN_COUNT, attack_set_from_scan, attacks_bb, check_pattern};
    use crate::bitboard::Bitboard;
    use crate::board::{Board, PATTERN_COUNT, pattern_of};
    use crate::color::Color;
    use crate::movegen::{attackers_bb, is_attacked_by, is_attacked_by_scan, try_find_king};
    use crate::piece::{Piece, PieceKind};
    use crate::position::Position;
    use crate::sfen::parse_sfen;
    use crate::square::Square;

    /// The fixtures the tests sweep: the perft parity positions plus the bench
    /// set.
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

    /// Rebuild the piece sets from scratch by scanning the board.
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

    /// A checkers scan, the oracle for the `attackers_bb`-based query.
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

        for sq in all_squares() {
            for attacker in [Color::Black, Color::White] {
                assert_eq!(
                    is_attacked_by(board, sq, attacker),
                    is_attacked_by_scan(board, sq, attacker),
                    "is_attacked_by {sq:?} {attacker:?} @ {ctx}",
                );
            }
        }

        // Every occupied square is tried as the discount square.
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
                // The representative pieces the check-info fill imagines on the
                // enemy king.
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

        for color in [Color::Black, Color::White] {
            if let Some(king) = try_find_king(board, color) {
                assert_eq!(
                    attackers_bb(board, king, color.flip()),
                    checkers_scan(board, king, color.flip()),
                    "attackers_bb (checkers) king {color:?} @ {ctx}",
                );
            }
        }

        // The pattern maps line up, which pins the numeric slots against drift.
        assert_eq!(CHECK_PATTERN_COUNT, PATTERN_COUNT);
        for p in [
            Piece::new(PieceKind::Gold, Color::Black),
            Piece::promoted(PieceKind::Pawn, Color::White).unwrap(),
            Piece::promoted(PieceKind::Rook, Color::Black).unwrap(),
        ] {
            assert_eq!(check_pattern(p), pattern_of(p));
        }
    }

    /// Run both checks at `pos`.
    fn assert_all(pos: &Position, ctx: &str) {
        assert_sets_consistent(pos, ctx);
        assert_attack_equiv(pos, ctx);
    }

    /// A tiny deterministic LCG.
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
    #[cfg_attr(miri, ignore)]
    fn gate_set_consistency_and_attack_equivalence_along_playouts() {
        const PLIES: usize = 32;
        for (fx, sfen) in FIXTURES.iter().enumerate() {
            let mut pos = parse_sfen(sfen).unwrap_or_else(|e| panic!("fixture {sfen}: {e:?}"));
            let mut rng = Lcg(0x9E3779B97F4A7C15 ^ (fx as u64).wrapping_mul(0x1000_0001));
            let mut buf = Vec::new();
            for ply in 0..PLIES {
                let ctx = format!("fx{fx} ply{ply}");
                assert_all(&pos, &ctx);

                // A null move leaves the board unchanged, so the sets must be
                // untouched by it and restored after the undo.
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

                let undo = pos.do_move(m);
                assert_sets_consistent(&pos, &format!("{ctx} post-do {m:?}"));
                pos.undo_move(m, undo);
                assert_sets_consistent(&pos, &format!("{ctx} post-undo {m:?}"));

                pos.do_move(m);
            }
        }
    }
}

// Node-count parity depends on the exact order moves come out, so the
// production generators must emit the byte-identical move list the scanning
// oracles do. These tests compare them move for move.
#[cfg(test)]
mod nifu_files_equivalence {
    use super::{ExtMove, nifu_blocked_files, nifu_blocked_files_scan};
    use crate::color::Color;
    use crate::move_::{Move, format_usi_move};
    use crate::position::{Position, Undo};
    use crate::sfen::parse_sfen;

    /// The six perft / bench parity fixtures. The second starts *in check*, so
    /// the evasion generator is exercised from ply 0.
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

    /// The four search generators, keyed by index, with their [`ExtMove`]s
    /// unwrapped.
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

    /// Every generator, under both `all` settings, emits the same move list as
    /// its scanning oracle.
    ///
    /// The evasion generator is target-restricted, so its *raw* emission is only
    /// a subsequence of the unrestricted twin's. What must match there is the
    /// **legal-filtered** sequence, each side filtered by its own predicate.
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
    #[cfg_attr(miri, ignore)]
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
                    // Terminal — unwind and continue the walk.
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

// The same emission-order constraint for the drop emitter, on both restriction
// masks it is called with and in positions where the nifu and uchifuzume
// exclusions actually fire.
#[cfg(test)]
mod drop_emitter_equivalence {
    use super::{ALL_SQUARES, ExtMove, nifu_blocked_files};
    use crate::bitboard::Bitboard;
    use crate::move_::{Move, format_usi_move};
    use crate::piece::PieceKind;
    use crate::position::{Position, Undo};
    use crate::sfen::parse_sfen;

    /// The six parity fixtures, plus a seventh supplying lance + knight in hand
    /// — which none of the six carry for the side to move — so that every rank
    /// band pass runs.
    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
        // An own pawn already on file 5 also fires the nifu file exclusion from
        // ply 0.
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

    /// Coverage witnesses, asserted non-zero at the end so that the test can
    /// never silently skip a required exclusion path.
    #[derive(Default)]
    struct Coverage {
        with_pawn: usize,
        with_lance: usize,
        with_knight: usize,
        empty_hand: usize,
        single_check: usize,
        nifu_fired: usize,
    }

    /// Assert the two emitters agree for `restrict`, returning the production
    /// list for further inspection.
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

    /// The single-checker interposition mask, or `None` outside a single check.
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

    /// Record which exclusion and hand conditions this position exercises.
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
        // Nifu counts as fired only with a pawn in hand and a blocked file, so
        // that a candidate square really was removed.
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
    #[cfg_attr(miri, ignore)]
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
                assert_drops_match(&p, ALL_SQUARES, &format!("fixture {fi} ply {plies} [all]"));
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

    /// Positions where the nifu and uchifuzume exclusions each remove a genuine
    /// candidate square.
    #[test]
    fn nifu_and_uchifuzume_exclusions_fire_and_match() {
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

        // Dropping a pawn in front of the cornered white king is mate, so that
        // one square must be excluded.
        let p = parse_sfen("k8/9/G1N6/9/9/9/9/9/8K b P 1").expect("valid SFEN");
        let mate_sq = crate::square::Square::new(8, 1).unwrap(); // 9b, in front of 9a king
        let drops = assert_drops_match(&p, ALL_SQUARES, "uchifuzume fixture");
        assert!(
            !drops.iter().any(|&m| m.is_drop() && m.to_sq() == mate_sq),
            "uchifuzume exclusion failed: the pawn-drop-mate square was emitted",
        );
        // A non-mating pawn drop nearby is still present, so pawn drops are
        // otherwise being generated.
        assert!(
            drops.iter().any(|&m| m.is_drop()
                && m.dropped_piece_kind() == PieceKind::Pawn
                && m.to_sq() != mate_sq),
            "uchifuzume fixture emitted no other pawn drops — exclusion untested",
        );
    }
}
