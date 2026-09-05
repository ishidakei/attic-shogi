//! Root search data types and helpers, ported from the reference's
//! `start_searching` / `iterative_deepening` / `search<Root>`.
//!
//! This module holds the *pure* pieces: the value types, the root-move list
//! builder and the nyugyoku predicate. The glue that drives iterative deepening
//! lives on [`crate::qsearch::QSearch`], so that it shares one node counter,
//! transposition table, search stack, history bundle and network with the
//! interior and quiescence searches it recurses into.

use attic_state::{Color, ExtMove, Move, PieceKind, Position, Square};
use attic_storage::Value;

/// `VALUE_INFINITE` (`types.h`).
const VALUE_INFINITE: Value = 32001;

/// `MAX_PLY` (`config.h` → `types.h`).
const MAX_PLY: Value = 246;
/// `VALUE_MATE` (`types.h`).
const VALUE_MATE: Value = 32000;
/// `VALUE_TB_WIN_IN_MAX_PLY` (`types.h`): the `is_win` / `is_loss`
/// threshold, `VALUE_MATE - MAX_PLY`.
const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY;

/// `is_win(v)` (`types.h`): a proven-win (mate/TB-win) score.
fn is_win(v: Value) -> bool {
    v >= VALUE_TB_WIN_IN_MAX_PLY
}

/// `is_loss(v)` (`types.h`): a proven-loss (mated/TB-loss) score.
fn is_loss(v: Value) -> bool {
    v <= -VALUE_TB_WIN_IN_MAX_PLY
}

/// `RootMove::meanSquaredScore` initial value (`search.h`). The reference
/// stores it in an `int`; here it is `i64`, so that the `value * abs(value)`
/// moving average cannot overflow.
const MEAN_SQUARED_INIT: i64 = -(VALUE_INFINITE as i64 * VALUE_INFINITE as i64);

/// One root move and the per-iteration statistics `search<Root>` maintains for
/// it (`search.h`).
#[derive(Clone, Debug)]
pub struct RootMove {
    /// `pv[0]` — the move itself.
    pub mv: Move,
    /// Principal variation: `mv` followed by the searched child PV.
    pub pv: Vec<Move>,
    /// This iteration's search score, `-VALUE_INFINITE` when the move neither
    /// is the PV nor improved alpha; the stable sort keeps those in place.
    pub score: Value,
    /// The USI-reported score (equals `score`, or the clamped window bound on a
    /// fail high/low).
    pub uci_score: Value,
    /// The previous iteration's `score`, the sort tie-break.
    pub previous_score: Value,
    /// Moving average of `score`, an aspiration input.
    pub average_score: Value,
    /// Moving average of `score * |score|` (aspiration `delta` input).
    pub mean_squared_score: i64,
    /// `selDepth` recorded when this move became the PV.
    pub sel_depth: i32,
    /// `uciScore` is a lower bound (fail high).
    pub score_lowerbound: bool,
    /// `uciScore` is an upper bound (fail low).
    pub score_upperbound: bool,
    /// Nodes spent searching this root move's subtree.
    pub effort: u64,
}

impl RootMove {
    /// A fresh root move (`search.h`), every score at its sentinel.
    pub fn new(m: Move) -> Self {
        Self {
            mv: m,
            pv: vec![m],
            score: -VALUE_INFINITE,
            uci_score: -VALUE_INFINITE,
            previous_score: -VALUE_INFINITE,
            average_score: -VALUE_INFINITE,
            mean_squared_score: MEAN_SQUARED_INIT,
            sel_depth: 0,
            score_lowerbound: false,
            score_upperbound: false,
            effort: 0,
        }
    }
}

/// How the root search resolved (the reference's three `search_skipped`
/// pre-search exits plus the normal search).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RootKind {
    /// A normal search produced the result.
    Normal,
    /// No legal move: `bestmove resign` with `mated_in(1)`.
    Resign,
    /// `Position::DeclarationWin()` succeeded: `bestmove win` with `mate_in(1)`.
    DeclarationWin,
}

/// Outcome of a root search — the fields the USI `info` / `bestmove` output
/// consumes.
#[derive(Clone, Debug)]
pub struct RootOutcome {
    /// The chosen move. A real move for [`RootKind::Normal`]; the
    /// [`Move::resign`] / [`Move::win`] sentinel otherwise.
    pub best_move: Move,
    /// Score from the root side-to-move's point of view (`rootMoves[0].uciScore`,
    /// or `mated_in(1)` / `mate_in(1)` for the skipped exits).
    pub score: Value,
    /// Total nodes searched (`do_move` count); `0` for the skipped exits.
    pub nodes: u64,
    /// Principal variation (`rootMoves[0].pv`, possibly ponder-extended).
    pub pv: Vec<Move>,
    /// The depth the reported result was completed at: the last fully-completed
    /// iteration's, so that a time-aborted search reports the depth it actually
    /// finished rather than the one it was cut off in. `0` for the pre-search
    /// exits, whose reply is a bare `bestmove`.
    pub depth: i32,
    /// Maximum selective depth reached.
    pub sel_depth: i32,
    /// Which resolution produced this outcome.
    pub kind: RootKind,
}

/// The per-worker facts the thread vote consumes. Note the **raw**
/// `rootMoves[0].score`, not the USI-clamped one.
#[derive(Clone, Debug)]
pub struct WorkerVote {
    /// `rootMoves[0].score` — the raw search score the vote weights by.
    pub score: Value,
    /// `rootMoves[0].pv[0]` — the move this worker would play.
    pub pv0: Move,
    /// `rootMoves[0].pv.size()` — read by the truncated-PV tiebreak.
    pub pv_len: usize,
    /// `completedDepth` — the last fully-completed iterative-deepening depth.
    pub completed_depth: i32,
}

/// `get_best_thread` (`yaneuraou-search.cpp`): the index of the worker
/// whose result the engine reports. `workers[0]` must be the main worker, and
/// the slice is never empty.
///
/// Each worker's `pv[0]` is voted by `(score - minScore + 14) * completedDepth`;
/// then a proven-win incumbent only yields to a shorter mate, a proven-loss one
/// only to a longer defence, and otherwise the challenger wins on a proven
/// decisive score or a higher vote total.
pub fn select_best_worker(workers: &[WorkerVote]) -> usize {
    debug_assert!(!workers.is_empty());

    // Seeded from `VALUE_NONE` as the reference does (609-611), though every
    // worker has a valid score so the seed is always superseded.
    let min_score = workers.iter().map(|w| w.score).min().unwrap_or(0);

    // Widened to `i64` (614-619), so that the product cannot overflow.
    let voting_value = |w: &WorkerVote| -> i64 {
        (w.score as i64 - min_score as i64 + 14) * w.completed_depth as i64
    };

    // `votes[pv[0]] += voting_value(w)` summed over workers (621-622).
    let mut votes: std::collections::HashMap<Move, i64> = std::collections::HashMap::new();
    for w in workers {
        *votes.entry(w.pv0).or_insert(0) += voting_value(w);
    }

    // `best` starts at the main worker, so its first comparison is a no-op.
    let mut best = 0usize;
    for i in 0..workers.len() {
        let cur = &workers[best];
        let th = &workers[i];

        let best_score = cur.score;
        let new_score = th.score;

        let best_vote = votes[&cur.pv0];
        let new_vote = votes[&th.pv0];

        let best_in_win = is_win(best_score);
        let new_in_win = is_win(new_score);
        let best_in_loss = best_score != -VALUE_INFINITE && is_loss(best_score);
        let new_in_loss = new_score != -VALUE_INFINITE && is_loss(new_score);

        // A vote total only counts when the PV has more than two moves.
        let better_voting_value =
            voting_value(th) * (th.pv_len > 2) as i64 > voting_value(cur) * (cur.pv_len > 2) as i64;

        if best_in_win {
            // Keep the shortest mate: switch only to a higher proven-win score.
            if new_score > best_score {
                best = i;
            }
        } else if best_in_loss {
            // Keep the longest defence: switch only to a lower proven loss.
            if new_in_loss && new_score < best_score {
                best = i;
            }
        } else if new_in_win
            || new_in_loss
            || (!is_loss(new_score)
                && (new_vote > best_vote || (new_vote == best_vote && better_voting_value)))
        {
            best = i;
        }
    }

    best
}

/// Build the root-move list in `MoveList<LEGAL>` order (`thread.cpp`).
///
/// The legality pass is the reference's **swap-with-tail** compaction
/// (`movegen.cpp`): an illegal move at the cursor is overwritten by the
/// current last move and the list shrinks. That is *not* an order-preserving
/// filter, and reproducing it exactly is what fixes `rootMoves[0]` — the move
/// the root search treats as its TT move — to the reference.
pub fn generate_root_moves(pos: &Position, all: bool) -> Vec<RootMove> {
    let mut pseudo: Vec<ExtMove> = Vec::new();
    if pos.in_check() {
        pos.generate_evasions(all, &mut pseudo);
    } else {
        pos.generate_non_evasions(all, &mut pseudo);
    }

    // `while (cur != last) if (!legal(*cur)) *cur = *(--last); else ++cur;`
    let mut cur = 0usize;
    let mut last = pseudo.len();
    while cur != last {
        if pos.is_legal(pseudo[cur].mv) {
            cur += 1;
        } else {
            pseudo[cur] = pseudo[last - 1];
            last -= 1;
        }
    }
    pseudo.truncate(last);

    pseudo.into_iter().map(|e| RootMove::new(e.mv)).collect()
}

/// The entering-king (nyugyoku) declaration rule (`types.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnteringKingRule {
    /// `EKR_NONE` / `NoEnteringKing` — declaration disabled everywhere.
    None,
    /// `EKR_24_POINT` / `CSARule24` — 24-point law (declare at 31+, both sides).
    Point24,
    /// `EKR_24_POINT_H` / `CSARule24H` — 24-point law, handicap-aware.
    Point24H,
    /// `EKR_27_POINT` / `CSARule27` — CSA 27-point law (Black 28, White 27).
    Point27,
    /// `EKR_27_POINT_H` / `CSARule27H` — CSA 27-point law, handicap-aware.
    Point27H,
    /// `EKR_TRY_RULE` / `TryRule` — reach the opponent king's initial square.
    Try,
}

impl EnteringKingRule {
    /// The `EKR_STRINGS` choice list (`types.cpp`) in the reference's
    /// exact order — the `var` values of the `EnteringKingRule` combo option.
    pub const STRINGS: [&'static str; 6] = [
        "NoEnteringKing",
        "CSARule24",
        "CSARule24H",
        "CSARule27",
        "CSARule27H",
        "TryRule",
    ];

    /// Map an option string to its rule (`types.cpp`). An unrecognised
    /// string falls back to [`EnteringKingRule::None`], as the reference's
    /// post-assert return does.
    pub fn from_option(s: &str) -> Self {
        match s {
            "NoEnteringKing" => Self::None,
            "CSARule24" => Self::Point24,
            "CSARule24H" => Self::Point24H,
            "CSARule27" => Self::Point27,
            "CSARule27H" => Self::Point27H,
            "TryRule" => Self::Try,
            _ => Self::None,
        }
    }
}

/// The entering-king rule plus its per-side point thresholds, precomputed once
/// per `go` from the root position (the reference's `set_ekr` state).
///
/// The total material across board and hands is invariant over a game — a
/// capture only moves a piece to a hand — so a root snapshot is exact for every
/// node of the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnteringKingConfig {
    rule: EnteringKingRule,
    /// `enteringKingPoint[]`, indexed by [`Color::index`] (Black 0, White 1).
    points: [i32; Color::COUNT],
}

impl EnteringKingConfig {
    /// Snapshot `rule` against `root` and precompute `enteringKingPoint[]`
    /// (`Position::update_entering_point`, `position.cpp`).
    pub fn new(rule: EnteringKingRule, root: &Position) -> Self {
        Self {
            rule,
            points: entering_king_points(rule, root),
        }
    }

    /// The configured rule.
    pub fn rule(&self) -> EnteringKingRule {
        self.rule
    }
}

impl Default for EnteringKingConfig {
    /// The option default with its fixed thresholds. The non-`_H` point rules
    /// never adjust for material, so this needs no position.
    fn default() -> Self {
        Self {
            rule: EnteringKingRule::Point27,
            points: [28, 27],
        }
    }
}

/// `Position::update_entering_point` (`position.cpp`): the per-side
/// entering-king point thresholds for `rule`, computed from `pos`.
fn entering_king_points(rule: EnteringKingRule, pos: &Position) -> [i32; Color::COUNT] {
    let mut points = match rule {
        EnteringKingRule::Point24 | EnteringKingRule::Point24H => [31, 31],
        EnteringKingRule::Point27 | EnteringKingRule::Point27H => [28, 27],
        // `EKR_NONE` / `EKR_TRY_RULE` never consult the thresholds.
        EnteringKingRule::None | EnteringKingRule::Try => return [0, 0],
    };

    if matches!(
        rule,
        EnteringKingRule::Point24H | EnteringKingRule::Point27H
    ) {
        // Every board piece scores 1, kings included, and a big piece 4 more;
        // hands add 1 per small piece and 5 per big one. A full set is 56.
        let mut p: i32 = 0;
        for sq in (0..Square::COUNT as u8).filter_map(Square::from_index) {
            if let Some(pc) = pos.board().get(sq) {
                p += 1;
                if matches!(pc.kind, PieceKind::Bishop | PieceKind::Rook) {
                    p += 4;
                }
            }
        }
        for color in [Color::Black, Color::White] {
            let h = pos.hand(color);
            let c = |k: PieceKind| h.count(k) as i32;
            p += c(PieceKind::Pawn)
                + c(PieceKind::Lance)
                + c(PieceKind::Knight)
                + c(PieceKind::Silver)
                + c(PieceKind::Gold)
                + (c(PieceKind::Bishop) + c(PieceKind::Rook)) * 5;
        }
        // The deficit from a full set is charged to White alone, the handicap
        // giver being treated as White (`position.cpp`).
        if p != 56 {
            points[Color::White.index()] -= 56 - p;
        }
    }

    points
}

/// `Position::DeclarationWin()` (`position.cpp`).
///
/// A point rule declares when the side to move's king is inside the enemy three
/// ranks, it is not in check, it has at least 11 pieces there counting the king,
/// and its point total reaches `config`'s per-side threshold. The try rule
/// instead returns the king move onto the opponent king's initial square, when
/// that square is adjacent, holds no own piece, and is unattacked once our king
/// vacates.
pub fn declaration_win(pos: &Position, config: &EnteringKingConfig) -> Option<Move> {
    match config.rule {
        EnteringKingRule::None => None,
        EnteringKingRule::Point24
        | EnteringKingRule::Point24H
        | EnteringKingRule::Point27
        | EnteringKingRule::Point27H => declaration_win_points(pos, config.points),
        EnteringKingRule::Try => declaration_win_try(pos),
    }
}

/// The CSA point-law branch (`position.cpp`) with the per-side
/// threshold taken from the precomputed `points` (`enteringKingPoint[us]`).
fn declaration_win_points(pos: &Position, points: [i32; Color::COUNT]) -> Option<Move> {
    let us = pos.side_to_move();

    // (e) not in check.
    if pos.in_check() {
        return None;
    }

    // The enemy field is the three ranks nearest the opponent: ranks 0..=2 for
    // Black (which advances toward rank 0), ranks 6..=8 for White.
    let in_enemy_field = |sq: Square| match us {
        Color::Black => sq.rank() <= 2,
        Color::White => sq.rank() >= 6,
    };

    // (b) the king is inside the enemy field.
    let king_sq = find_king(pos, us)?;
    if !in_enemy_field(king_sq) {
        return None;
    }

    // (d) at least 11 own pieces (king included) in the enemy field, and the
    // count of big pieces (bishop/rook incl. horse/dragon) there.
    let mut p1: i32 = 0;
    let mut p2: i32 = 0;
    for sq in (0..Square::COUNT as u8).filter_map(Square::from_index) {
        if let Some(p) = pos.board().get(sq)
            && p.color == us
            && in_enemy_field(sq)
        {
            p1 += 1;
            if matches!(p.kind, PieceKind::Bishop | PieceKind::Rook) {
                p2 += 1;
            }
        }
    }
    if p1 < 11 {
        return None;
    }

    // (c) point total: small pieces 1, big pieces 5, king excluded (`- 1`).
    let h = pos.hand(us);
    let count = |k: PieceKind| h.count(k) as i32;
    let score = p1 + p2 * 4 - 1
        + count(PieceKind::Pawn)
        + count(PieceKind::Lance)
        + count(PieceKind::Knight)
        + count(PieceKind::Silver)
        + count(PieceKind::Gold)
        + (count(PieceKind::Bishop) + count(PieceKind::Rook)) * 5;

    if score < points[us.index()] {
        return None;
    }

    Some(Move::win())
}

/// The try-rule branch (`position.cpp`).
fn declaration_win_try(pos: &Position) -> Option<Move> {
    let us = pos.side_to_move();

    // The opponent king's initial square (`position.cpp`). It is a fixed
    // on-board coordinate, so the `?` is only for totality.
    let try_sq = match us {
        Color::Black => Square::new(4, 0),
        Color::White => Square::new(4, 8),
    }?;

    let king_sq = find_king(pos, us)?;

    // 1) our king can step onto the try square (kingEffect adjacency).
    if !king_adjacent(king_sq, try_sq) {
        return None;
    }

    // An enemy piece on the try square is a capture-try and does not block.
    if matches!(pos.board().get(try_sq), Some(p) if p.color == us) {
        return None;
    }

    // 3) the opponent has no effect on the try square once our king leaves
    //    king_sq (`effected_to(~us, king_try_sq, king_sq)`).
    if pos.is_attacked_discounting(try_sq, us.flip(), king_sq) {
        return None;
    }

    // Encoded exactly as the generator would, and legal by the conditions
    // above, so the driver can emit it verbatim as `bestmove`.
    let king_piece = pos.board().get(king_sq)?;
    Some(Move::make(king_sq, try_sq, king_piece))
}

/// The side-to-move-agnostic king locator shared by the declaration branches.
fn find_king(pos: &Position, us: Color) -> Option<Square> {
    (0..Square::COUNT as u8)
        .filter_map(Square::from_index)
        .find(|&sq| {
            matches!(pos.board().get(sq), Some(p) if p.color == us && p.kind == PieceKind::King)
        })
}

/// True iff `sq` lies on one of `king_sq`'s eight king steps (`kingEffect`).
fn king_adjacent(king_sq: Square, sq: Square) -> bool {
    let df = (king_sq.file() as i8 - sq.file() as i8).abs();
    let dr = (king_sq.rank() as i8 - sq.rank() as i8).abs();
    df <= 1 && dr <= 1 && (df, dr) != (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use attic_state::parse_sfen;
    use std::collections::HashSet;

    fn pos(sfen: &str) -> Position {
        parse_sfen(sfen).expect("valid SFEN")
    }

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    const IN_CHECK: &str = "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1";
    const CHECKMATE: &str = "4K4/3ggg3/4k4/9/9/9/9/9/9 b - 1";
    // Black king on 5b inside the enemy field, 12 own pieces there (2 dragons +
    // 2 horses = 4 big) plus a rook in hand: 12 + 4*4 - 1 + 5 = 32 points, a CSA
    // 27-point win, and (unlike a fully walled-in king) it still has legal moves.
    const NYUGYOKU: &str = "+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/4k4 b R 1";

    #[test]
    fn root_moves_equal_legal_captures_and_quiets_when_not_in_check() {
        let p = pos(STARTPOS);
        assert!(!p.in_check());
        let rm: HashSet<Move> = generate_root_moves(&p, false)
            .into_iter()
            .map(|r| r.mv)
            .collect();
        assert!(!rm.is_empty());

        let mut caps: Vec<ExtMove> = Vec::new();
        p.generate_captures(false, &mut caps);
        let mut quiets: Vec<ExtMove> = Vec::new();
        p.generate_quiets(false, &mut quiets);
        let legal: HashSet<Move> = caps
            .into_iter()
            .chain(quiets)
            .map(|e| e.mv)
            .filter(|&m| p.is_legal(m))
            .collect();
        assert_eq!(rm, legal);
    }

    #[test]
    fn root_moves_equal_legal_evasions_when_in_check() {
        let p = pos(IN_CHECK);
        assert!(p.in_check());
        let rm: HashSet<Move> = generate_root_moves(&p, false)
            .into_iter()
            .map(|r| r.mv)
            .collect();

        let mut ev: Vec<ExtMove> = Vec::new();
        p.generate_evasions(false, &mut ev);
        let legal: HashSet<Move> = ev
            .into_iter()
            .map(|e| e.mv)
            .filter(|&m| p.is_legal(m))
            .collect();
        assert_eq!(rm, legal);
    }

    #[test]
    fn root_moves_empty_when_checkmated() {
        let p = pos(CHECKMATE);
        assert!(p.in_check());
        assert!(generate_root_moves(&p, false).is_empty());
    }

    #[test]
    fn declaration_win_declines_a_normal_position() {
        let cfg = EnteringKingConfig::default();
        assert!(declaration_win(&pos(STARTPOS), &cfg).is_none());
        // In check ⇒ no declaration even with pieces forward.
        assert!(declaration_win(&pos(IN_CHECK), &cfg).is_none());
    }

    #[test]
    fn declaration_win_detects_a_nyugyoku_position() {
        let p = pos(NYUGYOKU);
        assert!(!p.in_check());
        assert_eq!(
            declaration_win(&p, &EnteringKingConfig::default()),
            Some(Move::win())
        );
    }

    // -----------------------------------------------------------------
    // Rule-aware declaration. Positions are hand-built so a single
    // point of the score / count / threshold formula is exercised at a time.
    // -----------------------------------------------------------------

    /// A rule's config against a position (thresholds come from the position for
    /// the `_H` variants; the point/`None`/`Try` cases ignore it).
    fn cfg(rule: EnteringKingRule, p: &Position) -> EnteringKingConfig {
        EnteringKingConfig::new(rule, p)
    }

    #[test]
    fn point27_black_threshold_is_28() {
        // Black king + 10 golds in the enemy field (p1 = 11, board score 10) plus
        // a hand of 18 pawns → 28, exactly the Black threshold. (Hand counts are
        // capped at their real per-piece maxima — e.g. a rook holds at most 2 —
        // so pawns are used to dial the score freely.)
        let declare = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 18P 1");
        assert_eq!(
            declaration_win(&declare, &cfg(EnteringKingRule::Point27, &declare)),
            Some(Move::win())
        );
        // One point of hand less (17P → 17) → 27 < 28, no declaration.
        let decline = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 17P 1");
        assert!(declaration_win(&decline, &cfg(EnteringKingRule::Point27, &decline)).is_none());
    }

    #[test]
    fn point27_white_threshold_is_27() {
        // Mirror image: White king + 10 golds in White's enemy field (ranks g-i),
        // hand of 17 pawns → 27, exactly the White threshold.
        let declare = pos("8K/9/9/9/9/9/9/gg7/ggggkgggg w 17p 1");
        assert_eq!(
            declaration_win(&declare, &cfg(EnteringKingRule::Point27, &declare)),
            Some(Move::win())
        );
        // 16p → 26 < 27, no declaration.
        let decline = pos("8K/9/9/9/9/9/9/gg7/ggggkgggg w 16p 1");
        assert!(declaration_win(&decline, &cfg(EnteringKingRule::Point27, &decline)).is_none());
    }

    #[test]
    fn point24_threshold_is_31_both_colors() {
        // 24-point law: both sides need 31. Board score 10 + hand 21 (2R + 11P) = 31.
        let black_declare = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 2R11P 1");
        assert_eq!(
            declaration_win(
                &black_declare,
                &cfg(EnteringKingRule::Point24, &black_declare)
            ),
            Some(Move::win())
        );
        let black_decline = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 2R10P 1"); // 30
        assert!(
            declaration_win(
                &black_decline,
                &cfg(EnteringKingRule::Point24, &black_decline)
            )
            .is_none()
        );

        let white_declare = pos("8K/9/9/9/9/9/9/gg7/ggggkgggg w 2r11p 1");
        assert_eq!(
            declaration_win(
                &white_declare,
                &cfg(EnteringKingRule::Point24, &white_declare)
            ),
            Some(Move::win())
        );
        let white_decline = pos("8K/9/9/9/9/9/9/gg7/ggggkgggg w 2r10p 1"); // 30
        assert!(
            declaration_win(
                &white_decline,
                &cfg(EnteringKingRule::Point24, &white_decline)
            )
            .is_none()
        );
    }

    #[test]
    fn piece_count_gate_needs_eleven_in_the_enemy_field() {
        // Hand 19 (2R + 9P) makes the score pass for both fields on its own — the
        // 11-piece field scores 10 + 19 = 29 and the 10-piece field would score
        // 9 + 19 = 28, both ≥ 28. So the only thing that can decline the 10-piece
        // case is the count gate (condition (d)), which rejects before the score
        // is ever consulted.
        let eleven = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 2R9P 1"); // king + 10 golds
        assert_eq!(
            declaration_win(&eleven, &cfg(EnteringKingRule::Point27, &eleven)),
            Some(Move::win())
        );
        let ten = pos("GGGGKGGGG/G8/9/9/9/9/9/9/8k b 2R9P 1"); // king + 9 golds
        assert!(declaration_win(&ten, &cfg(EnteringKingRule::Point27, &ten)).is_none());
    }

    #[test]
    fn king_outside_enemy_field_never_declares() {
        // Black king on rank d (outside the enemy three ranks) with a large hand.
        let p = pos("9/9/9/4K4/9/9/9/9/8k b 2R4P 1");
        assert!(declaration_win(&p, &cfg(EnteringKingRule::Point27, &p)).is_none());
    }

    #[test]
    fn big_piece_on_board_counts_five() {
        // king + 8 golds + 1 dragon + 1 gold in the enemy field: p1 = 11, one big
        // piece → board score 11 + 4·1 − 1 = 14, plus hand 14 (2R4P) = 28. The
        // dragon's +4 over a small piece is the only reason it reaches 28.
        let with_dragon = pos("GGGGKGGGG/+RG7/9/9/9/9/9/9/8k b 2R4P 1");
        assert_eq!(
            declaration_win(&with_dragon, &cfg(EnteringKingRule::Point27, &with_dragon)),
            Some(Move::win())
        );
        // Same shape with the dragon replaced by a small piece → 24 < 28.
        let all_small = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 2R4P 1");
        assert!(declaration_win(&all_small, &cfg(EnteringKingRule::Point27, &all_small)).is_none());
    }

    #[test]
    fn handicap_variants_lower_white_threshold_by_the_deficit() {
        // Startpos with White's rook and bishop removed: full-set 56 minus two big
        // pieces (5 + 5) → material total 46, a deficit of 10.
        let handicap = pos("lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");

        // `_H` rules charge the whole deficit to White; Black never moves.
        assert_eq!(
            entering_king_points(EnteringKingRule::Point27H, &handicap),
            [28, 17]
        );
        assert_eq!(
            entering_king_points(EnteringKingRule::Point24H, &handicap),
            [31, 21]
        );
        // The non-handicap rules ignore the deficit entirely.
        assert_eq!(
            entering_king_points(EnteringKingRule::Point27, &handicap),
            [28, 27]
        );
        assert_eq!(
            entering_king_points(EnteringKingRule::Point24, &handicap),
            [31, 31]
        );
        // A complete set leaves every threshold at its base value, even under `_H`.
        let full = pos(STARTPOS);
        assert_eq!(
            entering_king_points(EnteringKingRule::Point27H, &full),
            [28, 27]
        );
        assert_eq!(
            entering_king_points(EnteringKingRule::Point24H, &full),
            [31, 31]
        );
    }

    #[test]
    fn no_entering_king_never_declares() {
        // A clearly declarable 27-point position yields nothing under `None`.
        let p = pos(NYUGYOKU);
        assert!(declaration_win(&p, &cfg(EnteringKingRule::None, &p)).is_none());
    }

    // --- TryRule (position.cpp). ---

    fn sq(file: u8, rank: u8) -> Square {
        Square::new(file, rank).expect("on-board square")
    }

    #[test]
    fn try_rule_black_declares_king_move_to_5a() {
        // Black king on 5b, the try square 5a empty and unattacked.
        let p = pos("9/4K4/9/9/9/9/9/9/8k b - 1");
        let king_sq = sq(4, 1);
        let try_sq = sq(4, 0);
        let expected = Move::make(king_sq, try_sq, p.board().get(king_sq).unwrap());
        assert_eq!(
            declaration_win(&p, &cfg(EnteringKingRule::Try, &p)),
            Some(expected)
        );
        // The returned move is exactly what the normal generator produces.
        let mut legal = Vec::new();
        p.generate_legal_all(&mut legal);
        assert!(legal.contains(&expected));
    }

    #[test]
    fn try_rule_white_declares_king_move_to_5i() {
        // White king on 5h, the try square 5i empty and unattacked.
        let p = pos("8K/9/9/9/9/9/9/4k4/9 w - 1");
        let king_sq = sq(4, 7);
        let try_sq = sq(4, 8);
        let expected = Move::make(king_sq, try_sq, p.board().get(king_sq).unwrap());
        assert_eq!(
            declaration_win(&p, &cfg(EnteringKingRule::Try, &p)),
            Some(expected)
        );
    }

    #[test]
    fn try_rule_requires_king_adjacency() {
        // Black king on 5c is two ranks from 5a → not adjacent → no declaration.
        let p = pos("9/9/4K4/9/9/9/9/9/8k b - 1");
        assert!(declaration_win(&p, &cfg(EnteringKingRule::Try, &p)).is_none());
    }

    #[test]
    fn try_rule_own_piece_on_the_square_blocks() {
        // Black gold sits on 5a → the try square holds an own piece → blocked.
        let p = pos("4G4/4K4/9/9/9/9/9/9/8k b - 1");
        assert!(declaration_win(&p, &cfg(EnteringKingRule::Try, &p)).is_none());
    }

    #[test]
    fn try_rule_enemy_piece_on_the_square_is_a_capture_try() {
        // A White knight on 5a does not defend its own square and (from 5a) does
        // not attack 5b, so the king may capture-try onto it.
        let p = pos("4n4/4K4/9/9/9/9/9/9/8k b - 1");
        let expected = Move::make(sq(4, 1), sq(4, 0), p.board().get(sq(4, 1)).unwrap());
        assert_eq!(
            declaration_win(&p, &cfg(EnteringKingRule::Try, &p)),
            Some(expected)
        );
    }

    #[test]
    fn try_rule_attacked_square_blocks() {
        // A White rook on 9a rakes rank a, so the try square 5a is attacked
        // (independently of our king) → blocked.
        let p = pos("r8/4K4/9/9/9/9/9/9/8k b - 1");
        assert!(declaration_win(&p, &cfg(EnteringKingRule::Try, &p)).is_none());
    }

    #[test]
    fn try_rule_attack_test_discounts_the_moving_king() {
        // A White rook on 5c is blocked from 5a by our own king on 5b. The attack
        // test removes the moving king from the occupancy (the reference's
        // `pieces() ^ kingSq`), revealing the rook's attack on 5a → blocked. A
        // naive check that left the king in place would wrongly allow the try.
        let p = pos("9/4K4/4r4/9/9/9/9/9/8k b - 1");
        let king_sq = sq(4, 1);
        let try_sq = sq(4, 0);
        assert!(declaration_win(&p, &cfg(EnteringKingRule::Try, &p)).is_none());
        // Discounting the king reveals the rook; discounting an unrelated empty
        // square leaves the king blocking it — pinpointing the king as the cause.
        assert!(p.is_attacked_discounting(try_sq, Color::White, king_sq));
        assert!(!p.is_attacked_discounting(try_sq, Color::White, sq(8, 8)));
    }

    // -----------------------------------------------------------------
    // Lazy-SMP thread vote (`select_best_worker` / `get_best_thread`).
    // -----------------------------------------------------------------

    /// Two distinct legal moves from startpos, used as the workers' `pv[0]` votes.
    fn two_moves() -> (Move, Move) {
        let rm = generate_root_moves(&pos(STARTPOS), false);
        (rm[0].mv, rm[1].mv)
    }

    fn vote(score: Value, pv0: Move, pv_len: usize, completed_depth: i32) -> WorkerVote {
        WorkerVote {
            score,
            pv0,
            pv_len,
            completed_depth,
        }
    }

    #[test]
    fn vote_single_worker_degenerates_to_main() {
        let (a, _) = two_moves();
        assert_eq!(select_best_worker(&[vote(100, a, 1, 5)]), 0);
    }

    #[test]
    fn vote_plain_winner_is_the_higher_vote_total() {
        let (a, b) = two_moves();
        // Equal scores; worker 1 searched deeper, so its move's vote total is
        // higher: votes[a] = 14*5 = 70, votes[b] = 14*10 = 140.
        let workers = [vote(100, a, 3, 5), vote(100, b, 3, 10)];
        assert_eq!(select_best_worker(&workers), 1);
        // Reversed depths ⇒ the main worker (index 0) keeps the higher total.
        let workers = [vote(100, a, 3, 10), vote(100, b, 3, 5)];
        assert_eq!(select_best_worker(&workers), 0);
    }

    #[test]
    fn vote_proven_win_prefers_the_shorter_mate() {
        let (a, b) = two_moves();
        // Incumbent (index 0) is a proven win (mate in 5); a higher proven-win
        // score (mate in 3) is a shorter mate and wins.
        let mate_in = |ply: Value| VALUE_MATE - ply;
        let workers = [vote(mate_in(5), a, 1, 8), vote(mate_in(3), b, 1, 8)];
        assert_eq!(select_best_worker(&workers), 1);
        // A *longer* mate (lower proven-win score) never displaces the incumbent.
        let workers = [vote(mate_in(5), a, 1, 8), vote(mate_in(7), b, 1, 8)];
        assert_eq!(select_best_worker(&workers), 0);
    }

    #[test]
    fn vote_proven_loss_prefers_the_lower_score() {
        let (a, b) = two_moves();
        // Incumbent (index 0) is a proven loss; a *lower* proven-loss score wins
        // (yaneuraou-search.cpp — `newThreadScore < bestThreadScore`).
        let mated_in = |ply: Value| -VALUE_MATE + ply;
        let workers = [vote(mated_in(7), a, 1, 8), vote(mated_in(3), b, 1, 8)];
        assert_eq!(select_best_worker(&workers), 1);
        // A higher proven-loss score does not displace the incumbent.
        let workers = [vote(mated_in(3), a, 1, 8), vote(mated_in(7), b, 1, 8)];
        assert_eq!(select_best_worker(&workers), 0);
    }

    #[test]
    fn vote_truncated_pv_breaks_a_vote_tie() {
        let (a, b) = two_moves();
        // Equal scores and equal depths ⇒ equal vote totals for the two moves.
        // The tie is broken by the truncated-PV guard: the worker whose PV has
        // more than two moves wins.
        let workers = [vote(50, a, 2, 5), vote(50, b, 3, 5)];
        assert_eq!(select_best_worker(&workers), 1);
        // If the incumbent has the longer PV instead, the tie leaves it in place
        // (the challenger's `pv.len() > 2` factor is 0, so it cannot exceed).
        let workers = [vote(50, a, 3, 5), vote(50, b, 2, 5)];
        assert_eq!(select_best_worker(&workers), 0);
    }
}
