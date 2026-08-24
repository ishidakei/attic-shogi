//! Search layer: greedy 1-ply NNUE move choice.
//!
//! This crate is the Search layer. The layering rules permit Search to depend
//! on Evaluation (`attic-eval`) and State (`attic-state`), never on
//! Protocol; the Protocol
//! layer, in turn, calls into this crate and never reaches Evaluation
//! directly.
//!
//! The policy is the simplest NNUE-guided one: greedy
//! 1-ply. [`Search::go`] enumerates the legal moves, scores each child with
//! the real network from the mover's point of view, and returns the maximum.
//! There is no alpha-beta, no quiescence, no transposition table, no time
//! management, and no reference-parity claim. The
//! point is a clean, tested search seam shaped like
//! `Search::go(&pos, limits, info) -> SearchResult`, which the USI driver can
//! call without importing the Evaluation layer.

use std::path::Path;

use attic_eval::{
    Accumulator, NnueError, NnueNetwork, evaluate_with, load_network, load_network_with_warnings,
};
use attic_state::{Move, Position};

pub mod book;
pub mod history;
pub mod movepick;
pub mod qsearch;
pub mod root;
pub mod timeman;
pub mod update;

pub use book::{BookConfig, BookHit, BookInfoLine, BookProbeResult, Prng, probe_book};
// Re-exported so the Protocol layer can drive the NNUE fixed-point output scale
// (the `FV_SCALE` USI option) without a direct Protocol -> Evaluation
// dependency, which the layering rules forbid. `set_fv_scale` writes the
// eval's live scale; `FV_SCALE_DEFAULT` is the reference default (16).
pub use attic_eval::{FV_SCALE_DEFAULT, fv_scale, set_fv_scale};
pub use timeman::{TimeInput, TimeManagement};

pub use history::{
    ButterflyHistory, CapturePieceToHistory, ContinuationCorrectionHistory, ContinuationHistory,
    CorrChannel, LOW_PLY_HISTORY_SIZE, LowPlyHistory, PieceToHistory, SharedHistories,
    TtMoveHistory, apply_gravity,
};
pub use movepick::MovePicker;
pub use qsearch::{
    PonderSignal, PvBound, PvInfo, PvOutputConfig, PvSink, QSearch, QSearchOutcome, SearchControl,
    TimeControl, WorkerResult, fail_lh_pv_gate,
};
pub use root::{
    EnteringKingConfig, EnteringKingRule, RootKind, RootMove, RootOutcome, WorkerVote,
    declaration_win, generate_root_moves, select_best_worker,
};
pub use update::{
    SearchStackCell, WorkerHistories, update_all_stats, update_continuation_histories,
    update_correction_history, update_quiet_histories,
};

/// Search limits for one `go` invocation.
///
/// Greedy 1-ply consults no limit. The type exists so [`Search::go`]'s
/// signature is stable once depth/time/node budgets are threaded in, and so the
/// USI driver can map its `GoLimits` onto a Search-layer type without this
/// crate depending on Protocol.
#[derive(Debug, Clone, Default)]
pub struct SearchLimits {}

/// One progress report emitted through an [`InfoSink`] during a search.
///
/// Mirrors the fields a USI `info` line carries; the driver formats them into
/// the wire syntax. Exact token order is not a parity target.
#[derive(Debug, Clone)]
pub struct SearchInfo {
    /// Search depth this report describes. Always `1` for greedy 1-ply.
    pub depth: u32,
    /// Score of the principal variation, in centipawns, from the root side to
    /// move's point of view.
    pub score_cp: i32,
    /// Number of nodes examined — here, the count of legal moves evaluated.
    pub nodes: u64,
    /// Principal variation. For greedy 1-ply this is the single chosen move.
    pub pv: Vec<Move>,
}

/// Sink for search progress reports.
///
/// Abstracts where `info` output goes so [`Search`] never touches the Protocol
/// layer. The USI driver supplies an implementation that formats each report
/// as an `info` line; tests supply a recording sink.
pub trait InfoSink {
    /// Receive one progress report.
    fn info(&mut self, info: &SearchInfo);
}

/// An [`InfoSink`] that discards every report.
///
/// Useful for callers (and tests) that only want the returned
/// [`SearchResult`].
#[derive(Debug, Default, Clone, Copy)]
pub struct NullInfoSink;

impl InfoSink for NullInfoSink {
    fn info(&mut self, _info: &SearchInfo) {}
}

/// Outcome of a [`Search::go`] call.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The chosen move, or `None` when the side to move has no legal move (the
    /// driver maps `None` to `bestmove resign`).
    pub best_move: Option<Move>,
    /// Score of the chosen move in centipawns, from the root side to move's
    /// point of view. `0` when there is no legal move.
    pub score_cp: i32,
    /// Number of legal moves evaluated.
    pub nodes: u64,
}

/// A greedy 1-ply search that owns a loaded NNUE network.
///
/// Construct it from a network file with [`Search::from_network_file`] (the
/// path the USI driver resolves from `EvalDir` at `isready`) or directly from
/// an in-memory [`NnueNetwork`] with [`Search::new`].
pub struct Search {
    net: NnueNetwork,
}

impl Search {
    /// Wrap an already-loaded network.
    pub fn new(net: NnueNetwork) -> Self {
        Self { net }
    }

    /// Load and validate the network at `path`, then wrap it.
    ///
    /// Surfaces the loader's [`NnueError`] unchanged so the caller can report a
    /// precise reason. Any non-fatal warnings are discarded; use
    /// [`Search::from_network_file_with_warnings`] to surface them.
    pub fn from_network_file(path: &Path) -> Result<Self, NnueError> {
        Ok(Self::new(load_network(path)?))
    }

    /// Like [`Search::from_network_file`], but also returns the loader's
    /// non-fatal warning bodies (hash mismatches) so the driver can emit them as
    /// `info string` lines before `readyok`. An empty vector means a clean load.
    pub fn from_network_file_with_warnings(path: &Path) -> Result<(Self, Vec<String>), NnueError> {
        let (net, warnings) = load_network_with_warnings(path)?;
        Ok((Self::new(net), warnings))
    }

    /// The network this search evaluates with.
    pub fn network(&self) -> &NnueNetwork {
        &self.net
    }

    /// A deep copy of this search with freshly allocated network storage
    /// ([`NnueNetwork::replicate`]).
    ///
    /// The driver runs this inside a NUMA-node-bound thread to place the copy's
    /// pages on that node (the in-process NNUE replication). The
    /// replica evaluates every position identically to `self`.
    pub fn replicate(&self) -> Self {
        Self::new(self.net.replicate())
    }

    /// Greedy 1-ply move choice.
    ///
    /// For every legal move, evaluates the child position with the real network
    /// from the mover's point of view — the child's [`evaluate_with`] is from
    /// the opponent's perspective, so it is negated — and picks the maximum.
    /// The tie-break is the first maximum in the deterministic legal-move
    /// generation order (a strict `>` comparison keeps the earliest).
    ///
    /// The child accumulator is derived incrementally from the root
    /// ([`Accumulator::update_after_move`]); the eval crate guarantees this is
    /// bit-identical to a from-scratch refresh, and a unit test pins the greedy
    /// choice to an independent full-refresh argmax.
    ///
    /// Emits one [`SearchInfo`] report (depth 1) through `info` when a move is
    /// chosen. Returns [`SearchResult::best_move`] `= None` — and emits no
    /// report — when the side to move has no legal move.
    ///
    /// # Panics
    /// Panics if `pos` (or any child) is missing either king, via the eval
    /// crate's feature extraction and bucket selection.
    pub fn go(
        &self,
        pos: &Position,
        limits: &SearchLimits,
        info: &mut dyn InfoSink,
    ) -> SearchResult {
        let _ = limits;

        let mut moves: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut moves);
        if moves.is_empty() {
            return SearchResult {
                best_move: None,
                score_cp: 0,
                nodes: 0,
            };
        }

        // A mutable working copy so the child accumulator can be built and the
        // child evaluated without disturbing the caller's position.
        let mut work = pos.clone();
        let mut root_acc = Accumulator::new();
        root_acc.refresh(&self.net, &work);

        let mut best_move = moves[0];
        let mut best_score = i32::MIN;
        for &mv in &moves {
            // `update_after_move` leaves `work` unchanged; apply the move for
            // real so `evaluate_with` sees the child's side to move and king
            // ranks, then undo it.
            let child_acc = root_acc.update_after_move(&self.net, &mut work, mv);
            let undo = work.do_move(mv);
            let score = -evaluate_with(&self.net, &child_acc, &work);
            work.undo_move(mv, undo);

            if score > best_score {
                best_score = score;
                best_move = mv;
            }
        }

        let nodes = moves.len() as u64;
        info.info(&SearchInfo {
            depth: 1,
            score_cp: best_score,
            nodes,
            pv: vec![best_move],
        });

        SearchResult {
            best_move: Some(best_move),
            score_cp: best_score,
            nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attic_eval::{
        FC_0_PADDED_INPUT_DIMS, HIDDEN_SIZE, HIDDEN1_DIMS, NUM_FEATURES, NetHeader, NnueNetwork,
        NnueNetworkBuilder, evaluate,
    };
    use attic_state::{Move, Position, format_usi_move, parse_sfen};
    use attic_storage::LargePageArray;

    // --- HalfKA_hm2 feature-plane geometry (mirrors attic-eval's features.rs).
    // A feature index is `E_KING_PERIOD * sq_k_code + p_adj`; board-piece
    // planes occupy `p_adj in [BOARD_LO, BOARD_HI)`, hand planes `[0, BOARD_LO)`,
    // and the shared king plane `[BOARD_HI, E_KING_PERIOD)`.
    const BOARD_LO: usize = 90;
    const BOARD_HI: usize = 1548;
    const E_KING_PERIOD: usize = 1629;

    // The two accumulator lanes ewm pairs into `transformed[0]`: the scalar
    // `ewm_one_perspective` multiplies lane `j` by lane `j + HIDDEN_SIZE/2`.
    const LANE_A: usize = 0;
    const LANE_B: usize = HIDDEN_SIZE / 2;

    /// Assemble a synthetic network from feature-transformer weights plus nine
    /// identical shortcut stacks, all in one arena. Biases are zero, so the
    /// accumulator is a pure sum of active feature columns; output row
    /// `HIDDEN1_DIMS` of every stack's fc_0 (the raw pre-ReLU term added to
    /// fc_2's output) reads `transformed[LANE_A]`/`transformed[LANE_B]` with
    /// weight 1 and every other weight/bias is 0, so the network output is
    /// `transformed[LANE_A] + transformed[LANE_B]`.
    fn net_with_ft(ft_weights: LargePageArray<i16>) -> NnueNetwork {
        let header = NetHeader {
            version: 0,
            hash: 0,
            arch_id: "synthetic".to_string(),
        };
        let mut b = NnueNetworkBuilder::new(header, [0u8; 32]);
        b.ft_weights_mut().copy_from_slice(&ft_weights);
        let row = HIDDEN1_DIMS * FC_0_PADDED_INPUT_DIMS;
        for s in 0..b.layer_stacks() {
            let w = b.fc_0_weights_mut(s);
            w[row + LANE_A] = 1;
            w[row + LANE_B] = 1;
        }
        b.build()
    }

    /// All-zero feature-transformer weights: every position evaluates to 0.
    fn zeroed_ft() -> LargePageArray<i16> {
        LargePageArray::zeroed(HIDDEN_SIZE * NUM_FEATURES)
    }

    /// Material-counting FT: every board-piece feature contributes `w` to
    /// lanes `LANE_A`/`LANE_B`; hand and king features contribute 0. Both
    /// perspectives see the same board-piece count, so the network output is a
    /// strictly increasing function of the number of pieces on the board (kept
    /// under the ewm clamp by a modest `w`). A capture removes a board piece
    /// (it moves to the mover's hand), lowering the child's eval, so negating
    /// it makes captures the most attractive moves.
    fn material_ft(w: i16) -> LargePageArray<i16> {
        let mut ft = LargePageArray::<i16>::zeroed(HIDDEN_SIZE * NUM_FEATURES);
        for f in 0..NUM_FEATURES {
            if (BOARD_LO..BOARD_HI).contains(&(f % E_KING_PERIOD)) {
                ft[f * HIDDEN_SIZE + LANE_A] = w;
                ft[f * HIDDEN_SIZE + LANE_B] = w;
            }
        }
        ft
    }

    /// Feature-index-dependent FT: every feature contributes a small,
    /// index-varying weight to the two live lanes. Evals then vary richly
    /// across sibling positions (both perspectives feed the score), which
    /// gives the argmax-equality tests something to discriminate. Weights stay
    /// small enough (`<= 5` per feature, `<= 40` active features) to keep the
    /// accumulator lanes under the ewm clamp of 254.
    fn patterned_ft() -> LargePageArray<i16> {
        let mut ft = LargePageArray::<i16>::zeroed(HIDDEN_SIZE * NUM_FEATURES);
        for f in 0..NUM_FEATURES {
            ft[f * HIDDEN_SIZE + LANE_A] = (f % 4 + 1) as i16;
            ft[f * HIDDEN_SIZE + LANE_B] = (f % 5 + 1) as i16;
        }
        ft
    }

    /// A recording [`InfoSink`] for asserting on emitted reports.
    #[derive(Default)]
    struct RecordingSink {
        reports: Vec<SearchInfo>,
    }

    impl InfoSink for RecordingSink {
        fn info(&mut self, info: &SearchInfo) {
            self.reports.push(info.clone());
        }
    }

    fn pos(sfen: &str) -> Position {
        parse_sfen(sfen).expect("valid SFEN")
    }

    fn legal_moves(p: &Position) -> Vec<Move> {
        let mut moves = Vec::new();
        p.generate_legal_all(&mut moves);
        moves
    }

    /// Independent full-refresh argmax: for each legal move, apply it, evaluate
    /// the child from scratch, negate for the mover, and keep the first
    /// maximum in generation order. Deliberately shares no code with the
    /// incremental path inside [`Search::go`].
    fn full_refresh_argmax(net: &NnueNetwork, p: &Position) -> Option<Move> {
        let mut work = p.clone();
        let mut best: Option<Move> = None;
        let mut best_score = i32::MIN;
        for mv in legal_moves(p) {
            let undo = work.do_move(mv);
            let score = -evaluate(net, &work);
            work.undo_move(mv, undo);
            if score > best_score {
                best_score = score;
                best = Some(mv);
            }
        }
        best
    }

    /// Is `mv` a capture in `p`? Drops never capture; otherwise the
    /// destination square is occupied by an enemy piece.
    fn is_capture(p: &Position, mv: Move) -> bool {
        !mv.is_drop() && p.board().get(mv.to_sq()).is_some()
    }

    const MATE_SFEN: &str = "4k4/4G4/3S5/9/9/9/9/9/4K4 w - 1";

    // A quiet-vs-one-capture position: the black gold on 5e can capture the
    // white pawn on 5d (its only capture). A gold never promotes, so that is a
    // single move — every other legal move (gold and king steps) is quiet, and
    // with no black hand pieces there are no drops. Four board pieces; the
    // capture is the only child with three.
    const ONE_CAPTURE_SFEN: &str = "4k4/9/9/4p4/4G4/9/9/9/4K4 b - 1";

    #[test]
    fn no_legal_move_returns_none() {
        let net = net_with_ft(patterned_ft());
        let search = Search::new(net);
        let mut sink = RecordingSink::default();
        let result = search.go(&pos(MATE_SFEN), &SearchLimits::default(), &mut sink);
        assert!(result.best_move.is_none());
        assert_eq!(result.nodes, 0);
        assert!(sink.reports.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn one_capture_is_chosen() {
        let p = pos(ONE_CAPTURE_SFEN);
        // Sanity: exactly one legal capture is available.
        let captures: Vec<Move> = legal_moves(&p)
            .into_iter()
            .filter(|&m| is_capture(&p, m))
            .collect();
        assert_eq!(
            captures.len(),
            1,
            "fixture must offer exactly one capture, got {:?}",
            captures
                .iter()
                .map(|&m| format_usi_move(m))
                .collect::<Vec<_>>()
        );

        let search = Search::new(net_with_ft(material_ft(40)));
        let mut sink = NullInfoSink;
        let result = search.go(&p, &SearchLimits::default(), &mut sink);
        assert_eq!(
            result.best_move,
            Some(captures[0]),
            "greedy search must pick the material-winning capture"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn tie_break_is_first_in_generation_order() {
        // The zero network scores every child 0, so all legal moves tie; the
        // strict-`>` tie-break must keep the first move in generation order.
        let p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        let moves = legal_moves(&p);
        assert!(moves.len() > 1);

        let search = Search::new(net_with_ft(zeroed_ft()));
        let mut sink = NullInfoSink;
        let result = search.go(&p, &SearchLimits::default(), &mut sink);
        assert_eq!(result.best_move, Some(moves[0]));
        assert_eq!(result.score_cp, 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn choice_is_deterministic_across_runs() {
        let p = pos(ONE_CAPTURE_SFEN);
        let search = Search::new(net_with_ft(patterned_ft()));
        let a = search.go(&p, &SearchLimits::default(), &mut NullInfoSink);
        let b = search.go(&p, &SearchLimits::default(), &mut NullInfoSink);
        assert_eq!(a.best_move, b.best_move);
        assert_eq!(a.score_cp, b.score_cp);
        assert_eq!(a.nodes, b.nodes);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn greedy_choice_equals_full_refresh_argmax() {
        // Several positions, including drops (hand pieces) and promotion-zone
        // moves, exercised against the incremental path inside `go`.
        const SFENS: &[&str] = &[
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
            "4k4/9/9/4p4/4G4/9/9/9/4K4 b - 1",                                 // one-capture
            "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
            "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",                   // promotion-zone
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game
        ];
        let net = net_with_ft(patterned_ft());
        let search = Search::new(net);
        for sfen in SFENS {
            let p = pos(sfen);
            let result = search.go(&p, &SearchLimits::default(), &mut NullInfoSink);
            let expected = full_refresh_argmax(search.network(), &p);
            assert_eq!(
                result.best_move, expected,
                "greedy vs full-refresh argmax mismatch on `{sfen}`"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn emits_one_info_report_with_expected_fields() {
        let p = pos(ONE_CAPTURE_SFEN);
        let search = Search::new(net_with_ft(material_ft(40)));
        let mut sink = RecordingSink::default();
        let result = search.go(&p, &SearchLimits::default(), &mut sink);

        assert_eq!(sink.reports.len(), 1);
        let report = &sink.reports[0];
        assert_eq!(report.depth, 1);
        assert_eq!(report.score_cp, result.score_cp);
        assert_eq!(report.nodes, result.nodes);
        assert_eq!(report.nodes, legal_moves(&p).len() as u64);
        assert_eq!(report.pv, vec![result.best_move.unwrap()]);
    }
}
