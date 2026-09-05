//! Search layer. [`Search::go`] is a greedy 1-ply NNUE move choice: it
//! enumerates the legal moves, scores each child with the real network from the
//! mover's point of view, and returns the maximum. That function alone claims
//! no alpha-beta, no quiescence, no transposition table, no time management and
//! no reference parity; the full search that does is [`qsearch`] and [`root`].
//!
//! It may depend on Evaluation and State, never on Protocol. Protocol calls
//! into this crate and so never reaches Evaluation directly.

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
// without the direct dependency on Evaluation the layering rules forbid.
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

/// Search limits for one `go` invocation. Greedy 1-ply consults none of them;
/// the type exists so that the USI driver can map its own limits onto a
/// Search-layer type without this crate depending on Protocol.
#[derive(Debug, Clone, Default)]
pub struct SearchLimits {}

/// One progress report emitted through an [`InfoSink`] during a search,
/// carrying the fields a USI `info` line does.
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

/// Sink for search progress reports, abstracting where `info` output goes so
/// that [`Search`] never touches the Protocol layer.
pub trait InfoSink {
    /// Receive one progress report.
    fn info(&mut self, info: &SearchInfo);
}

/// An [`InfoSink`] that discards every report.
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
pub struct Search {
    net: NnueNetwork,
}

impl Search {
    /// Wrap an already-loaded network.
    pub fn new(net: NnueNetwork) -> Self {
        Self { net }
    }

    /// Load and validate the network at `path`, then wrap it, discarding any
    /// non-fatal warnings.
    pub fn from_network_file(path: &Path) -> Result<Self, NnueError> {
        Ok(Self::new(load_network(path)?))
    }

    /// [`Search::from_network_file`] returning the loader's non-fatal warning
    /// bodies too; an empty vector means a clean load.
    pub fn from_network_file_with_warnings(path: &Path) -> Result<(Self, Vec<String>), NnueError> {
        let (net, warnings) = load_network_with_warnings(path)?;
        Ok((Self::new(net), warnings))
    }

    /// The network this search evaluates with.
    pub fn network(&self) -> &NnueNetwork {
        &self.net
    }

    /// A deep copy of this search with freshly allocated network storage. The
    /// driver runs it inside a NUMA-node-bound thread to place the copy's pages
    /// on that node.
    pub fn replicate(&self) -> Self {
        Self::new(self.net.replicate())
    }

    /// Greedy 1-ply move choice: evaluate every child from the mover's point of
    /// view — the child's own evaluation is from the opponent's, so it is
    /// negated — and pick the maximum, tie-breaking on the first in generation
    /// order.
    ///
    /// Emits one [`SearchInfo`] report when a move is chosen, and none, with a
    /// `None` best move, when the side to move has no legal move.
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
            // `update_after_move` leaves `work` unchanged, so the move is
            // applied for real to give `evaluate_with` the child's side to move
            // and king ranks.
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

    // The HalfKA_hm2 feature-plane geometry, mirroring `attic-eval`: a feature
    // index is `E_KING_PERIOD * sq_k_code + p_adj`, with board-piece planes at
    // `p_adj in [BOARD_LO, BOARD_HI)`, hand planes below and the shared king
    // plane above.
    const BOARD_LO: usize = 90;
    const BOARD_HI: usize = 1548;
    const E_KING_PERIOD: usize = 1629;

    // The two accumulator lanes the output transform pairs into
    // `transformed[0]`.
    const LANE_A: usize = 0;
    const LANE_B: usize = HIDDEN_SIZE / 2;

    /// A synthetic network whose zero biases make the accumulator a pure sum of
    /// active feature columns, and whose only nonzero FC weights route the two
    /// live lanes through the shortcut, so that the network output is
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

    /// A material-counting FT: only board-piece features contribute, so the
    /// network output increases strictly with the number of pieces on the board.
    /// A capture moves a piece to a hand and so lowers the child's eval, which
    /// negation turns into the most attractive move.
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

    /// An FT whose per-feature weight varies with the feature index, so that
    /// evals vary richly across siblings and the argmax tests have something to
    /// discriminate. The weights stay small enough to keep the accumulator lanes
    /// under the output transform's clamp.
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

    /// A full-refresh argmax, sharing no code with the incremental path inside
    /// [`Search::go`].
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

    /// Is `mv` a capture in `p`?
    fn is_capture(p: &Position, mv: Move) -> bool {
        !mv.is_drop() && p.board().get(mv.to_sq()).is_some()
    }

    const MATE_SFEN: &str = "4k4/4G4/3S5/9/9/9/9/9/4K4 w - 1";

    // Exactly one capture is available, by a gold, which never promotes — so it
    // is a single move, and with no hand pieces there are no drops either. Four
    // board pieces, of which the capture is the only child with three.
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
        // The zero network scores every child 0, so the tie-break must keep the
        // first move in generation order.
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
