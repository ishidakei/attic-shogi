//! Integration test: greedy 1-ply search with the real SFNN-1536 network.
//!
//! The network file is staged locally at
//! `eval/nn.bin` and is never committed. When it is
//! absent (a checkout without it staged) the test prints a notice and passes,
//! so the default `cargo test` run stays green everywhere — the same pattern
//! `attic-eval`'s integration tests use.

use std::path::PathBuf;

use attic_eval::{NnueNetwork, evaluate};
use attic_search::{NullInfoSink, Search, SearchLimits};
use attic_state::{Move, Position, parse_sfen};

fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

fn legal_moves(p: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    p.generate_legal_all(&mut moves);
    moves
}

/// Independent full-refresh argmax, sharing no code with `Search::go`'s
/// incremental accumulator path.
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

#[test]
#[cfg_attr(miri, ignore)]
fn startpos_choice_is_legal_deterministic_and_matches_full_refresh() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping startpos_choice_is_legal_deterministic_and_matches_full_refresh: {} is not present (staged only on the dev VM)",
            path.display()
        );
        return;
    }

    let search = Search::from_network_file(&path).expect("real nn.bin should load and validate");
    let p = parse_sfen(STARTPOS).expect("valid startpos SFEN");

    let a = search.go(&p, &SearchLimits::default(), &mut NullInfoSink);
    let chosen = a.best_move.expect("startpos has legal moves");

    // Legal.
    assert!(
        legal_moves(&p).contains(&chosen),
        "chosen move is not a startpos legal move"
    );

    // Deterministic across two runs.
    let b = search.go(&p, &SearchLimits::default(), &mut NullInfoSink);
    assert_eq!(a.best_move, b.best_move);
    assert_eq!(a.score_cp, b.score_cp);
    assert_eq!(a.nodes, b.nodes);

    // Equal to the independent full-refresh argmax.
    assert_eq!(
        a.best_move,
        full_refresh_argmax(search.network(), &p),
        "greedy choice diverged from the full-refresh argmax"
    );
}
