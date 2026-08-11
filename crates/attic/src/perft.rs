//! Perft: total number of leaf nodes reachable from a position at exactly
//! `depth` plies of legal play. Used as a parity gate against the reference
//! engine — see `crates/attic/tests/parity/perft.rs`.

use attic_state::{Move, Position};

pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    pos.generate_legal_all(&mut moves);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut total: u64 = 0;
    for m in moves {
        let undo = pos.do_move(m);
        total += perft(pos, depth - 1);
        pos.undo_move(m, undo);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perft_depth_zero_is_one() {
        let mut pos = Position::startpos();
        assert_eq!(perft(&mut pos, 0), 1);
    }

    #[test]
    fn perft_startpos_depth_one_is_thirty() {
        let mut pos = Position::startpos();
        assert_eq!(perft(&mut pos, 1), 30);
    }

    #[test]
    fn perft_startpos_depth_two_is_900() {
        let mut pos = Position::startpos();
        assert_eq!(perft(&mut pos, 2), 900);
    }

    #[test]
    fn perft_does_not_mutate_input() {
        let mut pos = Position::startpos();
        let snapshot = pos.clone();
        let _ = perft(&mut pos, 2);
        assert_eq!(pos, snapshot);
    }
}
