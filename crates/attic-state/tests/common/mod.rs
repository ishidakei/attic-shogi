//! Shared driver for the randomized-position proptest suites: a random
//! *reachable* position, meaning one on a random legal line played from
//! `startpos` with the real generator, rather than an arbitrary board.
//!
//! The randomness is threaded in as a plain `&[u16]` of per-ply choice indices
//! rather than an RNG seed, so that proptest shrinking stays meaningful:
//! dropping a trailing element shortens the line, and shrinking an element
//! towards `0` walks the choice back towards the first generated move.

use attic_state::{Move, Position, Undo};

/// Upper bound on the length of a generated line: deep enough to reach genuine
/// middlegame positions, short enough to keep a case fast.
pub const MAX_PLIES: usize = 48;

/// Visit every position on the line `choices` describes, leaf first, then each
/// ancestor back to `startpos`. Walking the whole line rather than only its leaf
/// is what makes rare states — in-check nodes above all — show up often enough
/// to matter.
///
/// `visit` may mutate the position but must leave it unchanged on return; the
/// unwind depends on it.
pub fn walk_line<E>(
    choices: &[u16],
    mut visit: impl FnMut(&mut Position) -> Result<(), E>,
) -> Result<(), E> {
    let (mut pos, mut frames) = play_line(choices);
    loop {
        visit(&mut pos)?;
        match frames.pop() {
            Some((mv, undo)) => pos.undo_move(mv, undo),
            None => return Ok(()),
        }
    }
}

/// Play `choices` from the start position, one legal move per element, stopping
/// early at a terminal (mate/stalemate) node.
///
/// Returns the reached position together with the `(move, undo)` frames in
/// play order, so a caller can unwind the whole line back to `startpos`.
///
/// # Check bias
///
/// Every other choice value steers the ply towards a checking move when the
/// position has one. Uniform random play gives check well under 1% of the time,
/// which would leave the in-check branch of every property here all but
/// untested. The bias only *reorders the preference* among moves the real
/// generator already emitted, so every position reached is still legal and
/// still reachable.
pub fn play_line(choices: &[u16]) -> (Position, Vec<(Move, Undo)>) {
    let mut pos = Position::startpos();
    let mut frames = Vec::with_capacity(choices.len());
    let mut legal: Vec<Move> = Vec::with_capacity(192);

    for &choice in choices {
        legal.clear();
        pos.generate_legal_all(&mut legal);
        if legal.is_empty() {
            break;
        }

        let checking: Vec<Move> = if choice % 2 == 0 {
            legal
                .iter()
                .copied()
                .filter(|&m| pos.gives_check(m))
                .collect()
        } else {
            Vec::new()
        };
        let pool = if checking.is_empty() {
            &legal[..]
        } else {
            &checking[..]
        };

        let mv = pool[(choice / 2) as usize % pool.len()];
        let undo = pos.do_move(mv);
        frames.push((mv, undo));
    }

    (pos, frames)
}
