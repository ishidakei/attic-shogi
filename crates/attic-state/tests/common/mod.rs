//! Shared driver for the randomized-position proptest suites.
//!
//! Both `do_undo_roundtrip.rs` and `movegen_invariants.rs` want the same notion
//! of "a random *reachable* position": not an arbitrary board (that is what
//! `sfen_roundtrip.rs`'s `arb_position` builds, and most such boards are not
//! reachable from the start position), but the positions along a random legal
//! line played from `startpos` with the real generator.
//!
//! The randomness is threaded in as a plain `&[u16]` of per-ply choice indices
//! rather than an RNG seed, so proptest shrinking is meaningful: dropping a
//! trailing element shortens the line, and shrinking an element towards `0`
//! walks the choice back towards the first generated move.

use attic_state::{Move, Position, Undo};

/// Upper bound on the length of a generated line. Deep enough to reach genuine
/// middlegame positions (captures in hand, promotions, checks) while keeping a
/// case at a few hundred microseconds.
pub const MAX_PLIES: usize = 48;

/// Visit every position on the line `choices` describes, leaf first, then each
/// ancestor back to `startpos`, calling `visit` on each.
///
/// Walking the whole line rather than only its leaf multiplies the positions a
/// single proptest case covers by the line length, which is what makes rarer
/// states — in-check nodes above all, but also nodes with a stuffed hand or a
/// promoted piece pinned to a king — show up often enough to matter.
///
/// `visit` may mutate the position (a do/undo probe, say) but must leave it
/// unchanged on return; the unwind depends on it.
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
/// position has one. Uniform random play from the start position gives check
/// well under 1% of the time (measured over this suite's corpus: 15 in-check
/// nodes out of 1623), which would leave the in-check branch of every property
/// here all but untested; the bias lifts that to 85 out of 1626.
///
/// The bias only *reorders the preference* among moves the real generator
/// already emitted, so every position reached is still legal and still
/// reachable — it just makes evasions, pins and double checks common enough to
/// be worth generating.
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
