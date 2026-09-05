//! Property test: `do_move` / `undo_move` is an exact round trip, over every
//! position on a random legal line from `startpos`.
//!
//! The Zobrist keys get their own assertions because [`Position`]'s `PartialEq`
//! is deliberately blind to them, so a broken incremental update would slip
//! straight through a bare position comparison.

mod common;

use attic_state::{Color, Position, format_sfen, format_usi_move};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

/// Every incrementally maintained key of `pos`, in one comparable array.
fn keys(pos: &Position) -> [u64; 7] {
    [
        pos.key(),
        pos.board_key(),
        pos.hand_key(),
        pos.pawn_key(),
        pos.minor_piece_key(),
        pos.non_pawn_key(Color::Black),
        pos.non_pawn_key(Color::White),
    ]
}

/// A random legal line, expressed as per-ply choice indices.
fn arb_line() -> impl Strategy<Value = Vec<u16>> {
    proptest::collection::vec(any::<u16>(), 0..=common::MAX_PLIES)
}

/// Do/undo every legal move at `pos` and assert nothing observable changed.
fn check_do_undo(pos: &mut Position) -> Result<(), TestCaseError> {
    let before = pos.clone();
    let before_sfen = format_sfen(&before);
    let before_keys = keys(&before);
    let before_in_check = before.in_check();
    let before_plies_from_null = before.plies_from_null();
    let before_occurrences = before.position_occurrences();

    let mut legal = Vec::new();
    pos.generate_legal_all(&mut legal);

    for mv in legal {
        let usi = format_usi_move(mv);
        let undo = pos.do_move(mv);

        // A real move always changes the position, so the round trip below is
        // never vacuous.
        prop_assert!(
            pos.side_to_move() != before.side_to_move(),
            "{before_sfen}: `{usi}` did not flip the side to move",
        );
        prop_assert!(
            *pos != before,
            "{before_sfen}: `{usi}` left the position unchanged",
        );

        pos.undo_move(mv, undo);

        // Compared component by component first, so that a failure names what
        // diverged, then as a whole, which also covers the move history.
        prop_assert!(
            pos.board() == before.board(),
            "{before_sfen}: board diverged after do/undo of `{usi}` (now {})",
            format_sfen(pos),
        );
        prop_assert!(
            pos.hand(Color::Black) == before.hand(Color::Black)
                && pos.hand(Color::White) == before.hand(Color::White),
            "{before_sfen}: hands diverged after do/undo of `{usi}` (now {})",
            format_sfen(pos),
        );
        prop_assert!(
            pos.side_to_move() == before.side_to_move(),
            "{before_sfen}: side to move diverged after do/undo of `{usi}`",
        );
        prop_assert!(
            pos.ply() == before.ply(),
            "{before_sfen}: ply diverged after do/undo of `{usi}` ({} vs {})",
            pos.ply(),
            before.ply(),
        );
        prop_assert!(
            *pos == before,
            "{before_sfen}: position diverged after do/undo of `{usi}` (now {})",
            format_sfen(pos),
        );

        prop_assert!(
            keys(pos) == before_keys,
            "{before_sfen}: keys diverged after do/undo of `{usi}`: {:?} vs {:?}",
            keys(pos),
            before_keys,
        );
        prop_assert!(
            pos.in_check() == before_in_check,
            "{before_sfen}: check info diverged after do/undo of `{usi}`",
        );
        prop_assert!(
            pos.plies_from_null() == before_plies_from_null,
            "{before_sfen}: plies_from_null diverged after do/undo of `{usi}`",
        );
        prop_assert!(
            pos.position_occurrences() == before_occurrences,
            "{before_sfen}: repetition count diverged after do/undo of `{usi}`",
        );
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    /// Every legal move at every position on a random line survives the round
    /// trip bit-identically.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn do_then_undo_restores_the_position(line in arb_line()) {
        common::walk_line(&line, check_do_undo)?;
    }

    /// Unwinding a whole random line returns to the start position.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn unwinding_a_whole_line_returns_to_startpos(line in arb_line()) {
        let (mut pos, mut frames) = common::play_line(&line);

        while let Some((mv, undo)) = frames.pop() {
            pos.undo_move(mv, undo);
        }

        let start = Position::startpos();
        prop_assert!(
            pos == start,
            "unwinding {} plies did not restore startpos: {}",
            line.len(),
            format_sfen(&pos),
        );
        prop_assert!(
            keys(&pos) == keys(&start),
            "unwinding {} plies left the keys stale: {:?} vs {:?}",
            line.len(),
            keys(&pos),
            keys(&start),
        );
    }
}
