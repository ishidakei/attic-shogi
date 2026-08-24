//! Property gate: invariants of the legal-move generator over random reachable
//! positions — every position on a random legal line from `startpos`, see
//! [`common::walk_line`].
//!
//! Three families:
//!
//! 1. **Well-formedness** — `generate_legal_all` emits no duplicates, every
//!    emitted move is `is_ok`, passes the crate's own `pseudo_legal` and
//!    `is_legal` predicates, and survives the `move16` narrow/widen round trip
//!    through `to_move` (the TT fragment path).
//! 2. **Independent legality oracle** — every emitted move, once played, really
//!    does leave the mover's king unattacked. This is a copy-apply-scan check
//!    (`is_attacked_discounting` over the post-move board) that shares no logic
//!    with `is_legal`'s constant-time pin test, so it is a genuine cross-check
//!    rather than a restatement.
//! 3. **Cross-path agreement** — the pseudo-legal-plus-filter route and the
//!    direct legal route agree as *sets*:
//!    * not in check: `generate_captures ∪ generate_quiets`, filtered by
//!      `is_legal`, equals `generate_legal_all` — which goes through the
//!      single-target `generate_non_evasions` instead, a different target mask
//!      and a different emission order;
//!    * in check: the *unrestricted* `generate_non_evasions`, filtered by the
//!      independent scan oracle above, equals `generate_legal_all` — which goes
//!      through the restricted `generate_evasions` plus `is_legal`. This is the
//!      strong direction: it says the restricted evasion generator loses no
//!      legal move and invents none. (`is_legal` cannot judge the non-evasion
//!      candidates here: its fast path is contract-bound to evasion output when
//!      the side to move is in check.)
//!
//! Move *order* is pinned elsewhere — the `search_movegen` scan-oracle sequence
//! gate — so these properties are deliberately order-blind.

mod common;

use std::collections::HashSet;

use attic_state::{ExtMove, Move, Position, format_sfen, format_usi_move};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

/// A random legal line, expressed as per-ply choice indices.
fn arb_line() -> impl Strategy<Value = Vec<u16>> {
    proptest::collection::vec(any::<u16>(), 0..=common::MAX_PLIES)
}

/// Copy-apply-scan legality oracle: play `mv`, ask whether the mover's king is
/// attacked on the resulting board, then take the move back.
///
/// `is_attacked_discounting(ksq, them, ksq)` is the plain "is this square
/// attacked" scan: discounting the square under test is documented as a no-op
/// for the piece standing on it — it can never block an attack aimed at itself
/// — and that argument is the only way to reach the scan from outside the
/// crate.
fn leaves_mover_king_safe(pos: &mut Position, mv: Move) -> bool {
    let mover = pos.side_to_move();
    let undo = pos.do_move(mv);
    let safe = match pos.king_square(mover) {
        Some(ksq) => !pos.is_attacked_discounting(ksq, mover.flip(), ksq),
        None => false,
    };
    pos.undo_move(mv, undo);
    safe
}

/// The set of moves in `buf` that pass `keep`.
fn filtered(buf: &[ExtMove], mut keep: impl FnMut(Move) -> bool) -> HashSet<Move> {
    buf.iter().map(|em| em.mv).filter(|&m| keep(m)).collect()
}

/// Render a move set as a sorted USI list for failure messages.
fn describe(moves: &HashSet<Move>) -> String {
    let mut v: Vec<String> = moves.iter().copied().map(format_usi_move).collect();
    v.sort();
    v.join(" ")
}

/// Generated legal moves are unique, well-formed, and accepted by the crate's
/// own predicates.
fn check_well_formed(pos: &mut Position) -> Result<(), TestCaseError> {
    let sfen = format_sfen(pos);

    let mut legal = Vec::new();
    pos.generate_legal_all(&mut legal);

    let unique: HashSet<Move> = legal.iter().copied().collect();
    prop_assert!(
        unique.len() == legal.len(),
        "{sfen}: generate_legal_all emitted {} moves but only {} distinct ones",
        legal.len(),
        unique.len(),
    );

    for &mv in &legal {
        let usi = format_usi_move(mv);
        prop_assert!(mv.is_ok(), "{sfen}: generated move `{usi}` is not is_ok");
        prop_assert!(
            pos.pseudo_legal(mv, true),
            "{sfen}: generated legal move `{usi}` fails pseudo_legal",
        );
        prop_assert!(
            pos.is_legal(mv),
            "{sfen}: generated legal move `{usi}` fails is_legal",
        );
        prop_assert!(
            pos.to_move(mv.move16()) == Some(mv),
            "{sfen}: `{usi}` did not survive the move16 round trip",
        );
    }

    Ok(())
}

/// Every generated legal move genuinely leaves the mover's king safe.
fn check_king_safety(pos: &mut Position) -> Result<(), TestCaseError> {
    let sfen = format_sfen(pos);

    let mut legal = Vec::new();
    pos.generate_legal_all(&mut legal);

    for &mv in &legal {
        prop_assert!(
            leaves_mover_king_safe(pos, mv),
            "{sfen}: generated legal move `{}` leaves the mover in check",
            format_usi_move(mv),
        );
    }

    Ok(())
}

/// The pseudo-legal-plus-filter route and the direct legal route agree.
fn check_routes_agree(pos: &mut Position) -> Result<(), TestCaseError> {
    let sfen = format_sfen(pos);

    let mut legal = Vec::new();
    pos.generate_legal_all(&mut legal);
    let direct: HashSet<Move> = legal.into_iter().collect();

    let in_check = pos.in_check();
    let mut buf: Vec<ExtMove> = Vec::new();
    let via_filter: HashSet<Move> = if in_check {
        pos.generate_non_evasions(true, &mut buf);
        let candidates: Vec<Move> = buf.iter().map(|em| em.mv).collect();
        candidates
            .into_iter()
            .filter(|&m| leaves_mover_king_safe(pos, m))
            .collect()
    } else {
        pos.generate_captures(true, &mut buf);
        pos.generate_quiets(true, &mut buf);
        filtered(&buf, |m| pos.is_legal(m))
    };

    prop_assert!(
        via_filter == direct,
        "{sfen} (in_check = {in_check}): routes disagree\n  filtered: {}\n  direct:   {}",
        describe(&via_filter),
        describe(&direct),
    );

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn generated_legal_moves_are_unique_and_well_formed(line in arb_line()) {
        common::walk_line(&line, check_well_formed)?;
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn generated_legal_moves_leave_the_king_safe(line in arb_line()) {
        common::walk_line(&line, check_king_safety)?;
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn generator_routes_agree_on_the_legal_move_set(line in arb_line()) {
        common::walk_line(&line, check_routes_agree)?;
    }
}
