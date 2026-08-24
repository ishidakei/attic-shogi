//! Property gate: the incremental accumulator equals a from-scratch refresh at
//! every ply of a *randomly generated* legal line.
//!
//! `incremental_parity.rs` drives the same invariant from fixed fixture lines
//! and from six hand-seeded xorshift playouts. This file drives it from
//! proptest instead, so the line shape — which pieces come into hand, when a
//! king moves (forcing a full-refresh perspective), when a promotion or a drop
//! lands — is search-generated rather than hand-picked, and a counterexample
//! shrinks to a short reproducible line.
//!
//! As in `incremental_parity.rs`, randomness enters as a `&[u16]` of per-ply
//! choice indices into the real generator's legal-move list, not as an RNG
//! seed: shrinking then shortens the line and walks each choice back towards
//! the first generated move.
//!
//! # Backend
//!
//! Kernel selection in this crate is a **compile-time** decision (see
//! `simd::active_backend`) driven by `.cargo/config.toml`'s
//! `-C target-cpu=native`, so an integration test cannot pick a backend from
//! inside itself: this property exercises whichever one the build compiled,
//! and the loader logs it once so a run always says which.
//!
//! The pure-Rust scalar path is the target here — the SIMD path is already
//! covered by the fixture-driven parity tests and by the per-kernel
//! SIMD-vs-scalar equivalence tests in `simd::avx512*`. On a host without
//! AVX-512 this build *is* the scalar path and there is nothing to do. On an
//! AVX-512 host, pin the scalar backend by overriding the target CPU:
//!
//! ```text
//! cargo test --config 'build.rustflags=["-C","target-cpu=x86-64-v2"]' \
//!     -p attic-eval --all-features --test incremental_parity_proptest -- --nocapture
//! ```
//!
//! which reports `running on the Scalar backend`.
//!
//! # Network file
//!
//! The network is staged locally at `eval/nn.bin` and
//! is never committed. When it is absent the test prints a notice and passes,
//! matching the other eval tests, so the default `cargo test` run stays green
//! everywhere.

use std::path::PathBuf;
use std::sync::OnceLock;

use attic_eval::{Accumulator, NnueNetwork, active_backend, evaluate, evaluate_with, load_network};
use attic_state::{Color, Move, Position, Undo, format_sfen, format_usi_move};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

/// Upper bound on the generated line length. Shorter than the state-crate
/// suites' bound because every ply here pays for a full accumulator refresh
/// plus two evaluations.
const MAX_PLIES: usize = 24;

fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("eval/nn.bin")
}

/// The real network, loaded once per test binary, or `None` when `nn.bin` is
/// not staged in this checkout.
fn network() -> Option<&'static NnueNetwork> {
    static NET: OnceLock<Option<NnueNetwork>> = OnceLock::new();
    NET.get_or_init(|| {
        let path = nn_bin_path();
        if !path.exists() {
            eprintln!(
                "skipping the incremental-parity proptest: {} is not present (staged only on the dev VM)",
                path.display(),
            );
            return None;
        }
        eprintln!(
            "incremental-parity proptest running on the {:?} backend",
            active_backend(),
        );
        Some(load_network(&path).expect("real nn.bin should load and validate"))
    })
    .as_ref()
}

/// A fresh accumulator refreshed from `pos`.
fn refreshed(net: &NnueNetwork, pos: &Position) -> Accumulator {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    acc
}

/// Both halves of `acc` must be bit-identical to a from-scratch refresh of
/// `pos`, and `evaluate_with` over `acc` must match the full-refresh
/// `evaluate`.
fn check_matches_refresh(
    net: &NnueNetwork,
    acc: &Accumulator,
    pos: &Position,
    ctx: &str,
) -> Result<(), TestCaseError> {
    let fresh = refreshed(net, pos);
    for color in [Color::Black, Color::White] {
        prop_assert!(
            acc.perspective(color) == fresh.perspective(color),
            "{ctx}: {color:?} half diverged from refresh at {}",
            format_sfen(pos),
        );
    }
    prop_assert!(
        evaluate_with(net, acc, pos) == evaluate(net, pos),
        "{ctx}: evaluate paths disagree at {} ({} vs {})",
        format_sfen(pos),
        evaluate_with(net, acc, pos),
        evaluate(net, pos),
    );
    Ok(())
}

/// A random legal line, expressed as per-ply choice indices.
fn arb_line() -> impl Strategy<Value = Vec<u16>> {
    proptest::collection::vec(any::<u16>(), 0..=MAX_PLIES)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 24,
        ..ProptestConfig::default()
    })]

    /// Thread an incremental accumulator through a random legal line and back
    /// out again, checking it against a from-scratch refresh after every `do`
    /// and every `undo`.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn incremental_accumulator_matches_refresh_along_random_lines(line in arb_line()) {
        let Some(net) = network() else { return Ok(()) };

        let mut pos = Position::startpos();
        let root = refreshed(net, &pos);
        check_matches_refresh(net, &root, &pos, "root")?;

        let mut stack: Vec<(Move, Undo, Accumulator)> = Vec::new();
        let mut legal: Vec<Move> = Vec::with_capacity(192);

        for (ply, &choice) in line.iter().enumerate() {
            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                // Terminal node: nothing left to apply.
                break;
            }
            // Check bias: every other ply steers towards a checking move when
            // one exists. Uniform random play from the start position gives
            // check well under 1% of the time, and it is the forced king
            // replies to a check that drive the *full-refresh* perspective
            // branch of `update_after_move` — the branch this property most
            // wants to exercise. The bias only reorders the preference among
            // moves the real generator emitted, so every position stays legal
            // and still reachable.
            let checking: Vec<Move> = if choice % 2 == 0 {
                legal.iter().copied().filter(|&m| pos.gives_check(m)).collect()
            } else {
                Vec::new()
            };
            let pool = if checking.is_empty() { &legal[..] } else { &checking[..] };
            let mv = pool[(choice / 2) as usize % pool.len()];

            let parent = stack.last().map_or(&root, |frame| &frame.2);
            let acc = parent.update_after_move(net, &mut pos, mv);
            let undo = pos.do_move(mv);

            check_matches_refresh(
                net,
                &acc,
                &pos,
                &format!("ply {ply} `{}` [after do]", format_usi_move(mv)),
            )?;
            stack.push((mv, undo, acc));
        }

        // Unwind: after each undo the parent's accumulator must still describe
        // the restored position exactly.
        while let Some((mv, undo, _acc)) = stack.pop() {
            pos.undo_move(mv, undo);
            let parent = stack.last().map_or(&root, |frame| &frame.2);
            check_matches_refresh(
                net,
                parent,
                &pos,
                &format!("unwind `{}` [after undo]", format_usi_move(mv)),
            )?;
        }
    }
}
