//! Runtime `FV_SCALE` gate: the final score is the raw network output divided
//! by the live [`attic_eval::fv_scale`], so changing the scale at runtime
//! rescales the eval exactly.
//!
//! Reuses the committed eval fixtures (`tests/fixtures/eval/*.json`) — the same
//! ones `attic-eval/tests/eval_parity.rs` captured at `FV_SCALE = 16`; nothing
//! is re-captured here. The real SFNN-1536 network is staged locally at
//! `eval/nn.bin` and is never committed, so the test
//! prints a notice and passes when it is absent (a checkout without it staged).
//!
//! This is the only test in its binary, so the process-global scale it toggles
//! never races another test; it is restored to the default on the way out
//! regardless.

use std::path::PathBuf;

use attic_eval::{FV_SCALE_DEFAULT, NnueNetwork, evaluate, fv_scale, load_network, set_fv_scale};
use attic_state::{Position, parse_sfen, parse_usi_move};

fn workspace_relative(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn position_for(sfen: &str, moves: &[String]) -> Position {
    let mut pos = parse_sfen(sfen).unwrap_or_else(|e| panic!("bad sfen `{sfen}`: {e:?}"));
    for mv in moves {
        let parsed = parse_usi_move(mv, &pos).unwrap_or_else(|e| panic!("bad move `{mv}`: {e:?}"));
        pos.do_move(parsed);
    }
    pos
}

/// Load one fixture's `(sfen, moves)` — the highest-magnitude fixture, so the
/// `/24` division is exercised on a value well above the divisor.
fn richest_fixture() -> (String, Vec<String>) {
    let dir = workspace_relative("tests/fixtures/eval");
    let mut best: Option<(i64, String, Vec<String>)> = None;
    for entry in std::fs::read_dir(&dir).expect("read fixtures dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read fixture");
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse fixture");
        let Some(sfen) = json["sfen"].as_str() else {
            continue;
        };
        let eval = json["eval"].as_i64().unwrap_or(0);
        let moves = match json.get("moves") {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|m| m.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        };
        if best.as_ref().is_none_or(|(b, ..)| eval.abs() > b.abs()) {
            best = Some((eval, sfen.to_string(), moves));
        }
    }
    let (_, sfen, moves) = best.expect("at least one eval fixture");
    (sfen, moves)
}

#[test]
#[cfg_attr(miri, ignore)]
fn fv_scale_24_divides_raw_output_by_24() {
    let nn_bin = workspace_relative("eval/nn.bin");
    if !nn_bin.exists() {
        eprintln!(
            "skipping fv_scale_24_divides_raw_output_by_24: {} is not present (staged only on the dev VM)",
            nn_bin.display()
        );
        return;
    }
    let net: NnueNetwork = load_network(&nn_bin).expect("real nn.bin should load and validate");

    let (sfen, moves) = richest_fixture();
    let pos = position_for(&sfen, &moves);

    // Recover the raw (pre-scale) network output by evaluating at scale 1.
    set_fv_scale(1);
    let raw = evaluate(&net, &pos);
    assert!(
        raw.abs() > 24,
        "fixture output {raw} too small to exercise a /24 division"
    );

    // At the default scale the eval is the raw output / 16 (the fixture-capture
    // condition), reconfirming `raw` is the true numerator.
    set_fv_scale(FV_SCALE_DEFAULT);
    assert_eq!(evaluate(&net, &pos), raw / FV_SCALE_DEFAULT);

    // The property under test: at FV_SCALE 24 the score is the raw output / 24.
    set_fv_scale(24);
    assert_eq!(fv_scale(), 24);
    assert_eq!(
        evaluate(&net, &pos),
        raw / 24,
        "FV_SCALE 24 must divide the raw network output by 24"
    );

    // Restore the process-global default for any later-loaded consumer.
    set_fv_scale(FV_SCALE_DEFAULT);
}
